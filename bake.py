import logging
from itertools import islice
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import TrainConfig

logger = logging.getLogger(__name__)

frame_group_size = 5


def embeddings_dir(cfg: TrainConfig) -> Path:
    """Directory holding the baked speech embedding shards."""
    return Path(cfg.output_path) / "speech_embeddings"


def bake_speech_embeddings(cfg: TrainConfig, device: torch.device) -> None:
    """Encode max_samples recordings with Whisper and save grouped embeddings to disk.

    Each shard holds one encoder batch: fp16 embeddings [n, 300, 3840], ids and transcripts.
    Existing shards are skipped unless cfg.overwrite is set, so an interrupted bake can resume.
    """
    if cfg.bake_batch_size < 1:
        raise ValueError("bake_batch_size must be at least 1.")
    out_dir = embeddings_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    speech_dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    speech_dataset = speech_dataset.cast_column("audio", Audio(sampling_rate=16000))
    speech_dataset = speech_dataset.take(cfg.max_samples)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = whisper_model.model.encoder.to(device).eval()

    iterator = iter(speech_dataset)
    baked = 0
    shard_index = 0
    while baked < cfg.max_samples:
        examples = list(islice(iterator, cfg.bake_batch_size))
        if not examples:
            break
        shard_path = out_dir / f"shard_{shard_index:05d}.pt"
        shard_index += 1
        baked += len(examples)
        if shard_path.exists() and not cfg.overwrite:
            logger.info("skipping existing %s", shard_path.name)
            continue

        # Pad to Whisper's 30-second window; the encoder always emits 1500 frames.
        audio_features = processor.feature_extractor(
            [example["audio"]["array"] for example in examples],
            sampling_rate=16000,
            padding="max_length",
            max_length=processor.feature_extractor.n_samples,
            truncation=True,
            return_tensors="pt",
        ).input_features.to(device)
        with torch.no_grad():
            speech_embeddings = encoder(audio_features).last_hidden_state  # [batch, 1500, 768]

        batch_size, frame_count, encoder_dim = speech_embeddings.shape
        grouped_frame_count = frame_count // frame_group_size
        grouped = speech_embeddings[:, : grouped_frame_count * frame_group_size].reshape(
            batch_size, grouped_frame_count, frame_group_size * encoder_dim
        )  # [batch, 300, 3840]

        torch.save(
            {
                "embeddings": grouped.to(torch.float16).cpu(),
                "ids": [example["id"] for example in examples],
                "texts": [example["text"] for example in examples],
            },
            shard_path,
        )
        logger.info("baked %d/%d -> %s", baked, cfg.max_samples, shard_path.name)

    if baked < cfg.max_samples:
        logger.warning("Requested %d examples, but the dataset only had %d.", cfg.max_samples, baked)


def load_baked_embeddings(cfg: TrainConfig) -> tuple[torch.Tensor, list[str], list[str]]:
    """Load every shard into memory: fp16 embeddings [N, 300, 3840], ids, transcripts."""
    shard_paths = sorted(embeddings_dir(cfg).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No baked embeddings in {embeddings_dir(cfg)}; run with stage=bake first."
        )
    # Stale shards from an earlier, larger bake may sit on disk; only use the first max_samples.
    shards = []
    loaded = 0
    for path in shard_paths:
        if loaded >= cfg.max_samples:
            break
        shard = torch.load(path)
        shards.append(shard)
        loaded += len(shard["ids"])
    embeddings = torch.cat([shard["embeddings"] for shard in shards])[: cfg.max_samples]
    ids = [i for shard in shards for i in shard["ids"]][: cfg.max_samples]
    texts = [t for shard in shards for t in shard["texts"]][: cfg.max_samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Baked shards contain duplicate example IDs.")
    return embeddings, ids, texts
