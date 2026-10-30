import io
import logging
from itertools import islice
from pathlib import Path

import numpy as np
import soundfile
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils import TrainConfig

logger = logging.getLogger(__name__)

frame_group_size = 5


SUBSETS = ("train", "val", "test")


def subset_spec(cfg: TrainConfig, subset: str) -> tuple[str | None, int, Path]:
    """Dataset split, number of samples and shard directory for a subset (train/val/test)."""
    if subset == "train":
        return cfg.dataset_split, cfg.max_samples, Path(cfg.output_path) / "speech_embeddings"
    if subset == "val":
        return cfg.val_split, cfg.val_samples, Path(cfg.output_path) / "speech_embeddings_val"
    if subset == "test":
        return cfg.test_split, cfg.test_samples, Path(cfg.output_path) / "speech_embeddings_test"
    raise ValueError(f"Unknown subset '{subset}'; expected one of {SUBSETS}.")


def bake_speech_embeddings(cfg: TrainConfig, device: torch.device) -> None:
    """Bake the train, val and test subsets that have a non-zero sample count."""
    if cfg.bake_batch_size < 1:
        raise ValueError("bake_batch_size must be at least 1.")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = whisper_model.model.encoder.to(device).eval()
    for subset in SUBSETS:
        split, sample_count, out_dir = subset_spec(cfg, subset)
        if sample_count > 0:
            logger.info(f"baking {subset}: {sample_count} examples from {split}")
            _bake_subset(cfg, processor, encoder, device, split, sample_count, out_dir)


def _decode_audio(audio: dict) -> np.ndarray:
    """Decode raw FLAC/WAV bytes with soundfile, so no FFmpeg/torchcodec is needed (LibriSpeech is already 16 kHz)."""
    array, sampling_rate = soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if sampling_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sampling_rate} Hz")
    return array


def _bake_subset(
    cfg: TrainConfig,
    processor: WhisperProcessor,
    encoder: torch.nn.Module,
    device: torch.device,
    split: str | None,
    max_samples: int,
    out_dir: Path,
) -> None:
    """Encode max_samples recordings with Whisper and save grouped embeddings to disk.

    Each shard holds one encoder batch: fp16 embeddings [n, 300, 3840], ids, transcripts and the
    unpadded audio length in 16 kHz samples (to tell speech frames from padding).
    Existing shards are skipped unless cfg.overwrite is set, so an interrupted bake can resume.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    speech_dataset = load_dataset(cfg.dataset_name, split=split, streaming=True)
    speech_dataset = speech_dataset.cast_column("audio", Audio(decode=False))
    speech_dataset = speech_dataset.take(max_samples)

    iterator = iter(speech_dataset)
    baked = 0
    shard_index = 0
    while baked < max_samples:
        examples = list(islice(iterator, cfg.bake_batch_size))
        if not examples:
            break
        for example in examples:
            example["audio"]["array"] = _decode_audio(example["audio"])
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
                "num_samples": torch.tensor([len(example["audio"]["array"]) for example in examples]),
            },
            shard_path,
        )
        logger.info("baked %d/%d -> %s", baked, max_samples, shard_path.name)

    if baked < max_samples:
        logger.warning("Requested %d examples, but the dataset only had %d.", max_samples, baked)


def speech_frame_counts(num_samples: torch.Tensor) -> torch.Tensor:
    """Number of grouped frames covering real audio; each covers 5 Whisper frames = 100 ms = 1600 samples."""
    return ((num_samples + 1599) // 1600).clamp(max=300)


def load_baked_embeddings(
    cfg: TrainConfig, subset: str = "train"
) -> tuple[torch.Tensor, list[str], list[str], torch.Tensor]:
    """Load a subset's shards into memory: fp16 embeddings [N, 300, 3840], ids, transcripts, audio lengths."""
    _, _, shard_dir = subset_spec(cfg, subset)
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No baked embeddings in {shard_dir}; run with stage=bake first.")
    shards = [torch.load(path) for path in shard_paths]
    embeddings = torch.cat([shard["embeddings"] for shard in shards])
    ids = [i for shard in shards for i in shard["ids"]]
    texts = [t for shard in shards for t in shard["texts"]]
    if any("num_samples" not in shard for shard in shards):
        raise ValueError(f"Shards in {shard_dir} lack audio lengths; re-bake with overwrite=True.")
    num_samples = torch.cat([shard["num_samples"] for shard in shards])
    if len(set(ids)) != len(ids):
        raise ValueError("Baked shards contain duplicate example IDs.")
    return embeddings, ids, texts, num_samples
