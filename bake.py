import bisect
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

FRAME_GROUP_SIZE = 5

SUBSETS = ("train", "val", "test")


def subset_spec(cfg: TrainConfig, subset: str) -> tuple[str | None, int, Path]:
    """Dataset split, sample count and shard directory of a subset."""
    if subset == "train":
        return cfg.dataset_split, cfg.max_samples, Path(cfg.output_path) / "speech_embeddings"
    if subset == "val":
        return cfg.val_split, cfg.val_samples, Path(cfg.output_path) / "speech_embeddings_val"
    if subset == "test":
        return cfg.test_split, cfg.test_samples, Path(cfg.output_path) / "speech_embeddings_test"
    raise ValueError(f"Unknown subset '{subset}'; expected one of {SUBSETS}.")


def bake_speech_embeddings(cfg: TrainConfig, device: torch.device) -> None:
    """Encode every subset with a non-zero sample count and save the shards."""
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = whisper_model.model.encoder.to(device).eval()
    for subset in SUBSETS:
        split, sample_count, out_dir = subset_spec(cfg, subset)
        if sample_count > 0:
            logger.info(f"baking {subset}: {sample_count} examples from {split}")
            _bake_subset(cfg, processor, encoder, device, split, sample_count, out_dir)


def decode_audio(audio: dict) -> np.ndarray:
    """Decode FLAC/WAV bytes with soundfile, so FFmpeg and torchcodec are not needed."""
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
    """Encode `max_samples` recordings with Whisper and save the grouped embeddings, one shard per batch.

    A shard holds fp16 embeddings [n, 300, 3840], ids, transcripts and the unpadded audio length in samples,
    which tells speech frames from padding. Existing shards are skipped unless cfg.overwrite is set, so an
    interrupted bake can resume.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(cfg.dataset_name, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False)).take(max_samples)

    iterator = iter(dataset)
    baked = 0
    shard_index = 0
    while baked < max_samples:
        examples = list(islice(iterator, cfg.bake_batch_size))
        if not examples:
            break
        for example in examples:
            example["audio"]["array"] = decode_audio(example["audio"])
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

        n, n_frames, dim = speech_embeddings.shape
        n_groups = n_frames // FRAME_GROUP_SIZE
        grouped = speech_embeddings[:, : n_groups * FRAME_GROUP_SIZE].reshape(
            n, n_groups, FRAME_GROUP_SIZE * dim
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
    """Grouped frames that cover real audio, at 1600 samples (100 ms, 5 Whisper frames) each."""
    return ((num_samples + 1599) // 1600).clamp(max=300)


class ShardedEmbeddings:
    """Memory-mapped shards indexed like one [N, 300, 3840] fp16 tensor.

    The train set does not fit in RAM, so rows are read from disk on demand. Takes an int, a slice or a
    list of indices.
    """

    def __init__(self, tensors: list[torch.Tensor]):
        self.tensors = tensors
        self.starts = [0]
        for tensor in tensors:
            self.starts.append(self.starts[-1] + len(tensor))
        self.shape = (self.starts[-1], *tensors[0].shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def _row(self, index: int) -> torch.Tensor:
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        shard = bisect.bisect_right(self.starts, index) - 1
        return self.tensors[shard][index - self.starts[shard]]

    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, int):
            return self._row(key)
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
        return torch.stack([self._row(int(index)) for index in key])


def load_baked_embeddings(
    cfg: TrainConfig, subset: str = "train", lazy: bool = False
) -> tuple[torch.Tensor | ShardedEmbeddings, list[str], list[str], torch.Tensor]:
    """Load a subset's shards: fp16 embeddings [N, 300, 3840], ids, transcripts and audio lengths.

    With lazy=True the embeddings stay memory-mapped (see ShardedEmbeddings) instead of being read into RAM.
    """
    _, _, shard_dir = subset_spec(cfg, subset)
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No baked embeddings in {shard_dir}; run with stage=bake first.")
    shards = [torch.load(path, mmap=lazy) for path in shard_paths]
    if lazy:
        embeddings = ShardedEmbeddings([shard["embeddings"] for shard in shards])
    else:
        embeddings = torch.cat([shard["embeddings"] for shard in shards])
    ids = [i for shard in shards for i in shard["ids"]]
    texts = [t for shard in shards for t in shard["texts"]]
    if any("num_samples" not in shard for shard in shards):
        raise ValueError(f"Shards in {shard_dir} lack audio lengths; re-bake with overwrite=True.")
    num_samples = torch.cat([shard["num_samples"] for shard in shards])
    if len(set(ids)) != len(ids):
        raise ValueError("Baked shards contain duplicate example IDs.")
    return embeddings, ids, texts, num_samples
