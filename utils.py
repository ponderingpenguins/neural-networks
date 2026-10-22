from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Configuration for the project."""

    dataset_split: str | None = None
    dataset_name: str = MISSING
    max_samples: int = MISSING
    batch_size: int = 4
    stage: str = "train"  # "bake" caches Whisper embeddings to disk, "train" trains on them
    bake_batch_size: int = 8

    output_path: Path = MISSING
    seed:int = 67
    overwrite: bool = False

    steps: int = 300
    lr: float = 1e-3
