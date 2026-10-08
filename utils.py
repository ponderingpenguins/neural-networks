from dataclasses import dataclass
from pathlib import Path
from omegaconf import MISSING
from typing import Optional

@dataclass
class TrainConfig:
    """Configuration for the project."""

    subset_name: Optional[str] = None
    dataset_name: str = MISSING
    max_samples: int = MISSING

    output_path: Path = MISSING
    seed:int = 67
    overwrite: bool = False
