import re
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

    # Evaluation; set *_samples to 0 to skip that split. Use 1 for a quick smoke test.
    val_split: str = "validation.clean"
    val_samples: int = 100
    test_split: str = "test.clean"
    test_samples: int = 100
    train_eval_samples: int = 100  # first N training examples used for the train WER curve
    eval_every: int = 100  # train/val WER every N training steps; 0 = only after training
    eval_batch_size: int = 8
    eval_max_new_tokens: int = 200

    output_path: Path = MISSING
    seed:int = 67
    overwrite: bool = False

    steps: int = 300
    lr: float = 1e-3


def normalize_text(text: str) -> str:
    """Uppercase and strip punctuation (keeping apostrophes) to match LibriSpeech transcripts."""
    return " ".join(re.sub(r"[^A-Z' ]", " ", text.upper()).split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
            )
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER: total word edits divided by total reference words."""
    errors = 0
    words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = normalize_text(reference).split()
        errors += _edit_distance(ref_words, normalize_text(hypothesis).split())
        words += len(ref_words)
    return errors / max(words, 1)


def plot_wer(history: list[dict], path: Path) -> None:
    """Plot train and validation WER against training step and save it to path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("train_wer", "train"), ("val_wer", "validation")):
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            ax.plot(*zip(*points), marker="o", label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("WER")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
