"""Compare short training runs: train vs validation loss, gradient norms, frame norms and learning rate.

Reads train_log.csv and wer_history.json from each run directory; legends use the directory names.

python analysis/plot_runs.py out=results/short_sweep/compare.png results/short_sweep/a results/short_sweep/b
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Moving mean, with the x offsets that line it up with the steps."""
    window = min(window, len(values))
    return np.arange(window - 1, len(values)), np.convolve(values, np.ones(window) / window, mode="valid")


def load(run: Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    with open(run / "train_log.csv", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        data = np.array([[float(x) for x in line.split(",")] for line in handle if line.strip()])
    history = json.loads((run / "wer_history.json").read_text()) if (run / "wer_history.json").exists() else []
    return {name: data[:, i] for i, name in enumerate(header)}, history


def main() -> None:
    args = sys.argv[1:]
    out = Path(next((a.split("=", 1)[1] for a in args if a.startswith("out=")), "compare.png"))
    runs = [Path(a) for a in args if not a.startswith("out=")]
    loaded = [(run.name, *load(run)) for run in runs]
    colours = plt.cm.tab10(np.arange(len(runs)))
    window = 25

    columns = max(len(runs), 3)
    fig, axes = plt.subplots(3, columns, figsize=(4.6 * columns, 11), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    # train vs validation loss, one panel per run
    for column, (name, log, history) in enumerate(loaded):
        ax = axes[0, column]
        ax.axis("on")
        steps = log["step"]
        ax.plot(steps, log["loss"], color="lightgray", lw=0.6, label="train (per batch)")
        x, y = smooth(log["loss"], window)
        ax.plot(steps[x], y, color="tab:blue", lw=1.6, label=f"train ({window}-step mean)")
        val = [(row["step"], row["val_loss"]) for row in history if "val_loss" in row and not row.get("final")]
        if val:
            ax.plot(*zip(*val), color="tab:red", marker="o", ms=3, lw=1.4, label="validation")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("step")
        ax.set_ylabel("cross-entropy loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # gradient norms, total and per layer, all runs on shared axes
    ax_total, ax_l1, ax_l2 = axes[1, 0], axes[1, 1], axes[1, 2]
    for ax, key, title in (
        (ax_total, "grad_norm", "gradient norm, whole projector (before clipping)"),
        (ax_l1, "grad_norm_layer1", "gradient norm, first linear layer"),
        (ax_l2, "grad_norm_layer2", "gradient norm, second linear layer"),
    ):
        ax.axis("on")
        for colour, (name, log, _) in zip(colours, loaded):
            ax.plot(log["step"], log[key], color=colour, lw=0.4, alpha=0.35)
            x, y = smooth(log[key], window)
            ax.plot(log["step"][x], y, color=colour, lw=1.5, label=name)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)

    # training loss, frame norm and learning rate across runs
    ax_loss, ax_norm, ax_lr = axes[2, 0], axes[2, 1], axes[2, 2]
    for ax in (ax_loss, ax_norm, ax_lr):
        ax.axis("on")
    for colour, (name, log, _) in zip(colours, loaded):
        x, y = smooth(log["loss"], window)
        ax_loss.plot(log["step"][x], y, color=colour, label=name)
        ax_norm.plot(log["step"], log["proj_norm"], color=colour, label=name)
        ax_lr.plot(log["step"], log["lr"], color=colour, label=name)
    ax_loss.set_title(f"training loss ({window}-step mean)", fontsize=9)
    ax_norm.set_title("mean norm of projected speech frames (text embeddings: ~0.9)", fontsize=9)
    ax_norm.set_yscale("log")
    ax_lr.set_title("learning rate", fontsize=9)
    for ax in (ax_loss, ax_norm, ax_lr):
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
