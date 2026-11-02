"""Train loss, validation loss and the audio gap against step, one panel per run.

Train loss: 21-step median of the per-step loss in train_log.csv. Validation loss: the right-audio loss from
analysis/dense_analysis.py (the same utterances as the swapped-audio loss). Audio gap (right axis): swapped minus right-audio
loss. The dotted vertical line is the first step where the smoothed train loss is below 3.0.

Usage: python analysis/plot_gap_loss.py out=results/gap_loss.png run_dir [run_dir ...]
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
out = Path(args.pop(0).removeprefix("out="))
runs = [Path(a) for a in args if (Path(a) / "dense_analysis" / "dense_analysis.json").exists()]

columns = min(4, len(runs))
rows = -(-len(runs) // columns)
fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.6 * rows), squeeze=False, sharey=True)
for ax, run in zip(axes.flat, runs):
    losses = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(run / "train_log.csv", encoding="utf-8"))}
    steps = sorted(losses)
    smooth = [statistics.median(losses[s] for s in range(t - 10, t + 11) if s in losses) for t in steps]
    dense = json.loads((run / "dense_analysis" / "dense_analysis.json").read_text())
    rows_ = [r for r in dense["rows"] if r["step"] > 0]
    x = [r["step"] for r in rows_]
    ax.plot(steps, smooth, color="C0", label="train loss")
    ax.plot(x, [r["loss_correct"] for r in rows_], "o-", ms=3, color="C1", label="val loss (right audio)")
    ax.plot(x, [r["loss_swapped"] for r in rows_], "o--", ms=3, color="C1", alpha=0.5, label="val loss (swapped audio)")
    drop = next((t for t, v in zip(steps, smooth) if t >= 100 and v < 3.0), None)
    if drop is not None:
        ax.axvline(drop, color="k", ls=":", lw=1)
    ax.set(title=run.name + (f" (drop at {drop})" if drop else ""), xlabel="step", ylim=(0, 7.2))
    ax.grid(alpha=0.3)
    twin = ax.twinx()
    twin.plot(x, [r["loss_swapped"] - r["loss_correct"] for r in rows_], "s-", ms=3, color="C3", label="audio gap")
    twin.set_ylim(0, 5.5)
    twin.tick_params(axis="y", colors="C3")
    if ax is axes.flat[0]:
        ax.set_ylabel("loss")
        ax.legend(fontsize=7, loc="upper right")
        twin.legend(fontsize=7, loc="center right")
    twin.set_ylabel("audio gap (loss, nats/token)", color="C3")
for ax in list(axes.flat)[len(runs):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out} with {len(runs)} runs")
