"""Summarise the runs of scripts/law_runs.sh: step of the loss drop and of the audio-gap milestones per run.

The drop is the first step where the 21-step median training loss is below 3.0. The audio gap is the loss with another
utterance's audio minus the loss with the right audio (analysis/dense_analysis.py); its milestones are the first checkpoints
where it reaches 0.1, 0.3 and 1.0 (linear interpolation between checkpoints).

Usage: python analysis/law_summary.py [results/law]
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/law")
runs = {"seed1 (dense)": (Path("results/dense/clip1"), 1, 1e-4, 4, 1.0)}
log = Path(str(out_dir) + ".log")
for line in log.read_text().splitlines() if log.exists() else []:
    match = re.match(r"=== (\S+): seed=(\S+) lr=(\S+) batch=(\S+) init_scale=(\S+) steps=(\S+)", line)
    if match:
        name, seed, lr, batch, scale, _ = match.groups()
        runs[name] = (out_dir / name, int(seed), float(lr), int(batch), float(scale))


def drop_step(path: Path, level: float = 3.0):
    losses = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(path / "train_log.csv", encoding="utf-8"))}
    for step in range(100, max(losses) - 10):
        if statistics.median(losses[s] for s in range(step - 10, step + 11) if s in losses) < level:
            return step
    return None


def milestones(path: Path, levels=(0.1, 0.3, 1.0)):
    file = path / "dense_analysis" / "dense_analysis.json"
    if not file.exists():
        return None, [None] * len(levels)
    rows = [r for r in json.loads(file.read_text())["rows"] if r["step"] > 0]
    steps = np.array([r["step"] for r in rows], dtype=float)
    gap = np.array([r["loss_swapped"] - r["loss_correct"] for r in rows])
    found = []
    for level in levels:
        above = np.where(gap >= level)[0]
        if len(above) == 0:
            found.append(None)
        elif above[0] == 0:
            found.append(float(steps[0]))
        else:
            i = above[0]
            found.append(float(steps[i - 1] + (level - gap[i - 1]) / (gap[i] - gap[i - 1]) * (steps[i] - steps[i - 1])))
    return (steps, gap), found


lines = [f"{'run':>14} {'seed':>4} {'lr':>7} {'bs':>3} {'init':>5} | {'drop<3':>7} | gap0.1 gap0.3 gap1.0 | drop-gap0.1"]
curves = {}
for name, (path, seed, lr, batch, scale) in runs.items():
    if not (path / "train_log.csv").exists():
        continue
    drop = drop_step(path)
    curve, found = milestones(path)
    if curve is not None:
        curves[name] = curve
    fmt = lambda v: f"{v:6.0f}" if v is not None else "     -"
    delta = f"{drop - found[0]:7.0f}" if drop is not None and found[0] is not None else "      -"
    lines.append(f"{name:>14} {seed:>4} {lr:>7g} {batch:>3} {scale:>5g} | {fmt(drop):>7} | {fmt(found[0])} {fmt(found[1])} {fmt(found[2])} | {delta}")
text = "\n".join(lines)
print(text)
(out_dir / "summary.txt").write_text(text + "\n")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (path, *_rest) in runs.items():
    if not (path / "train_log.csv").exists():
        continue
    losses = {int(r["step"]): float(r["loss"]) for r in csv.DictReader(open(path / "train_log.csv", encoding="utf-8"))}
    steps = sorted(losses)
    smooth = [statistics.median(losses[s] for s in range(t - 10, t + 11) if s in losses) for t in steps]
    axes[0].plot(steps, smooth, label=name)
    if name in curves:
        axes[1].plot(*curves[name], "o-", ms=3, label=name)
axes[0].set(title="training loss (21-step median)", xlabel="step")
axes[1].set(title="loss with other audio minus loss with right audio", xlabel="step", yscale="log", ylim=(0.01, 8))
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out_dir / "summary.png", dpi=130)
