"""Validation loss against step, mean +- std over seeds, for the linear baseline and the MLP with and without the 0.1 penalty.

Reads the run.log of every finished run (a "restored the best projector" line marks a finished run) and writes
results/seeds/seeds_drop.png plus a summary of final loss and drop step (first eval below DROP_LOSS) per group.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DROP_LOSS = 2.5
GROUPS = {
    "linear": ("Linear projector", ["results/seeds/linear_s{}".format(s) for s in (1, 2, 3)]),
    "mlp": ("MLP", ["results/short_sweep/clip1"] + ["results/seeds/mlp_s{}".format(s) for s in (2, 3)]),
    "mlppen": ("MLP, norm penalty 0.1", ["results/short_sweep/clip1_pen1e-1"] + ["results/seeds/mlppen_s{}".format(s) for s in (2, 3)]),
}
LINE = re.compile(r"step (\d+) val_loss ([\d.]+)")


def read_curve(run: Path):
    log = run / "run.log"
    if not log.exists():
        return None
    text = log.read_text(errors="ignore")
    if "restored the best projector" not in text:
        return None
    points = {int(s) + 1: float(v) for s, v in LINE.findall(text)}
    steps = sorted(points)
    return np.array(steps), np.array([points[s] for s in steps])


fig, (ax_loss, ax_drop) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
drop_data = {}
for key, (label, runs) in GROUPS.items():
    curves = [c for c in (read_curve(Path(r)) for r in runs) if c is not None]
    if not curves:
        continue
    steps = curves[0][0]
    losses = np.stack([c[1] for c in curves])
    mean, std = losses.mean(0), losses.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(steps, dtype=float)
    line = ax_loss.plot(steps, mean, label=f"{label} (n={len(curves)})")[0]
    ax_loss.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=0.25)
    drops = [c[0][np.argmax(c[1] < DROP_LOSS)] for c in curves]
    drop_data[label] = (drops, losses[:, -1], line.get_color())
    print(f"{label}: n={len(curves)} drop step {drops} mean {np.mean(drops):.0f} +- {np.std(drops, ddof=1) if len(drops) > 1 else 0:.0f}; "
          f"final val loss {losses[:, -1].mean():.3f} +- {losses[:, -1].std(ddof=1) if len(curves) > 1 else 0:.3f}")

ax_loss.set_xlabel("step")
ax_loss.set_ylabel("validation loss")
ax_loss.set_title("Validation loss, mean $\\pm$ std over seeds")
ax_loss.legend()
ax_loss.grid(alpha=0.3)

for i, (label, (drops, _, color)) in enumerate(drop_data.items()):
    ax_drop.scatter([i] * len(drops), drops, color=color, alpha=0.6)
    ax_drop.errorbar(i, np.mean(drops), yerr=np.std(drops, ddof=1) if len(drops) > 1 else 0, color="k", capsize=5, marker="_", markersize=14)
ax_drop.set_xticks(range(len(drop_data)))
ax_drop.set_xticklabels([l.replace(", ", ",\n") for l in drop_data], fontsize=8)
ax_drop.set_ylabel(f"first step with val loss < {DROP_LOSS}")
ax_drop.set_title("Drop step (points: seeds; bar: mean $\\pm$ std)")
ax_drop.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/seeds/seeds_drop.png", dpi=150)
