"""Summarise results/test_eval: test WER per model, seed mean +- std per group, and a 95% bootstrap interval over
utterances (resampling utterances, corpus-level WER each time) for single models and for the seed-pooled groups."""
import json
from pathlib import Path

import numpy as np

test_dir = Path("results/test_eval")
results = json.loads((test_dir / "wer_results.json").read_text())
results["whisper_small"] = json.loads((test_dir / "whisper_wer.json").read_text())
rng = np.random.default_rng(0)
GROUPS = {
    "Whisper-small (own decoder)": ["whisper_small"],
    "untrained MLP projector": ["init_mlp"],
    "untrained linear projector": ["init_linear"],
    "linear projector": ["linear_s1", "linear_s2", "linear_s3"],
    "MLP": ["mlp_s1", "mlp_s2", "mlp_s3"],
    "MLP + penalty 0.1": ["pen0.1_s1", "pen0.1_s2", "pen0.1_s3"],
    "MLP + penalty 1 (1,500 steps)": ["pen1"],
    "MLP + penalty 10 (1,500 steps)": ["pen10"],
    "MLP, 12,000 steps": ["long12000"],
}
lines = [f"{'group':34s} {'WER':>7s}  per-seed / bootstrap 95% CI"]
for group, names in GROUPS.items():
    wers = np.array([results[n]["wer"] for n in names])
    errors = np.array([results[n]["errors"] for n in names])
    words = np.array([results[n]["words"] for n in names])
    boot = []
    for _ in range(2000):
        index = rng.integers(0, errors.shape[1], errors.shape[1])
        boot.append(np.mean(errors[:, index].sum(1) / words[:, index].sum(1)))
    low, high = np.percentile(boot, [2.5, 97.5])
    spread = f" +- {wers.std(ddof=1):.3f} (seed std)" if len(wers) > 1 else ""
    lines.append(f"{group:34s} {wers.mean():7.3f}{spread}  [{', '.join(f'{w:.3f}' for w in wers)}]  CI [{low:.3f}, {high:.3f}]")
(test_dir / "wer_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
