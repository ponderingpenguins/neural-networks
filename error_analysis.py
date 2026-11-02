"""Error analysis of one model's test transcripts (CPU only): substitution/insertion/deletion split, WER by reference
length, utterance-level WER distribution, most frequent word confusions, and the worst utterances.

Usage: python error_analysis.py model=long12000 [compare=whisper] [test_dir=results/test_eval]
Reads <test_dir>/wer_results.json (hypotheses per model, from eval_wer.py) and the references from the baked test shards.
Writes <test_dir>/error_analysis_<model>.txt and .png.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import normalize_text

args = dict(a.split("=", 1) for a in sys.argv[1:])
test_dir = Path(args.get("test_dir", "results/test_eval"))
model = args["model"]

references = []
for shard in sorted((test_dir / "speech_embeddings_test").glob("shard_*.pt")):
    references += torch.load(shard, mmap=True)["texts"]
results = json.loads((test_dir / "wer_results.json").read_text())
results["whisper_small"] = json.loads((test_dir / "whisper_wer.json").read_text())  # Whisper's own decoder
hypotheses = results[model]["hypotheses"]
assert len(references) == len(hypotheses)


def align(ref: list[str], hyp: list[str]):
    """Edit operations (op, ref_word, hyp_word) from a minimum-edit alignment."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


counts, confusions, per_utt = Counter(), Counter(), []
for ref_text, hyp_text in zip(references, hypotheses):
    ref, hyp = normalize_text(ref_text).split(), normalize_text(hyp_text).split()
    ops = align(ref, hyp)
    c = Counter(op for op, _, _ in ops)
    counts.update(c)
    confusions.update((r, h) for op, r, h in ops if op == "sub")
    errors = c["sub"] + c["ins"] + c["del"]
    per_utt.append({"ref": ref_text, "hyp": hyp_text, "n_ref": len(ref), "n_hyp": len(hyp), "err": errors, "ops": c,
                    "del_words": [r for op, r, _ in ops if op == "del"]})

n_words = sum(u["n_ref"] for u in per_utt)
wers = np.array([u["err"] / max(u["n_ref"], 1) for u in per_utt])
lengths = np.array([u["n_ref"] for u in per_utt])
lines = [f"model {model}: corpus WER {sum(u['err'] for u in per_utt) / n_words:.4f} on {len(per_utt)} utterances, {n_words} words"]
lines.append(f"substitutions {counts['sub'] / n_words:.4f}  insertions {counts['ins'] / n_words:.4f}  deletions {counts['del'] / n_words:.4f} (per reference word)")
lines.append(f"utterances with WER 0: {np.mean(wers == 0):.1%}; median utterance WER {np.median(wers):.3f}; WER > 0.5: {np.mean(wers > 0.5):.1%}")
lines.append(f"hypothesis more than 2x reference length (runaway/repetition): {np.mean([u['n_hyp'] > 2 * u['n_ref'] for u in per_utt]):.1%}; "
             f"hypothesis under half reference length (early stop): {np.mean([u['n_hyp'] < 0.5 * u['n_ref'] for u in per_utt]):.1%}")
edges = [0, 10, 20, 35, 60, 1000]
lines.append("WER by reference length (words):")
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (lengths >= lo) & (lengths < hi)
    if mask.any():
        errs = sum(u["err"] for u, m in zip(per_utt, mask) if m)
        lines.append(f"  {lo:>3}-{hi - 1 if hi < 1000 else '+':<4} n={mask.sum():>3}  WER {errs / lengths[mask].sum():.3f}")
lines.append("most frequent substitutions (reference -> hypothesis):")
lines += [f"  {r} -> {h}: {n}" for (r, h), n in confusions.most_common(15)]
deleted = Counter(w for u in per_utt for w in u["del_words"])
lines.append("most frequently deleted reference words: " + ", ".join(f"{w} ({n})" for w, n in deleted.most_common(10)))
lines.append("worst 8 utterances:")
for index in np.argsort(-wers)[:8]:
    u = per_utt[index]
    lines.append(f"  WER {wers[index]:.2f} ({u['n_ref']} words)\n    REF: {u['ref']}\n    HYP: {u['hyp'][:300]}")
lines.append("5 random utterances with errors:")
rng = np.random.default_rng(0)
for index in rng.choice(np.flatnonzero((wers > 0) & (wers <= 0.5)), size=5, replace=False):
    u = per_utt[index]
    lines.append(f"  WER {wers[index]:.2f}\n    REF: {u['ref']}\n    HYP: {u['hyp']}")

(test_dir / f"error_analysis_{model}.txt").write_text("\n".join(lines))
print("\n".join(lines))

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
a.hist(wers, bins=np.arange(0, 1.05, 0.05), color="C0")
a.set_xlabel("utterance WER (capped at 1 for display)")
a.set_ylabel("utterances")
a.set_title(f"{model}: per-utterance WER")
b.scatter(lengths, np.minimum(wers, 1.5), s=8, alpha=0.5)
b.set_xlabel("reference length (words)")
b.set_ylabel("utterance WER")
b.set_title("WER vs utterance length")
fig.tight_layout()
fig.savefig(test_dir / f"error_analysis_{model}.png", dpi=150)
