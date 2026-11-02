"""What does the projector change across the loss drop? Weight, activation and function-level views of a dense run.

Uses the checkpoints saved every few steps by scripts/dense_run.sh. For every checkpoint (validation utterances):
  * weight change since the first checkpoint: Frobenius norm, stable rank and concentration of each layer's change
  * output anatomy: share of the projected frames' variance explained by frame position, by the utterance, and what is
    left within the utterance (the content-like part); effective dimension (participation ratio) of the hidden
    ReLU layer and of the output; share of active hidden units
Between a "before" and an "after" checkpoint (loss right/swapped audio, to read off the audio gap):
  * straight-line interpolation of all weights
  * layer transplants: which of the two layers carries the change
  * the layer's change truncated to its top-k singular directions (the rest of the network at "after")

Usage: python analysis/projector_anatomy.py output_path=results/dense/clip1 before=350 after=700
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pipeline modules live in the repo root
import dense_analysis as da
from bake import load_baked_embeddings, speech_frame_counts
from utils import TrainConfig, build_projector, build_prompt_embeddings, load_lm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
device = da.device


@dataclass
class AnatomyConfig:
    output_path: Path = Path("./results")  # run directory with checkpoints/ and the baked val embeddings
    before: int = 350  # checkpoint steps compared (their gap should be ~0 and large)
    after: int = 700
    extra_pairs: list[int] | None = None  # a second interpolation pair [before, after]; default [500, 650]
    max_examples: int = 32
    alphas: int = 21
    out_name: str = "projector_anatomy"


def load_state(directory: Path, step: int) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in torch.load(directory / f"projector_step{step}.pt", map_location=device).items()}


def participation_ratio(rows: torch.Tensor) -> float:
    """(sum of covariance eigenvalues)^2 / sum of squared eigenvalues: how many directions carry the variance."""
    eig = torch.linalg.svdvals(rows - rows.mean(0)) ** 2
    return float(eig.sum() ** 2 / (eig**2).sum())


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(AnatomyConfig), OmegaConf.from_cli())
    args = AnatomyConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    args.output_path = Path(args.output_path)
    ckpt_dir = args.output_path / "checkpoints"
    steps = sorted(int(re.search(r"step(\d+)", p.name).group(1)) for p in ckpt_dir.glob("projector_step*.pt"))
    pairs = [(args.before, args.after), tuple(args.extra_pairs or [500, 650])]

    cfg_train = TrainConfig(dataset_name="", max_samples=0, output_path=args.output_path)
    embeddings, _, texts, num_samples = load_baked_embeddings(cfg_train, "val")
    count = min(args.max_examples, len(texts))
    embeddings, texts = embeddings[:count], texts[:count]
    counts = speech_frame_counts(num_samples)[:count]
    tokenizer, lm = load_lm(device)
    da.LM = lm
    prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)
    token_ids = [torch.tensor(tokenizer(t, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]) for t in texts]
    in_dim, lm_dim = embeddings.shape[-1], lm.config.hidden_size

    def make(state):
        projector = build_projector(in_dim, lm_dim, device)
        projector.load_state_dict(state)
        return projector.eval()

    @torch.no_grad()
    def project(projector):
        return torch.cat([projector(embeddings[i : i + 4].to(device, torch.float32)) for i in range(0, count, 4)])

    @torch.no_grad()
    def losses(state):
        projected = project(make(state))
        right = da.mean_loss(prefix, suffix, token_ids, 4, lambda a, b: projected[a:b])
        swapped = da.mean_loss(prefix, suffix, token_ids, 4, lambda a, b: projected[[(i + 1) % count for i in range(a, b)]])
        return right, swapped

    # ---- per-checkpoint weight change and output anatomy
    first = load_state(ckpt_dir, steps[0])
    series = []
    for step in steps:
        state = load_state(ckpt_dir, step)
        row = {"step": step}
        for name, key in (("layer1", "0.weight"), ("layer2", "2.weight")):
            delta = state[key] - first[key]
            row[f"{name}_delta_norm"] = float(delta.norm())
            if step == steps[0]:
                row[f"{name}_stable_rank"] = float("nan")
                continue
            energy = torch.linalg.svdvals(delta) ** 2
            row[f"{name}_stable_rank"] = float(energy.sum() / energy[0])
        row_norms = (state["0.weight"] - first["0.weight"]).norm(dim=1) ** 2
        top = row_norms.topk(max(1, len(row_norms) // 20)).values.sum() / row_norms.sum().clamp_min(1e-12)
        row["layer1_top5pct_units_share"] = float(top)

        with torch.no_grad():
            projected = project(make(state))
            hidden = torch.relu(embeddings.to(device, torch.float32) @ state["0.weight"].T + state["0.bias"])
        valid = torch.zeros(count, projected.shape[1], dtype=torch.bool, device=device)
        for u in range(count):
            valid[u, : int(counts[u])] = True
        mask = valid[..., None]
        y = projected.float()
        mu = y[valid].mean(0)
        r1 = (y - mu) * mask
        total = (r1**2).sum()
        pos = r1.sum(0) / valid.sum(0).clamp_min(1)[:, None]  # mean over utterances at each frame position
        r2 = (r1 - pos[None]) * mask
        utt = r2.sum(1) / valid.sum(1).clamp_min(1)[:, None]  # what is left in each utterance's mean
        r3 = (r2 - utt[:, None]) * mask
        row["var_position"] = float(((pos[None] * mask) ** 2).sum() / total)
        row["var_utterance"] = float(((utt[:, None] * mask) ** 2).sum() / total)
        row["var_content"] = float((r3**2).sum() / total)
        sample = torch.linspace(0, int(valid.sum()) - 1, 4000, device=device).long()
        row["pr_output"] = participation_ratio(y[valid][sample])
        row["pr_hidden"] = participation_ratio(hidden[valid][sample])
        row["active_units"] = float((hidden[valid] > 0).float().mean())
        series.append(row)
        logger.info(f"step {step}: position {row['var_position']:.2f} utterance {row['var_utterance']:.2f} content {row['var_content']:.2f} "
                    f"PR out {row['pr_output']:.1f} hidden {row['pr_hidden']:.1f} active {row['active_units']:.2f}")

    # ---- interpolation, transplants, truncation
    interpolation = {}
    for before, after in pairs:
        a, b = load_state(ckpt_dir, before), load_state(ckpt_dir, after)
        rows = []
        for alpha in np.linspace(0, 1, args.alphas):
            right, swapped = losses({k: (1 - alpha) * a[k] + alpha * b[k] for k in a})
            rows.append({"alpha": float(alpha), "right": right, "swapped": swapped})
        interpolation[f"{before}->{after}"] = rows
        logger.info(f"interpolation {before}->{after}: " + " ".join(f"{r['alpha']:.2f}:{r['right']:.2f}" for r in rows[::4]))

    before_state, after_state = load_state(ckpt_dir, args.before), load_state(ckpt_dir, args.after)
    layer_keys = {"layer1": ["0.weight", "0.bias"], "layer2": ["2.weight", "2.bias"]}
    transplants = {}
    for label, (from1, from2) in {"before": (before_state, before_state), "layer1 after": (after_state, before_state),
                                  "layer2 after": (before_state, after_state), "after": (after_state, after_state)}.items():
        state = {**{k: from1[k] for k in layer_keys["layer1"]}, **{k: from2[k] for k in layer_keys["layer2"]}}
        transplants[label] = dict(zip(("right", "swapped"), losses(state)))
        logger.info(f"transplant {label}: {transplants[label]}")

    truncation = {}
    for name, keys in layer_keys.items():
        weight = keys[0]
        u, s, vh = torch.linalg.svd(after_state[weight] - before_state[weight], full_matrices=False)
        rows = []
        for k in (0, 1, 2, 4, 8, 16, 32, 64, 128, len(s)):
            truncated = (u[:, :k] * s[:k]) @ vh[:k]
            right, swapped = losses({**after_state, weight: before_state[weight] + truncated})
            rows.append({"k": k, "right": right, "swapped": swapped, "energy": float((s[:k] ** 2).sum() / (s**2).sum())})
        truncation[name] = rows
        logger.info(f"truncation {name}: " + " ".join(f"k={r['k']}:{r['right']:.2f}" for r in rows))

    out_dir = args.output_path / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "projector_anatomy.json").write_text(json.dumps(
        {"before": args.before, "after": args.after, "series": series, "interpolation": interpolation,
         "transplants": transplants, "truncation": truncation}, indent=2))
    plot(series, interpolation, transplants, truncation, args, out_dir)
    logger.info(f"wrote {out_dir}")


def plot(series, interpolation, transplants, truncation, args, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in series]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    for key, label in (("var_position", "frame position"), ("var_utterance", "which utterance"), ("var_content", "left within utterance")):
        ax.plot(steps, [r[key] for r in series], "o-", ms=3, label=label)
    ax.set(title="share of the projected frames' variance explained by...", xlabel="step", ylim=(0, 1))
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(steps, [r["pr_output"] for r in series], "o-", ms=3, label="output (1024-d)")
    ax.plot(steps, [r["pr_hidden"] for r in series], "o-", ms=3, label="hidden ReLU layer (2048-d)")
    ax.set(title="effective dimension (participation ratio)", xlabel="step", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for key, label in (("layer1_delta_norm", "layer 1"), ("layer2_delta_norm", "layer 2")):
        ax.plot(steps, [r[key] for r in series], "o-", ms=3, label=f"|W - W(first)|, {label}")
    twin = ax.twinx()
    for key, label in (("layer1_stable_rank", "layer 1"), ("layer2_stable_rank", "layer 2")):
        twin.plot(steps, [r[key] for r in series], "--", marker="s", ms=3, label=f"stable rank of change, {label}", alpha=0.6)
    ax.set(title="how far the weights moved (solid) and the rank of the change (dashed)", xlabel="step")
    ax.legend(fontsize=7, loc="upper left")
    twin.legend(fontsize=7, loc="lower right")
    ax = axes[1, 0]
    for index, (name, rows) in enumerate(interpolation.items()):
        alpha = [r["alpha"] for r in rows]
        ax.plot(alpha, [r["right"] for r in rows], "o-", ms=3, color=f"C{index}", label=f"{name}, right audio")
        ax.plot(alpha, [r["swapped"] for r in rows], "o--", ms=3, color=f"C{index}", alpha=0.5, label=f"{name}, swapped audio")
    ax.set(title="loss along the straight line between two checkpoints' weights", xlabel="alpha (0 = before, 1 = after)")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    labels = list(transplants)
    positions = np.arange(len(labels))
    ax.bar(positions - 0.2, [transplants[l]["right"] for l in labels], 0.4, label="right audio")
    ax.bar(positions + 0.2, [transplants[l]["swapped"] for l in labels], 0.4, label="swapped audio")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(title=f"layer transplants ({args.before} -> {args.after})", ylabel="val loss")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for name, rows in truncation.items():
        ax.plot([max(r["k"], 0.5) for r in rows], [r["right"] for r in rows], "o-", ms=4, label=f"{name} change truncated to top-k")
    ax.axhline(transplants["after"]["right"], color="k", ls=":", label="after")
    ax.axhline(transplants["before"]["right"], color="gray", ls=":", label="before")
    ax.set(title="val loss with only the top-k singular directions of one layer's change", xlabel="k", xscale="log")
    ax.legend(fontsize=7)
    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "projector_anatomy.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
