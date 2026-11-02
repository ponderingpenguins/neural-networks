"""Compare the LM's hidden states on projected speech with its hidden states on the transcript text, layer by layer.

analyze.py compares the projector's output to the LM's *input* embeddings. The LM reads speech through its layers, so
alignment could exist in the hidden states without showing there. Here the LM runs twice per utterance, with the same
prompt: once on the projected speech and once on the ground-truth transcript tokens in place of the speech. Per layer:
  * last-position retrieval: is the speech run's last-position state (where the LM has read everything) closest to
    the text run's state for the same utterance among all utterances? (chance: 1/N)
  * pooled retrieval: the same with states averaged over speech frames and over transcript tokens
  * monotone alignment advantage: the best in-order assignment of frames to transcript tokens scores higher with the
    true token order than with shuffled orders (0 = no order information)
  * logit lens: the share of transcript words among the top-k tokens of the speech positions' states decoded through
    the LM head, minus the same share against another utterance's tokens
Features are centred (and per-dimension standardised) separately for speech and text, so a constant offset between
the modalities does not decide the result. Intervals are bootstrap 95% over utterances.

Usage: python analysis/hidden_alignment.py output_path=./results subset=val
"""
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pipeline modules live in the repo root
from analyze import checkpoint_label, load_projector, word_token_variants
from bake import load_baked_embeddings, speech_frame_counts
from utils import TrainConfig, build_prompt_embeddings, load_lm, normalize_text

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class HiddenConfig:
    """Configuration for the hidden-state analysis."""

    output_path: Path = Path("./results")  # holds the baked shards and checkpoints; output goes to out_name/
    subset: str = "val"
    checkpoints: list[str] | None = None  # "init" and/or state_dict paths; None = init + all saved + final
    max_examples: int = 0  # 0 = every baked example
    layers: list[int] | None = None  # hidden-state indices (0 = input embeddings); None = every 4th
    batch_size: int = 4
    shuffles: int = 10  # token-order shuffles for the alignment control
    recall_k: int = 10  # tokens per speech position for the logit-lens recall
    standardise: bool = True  # divide by the per-dimension std (after centring), tames outlier dimensions
    bootstrap: int = 2000
    out_name: str = "hidden_alignment"
    seed: int = 67


def monotone_alignment_scores(sims: np.ndarray) -> np.ndarray:
    """Best in-order assignment of every frame to one token (each token gets >= 1 frame).

    sims is [B, tokens, frames]; returns the mean similarity of the assigned frame-token pairs, [B].
    """
    batch, tokens, frames = sims.shape
    dp = np.full((batch, tokens, frames), -np.inf, dtype=np.float32)
    dp[:, 0, 0] = sims[:, 0, 0]
    for j in range(1, frames):
        dp[:, 0, j] = dp[:, 0, j - 1] + sims[:, 0, j]
    for i in range(1, tokens):
        for j in range(i, frames):
            dp[:, i, j] = np.maximum(dp[:, i, j - 1], dp[:, i - 1, j - 1]) + sims[:, i, j]
    return dp[:, -1, -1] / frames


def standardiser(rows: torch.Tensor, standardise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std over rows [n, D] used to centre (and scale) features of one modality."""
    std = rows.std(0) + 1e-6 if standardise else torch.ones(rows.shape[-1])
    return rows.mean(0), std


def normalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return F.normalize((x - mean) / std, dim=-1)


def retrieval_scores(speech: torch.Tensor, text: torch.Tensor, standardise: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-utterance top-1 hit and reciprocal rank of the matching text vector for each speech vector [N, D].

    NaN when either side has no variation across utterances (e.g. layer 0 at the shared last prompt token), where
    every similarity ties and the ranking is meaningless.
    """
    if speech.std(0).max() < 1e-6 or text.std(0).max() < 1e-6:
        return np.full(len(speech), np.nan), np.full(len(speech), np.nan)
    sims =normalise(speech, *standardiser(speech, standardise)) @ normalise(text, *standardiser(text, standardise)).T
    rank = (sims > sims.diag()[:, None]).sum(1) + 1
    return (rank == 1).float().numpy(), (1.0 / rank).numpy()


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval over utterances (NaN utterances are ignored)."""
    values = values[~np.isnan(values)]
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def text_pass(lm, tokenizer, prefix, suffix, texts, layers):
    """Run the LM on prefix + transcript tokens + suffix. Returns per-utterance token states [L, T, D] and last states [L, D]."""
    embed = lm.get_input_embeddings()
    tokens, last = [], []
    final_normed = True
    for index, text in enumerate(texts):
        ids = torch.tensor(tokenizer(text, add_special_tokens=False).input_ids, device=device)
        inputs = torch.cat([prefix, embed(ids)[None], suffix], dim=1)
        with torch.no_grad():
            if index == 0:
                # Is the last hidden state already after the final norm? (decides how the logit lens is applied)
                out = lm(inputs_embeds=inputs, output_hidden_states=True)
                final_normed = torch.allclose(out.logits[0, -1], lm.lm_head(out.hidden_states[-1][0, -1]), atol=1e-3)
                states = out.hidden_states
            else:
                states = lm.model(inputs_embeds=inputs, output_hidden_states=True).hidden_states
        start = prefix.shape[1]
        tokens.append(torch.stack([states[layer][0, start : start + len(ids)] for layer in layers]).cpu())
        last.append(torch.stack([states[layer][0, -1] for layer in layers]).cpu())
    return tokens, torch.stack(last), final_normed


def speech_pass(projector, lm, embeddings, counts, prefix, suffix, layers, args, apply_norm, word_variants, tokenizer):
    """Run the LM on projected speech. Returns frame states, last states and the logit-lens recall per layer."""
    frames, last, lens_hits = [], [], []
    num_layers = lm.config.num_hidden_layers
    head = lm.get_output_embeddings().weight
    start = prefix.shape[1]
    neighbour_sets = []
    for begin in range(0, len(embeddings), args.batch_size):
        batch = embeddings[begin : begin + args.batch_size].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            projected = projector(batch)
            inputs = torch.cat([prefix.expand(len(batch), -1, -1), projected, suffix.expand(len(batch), -1, -1)], dim=1)
            states = lm.model(inputs_embeds=inputs, output_hidden_states=True).hidden_states
            for position in range(len(batch)):
                count = int(counts[begin + position])
                selected = [states[layer][position, start : start + count] for layer in layers]
                frames.append(torch.stack(selected).cpu())
                last.append(torch.stack([states[layer][position, -1] for layer in layers]).cpu())
                sets = []
                for layer, hidden in zip(layers, selected):
                    if apply_norm or layer < num_layers:
                        hidden = lm.model.norm(hidden)
                    top = (hidden @ head.T).topk(args.recall_k, dim=-1).indices
                    sets.append(set(top.flatten().tolist()))
                neighbour_sets.append(sets)
    recall = np.zeros((len(layers), len(neighbour_sets)))
    control = np.zeros_like(recall)
    for u, variants_per_word in enumerate(word_variants):
        other = (u + 1) % len(neighbour_sets)
        for li in range(len(layers)):
            recall[li, u] = np.mean([any(set(v) <= neighbour_sets[u][li] for v in variants) for variants in variants_per_word])
            control[li, u] = np.mean([any(set(v) <= neighbour_sets[other][li] for v in variants) for variants in variants_per_word])
    return frames, torch.stack(last), recall, control


def analyse_checkpoint(spec, args, ctx, rng) -> dict:
    layers, lm = ctx["layers"], ctx["lm"]
    projector = load_projector(spec, ctx["embeddings"].shape[-1], lm.config.hidden_size, args.seed)
    frames, last, recall, control = speech_pass(
        projector, lm, ctx["embeddings"], ctx["counts"], ctx["prefix"], ctx["suffix"], layers, args,
        not ctx["final_normed"], ctx["word_variants"], ctx["tokenizer"],
    )
    n_utt, n_layers = len(frames), len(layers)
    per_utt = {name: np.zeros((n_layers, n_utt)) for name in
               ("last_top1", "last_mrr", "pooled_top1", "pooled_mrr", "dtw_advantage", "lens_gain")}
    per_utt["lens_gain"] = recall - control
    pooled_speech = torch.stack([f.mean(1) for f in frames])  # [N, L, D]
    pooled_text = torch.stack([t.mean(1) for t in ctx["token_states"]])
    all_frames = torch.cat(frames, dim=1)  # [L, total frames, D]
    all_tokens = torch.cat(ctx["token_states"], dim=1)
    for li in range(n_layers):
        per_utt["last_top1"][li], per_utt["last_mrr"][li] = retrieval_scores(last[:, li], ctx["text_last"][:, li], args.standardise)
        per_utt["pooled_top1"][li], per_utt["pooled_mrr"][li] = retrieval_scores(pooled_speech[:, li], pooled_text[:, li], args.standardise)
    frame_stats = [standardiser(all_frames[li], args.standardise) for li in range(n_layers)]
    token_stats = [standardiser(all_tokens[li], args.standardise) for li in range(n_layers)]
    def advantage(tokens: torch.Tensor, frames_u: torch.Tensor) -> np.ndarray:
        """Per layer: alignment score with the true token order minus the mean over shuffled orders."""
        if frames_u.shape[1] < tokens.shape[1]:
            return np.full(n_layers, np.nan)
        sims = np.stack([
            (normalise(tokens[li], *token_stats[li]) @ normalise(frames_u[li], *frame_stats[li]).T).numpy()
            for li in range(n_layers)
        ])  # [L, tokens, frames]
        variants = [sims] + [sims[:, rng.permutation(sims.shape[1])] for _ in range(args.shuffles)]
        scores = monotone_alignment_scores(np.concatenate(variants)).reshape(args.shuffles + 1, n_layers)
        return scores[0] - scores[1:].mean(0)

    # The mismatched control scores each utterance's speech against another utterance's transcript: any advantage
    # left comes from position or length trends shared by every utterance, not from the content.
    per_utt["dtw_mismatched"] = np.full((n_layers, n_utt), np.nan)
    per_utt["dtw_advantage"][:] = np.nan
    for u in range(n_utt):
        per_utt["dtw_advantage"][:, u] = advantage(ctx["token_states"][u], frames[u])
        if ctx["partners"][u] is not None:
            per_utt["dtw_mismatched"][:, u] = advantage(ctx["token_states"][ctx["partners"][u]], frames[u])
    metrics = {}
    for name, values in per_utt.items():
        intervals = np.array([bootstrap_interval(values[li], args.bootstrap, rng) for li in range(n_layers)])
        metrics[name] = {"mean": intervals[:, 0].tolist(), "lo": intervals[:, 1].tolist(), "hi": intervals[:, 2].tolist()}
    metrics["lens_recall"] = {"mean": recall.mean(1).tolist()}
    metrics["lens_control"] = {"mean": control.mean(1).tolist()}
    return metrics


def plot_layerwise(rows: list[dict], layers: list[int], n_utt: int, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("last_top1", "last-position retrieval (top-1)", 1 / n_utt),
        ("pooled_top1", "pooled retrieval (top-1)", 1 / n_utt),
        ("dtw_advantage", "monotone alignment advantage\n(true token order minus shuffled)", 0.0),
        ("lens_gain", "logit-lens word recall\nminus other-utterance control", 0.0),
    )
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    colours = plt.cm.viridis(np.linspace(0, 0.95, max(len(rows) - 1, 1)))
    for ax, (key, title, chance) in zip(axes, panels):
        for index, row in enumerate(rows):
            data = row["metrics"][key]
            colour = "gray" if row["label"] == "init" else colours[index - 1]
            ax.plot(layers, data["mean"], marker="o", ms=3, color=colour, label=row["label"],
                    ls="--" if row["label"] == "init" else "-")
            ax.fill_between(layers, data["lo"], data["hi"], color=colour, alpha=0.12)
        if key == "dtw_advantage":
            mismatched = rows[-1]["metrics"]["dtw_mismatched"]
            ax.plot(layers, mismatched["mean"], color="k", ls="--", lw=1.2, label=f"{rows[-1]['label']}, mismatched transcript")
            ax.fill_between(layers, mismatched["lo"], mismatched["hi"], color="k", alpha=0.1)
            ax.legend(fontsize=7)
        ax.axhline(chance, color="k", ls=":", lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("layer (0 = input embeddings)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("speech vs transcript hidden states, per layer (dotted: chance / no-information level; bands: 95% bootstrap)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(HiddenConfig), OmegaConf.from_cli())
    try:
        args = HiddenConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    except TypeError as e:
        logger.error(f"Error: {e}\n\nUsage: python analysis/hidden_alignment.py output_path='./results' subset=val")
        sys.exit(1)
    args.output_path = Path(args.output_path)
    rng = np.random.default_rng(args.seed)

    checkpoints = args.checkpoints
    if checkpoints is None:
        saved = sorted((args.output_path / "checkpoints").glob("projector_step*.pt"),
                       key=lambda path: int(re.search(r"step(\d+)", path.name).group(1)))
        final = args.output_path / "projector.pt"
        checkpoints = ["init", *map(str, saved), *([str(final)] if final.exists() else [])]

    cfg_train = TrainConfig(dataset_name="", max_samples=0, output_path=args.output_path)
    embeddings, _, texts, num_samples = load_baked_embeddings(cfg_train, args.subset)
    if args.max_examples:
        embeddings, texts, num_samples = embeddings[: args.max_examples], texts[: args.max_examples], num_samples[: args.max_examples]
    counts = speech_frame_counts(num_samples)
    logger.info(f"{args.subset}: {len(texts)} examples, {counts.float().mean():.0f} speech frames on average")

    tokenizer, lm = load_lm(device)
    layers = args.layers or list(range(0, lm.config.num_hidden_layers + 1, 4))
    prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)
    logger.info("running the LM on the transcripts")
    token_states, text_last, final_normed = text_pass(lm, tokenizer, prefix, suffix, texts, layers)
    logger.info(f"layers {layers}; last hidden state already normalised: {final_normed}")
    word_variants = [[word_token_variants(tokenizer, word) for word in normalize_text(text).split()] for text in texts]
    token_counts = [states.shape[1] for states in token_states]
    partners = [
        next((v for v in ((u + k) % len(texts) for k in range(1, len(texts))) if token_counts[v] <= int(counts[u])), None)
        for u in range(len(texts))
    ]
    ctx = {
        "partners": partners, "lm": lm, "tokenizer": tokenizer, "prefix": prefix, "suffix": suffix, "layers": layers,
        "embeddings": embeddings, "counts": counts, "token_states": token_states, "text_last": text_last,
        "final_normed": final_normed, "word_variants": word_variants,
    }

    # Control: how often does length alone pick the right transcript? Speech duration and token count are correlated.
    log_frames = np.log(counts.numpy().astype(float))
    log_tokens = np.log(np.array([states.shape[1] for states in token_states], dtype=float))
    slope, intercept = np.polyfit(log_frames, log_tokens, 1)
    distance = np.abs(log_tokens[None, :] - (intercept + slope * log_frames)[:, None])  # [speech i, text j]
    length_only_top1 = float(((distance < np.diag(distance)[:, None]).sum(1) == 0).mean())
    logger.info(f"length-only retrieval top-1: {length_only_top1:.3f} (chance {1 / len(texts):.3f})")

    out_dir = args.output_path / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in checkpoints:
        label, step = checkpoint_label(spec)
        logger.info(f"analysing {label}")
        metrics = analyse_checkpoint(spec, args, ctx, rng)
        rows.append({"label": label, "step": step if step != float("inf") else None, "checkpoint": spec, "metrics": metrics})
        for li, layer in enumerate(layers):
            logger.info(
                f"  layer {layer:2d}: last_top1={metrics['last_top1']['mean'][li]:.3f} pooled_top1={metrics['pooled_top1']['mean'][li]:.3f} "
                f"dtw_adv={metrics['dtw_advantage']['mean'][li]:+.4f} lens_gain={metrics['lens_gain']['mean'][li]:+.4f}"
            )
        (out_dir / f"hidden_alignment_{args.subset}.json").write_text(
            json.dumps({"layers": layers, "n_utterances": len(texts), "length_only_top1": length_only_top1,
                        "rows": rows}, indent=2))
    plot_layerwise(rows, layers, len(texts), out_dir)
    logger.info(f"wrote results and figures to {out_dir}")


if __name__ == "__main__":
    main()
