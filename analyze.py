"""Analyse what the trained projector maps speech frames to in the LLM's embedding space.

For each projector checkpoint (plus an untrained control) this measures
  * nearest-neighbour retrieval of the spoken words among the LM's token embeddings,
  * cosine / norm statistics versus text-text baselines and a speech-vs-text linear probe,
  * the WER when projected frames are replaced by their nearest token embedding.

Usage: python analyze.py output_path=./results subset=val
"""
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from bake import load_baked_embeddings, speech_frame_counts
from utils import (
    TrainConfig,
    build_linear_projector,
    build_projector,
    build_prompt_embeddings,
    load_lm,
    normalize_text,
    transcribe,
    word_error_rate,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


@dataclass
class AnalyzeConfig:
    """Configuration for the analysis."""

    output_path: Path = Path("./results")  # holds the baked shards, checkpoints and the analysis/ output
    subset: str = "val"  # train, val or test
    checkpoints: list[str] | None = None  # "init" and/or state_dict paths; None = init + all saved + final
    max_examples: int = 0  # 0 = every baked example
    wer_examples: int = 100  # examples used for the replacement-WER test
    skip_wer: bool = False
    batch_size: int = 4  # the [B, 300, vocab] similarity matrix must fit in memory
    max_new_tokens: int = 200
    viz_examples: int = 3  # utterances shown frame by frame (nearest token per 100 ms frame); 0 = none
    out_name: str = "analysis"  # folder under output_path for the results; change it to keep an earlier analysis
    seed: int = 67


HIT_KS = (1, 5, 10)
REPLACEMENT_MODES = ("nn", "nn_rescaled", "random")


def word_token_variants(tokenizer, word: str) -> list[tuple[int, ...]]:
    """Token sequences that spell a word: cased and lower/capitalised, with and without a leading space."""
    forms = {word, word.lower(), word.capitalize()}
    forms |= {" " + form for form in forms}
    return [tuple(tokenizer(form, add_special_tokens=False).input_ids) for form in forms]


def word_recall(tokenizer, texts: list[str], neighbour_sets: list[set[int]]) -> tuple[float, float]:
    """Fraction of transcript words that have a variant fully contained in the utterance's neighbour set.

    Also returns the fraction of words that have a single-token variant (the words this metric is most exact for).
    """
    hits = words = single = 0
    for text, neighbours in zip(texts, neighbour_sets):
        for word in normalize_text(text).split():
            variants = word_token_variants(tokenizer, word)
            words += 1
            single += any(len(variant) == 1 for variant in variants)
            hits += any(set(variant) <= neighbours for variant in variants)
    return hits / max(words, 1), single / max(words, 1)


def separability_probe(speech: torch.Tensor, text: torch.Tensor, generator: torch.Generator) -> float:
    """Held-out accuracy of a logistic-regression probe separating speech frames from token embeddings."""
    count = min(len(speech), len(text), 5000)
    speech = speech[torch.randperm(len(speech), generator=generator)[:count]]
    text = text[torch.randperm(len(text), generator=generator)[:count]]
    features = torch.cat([speech, text])
    labels = torch.cat([torch.ones(count), torch.zeros(count)]).to(features.device)
    order = torch.randperm(len(features), generator=generator)
    features, labels = features[order], labels[order]
    split = int(0.8 * len(features))
    mean, std = features[:split].mean(0), features[:split].std(0) + 1e-6
    features = (features - mean) / std
    probe = torch.nn.Linear(features.shape[1], 1).to(features.device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(300):
        loss = F.binary_cross_entropy_with_logits(probe(features[:split]).squeeze(-1), labels[:split])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        predictions = (probe(features[split:]).squeeze(-1) > 0).float()
    return (predictions == labels[split:]).float().mean().item()


def load_projector(spec: str, in_dim: int, lm_dim: int, seed: int) -> torch.nn.Module:
    """'init' gives a freshly initialised (untrained) projector; anything else is a state_dict path."""
    if spec == "init":
        torch.manual_seed(seed)
        return build_projector(in_dim, lm_dim, device).eval()
    state = torch.load(spec, map_location=device)
    builder = build_linear_projector if "2.weight" not in state else build_projector  # the linear baseline has no hidden layer
    projector = builder(in_dim, lm_dim, device)
    projector.load_state_dict(state)
    return projector.eval()


def checkpoint_label(spec: str) -> tuple[str, float]:
    """Display label and x-axis position (training step) of a checkpoint spec."""
    if spec == "init":
        return "init", 0
    match = re.search(r"step(\d+)", spec)
    if match:
        return f"step{match.group(1)}", int(match.group(1))
    return "final", float("inf")


def analyse_checkpoint(spec, args, ctx) -> tuple[dict, dict]:
    """Compute every metric for one projector. Returns (metrics, arrays for plotting)."""
    tokenizer, lm = ctx["tokenizer"], ctx["lm"]
    text_embeddings = ctx["text_embeddings"]  # [V, D]
    text_normalised = F.normalize(text_embeddings, dim=-1)
    embeddings, texts, frame_counts = ctx["embeddings"], ctx["texts"], ctx["frame_counts"]
    prefix, suffix = ctx["prefix"], ctx["suffix"]
    projector = load_projector(spec, embeddings.shape[-1], text_embeddings.shape[-1], args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    wer_count = 0 if args.skip_wer else min(args.wer_examples, len(texts))

    sets_per_k: dict[int, list[set[int]]] = {k: [] for k in HIT_KS}
    speech_cos, silence_cos, speech_norm, top1_counter = [], [], [], Counter()
    speech_frames = []
    frame_views = []
    hypotheses = {mode: [] for mode in ("none", *REPLACEMENT_MODES)}
    for start in range(0, len(texts), args.batch_size):
        batch = embeddings[start : start + args.batch_size].to(device=device, dtype=torch.float32)
        counts = frame_counts[start : start + args.batch_size]
        speech_mask = torch.arange(batch.shape[1])[None, :] < counts[:, None]  # [B, 300]
        speech_mask = speech_mask.to(device)
        with torch.no_grad():
            projected = projector(batch)  # [B, 300, D]
            similarities = F.normalize(projected, dim=-1) @ text_normalised.T  # [B, 300, V]
            top_values, top_ids = similarities.topk(max(HIT_KS), dim=-1)
        for position in range(len(batch)):
            frames = top_ids[position, : int(counts[position])]
            for k in HIT_KS:
                sets_per_k[k].append(set(frames[:, :k].flatten().tolist()))
            if start + position < args.viz_examples:
                frame_views.append({
                    "text": texts[start + position],
                    "ids": frames[:, :3].cpu(),
                    "cos": top_values[position, : int(counts[position]), :3].cpu(),
                    "vectors": projected[position, : int(counts[position])].cpu(),
                })
        speech_cos.append(top_values[..., 0][speech_mask])
        silence_cos.append(top_values[..., 0][~speech_mask])
        speech_norm.append(projected.norm(dim=-1)[speech_mask])
        top1_counter.update(top_ids[..., 0][speech_mask].tolist())
        speech_frames.append(projected[speech_mask])

        if start < wer_count:
            top1 = text_embeddings[top_ids[..., 0]]
            replacements = {
                "none": projected,
                "nn": top1,
                "nn_rescaled": F.normalize(top1, dim=-1) * projected.norm(dim=-1, keepdim=True),
                "random": text_embeddings[
                    torch.randint(len(text_embeddings), top1.shape[:2], generator=generator).to(device)
                ],
            }
            for mode, replacement in replacements.items():
                # Only speech frames are replaced; padding frames keep the projector's output.
                speech = torch.where(speech_mask[..., None], replacement, projected)
                hypotheses[mode] += transcribe(tokenizer, lm, prefix, suffix, speech, args.max_new_tokens)

    speech_cos, silence_cos = torch.cat(speech_cos), torch.cat(silence_cos)
    speech_norm, speech_frames = torch.cat(speech_norm), torch.cat(speech_frames)
    top1_total = sum(top1_counter.values())

    metrics: dict = {
        "top1_cos_speech": speech_cos.mean().item(),
        "top1_cos_silence": silence_cos.mean().item() if len(silence_cos) else None,
        "text_text_top1_cos": ctx["text_text_cos"],
        "norm_speech": speech_norm.mean().item(),
        "norm_text": ctx["text_norm"],
        "distinct_top1_tokens": len(top1_counter),
        "top1_hub_share": sum(count for _, count in top1_counter.most_common(10)) / max(top1_total, 1),
        "top1_hub_tokens": [tokenizer.decode([token]) for token, _ in top1_counter.most_common(10)],
        "probe_accuracy": separability_probe(speech_frames, text_embeddings[: len(tokenizer)], generator),
    }
    metrics.update(hit_metrics(tokenizer, texts, sets_per_k))
    metrics["speech_var_in_text_pcs"] = subspace_fraction(speech_frames.cpu(), ctx["text_mean"], ctx["text_pcs"])
    metrics["text_var_in_text_pcs"] = ctx["text_pc_fraction"]
    for view in frame_views:
        view["word_cos"], view["order_corr"] = word_alignment(view, tokenizer, text_embeddings)
    if frame_views:
        metrics["word_order_corr"] = sum(view["order_corr"] for view in frame_views) / len(frame_views)
    if wer_count:
        references = texts[:wer_count]
        for mode, hyps in hypotheses.items():
            metrics[f"wer_{mode}"] = word_error_rate(references, hyps)
    return metrics, {
        "speech_cos": speech_cos.cpu(),
        "speech_frames": speech_frames[:3000].cpu(),
        "frame_views": frame_views,
    }


def hit_metrics(tokenizer, texts: list[str], sets_per_k: dict[int, list[set[int]]]) -> dict:
    """Bag-of-words recall of transcript words among each utterance's frame neighbours, per k.

    The control scores each utterance's words against a *different* utterance's neighbours, so any
    recall above it comes from the audio rather than from frequent words being everyone's neighbour.
    """
    out: dict = {}
    for k, sets in sets_per_k.items():
        recall, single = word_recall(tokenizer, texts, sets)
        shifted = sets[1:] + sets[:1]
        control, _ = word_recall(tokenizer, texts, shifted)
        out[f"word_recall@{k}"] = recall
        out[f"word_recall@{k}_other_utterance"] = control
        out["single_token_word_fraction"] = single
    return out


def token_text(tokenizer, token_id: int) -> str:
    """Decoded token with newlines and non-ASCII characters escaped, so plots do not need CJK glyphs."""
    return tokenizer.decode([token_id]).encode("unicode_escape").decode("ascii")


def write_frame_views(label: str, views: list[dict], tokenizer, out_dir: Path) -> None:
    """Show, per 100 ms frame, which token embedding the projected frame is closest to.

    Writes frames_<label>.txt (top-3 tokens per frame) and frames_<label>.png (top-1 cosine over time,
    runs of 2+ frames with the same nearest token are labelled; green = the token is a spoken word of 2+ letters).
    """
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    lines = []
    fig, axes = plt.subplots(len(views), 1, figsize=(14, 3.2 * len(views)), squeeze=False)
    for ax, view in zip(axes[:, 0], views):
        # Single-letter words are skipped: they match hub tokens like ' i' by chance.
        words = {word for word in normalize_text(view["text"]).lower().split() if len(word) > 1}
        top1 = view["ids"][:, 0].tolist()
        cos = view["cos"][:, 0].numpy()
        is_word = [token_text(tokenizer, token).strip().lower() in words for token in top1]
        lines += [f"== {view['text']}", f"top-1 token is a spoken word in {sum(is_word)}/{len(is_word)} frames"]
        for frame, (ids, values) in enumerate(zip(view["ids"].tolist(), view["cos"].tolist())):
            neighbours = "  ".join(f"{token_text(tokenizer, i)!r} {v:.2f}" for i, v in zip(ids, values))
            lines.append(f"{frame * 0.1:5.1f}s  {neighbours}")
        lines.append("")

        times = [frame * 0.1 for frame in range(len(top1))]
        ax.bar(times, cos, width=0.1, align="edge", color=["tab:green" if hit else "tab:gray" for hit in is_word])
        start = 0
        for frame in range(1, len(top1) + 1):
            if frame == len(top1) or top1[frame] != top1[start]:
                if frame - start >= 2:
                    ax.text((start + frame) * 0.05, cos[start:frame].max() + 0.01, repr(token_text(tokenizer, top1[start])),
                            rotation=90, ha="center", va="bottom", fontsize=7)
                start = frame
        ax.set_ylim(0, max(cos.max() * 1.5, 0.2))
        ax.set_ylabel("top-1 cosine")
        ax.set_title(view["text"], fontsize=8, loc="left", wrap=True)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(f"nearest token per frame: {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"frames_{label}.png", dpi=150)
    plt.close(fig)
    (out_dir / f"frames_{label}.txt").write_text("\n".join(lines))


def subspace_fraction(x: torch.Tensor, mean: torch.Tensor, pcs: torch.Tensor) -> float:
    """Average share of a vector's (mean-centred) squared norm that lies in the span of the given components."""
    centred = x.float() - mean
    return ((centred @ pcs).pow(2).sum(-1) / centred.pow(2).sum(-1)).mean().item()


def word_alignment(view: dict, tokenizer, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Cosine of every frame to every spoken word's embedding, and how well the best frames follow word order.

    A word's embedding is the mean of its token embeddings (leading-space, lower-case form). The order
    correlation is the Pearson correlation between a word's position in the transcript and the time of the
    frame most similar to it (after removing each word's mean over frames); about 1 means monotonic alignment.
    """
    words = normalize_text(view["text"]).lower().split()
    word_vectors = torch.stack([
        text_embeddings[tokenizer(" " + word, add_special_tokens=False).input_ids].mean(0) for word in words
    ]).cpu()
    cos = F.normalize(view["vectors"].float(), dim=-1) @ F.normalize(word_vectors, dim=-1).T  # [frames, words]
    best_frame = (cos - cos.mean(0)).argmax(0).float()
    corr = torch.corrcoef(torch.stack([torch.arange(len(words)).float(), best_frame]))[0, 1].item()
    return cos, corr


def write_word_heatmaps(label: str, views: list[dict], out_dir: Path) -> None:
    """Frame x word cosine heatmaps: a diagonal band means frames resemble the words in the order spoken."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(views), 2, figsize=(15, 5 * len(views)), squeeze=False)
    for row, view in zip(axes, views):
        words = normalize_text(view["text"]).lower().split()
        cos = view["word_cos"].T.numpy()  # [words, frames]
        extent = (0, cos.shape[1] * 0.1, len(words) - 0.5, -0.5)
        for ax, data, title in ((row[0], cos, "cosine"), (row[1], cos - cos.mean(1, keepdims=True), "cosine minus word mean")):
            image = ax.imshow(data, aspect="auto", extent=extent, cmap="viridis" if data is cos else "RdBu_r",
                              **({} if data is cos else {"vmin": -abs(data).max(), "vmax": abs(data).max()}))
            ax.set_yticks(range(len(words)), words, fontsize=6)
            ax.set_xlabel("time (s)")
            ax.set_title(f"{title}; order corr {view['order_corr']:.2f}", fontsize=9)
            fig.colorbar(image, ax=ax)
    fig.suptitle(f"frame vs. spoken-word embeddings: {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"word_heatmap_{label}.png", dpi=120)
    plt.close(fig)


def write_paper_views(
    arrays: dict, tokenizer, text_embeddings: torch.Tensor, text_sample: torch.Tensor, out_dir: Path
) -> None:
    """The two views of Kumar et al. (2024, arXiv 2411.03866, Figs. 4 and 5), for every checkpoint side by side.

    paper_fig4_token_cosine.png: cosine of every transcript token's embedding to every projected frame, on the
    paper's fixed -1..1 viridis scale (rows: utterances, columns: checkpoints). A diagonal means the frames
    follow the transcript tokens in order.
    nearest_tokens.txt: each utterance's reference text, then the nearest vocabulary token of every frame in
    time order (no collapsing of repeats), one line per checkpoint, as in the paper's Fig. 5.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(arrays)
    utterances = min(len(arrays[label]["frame_views"]) for label in labels)
    if utterances == 0:
        return
    fig, axes = plt.subplots(utterances, len(labels), figsize=(3.4 * len(labels) + 1, 3.4 * utterances), squeeze=False)
    panels: dict[tuple[int, int], torch.Tensor] = {}
    lines = []
    for row in range(utterances):
        lines += [f"reference: {arrays[labels[0]]['frame_views'][row]['text']}"]
        for column, label in enumerate(labels):
            view = arrays[label]["frame_views"][row]
            token_ids = tokenizer(view["text"], add_special_tokens=False).input_ids
            tokens = F.normalize(text_embeddings[token_ids].float().cpu(), dim=-1)
            cos = tokens @ F.normalize(view["vectors"].float(), dim=-1).T  # [transcript tokens, frames]
            panels[row, column] = cos
            ax = axes[row, column]
            image = ax.imshow(cos.numpy(), aspect="auto", cmap="viridis", vmin=-1, vmax=1,
                              extent=(0, cos.shape[1] * 0.1, len(token_ids) - 0.5, -0.5))
            if column == 0:
                ax.set_yticks(range(len(token_ids)), [token_text(tokenizer, i) for i in token_ids], fontsize=5)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=9)
            if row == utterances - 1:
                ax.set_xlabel("time (s)")
            nearest = " ".join(tokenizer.decode([i]).replace("\n", "\\n") for i in view["ids"][:, 0].tolist())
            lines.append(f"  {label:>10}: {nearest}")
        lines.append("")
    fig.colorbar(image, ax=axes, fraction=0.015, label="cosine")
    fig.suptitle("transcript token vs. projected frame cosine (Kumar et al. Fig. 4 style)", fontsize=10)
    fig.savefig(out_dir / "paper_fig4_token_cosine.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    (out_dir / "nearest_tokens.txt").write_text("\n".join(lines), encoding="utf-8")

    # The raw cosines all sit within about +-0.1, so the fixed -1..1 scale hides them. Same panels with each
    # token's mean over frames removed (it is a stripe that does not depend on time) on a symmetric-log scale,
    # shared by all panels so checkpoints stay comparable.
    from matplotlib.colors import SymLogNorm

    centred = {key: cos - cos.mean(1, keepdim=True) for key, cos in panels.items()}
    limit = torch.cat([value.abs().flatten() for value in centred.values()]).quantile(0.995).item()
    norm = SymLogNorm(linthresh=limit / 10, vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(utterances, len(labels), figsize=(3.4 * len(labels) + 1, 3.4 * utterances), squeeze=False)
    for (row, column), data in centred.items():
        ax = axes[row, column]
        image = ax.imshow(data.numpy(), aspect="auto", cmap="RdBu_r", norm=norm,
                          extent=(0, data.shape[1] * 0.1, data.shape[0] - 0.5, -0.5))
        ax.set_yticks([])
        if row == 0:
            ax.set_title(labels[column], fontsize=9)
        if row == utterances - 1:
            ax.set_xlabel("time (s)")
    fig.colorbar(image, ax=axes, fraction=0.015, label="cosine minus token mean (symlog)")
    fig.suptitle("transcript token vs. projected frame cosine, token mean removed, symmetric log scale", fontsize=10)
    fig.savefig(out_dir / "paper_fig4_token_cosine_rescaled.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Histograms of the same cosines. Two references: random vocabulary tokens against the same frames (if the
    # transcript tokens score no higher, the frames carry no token identity), and the spread of unrelated
    # random directions, a Gaussian with standard deviation 1/sqrt(dimension).
    random_tokens = F.normalize(text_sample.float(), dim=-1)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.4), squeeze=False, sharex=True, sharey=True)
    bins = torch.linspace(-0.4, 0.4, 81)
    for column, label in enumerate(labels):
        ax = axes[0, column]
        views = arrays[label]["frame_views"][:utterances]
        transcript = torch.cat([panels[row, column].flatten() for row in range(utterances)])
        frames = F.normalize(torch.cat([view["vectors"].float() for view in views]), dim=-1)
        control = (frames @ random_tokens.T).flatten()
        control = control[torch.randperm(len(control))[: len(transcript) * 20]]
        sigma = 1 / frames.shape[-1] ** 0.5
        centres = (bins[:-1] + bins[1:]) / 2
        density = torch.exp(-0.5 * (centres / sigma) ** 2) / (sigma * (2 * torch.pi) ** 0.5)
        ax.hist(control.numpy(), bins=bins.numpy(), density=True, histtype="step", color="gray",
                label=f"random tokens (mean {control.mean():+.3f}, sd {control.std():.3f})")
        ax.hist(transcript.numpy(), bins=bins.numpy(), density=True, histtype="step", color="tab:red",
                label=f"transcript tokens (mean {transcript.mean():+.3f}, sd {transcript.std():.3f})")
        ax.plot(centres.numpy(), density.numpy(), "k:", lw=1, label=f"unrelated directions (sd {sigma:.3f})")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, None)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("cosine")
        ax.legend(fontsize=5, loc="lower center")
    axes[0, 0].set_ylabel("density")
    fig.suptitle("frame-to-token cosine histograms", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_histograms.png", dpi=130)
    plt.close(fig)


def write_geometry(label: str, data: dict, ctx: dict, out_dir: Path) -> None:
    """Three views of the embedding space: PCA fitted on token embeddings only, directions only, and a time trajectory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speech, text = data["speech_frames"].float(), ctx["text_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) PCA fitted on token embeddings alone: how far does speech sit outside the text subspace?
    components = ctx["text_pcs"][:, :2]
    for points, name in ((text, "token embeddings"), (speech, "projected speech")):
        axes[0].scatter(*((points - ctx["text_mean"]) @ components).T.numpy(), s=3, alpha=0.4, label=name)
    axes[0].set_title("PCA fitted on token embeddings")
    axes[0].legend()

    # (b) + (c) unit-normalised vectors, joint PCA: removes the norm gap and shows direction overlap
    both = F.normalize(torch.cat([text, speech]), dim=-1)
    both = both - both.mean(0)
    _, _, directions = torch.pca_lowrank(both, q=2, center=False)
    coords = (both @ directions[:, :2]).numpy()
    text_coords, speech_coords = coords[: len(text)], coords[len(text) :]
    axes[1].scatter(*text_coords.T, s=3, alpha=0.4, label="token embeddings")
    axes[1].scatter(*speech_coords.T, s=3, alpha=0.4, label="projected speech")
    axes[1].set_title("directions only (unit norm), joint PCA")
    axes[1].legend()

    first = len(data["frame_views"][0]["vectors"]) if data["frame_views"] else min(150, len(speech))
    trajectory = speech_coords[:first]
    axes[2].scatter(*text_coords.T, s=2, alpha=0.15, color="gray")
    axes[2].plot(*trajectory.T, color="k", alpha=0.3, lw=0.8)
    points = axes[2].scatter(*trajectory.T, c=range(first), cmap="viridis", s=12)
    fig.colorbar(points, ax=axes[2], label="frame (100 ms)")
    axes[2].set_title("first utterance over time (directions PCA)")

    fig.suptitle(f"embedding-space geometry: {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"geometry_{label}.png", dpi=130)
    plt.close(fig)


def plot_results(rows: list[dict], arrays: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps ={row["label"]: i for i, row in enumerate(rows)}

    # Word recall vs training step, with the other-utterance control and the untrained projector.
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [steps[row["label"]] for row in rows]
    for k in HIT_KS:
        line, = ax.plot(xs, [row[f"word_recall@{k}"] for row in rows], marker="o", label=f"top-{k}")
        ax.plot(xs, [row[f"word_recall@{k}_other_utterance"] for row in rows], ls="--", color=line.get_color(), alpha=0.6)
    ax.set_xticks(xs, [row["label"] for row in rows], rotation=30)
    ax.set_ylabel("word recall in frame neighbours (dashed: other utterance)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "word_recall.png", dpi=150)
    plt.close(fig)

    # Top-1 cosine similarity distributions.
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, data in arrays.items():
        ax.hist(data["speech_cos"].numpy(), bins=60, alpha=0.5, label=label, density=True)
    ax.axvline(rows[0]["text_text_top1_cos"], color="k", ls="--", label="text-text top-1")
    ax.set_xlabel("cosine similarity to nearest token embedding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top1_cosine.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(AnalyzeConfig), OmegaConf.from_cli())
    try:
        args = AnalyzeConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    except TypeError as e:
        logger.error(f"Error: {e}\n\nUsage: python analyze.py output_path='./results' subset=val")
        sys.exit(1)
    if args.subset not in ("train", "val", "test"):
        logger.error(f"Unknown subset '{args.subset}'; expected train, val or test.")
        sys.exit(1)
    args.output_path = Path(args.output_path)

    checkpoints = args.checkpoints
    if checkpoints is None:
        saved = sorted((args.output_path / "checkpoints").glob("projector_step*.pt"),
                       key=lambda path: int(re.search(r"step(\d+)", path.name).group(1)))
        final = args.output_path / "projector.pt"
        checkpoints = ["init", *map(str, saved), *([str(final)] if final.exists() else [])]

    cfg = TrainConfig(dataset_name="", max_samples=0, output_path=args.output_path)
    embeddings, _, texts, num_samples = load_baked_embeddings(cfg, args.subset)
    if args.max_examples:
        embeddings, texts, num_samples = embeddings[: args.max_examples], texts[: args.max_examples], num_samples[: args.max_examples]
    frame_counts = speech_frame_counts(num_samples)
    logger.info(f"{args.subset}: {len(texts)} examples, {frame_counts.float().mean():.0f} speech frames on average")

    tokenizer, lm = load_lm(device)
    text_embeddings = lm.get_input_embeddings().weight.detach()  # [V, D]
    prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)

    # Text-text baseline: how close is a token's nearest *other* token? (sampled; the full V x V is too big)
    generator = torch.Generator().manual_seed(args.seed)
    sample = text_embeddings[torch.randperm(len(tokenizer), generator=generator)[:2000].to(device)]
    similarities = F.normalize(sample, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
    similarities[similarities > 0.9999] = -1  # drop each token's similarity to itself
    text_cpu = text_embeddings[: len(tokenizer)].float().cpu()
    text_mean = text_cpu.mean(0)
    _, _, text_pcs = torch.pca_lowrank(text_cpu - text_mean, q=50, center=False)
    text_sample = text_cpu[torch.randperm(len(text_cpu), generator=generator)[:3000]]
    ctx = {
        "text_mean": text_mean, "text_pcs": text_pcs, "text_sample": text_sample,
        "text_pc_fraction": subspace_fraction(text_sample, text_mean, text_pcs),
        "tokenizer": tokenizer, "lm": lm, "text_embeddings": text_embeddings, "prefix": prefix, "suffix": suffix,
        "embeddings": embeddings, "texts": texts, "frame_counts": frame_counts,
        "text_text_cos": similarities.max(dim=-1).values.mean().item(),
        "text_norm": text_embeddings.norm(dim=-1).mean().item(),
    }

    out_dir = args.output_path / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for spec in checkpoints:
        label, step = checkpoint_label(spec)
        logger.info(f"analysing {label}")
        metrics, plot_data = analyse_checkpoint(spec, args, ctx)
        rows.append({"label": label, "step": step if step != float("inf") else None, "checkpoint": spec, **metrics})
        arrays[label] = plot_data
        if plot_data["frame_views"]:
            write_frame_views(label, plot_data["frame_views"], tokenizer, out_dir)
            write_word_heatmaps(label, plot_data["frame_views"], out_dir)
        write_geometry(label, plot_data, ctx, out_dir)
        logger.info(" ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))
        (out_dir / f"analysis_{args.subset}.json").write_text(json.dumps(rows, indent=2))
    plot_results(rows, arrays, out_dir)
    write_paper_views(arrays, tokenizer, text_embeddings, ctx["text_sample"], out_dir)
    logger.info(f"wrote results and figures to {out_dir}")


if __name__ == "__main__":
    main()
