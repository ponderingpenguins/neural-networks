"""What does the trained projector turn speech frames into, in the LM's embedding space?

Per checkpoint (plus an untrained control): nearest-token retrieval of the spoken words, cosine and norm
statistics against text-text baselines, and the WER when frames are replaced by their nearest token.

python analyze.py output_path=./results subset=val
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
    output_path: Path = Path("./results")  # baked shards, checkpoints, and the output folder
    subset: str = "val"  # train, val or test
    checkpoints: list[str] | None = None  # "init" and/or state_dict paths; None = init + all saved + final
    max_examples: int = 0  # 0 = every baked example
    wer_examples: int = 100  # examples for the replacement WER
    skip_wer: bool = False
    batch_size: int = 4  # the [B, 300, vocab] similarity matrix must fit in memory
    max_new_tokens: int = 200
    viz_examples: int = 3  # utterances plotted frame by frame; 0 = none
    out_name: str = "analysis"  # output folder under output_path
    seed: int = 67


HIT_KS = (1, 5, 10)
REPLACEMENT_MODES = ("nn", "nn_rescaled", "random")


def word_token_variants(tokenizer, word: str) -> list[tuple[int, ...]]:
    """Token sequences that spell a word: three casings, with and without a leading space."""
    forms = {word, word.lower(), word.capitalize()}
    forms |= {" " + form for form in forms}
    return [tuple(tokenizer(form, add_special_tokens=False).input_ids) for form in forms]


def word_recall(tokenizer, texts: list[str], neighbour_sets: list[set[int]]) -> tuple[float, float]:
    """Share of transcript words with a spelling variant fully inside the utterance's neighbour set.

    Also returns the share of words that have a single-token variant, where the metric is most exact.
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
    """Held-out accuracy of a logistic probe that tells speech frames from token embeddings."""
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
    """'init' is an untrained projector; anything else is a state_dict path."""
    if spec == "init":
        torch.manual_seed(seed)
        return build_projector(in_dim, lm_dim, device).eval()
    state = torch.load(spec, map_location=device)
    builder = build_linear_projector if "2.weight" not in state else build_projector  # the linear baseline has no second layer
    projector = builder(in_dim, lm_dim, device)
    projector.load_state_dict(state)
    return projector.eval()


def checkpoint_label(spec: str) -> tuple[str, float]:
    """Label and training step of a checkpoint (init is step 0; no step in the name means the final one)."""
    if spec == "init":
        return "init", 0
    match = re.search(r"step(\d+)", spec)
    if match:
        return f"step{match.group(1)}", int(match.group(1))
    return "final", float("inf")


class FrameStats:
    """Frame-level numbers collected while one projector runs over the examples."""

    def __init__(self):
        self.neighbour_sets: dict[int, list[set[int]]] = {k: [] for k in HIT_KS}
        self.speech_cos, self.silence_cos, self.speech_norm, self.speech_frames = [], [], [], []
        self.top1_counts = Counter()
        self.views = []

    def add(self, projected, top_values, top_ids, counts, speech_mask, texts, n_views):
        """Add a batch; its first `n_views` utterances also get a frame view."""
        for i in range(len(projected)):
            n = int(counts[i])
            frames = top_ids[i, :n]
            for k in HIT_KS:
                self.neighbour_sets[k].append(set(frames[:, :k].flatten().tolist()))
            if i < n_views:
                self.views.append({
                    "text": texts[i],
                    "ids": frames[:, :3].cpu(),
                    "cos": top_values[i, :n, :3].cpu(),
                    "vectors": projected[i, :n].cpu(),
                })
        self.speech_cos.append(top_values[..., 0][speech_mask])
        self.silence_cos.append(top_values[..., 0][~speech_mask])
        self.speech_norm.append(projected.norm(dim=-1)[speech_mask])
        self.top1_counts.update(top_ids[..., 0][speech_mask].tolist())
        self.speech_frames.append(projected[speech_mask])


def replacement_hypotheses(projected, top_ids, speech_mask, ctx, generator, max_new_tokens) -> dict[str, list[str]]:
    """Transcripts when speech frames are swapped for token embeddings. Padding frames are left alone."""
    text_embeddings = ctx["text_embeddings"]
    top1 = text_embeddings[top_ids[..., 0]]
    replacements = {
        "none": projected,
        "nn": top1,
        "nn_rescaled": F.normalize(top1, dim=-1) * projected.norm(dim=-1, keepdim=True),
        "random": text_embeddings[torch.randint(len(text_embeddings), top1.shape[:2], generator=generator).to(device)],
    }
    return {
        mode: transcribe(
            ctx["tokenizer"], ctx["lm"], ctx["prefix"], ctx["suffix"],
            torch.where(speech_mask[..., None], replacement, projected), max_new_tokens,
        )
        for mode, replacement in replacements.items()
    }


def summarize(stats: FrameStats, ctx: dict, generator: torch.Generator) -> tuple[dict, dict]:
    """Metrics row and plot arrays from the collected frames."""
    tokenizer, text_embeddings = ctx["tokenizer"], ctx["text_embeddings"]
    speech_cos, silence_cos = torch.cat(stats.speech_cos), torch.cat(stats.silence_cos)
    speech_norm, speech_frames = torch.cat(stats.speech_norm), torch.cat(stats.speech_frames)
    hubs = stats.top1_counts.most_common(10)

    metrics: dict = {
        "top1_cos_speech": speech_cos.mean().item(),
        "top1_cos_silence": silence_cos.mean().item() if len(silence_cos) else None,
        "text_text_top1_cos": ctx["text_text_cos"],
        "norm_speech": speech_norm.mean().item(),
        "norm_text": ctx["text_norm"],
        "distinct_top1_tokens": len(stats.top1_counts),
        "top1_hub_share": sum(count for _, count in hubs) / max(sum(stats.top1_counts.values()), 1),
        "top1_hub_tokens": [tokenizer.decode([token]) for token, _ in hubs],
        "probe_accuracy": separability_probe(speech_frames, text_embeddings[: len(tokenizer)], generator),
    }
    metrics.update(hit_metrics(tokenizer, ctx["texts"], stats.neighbour_sets))
    metrics["speech_var_in_text_pcs"] = subspace_fraction(speech_frames.cpu(), ctx["text_mean"], ctx["text_pcs"])
    metrics["text_var_in_text_pcs"] = ctx["text_pc_fraction"]
    for view in stats.views:
        view["word_cos"], view["order_corr"] = word_alignment(view, tokenizer, text_embeddings)
    if stats.views:
        metrics["word_order_corr"] = sum(view["order_corr"] for view in stats.views) / len(stats.views)
    arrays = {
        "speech_cos": speech_cos.cpu(),
        "speech_frames": speech_frames[:3000].cpu(),
        "frame_views": stats.views,
    }
    return metrics, arrays


def analyse_checkpoint(spec, args, ctx) -> tuple[dict, dict]:
    """Metrics and plot arrays for one projector."""
    text_embeddings = ctx["text_embeddings"]  # [V, D]
    text_normalised = F.normalize(text_embeddings, dim=-1)
    embeddings, texts, frame_counts = ctx["embeddings"], ctx["texts"], ctx["frame_counts"]
    projector = load_projector(spec, embeddings.shape[-1], text_embeddings.shape[-1], args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    wer_count = 0 if args.skip_wer else min(args.wer_examples, len(texts))

    stats = FrameStats()
    hypotheses = {mode: [] for mode in ("none", *REPLACEMENT_MODES)}
    for start in range(0, len(texts), args.batch_size):
        end = start + args.batch_size
        batch = embeddings[start:end].to(device=device, dtype=torch.float32)
        counts = frame_counts[start:end]
        speech_mask = (torch.arange(batch.shape[1])[None, :] < counts[:, None]).to(device)  # [B, 300]
        with torch.no_grad():
            projected = projector(batch)  # [B, 300, D]
            similarities = F.normalize(projected, dim=-1) @ text_normalised.T  # [B, 300, V]
            top_values, top_ids = similarities.topk(max(HIT_KS), dim=-1)
        stats.add(projected, top_values, top_ids, counts, speech_mask, texts[start:end], args.viz_examples - start)
        if start < wer_count:
            new = replacement_hypotheses(projected, top_ids, speech_mask, ctx, generator, args.max_new_tokens)
            for mode, hyps in new.items():
                hypotheses[mode] += hyps

    metrics, arrays = summarize(stats, ctx, generator)
    if wer_count:
        for mode, hyps in hypotheses.items():
            metrics[f"wer_{mode}"] = word_error_rate(texts[:wer_count], hyps)
    return metrics, arrays


def hit_metrics(tokenizer, texts: list[str], sets_per_k: dict[int, list[set[int]]]) -> dict:
    """Word recall among each utterance's frame neighbours, per k.

    The control scores the words against another utterance's neighbours, so anything above it comes from
    the audio and not from frequent tokens that are everyone's neighbour.
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
    """Decoded token with newlines and non-ASCII escaped, so plots need no CJK glyphs."""
    return tokenizer.decode([token_id]).encode("unicode_escape").decode("ascii")


def write_frame_views(label: str, views: list[dict], tokenizer, out_dir: Path) -> None:
    """Nearest token of every 100 ms frame.

    Writes frames_<label>.txt (top 3 per frame) and frames_<label>.png (top-1 cosine over time; runs of 2+ equal
    tokens are labelled, green when the token is a spoken word).
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
    """Mean share of the centred squared norm that falls inside the span of `pcs`."""
    centred = x.float() - mean
    return ((centred @ pcs).pow(2).sum(-1) / centred.pow(2).sum(-1)).mean().item()


def word_alignment(view: dict, tokenizer, text_embeddings: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Cosine of every frame to every spoken word, and how well the best frames follow word order.

    A word's vector is the mean embedding of its tokens (leading space, lower case). Order correlation is the
    Pearson correlation between a word's position and the time of its best-matching frame, after removing each
    word's mean over frames; about 1 means the frames follow the words in order.
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
    """Frame x word cosine heatmaps; a diagonal band means the frames follow the words in order."""
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
    """Figs. 4 and 5 of Kumar et al. (2024, arXiv 2411.03866) for every checkpoint side by side."""
    labels = list(arrays)
    utterances = min(len(arrays[label]["frame_views"]) for label in labels)
    if utterances == 0:
        return
    panels = write_token_cosine(arrays, labels, utterances, tokenizer, text_embeddings, out_dir)
    write_centred_cosine(panels, labels, utterances, out_dir)
    write_cosine_histograms(arrays, panels, labels, utterances, text_sample, out_dir)


def write_token_cosine(arrays, labels, utterances, tokenizer, text_embeddings, out_dir):
    """Cosine of every transcript token to every frame; rows are utterances, columns checkpoints.

    Also writes nearest_tokens.txt and returns the cosine matrices.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    return panels


def write_centred_cosine(panels, labels, utterances, out_dir) -> None:
    """The same panels with each token's mean over frames removed, on one symlog scale.

    The raw cosines sit within about 0.1, which the fixed -1..1 scale hides.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


def write_cosine_histograms(arrays, panels, labels, utterances, text_sample, out_dir) -> None:
    """Histograms of those cosines next to two references: random vocabulary tokens against the same frames,
    and random directions (a Gaussian with sd 1/sqrt(dim)).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    """Three PCA views: fitted on token embeddings, unit-norm directions, and the first utterance over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speech, text = data["speech_frames"].float(), ctx["text_sample"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # fitted on token embeddings only: how far does speech sit outside the text subspace?
    components = ctx["text_pcs"][:, :2]
    for points, name in ((text, "token embeddings"), (speech, "projected speech")):
        axes[0].scatter(*((points - ctx["text_mean"]) @ components).T.numpy(), s=3, alpha=0.4, label=name)
    axes[0].set_title("PCA fitted on token embeddings")
    axes[0].legend()

    # unit-norm vectors, joint PCA: removes the norm gap
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

    steps = {row["label"]: i for i, row in enumerate(rows)}

    # word recall per checkpoint; dashed is the other-utterance control
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

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, data in arrays.items():
        ax.hist(data["speech_cos"].numpy(), bins=60, alpha=0.5, label=label, density=True)
    ax.axvline(rows[0]["text_text_top1_cos"], color="k", ls="--", label="text-text top-1")
    ax.set_xlabel("cosine similarity to nearest token embedding")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top1_cosine.png", dpi=150)
    plt.close(fig)


def parse_args() -> AnalyzeConfig:
    cfg = OmegaConf.merge(OmegaConf.structured(AnalyzeConfig), OmegaConf.from_cli())
    try:
        args = AnalyzeConfig(**OmegaConf.to_container(cfg, resolve=True))
    except TypeError as e:
        logger.error(f"Error: {e}\n\nUsage: python analyze.py output_path='./results' subset=val")
        sys.exit(1)
    if args.subset not in ("train", "val", "test"):
        logger.error(f"Unknown subset '{args.subset}'; expected train, val or test.")
        sys.exit(1)
    args.output_path = Path(args.output_path)
    return args


def find_checkpoints(args: AnalyzeConfig) -> list[str]:
    """The given checkpoints, or init plus every saved step and the final projector."""
    if args.checkpoints is not None:
        return args.checkpoints
    saved = sorted((args.output_path / "checkpoints").glob("projector_step*.pt"),
                   key=lambda path: int(re.search(r"step(\d+)", path.name).group(1)))
    final = args.output_path / "projector.pt"
    return ["init", *map(str, saved), *([str(final)] if final.exists() else [])]


def build_context(args: AnalyzeConfig) -> dict:
    """Shared inputs: baked speech, the LM and token-embedding baselines."""
    cfg = TrainConfig(dataset_name="", max_samples=0, output_path=args.output_path)
    embeddings, _, texts, num_samples = load_baked_embeddings(cfg, args.subset)
    if args.max_examples:
        embeddings, texts, num_samples = embeddings[: args.max_examples], texts[: args.max_examples], num_samples[: args.max_examples]
    frame_counts = speech_frame_counts(num_samples)
    logger.info(f"{args.subset}: {len(texts)} examples, {frame_counts.float().mean():.0f} speech frames on average")

    tokenizer, lm = load_lm(device)
    text_embeddings = lm.get_input_embeddings().weight.detach()  # [V, D]
    prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)

    # nearest other token of each token, on a sample (the full V x V matrix is too big)
    generator = torch.Generator().manual_seed(args.seed)
    sample = text_embeddings[torch.randperm(len(tokenizer), generator=generator)[:2000].to(device)]
    similarities = F.normalize(sample, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
    similarities[similarities > 0.9999] = -1  # drop each token's match with itself
    text_cpu = text_embeddings[: len(tokenizer)].float().cpu()
    text_mean = text_cpu.mean(0)
    _, _, text_pcs = torch.pca_lowrank(text_cpu - text_mean, q=50, center=False)
    text_sample = text_cpu[torch.randperm(len(text_cpu), generator=generator)[:3000]]
    return {
        "text_mean": text_mean, "text_pcs": text_pcs, "text_sample": text_sample,
        "text_pc_fraction": subspace_fraction(text_sample, text_mean, text_pcs),
        "tokenizer": tokenizer, "lm": lm, "text_embeddings": text_embeddings, "prefix": prefix, "suffix": suffix,
        "embeddings": embeddings, "texts": texts, "frame_counts": frame_counts,
        "text_text_cos": similarities.max(dim=-1).values.mean().item(),
        "text_norm": text_embeddings.norm(dim=-1).mean().item(),
    }


def main() -> None:
    args = parse_args()
    checkpoints = find_checkpoints(args)
    ctx = build_context(args)
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
            write_frame_views(label, plot_data["frame_views"], ctx["tokenizer"], out_dir)
            write_word_heatmaps(label, plot_data["frame_views"], out_dir)
        write_geometry(label, plot_data, ctx, out_dir)
        logger.info(" ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))
        (out_dir / f"analysis_{args.subset}.json").write_text(json.dumps(rows, indent=2))
    plot_results(rows, arrays, out_dir)
    write_paper_views(arrays, ctx["tokenizer"], ctx["text_embeddings"], ctx["text_sample"], out_dir)
    logger.info(f"wrote results and figures to {out_dir}")


if __name__ == "__main__":
    main()
