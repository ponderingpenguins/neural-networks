"""Does the LM use the speech? Per-checkpoint measurements around the loss drop, from a run saved every few steps.

For every projector checkpoint of a run directory (plus the untrained one):
  * loss with the right audio, loss with another utterance's audio (same transcripts), and the loss with no audio at
    all (checkpoint independent). Before the drop the first two should both sit at the no-audio level.
  * attention mass that the transcript positions put on the speech positions, per layer (mean over heads)
  * statistics of the projected frames: mean norm, mean cosine between the frames of one utterance, mean cosine between
    different utterances' mean frames, and the mean best cosine to any vocabulary embedding

Usage: python analysis/dense_analysis.py output_path=results/dense/clip1 max_examples=64
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
from analyze import checkpoint_label, load_projector
from bake import load_baked_embeddings
from utils import TrainConfig, build_prompt_embeddings, load_lm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class DenseConfig:
    """Configuration for the per-checkpoint analysis."""

    output_path: Path = Path("./results")  # run directory with checkpoints/ and the baked val embeddings
    subset: str = "val"
    max_examples: int = 64  # utterances for the losses
    attention_examples: int = 8  # utterances for the attention and frame statistics
    batch_size: int = 4
    out_name: str = "dense_analysis"


def make_sequences(prefix, suffix, token_ids, speech=None):
    """Prompt embeddings (+ projected speech) + transcript embeddings, right-padded; returns inputs, mask, labels."""
    embed = LM.get_input_embeddings()
    parts = []
    starts = []
    for index, ids in enumerate(token_ids):
        pieces = [prefix[0]] + ([speech[index]] if speech is not None else []) + [suffix[0]]
        starts.append(sum(p.shape[0] for p in pieces))
        parts.append(torch.cat(pieces + [embed(ids.to(device))], dim=0))
    inputs = torch.nn.utils.rnn.pad_sequence(parts, batch_first=True, padding_value=0.0)
    mask = torch.zeros(inputs.shape[:2], dtype=torch.long, device=device)
    labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    for index, ids in enumerate(token_ids):
        mask[index, : parts[index].shape[0]] = 1
        labels[index, starts[index] : starts[index] + len(ids)] = ids.to(device)
    return inputs, mask, labels, starts


@torch.no_grad()
def mean_loss(prefix, suffix, token_ids, batch, speech_fn):
    """Token-weighted mean teacher-forced loss; speech_fn(start, stop) gives projected speech or None."""
    total, count = 0.0, 0
    for start in range(0, len(token_ids), batch):
        ids = token_ids[start : start + batch]
        inputs, mask, labels, _ = make_sequences(prefix, suffix, ids, speech_fn(start, start + len(ids)))
        loss = LM(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss
        tokens = sum(len(i) for i in ids)
        total += loss.item() * tokens
        count += tokens
    return total / count


@torch.no_grad()
def attention_mass(prefix, suffix, token_ids, speech):
    """Mean over utterances and heads of the attention that transcript-prediction positions put on the speech positions, per layer."""
    mass = []
    for index in range(len(token_ids)):
        inputs, mask, _, starts = make_sequences(prefix, suffix, token_ids[index : index + 1], speech[index : index + 1])
        out = LM(inputs_embeds=inputs, attention_mask=mask, output_attentions=True)
        if out.attentions is None or out.attentions[0] is None:
            raise RuntimeError("the LM returned no attention weights (eager attention not active)")
        speech_slice = slice(prefix.shape[1], prefix.shape[1] + speech.shape[1])
        queries = slice(starts[0] - 1, inputs.shape[1] - 1)  # positions whose next token is a transcript token
        mass.append([layer[0, :, queries, speech_slice].sum(-1).mean().item() for layer in out.attentions])
    return np.mean(mass, axis=0)


@torch.no_grad()
def frame_statistics(projected, vocab):
    """Norm, within-utterance and between-utterance cosine, best vocabulary cosine of projected frames [N, 300, D]."""
    unit = F.normalize(projected, dim=-1)
    within = []
    for u in unit:
        sims = u @ u.T
        within.append(((sims.sum() - sims.diag().sum()) / (len(u) * (len(u) - 1))).item())
    means = F.normalize(projected.mean(1), dim=-1)
    sims = means @ means.T
    between = ((sims.sum() - sims.diag().sum()) / (len(means) * (len(means) - 1))).item()
    best = torch.stack([(u @ vocab.T).max(-1).values.mean() for u in unit]).mean().item()
    return {"frame_norm": projected.norm(dim=-1).mean().item(), "within_cosine": float(np.mean(within)),
            "between_cosine": between, "best_vocab_cosine": best}


def plot(rows, no_audio_loss, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    ax = axes[0, 0]
    ax.plot(steps, [r["loss_correct"] for r in rows], "o-", ms=3, label="right audio")
    ax.plot(steps, [r["loss_swapped"] for r in rows], "o-", ms=3, label="another utterance's audio")
    ax.axhline(no_audio_loss, color="k", ls=":", label="no audio")
    ax.set(title="val loss", xlabel="step", ylabel="loss")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(steps, [r["loss_swapped"] - r["loss_correct"] for r in rows], "o-", ms=3, color="C3")
    ax.axhline(0, color="k", ls=":")
    ax.set(title="loss gap: swapped minus right audio\n(0 = the audio is not used)", xlabel="step")
    ax = axes[0, 2]
    attention = np.array([r["attention_by_layer"] for r in rows]).T  # [layers, steps]
    image = ax.imshow(attention, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, len(steps), -0.5, attention.shape[0] - 0.5])
    ax.set_xticks(np.arange(len(steps)) + 0.5)
    ax.set_xticklabels(steps, rotation=90, fontsize=6)
    ax.set(title="attention mass on speech positions\n(transcript queries, mean over heads)", ylabel="layer")
    fig.colorbar(image, ax=ax)
    ax = axes[1, 0]
    ax.plot(steps, [np.mean(r["attention_by_layer"]) for r in rows], "o-", ms=3, label="mean over layers")
    ax.plot(steps, [np.max(r["attention_by_layer"]) for r in rows], "o-", ms=3, label="best layer")
    ax.set(title="attention on speech", xlabel="step")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(steps, [r["frame_norm"] for r in rows], "o-", ms=3)
    ax.set(title="mean norm of projected frames (text embeddings: about 0.9)", xlabel="step", yscale="log")
    ax = axes[1, 2]
    for key in ("within_cosine", "between_cosine", "best_vocab_cosine"):
        ax.plot(steps, [r[key] for r in rows], "o-", ms=3, label=key)
    ax.set(title="projected frames: cosine statistics", xlabel="step")
    ax.legend(fontsize=8)
    for a in axes.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "dense_analysis.png", dpi=130)
    plt.close(fig)


def main() -> None:
    global LM
    cfg = OmegaConf.merge(OmegaConf.structured(DenseConfig), OmegaConf.from_cli())
    try:
        args = DenseConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    except TypeError as e:
        logger.error(f"Error: {e}\n\nUsage: python analysis/dense_analysis.py output_path=results/dense/clip1")
        sys.exit(1)
    args.output_path = Path(args.output_path)

    cfg_train = TrainConfig(dataset_name="", max_samples=0, output_path=args.output_path)
    embeddings, _, texts, _ = load_baked_embeddings(cfg_train, args.subset)
    count = min(args.max_examples, len(texts))
    embeddings, texts = embeddings[:count], texts[:count]
    tokenizer, LM = load_lm(device)
    prefix, suffix = build_prompt_embeddings(tokenizer, LM, device)
    token_ids = [torch.tensor(tokenizer(t, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]) for t in texts]
    vocab = F.normalize(LM.get_input_embeddings().weight.detach()[: len(tokenizer)], dim=-1)
    no_audio = mean_loss(prefix, suffix, token_ids, args.batch_size, lambda a, b: None)
    logger.info(f"{count} utterances; loss with no audio at all: {no_audio:.4f}")

    checkpoints = sorted((args.output_path / "checkpoints").glob("projector_step*.pt"),
                         key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)))
    specs = ["init", *map(str, checkpoints)]
    out_dir = args.output_path / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        label, step = checkpoint_label(spec)
        projector = load_projector(spec, embeddings.shape[-1], LM.config.hidden_size, 67)
        with torch.no_grad():
            projected = torch.cat([projector(embeddings[i : i + args.batch_size].to(device, torch.float32))
                                   for i in range(0, count, args.batch_size)])
        right = lambda a, b: projected[a:b]
        swapped = lambda a, b: projected[[(i + 1) % count for i in range(a, b)]]
        row = {"label": label, "step": step,
               "loss_correct": mean_loss(prefix, suffix, token_ids, args.batch_size, right),
               "loss_swapped": mean_loss(prefix, suffix, token_ids, args.batch_size, swapped)}
        n_att = min(args.attention_examples, count)
        try:
            LM.set_attn_implementation("eager")
        except AttributeError:
            LM.config._attn_implementation = "eager"
        row["attention_by_layer"] = attention_mass(prefix, suffix, token_ids[:n_att], projected[:n_att]).tolist()
        try:
            LM.set_attn_implementation("sdpa")
        except AttributeError:
            LM.config._attn_implementation = "sdpa"
        row.update(frame_statistics(projected[:n_att], vocab))
        rows.append(row)
        logger.info(f"{label}: right {row['loss_correct']:.3f} swapped {row['loss_swapped']:.3f} "
                    f"attention {np.mean(row['attention_by_layer']):.3f} norm {row['frame_norm']:.1f} "
                    f"within {row['within_cosine']:.3f} between {row['between_cosine']:.3f} vocab {row['best_vocab_cosine']:.3f}")
        (out_dir / "dense_analysis.json").write_text(json.dumps({"no_audio_loss": no_audio, "rows": rows}, indent=2))
    plot(rows, no_audio, out_dir)
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
