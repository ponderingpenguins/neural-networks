import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from bake import bake_speech_embeddings, load_baked_embeddings
from utils import (
    TrainConfig,
    build_linear_projector,
    build_projector,
    build_prompt_embeddings,
    load_lm,
    plot_wer,
    transcribe,
    word_error_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub", "fsspec", "datasets"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


class SpeechLM:
    """Frozen LM that reads projected speech between a fixed prompt prefix and suffix."""

    def __init__(self, projector_type: str, in_dim: int, n_frames: int):
        self.tokenizer, self.lm = load_lm(device)
        lm_dim = self.lm.config.hidden_size
        if projector_type == "mlp":
            self.projector = build_projector(in_dim, lm_dim, device)
        elif projector_type == "linear":
            self.projector = build_linear_projector(in_dim, lm_dim, device)
        else:
            raise ValueError(f"unknown projector_type {projector_type!r}")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token for transcript targets.")
        self.prefix, self.suffix = build_prompt_embeddings(self.tokenizer, self.lm, device)
        self.transcript_start = self.prefix.shape[1] + n_frames + self.suffix.shape[1]

    def token_ids(self, texts: list[str]) -> list[torch.Tensor]:
        """Token ids of each transcript followed by EOS; kept on the CPU and moved per batch."""
        eos = self.tokenizer.eos_token_id
        return [
            torch.tensor(self.tokenizer(text, add_special_tokens=False).input_ids + [eos], dtype=torch.long)
            for text in texts
        ]

    def loss(self, speech: torch.Tensor, ids: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced loss on the transcripts, and the projected frames [batch, 300, lm_dim]."""
        projected = self.projector(speech).to(self.lm.dtype)
        seqs = [
            torch.cat(
                [self.prefix[0], projected[i], self.suffix[0], self.lm.get_input_embeddings()(tokens.to(device))],
                dim=0,
            )
            for i, tokens in enumerate(ids)
        ]
        inputs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
        mask = torch.zeros(inputs.shape[:2], dtype=torch.long, device=device)
        labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
        for i, (seq, tokens) in enumerate(zip(seqs, ids)):
            mask[i, : seq.shape[0]] = 1
            labels[i, self.transcript_start : self.transcript_start + len(tokens)] = tokens.to(device)
        loss = self.lm(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss
        return loss, projected

    def val_loss(self, embeddings, ids: list[torch.Tensor], batch_size: int) -> float:
        """Mean loss over a set, in training-sized batches (the vocabulary logits are big)."""
        self.projector.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start : start + batch_size]
                speech = embeddings[start : start + batch_size].to(device=device, dtype=torch.float32)
                total += self.loss(speech, batch_ids)[0].item() * len(batch_ids)
        self.projector.train()
        return total / len(ids)

    def wer(self, embeddings, texts: list[str], batch_size: int, max_new_tokens: int) -> float:
        """Greedy-decode a set and return its corpus WER."""
        self.projector.eval()
        hypotheses: list[str] = []
        for start in range(0, len(texts), batch_size):  # all utterances are padded to 30 s, so no length padding
            speech = embeddings[start : start + batch_size].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                speech = self.projector(speech)
            hypotheses += transcribe(self.tokenizer, self.lm, self.prefix, self.suffix, speech, max_new_tokens)
        self.projector.train()
        return word_error_rate(texts, hypotheses)


class BestModel:
    """Remembers the projector with the lowest validation loss and how long ago it was seen."""

    def __init__(self, min_delta: float, min_steps: int):
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.loss = math.inf
        self.step = 0
        self.state: dict[str, torch.Tensor] | None = None
        self.evals_since_best = 0

    def update(self, loss: float, step: int, projector: torch.nn.Module) -> bool:
        """Record a validation loss; return True if it is a new best."""
        if loss < self.loss - self.min_delta:
            self.loss, self.step, self.evals_since_best = loss, step, 0
            self.state = {name: t.detach().cpu().clone() for name, t in projector.state_dict().items()}
            return True
        if step >= self.min_steps:  # the loss plateaus before the projector aligns, so wait
            self.evals_since_best += 1
        return False


class TrainLog:
    """Per-step CSV with the loss, penalty, gradient norms, frame norm and learning rate."""

    HEADER = "step,loss,penalty,grad_norm,grad_norm_layer1,grad_norm_layer2,proj_norm,lr\n"

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.file.write(self.HEADER)

    def write(self, step: int, s: dict) -> None:
        self.file.write(
            f"{step},{s['loss']:.6f},{s['penalty']:.4f},{s['grad_norm']:.5f},"
            f"{s['grad_layer1']:.5f},{s['grad_layer2']:.5f},{s['proj_norm']:.4f},{s['lr']:.3e}\n"
        )
        if step % 50 == 0:
            self.file.flush()

    def close(self) -> None:
        self.file.close()


def batch_indices(count: int, size: int, seed: int):
    """Endless batches of indices; reshuffles when fewer than `size` are left."""
    generator = torch.Generator().manual_seed(seed)
    left: list[int] = []
    while True:
        if len(left) < size:
            left = torch.randperm(count, generator=generator).tolist()
        yield left[:size]
        left = left[size:]


def load_eval_sets(cfg: TrainConfig, train_embeddings, train_texts) -> dict:
    """{"train": ..., "val": ..., "test": ...} of (embeddings, texts); a split with 0 samples is left out."""
    sets = {}
    n_train = min(cfg.train_eval_samples, train_embeddings.shape[0])
    if n_train > 0:
        sets["train"] = (train_embeddings[:n_train], train_texts[:n_train])
    for subset, n_samples in (("val", cfg.val_samples), ("test", cfg.test_samples)):
        if n_samples > 0:
            embeddings, _, texts, _ = load_baked_embeddings(cfg, subset)
            sets[subset] = (embeddings, texts)
            logger.info(f"loaded {subset} embeddings: {tuple(embeddings.shape)}")
    return sets


def train_step(model: SpeechLM, opt, scheduler, cfg: TrainConfig, speech, ids, text_norm: float) -> dict:
    """One optimiser step; returns the numbers that go into train_log.csv."""
    loss, projected = model.loss(speech, ids)
    frame_norms = projected.norm(dim=-1)
    penalty = ((frame_norms - text_norm) ** 2).mean()
    opt.zero_grad()
    (loss + cfg.norm_penalty * penalty if cfg.norm_penalty > 0 else loss).backward()
    first, last = model.projector[0], model.projector[-1]
    grad_layer1, grad_layer2 = first.weight.grad.norm().item(), last.weight.grad.norm().item()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.projector.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
    ).item()
    opt.step()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()
    return {
        "loss": loss.item(),
        "penalty": penalty.item(),
        "grad_norm": grad_norm,
        "grad_layer1": grad_layer1,
        "grad_layer2": grad_layer2,
        "proj_norm": frame_norms.mean().item(),
        "lr": lr,
    }


def evaluate(cfg, model, eval_sets, val_ids, best, row, step, ckpt_dir) -> bool:
    """Scheduled validation loss and WER evals. Adds results to `row`; True if anything ran."""
    ran = False
    last_step = step == cfg.steps
    if "val" in eval_sets and cfg.eval_every and (step % cfg.eval_every == 0 or last_step):
        row["val_loss"] = model.val_loss(eval_sets["val"][0], val_ids, cfg.batch_size)
        ran = True
        if best.update(row["val_loss"], step, model.projector):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best.state, ckpt_dir / "projector_best.pt")
        logger.info(
            f"step {step - 1} val_loss {row['val_loss']:.4f} best {best.loss:.4f} (step {best.step}) "
            f"evals without improvement {best.evals_since_best}"
        )
    if cfg.wer_every and step % cfg.wer_every == 0 and not last_step:  # the final WER runs after training
        for subset in ("train", "val"):
            if subset in eval_sets:
                row[f"{subset}_wer"] = model.wer(*eval_sets[subset], cfg.eval_batch_size, cfg.eval_max_new_tokens)
        ran = True
        logger.info(f"step {step - 1} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k.endswith("_wer")))
    return ran


def final_evaluation(cfg, model, eval_sets, best, completed_steps, history) -> float | None:
    """Restore the best projector, add the final row to `history`, return the test WER if it was measured."""
    final_step = completed_steps
    if best.state is not None:
        model.projector.load_state_dict(best.state)
        final_step = best.step
        logger.info(f"restored the best projector from step {best.step} (val_loss {best.loss:.4f})")
    final_row: dict = {"step": final_step, "final": True}
    if best.state is not None:
        final_row["val_loss"] = best.loss
    for subset in ("train", "val") if cfg.final_wer else ():
        if subset in eval_sets:
            final_row[f"{subset}_wer"] = model.wer(*eval_sets[subset], cfg.eval_batch_size, cfg.eval_max_new_tokens)
    history.append(final_row)
    logger.info("final " + " ".join(f"{k} {v:.4f}" for k, v in final_row.items() if k.endswith("_wer")))

    if cfg.final_wer and "test" in eval_sets:
        test_wer = model.wer(*eval_sets["test"], cfg.eval_batch_size, cfg.eval_max_new_tokens)
        logger.info(f"final test WER {test_wer:.4f} ({len(eval_sets['test'][1])} examples)")
        return test_wer
    return None


def save_results(cfg, model, best, history, completed_steps, stopped_early, test_wer) -> None:
    out = Path(cfg.output_path)
    (out / "wer_history.json").write_text(json.dumps(history, indent=2))
    (out / "run_summary.json").write_text(json.dumps({
        "max_steps": cfg.steps,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "best_step": best.step if best.state is not None else None,
        "best_val_loss": best.loss if best.state is not None else None,
        "test_wer": test_wer,
    }, indent=2))
    plot_wer(history, out / "wer.png")
    torch.save(model.projector.state_dict(), out / "projector.pt")
    logger.info(f"saved WER history, plot and projector to {out}")


def print_examples(model: SpeechLM, embeddings, example_ids, texts, n: int = 3) -> None:
    """Print a few greedy transcripts next to the reference."""
    model.projector.eval()
    for i in range(min(n, len(texts))):
        with torch.no_grad():
            speech = model.projector(embeddings[i : i + 1].to(device=device, dtype=torch.float32))
            inputs = torch.cat([model.prefix, speech.to(model.lm.dtype), model.suffix], dim=1)
            out = model.lm.generate(inputs_embeds=inputs, max_new_tokens=64, do_sample=False)
        print(f"[{example_ids[i]}]")
        print("output:                  ", model.tokenizer.batch_decode(out, skip_special_tokens=True)[0])
        print("ground truth transcript: ", texts[i])


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if cfg.patience > 0 and (cfg.val_samples < 1 or cfg.eval_every < 1):
        raise ValueError("patience needs val_samples > 0 and eval_every > 0 to measure the validation loss.")

    # Train embeddings stay memory-mapped: the full set does not fit in RAM.
    embeddings, example_ids, transcripts, _ = load_baked_embeddings(cfg, lazy=True)  # fp16 [N, 300, 3840]
    count, n_frames, in_dim = embeddings.shape
    if count < cfg.batch_size:
        raise ValueError(f"batch_size {cfg.batch_size} exceeds the {count} baked examples.")
    logger.info(f"loaded baked embeddings: {tuple(embeddings.shape)}")
    eval_sets = load_eval_sets(cfg, embeddings, transcripts)

    model = SpeechLM(cfg.projector_type, in_dim, n_frames)
    train_ids = model.token_ids(transcripts)
    val_ids = model.token_ids(eval_sets["val"][1]) if "val" in eval_sets else []

    opt = torch.optim.AdamW(model.projector.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / cfg.warmup_steps) if cfg.warmup_steps > 0 else 1.0
    )
    text_norm = model.lm.get_input_embeddings().weight.detach()[: len(model.tokenizer)].norm(dim=-1).mean().item()
    ckpt_dir = Path(cfg.output_path) / "checkpoints"
    log = TrainLog(Path(cfg.output_path) / "train_log.csv")
    best = BestModel(cfg.min_delta, cfg.min_steps)
    batches = batch_indices(count, cfg.batch_size, cfg.seed)
    history: list[dict] = []
    stopped_early = False
    completed_steps = 0

    for step in range(cfg.steps):
        indices = next(batches)
        stats = train_step(
            model, opt, scheduler, cfg,
            embeddings[indices].to(device=device, dtype=torch.float32),
            [train_ids[i] for i in indices],
            text_norm,
        )
        completed_steps = step + 1
        log.write(completed_steps, stats)
        logger.info(
            f"step {step} loss {stats['loss']:.4f} grad_norm {stats['grad_norm']:.3f} "
            f"proj_norm {stats['proj_norm']:.2f}"
        )
        if cfg.save_every and completed_steps % cfg.save_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.projector.state_dict(), ckpt_dir / f"projector_step{completed_steps}.pt")

        row: dict = {"step": completed_steps, "loss": stats["loss"]}
        if evaluate(cfg, model, eval_sets, val_ids, best, row, completed_steps, ckpt_dir):
            history.append(row)
        if cfg.patience > 0 and best.evals_since_best >= cfg.patience:
            stopped_early = True
            logger.info(f"early stop at step {completed_steps}: no val_loss improvement for {cfg.patience} evals")
            break

    log.close()
    test_wer = final_evaluation(cfg, model, eval_sets, best, completed_steps, history)
    save_results(cfg, model, best, history, completed_steps, stopped_early, test_wer)
    print_examples(model, embeddings, example_ids, transcripts)


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.from_cli())
    try:
        cfg = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    except TypeError as e:
        logger.error(f"Error: {e}\n\nUsage: python main.py dataset_name='mydataset' output_path='./results'")
        sys.exit(1)

    if cfg.stage == "bake":
        bake_speech_embeddings(cfg, device)
        # The streaming dataset leaves non-daemon threads alive that keep the process from exiting, which
        # stalls the next stage in pipeline_bake_train_analyze.sh. The shards are already saved.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif cfg.stage == "train":
        train(cfg)
    else:
        logger.error(f"Unknown stage '{cfg.stage}'; expected 'bake' or 'train'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
