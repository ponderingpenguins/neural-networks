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

def train_projector_and_transcribe(cfg: TrainConfig) -> None:
    """Train the speech projector on baked Whisper embeddings and transcribe a few examples."""
    torch.manual_seed(cfg.seed)
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if cfg.patience > 0 and (cfg.val_samples < 1 or cfg.eval_every < 1):
        raise ValueError("patience needs val_samples > 0 and eval_every > 0 to measure the validation loss.")

    # The train embeddings stay memory-mapped on disk: the full dataset does not fit in RAM.
    baked_embeddings, example_ids, transcripts, _ = load_baked_embeddings(cfg, lazy=True)  # fp16 [N, 300, 3840]
    example_count, grouped_frame_count, grouped_dim = baked_embeddings.shape
    if example_count < cfg.batch_size:
        raise ValueError(f"batch_size {cfg.batch_size} exceeds the {example_count} baked examples.")
    logger.info(f"loaded baked embeddings: {tuple(baked_embeddings.shape)}")
    eval_sets = {}
    train_eval_count = min(cfg.train_eval_samples, example_count)
    if train_eval_count > 0:
        eval_sets["train"] = (baked_embeddings[:train_eval_count], transcripts[:train_eval_count])
    for subset, sample_count in (("val", cfg.val_samples), ("test", cfg.test_samples)):
        if sample_count > 0:
            eval_embeddings, _, eval_texts, _ = load_baked_embeddings(cfg, subset)
            eval_sets[subset] = (eval_embeddings, eval_texts)
            logger.info(f"loaded {subset} embeddings: {tuple(eval_embeddings.shape)}")

    # Language Model (frozen) and projector
    tokenizer, lm = load_lm(device)
    if cfg.projector_type == "mlp":
        projector = build_projector(grouped_dim, lm.config.hidden_size, device)
    elif cfg.projector_type == "linear":
        projector = build_linear_projector(grouped_dim, lm.config.hidden_size, device)
    else:
        raise ValueError(f"unknown projector_type {cfg.projector_type!r}")
    token_embedding_layer = lm.get_input_embeddings()

    # Template(E_s, E_p), eq. 7, plus each transcript as its target.
    prompt_prefix_embeddings, prompt_suffix_embeddings = build_prompt_embeddings(tokenizer, lm, device)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token for transcript targets.")

    def tokenize_transcripts(texts: list[str]) -> list[torch.Tensor]:
        """Token ids of each transcript followed by EOS, kept on the CPU and moved per batch."""
        return [
            torch.tensor(
                tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id],
                dtype=torch.long,
            )
            for text in texts
        ]

    transcript_token_ids = tokenize_transcripts(transcripts)
    val_token_ids = tokenize_transcripts(eval_sets["val"][1]) if "val" in eval_sets else []
    transcript_start = (
        prompt_prefix_embeddings.shape[1]
        + grouped_frame_count
        + prompt_suffix_embeddings.shape[1]
    )

    def batch_loss(
        batch_embeddings: torch.Tensor, batch_token_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced cross-entropy of the transcripts given the projected speech (fp32 [batch, 300, 3840]).

        Also returns the projected speech embeddings [batch, 300, lm_dim], for the norm penalty and logging.
        """
        projected_speech_embeddings = projector(batch_embeddings).to(lm.dtype)  # [batch, 300, lm_dim]
        training_sequences = [
            torch.cat(
                [
                    prompt_prefix_embeddings[0],
                    projected_speech_embeddings[position],
                    prompt_suffix_embeddings[0],
                    token_embedding_layer(token_ids.to(device)),
                ],
                dim=0,
            )
            for position, token_ids in enumerate(batch_token_ids)
        ]
        # Right-pad complete sequences with zero embeddings to the longest sequence.
        inputs_embeds = torch.nn.utils.rnn.pad_sequence(
            training_sequences, batch_first=True, padding_value=0.0
        )  # [batch_size, max_sequence_length, lm_dim]
        # Attend to real sequence positions, including Whisper's encoded silence.
        attention_mask = torch.zeros(
            inputs_embeds.shape[:2], dtype=torch.long, device=device
        )
        for position, sequence in enumerate(training_sequences):
            attention_mask[position, :sequence.shape[0]] = 1
        labels = torch.full(inputs_embeds.shape[:2], -100, dtype=torch.long, device=device)
        for position, token_ids in enumerate(batch_token_ids):
            labels[position, transcript_start : transcript_start + len(token_ids)] = token_ids.to(device)
        loss = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        return loss, projected_speech_embeddings

    def evaluate_wer(subset: str) -> float:
        """Greedy-decode every example of a subset and return its corpus-level WER."""
        eval_embeddings, eval_texts = eval_sets[subset]
        projector.eval()
        hypotheses: list[str] = []
        # Every example has the same length (30 s padded audio), so batches need no padding.
        for start in range(0, len(eval_texts), cfg.eval_batch_size):
            batch = eval_embeddings[start : start + cfg.eval_batch_size]
            with torch.no_grad():
                speech = projector(batch.to(device=device, dtype=torch.float32))
            hypotheses += transcribe(
                tokenizer, lm, prompt_prefix_embeddings, prompt_suffix_embeddings, speech,
                cfg.eval_max_new_tokens,
            )
        projector.train()
        return word_error_rate(eval_texts, hypotheses)

    def evaluate_val_loss() -> float:
        """Mean teacher-forced loss over the validation set; one forward pass per batch, no generation."""
        eval_embeddings, _ = eval_sets["val"]
        projector.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            # Batches of the training size: the full-vocabulary logits are too big for larger ones on a small GPU.
            for start in range(0, len(val_token_ids), cfg.batch_size):
                batch = eval_embeddings[start : start + cfg.batch_size]
                batch_ids = val_token_ids[start : start + cfg.batch_size]
                loss, _ = batch_loss(batch.to(device=device, dtype=torch.float32), batch_ids)
                total += loss.item() * len(batch_ids)
                count += len(batch_ids)
        projector.train()
        return total / count

    generator = torch.Generator().manual_seed(cfg.seed)
    permutation: list[int] = []

    def next_batch_indices() -> list[int]:
        """Draw batch_size indices, reshuffling once an epoch is exhausted."""
        nonlocal permutation
        if len(permutation) < cfg.batch_size:
            permutation = torch.randperm(example_count, generator=generator).tolist()
        indices, permutation = permutation[: cfg.batch_size], permutation[cfg.batch_size :]
        return indices

    checkpoint_dir = Path(cfg.output_path) / "checkpoints"
    history: list[dict] = []
    best_val_loss = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    evals_without_improvement = 0
    stopped_early = False
    completed_steps = 0
    opt = torch.optim.AdamW(projector.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / cfg.warmup_steps) if cfg.warmup_steps > 0 else 1.0
    )
    # Per-step training log (train_log.csv) for plotting: loss, the norm penalty, gradient norms (before clipping),
    # the mean norm of the projected speech frames and the learning rate.
    text_norm = token_embedding_layer.weight.detach()[: len(tokenizer)].norm(dim=-1).mean().item()
    first_layer, last_layer = projector[0], projector[-1]
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)
    train_log = open(Path(cfg.output_path) / "train_log.csv", "w", encoding="utf-8")
    train_log.write("step,loss,penalty,grad_norm,grad_norm_layer1,grad_norm_layer2,proj_norm,lr\n")
    for step in range(cfg.steps):
        indices = next_batch_indices()
        loss, projected = batch_loss(
            baked_embeddings[indices].to(device=device, dtype=torch.float32),
            [transcript_token_ids[index] for index in indices],
        )
        frame_norms = projected.norm(dim=-1)
        penalty = ((frame_norms - text_norm) ** 2).mean()
        opt.zero_grad()
        (loss + cfg.norm_penalty * penalty if cfg.norm_penalty > 0 else loss).backward()
        layer_grad_norms = (first_layer.weight.grad.norm().item(), last_layer.weight.grad.norm().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            projector.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        ).item()
        opt.step()
        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()
        completed_steps = step + 1
        train_log.write(
            f"{completed_steps},{loss.item():.6f},{penalty.item():.4f},{grad_norm:.5f},"
            f"{layer_grad_norms[0]:.5f},{layer_grad_norms[1]:.5f},"
            f"{frame_norms.mean().item():.4f},{learning_rate:.3e}\n"
        )
        if completed_steps % 50 == 0:
            train_log.flush()
        logger.info(
            f"step {step} loss {loss.item():.4f} grad_norm {grad_norm:.3f} proj_norm {frame_norms.mean().item():.2f}"
        )
        if cfg.save_every and completed_steps % cfg.save_every == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(projector.state_dict(), checkpoint_dir / f"projector_step{completed_steps}.pt")

        row: dict = {"step": completed_steps, "loss": loss.item()}
        evaluated = False
        if "val" in eval_sets and cfg.eval_every and (completed_steps % cfg.eval_every == 0 or completed_steps == cfg.steps):
            val_loss = evaluate_val_loss()
            row["val_loss"] = val_loss
            evaluated = True
            if val_loss < best_val_loss - cfg.min_delta:
                best_val_loss, best_step, evals_without_improvement = val_loss, completed_steps, 0
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in projector.state_dict().items()}
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_dir / "projector_best.pt")
            elif completed_steps >= cfg.min_steps:
                evals_without_improvement += 1
            logger.info(
                f"step {step} val_loss {val_loss:.4f} best {best_val_loss:.4f} (step {best_step}) "
                f"evals without improvement {evals_without_improvement}"
            )
        if cfg.wer_every and completed_steps % cfg.wer_every == 0 and completed_steps != cfg.steps:
            for subset in ("train", "val"):
                if subset in eval_sets:
                    row[f"{subset}_wer"] = evaluate_wer(subset)
            evaluated = True
            logger.info(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k.endswith("_wer")))
        if evaluated:
            history.append(row)
        if cfg.patience > 0 and evals_without_improvement >= cfg.patience:
            stopped_early = True
            logger.info(f"early stop at step {completed_steps}: no val_loss improvement for {cfg.patience} evals")
            break

    # Report the best validation-loss projector, not whatever the last step happened to be.
    final_step = completed_steps
    if best_state is not None:
        projector.load_state_dict(best_state)
        final_step = best_step
        logger.info(f"restored the best projector from step {best_step} (val_loss {best_val_loss:.4f})")
    final_row: dict = {"step": final_step, "final": True}
    if best_state is not None:
        final_row["val_loss"] = best_val_loss
    train_log.close()
    for subset in ("train", "val") if cfg.final_wer else ():
        if subset in eval_sets:
            final_row[f"{subset}_wer"] = evaluate_wer(subset)
    history.append(final_row)
    logger.info("final " + " ".join(f"{k} {v:.4f}" for k, v in final_row.items() if k.endswith("_wer")))

    test_wer = None
    if cfg.final_wer and "test" in eval_sets:
        test_wer = evaluate_wer("test")
        logger.info(f"final test WER {test_wer:.4f} ({len(eval_sets['test'][1])} examples)")
    output_dir = Path(cfg.output_path)
    (output_dir / "wer_history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "run_summary.json").write_text(json.dumps({
        "max_steps": cfg.steps,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "best_step": best_step if best_state is not None else None,
        "best_val_loss": best_val_loss if best_state is not None else None,
        "test_wer": test_wer,
    }, indent=2))
    plot_wer(history, output_dir / "wer.png")
    torch.save(projector.state_dict(), output_dir / "projector.pt")
    logger.info(f"saved WER history, plot and projector to {output_dir}")

    # Generate "transcripts" with the LM and the projected speech embeddings
    projector.eval()
    for index in range(min(3, example_count)):
        with torch.no_grad():
            inputs_embeds = torch.cat(
                [prompt_prefix_embeddings,
                 projector(baked_embeddings[index : index + 1].to(device=device, dtype=torch.float32)).to(lm.dtype),
                 prompt_suffix_embeddings], dim=1
            )
            generated_token_ids = lm.generate(inputs_embeds=inputs_embeds, max_new_tokens=64, do_sample=False)
        print(f"[{example_ids[index]}]")
        print("output:                  ", tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0])
        print("ground truth transcript: ", transcripts[index])

def main() -> None:
    """main function"""
    cfg = OmegaConf.structured(TrainConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainConfig(**cfg) # type: ignore
    except TypeError as e:  # pylint: disable=broad-exception-raised
        logger.error(f"Error: {e}\n\nUsage: python main.py dataset_name='mydataset' output_path='./results'")
        sys.exit(1)

    if cfg.stage == "bake":
        bake_speech_embeddings(cfg, device) # type: ignore
        # The streaming dataset leaves non-daemon threads alive that keep the process from
        # exiting, which stalls the next stage in pipeline_bake_train_analyze.sh until Ctrl+C. Shards are already saved.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif cfg.stage == "train":
        train_projector_and_transcribe(cfg) # type: ignore
    else:
        logger.error(f"Unknown stage '{cfg.stage}'; expected 'bake' or 'train'.") # type: ignore
        sys.exit(1)


if __name__ == "__main__":
    main()
