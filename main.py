import json
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from bake import bake_speech_embeddings, load_baked_embeddings
from utils import TrainConfig, plot_wer, word_error_rate

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

    baked_embeddings, example_ids, transcripts = load_baked_embeddings(cfg)  # fp16 [N, 300, 3840]
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
            eval_embeddings, _, eval_texts = load_baked_embeddings(cfg, subset)
            eval_sets[subset] = (eval_embeddings, eval_texts)
            logger.info(f"loaded {subset} embeddings: {tuple(eval_embeddings.shape)}")

    # Language Model
    lm_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm = AutoModelForCausalLM.from_pretrained(
        lm_name,
        dtype=torch.float32,
    ).to(device)
    # Projector
    lm_dim = lm.config.hidden_size
    proj_hidden_dim = 2048
    projector = torch.nn.Sequential(
        torch.nn.Linear(grouped_dim, proj_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_hidden_dim, lm_dim),
    ).to(device)

    # Freeze the LM, only train the projector
    lm.requires_grad_(False)
    token_embedding_layer = lm.get_input_embeddings()

    def embed_text(text: str) -> torch.Tensor:
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        return token_embedding_layer(token_ids)

    # Template(E_s, E_p), eq. 7, plus each transcript as its target.
    prompt_prefix_embeddings = embed_text("USER:")  # [1, prefix_len, lm_dim]
    prompt_suffix_embeddings = embed_text(" Transcribe speech to text. ASSISTANT:")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token for transcript targets.")
    transcript_token_ids = [
        torch.tensor(
            tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id],
            dtype=torch.long,
            device=device,
        )
        for text in transcripts
    ]
    transcript_embeddings = [token_embedding_layer(ids) for ids in transcript_token_ids]
    transcript_start = (
        prompt_prefix_embeddings.shape[1]
        + grouped_frame_count
        + prompt_suffix_embeddings.shape[1]
    )

    def evaluate_wer(subset: str) -> float:
        """Greedy-decode every example of a subset and return its corpus-level WER."""
        eval_embeddings, eval_texts = eval_sets[subset]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        projector.eval()
        hypotheses: list[str] = []
        # Every example has the same length (30 s padded audio), so batches need no padding.
        for start in range(0, len(eval_texts), cfg.eval_batch_size):
            batch = eval_embeddings[start : start + cfg.eval_batch_size]
            with torch.no_grad():
                speech = projector(batch.to(device=device, dtype=torch.float32)).to(lm.dtype)
                inputs_embeds = torch.cat(
                    [prompt_prefix_embeddings.expand(len(batch), -1, -1),
                     speech,
                     prompt_suffix_embeddings.expand(len(batch), -1, -1)], dim=1
                )
                generated_token_ids = lm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device),
                    max_new_tokens=cfg.eval_max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                )
            hypotheses += tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        projector.train()
        return word_error_rate(eval_texts, hypotheses)

    generator = torch.Generator().manual_seed(cfg.seed)
    permutation: list[int] = []

    def next_batch_indices() -> list[int]:
        """Draw batch_size indices, reshuffling once an epoch is exhausted."""
        nonlocal permutation
        if len(permutation) < cfg.batch_size:
            permutation = torch.randperm(example_count, generator=generator).tolist()
        indices, permutation = permutation[: cfg.batch_size], permutation[cfg.batch_size :]
        return indices

    history: list[dict] = []
    opt = torch.optim.AdamW(projector.parameters(), lr=cfg.lr)
    for step in range(cfg.steps):
        indices = next_batch_indices()
        batch_embeddings = baked_embeddings[indices].to(device=device, dtype=torch.float32)
        projected_speech_embeddings = projector(batch_embeddings).to(lm.dtype)  # [batch, 300, lm_dim]
        training_sequences = [
            torch.cat(
                [
                    prompt_prefix_embeddings[0],
                    projected_speech_embeddings[position],
                    prompt_suffix_embeddings[0],
                    transcript_embeddings[index],
                ],
                dim=0,
            )
            for position, index in enumerate(indices)
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
        for position, index in enumerate(indices):
            token_ids = transcript_token_ids[index]
            labels[position, transcript_start : transcript_start + len(token_ids)] = token_ids
        loss = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.info(f"step {step} loss {loss.item():.4f}")
        if (cfg.eval_every and (step + 1) % cfg.eval_every == 0) or step + 1 == cfg.steps:
            row: dict = {"step": step + 1, "loss": loss.item()}
            for subset in ("train", "val"):
                if subset in eval_sets:
                    row[f"{subset}_wer"] = evaluate_wer(subset)
            history.append(row)
            logger.info(f"step {step} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if k.endswith("_wer")))

    if "test" in eval_sets:
        logger.info(f"final test WER {evaluate_wer('test'):.4f} ({len(eval_sets['test'][1])} examples)")
    output_dir = Path(cfg.output_path)
    (output_dir / "wer_history.json").write_text(json.dumps(history, indent=2))
    plot_wer(history, output_dir / "wer.png")
    logger.info(f"saved WER history and plot to {output_dir}")

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
        # exiting, which stalls the next stage in train.sh until Ctrl+C. Shards are already saved.
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
