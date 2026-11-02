import re
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from transformers import AutoModelForCausalLM, AutoTokenizer

LM_NAME = "Qwen/Qwen3-0.6B"
PROJ_HIDDEN_DIM = 2048


@dataclass
class TrainConfig:
    """Configuration for the project."""

    dataset_split: str | None = None
    dataset_name: str = MISSING
    max_samples: int = MISSING
    batch_size: int = 4
    stage: str = "train"  # "bake" caches Whisper embeddings to disk, "train" trains on them
    bake_batch_size: int = 8

    # Evaluation; set *_samples to 0 to skip that split. Use 1 for a quick smoke test.
    val_split: str = "validation.clean"
    val_samples: int = 100
    test_split: str = "test.clean"
    test_samples: int = 100
    train_eval_samples: int = 100  # first N training examples used for the train WER curve
    eval_every: int = 100  # validation loss every N training steps (cheap, teacher-forced); 0 = only after training
    wer_every: int = 100  # train/val WER (greedy decoding, slow) every N training steps; 0 = only after training
    eval_batch_size: int = 8
    patience: int = 0  # stop after this many validation-loss evals without improvement; 0 = train for all steps
    min_steps: int = 0  # never stop early before this step (the loss plateaus before the projector aligns)
    min_delta: float = 0.0  # smallest validation-loss drop that counts as an improvement
    eval_max_new_tokens: int = 200

    output_path: Path = MISSING
    seed:int = 67
    overwrite: bool = False

    grad_clip: float = 0.0  # clip the projector's gradient norm to this value; 0 = no clipping
    norm_penalty: float = 0.0  # weight on mean (projected frame norm - mean text embedding norm)^2; 0 = off
    projector_type: str = "mlp"  # "mlp": two-layer MLP; "linear": single linear layer (baseline)
    final_wer: bool = True  # False skips the (slow) greedy-decoding WER after training; val/test loss still run
    steps: int = 300  # maximum number of training steps; patience may stop earlier
    lr: float = 1e-3  # peak learning rate
    warmup_steps: int = 1000  # linear warmup from 0 to lr; the rate then stays constant (as in the paper)
    save_every: int = 100  # save a projector checkpoint every N training steps; 0 = only the final one


def normalize_text(text: str) -> str:
    """Uppercase and strip punctuation (keeping apostrophes) to match LibriSpeech transcripts."""
    return " ".join(re.sub(r"[^A-Z' ]", " ", text.upper()).split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word))
            )
        previous = current
    return previous[-1]


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER: total word edits divided by total reference words."""
    errors = 0
    words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = normalize_text(reference).split()
        errors += _edit_distance(ref_words, normalize_text(hypothesis).split())
        words += len(ref_words)
    return errors / max(words, 1)


def plot_wer(history: list[dict], path: Path) -> None:
    """Plot train and validation WER against training step and save it to path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("train_wer", "train"), ("val_wer", "validation")):
        points = [(row["step"], row[key]) for row in history if key in row]
        if points:
            ax.plot(*zip(*points), marker="o", label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("WER")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_lm(device: torch.device):
    """Load the frozen language model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(LM_NAME)
    lm = AutoModelForCausalLM.from_pretrained(LM_NAME, dtype=torch.float32).to(device)
    lm.requires_grad_(False)
    return tokenizer, lm


def build_projector(in_dim: int, lm_dim: int, device: torch.device) -> torch.nn.Module:
    """The two-layer MLP that maps grouped Whisper frames into the LM embedding space."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, PROJ_HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Linear(PROJ_HIDDEN_DIM, lm_dim),
    ).to(device)


def build_linear_projector(in_dim: int, lm_dim: int, device: torch.device) -> torch.nn.Module:
    """Elementary baseline: a single linear map from grouped Whisper frames to the LM embedding space."""
    return torch.nn.Sequential(torch.nn.Linear(in_dim, lm_dim)).to(device)


def embed_text(tokenizer, lm, text: str, device: torch.device) -> torch.Tensor:
    """Embed text without special tokens: [1, n_tokens, lm_dim]."""
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return lm.get_input_embeddings()(token_ids)


def build_prompt_embeddings(tokenizer, lm, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Template(E_s, E_p), eq. 7: embeddings placed before and after the speech embeddings."""
    return (
        embed_text(tokenizer, lm, "USER:", device),
        embed_text(tokenizer, lm, " Transcribe speech to text. ASSISTANT:", device),
    )


@torch.no_grad()
def transcribe(
    tokenizer,
    lm,
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    speech: torch.Tensor,
    max_new_tokens: int,
) -> list[str]:
    """Greedy-decode a batch of speech embeddings [batch, frames, lm_dim] (already in the LM space)."""
    device = speech.device
    inputs_embeds = torch.cat(
        [prefix.expand(len(speech), -1, -1), speech.to(lm.dtype), suffix.expand(len(speech), -1, -1)],
        dim=1,
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    generated_token_ids = lm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
    )
    return tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
