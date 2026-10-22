import logging
import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from bake import bake_speech_embeddings, load_baked_embeddings
from utils import TrainConfig

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

    generator = torch.Generator().manual_seed(cfg.seed)
    permutation: list[int] = []

    def next_batch_indices() -> list[int]:
        """Draw batch_size indices, reshuffling once an epoch is exhausted."""
        nonlocal permutation
        if len(permutation) < cfg.batch_size:
            permutation = torch.randperm(example_count, generator=generator).tolist()
        indices, permutation = permutation[: cfg.batch_size], permutation[cfg.batch_size :]
        return indices

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
