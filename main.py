import logging
import sys
from itertools import islice

import torch
from datasets import Audio, load_dataset
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from utils import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
frame_group_size = 5

def train_projector_and_transcribe(cfg: TrainConfig) -> None:
    """Train the speech projector on one example and generate its transcript."""
    torch.manual_seed(cfg.seed)
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if cfg.max_samples and cfg.max_samples < cfg.batch_size:
        raise ValueError("max_samples must be at least batch_size.")
    speech_dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    speech_dataset = speech_dataset.cast_column("audio", Audio(sampling_rate=16000))
    if cfg.max_samples:
        speech_dataset = speech_dataset.take(cfg.max_samples)

    # Select once so every training step can reuse the same audio/transcript pairs.
    fixed_batch = list(islice(speech_dataset, cfg.batch_size))
    if len(fixed_batch) < cfg.batch_size:
        raise ValueError(
            f"Requested {cfg.batch_size} examples, but only found {len(fixed_batch)}."
        )
    example_ids = [example["id"] for example in fixed_batch]
    if len(set(example_ids)) != cfg.batch_size:
        raise ValueError("The fixed batch must contain distinct dataset example IDs.")
    logger.info("Selected fixed batch of %d examples: %s", len(fixed_batch), example_ids)

    # Speech embedding model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = whisper_model.model.encoder.to(device).eval()

    # Language Model
    lm_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm = AutoModelForCausalLM.from_pretrained(
        lm_name,
        dtype=torch.float32,
    ).to(device)
    # Projector
    enc_dim = encoder.config.d_model
    lm_dim = lm.config.hidden_size
    proj_hidden_dim = 2048
    projector = torch.nn.Sequential(
        torch.nn.Linear(enc_dim * frame_group_size, proj_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_hidden_dim, lm_dim),
    ).to(device)

    # Calculate speech embeding
    # Use the first selected pair until audio and transcript batching is implemented.
    example = fixed_batch[0]

    audio_sample = example["audio"]
    transcript = example["text"]
    audio_features = processor.feature_extractor(
        audio_sample["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        speech_embeddings = encoder(audio_features).last_hidden_state # [1, 1500, 768]
    logger.info(f"encoder out: {tuple(speech_embeddings.shape)}")


    # Downsample just like in section 2.3, eq. 2 and 3
    batch_size, frame_count, encoder_dim = speech_embeddings.shape
    grouped_frame_count = frame_count // frame_group_size
    grouped_speech_embeddings = speech_embeddings[
        :, : grouped_frame_count * frame_group_size
    ].reshape(batch_size, grouped_frame_count, frame_group_size * encoder_dim) # [1, 300, 3840]
    logger.info(f"downsampled: {tuple(grouped_speech_embeddings.shape)}")

    # Freeze the LM, only train the projector
    lm.requires_grad_(False)
    token_embedding_layer = lm.get_input_embeddings()

    def embed_text(text: str) -> torch.Tensor:
        token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        return token_embedding_layer(token_ids)

    # Template(E_s, E_p), eq. 7, plus the transcript as the target
    prompt_prefix_embeddings = embed_text("USER:")
    prompt_suffix_embeddings = embed_text(" Transcribe speech to text. ASSISTANT:")
    transcript_token_ids = tokenizer(transcript + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    opt = torch.optim.AdamW(projector.parameters(), lr=cfg.lr)
    for step in range(cfg.steps):
        # Run the pre-computed speech embedding (grouped_speech_embeddings) through the projector
        projected_speech_embeddings = projector(grouped_speech_embeddings).to(lm.dtype)  # [1, 300, 1024]
        inputs_embeds = torch.cat([prompt_prefix_embeddings, projected_speech_embeddings, prompt_suffix_embeddings, token_embedding_layer(transcript_token_ids)], dim=1)
        # loss only on the transcript tokens
        labels = torch.full(inputs_embeds.shape[:2], -100, device=device)
        labels[:, -transcript_token_ids.shape[1]:] = transcript_token_ids
        loss = lm(inputs_embeds=inputs_embeds, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.info(f"step {step} loss {loss.item():.4f}")

    # Generate "transcript" with the LM and the projected speech embeddings
    with torch.no_grad():
        inputs_embeds = torch.cat([prompt_prefix_embeddings, projector(grouped_speech_embeddings).to(lm.dtype), prompt_suffix_embeddings], dim=1)
        generated_token_ids = lm.generate(inputs_embeds=inputs_embeds, max_new_tokens=64, do_sample=False)
    print("output:                  ", tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0])
    print("ground truth transcript: ", transcript)

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

    train_projector_and_transcribe(cfg) # type: ignore


if __name__ == "__main__":
    main()
