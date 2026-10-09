from omegaconf import OmegaConf
import logging
from utils import TrainConfig
import sys
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import torch



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
k = 5

def fooberino(cfg: TrainConfig) -> None:
    """A fooberino function"""
    torch.manual_seed(cfg.seed)
    ds = load_dataset(cfg.dataset_name, split=cfg.subset_name, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if cfg.max_samples:
        ds = ds.take(cfg.max_samples)

    # Speech embedding model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = model.model.encoder.to(device).eval()

    # LM
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
        torch.nn.Linear(enc_dim * k, proj_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_hidden_dim, lm_dim),
    ).to(device)

    # Calculate speech embeding
    example = next(iter(ds))

    sample = example["audio"]
    transcript = example["text"]
    feats = processor.feature_extractor(
        sample["array"], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        h = encoder(feats).last_hidden_state # [1, 1500, 768]
    logger.info(f"encoder out: {tuple(h.shape)}")


    # Downsample just like in section 2.3, eq. 2 and 3
    B, T, D = h.shape
    N = T // k
    z = h[:, : N * k].reshape(B, N, k * D) # [1, 300, 3840]
    logger.info(f"downsampled: {tuple(z.shape)}")

    # Freeze the LM, only train the projector
    lm.requires_grad_(False)
    embed = lm.get_input_embeddings()

    def embed_text(s: str) -> torch.Tensor:
        ids = tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        return embed(ids)

    # Template(E_s, E_p), eq. 7, plus the transcript as the target
    prefix = embed_text("USER:")
    suffix = embed_text(" Transcribe speech to text. ASSISTANT:")
    target = tokenizer(transcript + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    opt = torch.optim.AdamW(projector.parameters(), lr=cfg.lr)
    for step in range(cfg.steps):
        # Run the pre-computed speech embedding (z) through the projector
        e_s = projector(z).to(lm.dtype)  # [1, 300, 1024]
        inputs_embeds = torch.cat([prefix, e_s, suffix, embed(target)], dim=1)
        # loss only on the transcript tokens
        labels = torch.full(inputs_embeds.shape[:2], -100, device=device)
        labels[:, -target.shape[1]:] = target
        loss = lm(inputs_embeds=inputs_embeds, labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.info(f"step {step} loss {loss.item():.4f}")

    # Generate "transcript" with the LM and the projected speech embeddings
    with torch.no_grad():
        inputs_embeds = torch.cat([prefix, projector(z).to(lm.dtype), suffix], dim=1)
        out_ids = lm.generate(inputs_embeds=inputs_embeds, max_new_tokens=64, do_sample=False)
    print("output:                  ", tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0])
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

    fooberino(cfg) # type: ignore


if __name__ == "__main__":
    main()
