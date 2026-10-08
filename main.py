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

def fooberino(cfg: TrainConfig) -> None:
    """A fooberino function"""
    ds = load_dataset(cfg.dataset_name, split=cfg.subset_name, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if cfg.max_samples:
        ds = ds.take(cfg.max_samples)

    # Speech embedding model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    encoder = model.model.encoder

    sample = next(iter(ds))["audio"]
    feats = processor.feature_extractor(sample["array"], sampling_rate=16000, return_tensors="pt").input_features
    speech_embedding = encoder(feats)
    speech_embedding.last_hidden_state.shape
    logger.info(torch.Size([1, 1500, 768])) # torch.Size([1, 1500, 768])

    # LM
    lm_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm = AutoModelForCausalLM.from_pretrained(
        lm_name,
        torch_dtype="auto",
    ).to(device)

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(lm.device)

    # conduct text completion
    generated_ids = lm.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


    breakpoint()
    pass

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
