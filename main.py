from omegaconf import OmegaConf
import logging
from utils import TrainConfig
import sys
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration



logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def fooberino(cfg: TrainConfig) -> None:
    """A fooberino function"""
    ds = load_dataset(cfg.dataset_name, split=cfg.subset_name, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if cfg.max_samples:
        ds = ds.take(cfg.max_samples)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    sample = next(iter(ds))["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features, language="en", task="transcribe")

    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    print(transcription)



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
