"""Reference: WER of Whisper-small used as a normal ASR system (its own decoder) on the first N test.clean utterances.

Usage: python whisper_wer.py n=500 out=results/test_eval/whisper_wer.json
Same utterances, same order and same text normalisation as eval_wer.py, so the numbers are comparable.
"""
import json
import sys
from itertools import islice
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from bake import decode_audio
from utils import edit_distance, normalize_text

args = dict(a.split("=", 1) for a in sys.argv[1:])
count, out_path = int(args["n"]), Path(args["out"])
device = torch.device("cuda")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device).eval()
dataset = load_dataset("openslr/librispeech_asr", split="test.clean", streaming=True)
dataset = dataset.cast_column("audio", Audio(decode=False)).take(count)
iterator = iter(dataset)

errors, words, hypotheses = [], [], []
while len(errors) < count:
    examples = list(islice(iterator, 8))
    if not examples:
        break
    audio = [decode_audio(example["audio"]) for example in examples]
    features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        generated = model.generate(features, language="en", task="transcribe", max_new_tokens=200, do_sample=False)
    for example, hypothesis in zip(examples, processor.batch_decode(generated, skip_special_tokens=True)):
        ref_words = normalize_text(example["text"]).split()
        errors.append(edit_distance(ref_words, normalize_text(hypothesis).split()))
        words.append(len(ref_words))
        hypotheses.append(hypothesis)
wer = sum(errors) / sum(words)
out_path.write_text(json.dumps({"wer": wer, "errors": errors, "words": words, "hypotheses": hypotheses}))
print(f"whisper-small WER {wer:.4f} on {len(errors)} utterances")
import os; os._exit(0)
