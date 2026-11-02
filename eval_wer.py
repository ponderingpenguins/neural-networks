"""Test WER of trained projectors (and of an untrained one) on baked test.clean embeddings.

Usage: python eval_wer.py test_dir=results/test_eval out=results/test_eval/wer_results.json
Each model is "name:projector_type:path" ("init" as path = freshly initialised, untrained projector, seed 1).
Writes per-model corpus WER plus per-utterance edit counts (for bootstrap intervals) and hypotheses.
"""
import json
import sys
from pathlib import Path

import torch

from bake import load_baked_embeddings
from utils import (
    TrainConfig,
    _edit_distance,
    build_linear_projector,
    build_projector,
    build_prompt_embeddings,
    load_lm,
    normalize_text,
    transcribe,
)

args = dict(a.split("=", 1) for a in sys.argv[1:])
test_dir = Path(args["test_dir"])
out_path = Path(args["out"])
models = args["models"].split(",")
device = torch.device("cuda")

cfg = TrainConfig(dataset_name="openslr/librispeech_asr", max_samples=0, output_path=test_dir, test_samples=1)
embeddings, ids, texts, _ = load_baked_embeddings(cfg, "test")
tokenizer, lm = load_lm(device)
prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)
results = json.loads(out_path.read_text()) if out_path.exists() else {}

for spec in models:
    name, kind, path = spec.split(":", 2)
    if name in results:
        continue
    builder = build_linear_projector if kind == "linear" else build_projector
    projector = builder(embeddings.shape[-1], lm.config.hidden_size, device)
    if path == "init":
        torch.manual_seed(1)
        projector = builder(embeddings.shape[-1], lm.config.hidden_size, device)
    else:
        projector.load_state_dict(torch.load(path, map_location=device))
    projector.eval()
    hypotheses = []
    for start in range(0, len(texts), 8):
        batch = embeddings[start : start + 8].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            speech = projector(batch)
        hypotheses += transcribe(tokenizer, lm, prefix, suffix, speech, 200)
    errors, words = [], []
    for reference, hypothesis in zip(texts, hypotheses):
        ref_words = normalize_text(reference).split()
        errors.append(_edit_distance(ref_words, normalize_text(hypothesis).split()))
        words.append(len(ref_words))
    wer = sum(errors) / sum(words)
    results[name] = {"wer": wer, "errors": errors, "words": words, "hypotheses": hypotheses}
    out_path.write_text(json.dumps(results))
    print(f"{name}: test WER {wer:.4f} on {len(texts)} utterances", flush=True)
