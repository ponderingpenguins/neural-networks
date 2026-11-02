"""What do the vocabulary-mixture projector's per-frame word distributions look like?

For validation utterances: mean entropy and max weight of the per-frame distribution over tokens, how much of each
transcript's token set shows up among the frames' arg-max tokens (against the same count for another utterance's
transcript), the share of frames whose arg-max token belongs to the utterance's own transcript, and the collapsed arg-max
token sequence next to the transcript for a few utterances.

Usage: python vocab_eval.py output_path=results/vocab/clip1_vocab
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from bake import load_baked_embeddings, speech_frame_counts
from utils import TrainConfig, VocabMixtureProjector, load_lm

args = dict(a.split("=") for a in sys.argv[1:])
run = Path(args.get("output_path", "results/vocab/clip1_vocab"))
count = int(args.get("max_examples", 64))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings, _, texts, num_samples = load_baked_embeddings(TrainConfig(dataset_name="", max_samples=0, output_path=run), "val")
embeddings, texts = embeddings[:count], texts[:count]
counts = speech_frame_counts(num_samples)[:count]
tokenizer, _ = load_lm(torch.device("cpu"))
state = {k: v.to(device) for k, v in torch.load(run / "projector.pt", map_location=device).items()}
projector = VocabMixtureProjector(embeddings.shape[-1], state["vocab_ids"], state["vocab_embeddings"], float(args.get("vocab_scale", 1.0))).to(device)
projector.load_state_dict(state)
projector.eval()
ids = state["vocab_ids"].cpu()

entropy, max_weight, recall, recall_other, precision, precision_other = [], [], [], [], [], []
argmax_sets, own_sets, shown = [], [], []
with torch.no_grad():
    for u in range(count):
        n = int(counts[u])
        p = projector.distribution(embeddings[u : u + 1, :n].to(device, torch.float32))[0]  # [n, K]
        entropy.append(float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()))
        max_weight.append(float(p.max(-1).values.mean()))
        top = ids[p.argmax(-1).cpu()]
        argmax_sets.append(top)
        own_sets.append(set(tokenizer(texts[u], add_special_tokens=False).input_ids))
        shown.append(top.tolist())
for u in range(count):
    other = (u + 1) % count
    top_set, own, other_own = set(argmax_sets[u].tolist()), own_sets[u], own_sets[other]
    recall.append(len(own & top_set) / len(own))
    recall_other.append(len(other_own & top_set) / len(other_own))
    frames = argmax_sets[u].tolist()
    precision.append(np.mean([t in own for t in frames]))
    precision_other.append(np.mean([t in other_own for t in frames]))
summary = {
    "mean_entropy_nats": float(np.mean(entropy)), "log_vocab": float(np.log(len(ids))), "mean_max_weight": float(np.mean(max_weight)),
    "transcript_token_recall": float(np.mean(recall)), "recall_with_other_transcript": float(np.mean(recall_other)),
    "argmax_in_own_transcript": float(np.mean(precision)), "argmax_in_other_transcript": float(np.mean(precision_other)),
    "distinct_argmax_tokens_per_utterance": float(np.mean([len(set(t.tolist())) for t in argmax_sets])),
}
print(json.dumps(summary, indent=1))
for u in range(3):
    seq = [t for i, t in enumerate(shown[u]) if i == 0 or t != shown[u][i - 1]]
    print(f"\ntranscript: {texts[u][:150]}\nargmax tokens (repeats collapsed): {tokenizer.decode(seq)[:300]!r}")
(run / "vocab_eval.json").write_text(json.dumps(summary, indent=2))
