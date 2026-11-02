"""Does the frozen LM read the few output directions that the projector's second layer learned to write into?

The change of the projector's second layer between a "before" and an "after" checkpoint is close to rank 1-8 (see
analysis/projector_anatomy.py). Here those directions in the LM's embedding space are compared with what the LM reads:
  * revert test: in the after-model's projected frames, put the before-model's values back along k directions
  * inject test: in the before-model's frames, put the after-model's values along k directions
  * gradient test: the share of the loss gradient w.r.t. the projected frames that lies in the k directions
  * first-layer gain: how strongly the LM's first attention layer (q, k, v) amplifies the directions
Subspaces compared: top-k left singular vectors of the second layer's weight change, top-k principal directions of the
change in projected frames, and random k-dimensional subspaces (3 draws, mean).

Usage: python analysis/direction_probe.py output_path=results/dense/clip1 before=350 after=700
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # pipeline modules live in the repo root
import dense_analysis as da
from bake import load_baked_embeddings, speech_frame_counts
from utils import TrainConfig, build_projector, build_prompt_embeddings, load_lm

args = dict(a.split("=") for a in sys.argv[1:])
run = Path(args.get("output_path", "results/dense/clip1"))
before_step, after_step = int(args.get("before", 350)), int(args.get("after", 700))
count = int(args.get("max_examples", 32))
device = da.device
ks = (1, 2, 4, 8, 16)

embeddings, _, texts, num_samples = load_baked_embeddings(TrainConfig(dataset_name="", max_samples=0, output_path=run), "val")
embeddings, texts = embeddings[:count], texts[:count]
counts = speech_frame_counts(num_samples)[:count]
tokenizer, lm = load_lm(device)
da.LM = lm
prefix, suffix = build_prompt_embeddings(tokenizer, lm, device)
token_ids = [torch.tensor(tokenizer(t, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]) for t in texts]
valid = torch.zeros(count, 300, dtype=torch.bool, device=device)
for u in range(count):
    valid[u, : int(counts[u])] = True


def state(step):
    return {k: v.to(device) for k, v in torch.load(run / "checkpoints" / f"projector_step{step}.pt", map_location=device).items()}


def project(s):
    projector = build_projector(embeddings.shape[-1], lm.config.hidden_size, device)
    projector.load_state_dict(s)
    with torch.no_grad():
        return torch.cat([projector.eval()(embeddings[i : i + 4].to(device, torch.float32)) for i in range(0, count, 4)])


@torch.no_grad()
def loss(y, swapped=False):
    order = [(i + 1) % count for i in range(count)] if swapped else list(range(count))
    return da.mean_loss(prefix, suffix, token_ids, 4, lambda a, b: y[order[a:b]])


def gradient(y):
    """Gradient of the mean batch loss w.r.t. the projected frames, [N, 300, D]."""
    grads = []
    for a in range(0, count, 4):
        yb = y[a : a + 4].clone().requires_grad_(True)
        ids = token_ids[a : a + 4]
        inputs, mask, labels, _ = da.make_sequences(prefix, suffix, ids, yb)
        lm(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss.backward()
        grads.append(yb.grad.detach())
    return torch.cat(grads)


def mix(y_keep, y_take, q):
    """y_keep with its components along the orthonormal columns of q replaced by y_take's."""
    return y_keep - (y_keep @ q) @ q.T + (y_take @ q) @ q.T


before, after = state(before_step), state(after_step)
y_before, y_after = project(before), project(after)
u, s, _ = torch.linalg.svd(after["2.weight"] - before["2.weight"], full_matrices=False)
delta = (y_after - y_before)[valid]
_, _, vt = torch.linalg.svd(delta - delta.mean(0), full_matrices=False)
rng = torch.Generator(device="cpu").manual_seed(0)
dim = lm.config.hidden_size


def random_q(k):
    return torch.linalg.qr(torch.randn(dim, k, generator=rng).to(device))[0]


subspaces = {
    "dW2 singular vectors": lambda k: u[:, :k],
    "PCA of output change": lambda k: vt[:k].T,
    "random": random_q,
}
results = {"before_loss": loss(y_before), "after_loss": loss(y_after), "singular_values": s[:16].tolist(), "tests": {}}
print(f"before {results['before_loss']:.3f} after {results['after_loss']:.3f}; top singular values {[round(float(x), 2) for x in s[:6]]}")

g_after, g_before = gradient(y_after), gradient(y_before)
for name, make in subspaces.items():
    rows = []
    for k in ks:
        draws = 3 if name == "random" else 1
        row = {"k": k, "revert": 0.0, "inject": 0.0, "inject_swapped": 0.0, "grad_after": 0.0, "grad_before": 0.0}
        for _ in range(draws):
            q = make(k)
            row["revert"] += loss(mix(y_after, y_before, q)) / draws
            injected = mix(y_before, y_after, q)
            row["inject"] += loss(injected) / draws
            row["inject_swapped"] += loss(injected, swapped=True) / draws
            for key, g in (("grad_after", g_after), ("grad_before", g_before)):
                row[key] += float(((g[valid] @ q) ** 2).sum() / (g[valid] ** 2).sum()) / draws
        rows.append(row)
        print(f"{name:>22} k={k:>2}: revert {row['revert']:.2f}  inject {row['inject']:.2f} (swapped {row['inject_swapped']:.2f})  "
              f"gradient share after {row['grad_after']:.3f} before {row['grad_before']:.3f} (isotropic {k / dim:.4f})")
    results["tests"][name] = rows

# How strongly the LM's first attention layer amplifies a direction (input norm weight included), relative to a random one.
layer = lm.model.layers[0]
gain = {}
for proj in ("q_proj", "k_proj", "v_proj"):
    w = getattr(layer.self_attn, proj).weight * layer.input_layernorm.weight[None]
    typical = (w**2).sum() / dim
    gain[proj] = [float((w @ u[:, i]).pow(2).sum() / typical) for i in range(4)]
results["layer0_gain_of_dW2_singular_vectors"] = gain
print("layer-0 gain (1 = a random direction):", {k: [round(x, 2) for x in v] for k, v in gain.items()})
out = run / "projector_anatomy"
out.mkdir(exist_ok=True)
(out / "direction_probe.json").write_text(json.dumps(results, indent=2))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name, rows in results["tests"].items():
    axes[0].plot(ks, [r["revert"] for r in rows], "o-", label=name)
    axes[1].plot(ks, [r["inject"] for r in rows], "o-", label=name)
    axes[2].plot(ks, [r["grad_before"] for r in rows], "o-", label=f"{name}, before-model")
axes[2].plot(ks, [k / dim for k in ks], "k:", label="isotropic")
axes[0].axhline(results["after_loss"], color="k", ls=":")
axes[0].set(title="revert k directions to the before-model\n(after-model's loss: dotted)", xlabel="k", xscale="log", ylabel="val loss")
axes[1].axhline(results["before_loss"], color="gray", ls=":")
axes[1].set(title="inject k directions into the before-model\n(before-model's loss: dotted)", xlabel="k", xscale="log")
axes[2].set(title="share of the loss gradient w.r.t. the frames\nlying in the k directions", xlabel="k", xscale="log", yscale="log")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "direction_probe.png", dpi=130)
