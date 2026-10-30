# neural-networks

## Quick-lookup

paper: https://arxiv.org/pdf/2402.08846
code: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/asr_librispeech


## Micro abstract



Ma et al. (2024) systematically explored how LLM-based ASR systems can be constructed by only training a linear projector to align the speech encoder and the LLM. However, the paper did not analyse how the projector maps the speech embeddings to embeddings usefull to the LLM. Our research question is: after training, are the projected speech embeddings similar to the text embeddings of the spoken words, or do they occupy a seperate region of the LLM's embedding space that the LLM learns to read? Using the SLAM-ASR method proposed in the paper (frozen WHisper encoder and TinyLLaMA-1.1B-chat, trained on LibriSpeech), we compare each projected speech embeddings to the LLM's text embeddings using cosine similarity, then select the K nearest neighbours to analyse if neighbours align with the spoken words. We will also replace the projected speech embeddings with the nearest neighbour in the LLM's vocabulary to observe the effect on WER. We will also compare our results against an untrained projector for control. Our hypothesis is that the projected speech embeddings do align with the text embeddings of the words spoken.


## Development

Make sure to be in the root directory of the project `cd neural-networks`

To load the environment

```
uv sync
```

## Training

`scripts/quick_test.sh` is a one-example check. To bake, train (3 seeds) and analyse the full run on a Habrok A100:

```
sbatch scripts/train.sh
```

For a short end-to-end check on the cluster first (10% of the data, one seed, about 45 minutes or less):

```
sbatch scripts/short_run.sh
```

Settings can be overridden from the environment, e.g. `sbatch --export=ALL,SEEDS=1,STEPS=200 scripts/train.sh`.

### Running locally

To run a short experiment on a single consumer GPU (built for an RTX 3060 with 12 GB), no Slurm needed. Create the virtual environment first if it does not exist (`uv venv`), then from the repository root:

```
bash scripts/local_run.sh
```

It runs the same bake, train and analyse pipeline as `train.sh` on 500 training samples, one seed and 300 steps, and writes to `results/local_test`. It exits early if no NVIDIA GPU is found. The batch size defaults to 4 because the language model runs in fp32; if you hit CUDA out-of-memory, lower it, e.g. `BATCH_SIZE=2 bash scripts/local_run.sh`. Other settings (`STEPS`, `MAX_SAMPLES`, `OUT`, ...) can be overridden the same way.

If it does not work, you might need to give it excecutable permsissions first:

```
chmod +x ./scripts/*.sh
```
