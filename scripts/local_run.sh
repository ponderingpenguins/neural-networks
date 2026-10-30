#!/bin/bash
# Short end-to-end experiment on a single consumer GPU (built for an RTX 3060, 12 GB), no Slurm needed.
# Run from the repository root with `bash scripts/local_run.sh`. Output goes to results/local_test.
# It reuses scripts/train.sh with a small configuration; everything can be overridden from the environment,
# e.g. `STEPS=1000 BATCH_SIZE=8 bash scripts/local_run.sh`. If you hit CUDA out-of-memory, lower BATCH_SIZE.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No NVIDIA GPU found; this script is meant for the RTX 3060 machine." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export MAX_SAMPLES=${MAX_SAMPLES:-500}
export VAL_SAMPLES=${VAL_SAMPLES:-100}
export TEST_SAMPLES=${TEST_SAMPLES:-100}
export SEEDS=${SEEDS:-1}
export STEPS=${STEPS:-300}
export BATCH_SIZE=${BATCH_SIZE:-4}   # the LM runs in fp32 and is backpropagated through; 12 GB is tight at 16
export WARMUP_STEPS=${WARMUP_STEPS:-30}
export EVAL_EVERY=${EVAL_EVERY:-100}
export SAVE_EVERY=${SAVE_EVERY:-100}
export TRAIN_EVAL_SAMPLES=${TRAIN_EVAL_SAMPLES:-50}
export WER_EXAMPLES=${WER_EXAMPLES:-20}
export OUT=${OUT:-./results/local_test}

exec bash scripts/train.sh
