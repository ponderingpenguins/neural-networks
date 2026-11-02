#!/bin/bash
# Long single-seed run on the RTX 3070 (8 GB, ~60 GB of free disk), no Slurm needed: a larger slice of train.clean.100
# than local_run.sh, with early stopping. Run from the repository root with `bash scripts/local_full.sh`; everything the
# run prints is also saved to results/local_full/local-YYYYmmdd-HHMMSS.out. It resumes where it left off if the bake is
# interrupted (existing shards are skipped). Everything can be overridden from the environment.
# 12000 samples take ~28 GB of baked shards (~2.3 MB each); all 28539 would need ~65 GB.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "No NVIDIA GPU found; this script is meant for the local GPU." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export MAX_SAMPLES=${MAX_SAMPLES:-12000}
export VAL_SAMPLES=${VAL_SAMPLES:-200}
export TEST_SAMPLES=${TEST_SAMPLES:-100}
export SEEDS=${SEEDS:-1}
export STEPS=${STEPS:-12000}          # ~1.5 s/step at batch size 4; one epoch is 3000 steps
export BATCH_SIZE=${BATCH_SIZE:-4}    # the LM runs in fp32 and is backpropagated through; 8 GB is tight
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
export ANALYZE_BATCH_SIZE=${ANALYZE_BATCH_SIZE:-4}
export PATIENCE=${PATIENCE:-5}
export MIN_STEPS=${MIN_STEPS:-3000}
export EVAL_EVERY=${EVAL_EVERY:-500}
export WER_EVERY=${WER_EVERY:-4000}   # greedy decoding is slow on this GPU
export SAVE_EVERY=${SAVE_EVERY:-3000}
export TRAIN_EVAL_SAMPLES=${TRAIN_EVAL_SAMPLES:-50}
export WER_EXAMPLES=${WER_EXAMPLES:-30}
export OUT=${OUT:-./results/local_full}

mkdir -p "$OUT"
LOG="$OUT/local-$(date +%Y%m%d-%H%M%S).out"
echo "Logging to $LOG"

# pipefail (set above) keeps train.sh's exit status through the pipe to tee
bash scripts/train.sh 2>&1 | tee "$LOG"
