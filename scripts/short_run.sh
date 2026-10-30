#!/bin/bash
#SBATCH --job-name=slam-short
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
# Short end-to-end check on Habrok: the same pipeline as train.sh on a tenth of the data, one seed, 300 steps.
# Submit from the repository root with `sbatch scripts/short_run.sh`. Output goes to results/short_test.
set -euo pipefail

export MAX_SAMPLES=${MAX_SAMPLES:-500}
export VAL_SAMPLES=${VAL_SAMPLES:-100}
export TEST_SAMPLES=${TEST_SAMPLES:-100}
export SEEDS=${SEEDS:-1}
export STEPS=${STEPS:-300}
export BATCH_SIZE=${BATCH_SIZE:-16}
export WARMUP_STEPS=${WARMUP_STEPS:-30}
export EVAL_EVERY=${EVAL_EVERY:-100}
export SAVE_EVERY=${SAVE_EVERY:-100}
export TRAIN_EVAL_SAMPLES=${TRAIN_EVAL_SAMPLES:-50}
export WER_EXAMPLES=${WER_EXAMPLES:-20}
export OUT=${OUT:-./results/short_test}

exec bash scripts/train.sh
