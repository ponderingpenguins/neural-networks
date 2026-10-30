#!/bin/bash
#SBATCH --job-name=slam-projector
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Full run on Habrok: bake once, train the projector for several seeds, analyse every checkpoint.
# Submit from the repository root with `sbatch scripts/train.sh`; the gpu partition allows at most 3 days.
# Every setting can be overridden from the environment, e.g. `sbatch --export=ALL,SEEDS=1,STEPS=200 scripts/train.sh`.
# Use scripts/quick_test.sh for a one-example check first.
set -euo pipefail

# Optional: ping an endpoint on a VPS when the job ends (success, failure or timeout kill); the VPS sends the email
# (see scripts/notify_server.py). Nothing happens unless NOTIFY_URL and NOTIFY_TOKEN are both set.
notify() {
    local status=$?
    if [[ -n "${NOTIFY_URL:-}" && -n "${NOTIFY_TOKEN:-}" ]]; then
        curl -fsS --max-time 20 -H "X-Token: $NOTIFY_TOKEN" \
            --data-urlencode "subject=slam job ${SLURM_JOB_ID:-local} exited with status $status" \
            --data-urlencode "body=Job ${SLURM_JOB_NAME:-train.sh} on $(hostname) finished after ${SECONDS}s with exit status $status. Output: slurm-${SLURM_JOB_ID:-local}.out" \
            "$NOTIFY_URL" || true
    fi
}
trap notify EXIT

source .venv/bin/activate
uv sync

DATASET=${DATASET:-openslr/librispeech_asr}
SPLIT=${SPLIT:-train.clean.100}
MAX_SAMPLES=${MAX_SAMPLES:-5000}   # all baked embeddings are held in RAM: ~2.3 MB each (5000 -> ~11.5 GB)
VAL_SAMPLES=${VAL_SAMPLES:-500}
TEST_SAMPLES=${TEST_SAMPLES:-500}
SEEDS=${SEEDS:-"1 2 3"}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}                     # the paper's peak learning rate
WARMUP_STEPS=${WARMUP_STEPS:-1000}
TRAIN_EVAL_SAMPLES=${TRAIN_EVAL_SAMPLES:-200}
WER_EXAMPLES=${WER_EXAMPLES:-200}  # examples used for the replacement-WER test in the analysis
EVAL_EVERY=${EVAL_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-500}
OUT=${OUT:-./results}

# Stage 1: encode the audio with Whisper once. Shards are shared by all seeds, so bake into one directory.
BAKED="$OUT/baked"
uv run main.py stage=bake dataset_name="$DATASET" dataset_split="$SPLIT" output_path="$BAKED" \
    max_samples="$MAX_SAMPLES" val_samples="$VAL_SAMPLES" test_samples="$TEST_SAMPLES" bake_batch_size=32

for SEED in $SEEDS; do
    RUN="$OUT/seed_$SEED"
    mkdir -p "$RUN"
    for DIR in speech_embeddings speech_embeddings_val speech_embeddings_test; do
        ln -sfn "$(cd "$BAKED" && pwd)/$DIR" "$RUN/$DIR"
    done

    # Stage 2: train the projector on the cached embeddings; checkpoints go to $RUN/checkpoints
    uv run main.py stage=train dataset_name="$DATASET" dataset_split="$SPLIT" output_path="$RUN" \
        max_samples="$MAX_SAMPLES" val_samples="$VAL_SAMPLES" test_samples="$TEST_SAMPLES" \
        seed="$SEED" steps="$STEPS" batch_size="$BATCH_SIZE" lr="$LR" warmup_steps="$WARMUP_STEPS" \
        eval_every="$EVAL_EVERY" save_every="$SAVE_EVERY" train_eval_samples="$TRAIN_EVAL_SAMPLES" eval_batch_size=32

    # Stage 3: analyse the untrained projector, every checkpoint and the final projector
    uv run analyze.py output_path="$RUN" subset=val seed="$SEED" batch_size=16 wer_examples="$WER_EXAMPLES"
done
