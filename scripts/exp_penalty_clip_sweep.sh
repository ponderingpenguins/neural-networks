#!/bin/bash
# A few very short training runs on the already baked embeddings, to get a feeling for warmup, gradient clipping and
# the output-norm penalty. Run from the repository root with `bash scripts/exp_penalty_clip_sweep.sh`; each run writes to
# results/short_sweep/<name> (train_log.csv, wer_history.json) and is compared with analysis/plot_runs.py at the end.
# Override from the environment, e.g. `STEPS=800 bash scripts/exp_penalty_clip_sweep.sh`. Needs the baked shards in BAKED
# (made by scripts/exp_long_mlp_12000steps.sh) and no GPU memory used by anything else.
set -euo pipefail

BAKED=${BAKED:-results/local_full/baked}
OUT=${OUT:-results/short_sweep}
STEPS=${STEPS:-1500}
SEED=${SEED:-1}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-50}
EVAL_EVERY=${EVAL_EVERY:-100}
# Each run is "name:grad_clip:norm_penalty"; grad_clip 0 and norm_penalty 0 mean off.
RUNS=${RUNS:-"noclip:0:0 clip1:1.0:0 clip1_pen1e-3:1.0:0.001 clip1_pen1e-2:1.0:0.01"}

source .venv/bin/activate

for RUN in $RUNS; do
    IFS=: read -r NAME CLIP PENALTY <<< "$RUN"
    DIR="$OUT/$NAME"
    mkdir -p "$DIR"
    for D in speech_embeddings speech_embeddings_val; do
        ln -sfn "$(cd "$BAKED" && pwd)/$D" "$DIR/$D"
    done
    echo "=== $NAME: grad_clip=$CLIP norm_penalty=$PENALTY warmup=$WARMUP_STEPS lr=$LR steps=$STEPS"
    python main.py stage=train dataset_name=openslr/librispeech_asr dataset_split=train.clean.100 \
        output_path="$DIR" max_samples=12000 val_samples=100 test_samples=0 seed="$SEED" \
        steps="$STEPS" batch_size=4 lr="$LR" warmup_steps="$WARMUP_STEPS" \
        grad_clip="$CLIP" norm_penalty="$PENALTY" \
        eval_every="$EVAL_EVERY" wer_every=0 save_every=0 train_eval_samples=0 eval_batch_size=4 \
        patience=0 final_wer=false > "$DIR/run.log" 2>&1
    grep -E 'val_loss' "$DIR/run.log" | tail -n 1
done

python analysis/plot_runs.py out="$OUT/compare.png" $(for RUN in $RUNS; do echo "$OUT/${RUN%%:*}"; done)
