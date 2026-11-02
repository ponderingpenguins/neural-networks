#!/bin/bash
# Linear-projector baseline (single Linear layer) against the MLP, plus extra seeds for the MLP with and without the
# small norm penalty, all at the settings of scripts/exp_penalty_clip_sweep.sh. Run from the repository root; each run writes to
# results/seeds/<name>_s<seed>. Needs no other GPU use.
set -euo pipefail
source .venv/bin/activate

BAKED=${BAKED:-results/local_full/baked}
OUT=results/seeds
STEPS=${STEPS:-1500}

# "name:seed:projector_type:norm_penalty"
RUNS=${RUNS:-"linear:1:linear:0 linear:2:linear:0 linear:3:linear:0 mlp:2:mlp:0 mlp:3:mlp:0 mlppen:2:mlp:0.1 mlppen:3:mlp:0.1"}

for RUN in $RUNS; do
    IFS=: read -r NAME SEED TYPE PENALTY <<< "$RUN"
    DIR="$OUT/${NAME}_s${SEED}"
    mkdir -p "$DIR"
    for D in speech_embeddings speech_embeddings_val; do
        ln -sfn "$(cd "$BAKED" && pwd)/$D" "$DIR/$D"
    done
    echo "=== $NAME seed=$SEED projector=$TYPE norm_penalty=$PENALTY"
    python main.py stage=train dataset_name=openslr/librispeech_asr dataset_split=train.clean.100 \
        output_path="$DIR" max_samples=12000 val_samples=100 test_samples=0 seed="$SEED" \
        steps="$STEPS" batch_size=4 lr=1e-4 warmup_steps=50 grad_clip=1.0 norm_penalty="$PENALTY" \
        projector_type="$TYPE" eval_every=100 wer_every=0 save_every=0 train_eval_samples=0 eval_batch_size=4 \
        patience=0 final_wer=false > "$DIR/run.log" 2>&1
    grep -E 'val_loss' "$DIR/run.log" | tail -n 1
done
