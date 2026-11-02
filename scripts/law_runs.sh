#!/bin/bash
# Short clipped runs that vary one thing at a time (seed, learning rate, batch size, initial projector scale), to find
# what sets the step of the loss drop. Each run saves a checkpoint every SAVE_EVERY steps, is analysed with
# analysis/dense_analysis.py (audio-swap loss gap, attention on speech, frame statistics) and its checkpoints are then deleted.
# Run from the repository root with `bash scripts/law_runs.sh`; results go to results/law/<name>.
# Each run is "name:seed:lr:batch_size:init_scale:steps".
set -euo pipefail

BAKED=${BAKED:-results/local_full/baked}
OUT=${OUT:-results/law}
SAVE_EVERY=${SAVE_EVERY:-50}
RUNS=${RUNS:-"s2:2:1e-4:4:1:1000 s3:3:1e-4:4:1:1000 lr2e-4:1:2e-4:4:1:800 init0.5:1:1e-4:4:0.5:1000 init2:1:1e-4:4:2:1000 lr5e-5:1:5e-5:4:1:1600 bs2:1:1e-4:2:1:1500"}

source .venv/bin/activate

for RUN in $RUNS; do
    IFS=: read -r NAME SEED LR BATCH SCALE STEPS <<< "$RUN"
    DIR="$OUT/$NAME"
    mkdir -p "$DIR"
    for D in speech_embeddings speech_embeddings_val; do
        ln -sfn "$(cd "$BAKED" && pwd)/$D" "$DIR/$D"
    done
    echo "=== $NAME: seed=$SEED lr=$LR batch=$BATCH init_scale=$SCALE steps=$STEPS"
    python main.py stage=train dataset_name=openslr/librispeech_asr dataset_split=train.clean.100 \
        output_path="$DIR" max_samples=12000 val_samples=100 test_samples=0 seed="$SEED" \
        steps="$STEPS" batch_size="$BATCH" lr="$LR" warmup_steps=50 grad_clip=1.0 norm_penalty=0 init_scale="$SCALE" \
        eval_every=100 wer_every=0 save_every="$SAVE_EVERY" train_eval_samples=0 eval_batch_size=4 \
        patience=0 final_wer=false > "$DIR/run.log" 2>&1
    python analysis/dense_analysis.py output_path="$DIR" max_examples=32 attention_examples=4 batch_size=4 > "$DIR/dense_analysis.log" 2>&1
    rm -f "$DIR"/checkpoints/projector_step*.pt
    echo "done $NAME"
done
