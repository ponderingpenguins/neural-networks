#!/bin/bash
# One short clipped run that saves a projector checkpoint every few steps, then analysis/dense_analysis.py on all of them
# (does the LM use the audio, where does its attention go, what do the projected frames look like). Run from the
# repository root with `bash scripts/dense_run.sh`. Each checkpoint is about 40 MB, so STEPS/SAVE_EVERY of them are
# written to $OUT/checkpoints. Needs the baked shards in BAKED (made by scripts/local_full.sh).
set -euo pipefail

BAKED=${BAKED:-results/local_full/baked}
OUT=${OUT:-results/dense/clip1}
STEPS=${STEPS:-900}
SAVE_EVERY=${SAVE_EVERY:-25}
SEED=${SEED:-1}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-50}
GRAD_CLIP=${GRAD_CLIP:-1.0}

source .venv/bin/activate

mkdir -p "$OUT"
for D in speech_embeddings speech_embeddings_val; do
    ln -sfn "$(cd "$BAKED" && pwd)/$D" "$OUT/$D"
done
python main.py stage=train dataset_name=openslr/librispeech_asr dataset_split=train.clean.100 \
    output_path="$OUT" max_samples=12000 val_samples=100 test_samples=0 seed="$SEED" \
    steps="$STEPS" batch_size=4 lr="$LR" warmup_steps="$WARMUP_STEPS" grad_clip="$GRAD_CLIP" norm_penalty=0 \
    eval_every=100 wer_every=0 save_every="$SAVE_EVERY" train_eval_samples=0 eval_batch_size=4 \
    patience=0 final_wer=false > "$OUT/run.log" 2>&1
python analysis/dense_analysis.py output_path="$OUT" max_examples=64 attention_examples=8 batch_size=4 > "$OUT/dense_analysis.log" 2>&1
tail -n 40 "$OUT/dense_analysis.log"
