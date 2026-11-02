#!/bin/bash
# Last overnight batch, after the test-WER queue and the long penalty-1 run:
#  1. alignment analysis (word recall, frame norm) on the seed runs, for error bars on the central question;
#  2. vocabulary-mixture projector with a logit scale (the first attempt, entry 11, used unscaled logits, i.e. a
#     near-uniform softmax at init): scale 10 at lr 1e-4, then scale 10 at lr 3e-4.
# Waits for the earlier queues. Outputs: results/seeds/*/analysis_norm, results/vocab/scale10_lr1e-4, scale10_lr3e-4.
set -uo pipefail
source .venv/bin/activate
while pgrep -f '[t]est_eval.sh|[p]en1_long.sh|[p]en_analyze.sh' > /dev/null; do sleep 60; done

for D in results/seeds/mlp_s2 results/seeds/mlp_s3 results/seeds/mlppen_s2 results/seeds/mlppen_s3 \
         results/seeds/linear_s1 results/seeds/linear_s2 results/seeds/linear_s3; do
    python analyze.py output_path=$D subset=val max_examples=64 wer_examples=30 viz_examples=0 \
        "checkpoints=[$D/projector.pt]" out_name=analysis_norm > $D/analysis_norm.log 2>&1
done

for RUN in "scale10_lr1e-4:1e-4" "scale10_lr3e-4:3e-4"; do
    IFS=: read -r NAME LR <<< "$RUN"
    D=results/vocab/$NAME
    mkdir -p $D
    for X in speech_embeddings speech_embeddings_val; do
        ln -sfn "$(cd results/local_full/baked && pwd)/$X" "$D/$X"
    done
    python main.py stage=train dataset_name=openslr/librispeech_asr dataset_split=train.clean.100 \
        output_path=$D max_samples=12000 val_samples=100 test_samples=0 seed=1 \
        steps=3000 batch_size=4 lr=$LR warmup_steps=50 grad_clip=1.0 norm_penalty=0 projector_type=vocab vocab_scale=10 \
        eval_every=250 wer_every=0 save_every=0 train_eval_samples=0 eval_batch_size=4 \
        patience=0 final_wer=false > $D/run.log 2>&1
    python vocab_eval.py output_path=$D vocab_scale=10 > $D/vocab_eval.log 2>&1
done
echo done
