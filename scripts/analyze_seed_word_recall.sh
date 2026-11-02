#!/bin/bash
# Alignment analysis (word recall, frame norm) on the seed runs, for error bars on the central question.
# Waits for eval_test_wer.sh and exp_penalty1_4000steps.sh. Output: results/seeds/*/analysis_norm.
set -uo pipefail
source .venv/bin/activate
while pgrep -f '[e]val_test_wer.sh|[e]xp_penalty1_4000steps.sh' > /dev/null; do sleep 60; done

for D in results/seeds/mlp_s2 results/seeds/mlp_s3 results/seeds/mlppen_s2 results/seeds/mlppen_s3 \
         results/seeds/linear_s1 results/seeds/linear_s2 results/seeds/linear_s3; do
    python analyze.py output_path=$D subset=val max_examples=64 wer_examples=30 viz_examples=0 \
        "checkpoints=[$D/projector.pt]" out_name=analysis_norm > $D/analysis_norm.log 2>&1
done

echo done
