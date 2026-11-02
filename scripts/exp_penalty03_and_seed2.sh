#!/bin/bash
# Penalty 0.3 (seed 1), between 0.1 (drops) and 1 (no drop in 1,500 steps), and penalty 1 with seed 2.
# Both at the sweep settings (clip 1.0, lr 1e-4, 1,500 steps), then the alignment analysis on each.
# Waits for the other overnight scripts.
set -uo pipefail
source .venv/bin/activate
while pgrep -f '[e]val_test_wer.sh|[e]xp_penalty1_4000steps.sh|[a]nalyze_seed_word_recall.sh' > /dev/null; do sleep 60; done

RUNS="clip1_pen3e-1:1.0:0.3" bash scripts/exp_penalty_clip_sweep.sh > results/short_sweep_pen03.log 2>&1
SEED=2 OUT=results/seeds RUNS="pen1_s2:1.0:1.0" bash scripts/exp_penalty_clip_sweep.sh > results/seeds_pen1_s2.log 2>&1
for D in results/short_sweep/clip1_pen3e-1 results/seeds/pen1_s2; do
    python analyze.py output_path=$D subset=val max_examples=64 wer_examples=30 viz_examples=0 \
        "checkpoints=[$D/projector.pt]" out_name=analysis_norm > $D/analysis_norm.log 2>&1
done
echo done
