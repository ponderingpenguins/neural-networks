#!/bin/bash
# Does penalty 1 only delay the loss drop? Same run as clip1_pen1 (seed 1, lr 1e-4, clip 1.0) for 4,000 steps.
# Waits for eval_test_wer.sh. Output: results/pen1_long/pen1_long
set -euo pipefail
while pgrep -f '[e]val_test_wer.sh' > /dev/null; do sleep 60; done
OUT=results/pen1_long STEPS=4000 RUNS="pen1_long:1.0:1.0" bash scripts/exp_penalty_clip_sweep.sh
