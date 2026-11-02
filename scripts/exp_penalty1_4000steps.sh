#!/bin/bash
# Does norm penalty 1 only delay the drop? Same run as clip1_pen1 (seed 1, lr 1e-4, clip 1.0) but for 4,000 steps.
# Waits until the test-WER queue (scripts/eval_test_wer.sh) is done. Output: results/pen1_long/pen1_long
set -euo pipefail
while pgrep -f '[e]val_test_wer.sh' > /dev/null; do sleep 60; done
OUT=results/pen1_long STEPS=4000 RUNS="pen1_long:1.0:1.0" bash scripts/exp_penalty_clip_sweep.sh
