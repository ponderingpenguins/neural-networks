#!/bin/bash
# Nearest-token replacement WER on the first 100 test.clean utterances (the baked 500 of eval_test_wer.sh), for the
# 12,000-step two-layer MLP and the 1,500-step clip-only / penalty 1e-3 / penalty 1e-2 runs.
# Results: results/test_eval/replacement_wer100_<model>/analysis_test.json. Run from the repository root.
set -euo pipefail
source .venv/bin/activate

run() {  # run <name> <projector.pt>
    python analyze.py output_path=results/test_eval subset=test max_examples=100 wer_examples=100 viz_examples=0 \
        "checkpoints=[$2]" out_name="replacement_wer100_$1" > "results/test_eval/replacement_wer100_$1.log" 2>&1
}

run long12000 results/local_full/seed_1/projector.pt
run clip1 results/short_sweep/clip1/projector.pt
run pen1e-3 results/short_sweep/clip1_pen1e-3/projector.pt
run pen1e-2 results/short_sweep/clip1_pen1e-2/projector.pt
echo done
