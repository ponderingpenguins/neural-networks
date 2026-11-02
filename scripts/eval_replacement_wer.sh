#!/bin/bash
# Nearest-token replacement WER on 100 validation utterances, the same analysis as analyze.py. Output goes to
# analysis_wer100/ so the earlier 30-utterance results are kept. Run from the repository root.
set -euo pipefail
source .venv/bin/activate

run() {  # run <output_path> <projector.pt>
    python analyze.py output_path="$1" subset=val max_examples=100 wer_examples=100 viz_examples=0 \
        "checkpoints=[$2]" out_name=analysis_wer100 > "$1/analysis_wer100.log" 2>&1
}

run results/local_full/seed_1 results/local_full/seed_1/projector.pt   # 12,000-step two-layer MLP
for D in clip1 clip1_pen1e-3 clip1_pen1e-2; do                          # 1,500-step runs (clip only, penalty 1e-3, 1e-2)
    run results/short_sweep/$D results/short_sweep/$D/projector.pt
done
echo done
