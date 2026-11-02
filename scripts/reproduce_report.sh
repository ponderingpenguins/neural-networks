#!/bin/bash
# Every experiment behind the report, in the order it was run (about 1.5 days on one RTX 3070 8 GB). Each step is its
# own script and can be run alone; this file only fixes the order. Run from the repository root. Needs ~60 GB of free
# disk for the baked embeddings (step 1). The numbers of each step are logged in DEVLOG.md (entry numbers in brackets).
set -euo pipefail

bash scripts/local_full.sh       # [1-3]  bake 12,000 train utterances, train the MLP projector for 12,000 steps
bash scripts/dense_run.sh        # [5-6]  checkpoints every 25 steps + audio-swap gap, attention, frame statistics
bash scripts/law_runs.sh         # [7-9]  seed / lr / batch size / init scale vs the step of the loss drop
bash scripts/short_sweep.sh      # [10-12] warmup, gradient clipping and norm penalty 0.001-10 (1,500 steps each)
bash scripts/baseline_seeds.sh   # [13]   linear baseline and extra seeds (results/seeds)
bash scripts/test_eval.sh        # [14]   test WER on 500 test.clean utterances for every trained model
bash scripts/pen1_long.sh        # [14]   penalty 1, 4,000 steps
bash scripts/overnight2.sh       # [15]   word recall on the seed runs, vocabulary-mixture projector with logit scale
bash scripts/overnight3.sh       # [15]   penalty 0.3 and a second seed of penalty 1

source .venv/bin/activate
python analysis/plot_seeds.py            # results/seeds/seeds_drop.png
python wer_summary.py                    # results/test_eval/wer_summary.txt
python error_analysis.py model=long12000 # results/test_eval/error_analysis_long12000.{txt,png}
