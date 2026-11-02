#!/bin/bash
# Every experiment behind the report, in the order it was run (about 1.5 days on one RTX 3070 8 GB). Each step is its
# own script and can be run alone; this file only fixes the order. Run from the repository root. Needs ~60 GB of free
# disk for the baked embeddings (step 1). The numbers of each step are logged in DEVLOG.md (entry numbers in brackets).
set -euo pipefail

bash scripts/exp_long_mlp_12000steps.sh       # [1-3]  bake 12,000 train utterances, train the MLP projector for 12,000 steps
bash scripts/exp_penalty_clip_sweep.sh      # [10-12] warmup, gradient clipping and norm penalty 0.001-10 (1,500 steps each)
bash scripts/exp_linear_vs_mlp_seeds.sh   # [13]   linear baseline and extra seeds (results/seeds)
bash scripts/eval_test_wer.sh        # [14]   test WER on 500 test.clean utterances for every trained model
bash scripts/exp_penalty1_4000steps.sh        # [14]   penalty 1, 4,000 steps
bash scripts/analyze_seed_word_recall.sh       # [15]   word recall on the seed runs
bash scripts/exp_penalty03_and_seed2.sh       # [15]   penalty 0.3 and a second seed of penalty 1

source .venv/bin/activate
python analysis/plot_seeds.py            # results/seeds/seeds_drop.png
python wer_summary.py                    # results/test_eval/wer_summary.txt
python error_analysis.py model=long12000 # results/test_eval/error_analysis_long12000.{txt,png}
