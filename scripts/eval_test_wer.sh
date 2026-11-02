#!/bin/bash
# Test WER on the first 500 test.clean utterances for Whisper-small (reference), untrained projectors and every
# trained projector. Waits for the GPU to be free. Output: results/test_eval/*.json
set -euo pipefail
source .venv/bin/activate
while pgrep -f '[m]ain.py|[a]nalyze.py' > /dev/null; do sleep 30; done

OUT=results/test_eval
mkdir -p $OUT
python main.py stage=bake dataset_name=openslr/librispeech_asr max_samples=0 val_samples=0 test_samples=500 \
    output_path=$OUT > $OUT/bake.log 2>&1
python whisper_wer.py n=500 out=$OUT/whisper_wer.json > $OUT/whisper.log 2>&1
MODELS="init_mlp:mlp:init,init_linear:linear:init"
for S in 1 2 3; do MODELS+=",linear_s$S:linear:results/seeds/linear_s$S/projector.pt"; done
MODELS+=",mlp_s1:mlp:results/short_sweep/clip1/projector.pt,mlp_s2:mlp:results/seeds/mlp_s2/projector.pt,mlp_s3:mlp:results/seeds/mlp_s3/projector.pt"
MODELS+=",pen0.1_s1:mlp:results/short_sweep/clip1_pen1e-1/projector.pt,pen0.1_s2:mlp:results/seeds/mlppen_s2/projector.pt,pen0.1_s3:mlp:results/seeds/mlppen_s3/projector.pt"
MODELS+=",pen1:mlp:results/short_sweep/clip1_pen1/projector.pt,pen10:mlp:results/short_sweep/clip1_pen10/projector.pt"
MODELS+=",long12000:mlp:results/local_full/seed_1/projector.pt"
python eval_wer.py test_dir=$OUT out=$OUT/wer_results.json models=$MODELS > $OUT/eval.log 2>&1
echo finished
