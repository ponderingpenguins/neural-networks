# neural-networks

SLAM-ASR-style speech recognition with a frozen speech encoder and a frozen language model, connected by a small trained projector. Semester project for Neural Networks (Jaeger).

Paper: https://arxiv.org/pdf/2402.08846 (Ma et al., 2024). Original code: https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/asr_librispeech

## Research question

After training, are the projected speech embeddings similar to the text embeddings of the spoken words, or do they occupy a separate region of the LM's embedding space that the LM learns to read?

## Setup

- Whisper-small encoder (frozen), frames grouped 5 at a time (300 x 3840 per utterance, baked once to disk in fp16)
- Projector (trained): 2-layer MLP 3840 -> 2048 -> 1024, or one linear layer 3840 -> 1024 as the baseline
- Qwen3-0.6B (frozen), prompted with the projected speech embeddings and trained to emit the transcript
- Data: 12,000 utterances of LibriSpeech `train.clean.100` (`openslr/librispeech_asr` on Hugging Face), validation on 100 utterances of `validation.clean`, test on the first 500 of `test.clean`

## Main results (500 test utterances, WER)

| Model | WER |
|---|---|
| Whisper-small with its own decoder (reference) | 4.5% |
| Untrained projector (MLP / linear) | 169% / 117% |
| Linear projector, 1,500 steps, 3 seeds | 20.3% +- 0.8 |
| MLP projector, 1,500 steps, 3 seeds | 18.8% +- 0.4 |
| MLP + norm penalty 0.1, 3 seeds | 29.6% +- 7.9 |
| MLP, 12,000 steps, 1 seed | 12.3% |

Without a penalty the projected frames have norm 39-46 against about 0.93 for the LM's word embeddings, and the nearest vocabulary token is almost never a spoken word (recall about 0). A norm penalty makes the output partly text-like (recall about 0.33 at penalty 0.1) but costs accuracy. Details: `DEVLOG.md` (day-by-day log with every experiment) and `docs/RESULTS_COURSE_VOCABULARY.md` (results in the vocabulary of the course notes). Summary tables are in `results_summary/`, figures in `figures/`.

## Layout

```
main.py                 bake / train / evaluate entry point (stage=bake|train)
utils.py, bake.py       model, projectors, training config, Whisper baking
analyze.py              projector vs. text-embedding analysis (nearest tokens, norms, word recall)
eval_wer.py             test WER of trained projectors
whisper_wer.py          test WER of Whisper-small alone
wer_summary.py          group means, seed std and bootstrap intervals
error_analysis.py       substitutions / insertions / deletions, WER by length
analysis/               run-comparison and seed plots
scripts/                one shell script per experiment, see "Scripts" below; reproduce_report.sh gives the order
docs/                   results written in the course vocabulary
figures/, results_summary/   report figures and result tables (results/ itself is not tracked)
DEVLOG.md               experiment log
```

## Running

Needs a CUDA GPU (developed on an RTX 3070 8 GB, fp32 LM, batch size 4) and about 60 GB of free disk for the baked embeddings.

```
uv sync
bash scripts/exp_long_mlp_12000steps.sh    # bake 12,000 utterances + one long training run -> results/local_full
bash scripts/exp_penalty_clip_sweep.sh     # short runs on the baked embeddings (warmup, clipping, norm penalty)
bash scripts/exp_linear_vs_mlp_seeds.sh    # linear baseline and extra seeds -> results/seeds
bash scripts/eval_test_wer.sh              # test WER of all models -> results/test_eval
```

All scripts run from the repository root, take overrides from the environment (`STEPS`, `OUT`, `BAKED`, ...), and activate `.venv`. `bash scripts/reproduce_report.sh` runs every experiment in the order it was originally run.

## Scripts

The prefix says what a script does: `pipeline_` is the reusable bake/train/analyze flow, `exp_` trains models for one experiment, `eval_` measures WER of trained models, `analyze_` runs the embedding analysis on existing runs.

| Script | What it does |
|---|---|
| `pipeline_bake_train_analyze.sh` | Bake Whisper embeddings, train projector(s), analyse every checkpoint. The original Slurm (Habrok) script; the local scripts wrap it. |
| `exp_long_mlp_12000steps.sh` | Long local run: bake 12,000 utterances, train the two-layer MLP for 12,000 steps with early stopping |
| `exp_penalty_clip_sweep.sh` | 1,500-step runs on the baked embeddings: warmup, gradient clipping, norm penalty |
| `exp_linear_vs_mlp_seeds.sh` | Linear baseline against the MLP over 3 seeds, with and without penalty 0.1 |
| `exp_penalty1_4000steps.sh` | Norm penalty 1 for 4,000 steps: is the loss drop delayed or prevented? |
| `exp_penalty03_and_seed2.sh` | Penalty 0.3 (seed 1) and penalty 1 (seed 2), then alignment analysis on both |
| `eval_test_wer.sh` | Test WER on 500 `test.clean` utterances for Whisper-small, untrained projectors and every trained model |
| `eval_replacement_wer.sh` | Nearest-token replacement WER on 100 validation utterances |
| `eval_replacement_wer_test.sh` | The same replacement WER on the first 100 `test.clean` utterances, for the long run and three short runs |
| `analyze_seed_word_recall.sh` | Word recall and frame norms on the seed runs |
| `reproduce_report.sh` | Runs the experiments above in their original order |
