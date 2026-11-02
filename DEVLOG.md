# Devlog

Running log of experiments and results for the SLAM-ASR-style replication (frozen Whisper-small encoder → trainable 2-layer MLP projector 3840→2048→1024 → frozen Qwen3-0.6B; LibriSpeech train.clean.100). One entry per experiment: what was run, what was found, what it means, where the files are. Newest at the bottom. Claims marked *(hypothesis)* are not tested.

**Research question (assignment):** after training, are the projected speech embeddings similar to the text embeddings of the spoken words, or do they occupy a separate region that the LLM learns to read?

**Short answer so far:** a separate region the LM learns to read. Original hypothesis (alignment with word embeddings) is not supported at the input-embedding level.

All paths are relative to the repository root. Times are local (WSL on the RTX 3070 8 GB PC). Seed 1 unless noted.

---

## 0. Setup and infrastructure
- Baked Whisper encoder output once (5-frame grouping → 300×3840 fp16 per utterance, ~27 GB for 12,000 train + val) in `results/local_full/baked`; all later runs reuse it through symlinks.
- Training script `main.py`: val-loss patience / early stopping (`patience`, `min_steps`, `min_delta`), best-checkpoint restore, lazy mmap loading (`ShardedEmbeddings` in `bake.py`), per-step `train_log.csv` (loss, penalty, grad norms before clipping, per-layer grad norms, projected-frame norm, lr), options `grad_clip`, `norm_penalty`, `init_scale`, `final_wer`.
- Habrok (cluster) was busy, so everything ran on the local PC. WSL restarted by itself twice (22:05 on 2026-10-31 killed the 5e-5 run at step 899; it was rerun from scratch). Use `setsid nohup`, and expect restarts.
- Do not edit a bash script that is being executed (bash reads it incrementally); copy to a new script instead.
- Commit message rule: no attribution / Co-Authored-By lines. Nothing committed yet.

## 1. Full local run (12,000 utterances, 12,000 steps, batch 4, lr 1e-4, warmup 1000, no clipping)
- Output: `results/local_full/seed_1/`.
- Loss plateau ≈ 3.66 until step ≈ 2100, then an abrupt drop (3.0 → 1.0 between steps 2127 and 2196). Best val loss 0.2551 at step 11,500. WER: train 2.0%, val 9.7%, test 8.3%.
- So the model works as an ASR system. The sudden drop is the thing to explain.

## 2. Input-embedding analyses (projected speech vs LM text embeddings)
- Scripts: `analyze.py` (cosine heatmaps, nearest vocabulary tokens, histograms, Kumar et al. Fig. 4/5 style views), output in `results/local_full/seed_1/analysis*/`.
- Result: no word alignment at the input level. Nearest vocabulary tokens are junk, cosine to the spoken words' embeddings is barely above random; speech embeddings sit in a separate region. Cosine histograms: transcript tokens vs random tokens vs N(0, 1/d) differ only slightly.
- Literature: Ma et al. 2024 (SLAM-ASR, arXiv 2402.08846); Kumar et al. 2024 (arXiv 2411.03866) find the same with a frozen LM (nearest tokens are gibberish; LoRA aligns better).

## 3. Hidden-state alignment (layer by layer)
- Script `analysis/hidden_alignment.py`; output `results/local_full/seed_1/hidden_alignment/`. Runs the LM on projected speech and on the true transcript tokens, compares hidden states per layer: last-position and pooled retrieval, monotone (DTW-like) alignment advantage vs shuffled and mismatched-transcript controls, logit lens.
- Result: transcript content and order are present in the LM's hidden states, strongest at layers 20–24, already present at step 3000. Controls (length-only retrieval top-1 0.14, mismatched transcript) rule out length / position artefacts.
- Interpretation: speech is read through the LM's layers even though the input embeddings are not word-like.

## 4. Short sweep: warmup, clipping, norm penalty (1,500 steps each, lr 1e-4, warmup 50)
- Script `scripts/short_sweep.sh`, plots `analysis/plot_runs.py`; output `results/short_sweep/`.
- Drop step (first step where the 21-step median training loss is below 3.0), val loss at step 1500:

| run | drop | val loss @1500 |
|---|---|---|
| noclip | 1487 | 1.80 (still falling) |
| clip 1.0 | 613 | 0.365 |
| clip 1.0 + norm penalty 1e-3 | 437 | 0.397 |
| clip 1.0 + norm penalty 1e-2 | 450 | 0.42 |

- Reducing warmup 1000 → 50 moved the unclipped drop from ≈ 2128 to 1487. Clipping moved it to ≈ 600; the penalty a bit further. Train and val loss stay close (no overfitting at this length).
- Gradient norm rises 3–4× during the drop and does not ramp up before it (plateau flat). Projected-frame norm is 7–44 in all runs (text embeddings: ≈ 0.93); it does not change at the drop.
- The norm penalty at 1e-3 only brought the norm to ≈ 7 (still 7× text norm).

## 5. Learning-rate tests (no clipping, warmup 50)
- Output `results/lr_test/` (`compare.png`, `onsets.txt`).

| lr | drop below 3.0 | summed lr by then |
|---|---|---|
| 5e-5 | 2038 | 0.101 |
| 1e-4 | 1487 | 0.146 |
| 2e-4 | 1043 | 0.204 |

- Onset ∝ lr^-0.5 (onset × √lr = 0.0144 / 0.0149 / 0.0148). The "cumulative learning rate" idea (same summed lr at the drop) is refuted. Single seed.
- A sqrt law is what noise-driven escape from a plateau might give *(hypothesis)*.

## 6. Dense run around the drop (clip 1.0, lr 1e-4, warmup 50, checkpoints every 25 steps, 900 steps)
- Script `scripts/dense_run.sh`, `analysis/dense_analysis.py`, hidden alignment on steps 500–900; output `results/dense/clip1/` (`dense_analysis/`, `hidden_alignment/`). Drop at step 582 (613 in the earlier sweep run; reruns differ by ~5%).
- **Audio gap** = val loss with another utterance's audio minus loss with the right audio (0 = audio not used).
  - Gap: 0.05 @300, 0.14 @400, 0.42 @500, 1.2 @575, 3.2 @600, ≈ 4.5 from 700 on. Grows smoothly (roughly exponentially) for ~200 steps before the training loss drops.
  - Loss with no audio at all: 4.99. The plateau loss (≈ 3.5) is *below* that: on the plateau the projector acts as a content-independent soft prompt (swapped = right-audio loss). After the drop, wrong audio is as bad as no audio (4.8–5.1).
- Attention from transcript positions to speech positions: mean 0.08 → 0.20 between steps 450 and 625; strongest layer moves from 18 (before) to 16 (after). Attention mass is not causal evidence, and stays below the uniform share (~0.77).
- Hidden alignment: pooled retrieval is already ≈ 0.2 at layers 12–24 at step 500 (chance 0.01), reaches 1.0 at layer 20 by step 625; mismatched-transcript control stays at the shuffled level; last-position retrieval appears only at layer 28.
- Projected frames: norm 12 → 39 then flat; best cosine to any vocabulary embedding is 0.14 at every checkpoint (= chance for a max over 150k random vectors, ≈ init); between-utterance cosine 0.999 (huge shared component).

## 7. Law-finding runs (7 runs, clip 1.0, warmup 50, checkpoint every 50 steps, analysed then deleted)
- Script `scripts/law_runs.sh`, summary `analysis/law_summary.py`, plots `analysis/plot_gap_loss.py`; output `results/law/` (`summary.txt`, `summary.png`), `results/gap_loss.png` (8 panels: train loss, val loss, audio gap).
- Drop step and step where the gap reaches 0.3:

| run | drop | gap = 0.3 | drop ÷ gap-0.3 step |
|---|---|---|---|
| seed 1 (dense run) | 582 | 483 | 1.20 |
| seed 2 | 631 | 561 | 1.13 |
| seed 3 | 730 | 626 | 1.17 |
| lr 2e-4 | 473 | 407 | 1.16 |
| lr 5e-5 | 660 | 561 | 1.18 |
| init scale 0.5 | 870 | 806 | 1.08 |
| init scale 2 | none by 1000 | not reached | – |
| batch size 2 | 749 | 672 | 1.11 |

- Seed spread of the drop with identical settings is ≈ ±12%.
- **No clean law for the drop step in the knobs:** with clipping lr 5e-5 ≈ lr 1e-4 (660 vs ≈ 648), only 2e-4 clearly earlier; the unclipped lr^-0.5 relation does not carry over. Init scale is slow in *both* directions (0.5× → 870, 2× → no drop; its frame norm explodes to 116 by step 600 vs 39 baseline *(hypothesis: huge-norm frames are hard for the LM to read, untested)*). Batch size 2 is later in steps (749).
- **Robust relation:** once the audio gap reaches 0.3 the drop follows at 1.15 ± 0.04 × that step (65–105 steps later) in every run that reached it. This is a warning signal, not an independent predictor: the gap is itself a loss quantity.

## 8. Projector anatomy (what the two layers change across the drop)
- Script `analysis/projector_anatomy.py` on `results/dense/clip1` checkpoints, before = step 350, after = step 700; output `results/dense/clip1/projector_anatomy/` (`projector_anatomy.png`, `.json`).
- Output variance decomposition (val frames): share explained by frame position 0.30 (step 25) → 0.48 (step 350) → 0.26 (step 900); share left within the utterance 0.68 → 0.50 → 0.71; utterance-mean share stays ≈ 0.02. Effective output dimension (participation ratio) 47 → 11 (step 450) → 16; hidden layer 51 → 35 → 53. The reversal starts around steps 450–500, ~100 steps before the loss drop. Caveat: "within-utterance" variance may include non-word factors (loudness etc.).
- Straight-line weight interpolation 350 → 700: val loss 3.33, 3.28, 3.04, 1.47, 0.67, 0.57 at α = 0, 0.2, 0.4, 0.6, 0.8, 1. Smooth transition, no barrier; the 500 → 650 path looks the same.
- Layer transplants (val loss right audio / swapped): before 3.33 / 3.39; layer 1 from after only 2.89 / 3.42; layer 2 from after only 3.34 / 3.70; after 0.57 / 4.73. Both layers' changes are needed (co-adaptation).
- Rank of each layer's change (top-k singular directions applied to the after-model): layer 2: loss 2.89 (k=0), 1.03 (k=1), 0.65 (k=8) → effectively rank 1–8. Layer 1: 2.81 (k=8), 1.05 (k=32), 0.73 (k=64) → dozens of directions.

## 9. Direction probe (does the LM read layer 2's few directions?)
- Script `analysis/direction_probe.py` (before = 350, after = 700); output `results/dense/clip1/projector_anatomy/direction_probe.{json,png}`. Subspaces: top-k left singular vectors of the layer-2 weight change, top-k principal directions of the change in projected frames, random k-dim subspaces (3 draws). Before-model val loss 3.33, after-model 0.57.
- **Revert** k directions of the after-model's frames to the before-model's values: ΔW2 directions k = 1/2/4/8/16 → loss 1.79 / 2.35 / 2.95 / 3.10 / 3.21; PCA-of-change directions 1.84 / 2.22 / 2.57 / 3.57 / 3.48; random directions: 0.57–0.58 (no effect). So what the LM uses is concentrated in a handful of directions (reverting one direction already costs 1.2 nats).
- **Inject** k after-model directions into the before-model's frames: loss 3.2–3.8 (no improvement), swapped-audio loss rises with k (3.51 at k=1 → 4.25 at k=16, so a gap of ≈ 1.0 appears at k=16 even though the right-audio loss does not fall). Necessary but not sufficient on their own; the rest of the output (the soft-prompt component) and the layer-1 state matter.
- **Gradient share:** share of the loss gradient w.r.t. frames lying in the k directions: ΔW2 directions 0.014 / 0.032 / 0.080 at k = 4 / 8 / 16 (after-model) against 0.0039 / 0.0078 / 0.0156 isotropic, i.e. 3.5–5× enriched, and nearly as enriched in the before-model (0.012 / 0.025 / 0.055). So the LM was already moderately more sensitive to these directions before the drop (not an extreme concentration; k=1 is at the isotropic level).
- **Layer-0 gain:** the LM's first attention layer (q, k, v, with the input norm weight) amplifies the top singular vectors about as much as a random direction (0.82–1.24). The directions are not specially amplified at layer 0, so the read-out happens deeper or through other paths.
- Single run (seed 1).


## 10. Alignment analysis on the norm-penalty projectors (2026-11-01)
- Why: entry 4 showed the penalty shrinks the frame norm and brings the drop forward, but we had not checked whether it changes the alignment picture. User's hypothesis: the projector "should be text-like".
- Run: `analyze.py` on the final `projector.pt` of the short-sweep runs `clip1`, `clip1_pen1e-3`, `clip1_pen1e-2` (1,500 steps, seed 1), 64 val utterances, replacement WER on 30 utterances; output `results/short_sweep/<run>/analysis_norm/`. `word_recall@k` = fraction of transcript words with a token variant fully inside the set of the k nearest vocabulary tokens of the utterance's frames; the control uses another utterance's frames.
- Results:

| projector | mean frame norm (val) | cosine to nearest token | share of frames on the 10 most common nearest tokens | word recall@1 (control) | word recall@10 (control) | WER (30 utt.) |
|---|---|---|---|---|---|---|
| untrained | 11.5 | 0.139 | 0.13 | 0.001 (0.001) | – | – |
| clip only | 39.4 | 0.133 | 0.60 | 0.004 (0.003) | 0.015 (0.009) | 8.1% |
| clip + penalty 1e-3 | 9.8 | 0.156 | 0.14 | 0.135 (0.056) | 0.327 (0.144) | 8.7% |
| clip + penalty 1e-2 | 4.7 | 0.168 | 0.14 | 0.350 (0.126) | 0.605 (0.225) | 9.4% |

- **The norm penalty makes the nearest-token reading start to follow the spoken words**, with a dose-response over three points (0 / 1e-3 / 1e-2). Without the penalty there is no signal at all, and 60% of frames pile onto a few "hub" tokens (a huge shared component). So the earlier "no word alignment" result is not a fixed property of the projector; it depends on the norm regime.
- Replacing every frame by its nearest text embedding still destroys transcription in all three (WER 3.0 / 1.9 / 1.6, random replacement 1.5–2.0), so the model is still not working in text space: partial alignment, not a text-like projector.
- The penalty costs little: val loss 0.365 / 0.397 / 0.42 at step 1,500, WER 8.1% / 8.7% / 9.4% on 30 utterances (noise level).
- Caveats: one seed per setting, 1,500 steps, penalty strengths 1e-3 and 1e-2 only, frame norm still 5× the text norm at 1e-2. Mean-centring / cosine on the unpenalised run was analysed earlier (entry 2); the hub share suggests the shared component hides weak alignment there. Hypothesis: a stronger penalty gives more alignment *(untested)*.

## 11. Vocabulary-mixture projector (text-like by construction)
- Code: `projector_type=vocab` (`VocabMixtureProjector` in `utils.py`, selected in `main.py`). The MLP gives a query q, weights = softmax(q · E^T) over the 3,593 tokens of the training transcripts, output = weights · E (E = the LM's frozen word embeddings), so every output is a blend of real word embeddings.
- Run: clip 1.0, lr 1e-4, warmup 50, seed 1, 3,000 steps, val loss every 250 steps, output `results/vocab/clip1_vocab/` (`run.log`, `vocab_eval.json`), diagnostics from `vocab_eval.py` (entropy, arg-max token overlap with the transcript, decoded top-word sequence). Started 2026-11-01 11:30. At step 0: loss 5.60, frame norm 0.33.
- Compared with the free projector (entry 6).
- **Result (2026-11-01 12:30): it never leaves the plateau.** Val loss 4.71 (step 250), 4.50 (1,000), 4.39 (2,000), 4.334 at step 3,000 (best, still falling very slowly); the no-audio loss is 4.99 and the free projector (entry 6) was at 3.4–3.5 before its drop at ~600 and 0.365 at step 1,500. No drop in 3,000 steps; the frame norm stays at ≈ 0.9 (a blend of word embeddings). Final val WER 621% (long hallucinated outputs; no usable transcription).
- Diagnostics (`vocab_eval.py`, 64 val utterances): mean entropy of the per-frame word weights 1.23 nats (max possible 8.19), mean top weight 0.69, i.e. fairly peaked, but the arg-max tokens are unrelated to the utterance: transcript-token recall 0.17% (control 0.17%), arg-max token in own transcript 0.34% (other utterance 0.23%). Decoded arg-max sequences are junk ("BYSCI RIGHTS OTHERWISE ... HOWEVER WAR ... WASHINGTON"), the same for every utterance.
- Interpretation: forcing outputs onto the convex hull of word embeddings trains far worse here, and the weights collapse onto a few unrelated words that act like a fixed prompt. Consistent with the free, large-norm region doing necessary work (the soft-prompt stage in entry 6 needs vectors that a blend of word embeddings cannot form, *hypothesis*). Not conclusive: one seed, lr 1e-4 as for the free projector, 3,000 steps, vocabulary restricted to 3,593 tokens, no temperature / lr tuning; a slower escape later cannot be excluded.

## 12. Stronger norm penalties (2026-11-01, `scripts/short_sweep.sh`, clip 1.0, lr 1e-4, seed 1, 1,500 steps)
- Training only; `analyze.py` (word recall, frame norm) on these three is *pending*, GPU busy with entry 13.
- Penalty 0.1: plateau ~3.5, drop at step ~750–800 (val 3.31 → 1.42), val loss 0.537 at step 1,500.
- Penalty 1: no drop in 1,500 steps, val loss 3.48 at step 1,500 (still on the plateau, slowly falling).
- Penalty 10: no drop, best val loss 3.528 (step 1,200), 3.54 at step 1,500.
- So the drop is delayed by 0.1 and prevented (within 1,500 steps) by 1 and 10: a strong norm penalty removes the audio-using solution in this budget. One seed each, 1,500 steps; a later escape is not excluded.

## 13. Linear baseline and extra seeds (2026-11-01 15:33, `scripts/baseline_seeds.sh`)
- Code: `projector_type=linear` (`build_linear_projector`, one Linear 3840→1024; no hidden layer). Same settings as entry 4 (clip 1.0, lr 1e-4, warmup 50, 1,500 steps).
- Queue: linear seeds 1–3; MLP seeds 2, 3; MLP + penalty 0.1 seeds 2, 3. Outputs `results/seeds/<name>_s<seed>/run.log`. Plot/summary: `analysis/plot_seeds.py` → `results/seeds/seeds_drop.png` (seed 1 of MLP and MLP+penalty are the earlier `short_sweep/clip1` and `clip1_pen1e-1` runs).
- **Result (3 seeds each, finished 2026-11-01 ~23:00):** drop step (first 100-step eval with val loss < 2.5; resolution 100): linear 600/900/900 (mean 800 ± 173), MLP 700/700/800 (733 ± 58), MLP + penalty 0.1 800/800/800. Final val loss at step 1,500: linear 0.380 ± 0.008, MLP 0.375 ± 0.019, MLP + penalty 0.1 0.527 ± 0.013 (std, ddof=1).
- Reading: a single linear layer matches the MLP in final loss and also shows plateau-then-drop, so the hidden layer is not needed; the drop-step differences between groups are within the 100-step resolution and seed spread. The penalty 0.1 costs about 0.15 val loss, outside the seed spread. WER not computed for these runs.

## 14. Test WER and long penalty-1 run (queued 2026-11-02, overnight)
- `scripts/test_eval.sh`: bakes 500 test.clean utterances, then `whisper_wer.py` (Whisper-small with its own decoder, reference) and `eval_wer.py` (untrained MLP/linear projector, linear s1–3, MLP s1–3, penalty 0.1 s1–3, penalty 1, penalty 10, long 12,000-step run). Output `results/test_eval/` (`whisper_wer.json`, `wer_results.json`, per-utterance error counts for bootstrap CIs). Results *pending*.
- `scripts/pen1_long.sh`: penalty 1, seed 1, 4,000 steps (entry 12 saw no drop in 1,500), starts after the test-WER queue. Output `results/pen1_long/pen1_long`. Question: delayed or prevented? Results *pending*.
- `scripts/overnight2.sh` (queued after the above): (a) `analyze.py` (word recall, frame norm; `analysis_norm/`) on mlp s2–3, MLP+penalty 0.1 s2–3, linear s1–3 for error bars on the alignment question (`analyze.py` now also loads the linear projector); (b) vocab projector with logit scale 10 (`vocab_scale=10`; entry 11 had unscaled logits, near-uniform softmax at init), lr 1e-4 and 3e-4, 3,000 steps, `results/vocab/scale10_lr1e-4`, `scale10_lr3e-4`. Results *pending*.
- `scripts/overnight3.sh` (last in the chain): penalty 0.3 seed 1 and penalty 1 seed 2 (1,500 steps each, short-sweep settings), then `analyze.py` on both. Results *pending*.
- **Test WER so far** (500 utterances, first results at 23:40): Whisper-small own decoder 4.5%; untrained MLP 169%, untrained linear 117%; linear s1/s2/s3 20.6/20.8/19.4%; MLP s1 18.7%.
- **Error analysis** (`error_analysis.py model=<name>`, CPU only, writes `results/test_eval/error_analysis_<name>.txt/.png`). First look at mlp_s1 (18.7%): substitutions 13.6%, insertions 4.4%, deletions 0.8% per reference word; 12.6% of utterances perfect, median utterance WER 0.14, 3% above 0.5. Insertions are dominated by a few runaway outputs (repetition loops such as "FIVE FIVE FIVE…" until the 200-token cap; 0.4% of utterances have a hypothesis > 2× the reference length) - these two utterances alone have WER 4.0 and 2.1. Remaining errors are mostly acoustic/spelling confusions (names: HAWKEYE→HOCKEY, UNCAS→UNKAS; homophones PANES→PAINS; function words AND↔IN, THE↔A; inflection SUPPOSE→SUPPOSED), with word-boundary errors (RIGHTMINDED). No length trend except the 11 utterances with 60+ words (WER 0.44, includes a runaway). To be repeated for the best model (long12000) when its WER is in.

## 15. Overnight results (2026-11-02 08:40)
- **Correction to entry 12: penalty 1 does not prevent the drop, it delays it.** Seed 1 for 4,000 steps (`results/pen1_long/pen1_long`): plateau at 3.51 until step ~1,600, val 1.51 at step 2,000, 0.76 at 2,400, 0.484 at 4,000. Seed 2 (1,500 steps) is already dropping at the end (val 3.45 at 1,200, 1.26 at 1,500). Penalty 0.3 (seed 1): plateau 3.52 at 1,200, 1.57 at 1,500. So drop onset grows with the penalty: ≈ 700 (none) / 800 (0.1) / ≈ 1,450 (0.3) / ≈ 1,450–1,900 (1). Penalty 10 not rerun for longer (no drop in 1,500 steps, still possible later).
- **Word recall @1 (64 val utterances, other-utterance control in brackets)**, `analysis_norm`: plain MLP 0.004 / 0.000 / 0.001 (seeds 1–3; frame norm 39–44), linear 0.002 / 0.006 / 0.021 (norm 46), MLP + penalty 0.1: 0.329 (0.116) / 0.336 (0.114) / 0.318 (0.108) (norm 2.2–2.3), penalty 0.3: 0.191 (0.090), penalty 1 seed 2: 0.232 (0.096) (norm 1.56), penalty 1 seed 1 and 10 at 1,500 steps (not dropped): 0.006 / 0.001. So plain projectors are not text-like in any seed; the penalty-0.1 result replicates in all three seeds (recall about three times its control). The 0.3 and 1 runs were analysed at the end of 1,500 steps, shortly after (or at) their drop, so their recall is not comparable to the fully trained 0.1 runs.
- **Vocab mixture with logit scale 10** (`results/vocab/scale10_lr1e-4`, `scale10_lr3e-4`, 3,000 steps): still no learning: val loss 4.727 (step 3,000, lr 1e-4) and 4.789 (best, lr 3e-4), against no-audio loss 4.99. Mean softmax entropy 0.002 nats, i.e. one-hot, so with scale 10 the failure is a *collapse onto a single word* (the first attempt, scale 1, was a near-uniform blend); transcript-token recall 0.0 / 0.001. The convex-hull projector failed in two settings, but neither is a tuned run: no setting between the two scales (e.g. 3) or temperature annealing was tried.
- **Test WER (500 test.clean utterances; `wer_summary.py` → `results/test_eval/wer_summary.txt`, bootstrap over utterances):** Whisper-small own decoder 4.5% [4.0, 5.1]; untrained MLP projector 168.8%, untrained linear 117.0%; linear 20.3% ± 0.8 (seed std; per seed 20.6/20.8/19.4); MLP 18.8% ± 0.4 (18.7/19.2/18.4), CI [16.4, 21.5]; MLP + penalty 0.1 29.6% ± 7.9 (38.6/24.1/26.0); penalty 1 (1,500 steps, seed 1, not dropped) 207.6%; penalty 10 (not dropped) 117.9%; MLP 12,000 steps (single seed) 12.3% [10.4, 14.8]. The seed-pooled bootstrap intervals do not separate linear from MLP (20.3 vs 18.8, overlapping), but the penalty 0.1 gap does separate from the plain MLP.
- **Error analysis of the best model** (long12000, `error_analysis.py model=long12000` → `results/test_eval/error_analysis_long12000.txt/.png`): substitutions 9.2%, insertions 1.8%, deletions 1.3%; 26% of utterances perfect, median utterance WER 0.089, 1.4% above 0.5; no runaway repetitions (0%, against 0.4% for the 1,500-step model, where two looping outputs inflated WER). Most frequent substitutions: AND→IN (13), THE→A (9), names and spelling variants (UNCAS→UNCUS, DAIRY→DERRY, HONOR→HONOUR). Longer references (60+ words, n=11) have WER 0.36, 3× the rest.

---

## Open questions / next
- Which output directions are these (what do the top-1 and top-2 directions encode: position, energy, phonetic class)? Not tested.
- Find which LM layers read the directions (attention heads at layers 16/18, residual stream) and whether the pre-drop gradient enrichment grows as the drop approaches.
- More seeds for the init-scale runs (870 and the init-2 failure may be partly noise); init 2 with a norm penalty to test the large-norm explanation.
- Which controllable condition sets the *start* of the audio-gap rise (a predictor independent of the loss) is still unknown.
- Optional: Kumar-style tempo/noise perturbation tests, LoRA / alignment-loss / output-scale experiments, within-utterance token retrieval test.
- Commit the code (patience, lazy loading, local_full.sh, analyze.py additions, analysis/hidden_alignment.py, analysis/plot_runs.py, short_sweep.sh, analysis/dense_analysis.py, law_runs.sh, analysis/law_summary.py, analysis/plot_gap_loss.py, analysis/projector_anatomy.py, analysis/direction_probe.py, init_scale option) if wanted, without attribution lines.
