# BVD features vs raw impedance: leave-one-probe-out comparison

Notebook: `ML_Raw_data_CNN.ipynb` (BVD arm and raw arm run in the same notebook). Date: 2026-10-08.

## Setup (identical for both arms)

- Same 46 (probe, timestamp) rows: sample_0 (16), sample_5 (14), sample_10 (16).
- Leave-one-probe-out: train on 2 probes, score the third. Scored on **real held-out rows only** (15, 13, 15 rows).
- Same CNN class, AdamW + cosine LR, fixed epochs (no checkpoint chosen on the held-out probe), mean of 5 seeds.
- Same augmentation: PCHIP interpolation between measurements, applied to the training probes only.
- Only the input differs:
  - **BVD arm:** 4 baseline-relative features (`rel_R0`, `rel_R1`, `rel_L1`, `rel_C0`), 300 epochs.
  - **Raw arm:** the 50-350 kHz spectrum (log|Z| and phase relative to each probe's first spectrum, 30-point block means), 200 epochs. Same Z scaling as `bvd_utils`.

## Results (final epoch, mean of 5 seeds)

Held-out RMSE in grams:

| Model | sample_0 | sample_5 | sample_10 | Mean RMSE | Mean MAE | Mean R² |
|---|---|---|---|---|---|---|
| Train mean (baseline) | 13.58 | 15.01 | 15.56 | 14.71 | 12.95 | -0.03 |
| Raw impedance | 17.34 | 20.13 | 8.69 | 15.39 | 12.60 | -0.30 |
| BVD features | 3.13 | 4.97 | 3.13 | 3.74 | 2.86 | 0.93 |

Supporting numbers (mean over the 3 held-out probes):

| | Raw impedance | BVD features |
|---|---|---|
| Seed spread of RMSE (std over 5 seeds) | 1.71 g | 1.36 g |
| Train RMSE, real rows | 6.98 g | 3.07 g |
| Generalization gap (held-out minus train) | 8.41 g | 0.67 g |
| RMSE at the oracle best epoch (not a valid score, shows the best case) | 12.07 g | 4.13 g |
| Model inputs | 200 (100 bins x 2 channels) | 4 |
| Parameters | 3,409 | 3,473 |
| Probes where it beats the other arm | 0/3 | 3/3 |

Raw-arm sensitivity: 13 setups tried (absolute vs baseline-relative input, Re/Im, per-frequency scaling,
frequency coordinate channel, normalization, dropout, noise, depth, narrower bands), 2-seed ensembles.
Mean RMSE ranged from 13.4 g to 46.9 g. Only 3 of 13 were below the train-mean baseline (14.7 g),
and the best was 1.3 g below it (band 150-260 kHz, 13.4 g).

BVD without `rel_L1` (same settings): 33.8 g (R0, R1, C0) and 25.4 g (R0, R1, C0, C1).

## Conclusions we can draw

1. **In this setup BVD features generalize across probes and raw impedance does not.** BVD reaches 3.7 g
   (R² 0.93) and beats raw on all three held-out probes. Raw is no better than predicting the training mean
   (15.4 g vs 14.7 g, R² -0.30).
2. **Raw overfits to the training probes.** Its held-out error is 8.4 g above its training error, against 0.7 g
   for BVD. On sample_5 the raw model predicts about 37 g almost regardless of the true value.
3. **It is not a stopping artifact.** Even with the best epoch picked on the held-out probe, raw reaches 12.1 g
   against 4.1 g for BVD.
4. **It is not one unlucky raw setup.** None of the 13 raw variants approaches the BVD result.
5. **Plausible reason (not tested here):** 200 inputs against about 28 real training rows, and the raw spectrum
   contains other peaks (for example near 280 and 330 kHz, see `06_input_compression.png`) that the BVD fit
   ignores. The BVD fit compresses each measurement to a few resonance parameters.

## Limitations to state alongside the result

- **Only 3 probes and 13-16 held-out rows per probe.** This is a feasibility result, not a generalization
  guarantee. There is no independent test set; leave-one-probe-out is the evaluation.
- **BVD was tuned much more than raw.** The BVD settings and feature subset were chosen on these same folds
  (dozens of runs), raw got the 13 variants above. The size of the gap is probably inflated even if the
  ranking holds.
- **The BVD result rests on `rel_L1`, by design.** L1 is the acoustic-mass parameter. It is fixed at 0.020 H on
  each probe's first fit because all probes start with the same weight, and it can only decrease afterwards
  because the probe only loses mass. The values are tiny because only the relative change is used. `rel_C1` is
  left out because L1 and C1 overlap in information (resonance depends on L1·C1). Without `rel_L1` the other
  parameters do not predict mass loss across probes (25 to 34 g). Because L1 is constrained to decrease, a
  reader may ask whether it behaves like a time proxy; state openly that the result rests on this channel.
- **Mass loss is close to linear in elapsed time for all three probes** (the notebook's baseline cell prints the
  calendar-time reference), so a monotone drifting feature can score well for that reason alone.
- **Only one raw encoder was tried** (a CNN over the frequency axis). Other raw encoders were not explored.
- **Same 46 rows for both arms.** These are the rows with a usable BVD fit (R² on the real part above 0.1).

## Why there is no separate test set

- **The probe is the only valid unit to split on.** Readings from one probe are close to duplicates of each
  other, so a row-level split would put near-identical data in train and test and measure memorization.
- **There are only three probes.** A separate test probe leaves two for everything else. Tuning inside those two
  (train on one, validate on the other) means each model sees a single probe, too little to select anything
  meaningfully. One test probe would also give one noisy score instead of three.
- **Leave-one-probe-out is the standard answer when independent groups are few.** Each probe is scored by a
  model that never saw it, and the three scores use all the data. The held-out probe plays the test role:
  nothing in training reads it (fixed epochs, no early stopping or checkpoint choice, scaler, noise scale and
  augmentation fitted on the training probes only, scoring on real rows only).
- **The claim is a comparison, not an absolute error.** Both arms are evaluated under the same protocol, so the
  question "does BVD generalize better than raw" does not need an unbiased absolute RMSE.
- **What this does not cover.** The settings were chosen by looking at these folds, so absolute errors are
  best-case (validation-selected), and BVD was tuned more than raw. Say so, report per-probe results (BVD wins
  3/3), and name an independent probe as the next step.

## Suggested wording

> With leave-one-probe-out validation on three probes, a CNN trained on four BVD-derived features reached
> 3.7 g held-out RMSE (R² 0.93) and beat a CNN trained on the full impedance spectrum on all three probes
> (15.4 g, no better than the training mean), with a small train-to-held-out gap (0.7 g vs 8.4 g). Settings were
> chosen on the same folds, so the absolute errors are best-case estimates, and confirming generalization
> requires an independent probe.

## Files in this folder

| File | Content |
|---|---|
| `01_rmse_per_probe.png` | Held-out RMSE per probe with seed spread, plus train-mean baseline |
| `02_parity_bvd_vs_raw.png` | Predicted vs actual, BVD (top) and raw (bottom), real held-out points |
| `03_prediction_over_time.png` | Actual and predicted mass loss over the measurement period |
| `04_generalization_gap.png` | Train vs held-out RMSE for each arm |
| `05_raw_setups_tried.png` | Mean RMSE of the 13 raw variants against BVD and the baseline |
| `06_input_compression.png` | One spectrum next to its BVD fit and the 4 fitted numbers |
| `07_limitation_rel_L1.png` | BVD with and without `rel_L1` |
| `metrics_per_probe.csv`, `metrics_summary.csv` | All metrics in the tables above |
| `raw_sweep.csv`, `bvd_without_rel_L1.csv` | Raw sweep and BVD ablation results |
