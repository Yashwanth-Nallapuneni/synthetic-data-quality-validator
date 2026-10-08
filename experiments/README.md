# Experiments: validating the validator

Reproduce everything with two commands from the repo root (with the dev environment active):

```bash
python -m experiments.benchmark                          # about 3 minutes on 8 cores, needs network once
python -m experiments.calibrate --update-readme README.md
```

`benchmark` downloads the three datasets from OpenML into `experiments/cache/` (git-ignored), runs every generator on every dataset and seed and writes the raw statistics to `results/runs.jsonl` plus the environment to `results/meta.json`. `calibrate` reads those, learns the thresholds, writes `app/calibrated_thresholds.json` and the evaluation files in `results/`, and pastes the detection matrix into the main README between its `detection-matrix` markers. Only the date inside the calibrated file and the run times change when the commands are repeated on the same machine and package versions.

## Files

| File | Purpose |
|---|---|
| `generators.py` | Gaussian copula and the six failure injections |
| `benchmark.py` | Splits, runs, records raw statistics (`stats`) and provisional-threshold warning codes per run |
| `calibrate.py` | Threshold learning, test-half evaluation, leave-one-dataset-out variant |
| `results/runs.jsonl` | One line per (dataset, seed, generator): statistics, runtime, errors |
| `results/meta.json` | Datasets (id, name, version, licence), seeds, split, package versions, total runtime |
| `results/detection_matrix.md` and `.json` | Hit rate per generator and warning code on the test half, intended and any-warning rates, per-dataset breakdown |
| `results/detection_matrix_details.md` | Per-dataset table and the leave-one-dataset-out matrix |
| `results/summary.json` | Headline: failure modes detected, `heldout_real` false-alarm count, copula rate, leave-one-dataset-out headline |

## Design

- Datasets: Adult (OpenML id 1590, version 2), German credit (id 31, `credit-g` version 1) and CMC (id 23, version 1). Adult is subsampled to 6,000 rows per seed (stratified by label, seeded) so a full run takes minutes. All three list the licence as Public on OpenML
- Per (dataset, seed): a stratified 50 / 25 / 25 split into T (generator training set and `real_df`), H_gold (output of the `heldout_real` generator) and H_base (`holdout_df`, the privacy baseline). Every generator makes `len(H_gold)` rows from T, so row counts are equal across generators
- Seeds 0 to 19. The seed controls the subsample, the split, the generator and the validator's own randomness
- Statistics recorded per run are exactly the numbers `app/checks.py` compares with thresholds (`extract_stats`), so thresholds can be recalibrated without re-running any metric. `tests/test_benchmark.py` checks that re-applying thresholds to recorded statistics gives the same warning codes as `generate_trust_report` on the same inputs
- Threshold rule and the intended-warning mapping are written in the `calibrate.py` module docstring and in the main README

## Failure injections

| Name | Construction |
|---|---|
| `exact_copy` | `len(H_gold)` rows of T sampled without replacement |
| `near_copy` | the same, plus Gaussian noise with std 1% of each numeric column's std; categoricals unchanged |
| `near_copy_5pct` | as `near_copy` with 5% (reported, not in the headline count) |
| `column_shuffle` | bootstrap of T, then every column (label included) permuted independently |
| `mode_collapse` | 5 random rows of T, sampled with replacement, plus 1% numeric jitter |
| `category_oversample` | bootstrap of T; the categorical feature with the most categories in T has 50% of its rows set to its rarest category in T |
| `label_shuffle` | bootstrap of T with the label column permuted |

The Gaussian copula maps numeric columns through the empirical CDF to normal scores and categorical columns (label included) to jittered intervals by frequency, fits a multivariate normal to the scores and inverts the mapping. Integer-like columns are rounded back. Missing categoricals are their own category and missing numerics are re-created at the training missing rate.

## Changes made after the first run

- `privacy_medium_share` was initially left at the provisional value, which ended up above the calibrated `privacy_high_share`. It is now set to the largest null value of the privacy statistic with no margin. It only changes the displayed risk level (MEDIUM versus LOW), never a warning, so no result in the detection matrix changed
- The shape of the generated markdown was fixed (the leave-one-dataset-out sentence had landed above its table). No numbers changed

Injection strengths, thresholds rule, seeds and intended-warning mapping were not changed after the results were seen. The weaknesses in the results (near copies on CMC, shuffled labels) are reported as they are.
