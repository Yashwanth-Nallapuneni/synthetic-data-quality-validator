# Experiments: validating the validator

This folder holds two protocol versions. **v2** (current) is described first. **v1 (first run)** is archived unchanged in `results/v1/` and described at the end.

Reproduce everything with two commands from the repo root (with the dev environment active):

```bash
python -m experiments.benchmark                          # 40 seeds, several minutes on 8 cores, needs network once
python -m experiments.calibrate --update-readme docs/VALIDATION.md
```

`benchmark` downloads the three datasets from OpenML into `experiments/cache/` (git-ignored), runs every generator on every dataset and seed and writes the raw statistics to `results/runs.jsonl` plus the environment to `results/meta.json`. `calibrate` reads those, learns the thresholds, writes `app/calibrated_thresholds.json` and the evaluation files in `results/` and pastes the detection matrix into docs/VALIDATION.md between its `detection-matrix` markers. Only the date inside the calibrated file and the run times change when the commands are repeated on the same machine and package versions.

## Files

| File | Purpose |
|---|---|
| `generators.py` | Gaussian copula and the six failure injections (`fresh` rows for the four isolated ones) |
| `benchmark.py` | Splits, runs, records raw statistics (`stats`) and provisional-threshold warning codes per run |
| `calibrate.py` | Threshold learning, test-set evaluation, leave-one-dataset-out variant |
| `compare_sdmetrics.py` | Runs SDMetrics on the same tables (needs the `compare` extra), re-times our validator |
| `compare_report.py` | Calibrates SDMetrics thresholds with the same rule and writes `comparison.md` and `comparison.json` |
| `results/runs.jsonl` | One line per (dataset, seed, generator): statistics, runtime, errors |
| `results/meta.json` | Datasets (id, name, version, licence), seeds, split, package versions, total runtime |
| `results/detection_matrix.md` and `.json` | Hit rate per generator and warning code on the test seeds, intended and any-warning rates, per-dataset breakdown |
| `results/detection_matrix_details.md` | Per-dataset table and the leave-one-dataset-out matrix |
| `results/sdmetrics_runs.jsonl`, `sdmetrics_meta.json`, `sdmetrics_directions.json` | Raw SDMetrics scores, runtimes and errors per run, versions, and the hand-built direction checks |
| `results/comparison.md` and `.json` | The generated SDMetrics comparison |
| `results/summary.json` | Headline: failure modes detected, `heldout_real` false-alarm count, copula rate, leave-one-dataset-out headline |

## Design

- Datasets: Adult (OpenML id 1590, version 2), German credit (id 31, `credit-g` version 1) and CMC (id 23, version 1). Adult is subsampled to 6,000 rows per seed (stratified by label, seeded) so a full run takes minutes. All three list the licence as Public on OpenML
- Per (dataset, seed): a stratified 50 / 25 / 25 split into T (generator training set and `real_df`), H_gold (output of the `heldout_real` generator) and H_base (`holdout_df`, the privacy baseline). Every generator makes `len(H_gold)` rows, so row counts are equal across generators. `exact_copy`, `near_copy` and the copula use T. The other four injections start from H_gold (see Changes between v1 and v2)
- Seeds 0 to 39. Seeds 0 to 19 are the calibration set and seeds 20 to 39 are the test set. All headline numbers come from seeds 20 to 39. The seed controls the subsample, the split, the generator and the validator's own randomness
- Statistics recorded per run are exactly the numbers `app/checks.py` compares with thresholds (`extract_stats`), so thresholds can be recalibrated without re-running any metric. `tests/test_benchmark.py` checks that re-applying thresholds to recorded statistics gives the same warning codes as `generate_trust_report` on the same inputs
- Threshold rule and the intended-warning mapping are written in the `calibrate.py` module docstring and in docs/VALIDATION.md

## Failure injections

| Name | Construction |
|---|---|
| `exact_copy` | `len(H_gold)` rows of T sampled without replacement |
| `near_copy` | the same, plus Gaussian noise with std 1% of each numeric column's std; categoricals unchanged |
| `near_copy_5pct` | as `near_copy` with 5% (reported, not in the headline count) |
| `column_shuffle` | bootstrap of H_gold, then every column (label included) permuted independently |
| `mode_collapse` | 5 random rows of H_gold, sampled with replacement, plus 1% numeric jitter (scaled by the training column's std) |
| `category_oversample` | bootstrap of H_gold; the categorical feature with the most categories in T has 50% of its rows set to its rarest category in T |
| `label_shuffle` | bootstrap of H_gold with the label column permuted |

The Gaussian copula maps numeric columns through the empirical CDF to normal scores and categorical columns (label included) to jittered intervals by frequency, fits a multivariate normal to the scores and inverts the mapping. Integer-like columns are rounded back. Missing categoricals are their own category and missing numerics are re-created at the training missing rate.

## Comparison with SDMetrics

Reproduce with (about 80 minutes with `--jobs 3`, about 1.3 GB of memory per job; the benchmark results above must exist):

```bash
pip install -e ".[compare]"
python -m experiments.compare_sdmetrics --jobs 3
python -m experiments.compare_sdmetrics --worst-only --jobs 3   # added later, see "Worst-column variants"
python -m experiments.compare_report --update-readme docs/VALIDATION.md --update-full experiments/README.md
```

`compare_sdmetrics` rebuilds T, H_gold, H_base and the synthetic table of all 1,080 runs through `benchmark.split_three` and `benchmark.build_synth` (the only change to `benchmark.py` was extracting `build_synth` from `run_unit`). It also re-runs our validator on every run, times it and compares its statistics with `runs.jsonl`: 0 of 1,080 differed by more than 1e-9, so the comparison used the same inputs and no benchmark result changed. `compare_report` then calibrates and evaluates.

**SDMetrics 0.32.0 scores.** QualityReport (overall score, Column Shapes, Column Pair Trends; the `sdmetrics.reports.single_table` class, which prints a deprecation warning but accepts single-table metadata), LogisticDetection(T, S), DCROverfittingProtection(T as training data, S, H_base as validation data, so it gets the same holdout as our privacy check, no subsampling), NewRowSynthesis(T, S) and ML efficacy. Efficacy uses BinaryLogisticRegression for two classes and MulticlassDecisionTreeClassifier for CMC, trained on the synthetic table and on T and tested on H_base. The calibrated statistic is the ratio TSTR / TRTR (SDMetrics' default F1 or macro F1), so that datasets with different baseline scores can share one threshold, as our utility check does. Metadata comes from `detect_column_types` plus our label typing, so both tools see the same numeric and categorical columns.

**Directions verified on hand-built cases** (`results/sdmetrics_directions.json`, run before every comparison): every score is higher is better, so every check warns when the score is low. An exact copy of T scores 0.0 on DCR overfitting protection and 0.0 on NewRowSynthesis against about 0.5 and 1.0 for a fresh sample. LogisticDetection is 1 minus the mean over folds of (2 max(AUC, 0.5) - 1), so a shifted sample scores 0.0 and an undetectable one near 1.0. Column Shapes, Column Pair Trends and the overall score fall for shifted or column-permuted samples. The efficacy ratio falls from about 1.0 to 0.77 when the label is permuted.

**Protocol.** Identical to the main benchmark: SDMetrics thresholds come only from `heldout_real` runs of seeds 0 to 19 pooled over the three datasets (60 values per score), rule null extreme minus max(10%, 0.01) for these warn-below scores, then applied unchanged to seeds 20 to 39. Detected means the intended check fires in at least 90% of the 60 test runs. No threshold of either tool was changed after seeing test seeds.

**Mapping of checks to failure modes.**

| Failure mode | Ours (intended) | SDMetrics strict |
|---|---|---|
| `exact_copy`, `near_copy` | `privacy_memorisation` | DCR overfitting protection or NewRowSynthesis |
| `column_shuffle` | `dependence_shift` or `detectability_high_auc` | Column Pair Trends or LogisticDetection |
| `mode_collapse` | `diversity_mode_collapse` or `diversity_low_coverage` | none (SDMetrics has no diversity check) |
| `category_oversample` | any `categorical_*` code | Column Shapes |
| `label_shuffle` | `utility_degradation` | ML efficacy ratio |

The lenient view counts any of the six individual SDMetrics checks. The overall-score view counts only the QualityReport overall score. Any-warning counts for the original strict and lenient views use the six individual checks (they share them). The two "worst column/pair" views are described next.

**Worst-column variants (added after seeing the comparison results).** A fairness review pointed out that our marginal checks use the worst column (maximum KS, largest category shift) while the strict mapping used the averaged QualityReport properties, Column Shapes and Column Pair Trends, which dilute damage to a single column. This was changed after the first results were read, so it is a post-hoc change. It can only make SDMetrics look better, and it was made for that reason. `compare_sdmetrics --worst-only` re-runs only QualityReport (about 1 minute for all 1,080 runs; DCR and NewRowSynthesis were not re-run) on the same regenerated frames, with the same unseen-category handling (QualityReport sees the raw tables), and writes `results/sdmetrics_worst.jsonl`. It records the smallest per-column score of `get_details("Column Shapes")` (`shape_min`, plus `shape_min_numeric` over KSComplement columns and `shape_min_categorical` over TVComplement columns) and the smallest per-pair score of `get_details("Column Pair Trends")` (`pair_min`; pairs SDMetrics cannot score are NaN and skipped, as in its own average). The existing `sdmetrics_runs.jsonl` was not modified, and the re-computed averages match the stored ones in 1,080 of 1,080 runs. The four statistics are calibrated with exactly the protocol above (`heldout_real`, seeds 0 to 19, same rule, warn below). Two columns were added and the existing ones are unchanged: "SDMetrics strict (worst column/pair)" maps `category_oversample` to `shape_min` and `column_shuffle` to `pair_min` or LogisticDetection (memorisation, `label_shuffle` as before, `mode_collapse` still has no check), and "SDMetrics lenient (worst column/pair)" counts every individual check including the four new ones. Both worst views share the any-warning count, which includes all ten checks.

**Result.** The table below is generated by `compare_report` (the same table is in docs/VALIDATION.md).

<!-- sdmetrics-full:start -->

Test seeds 20 to 39 (60 runs per generator, 3 datasets). Thresholds for both tools come from `heldout_real` runs of seeds 0 to 19 with the same rule. A mode counts as detected when its intended check fires in at least 90% of test runs. Cells are the fraction of test runs in which the intended check fired. SDMetrics strict uses the analogous SDMetrics check and has none for `mode_collapse` (SDMetrics has no diversity check). SDMetrics lenient counts any individual SDMetrics check. The two worst column/pair columns were added after the first results were read, following a fairness review, and can only favour SDMetrics: they use the smallest per-column Column Shapes score and the smallest per-pair Column Pair Trends score instead of the averages (`category_oversample` is mapped to the worst column shape, `column_shuffle` to the worst pair or LogisticDetection; the lenient variant counts every individual check including these). The last column uses only the overall Quality score.

| Failure mode | Ours (intended warning) | SDMetrics strict | SDMetrics strict (worst column/pair) | SDMetrics lenient | SDMetrics lenient (worst column/pair) | SDMetrics overall score only |
|---|---|---|---|---|---|---|
| `exact_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `near_copy` | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 1.00 detected | 0.00 |
| `column_shuffle` | 1.00 detected | 0.55 | 0.40 | 0.78 | 0.78 | 0.00 |
| `mode_collapse` | 1.00 detected | 0.00 | 0.00 | 1.00 detected | 1.00 detected | 1.00 detected |
| `category_oversample` | 1.00 detected | 0.00 | 1.00 detected | 0.67 | 1.00 detected | 0.17 |
| `label_shuffle` | 0.92 detected | 0.38 | 0.38 | 0.43 | 0.48 | 0.00 |
| `near_copy_5pct` (not in count) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| Failure modes detected | 6 of 6 | 2 of 6 | 3 of 6 | 3 of 6 | 4 of 6 | 1 of 6 |
| `heldout_real` runs with any warning | 4 of 60 | 0 of 60 | 0 of 60 | 0 of 60 | 0 of 60 | 0 of 60 |
| `gaussian_copula` runs with any warning | 60 of 60 | 15 of 60 | 30 of 60 | 15 of 60 | 30 of 60 | 0 of 60 |

Per dataset (intended check fired, then any-warning counts):

| Failure mode | View | adult | cmc | german_credit |
|---|---|---|---|---|
| `exact_copy` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics strict (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics lenient (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `exact_copy` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `near_copy` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics strict (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics lenient (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `near_copy` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `column_shuffle` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `column_shuffle` | SDMetrics strict | 1.00 | 0.10 | 0.55 |
| `column_shuffle` | SDMetrics strict (worst column/pair) | 1.00 | 0.00 | 0.20 |
| `column_shuffle` | SDMetrics lenient | 1.00 | 0.55 | 0.80 |
| `column_shuffle` | SDMetrics lenient (worst column/pair) | 1.00 | 0.55 | 0.80 |
| `column_shuffle` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `mode_collapse` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `mode_collapse` | SDMetrics strict | 0.00 | 0.00 | 0.00 |
| `mode_collapse` | SDMetrics strict (worst column/pair) | 0.00 | 0.00 | 0.00 |
| `mode_collapse` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `mode_collapse` | SDMetrics lenient (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `mode_collapse` | SDMetrics overall score only | 1.00 | 1.00 | 1.00 |
| `category_oversample` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics strict | 0.00 | 0.00 | 0.00 |
| `category_oversample` | SDMetrics strict (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics lenient | 0.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics lenient (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `category_oversample` | SDMetrics overall score only | 0.00 | 0.50 | 0.00 |
| `label_shuffle` | Ours (intended warning) | 1.00 | 0.90 | 0.85 |
| `label_shuffle` | SDMetrics strict | 0.40 | 0.45 | 0.30 |
| `label_shuffle` | SDMetrics strict (worst column/pair) | 0.40 | 0.45 | 0.30 |
| `label_shuffle` | SDMetrics lenient | 0.45 | 0.45 | 0.40 |
| `label_shuffle` | SDMetrics lenient (worst column/pair) | 0.60 | 0.45 | 0.40 |
| `label_shuffle` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `near_copy_5pct` | Ours (intended warning) | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics strict | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics strict (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics lenient | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics lenient (worst column/pair) | 1.00 | 1.00 | 1.00 |
| `near_copy_5pct` | SDMetrics overall score only | 0.00 | 0.00 | 0.00 |
| `heldout_real` any warning | Ours (intended warning) | 0 of 20 | 2 of 20 | 2 of 20 |
| `heldout_real` any warning | SDMetrics strict | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics strict (worst column/pair) | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics lenient | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics lenient (worst column/pair) | 0 of 20 | 0 of 20 | 0 of 20 |
| `heldout_real` any warning | SDMetrics overall score only | 0 of 20 | 0 of 20 | 0 of 20 |
| `gaussian_copula` any warning | Ours (intended warning) | 20 of 20 | 20 of 20 | 20 of 20 |
| `gaussian_copula` any warning | SDMetrics strict | 6 of 20 | 6 of 20 | 3 of 20 |
| `gaussian_copula` any warning | SDMetrics strict (worst column/pair) | 20 of 20 | 6 of 20 | 4 of 20 |
| `gaussian_copula` any warning | SDMetrics lenient | 6 of 20 | 6 of 20 | 3 of 20 |
| `gaussian_copula` any warning | SDMetrics lenient (worst column/pair) | 20 of 20 | 6 of 20 | 4 of 20 |
| `gaussian_copula` any warning | SDMetrics overall score only | 0 of 20 | 0 of 20 | 0 of 20 |

SDMetrics thresholds and false alarms of each score on the `heldout_real` test runs:

| SDMetrics score | Warns when | Calibrated threshold | Null extreme | Null values | `heldout_real` test false alarms |
|---|---|---|---|---|---|
| `column_shapes` | below | 0.8543 | 0.9492 | 60 | 0 of 60 |
| `column_pair_trends` | below | 0.7660 | 0.8511 | 60 | 0 of 60 |
| `logistic_detection` | below | 0.7521 | 0.8357 | 60 | 0 of 60 |
| `dcr_overfitting` | below | 0.4536 | 0.5040 | 60 | 0 of 60 |
| `new_row_synthesis` | below | 0.8584 | 0.9538 | 60 | 0 of 60 |
| `efficacy_ratio` | below | 0.7298 | 0.8109 | 60 | 0 of 60 |
| `shape_min` | below | 0.7686 | 0.8540 | 60 | 0 of 60 |
| `shape_min_numeric` | below | 0.7884 | 0.8760 | 60 | 0 of 60 |
| `shape_min_categorical` | below | 0.7686 | 0.8540 | 60 | 0 of 60 |
| `pair_min` | below | 0.4986 | 0.5540 | 60 | 0 of 60 |
| `quality_score` | below | 0.8132 | 0.9035 | 60 | 0 of 60 |

Runtime (all runs; both tools timed inside the same parallel job, one thread each):

| Tool | Mean seconds per run |
|---|---|
| Ours (`generate_trust_report`, validator only) | 1.91 |
| SDMetrics (all metrics below) | 11.80 |
| SDMetrics `quality_score` | 0.21 |
| SDMetrics `logistic_detection` | 0.09 |
| SDMetrics `dcr_overfitting` | 8.27 |
| SDMetrics `new_row_synthesis` | 3.14 |
| SDMetrics `efficacy_ratio` | 0.10 |

SDMetrics errors: none.
Runs where our re-run did not reproduce the stored statistics: 0.
Worst-column pass: 1080 runs, re-computed averaged scores differing from the stored ones: 0.

<!-- sdmetrics-full:end -->

**Reading the result.**
- Exact and near copies are caught by both tools in every run. This is the one failure mode where SDMetrics matches this tool
- This tool detects 6 of 6 modes. SDMetrics detects 2 of 6 (strict), 3 of 6 (lenient), 3 of 6 (strict, worst column/pair), 4 of 6 (lenient, worst column/pair) and 1 of 6 (overall score alone). The lenient and overall-score views only catch `mode_collapse` because the collapsed sample is so far from the real data that fidelity scores drop; that is a coincidence of the check, not a diversity check
- `category_oversample`: the first result (Column Shapes missed it in every run) was largely an averaging effect, as the reviewer suspected. Column Shapes averages over all columns (15 on Adult), so changing one column dilutes to a small drop that the pooled threshold absorbs. With the worst column (`shape_min`) SDMetrics detects it in every test run (1.00), and the mode is no longer a difference between the tools
- `column_shuffle`: the worst pair did not help. `pair_min` (threshold 0.4986) is a much noisier statistic than the average, so the strict worst variant, which replaces Column Pair Trends by `pair_min`, catches 40% of runs, fewer than the 55% of the averaged strict view. The worst-pair lenient view stays at 78%. The strict worst view is therefore not a superset of the strict view; the lenient worst view, which does include every check, reaches 78% (the same as before)
- `label_shuffle` is caught by SDMetrics in 38% of runs (strict) and 48% (lenient, worst column/pair). LogisticDetection is linear and does not see broken dependence, and the efficacy ratio threshold is pooled over datasets. Nothing in the worst-column change touches this mode
- `mode_collapse` still has no SDMetrics check in the strict views
- `heldout_real`: this tool raised a warning in 4 of 60 test runs and SDMetrics in 0 of 60 under every view, including the worst-column ones (the new thresholds absorb all null values). SDMetrics' thresholds are wide because the margin (10% of the extreme) applies to scores near 0.9 to 0.95, so its better false-alarm count partly reflects lower sensitivity. I did not search for a threshold that equalises the two
- `gaussian_copula`: warnings in 60 of 60 runs for this tool, 15 of 60 for SDMetrics (original views) and 30 of 60 with the worst-column checks. The copula attenuates dependence, so a warning is defensible, but this is not a false alarm count for either tool
- Runtime: SDMetrics is about 6 times slower per run here, almost all of it DCR overfitting protection (8.3 s per run on average, 15 s on Adult) and NewRowSynthesis (3.1 s). Both tools were timed in the same parallel job, one thread each. Our figure is `generate_trust_report` only

**Honest differentiation.** Detection of memorisation is a tie, and after the worst-column change `category_oversample` is a tie too (and `mode_collapse` is a tie only under the lenient view, by coincidence). On `column_shuffle` and `label_shuffle`, and on `mode_collapse` with a strict mapping, this benchmark still favours this tool, but the benchmark was designed around this tool's checks and the injections are strong, so it does not show that SDMetrics is a bad library. The differences that do not depend on the benchmark: every check has its own threshold learned from real held-out data with a stated false-alarm target (SDMetrics ships scores without thresholds, so the threshold is the user's problem), warnings say which column or model and what value, the privacy check takes a holdout and uses a tie-robust rank statistic, and there is an API, a CLI and a UI.

**Limitations.**
- The SDMetrics threshold choice follows our rule and pools the datasets. Per-dataset thresholds or a different margin could raise SDMetrics' sensitivity, and I did not try them
- The mapping of SDMetrics checks to failure modes and the efficacy ratio are my choices. A different efficacy model or a raw TSTR score would change the `label_shuffle` row
- LogisticDetection and the efficacy metrics error out when one frame holds a categorical value the other lacks (this happens on genuine held-out real data). For those two calls only, unseen values were replaced by the reference frame's most common value (`mask_unseen_categories`). With that in place there were 0 metric errors in 1,080 runs. In an earlier smoke test without the replacement these errors appeared on `heldout_real` runs
- Three datasets, a weak copula as the only "good but imperfect" generator and injections chosen by me, as for the main benchmark
- ML efficacy is classification only (no regression dataset)

**Log of changes during the comparison.** The first full launch used 8 parallel jobs and was stopped after about 35 minutes because 8 jobs of 1.3 GB each exceeded the 8 GB of memory and the machine thrashed; it produced no results and was restarted with 3 jobs. The first version of the unseen-category handling replaced values by missing; it failed on frames where only one side had missing values, so it was changed to the mode replacement before any full run. The direction check initially required a 0.3 drop for the efficacy ratio on a toy case, which was too strict for F1 (the drop is 0.23), so it was relaxed to 0.15; this affected only the self-test. No bug in the comparison code was found after looking at the test-seed results.

## Changes between v1 and v2

v1 (seeds 0 to 19, even seeds calibration and odd seeds test) exposed a design flaw and two metric weaknesses. These changes were motivated by looking at the v1 test-half results, so the v1 test seeds were seen and are now part of the calibration set. To keep the evidence clean, v2 uses seeds 0 to 19 for calibration only and reports every headline number from new seeds 20 to 39, which nobody had looked at. All choices below were fixed before seeds 20 to 39 were run and are not to be re-tuned after seeing v2 results. Anything wrong found after the v2 run is documented, not fixed.

1. **Isolated failure modes.** In v1, `column_shuffle`, `mode_collapse`, `category_oversample` and `label_shuffle` bootstrapped from T, the rows the privacy check compares against, so `privacy_memorisation` fired on all of them for reasons unrelated to the injected failure and the any-warning rate was inflated. In v2 these four start from H_gold (real rows not in T). Only `exact_copy` and `near_copy` use T. The row count is still `len(H_gold)`
2. **Tie-robust privacy statistic.** The v1 privacy rule counts synthetic rows below the 5th percentile of holdout DCR. On data with many duplicate rows (CMC) that percentile is exactly 0, so a noisy copy is never counted and `near_copy` was missed on CMC. v2 adds `dcr_closeness_auc` = P(synthetic DCR < holdout DCR) + 0.5 P(equal), computed from ranks. It is 0.5 without memorisation. `privacy_memorisation` now fires if the AUC, the share or the exact-match excess exceeds its calibrated threshold. The share and exact-match statistics are still reported
3. **Balanced accuracy for utility.** Plain accuracy hides label damage on imbalanced labels (a majority-class predictor still scores about 0.76 on Adult). For classification `utility_degradation` now uses the TSTR / TRTR balanced-accuracy ratio, whose chance level is 1/k whatever the class balance. Accuracy and F1 are still reported. The threshold field was renamed `utility_balanced_accuracy_ratio`. Regression still uses the R2 drop
4. **Calibration rule unchanged.** Null extreme over the `heldout_real` calibration runs plus a margin of max(10%, 0.01). Calibration now uses 20 seeds per dataset (60 null runs) instead of 10 (30). The empirical false-alarm rate of each check on the 60 test null runs is reported
5. **Leave-one-dataset-out variant kept.** Its thresholds now use only calibration-seed null runs of the other datasets
6. **Layout.** The v1 results are archived in `results/v1/` unchanged, together with the v1 calibrated thresholds

## Changes made inside v1 (after its first run)

- `privacy_medium_share` was initially left at the provisional value, which ended up above the calibrated `privacy_high_share`. It is now set to the largest null value of the privacy statistic with no margin. It only changes the displayed risk level (MEDIUM versus LOW), never a warning, so no result in the detection matrix changed
- The shape of the generated markdown was fixed (the leave-one-dataset-out sentence had landed above its table). No numbers changed

Injection strengths, thresholds rule, seeds and intended-warning mapping were not changed after the results were seen. The weaknesses in the results (near copies on CMC, shuffled labels) are reported as they are.

## v1 (first run)

Archived unchanged in `results/v1/` (runs, matrix, summary and the v1 calibrated thresholds). Seeds 0 to 19, calibration on even seeds and test on odd seeds. The text below is the README section as it stood after that run. Its numbers are v1 numbers and are superseded by v2 above.

**Method**
- Datasets, from OpenML with pinned ids: Adult (id 1590, 48,842 rows, subsampled to 6,000 rows per seed for runtime), German credit (id 31, 1,000 rows) and CMC (id 23, 1,473 rows, 3 classes). All three list the licence as Public on OpenML
- 20 seeds per dataset (0 to 19). Per seed the rows are split at random, stratified by label, into the generator training set T (50%), a held-out gold set (25%) and a held-out baseline set (25%). The baseline set is the `holdout_df` for the privacy check. Every generator produces as many rows as the gold set, from T only
- Good generators: `heldout_real` (the gold set itself, the null) and `gaussian_copula` (written from scratch in `experiments/generators.py`; categorical columns use interval jitter, so it attenuates dependence)
- Failure injections, each from T: `exact_copy` (rows without replacement), `near_copy` (plus Gaussian noise of 1% of each numeric column's std, with `near_copy_5pct` as a stronger variant that is not counted in the headline), `column_shuffle` (every column permuted independently), `mode_collapse` (a bootstrap of 5 random rows plus 1% numeric jitter), `category_oversample` (50% of rows of the categorical feature with the most categories set to its rarest category) and `label_shuffle` (bootstrap with the label column permuted)
- Calibration: even seeds are the calibration half and odd seeds the test half. Each threshold is set from the `heldout_real` calibration runs only: for a check that warns when a statistic is high, threshold = max over those runs + max(10% of the max, 0.01), and for a check that warns when it is low, threshold = min over those runs - max(10% of the min, 0.01). The broken generators and the copula play no part in choosing thresholds. With n pooled null values a fresh null value falls beyond the observed maximum with probability 1 / (n + 1) per check, which is about 3% for n = 30, before the margin. Several checks run on each report, so the any-warning false-alarm rate is higher than any one check's
- Evaluation: the thresholds are applied to the recorded statistics of the test half. A hit means the intended warning fired. Intended warnings: `exact_copy` and `near_copy` need `privacy_memorisation`; `column_shuffle` needs `dependence_shift` or `detectability_high_auc`; `mode_collapse` needs `diversity_mode_collapse` or `diversity_low_coverage`; `category_oversample` needs any `categorical_*` code; `label_shuffle` needs `utility_degradation`. A failure mode counts as detected when its intended warning fires in at least 90% of test runs

**Calibrated thresholds** (from `app/calibrated_thresholds.json`, which also records the rule, seeds and date)

| Threshold | Provisional | Calibrated | Null values pooled |
|---|---|---|---|
| `privacy_high_share` | 0.15 | 0.099674 | 30 |
| `privacy_medium_share` | 0.1 | 0.089674 | derived (display only) |
| `privacy_exact_match_margin` | 0.05 | 0.023771 | 30 |
| `variance_ratio_low` | 0.5 | 0.769212 | 30 |
| `variance_ratio_high` | 1.5 | 1.385892 | 30 |
| `coverage_min` | 0.5 | 0.7362 | 30 |
| `numeric_max_ks` | 0.2 | 0.1364 | 30 |
| `utility_accuracy_ratio` | 0.8 | 0.800847 | 60 |
| `utility_r2_drop` | 0.2 | 0.2 | 0 (no regression dataset, provisional value kept) |
| `utility_ece_increase` | 0.1 | 0.105622 | 60 |
| `categorical_mean_l1` | 0.3 | 0.126585 | 30 |
| `categorical_mean_js` | 0.2 | 0.061616 | 30 |
| `categorical_worst_shift` | 0.25 | 0.11 | 30 |
| `dependence_mean_abs_diff` | 0.1 | 0.074027 | 30 |
| `c2st_auc` | 0.7 | 0.606936 | 30 |

**Results on the test half**


Test half: odd seeds 1 to 19 on 3 datasets (30 runs per generator). Cells are the fraction of runs in which the warning fired. Columns for codes that never fired are omitted.

| generator | n | privacy_memorisation | diversity_mode_collapse | diversity_excess_variance | diversity_low_coverage | numeric_marginal_shift | categorical_mean_l1 | categorical_mean_js | categorical_worst_shift | dependence_shift | detectability_high_auc | utility_degradation | utility_calibration_degradation | utility_synthetic_single_class | intended warning fired | any warning fired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `heldout_real` | 30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 |
| `gaussian_copula` | 30 | 0.00 | 0.00 | 0.00 | 0.57 | 0.00 | 0.00 | 0.00 | 0.00 | 0.97 | 0.70 | 0.10 | 0.00 | 0.00 | - | 1.00 |
| `exact_copy` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `near_copy` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `near_copy_5pct` | 30 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| `column_shuffle` | 30 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.43 | 0.13 | 0.00 | 1.00 | 1.00 |
| `mode_collapse` | 30 | 0.67 | 0.67 | 0.10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.47 | 1.00 | 0.10 | 1.00 | 1.00 |
| `category_oversample` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.90 | 0.90 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| `label_shuffle` | 30 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.17 | 0.00 | 0.40 | 1.00 |

False-alarm rate on `heldout_real` (any warning): 0 of 30 test runs (0.000).

Warnings on `gaussian_copula` (any warning): 30 of 30 test runs (1.000). A copula legitimately differs from real data, so these are not strictly false alarms.

Failure modes whose intended warning fires in at least 90% of test runs: 4 of 6.


Per-dataset rates and a leave-one-dataset-out variant are in `results/v1/detection_matrix_details.md`.

**Reading the results**
- Exact copies, independently shuffled columns, mode collapse and a heavily oversampled category are caught in every test run on all three datasets
- Near copies are caught on Adult and German credit but never on CMC. The privacy check flags rows that sit closer to training rows than the 5th percentile of the holdout distance. In CMC many rows are duplicated (for seed 1, 8.4% of holdout rows have an exact twin in the training set), so that percentile is exactly 0 and a noisy copy, whose distance is small but not 0, is never closer. This is a weakness of the check on data with many duplicate rows
- Shuffled labels are caught by `utility_degradation` in only part of the runs (see the table), mostly on CMC. The accuracy ratio of a label-shuffled bootstrap is often not far enough below the real-data ratio to cross the calibrated threshold. The other warnings that fire on those runs are a side effect of the injection, which copies feature rows from T, and do not show the label damage
- `heldout_real` produced no warnings in the test half. The copula triggered warnings in every test run, mostly `dependence_shift` and `detectability_high_auc`. That is expected for a weak copula and says the thresholds are strict enough to separate it from real data, not that the validator misfires
- The leave-one-dataset-out thresholds are less safe: they raised a warning on some `heldout_real` test runs, so thresholds learned from other datasets do not transfer perfectly

**Limitations of this evidence**
- Only three datasets, all small or subsampled, and no regression dataset (so `utility_r2_drop` stays at its provisional value)
- The failure injections are simple and their strength was fixed before any results were seen. Detection rates depend on that strength: a milder oversample or a smaller shuffled fraction would be harder to catch
- The copula is a weak generator. No modern deep generator (CTGAN, diffusion) was tested, so the false-alarm rate on a good-but-imperfect generator is not measured
- The false-alarm rate on `heldout_real` has a small denominator (30 test runs), so a rate of 0 does not mean the true rate is 0
- The thresholds are tuned to the dataset sizes used here (a few hundred to a few thousand rows). Other sizes may need recalibration with `python -m experiments.calibrate`
