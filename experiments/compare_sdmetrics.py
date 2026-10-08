"""Run SDMetrics on the exact tables the benchmark gives our validator.

    pip install -e ".[compare]"
    python -m experiments.compare_sdmetrics [--seeds 40] [--jobs 8] [--out experiments/results]
    python -m experiments.compare_sdmetrics --worst-only   # added later, see below
    python -m experiments.compare_report

For every (dataset, seed, generator) of experiments.benchmark the same T, H_gold, H_base and
synthetic table are rebuilt (benchmark.split_three and benchmark.build_synth) and scored with
SDMetrics 0.32.0:

- QualityReport: overall score, Column Shapes, Column Pair Trends
- LogisticDetection(T, S)
- DCROverfittingProtection(T as training data, S, H_base as validation data), the same
  holdout our privacy check gets
- NewRowSynthesis(T, S)
- ML efficacy for the label, trained on S and tested on H_base (TSTR) and trained on T and
  tested on H_base (TRTR): BinaryLogisticRegression for two classes, MulticlassDecisionTree
  Classifier for more. The calibrated score is the ratio TSTR / TRTR so that datasets with
  different baseline scores can share one threshold, as our utility check does

LogisticDetection and the efficacy metrics raise an error when one frame holds a category the
other lacks, so categories unseen in the reference frame are replaced by its most common value
for those calls only (mask_unseen_categories); QualityReport, DCR and NewRowSynthesis see the
raw tables.

Both tools see the same column types: detect_column_types on the real features plus the label
typing of app.metrics.utility.detect_task. Our own validator is re-run on each run too, timed,
and its recorded statistics are compared with runs.jsonl so the comparison provably used the
same inputs. Each metric is timed and any exception is recorded in `errors` (the value is then
null); no run is dropped. All SDMetrics scores are higher-is-better (see verify_directions).

Worst-column pass (--worst-only). Added AFTER the comparison was first read, because a fairness
review pointed out that our marginal checks use the worst column while the averaged Column
Shapes and Column Pair Trends dilute single-column damage. It re-runs only QualityReport on the
same regenerated frames and records the smallest per-column score (all, numeric KSComplement,
categorical TVComplement) and the smallest per-pair score, in sdmetrics_worst.jsonl. It can
only help the baseline and leaves every existing result untouched.
"""

import argparse
import json
import random
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.checks import extract_stats
from app.metrics.utility import detect_task
from app.report import generate_trust_report
from app.thresholds import Thresholds
from app.utils.column_types import detect_column_types
from experiments.benchmark import (
    DATASETS,
    SEEDS,
    Dataset,
    build_synth,
    load_dataset,
    split_three,
    versions,
)
from experiments.generators import GENERATORS, HELDOUT

RESULTS = Path(__file__).resolve().parent / "results"
SDMETRICS_VERSION = "0.32.0"
STATS_TOLERANCE = 1e-9
WORST_FILE = "sdmetrics_worst.jsonl"


def build_metadata(real: pd.DataFrame, label: str) -> dict[str, Any]:
    """SDMetrics single-table metadata from our column typing.

    Feature columns use detect_column_types (numeric, categorical or empty) on the real
    features. The label is categorical for classification and numerical for regression
    (app.metrics.utility.detect_task). Empty columns are left out, as our validator ignores them.
    """
    types = detect_column_types(real.drop(columns=[label]))
    columns: dict[str, dict[str, str]] = {}
    for col in real.columns:
        if col == label:
            labels = real[label].dropna()
            numeric = not labels.empty and detect_task(labels) == "regression"
            columns[col] = {"sdtype": "numerical" if numeric else "categorical"}
        elif col in types["numeric"]:
            columns[col] = {"sdtype": "numerical"}
        elif col in types["categorical"]:
            columns[col] = {"sdtype": "categorical"}
    return {"columns": columns}


def prepare_frame(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Keep the metadata columns; numerical -> float, categorical -> string (missing stays NaN)."""
    out = pd.DataFrame(index=range(len(df)))
    for col, spec in metadata["columns"].items():
        s = df[col].reset_index(drop=True)
        if spec["sdtype"] == "numerical":
            out[col] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            out[col] = s.astype(object).map(lambda v: v if pd.isna(v) else str(v))
    return out


def mask_unseen_categories(
    df: pd.DataFrame, reference: pd.DataFrame, metadata: dict[str, Any], skip: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Replace categorical values (missing included) absent from `reference` by its mode.

    SDMetrics fits its one-hot encoder on one frame and raises "Found unknown categories" when
    the other frame has a value the first lacks. That happens on genuinely unseen real data (a
    rare category in H_gold but not in T), so LogisticDetection and the efficacy metrics would
    fail on heldout_real runs. Replacing the unseen values (like a mode imputer) keeps them
    computable. The `skip` column (the efficacy target) is never changed. Returns the new frame
    and the number of replaced cells.
    """
    out = df.copy()
    n = 0
    for col, spec in metadata["columns"].items():
        if spec["sdtype"] != "categorical" or col == skip:
            continue
        seen = reference[col].dropna()
        if seen.empty:
            continue
        unseen = ~out[col].isin(seen.unique())
        if reference[col].isna().any():
            unseen &= out[col].notna()  # missing values are only "unseen" if the reference has none
        n += int(unseen.sum())
        out.loc[unseen, col] = seen.mode().iloc[0]
    return out, n


NUMERIC_SHAPE_METRIC = "KSComplement"
CATEGORICAL_SHAPE_METRIC = "TVComplement"
WORST_KEYS = ("shape_min", "shape_min_numeric", "shape_min_categorical", "pair_min")


def _min_score(scores: pd.Series) -> float | None:
    """Smallest finite score, or None when there is none (unscorable pairs are NaN)."""
    vals = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.min()) if vals.size else None


def worst_scores(shapes: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, float | None]:
    """Worst single column and worst single pair from the QualityReport detail tables.

    `shapes` is get_details("Column Shapes") (columns Column, Metric, Score) and `pairs` is
    get_details("Column Pair Trends"). The averaged properties hide one badly damaged column;
    these are the per-column analogue of our max KS and worst category shift. shape_min is the
    smallest score of all columns, shape_min_numeric the smallest KSComplement and
    shape_min_categorical the smallest TVComplement. pair_min is the smallest pair score;
    pairs SDMetrics could not score (NaN) are skipped, as its own average skips them. A group
    with no scored member is None.
    """
    metric = shapes["Metric"]
    return {
        "shape_min": _min_score(shapes["Score"]),
        "shape_min_numeric": _min_score(shapes.loc[metric == NUMERIC_SHAPE_METRIC, "Score"]),
        "shape_min_categorical": _min_score(
            shapes.loc[metric == CATEGORICAL_SHAPE_METRIC, "Score"]
        ),
        "pair_min": _min_score(pairs["Score"]),
    }


def score_worst(train: pd.DataFrame, synth: pd.DataFrame, label: str, seed: int) -> dict[str, Any]:
    """Worst-column and worst-pair QualityReport scores of one synthetic table.

    Same metadata, prepared frames, raw (unmasked) tables and seeding as score_run's quality
    call, so the averaged Column Shapes and Column Pair Trends are recorded again as a check.
    """
    from sdmetrics.reports.single_table import QualityReport

    md = build_metadata(train, label)
    real, fake = (prepare_frame(d, md) for d in (train, synth))
    values: dict[str, float | None] = {}
    runtimes: dict[str, float] = {}
    errors: dict[str, str] = {}

    def quality() -> float:
        report = QualityReport()
        report.generate(real, fake, md, verbose=False)
        props = report.get_properties().set_index("Property")["Score"]
        values["column_shapes"] = float(props["Column Shapes"])
        values["column_pair_trends"] = float(props["Column Pair Trends"])
        values.update(
            worst_scores(
                report.get_details("Column Shapes"), report.get_details("Column Pair Trends")
            )
        )
        return float(report.get_score())

    _timed("quality_score", quality, seed, values, runtimes, errors)
    if errors:
        values.update(dict.fromkeys(("column_shapes", "column_pair_trends", *WORST_KEYS)))
    return {"metrics": values, "runtime_s": runtimes, "errors": errors}


def efficacy_class(labels: pd.Series) -> str | None:
    """Name of the SDMetrics efficacy metric for the label, or None for regression."""
    if detect_task(labels.dropna()) == "regression":
        return None
    return "BinaryLogisticRegression" if labels.nunique() == 2 else "MulticlassDecisionTree"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))


def _timed(
    name: str,
    fn: Callable[[], Any],
    seed: int,
    values: dict[str, float | None],
    runtimes: dict[str, float],
    errors: dict[str, str],
) -> None:
    """Run one metric, seeded and timed. An exception is recorded, never raised."""
    _seed_all(seed)
    t0 = time.perf_counter()
    try:
        score = float(fn())
        values[name] = score if np.isfinite(score) else None
        if not np.isfinite(score):
            errors[name] = "non-finite score"
    except Exception as e:  # recorded, never silently dropped
        values[name] = None
        errors[name] = f"{type(e).__name__}: {e}"
    runtimes[name] = round(time.perf_counter() - t0, 3)


def score_run(
    train: pd.DataFrame,
    synth: pd.DataFrame,
    base: pd.DataFrame,
    label: str,
    seed: int,
) -> dict[str, Any]:
    """All SDMetrics scores of one synthetic table."""
    from sdmetrics.reports.single_table import QualityReport
    from sdmetrics.single_table import (
        BinaryLogisticRegression,
        DCROverfittingProtection,
        LogisticDetection,
        MulticlassDecisionTreeClassifier,
        NewRowSynthesis,
    )

    md = build_metadata(train, label)
    real, fake, hold = (prepare_frame(d, md) for d in (train, synth, base))
    values: dict[str, float | None] = {}
    runtimes: dict[str, float] = {}
    errors: dict[str, str] = {}

    def quality() -> float:
        report = QualityReport()
        report.generate(real, fake, md, verbose=False)
        props = report.get_properties().set_index("Property")["Score"]
        values["column_shapes"] = float(props["Column Shapes"])
        values["column_pair_trends"] = float(props["Column Pair Trends"])
        return float(report.get_score())

    _timed("quality_score", quality, seed, values, runtimes, errors)
    if "quality_score" in errors:
        values.update(column_shapes=None, column_pair_trends=None)
    fake_seen, n_replaced_detection = mask_unseen_categories(fake, real, md)
    _timed(
        "logistic_detection",
        lambda: LogisticDetection.compute(real, fake_seen, md),
        seed,
        values,
        runtimes,
        errors,
    )
    _timed(
        "dcr_overfitting",
        lambda: DCROverfittingProtection.compute(real, fake, hold, md, None),
        seed,
        values,
        runtimes,
        errors,
    )
    _timed(
        "new_row_synthesis",
        lambda: NewRowSynthesis.compute(real, fake, md),
        seed,
        values,
        runtimes,
        errors,
    )

    kind = efficacy_class(train[label])
    if kind is not None:
        metric = (
            BinaryLogisticRegression
            if kind == "BinaryLogisticRegression"
            else (MulticlassDecisionTreeClassifier)
        )

        def efficacy() -> float:
            hold_vs_fake, _ = mask_unseen_categories(hold, fake, md, label)
            hold_vs_real, _ = mask_unseen_categories(hold, real, md, label)
            tstr = float(metric.compute(hold_vs_fake, fake, md, None, label))
            trtr = float(metric.compute(hold_vs_real, real, md, None, label))
            values["efficacy_tstr"], values["efficacy_trtr"] = tstr, trtr
            if trtr <= 0:
                raise ValueError("TRTR efficacy is 0, ratio undefined")
            return tstr / trtr

        _timed("efficacy_ratio", efficacy, seed, values, runtimes, errors)
        values.setdefault("efficacy_tstr", None)
        values.setdefault("efficacy_trtr", None)
    return {
        "metrics": values,
        "runtime_s": runtimes,
        "runtime_total_s": round(sum(runtimes.values()), 3),
        "errors": errors,
        "efficacy_model": kind,
        "n_unseen_category_cells_replaced_for_detection": n_replaced_detection,
    }


def stats_match(a: dict[str, float | None], b: dict[str, float | None]) -> bool:
    """True when two stats dicts have the same keys and values within STATS_TOLERANCE."""
    if a.keys() != b.keys():
        return False
    for k, x in a.items():
        y = b[k]
        if (x is None) != (y is None):
            return False
        if x is not None and y is not None and abs(x - y) > STATS_TOLERANCE:
            return False
    return True


def run_unit(
    df: pd.DataFrame, ds: Dataset, seed: int, stored: dict[str, dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """All generators of one (dataset, seed): SDMetrics scores plus a timed re-run of ours."""
    warnings.filterwarnings("ignore")
    train, gold, base = split_three(df, ds.label, seed, ds.max_rows)
    out = []
    with threadpool_limits(limits=1):
        for name in (HELDOUT, *GENERATORS):
            rec: dict[str, Any] = {"dataset": ds.name, "seed": seed, "generator": name}
            try:
                synth = build_synth(name, train, gold, ds.label, seed)
                rec.update(score_run(train, synth, base, ds.label, seed))
                if stored is not None:
                    t0 = time.perf_counter()
                    report = generate_trust_report(
                        train, synth, ds.label, seed=seed, thresholds=Thresholds(), holdout_df=base
                    )
                    elapsed = round(time.perf_counter() - t0, 3)
                    match = stats_match(extract_stats(report), stored[name]["stats"])
                    rec["ours_rerun"] = {"runtime_s": elapsed, "stats_match": match}
            except Exception as e:  # a failure outside the metrics, recorded
                rec.setdefault("metrics", {})
                rec.setdefault("runtime_s", {})
                rec.setdefault("runtime_total_s", 0.0)
                rec.setdefault("errors", {})["run"] = f"{type(e).__name__}: {e}"
            out.append(rec)
    return out


def run_worst_unit(df: pd.DataFrame, ds: Dataset, seed: int) -> list[dict[str, Any]]:
    """The worst-column QualityReport pass for all generators of one (dataset, seed)."""
    warnings.filterwarnings("ignore")
    train, gold, _ = split_three(df, ds.label, seed, ds.max_rows)
    out = []
    with threadpool_limits(limits=1):
        for name in (HELDOUT, *GENERATORS):
            rec: dict[str, Any] = {"dataset": ds.name, "seed": seed, "generator": name}
            try:
                synth = build_synth(name, train, gold, ds.label, seed)
                rec.update(score_worst(train, synth, ds.label, seed))
            except Exception as e:  # recorded, never dropped
                rec.update(metrics={}, runtime_s={}, errors={"run": f"{type(e).__name__}: {e}"})
            out.append(rec)
    return out


def verify_directions() -> dict[str, Any]:
    """Score hand-built cases to check which direction is good for every metric.

    Uses a fresh sample, an exact copy of the training rows, a copy with every column permuted
    independently, a copy with only the label permuted and an easily detectable shifted sample.
    Raises AssertionError when a metric does not behave as the comparison assumes.
    """
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)

    def make(n: int) -> pd.DataFrame:
        x = rng.normal(size=n)
        y = x + 0.5 * rng.normal(size=n)
        c = rng.choice(list("abc"), n)
        lab = np.where(x + rng.normal(size=n) > 0, "pos", "neg")
        return pd.DataFrame({"x": x, "y": y, "c": c, "label": lab})

    train, hold, fresh = make(600), make(300), make(300)
    copy = train.sample(300, random_state=0).reset_index(drop=True)
    shuffled = fresh.apply(lambda s: s.sample(frac=1, random_state=int(rng.integers(10**6))).values)
    labelshuf = fresh.assign(label=fresh["label"].sample(frac=1, random_state=3).to_numpy())
    shifted = fresh.assign(x=fresh["x"] + 4.0, y=fresh["y"] * 0.1)
    cases = {
        "fresh_sample": fresh,
        "exact_copy_of_train": copy,
        "all_columns_permuted": shuffled,
        "label_permuted": labelshuf,
        "shifted_columns": shifted,
    }
    result = {name: score_run(train, s, hold, "label", 0)["metrics"] for name, s in cases.items()}
    r = result
    checks = {
        "dcr_overfitting: exact copy scores far lower than fresh (low = memorised)": r[
            "exact_copy_of_train"
        ]["dcr_overfitting"]
        < r["fresh_sample"]["dcr_overfitting"] - 0.3,
        "new_row_synthesis: exact copy scores 0, fresh near 1 (low = memorised)": r[
            "exact_copy_of_train"
        ]["new_row_synthesis"]
        < 0.05
        and r["fresh_sample"]["new_row_synthesis"] > 0.9,
        "logistic_detection: shifted sample scores far lower than fresh (low = detectable)": r[
            "shifted_columns"
        ]["logistic_detection"]
        < r["fresh_sample"]["logistic_detection"] - 0.3,
        "column_pair_trends: permuted columns score lower than fresh (low = worse)": r[
            "all_columns_permuted"
        ]["column_pair_trends"]
        < r["fresh_sample"]["column_pair_trends"] - 0.05,
        "column_shapes: shifted sample scores lower than fresh (low = worse)": r["shifted_columns"][
            "column_shapes"
        ]
        < r["fresh_sample"]["column_shapes"] - 0.1,
        "quality_score: shifted sample scores lower than fresh (low = worse)": r["shifted_columns"][
            "quality_score"
        ]
        < r["fresh_sample"]["quality_score"] - 0.05,
        "efficacy_ratio: permuted label scores lower than fresh (low = worse)": r["label_permuted"][
            "efficacy_ratio"
        ]
        < r["fresh_sample"]["efficacy_ratio"] - 0.15,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"SDMetrics direction assumptions failed: {failed}")
    return {
        "scores": result,
        "assumptions_verified": list(checks),
        "note": "every SDMetrics score used is higher = better, so each check warns when LOW",
    }


def sdmetrics_versions() -> dict[str, str]:
    import sdmetrics

    out = {"sdmetrics": sdmetrics.__version__}
    try:
        import copulas

        out["copulas"] = copulas.__version__
    except ImportError:  # pragma: no cover
        pass
    return out


def main_worst(args: argparse.Namespace) -> int:
    chosen = [d for d in DATASETS if d.name in args.datasets]
    loaded = {d.name: load_dataset(d)[0] for d in chosen}
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    units = [(d, s) for d in chosen for s in range(args.seeds)]
    results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(run_worst_unit)(loaded[d.name], d, s) for d, s in units
    )
    runs = [r for unit in results for r in unit]
    with open(args.out / WORST_FILE, "w") as f:
        for r in runs:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    n_err = sum(bool(r["errors"]) for r in runs)
    print(f"{len(runs)} worst-column runs, {n_err} with errors, {time.perf_counter() - t0:.0f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=SEEDS, help="seeds 0..N-1 (default 40)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--datasets", nargs="*", default=[d.name for d in DATASETS])
    ap.add_argument(
        "--worst-only",
        action="store_true",
        help="only re-run QualityReport and write the worst-column scores to "
        "sdmetrics_worst.jsonl (existing result files are not touched)",
    )
    ap.add_argument(
        "--no-rerun-ours", action="store_true", help="skip re-running our validator (no timing)"
    )
    args = ap.parse_args(argv)

    if args.worst_only:
        return main_worst(args)
    directions = verify_directions()
    chosen = [d for d in DATASETS if d.name in args.datasets]
    loaded = {d.name: load_dataset(d)[0] for d in chosen}
    stored: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    if not args.no_rerun_ours:
        with open(RESULTS / "runs.jsonl") as f:
            for line in f:
                r = json.loads(line)
                stored.setdefault((r["dataset"], r["seed"]), {})[r["generator"]] = r
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    units = [(d, s) for d in chosen for s in range(args.seeds)]
    results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(run_unit)(loaded[d.name], d, s, stored.get((d.name, s))) for d, s in units
    )
    total = time.perf_counter() - t0
    runs = [r for unit in results for r in unit]
    with open(args.out / "sdmetrics_runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    (args.out / "sdmetrics_directions.json").write_text(json.dumps(directions, indent=2) + "\n")
    meta = {
        "command": "python -m experiments.compare_sdmetrics " + " ".join(sys.argv[1:]),
        "versions": {**versions(), **sdmetrics_versions()},
        "n_runs": len(runs),
        "n_runs_with_metric_errors": sum(bool(r["errors"]) for r in runs),
        "n_ours_stats_mismatch": sum(
            1 for r in runs if r.get("ours_rerun") and not r["ours_rerun"]["stats_match"]
        ),
        "wall_clock_seconds": round(total, 1),
        "jobs": args.jobs,
        "efficacy_models": "BinaryLogisticRegression (2 classes), MulticlassDecisionTreeClassifier",
        "efficacy_score": "SDMetrics default (F1 binary, macro F1 multiclass), TSTR / TRTR ratio",
    }
    (args.out / "sdmetrics_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(runs)} runs, {meta['n_runs_with_metric_errors']} with metric errors, {total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
