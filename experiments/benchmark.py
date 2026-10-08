"""Phase 3 benchmark: run the validator against good and deliberately broken generators.

    python -m experiments.benchmark [--seeds 20] [--jobs 8] [--out experiments/results]

For each (dataset, seed) the real data is split into three disjoint parts:
- T       50%  generator training set, passed to generate_trust_report as `real_df`
- H_gold  25%  held-out real rows, used as the "good" generator output (heldout_real)
- H_base  25%  held-out real rows, passed as `holdout_df` (the privacy baseline)

Every generator makes n_synth = len(H_gold) rows. The copula and the two memorisation
injections (exact_copy, near_copy) use T only; the other four injections start from H_gold
(real rows not in T) so that the privacy check has no training rows to find in them.
The splits are stratified by the label. The Adult data (48,842 rows) is subsampled to
ADULT_ROWS rows per seed before splitting to keep the runtime sane; the other datasets
are used whole.

For each run the raw statistics every warning uses (app/checks.py) are recorded, so
thresholds can be recalibrated offline with experiments/calibrate.py. The warning codes
stored in runs.jsonl use the PROVISIONAL default thresholds and are for reference only.
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from app.checks import extract_stats
from app.report import generate_trust_report
from app.thresholds import Thresholds
from experiments.generators import GENERATORS, HELDOUT

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
ADULT_ROWS = 6000
TRAIN_FRAC = 0.5
SEEDS = 40


@dataclass(frozen=True)
class Dataset:
    name: str
    data_id: int
    label: str
    max_rows: int | None = None


DATASETS = (
    Dataset("adult", 1590, "class", ADULT_ROWS),
    Dataset("german_credit", 31, "class"),
    Dataset("cmc", 23, "Contraceptive_method_used"),
)


def load_dataset(ds: Dataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    bunch = fetch_openml(data_id=ds.data_id, as_frame=True, parser="auto", data_home=str(CACHE))
    df = bunch.frame.reset_index(drop=True)
    details = bunch.details
    info = {
        "name": ds.name,
        "openml_name": details.get("name"),
        "data_id": ds.data_id,
        "version": details.get("version"),
        "licence": details.get("licence"),
        "label": ds.label,
        "n_rows_full": int(len(df)),
        "n_columns": int(df.shape[1]),
        "max_rows_per_seed": ds.max_rows,
    }
    return df, info


def split_three(
    df: pd.DataFrame, label: str, seed: int, max_rows: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """T (50%), H_gold (25%), H_base (25%), disjoint, stratified by label."""
    if max_rows is not None and len(df) > max_rows:
        df, _ = train_test_split(df, train_size=max_rows, random_state=seed, stratify=df[label])
    train, rest = train_test_split(df, train_size=TRAIN_FRAC, random_state=seed, stratify=df[label])
    gold, base = train_test_split(rest, train_size=0.5, random_state=seed, stratify=rest[label])
    return (
        train.reset_index(drop=True),
        gold.reset_index(drop=True),
        base.reset_index(drop=True),
    )


def build_synth(
    name: str, train: pd.DataFrame, gold: pd.DataFrame, label: str, seed: int
) -> pd.DataFrame:
    """The synthetic table of generator `name` for one split (`heldout_real` is H_gold)."""
    if name == HELDOUT:
        return gold
    return GENERATORS[name](train, len(gold), label, seed, fresh=gold)


def run_unit(df: pd.DataFrame, ds: Dataset, seed: int) -> list[dict[str, Any]]:
    """All generators for one (dataset, seed)."""
    train, gold, base = split_three(df, ds.label, seed, ds.max_rows)
    n_synth = len(gold)
    runs = []
    with threadpool_limits(limits=1):
        for name in (HELDOUT, *GENERATORS):
            t0 = time.perf_counter()
            error = None
            stats, codes = {}, []
            try:
                synth = build_synth(name, train, gold, ds.label, seed)
                report = generate_trust_report(
                    train,
                    synth,
                    ds.label,
                    seed=seed,
                    thresholds=Thresholds(),
                    holdout_df=base,
                )
                stats = extract_stats(report)
                codes = sorted({w["code"] for w in report["warnings"]})
            except Exception as e:  # recorded, never silently dropped
                error = f"{type(e).__name__}: {e}"
            runs.append(
                {
                    "dataset": ds.name,
                    "seed": seed,
                    "generator": name,
                    "n_train": len(train),
                    "n_holdout": len(base),
                    "n_synth": n_synth,
                    "runtime_s": round(time.perf_counter() - t0, 3),
                    "provisional_warning_codes": codes,
                    "stats": stats,
                    "error": error,
                }
            )
    return runs


def versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=SEEDS, help="seeds 0..N-1 (default 40)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--datasets", nargs="*", default=[d.name for d in DATASETS])
    args = ap.parse_args(argv)

    chosen = [d for d in DATASETS if d.name in args.datasets]
    loaded = {d.name: load_dataset(d) for d in chosen}
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    units = [(d, s) for d in chosen for s in range(args.seeds)]
    results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(run_unit)(loaded[d.name][0], d, s) for d, s in units
    )
    total = time.perf_counter() - t0
    runs = [r for unit in results for r in unit]

    with open(args.out / "runs.jsonl", "w") as f:
        for r in runs:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    meta = {
        "command": "python -m experiments.benchmark " + " ".join(sys.argv[1:]),
        "seeds": list(range(args.seeds)),
        "generators": [HELDOUT, *GENERATORS],
        "split": {"train": TRAIN_FRAC, "heldout_gold": 0.25, "heldout_baseline": 0.25},
        "n_synth_rule": "len(H_gold) for every generator",
        "datasets": [loaded[d.name][1] for d in chosen],
        "versions": versions(),
        "n_runs": len(runs),
        "n_errors": sum(r["error"] is not None for r in runs),
        "wall_clock_seconds": round(total, 1),
        "sum_of_run_seconds": round(sum(r["runtime_s"] for r in runs), 1),
        "jobs": args.jobs,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(runs)} runs, {meta['n_errors']} errors, {total:.0f}s wall clock")
    return 0


if __name__ == "__main__":
    sys.exit(main())
