"""Calibration and evaluation for the SDMetrics comparison (no SDMetrics import needed).

    python -m experiments.compare_report [--results experiments/results]

Reads experiments/results/runs.jsonl (our validator, written by experiments.benchmark) and
experiments/results/sdmetrics_runs.jsonl (written by experiments.compare_sdmetrics) and writes
comparison.json and comparison.md.

SDMetrics thresholds are learned with exactly the protocol used for our own thresholds
(experiments/calibrate.py): from the heldout_real runs of seeds 0 to 19 only, pooled over the
three datasets, with the rule "null extreme -/+ max(10% of the extreme, 0.01)". Every SDMetrics
score used here is a higher-is-better score (verified on hand-built cases, see
results/sdmetrics_directions.json), so every check warns when the score is BELOW its
threshold; the code supports both directions. The thresholds are then applied to seeds 20 to 39.

Four views are compared:
- ours: the intended-warning mapping of calibrate.INTENDED
- sdmetrics_strict: the analogous SDMetrics check for the same failure mode (SD_INTENDED).
  SDMetrics has no diversity or collapse check, so mode_collapse has no intended check and
  can never count as detected in this view
- sdmetrics_lenient: any of the individual SDMetrics checks fires
- sdmetrics_overall: only the single QualityReport overall score, one calibrated threshold
"""

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.thresholds import Thresholds
from experiments.calibrate import (
    ABS_MARGIN,
    DETECTION_TARGET,
    HEADLINE_MODES,
    INTENDED,
    N_CAL_SEEDS,
    NULL,
    REL_MARGIN,
    calibrate_thresholds,
    embed_in_readme,
    fired_codes,
)
from experiments.generators import BROKEN

RESULTS = Path(__file__).resolve().parent / "results"
ABOVE = "above"
BELOW = "below"

# Individual SDMetrics checks -> direction in which a value is a warning. All of them are
# higher-is-better scores, so they warn when the score is low.
SD_CHECKS: dict[str, str] = {
    "column_shapes": BELOW,
    "column_pair_trends": BELOW,
    "logistic_detection": BELOW,
    "dcr_overfitting": BELOW,
    "new_row_synthesis": BELOW,
    "efficacy_ratio": BELOW,
}
OVERALL = "quality_score"
OVERALL_DIRECTION = BELOW

# Failure mode -> the SDMetrics checks that are the analogue of our intended warnings.
# mode_collapse is empty on purpose: SDMetrics has no diversity or collapse check.
SD_INTENDED: dict[str, tuple[str, ...]] = {
    "exact_copy": ("dcr_overfitting", "new_row_synthesis"),
    "near_copy": ("dcr_overfitting", "new_row_synthesis"),
    "near_copy_5pct": ("dcr_overfitting", "new_row_synthesis"),
    "column_shuffle": ("column_pair_trends", "logistic_detection"),
    "mode_collapse": (),
    "category_oversample": ("column_shapes",),
    "label_shuffle": ("efficacy_ratio",),
}
MODES = (*HEADLINE_MODES, "near_copy_5pct")
VIEWS = ("ours", "sdmetrics_strict", "sdmetrics_lenient", "sdmetrics_overall")
ORDER = (NULL, "gaussian_copula", *BROKEN)


def threshold_for(values: Iterable[float], direction: str) -> tuple[float, dict[str, Any]]:
    """The calibrated threshold of one score from its null values.

    Same rule as experiments.calibrate: for a warn-above statistic the maximum null value plus
    max(10% of its magnitude, 0.01); for a warn-below statistic the minimum minus the same.
    """
    pool = list(values)
    if not pool:
        raise ValueError("no null values to calibrate on")
    if direction == ABOVE:
        extreme = max(pool)
        thr = extreme + max(REL_MARGIN * abs(extreme), ABS_MARGIN)
    elif direction == BELOW:
        extreme = min(pool)
        thr = extreme - max(REL_MARGIN * abs(extreme), ABS_MARGIN)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    info = {
        "direction": direction,
        "n_null_values": len(pool),
        "null_extreme": extreme,
        "threshold": round(thr, 6),
        "max_per_check_false_alarm_rate_before_margin": 1.0 / (len(pool) + 1),
    }
    return round(thr, 6), info


def fires(value: float | None, threshold: float, direction: str) -> bool:
    """True when the value is a warning. A missing value (metric errored) never fires."""
    if value is None:
        return False
    return value > threshold if direction == ABOVE else value < threshold


def calibrate_sdmetrics(
    null_runs: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Thresholds for every SDMetrics score from heldout_real calibration runs only."""
    if any(r["generator"] != NULL for r in null_runs):
        raise ValueError("calibration may only use heldout_real runs")
    if any(r["seed"] >= N_CAL_SEEDS for r in null_runs):
        raise ValueError(f"calibration may only use seeds below {N_CAL_SEEDS}")
    directions = {**SD_CHECKS, OVERALL: OVERALL_DIRECTION}
    thresholds: dict[str, float] = {}
    details: dict[str, Any] = {}
    for name, direction in directions.items():
        pool = [r["metrics"][name] for r in null_runs if r["metrics"].get(name) is not None]
        n_missing = len(null_runs) - len(pool)
        if not pool:
            details[name] = {"direction": direction, "note": "no null values, never fires"}
            continue
        thr, info = threshold_for(pool, direction)
        info["n_null_runs_without_value"] = n_missing
        thresholds[name] = thr
        details[name] = info
    return thresholds, details


def sd_fired(run: dict[str, Any], thresholds: dict[str, float]) -> set[str]:
    """Individual SDMetrics checks that warn on a run (the overall score is separate)."""
    out = set()
    for name, direction in SD_CHECKS.items():
        if name in thresholds and fires(run["metrics"].get(name), thresholds[name], direction):
            out.add(name)
    return out


def overall_fired(run: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return OVERALL in thresholds and fires(
        run["metrics"].get(OVERALL), thresholds[OVERALL], OVERALL_DIRECTION
    )


def join_runs(ours: list[dict[str, Any]], sd: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every SDMetrics record with our benchmark record for the same run."""

    def key(r: dict[str, Any]) -> tuple[str, int, str]:
        return (r["dataset"], r["seed"], r["generator"])

    by_key = {key(r): r for r in ours}
    missing = [key(r) for r in sd if key(r) not in by_key]
    if missing:
        raise ValueError(
            f"{len(missing)} SDMetrics runs have no benchmark run, first: {missing[0]}"
        )
    return [{**r, "ours": by_key[key(r)]} for r in sd]


def evaluate(
    paired: list[dict[str, Any]], ours_th: Thresholds, sd_th: dict[str, float]
) -> list[dict[str, Any]]:
    """Per run: which warnings each tool raised and whether its intended check hit."""
    rows = []
    for r in paired:
        gen = r["generator"]
        ours = fired_codes(r["ours"], ours_th)
        sd = sd_fired(r, sd_th)
        o_int, s_int = INTENDED.get(gen), SD_INTENDED.get(gen)
        rows.append(
            {
                "dataset": r["dataset"],
                "seed": r["seed"],
                "generator": gen,
                "ours": {
                    "any": bool(ours),
                    "intended": bool(set(o_int) & ours) if o_int else None,
                },
                "sdmetrics_strict": {
                    "any": bool(sd),
                    "intended": bool(set(s_int) & sd) if s_int is not None else None,
                },
                "sdmetrics_lenient": {"any": bool(sd), "intended": bool(sd) if o_int else None},
                "sdmetrics_overall": {
                    "any": overall_fired(r, sd_th),
                    "intended": overall_fired(r, sd_th) if o_int else None,
                },
                "sd_fired": sorted(sd),
            }
        )
    return rows


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per generator and view: n, intended-hit rate, any-warning rate and counts."""
    out: dict[str, Any] = {}
    for gen in ORDER:
        rs = [r for r in rows if r["generator"] == gen]
        if not rs:
            continue
        entry: dict[str, Any] = {"n_runs": len(rs)}
        for v in VIEWS:
            intended = [r[v]["intended"] for r in rs if r[v]["intended"] is not None]
            anys = [r[v]["any"] for r in rs]
            entry[v] = {
                "intended_fired": _rate(intended),
                "n_intended_fired": sum(intended),
                "any_warning": _rate(anys),
                "n_any_warning": sum(anys),
            }
        out[gen] = entry
    return out


def view_headline(table: dict[str, Any], view: str) -> dict[str, Any]:
    """Failure modes detected (intended hit rate at least DETECTION_TARGET) for one view."""
    modes = {}
    for m in MODES:
        rate = table[m][view]["intended_fired"]
        modes[m] = {"rate": rate, "detected": rate is not None and rate >= DETECTION_TARGET}
    null, cop = table[NULL][view], table["gaussian_copula"][view]
    n_runs = table[NULL]["n_runs"]
    return {
        "n_detected": sum(modes[m]["detected"] for m in HEADLINE_MODES),
        "n_modes": len(HEADLINE_MODES),
        "modes": modes,
        "heldout_real_false_alarm": {
            "k": null["n_any_warning"],
            "n": n_runs,
            "rate": null["any_warning"],
        },
        "gaussian_copula_any_warning": {
            "k": cop["n_any_warning"],
            "n": table["gaussian_copula"]["n_runs"],
            "rate": cop["any_warning"],
        },
    }


def runtime_summary(paired: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean seconds per run. Ours is the validator alone, re-timed in the same job."""
    n = len(paired)
    if not n:
        return {}
    ours = [r["ours_rerun"]["runtime_s"] for r in paired if r.get("ours_rerun")]
    sd_total = [r["runtime_total_s"] for r in paired]
    per_metric: dict[str, list[float]] = {}
    for r in paired:
        for k, v in r["runtime_s"].items():
            per_metric.setdefault(k, []).append(v)
    return {
        "n_runs": n,
        "ours_mean_s_per_run": sum(ours) / len(ours) if ours else None,
        "sdmetrics_mean_s_per_run": sum(sd_total) / n,
        "sdmetrics_mean_s_per_metric": {k: sum(v) / len(v) for k, v in per_metric.items()},
    }


def error_summary(sd_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """How many runs had no value for each metric, and the first error text."""
    out: dict[str, Any] = {}
    for r in sd_runs:
        for name, msg in r["errors"].items():
            e = out.setdefault(name, {"n": 0, "first": msg})
            e["n"] += 1
    return out


def build_comparison(
    ours_runs: list[dict[str, Any]], sd_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    paired = join_runs(ours_runs, sd_runs)
    cal = [r for r in paired if r["seed"] < N_CAL_SEEDS]
    test = [r for r in paired if r["seed"] >= N_CAL_SEEDS]
    null_cal = [r for r in cal if r["generator"] == NULL]
    ours_th, _ = calibrate_thresholds([r["ours"] for r in null_cal])
    sd_th, sd_details = calibrate_sdmetrics(null_cal)

    rows = evaluate(test, ours_th, sd_th)
    table = rates(rows)
    datasets = sorted({r["dataset"] for r in test})
    by_ds = {d: rates([r for r in rows if r["dataset"] == d]) for d in datasets}

    null_rows = [r for r in rows if r["generator"] == NULL]
    per_check_fa = {name: sum(name in r["sd_fired"] for r in null_rows) for name in SD_CHECKS}
    per_check_fa[OVERALL] = sum(r["sdmetrics_overall"]["any"] for r in null_rows)

    return {
        "protocol": {
            "calibration_seeds": sorted({r["seed"] for r in cal}),
            "test_seeds": sorted({r["seed"] for r in test}),
            "n_test_runs_per_generator": len(null_rows),
            "rule": (
                f"threshold = null extreme -/+ max({REL_MARGIN} * |extreme|, {ABS_MARGIN}) from "
                "heldout_real runs of the calibration seeds, pooled over datasets"
            ),
            "detection_rule": f"intended check fires in at least {DETECTION_TARGET:.0%} of runs",
            "ours_intended": {m: list(INTENDED[m]) for m in MODES},
            "sdmetrics_strict_intended": {m: list(SD_INTENDED[m]) for m in MODES},
            "sdmetrics_lenient": "any individual SDMetrics check fires",
            "sdmetrics_overall": f"only the QualityReport overall score ({OVERALL})",
        },
        "sdmetrics_thresholds": sd_th,
        "sdmetrics_threshold_details": sd_details,
        "table": table,
        "headline": {v: view_headline(table, v) for v in VIEWS},
        "per_dataset": by_ds,
        "per_dataset_headline": {
            d: {v: view_headline(by_ds[d], v) for v in VIEWS} for d in datasets
        },
        "sdmetrics_heldout_real_false_alarms_per_check": {
            "k": per_check_fa,
            "n": len(null_rows),
        },
        "runtime": runtime_summary(paired),
        "sdmetrics_errors": error_summary(sd_runs),
        "ours_stats_mismatch_runs": sum(
            1 for r in paired if r.get("ours_rerun") and not r["ours_rerun"]["stats_match"]
        ),
    }


# Markdown ---------------------------------------------------------------------------------

VIEW_TITLES = {
    "ours": "Ours (intended warning)",
    "sdmetrics_strict": "SDMetrics strict",
    "sdmetrics_lenient": "SDMetrics lenient",
    "sdmetrics_overall": "SDMetrics overall score only",
}


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def main_table_markdown(cmp: dict[str, Any]) -> str:
    table, head = cmp["table"], cmp["headline"]
    lines = [
        "| Failure mode | " + " | ".join(VIEW_TITLES[v] for v in VIEWS) + " |",
        "|---|" + "|".join(["---"] * len(VIEWS)) + "|",
    ]
    for m in MODES:
        label = f"`{m}`" + (" (not in count)" if m not in HEADLINE_MODES else "")
        cells = []
        for v in VIEWS:
            rate = table[m][v]["intended_fired"]
            if m != "near_copy_5pct" and head[v]["modes"][m]["detected"]:
                cells.append(f"{_f(rate)} detected")
            else:
                cells.append(_f(rate))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| Failure modes detected | "
        + " | ".join(f"{head[v]['n_detected']} of {head[v]['n_modes']}" for v in VIEWS)
        + " |"
    )
    fa = [head[v]["heldout_real_false_alarm"] for v in VIEWS]
    lines.append(
        "| `heldout_real` runs with any warning | "
        + " | ".join(f"{x['k']} of {x['n']}" for x in fa)
        + " |"
    )
    cp = [head[v]["gaussian_copula_any_warning"] for v in VIEWS]
    lines.append(
        "| `gaussian_copula` runs with any warning | "
        + " | ".join(f"{x['k']} of {x['n']}" for x in cp)
        + " |"
    )
    return "\n".join(lines)


def per_dataset_markdown(cmp: dict[str, Any]) -> str:
    ds = list(cmp["per_dataset"])
    lines = [
        "| Failure mode | View | " + " | ".join(ds) + " |",
        "|---|---|" + "|".join(["---"] * len(ds)) + "|",
    ]
    for m in MODES:
        for v in VIEWS:
            cells = [_f(cmp["per_dataset"][d][m][v]["intended_fired"]) for d in ds]
            lines.append(f"| `{m}` | {VIEW_TITLES[v]} | " + " | ".join(cells) + " |")
    for gen in (NULL, "gaussian_copula"):
        for v in VIEWS:
            cells = []
            for d in ds:
                x = cmp["per_dataset"][d][gen][v]
                cells.append(f"{x['n_any_warning']} of {cmp['per_dataset'][d][gen]['n_runs']}")
            lines.append(f"| `{gen}` any warning | {VIEW_TITLES[v]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def thresholds_markdown(cmp: dict[str, Any]) -> str:
    fa = cmp["sdmetrics_heldout_real_false_alarms_per_check"]
    lines = [
        "| SDMetrics score | Warns when | Calibrated threshold | Null extreme | Null values "
        "| `heldout_real` test false alarms |",
        "|---|---|---|---|---|---|",
    ]
    for name, d in cmp["sdmetrics_threshold_details"].items():
        if "threshold" not in d:
            lines.append(f"| `{name}` | {d['direction']} | n/a | n/a | 0 | n/a |")
            continue
        lines.append(
            f"| `{name}` | {d['direction']} | {d['threshold']:.4f} | {d['null_extreme']:.4f} "
            f"| {d['n_null_values']} | {fa['k'][name]} of {fa['n']} |"
        )
    return "\n".join(lines)


def runtime_markdown(cmp: dict[str, Any]) -> str:
    rt = cmp["runtime"]
    lines = [
        "| Tool | Mean seconds per run |",
        "|---|---|",
        f"| Ours (`generate_trust_report`, validator only) | {_f(rt['ours_mean_s_per_run'])} |",
        f"| SDMetrics (all metrics below) | {_f(rt['sdmetrics_mean_s_per_run'])} |",
    ]
    for k, v in rt["sdmetrics_mean_s_per_metric"].items():
        lines.append(f"| SDMetrics `{k}` | {v:.2f} |")
    return "\n".join(lines)


def comparison_markdown(cmp: dict[str, Any]) -> str:
    p = cmp["protocol"]
    t0, t1 = p["test_seeds"][0], p["test_seeds"][-1]
    c0, c1 = p["calibration_seeds"][0], p["calibration_seeds"][-1]
    err = cmp["sdmetrics_errors"]
    err_line = (
        "SDMetrics errors: none."
        if not err
        else "SDMetrics errors (runs with no value for the metric): "
        + ", ".join(f"`{k}` {v['n']}" for k, v in err.items())
        + "."
    )
    parts = [
        f"Test seeds {t0} to {t1} ({p['n_test_runs_per_generator']} runs per generator, 3 "
        f"datasets). Thresholds for both tools come from `heldout_real` runs of seeds {c0} to "
        f"{c1} with the same rule. A mode counts as detected when its intended check fires in at "
        f"least {DETECTION_TARGET:.0%} of test runs. Cells are the fraction of test runs in "
        "which the intended check fired. SDMetrics strict uses the analogous SDMetrics check "
        "and has none for `mode_collapse` (SDMetrics has no diversity check). SDMetrics lenient "
        "counts any individual SDMetrics check. The last column uses only the overall Quality "
        "score.",
        "",
        main_table_markdown(cmp),
        "",
        "Per dataset (intended check fired, then any-warning counts):",
        "",
        per_dataset_markdown(cmp),
        "",
        "SDMetrics thresholds and false alarms of each score on the `heldout_real` test runs:",
        "",
        thresholds_markdown(cmp),
        "",
        "Runtime (all runs; both tools timed inside the same parallel job, one thread each):",
        "",
        runtime_markdown(cmp),
        "",
        err_line,
        f"Runs where our re-run did not reproduce the stored statistics: "
        f"{cmp['ours_stats_mismatch_runs']}.",
        "",
    ]
    return "\n".join(parts)


START_MARK = "<!-- sdmetrics-comparison:start -->"
END_MARK = "<!-- sdmetrics-comparison:end -->"
FULL_START_MARK = "<!-- sdmetrics-full:start -->"
FULL_END_MARK = "<!-- sdmetrics-full:end -->"


def readme_markdown(cmp: dict[str, Any]) -> str:
    """The short block pasted into the README: one sentence of method and the main table."""
    p = cmp["protocol"]
    return (
        f"Test seeds {p['test_seeds'][0]} to {p['test_seeds'][-1]}, "
        f"{p['n_test_runs_per_generator']} runs per generator. Cells are the fraction of test "
        'runs in which the intended check fired ("detected" means at least '
        f"{DETECTION_TARGET:.0%}). Both tools' thresholds come from `heldout_real` seeds "
        f"{p['calibration_seeds'][0]} to {p['calibration_seeds'][-1]} with the same rule.\n\n"
        + main_table_markdown(cmp)
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument(
        "--update-readme",
        type=Path,
        default=None,
        help="README whose sdmetrics-comparison markers are replaced with the main table",
    )
    ap.add_argument(
        "--update-full",
        type=Path,
        default=None,
        help="document whose sdmetrics-full markers are replaced with the whole comparison.md",
    )
    args = ap.parse_args(argv)
    cmp = build_comparison(
        load_jsonl(args.results / "runs.jsonl"), load_jsonl(args.results / "sdmetrics_runs.jsonl")
    )
    md = comparison_markdown(cmp)
    (args.results / "comparison.json").write_text(json.dumps(cmp, indent=2) + "\n")
    (args.results / "comparison.md").write_text(md)
    if args.update_readme:
        embed_in_readme(args.update_readme, readme_markdown(cmp), START_MARK, END_MARK)
    if args.update_full:
        embed_in_readme(args.update_full, md, FULL_START_MARK, FULL_END_MARK)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
