"""Phase 3 calibration and evaluation.

    python -m experiments.calibrate [--results experiments/results]

Reads the raw runs written by experiments.benchmark and:

1. Splits seeds into a CALIBRATION set (seeds 0 to 19, all of which were seen in the v1
   protocol) and a TEST set (seeds 20 to 39, never looked at before the v2 design was fixed),
   across all datasets.
2. Learns every warning threshold from the heldout_real runs of the calibration half only.
   The broken generators and the Gaussian copula play no part in choosing thresholds.
3. Re-applies the thresholds to the recorded statistics of the TEST half and writes the
   detection matrix, false-alarm rates and a summary.
4. Also reports a leave-one-dataset-out (LODO) variant: thresholds learned from the
   heldout_real runs of the other two datasets, scored on the held-out dataset's test seeds.

Threshold rule. heldout_real is the null (the synthetic data is genuinely unseen real data).
For a check that warns when the statistic is ABOVE its threshold:
    threshold = max_null + max(0.10 * |max_null|, 0.01)
and for a check that warns when the statistic is BELOW its threshold:
    threshold = min_null - max(0.10 * |min_null|, 0.01)
where max_null / min_null are taken over every calibration null value of the statistic
(pooled over datasets, seeds and, for utility, models). With n exchangeable null values the
chance that a fresh null value lands beyond the observed maximum is 1 / (n + 1), so the
target per-check false-alarm rate is at most 1 / (n + 1) before the margin (n is recorded per
threshold in the output). The margin only lowers it. Several checks run per report, so the
overall any-warning false-alarm rate is higher than any single check's.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from app.checks import ABOVE, CHECKS, CODES, evaluate_checks, matching_stats
from app.thresholds import CALIBRATED_PATH, Thresholds, load_thresholds
from experiments.generators import BROKEN, HELDOUT

RESULTS = Path(__file__).resolve().parent / "results"
DOCS = Path(__file__).resolve().parent.parent / "docs"
VALIDATION_DOC = DOCS / "VALIDATION.md"
METRICS_DOC = DOCS / "METRICS.md"
REL_MARGIN = 0.10
ABS_MARGIN = 0.01
N_CAL_SEEDS = 20
DETECTION_TARGET = 0.9
NULL = HELDOUT

# Failure mode -> warning codes that count as the intended detection.
INTENDED = {
    "exact_copy": ("privacy_memorisation",),
    "near_copy": ("privacy_memorisation",),
    "near_copy_5pct": ("privacy_memorisation",),
    "column_shuffle": ("dependence_shift", "detectability_high_auc"),
    "mode_collapse": ("diversity_mode_collapse", "diversity_low_coverage"),
    "category_oversample": (
        "categorical_mean_l1",
        "categorical_mean_js",
        "categorical_worst_shift",
    ),
    "label_shuffle": ("utility_degradation",),
}
# Failure modes counted in the headline (near_copy_5pct is a stronger variant of near_copy).
HEADLINE_MODES = (
    "exact_copy",
    "near_copy",
    "column_shuffle",
    "mode_collapse",
    "category_oversample",
    "label_shuffle",
)
ORDER = (HELDOUT, "gaussian_copula", *BROKEN)


def load_runs(results: Path) -> list[dict[str, Any]]:
    with open(results / "runs.jsonl") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    bad = [r for r in runs if r["error"]]
    if bad:
        raise SystemExit(f"{len(bad)} benchmark runs failed, first: {bad[0]['error']}")
    return runs


def split_runs(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(calibration runs on seeds below N_CAL_SEEDS, test runs on the remaining seeds)."""
    return [r for r in runs if r["seed"] < N_CAL_SEEDS], [
        r for r in runs if r["seed"] >= N_CAL_SEEDS
    ]


def calibrate_thresholds(null_runs: list[dict[str, Any]]) -> tuple[Thresholds, dict[str, Any]]:
    """Learn thresholds from null runs only. Returns (thresholds, per-threshold details)."""
    if any(r["generator"] != NULL for r in null_runs):
        raise ValueError("calibration may only use heldout_real runs")
    values: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for check in CHECKS:
        if not isinstance(check.threshold, str):
            continue
        pool = [v for r in null_runs for _, v in matching_stats(check, r["stats"])]
        info = {
            "code": check.code,
            "stat": check.stat,
            "direction": check.direction,
            "provisional_default": getattr(Thresholds(), check.threshold),
            "n_null_values": len(pool),
        }
        if not pool:
            info["note"] = "no null values recorded, provisional default kept"
            details[check.threshold] = info
            continue
        if check.direction == ABOVE:
            extreme = max(pool)
            thr = extreme + max(REL_MARGIN * abs(extreme), ABS_MARGIN)
        else:
            extreme = min(pool)
            thr = extreme - max(REL_MARGIN * abs(extreme), ABS_MARGIN)
        values[check.threshold] = round(thr, 6)
        info.update(
            {
                "null_extreme": extreme,
                "threshold": round(thr, 6),
                "max_per_check_false_alarm_rate_before_margin": 1.0 / (len(pool) + 1),
            }
        )
        details[check.threshold] = info
    if "privacy_high_share" in values:
        # The MEDIUM risk level (display only, no warning): above every null run but not yet
        # past the margin that triggers HIGH.
        values["privacy_medium_share"] = round(details["privacy_high_share"]["null_extreme"], 6)
        details["privacy_medium_share"] = {
            "rule": "max_null of the privacy share statistic (no margin); display only",
            "threshold": values["privacy_medium_share"],
            "provisional_default": Thresholds().privacy_medium_share,
        }
    return Thresholds(**values), details


def fired_codes(run: dict[str, Any], th: Thresholds) -> set[str]:
    return {w["code"] for w in evaluate_checks(run["stats"], th)}


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def detection_table(
    runs: list[dict[str, Any]], th: Thresholds | dict[str, Thresholds]
) -> dict[str, Any]:
    """Per generator: n, hit rate per code, intended-fired rate and any-warning rate.

    `th` is one Thresholds for every run, or a dict giving the thresholds per dataset name.
    """
    table: dict[str, dict[str, Any]] = {}
    for gen in ORDER:
        rs = [r for r in runs if r["generator"] == gen]
        if not rs:
            continue
        fired = [fired_codes(r, th[r["dataset"]] if isinstance(th, dict) else th) for r in rs]
        intended = INTENDED.get(gen)
        table[gen] = {
            "n_runs": len(rs),
            "code_hit_rate": {c: _rate([c in f for f in fired]) for c in CODES},
            "intended_codes": list(intended) if intended else None,
            "intended_fired": (
                _rate([bool(set(intended) & f) for f in fired]) if intended else None
            ),
            "any_warning": _rate([bool(f) for f in fired]),
            "n_any_warning": sum(bool(f) for f in fired),
        }
    return table


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def matrix_markdown(table: dict[str, Any]) -> str:
    shown = [c for c in CODES if any(t["code_hit_rate"][c] for t in table.values())]
    head = ["generator", "n", *shown, "intended warning fired", "any warning fired"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for gen, t in table.items():
        row = [f"`{gen}`", str(t["n_runs"])]
        row += [_pct(t["code_hit_rate"][c]) for c in shown]
        row += [_pct(t["intended_fired"]), _pct(t["any_warning"])]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def null_false_alarm_markdown(per_code: dict[str, Any]) -> str:
    lines = ["| warning code | k of n | rate |", "|---|---|---|"]
    for c, v in per_code.items():
        lines.append(f"| `{c}` | {v['k']} of {v['n']} | {v['rate']:.3f} |")
    return "\n".join(lines)


def per_dataset(runs: list[dict[str, Any]], th: Thresholds, datasets: list[str]) -> dict[str, Any]:
    return {d: detection_table([r for r in runs if r["dataset"] == d], th) for d in datasets}


def per_dataset_markdown(by_ds: dict[str, Any]) -> str:
    ds = list(by_ds)
    head = ["generator", *[f"{d}: intended / any" for d in ds]]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for gen in ORDER:
        cells = [f"`{gen}`"]
        for d in ds:
            t = by_ds[d].get(gen)
            cells.append(
                "-" if t is None else f"{_pct(t['intended_fired'])} / {_pct(t['any_warning'])}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(table: dict[str, Any]) -> dict[str, Any]:
    modes = {}
    for m in HEADLINE_MODES:
        rate = table[m]["intended_fired"]
        modes[m] = {
            "intended_codes": INTENDED[m],
            "intended_fired_rate": rate,
            "detected": rate >= DETECTION_TARGET,
        }
    null, copula = table[NULL], table["gaussian_copula"]
    n_det, n_modes = sum(v["detected"] for v in modes.values()), len(HEADLINE_MODES)
    return {
        "headline": (
            f"detects {n_det} of {n_modes} injected failure modes (intended warning in at least "
            f"{DETECTION_TARGET:.0%} of held-out runs) with {null['n_any_warning']} of "
            f"{null['n_runs']} held-out-real runs raising any warning"
        ),
        "detection_rule": f"intended warning fires in at least {DETECTION_TARGET:.0%} of test runs",
        "n_failure_modes": len(HEADLINE_MODES),
        "n_detected": sum(v["detected"] for v in modes.values()),
        "failure_modes": modes,
        "variant_near_copy_5pct_intended_fired_rate": table["near_copy_5pct"]["intended_fired"],
        "heldout_real_false_alarm": {
            "k": null["n_any_warning"],
            "n": null["n_runs"],
            "rate": null["any_warning"],
            "per_code": {
                c: {"k": round(r * null["n_runs"]), "n": null["n_runs"], "rate": r}
                for c, r in null["code_hit_rate"].items()
            },
        },
        "gaussian_copula_any_warning": {
            "k": copula["n_any_warning"],
            "n": copula["n_runs"],
            "rate": copula["any_warning"],
            "note": "A copula legitimately differs from real data, so these are not strictly "
            "false alarms.",
        },
    }


def lodo_table(runs: list[dict[str, Any]], datasets: list[str]) -> dict[str, Any]:
    """Detection table where each test run is scored with thresholds learned on the other
    datasets' heldout_real calibration-seed runs, so its own dataset never informs its
    thresholds."""
    ths = {}
    for d in datasets:
        null = [
            r
            for r in runs
            if r["generator"] == NULL and r["dataset"] != d and r["seed"] < N_CAL_SEEDS
        ]
        ths[d] = calibrate_thresholds(null)[0]
    test = [r for r in runs if r["seed"] >= N_CAL_SEEDS]
    return {
        "table": detection_table(test, ths),
        "thresholds": {d: t.to_dict() for d, t in ths.items()},
    }


START_MARK = "<!-- detection-matrix:start -->"
END_MARK = "<!-- detection-matrix:end -->"


def embed_in_readme(readme: Path, text: str, start: str = START_MARK, end: str = END_MARK) -> None:
    """Replace the text between the `start` and `end` markers in `readme` with `text`."""
    body = readme.read_text()
    a, b = body.index(start) + len(start), body.index(end)
    readme.write_text(body[:a] + "\n\n" + text.strip("\n") + "\n\n" + body[b:])


WARN_START_MARK = "<!-- warnings-table:start -->"
WARN_END_MARK = "<!-- warnings-table:end -->"

# (warning code, what makes it fire, threshold keys it references). The prose names keys only;
# every number in the README table comes from Thresholds() and the calibrated JSON.
WARNING_ROWS: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "privacy_memorisation",
        "any of: the share of synthetic rows closer to train than the holdout 5th percentile is "
        "above `privacy_high_share`; the exact-match rate exceeds the holdout rate by more than "
        "`privacy_exact_match_margin`; the closeness AUC (P(synthetic DCR < holdout DCR) with "
        "ties counted half) is above `privacy_closeness_auc`",
        ("privacy_high_share", "privacy_exact_match_margin", "privacy_closeness_auc"),
    ),
    (
        "diversity_mode_collapse",
        "mean variance ratio is below `variance_ratio_low`",
        ("variance_ratio_low",),
    ),
    (
        "diversity_excess_variance",
        "mean variance ratio is above `variance_ratio_high`",
        ("variance_ratio_high",),
    ),
    (
        "diversity_low_coverage",
        "k-NN coverage is below `coverage_min`",
        ("coverage_min",),
    ),
    (
        "numeric_marginal_shift",
        "the largest per-column KS statistic is above `numeric_max_ks`",
        ("numeric_max_ks",),
    ),
    (
        "categorical_mean_l1",
        "mean categorical L1 is above `categorical_mean_l1`",
        ("categorical_mean_l1",),
    ),
    (
        "categorical_mean_js",
        "mean categorical JS is above `categorical_mean_js`",
        ("categorical_mean_js",),
    ),
    (
        "categorical_worst_shift",
        "the largest single-category share shift is above `categorical_worst_shift`",
        ("categorical_worst_shift",),
    ),
    (
        "dependence_shift",
        "the mean absolute dependence difference is above `dependence_mean_abs_diff`",
        ("dependence_mean_abs_diff",),
    ),
    (
        "detectability_high_auc",
        "the C2ST AUC is above `c2st_auc`",
        ("c2st_auc",),
    ),
    (
        "utility_degradation",
        "for any model, TSTR balanced accuracy is below `utility_balanced_accuracy_ratio` times "
        "TRTR balanced accuracy (balanced accuracy has chance level 1/k whatever the class "
        "balance, so a majority-class predictor does not hide label damage), or the TSTR R2 drop "
        "is above `utility_r2_drop`. The model is named in the message",
        ("utility_balanced_accuracy_ratio", "utility_r2_drop"),
    ),
    (
        "utility_calibration_degradation",
        "for any model, TSTR ECE is more than `utility_ece_increase` above TRTR ECE",
        ("utility_ece_increase",),
    ),
    (
        "utility_synthetic_single_class",
        "synthetic labels contain a single class",
        (),
    ),
]


def _sig(x: float) -> str:
    return f"{x:.4g}"


def warnings_table_markdown(calibrated_path: Path = CALIBRATED_PATH) -> str:
    """The warnings table for docs/METRICS.md, values read from the code and calibrated JSON."""
    provisional = Thresholds()
    calibrated = load_thresholds(calibrated_path)
    meta = json.loads(Path(calibrated_path).read_text()).get("calibration", {})
    not_calibrated = set(meta.get("not_calibrated", []))

    def cell(keys: tuple[str, ...], th: Thresholds, mark: bool) -> str:
        parts = []
        for k in keys:
            note = " (not calibrated)" if mark and k in not_calibrated else ""
            parts.append(f"`{k}`: {_sig(getattr(th, k))}{note}")
        return "<br>".join(parts) if parts else "n/a"

    lines = [
        "| Code | Fires when | Calibrated default | Provisional fallback |",
        "|---|---|---|---|",
    ]
    for code, fires, keys in WARNING_ROWS:
        lines.append(
            f"| `{code}` | {fires} | {cell(keys, calibrated, True)} "
            f"| {cell(keys, provisional, False)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--thresholds-out", type=Path, default=CALIBRATED_PATH)
    ap.add_argument(
        "--update-readme",
        type=Path,
        nargs="?",
        const=VALIDATION_DOC,
        default=None,
        help="document whose detection-matrix markers are replaced with detection_matrix.md "
        "(default when given without a path: docs/VALIDATION.md)",
    )
    ap.add_argument(
        "--update-warnings-table",
        type=Path,
        nargs="?",
        const=METRICS_DOC,
        default=None,
        help="document whose warnings-table markers are replaced with the threshold table "
        "(default when given without a path: docs/METRICS.md); "
        "needs no benchmark results, only the calibrated JSON",
    )
    args = ap.parse_args(argv)

    if args.update_warnings_table:
        embed_in_readme(
            args.update_warnings_table, warnings_table_markdown(), WARN_START_MARK, WARN_END_MARK
        )
        return 0

    runs = load_runs(args.results)
    meta = json.loads((args.results / "meta.json").read_text())
    datasets = [d["name"] for d in meta["datasets"]]
    cal, test = split_runs(runs)

    null_cal = [r for r in cal if r["generator"] == NULL]
    th, details = calibrate_thresholds(null_cal)
    cal_seeds = sorted({r["seed"] for r in cal})
    test_seeds = sorted({r["seed"] for r in test})

    calibration = {
        "rule": (
            "From heldout_real runs of the calibration seeds only. Warn-above checks: "
            f"threshold = max_null + max({REL_MARGIN} * |max_null|, {ABS_MARGIN}). "
            f"Warn-below checks: threshold = min_null - max({REL_MARGIN} * |min_null|, "
            f"{ABS_MARGIN}). Null values are pooled over datasets, seeds and models. Target "
            "per-check false-alarm rate on fresh null data: at most 1 / (n_null_values + 1) "
            "before the margin."
        ),
        "date": datetime.date.today().isoformat(),
        "calibration_seeds": cal_seeds,
        "test_seeds": test_seeds,
        "datasets": datasets,
        "n_null_runs": len(null_cal),
        "generators_used_for_calibration": [NULL],
        "source": "experiments/results/runs.jsonl (protocol v2)",
        "per_threshold": details,
        "not_calibrated": sorted(
            set(Thresholds().to_dict()) - {k for k, v in details.items() if "threshold" in v}
        ),
    }
    args.thresholds_out.write_text(
        json.dumps({"thresholds": th.to_dict(), "calibration": calibration}, indent=2) + "\n"
    )

    table = detection_table(test, th)
    by_ds = per_dataset(test, th, datasets)
    summary = {
        "n_test_runs": len(test),
        "test_seeds": test_seeds,
        "calibration_seeds": cal_seeds,
        "datasets": datasets,
        **headline(table),
    }
    lo = lodo_table(runs, datasets)
    summary["leave_one_dataset_out"] = {
        "headline": headline(lo["table"]),
        "thresholds_per_held_out_dataset": lo["thresholds"],
    }

    out = args.results
    (out / "detection_matrix.json").write_text(
        json.dumps({"overall": table, "per_dataset": by_ds}, indent=2) + "\n"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    h = summary
    fa = h["heldout_real_false_alarm"]
    cp = h["gaussian_copula_any_warning"]
    md = [
        f"Test set: seeds {test_seeds[0]} to {test_seeds[-1]} on {len(datasets)} datasets "
        f"({table[NULL]['n_runs']} runs per generator). Cells are the fraction of runs in which "
        "the warning fired. Columns for codes that never fired are omitted.",
        "",
        matrix_markdown(table),
        "",
        f"False-alarm rate on `{NULL}` (any warning): {fa['k']} of {fa['n']} test runs "
        f"({fa['rate']:.3f}).",
        "",
        f"Warnings on `gaussian_copula` (any warning): {cp['k']} of {cp['n']} test runs "
        f"({cp['rate']:.3f}). A copula legitimately differs from real data, so these are not "
        "strictly false alarms.",
        "",
        f"Failure modes whose intended warning fires in at least {DETECTION_TARGET:.0%} of test "
        f"runs: {h['n_detected']} of {h['n_failure_modes']}.",
        "",
        f"Headline: {h['headline']}.",
        "",
        f"False-alarm rate of each check on the `{NULL}` test runs (k of {fa['n']} runs):",
        "",
        null_false_alarm_markdown(fa["per_code"]),
        "",
    ]
    md_extra = [
        "Per dataset (intended warning fired / any warning fired, "
        f"{len(test_seeds)} test runs per cell):",
        "",
        per_dataset_markdown(by_ds),
        "",
        "Leave-one-dataset-out (thresholds learned from the other two datasets, scored on the "
        "held-out dataset's test seeds):",
        "",
        matrix_markdown(lo["table"]),
    ]
    lh = summary["leave_one_dataset_out"]["headline"]
    md_extra += [
        "",
        f"LODO false-alarm rate on `{NULL}`: {lh['heldout_real_false_alarm']['k']} of "
        f"{lh['heldout_real_false_alarm']['n']} runs. Failure modes detected in at least "
        f"{DETECTION_TARGET:.0%} of runs: {lh['n_detected']} of {lh['n_failure_modes']}.",
        "",
    ]
    (out / "detection_matrix.md").write_text("\n".join(md))
    (out / "detection_matrix_details.md").write_text("\n".join(md_extra))
    if args.update_readme:
        embed_in_readme(args.update_readme, (out / "detection_matrix.md").read_text())
    print("\n".join(md + md_extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
