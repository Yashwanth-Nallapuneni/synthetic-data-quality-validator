"""Warning checks as data: raw statistics in, structured warnings out.

`extract_stats(report)` flattens the numbers every check looks at into a dict keyed by
metric path. `evaluate_checks(stats, thresholds)` turns those numbers into warnings. The
report builder uses both, and the Phase 3 benchmark records the stats so thresholds can be
recalibrated offline and re-applied without re-running any metric.

A warning is {"code", "message", "metric", "value", "threshold"}. The code is a stable
snake_case id, one per kind of check.
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase

from app.thresholds import Thresholds

ABOVE = "above"  # warn when the statistic is greater than the threshold
BELOW = "below"  # warn when the statistic is lower than the threshold


@dataclass(frozen=True)
class Check:
    code: str
    stat: str  # metric path, may contain '*' for per-model utility statistics
    threshold: str | float  # Thresholds field name, or a fixed number
    direction: str
    message: str  # may use {model}, {value}, {threshold}


CHECKS: tuple[Check, ...] = (
    Check(
        "privacy_memorisation",
        "privacy.share_closer_than_holdout_p5",
        "privacy_high_share",
        ABOVE,
        "High memorisation risk: too many synthetic rows are closer to the training rows "
        "than unseen real rows are",
    ),
    Check(
        "privacy_memorisation",
        "privacy.exact_match_excess",
        "privacy_exact_match_margin",
        ABOVE,
        "High memorisation risk: synthetic rows match training rows exactly more often "
        "than unseen real rows do",
    ),
    Check(
        "diversity_mode_collapse",
        "diversity.mean_variance_ratio",
        "variance_ratio_low",
        BELOW,
        "Synthetic data shows signs of mode collapse (low variance ratio)",
    ),
    Check(
        "diversity_excess_variance",
        "diversity.mean_variance_ratio",
        "variance_ratio_high",
        ABOVE,
        "Synthetic data shows excessive variance or instability (high variance ratio)",
    ),
    Check(
        "diversity_low_coverage",
        "diversity.coverage",
        "coverage_min",
        BELOW,
        "Low coverage: parts of the real data have no nearby synthetic row",
    ),
    Check(
        "numeric_marginal_shift",
        "fidelity.numeric.max_ks",
        "numeric_max_ks",
        ABOVE,
        "A numeric column's distribution differs from the real data (max KS statistic)",
    ),
    Check(
        "categorical_mean_l1",
        "fidelity.categorical.mean_l1",
        "categorical_mean_l1",
        ABOVE,
        "Significant categorical distribution shift (mean L1 distance)",
    ),
    Check(
        "categorical_mean_js",
        "fidelity.categorical.mean_js",
        "categorical_mean_js",
        ABOVE,
        "Categorical distributions differ substantially (mean JS distance)",
    ),
    Check(
        "categorical_worst_shift",
        "fidelity.categorical.max_abs_worst_shift",
        "categorical_worst_shift",
        ABOVE,
        "Severe distortion of a single category's share",
    ),
    Check(
        "dependence_shift",
        "dependence.mean_abs_diff",
        "dependence_mean_abs_diff",
        ABOVE,
        "Pairwise dependencies differ from the real data",
    ),
    Check(
        "detectability_high_auc",
        "detectability.auc_mean",
        "c2st_auc",
        ABOVE,
        "Synthetic rows are easily distinguishable from real rows (C2ST AUC)",
    ),
    Check(
        "utility_degradation",
        "utility.*.accuracy_ratio",
        "utility_accuracy_ratio",
        BELOW,
        "Significant downstream utility degradation ({model}, TSTR/TRTR accuracy ratio)",
    ),
    Check(
        "utility_degradation",
        "utility.*.r2_drop",
        "utility_r2_drop",
        ABOVE,
        "Significant downstream utility degradation ({model}, R2 drop)",
    ),
    Check(
        "utility_calibration_degradation",
        "utility.*.ece_increase",
        "utility_ece_increase",
        ABOVE,
        "Synthetic data significantly degrades probability calibration ({model}, ECE increase)",
    ),
    Check(
        "utility_synthetic_single_class",
        "utility.synthetic_single_class",
        0.5,
        ABOVE,
        "Synthetic labels contain a single class",
    ),
)

# Warning codes in a stable order (for matrices and docs).
CODES: tuple[str, ...] = tuple(dict.fromkeys(c.code for c in CHECKS))


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def extract_stats(report: dict) -> dict[str, float | None]:
    """Every raw number the checks use, keyed by metric path (None when not computed)."""
    s: dict[str, float | None] = {}

    p = report.get("privacy") or {}
    if p.get("status") == "ok":
        s["privacy.share_closer_than_holdout_p5"] = _num(p["share_closer_than_holdout_p5"])
        excess = p["exact_match_rate"] - p["holdout_exact_match_rate"]
        s["privacy.exact_match_excess"] = _num(excess)

    d = report.get("diversity") or {}
    s["diversity.mean_variance_ratio"] = _num(d.get("mean_variance_ratio"))
    cov = d.get("coverage")
    s["diversity.coverage"] = _num(cov["value"]) if cov else None

    fid = report.get("fidelity") or {}
    num = fid.get("numeric")
    if num:
        ks = [v["ks_statistic"] for v in num["per_feature"].values()]
        s["fidelity.numeric.max_ks"] = _num(max(ks)) if ks else None
        s["fidelity.numeric.mean_ks"] = _num(num["summary"]["mean_ks"])
        s["fidelity.numeric.mean_js"] = _num(num["summary"]["mean_js"])
    cat = fid.get("categorical")
    if cat:
        shifts = [abs(v["worst_shift"]) for v in cat["per_feature"].values()]
        s["fidelity.categorical.mean_l1"] = _num(cat["summary"]["mean_l1"])
        s["fidelity.categorical.mean_js"] = _num(cat["summary"]["mean_js"])
        s["fidelity.categorical.max_abs_worst_shift"] = _num(max(shifts)) if shifts else None

    dep = report.get("dependence") or {}
    if dep.get("status") == "ok":
        s["dependence.mean_abs_diff"] = _num(dep["summary"]["mean_abs_diff"])

    det = report.get("detectability") or {}
    if det.get("status") == "ok":
        s["detectability.auc_mean"] = _num(det["auc_mean"])

    u = report.get("utility") or {}
    if u.get("status") == "ok":
        if u.get("task") == "classification":
            s["utility.synthetic_single_class"] = 1.0 if u.get("synthetic_single_class") else 0.0
        for name, m in u["models"].items():
            for k, v in m["tstr_vs_trtr"].items():
                s[f"utility.{name}.{k}"] = _num(v)
    return s


def extract_context(report: dict) -> dict[str, str]:
    """Human detail for messages that cannot be rebuilt from the numbers alone."""
    ctx: dict[str, str] = {}
    cat = (report.get("fidelity") or {}).get("categorical")
    if cat and cat["per_feature"]:
        col, st = max(cat["per_feature"].items(), key=lambda kv: abs(kv[1]["worst_shift"]))
        ctx["fidelity.categorical.max_abs_worst_shift"] = (
            f" (feature '{col}', category '{st['worst_category']}')"
        )
    return ctx


def _threshold_value(check: Check, th: Thresholds) -> float:
    if isinstance(check.threshold, str):
        return float(getattr(th, check.threshold))
    return float(check.threshold)


def matching_stats(check: Check, stats: dict) -> list[tuple[str, float]]:
    """(key, value) pairs for the check's statistic, skipping missing values."""
    return [
        (k, v) for k, v in stats.items() if v is not None and fnmatchcase(k, check.stat)
    ]


def evaluate_checks(
    stats: dict[str, float | None], th: Thresholds, context: dict[str, str] | None = None
) -> list[dict]:
    """Structured warnings for the given statistics and thresholds."""
    context = context or {}
    out: list[dict] = []
    for check in CHECKS:
        thr = _threshold_value(check, th)
        for key, value in matching_stats(check, stats):
            fired = value > thr if check.direction == ABOVE else value < thr
            if not fired:
                continue
            model = key.split(".")[1] if "*" in check.stat else ""
            cmp_ = ">" if check.direction == ABOVE else "<"
            msg = check.message.format(model=model) + context.get(key, "")
            out.append(
                {
                    "code": check.code,
                    "message": f"{msg}: {key} = {value:.3f} {cmp_} {thr:.3f}",
                    "metric": key,
                    "value": value,
                    "threshold": thr,
                }
            )
    return out
