"""Pure functions that turn a validation report into tables and plot data for the UI.

Nothing here imports Streamlit, so the logic is unit-tested without a running app.
`report` is the `report` object of the API response (see app/schemas.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIMENSIONS = (
    ("fidelity", "Fidelity"),
    ("dependence", "Dependence"),
    ("detectability", "Detectability"),
    ("utility", "Utility"),
    ("privacy", "Privacy"),
    ("diversity", "Diversity"),
)

MAX_CATEGORIES_PLOTTED = 20

PRIVACY_EXPLANATION = (
    "Each synthetic row is compared with its closest row in the training data (distance to "
    "closest record, DCR). The same is done for real rows the generator never saw, which "
    "shows how close an honest new row normally gets. If synthetic rows sit closer to the "
    "training rows than those unseen rows do, the generator may have memorised training "
    "records."
)


def warnings_table(report: dict) -> pd.DataFrame:
    cols = ["code", "message", "value", "threshold"]
    rows = [{c: w.get(c) for c in cols} for w in report.get("warnings", [])]
    return pd.DataFrame(rows, columns=cols)


def numeric_fidelity_table(report: dict) -> pd.DataFrame:
    cols = ["feature", "KS", "Wasserstein (standardised)", "JS"]
    block = (report.get("fidelity") or {}).get("numeric")
    if not block:
        return pd.DataFrame(columns=cols)
    rows = [
        {
            "feature": name,
            "KS": m["ks_statistic"],
            "Wasserstein (standardised)": m["wasserstein_standardised"],
            "JS": m["js_divergence"],
        }
        for name, m in block["per_feature"].items()
    ]
    return pd.DataFrame(rows, columns=cols)


def categorical_fidelity_table(report: dict) -> pd.DataFrame:
    cols = ["feature", "L1", "JS", "worst category", "worst shift"]
    block = (report.get("fidelity") or {}).get("categorical")
    if not block:
        return pd.DataFrame(columns=cols)
    rows = [
        {
            "feature": name,
            "L1": m["l1_distance"],
            "JS": m["js_divergence"],
            "worst category": m["worst_category"],
            "worst shift": m["worst_shift"],
        }
        for name, m in block["per_feature"].items()
    ]
    return pd.DataFrame(rows, columns=cols)


def utility_table(report: dict) -> pd.DataFrame:
    """One row per model and metric: TRTR, TSTR and TSTR minus TRTR."""
    cols = ["model", "metric", "TRTR", "TSTR", "TSTR minus TRTR"]
    util = report.get("utility") or {}
    if util.get("status") != "ok":
        return pd.DataFrame(columns=cols)
    metrics = (
        ["r2", "mae"]
        if util["task"] == "regression"
        else [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "ece",
        ]
    )
    rows = []
    for model, res in util["models"].items():
        for m in metrics:
            a, b = res["trtr"][m], res["tstr"][m]
            rows.append(
                {
                    "model": model,
                    "metric": m,
                    "TRTR": a,
                    "TSTR": b,
                    "TSTR minus TRTR": None if a is None or b is None else b - a,
                }
            )
    return pd.DataFrame(rows, columns=cols)


def privacy_table(report: dict) -> pd.DataFrame:
    """Headline privacy numbers as metric / value / expected-without-memorisation rows."""
    cols = ["metric", "value", "expected without memorisation"]
    p = report.get("privacy") or {}
    if p.get("status") != "ok":
        return pd.DataFrame(columns=cols)
    rows = [
        ("Synthetic to train DCR, median", p["synthetic_dcr"]["median"], None),
        ("Holdout to train DCR, median", p["holdout_dcr"]["median"], None),
        (
            "Share of synthetic rows closer than the holdout 5th percentile",
            p["share_closer_than_holdout_p5"],
            p["expected_share_without_memorisation"],
        ),
        (
            "Closeness AUC",
            p["dcr_closeness_auc"],
            p["expected_closeness_auc_without_memorisation"],
        ),
        ("Exact-match rate, synthetic", p["exact_match_rate"], p["holdout_exact_match_rate"]),
    ]
    return pd.DataFrame(rows, columns=cols)


def diversity_table(report: dict) -> pd.DataFrame:
    cols = ["measure", "value"]
    d = report.get("diversity") or {}
    rows = []
    for name, ratio in (d.get("variance_ratio") or {}).items():
        rows.append((f"Variance ratio: {name}", ratio))
    if d.get("mean_variance_ratio") is not None:
        rows.append(("Mean variance ratio", d["mean_variance_ratio"]))
    if d.get("pca_coverage_ratio") is not None:
        rows.append(("PCA variance ratio", d["pca_coverage_ratio"]))
    cov = d.get("coverage")
    if cov and cov.get("value") is not None:
        rows.append((f"Coverage (k={cov['k']})", cov["value"]))
    return pd.DataFrame(rows, columns=cols)


def detectability_table(report: dict) -> pd.DataFrame:
    cols = ["measure", "value"]
    d = report.get("detectability") or {}
    if d.get("status") != "ok":
        return pd.DataFrame(columns=cols)
    rows = [
        ("AUC, mean over folds", d["auc_mean"]),
        ("AUC, standard deviation", d["auc_std"]),
        ("Rows per side", float(d["n_per_side"])),
    ]
    return pd.DataFrame(rows, columns=cols)


def _dimension_of(metric: str) -> str:
    return metric.split(".", 1)[0]


def dimension_verdicts(report: dict) -> pd.DataFrame:
    """Per-dimension verdict: which checks fired, or whether the dimension was not computed.

    This replaces an aggregate score by design: each dimension is judged on its own checks.
    """
    cols = ["dimension", "verdict", "checks fired"]
    fired: dict[str, list[str]] = {}
    for w in report.get("warnings", []):
        fired.setdefault(_dimension_of(w["metric"]), []).append(w["code"])
    rows = []
    for key, label in DIMENSIONS:
        section = report.get(key) or {}
        codes = list(dict.fromkeys(fired.get(key, [])))
        if codes:
            verdict = "warn"
        elif section.get("status") == "insufficient_data":
            verdict = "not computed"
        else:
            verdict = "pass"
        rows.append({"dimension": label, "verdict": verdict, "checks fired": ", ".join(codes)})
    return pd.DataFrame(rows, columns=cols)


def column_kind(report: dict, name: str, series: pd.Series) -> str:
    """'numeric' or 'categorical' for plotting, using the report's own column typing."""
    fid = report.get("fidelity") or {}
    if name in ((fid.get("numeric") or {}).get("per_feature") or {}):
        return "numeric"
    if name in ((fid.get("categorical") or {}).get("per_feature") or {}):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 20:
        return "numeric"
    return "categorical"


def histogram_frame(real: pd.Series, synth: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Density per bin for both samples on shared bin edges (long format for plotting)."""
    r = pd.to_numeric(real, errors="coerce").dropna().to_numpy(float)
    s = pd.to_numeric(synth, errors="coerce").dropna().to_numpy(float)
    cols = ["bin_start", "bin_end", "dataset", "density"]
    if len(r) == 0 and len(s) == 0:
        return pd.DataFrame(columns=cols)
    both = np.concatenate([r, s])
    lo, hi = float(both.min()), float(both.max())
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    rows = []
    for name, x in (("real", r), ("synthetic", s)):
        if len(x) == 0:
            continue
        dens, _ = np.histogram(x, bins=edges)
        dens = dens / len(x)
        for i, d in enumerate(dens):
            rows.append(
                {"bin_start": edges[i], "bin_end": edges[i + 1], "dataset": name, "density": d}
            )
    return pd.DataFrame(rows, columns=cols)


def category_share_frame(
    real: pd.Series, synth: pd.Series, max_categories: int = MAX_CATEGORIES_PLOTTED
) -> pd.DataFrame:
    """Share of rows per category for both samples, limited to the most common in real."""
    r = real.dropna().astype(str)
    s = synth.dropna().astype(str)
    cols = ["category", "dataset", "share"]
    if r.empty and s.empty:
        return pd.DataFrame(columns=cols)
    order = list(r.value_counts().index[:max_categories])
    extra = [c for c in s.value_counts().index if c not in order]
    order += extra[: max(0, max_categories - len(order))]
    rows = []
    for name, x in (("real", r), ("synthetic", s)):
        if x.empty:
            continue
        shares = x.value_counts(normalize=True)
        for c in order:
            rows.append({"category": c, "dataset": name, "share": float(shares.get(c, 0.0))})
    return pd.DataFrame(rows, columns=cols)


def dependence_heatmap_frame(report: dict) -> pd.DataFrame:
    """Long-format absolute association differences (x, y, abs_diff) over all column pairs.

    Empty when dependence was not computed or the matrices were left out (very wide data).
    """
    cols = ["x", "y", "abs_diff"]
    dep = report.get("dependence") or {}
    if dep.get("status") != "ok":
        return pd.DataFrame(columns=cols)
    cells: dict[tuple[str, str], float] = {}
    for kind in ("numeric_numeric", "categorical_categorical", "numeric_categorical"):
        block = dep.get(kind)
        if not block or block.get("abs_diff") is None:
            continue
        for i, row in enumerate(block["rows"]):
            for j, col in enumerate(block["columns"]):
                v = block["abs_diff"][i][j]
                if v is None or row == col:
                    continue
                cells[(row, col)] = v
                cells[(col, row)] = v
    rows = [{"x": x, "y": y, "abs_diff": v} for (x, y), v in cells.items()]
    return pd.DataFrame(rows, columns=cols)
