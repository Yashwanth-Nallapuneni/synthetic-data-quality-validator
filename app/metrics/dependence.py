"""Pairwise dependence fidelity: how well association structure is preserved.

Associations are computed separately on the real and synthetic data and compared pair by
pair:
- numeric/numeric: Pearson correlation (range -1 to 1, compared as absolute difference)
- categorical/categorical: bias-corrected Cramer's V (Bergsma 2013, range 0 to 1)
- numeric/categorical: correlation ratio eta, adjusted for the number of groups (0 to 1)

Rows with a missing value in either column of a pair are dropped for that pair. A pair
whose association is undefined on either side (constant column, one category) is skipped.
"""

from itertools import combinations

import numpy as np
import pandas as pd

MAX_PAIRS_LISTED = 20
MAX_MATRIX_COLUMNS = 50
MIN_ROWS = 10


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length arrays; NaN when either is constant."""
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    if denom == 0:
        return float("nan")
    return float((xc @ yc) / denom)


def cramers_v_bias_corrected(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramer's V (Bergsma) between two arrays of category labels.

    phi2 = chi2 / n, corrected as max(0, phi2 - (r-1)(k-1)/(n-1)), with r and k replaced by
    r - (r-1)^2/(n-1) and k - (k-1)^2/(n-1). V = sqrt(phi2corr / min(rcorr-1, kcorr-1)).
    NaN when either variable has fewer than 2 categories or n < 2.
    """
    n = len(a)
    if n < 2:
        return float("nan")
    a_codes, a_levels = pd.factorize(a)
    b_codes, b_levels = pd.factorize(b)
    r, k = len(a_levels), len(b_levels)
    if r < 2 or k < 2:
        return float("nan")
    table = np.bincount(a_codes * k + b_codes, minlength=r * k).reshape(r, k).astype(float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2corr / denom))


def correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
    """Correlation ratio eta (numeric by category), adjusted for the number of groups.

    eta^2 = SS_between / SS_total. The adjusted value 1 - (1 - eta^2)(n-1)/(n-g) (floored
    at 0) removes the upward bias that many groups cause at small n. NaN when there is
    one group, no variance or n <= g.
    """
    n = len(values)
    codes, levels = pd.factorize(groups)
    g = len(levels)
    if g < 2 or n <= g:
        return float("nan")
    sizes = np.bincount(codes, minlength=g).astype(float)
    sums = np.bincount(codes, weights=values, minlength=g)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = float((sizes * (sums / sizes - grand) ** 2).sum())
    eta2 = ss_between / ss_total
    adj = max(0.0, 1.0 - (1.0 - eta2) * (n - 1) / (n - g))
    return float(np.sqrt(adj))


def _pair_values(df: pd.DataFrame, a: str, b: str):
    sub = df[[a, b]].dropna()
    return sub[a].to_numpy(), sub[b].to_numpy()


def _numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    m = np.full((len(cols), len(cols)), np.nan)
    for i, j in combinations(range(len(cols)), 2):
        x, y = _pair_values(df, cols[i], cols[j])
        m[i, j] = m[j, i] = pearson(x.astype(float), y.astype(float))
    return m


def _categorical_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    m = np.full((len(cols), len(cols)), np.nan)
    for i, j in combinations(range(len(cols)), 2):
        a, b = _pair_values(df, cols[i], cols[j])
        m[i, j] = m[j, i] = cramers_v_bias_corrected(a, b)
    return m


def _mixed_matrix(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    m = np.full((len(num_cols), len(cat_cols)), np.nan)
    for i, nc in enumerate(num_cols):
        for j, cc in enumerate(cat_cols):
            v, g = _pair_values(df, nc, cc)
            m[i, j] = correlation_ratio(v.astype(float), g)
    return m


def _block(real_m, synth_m, rows, cols, kind, symmetric):
    """Compare one pair of matrices and list its pairs (upper triangle when symmetric)."""
    diff = np.abs(real_m - synth_m)
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1 if symmetric else 0, len(cols)):
            if np.isfinite(diff[i, j]):
                pairs.append(
                    {
                        "a": rows[i],
                        "b": cols[j],
                        "kind": kind,
                        "real": float(real_m[i, j]),
                        "synthetic": float(synth_m[i, j]),
                        "abs_diff": float(diff[i, j]),
                    }
                )
    out = {
        "measure": {
            "numeric_numeric": "pearson",
            "categorical_categorical": "cramers_v_bias_corrected",
            "numeric_categorical": "correlation_ratio_adjusted",
        }[kind],
        "rows": rows,
        "columns": cols,
        "n_pairs": len(pairs),
        "mean_abs_diff": float(np.mean([p["abs_diff"] for p in pairs])) if pairs else None,
        "max_abs_diff": float(max(p["abs_diff"] for p in pairs)) if pairs else None,
        "matrices_omitted": max(len(rows), len(cols)) > MAX_MATRIX_COLUMNS,
        "real": None,
        "synthetic": None,
        "abs_diff": None,
    }
    if not out["matrices_omitted"]:
        out["real"], out["synthetic"], out["abs_diff"] = real_m, synth_m, diff
    return out, pairs


def compute_dependence(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict:
    """Compare pairwise association structure between real and synthetic data.

    Returns status, per-kind blocks (association matrices for real and synthetic plus the
    absolute difference matrix, ready for a heatmap), a `summary` over all pairs (mean and
    max absolute difference and the worst pair) and `worst_pairs`, the largest differences
    capped at MAX_PAIRS_LISTED. The overall mean pools the three kinds, whose scales are
    comparable (all roughly 0 to 1 in magnitude) but not identical.
    """
    numeric_cols, categorical_cols = list(numeric_cols), list(categorical_cols)
    if len(numeric_cols) + len(categorical_cols) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Need at least 2 columns to measure dependence",
        }
    if len(real_df) < MIN_ROWS or len(synth_df) < MIN_ROWS:
        return {
            "status": "insufficient_data",
            "reason": f"Need at least {MIN_ROWS} rows in both datasets",
        }

    blocks = {}
    all_pairs: list[dict] = []

    if len(numeric_cols) >= 2:
        blocks["numeric_numeric"], p = _block(
            _numeric_matrix(real_df, numeric_cols),
            _numeric_matrix(synth_df, numeric_cols),
            numeric_cols,
            numeric_cols,
            "numeric_numeric",
            True,
        )
        all_pairs += p
    if len(categorical_cols) >= 2:
        blocks["categorical_categorical"], p = _block(
            _categorical_matrix(real_df, categorical_cols),
            _categorical_matrix(synth_df, categorical_cols),
            categorical_cols,
            categorical_cols,
            "categorical_categorical",
            True,
        )
        all_pairs += p
    if numeric_cols and categorical_cols:
        blocks["numeric_categorical"], p = _block(
            _mixed_matrix(real_df, numeric_cols, categorical_cols),
            _mixed_matrix(synth_df, numeric_cols, categorical_cols),
            numeric_cols,
            categorical_cols,
            "numeric_categorical",
            False,
        )
        all_pairs += p

    if not all_pairs:
        return {"status": "insufficient_data", "reason": "No column pair has a defined association"}

    all_pairs.sort(key=lambda p: p["abs_diff"], reverse=True)
    worst = all_pairs[0]
    return {
        "status": "ok",
        "reason": None,
        **blocks,
        "summary": {
            "n_pairs": len(all_pairs),
            "mean_abs_diff": float(np.mean([p["abs_diff"] for p in all_pairs])),
            "max_abs_diff": worst["abs_diff"],
            "worst_pair": [worst["a"], worst["b"]],
        },
        "worst_pairs": all_pairs[:MAX_PAIRS_LISTED],
    }
