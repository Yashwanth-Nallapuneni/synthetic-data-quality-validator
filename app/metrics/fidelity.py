"""Numeric marginal fidelity: KS, standardised Wasserstein and quantile-bin JS."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

N_QUANTILE_BINS = 10


def quantile_js_distance(real: np.ndarray, synth: np.ndarray, n_bins: int = N_QUANTILE_BINS):
    """Jensen-Shannon distance (base 2, range 0 to 1) on bins built from the real data.

    The bin edges are the real column's quantiles (n_bins bins holding roughly equal real
    mass), shared by both samples, with open-ended first and last bins so no synthetic
    value falls outside. Counts are normalised directly, with no epsilon added. Because
    the real mass per bin is fixed by construction, the value does not depend on the
    column's scale and a fresh sample from the same distribution scores near 0. Sampling
    noise still gives a small positive bias that shrinks roughly as 1/sqrt(n). Because
    the end bins are open, the maximum is below 1 when the supports are disjoint.
    """
    inner = np.unique(np.quantile(real, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    real_counts = np.bincount(np.searchsorted(inner, real, side="left"), minlength=len(inner) + 1)
    synth_counts = np.bincount(np.searchsorted(inner, synth, side="left"), minlength=len(inner) + 1)
    p = real_counts / real_counts.sum()
    q = synth_counts / synth_counts.sum()
    return float(jensenshannon(p, q, base=2))


def standardised_wasserstein(real: np.ndarray, synth: np.ndarray):
    """1-Wasserstein distance divided by the standard deviation of the real column.

    The result is in units of real standard deviations, so it is comparable across
    columns. None when the real column is constant (the scale is undefined).
    """
    std = float(np.std(real))
    if std == 0.0:
        return None
    return float(wasserstein_distance(real, synth) / std)


def compute_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    Computes distribution similarity metrics between real and synthetic numeric columns.
    Assumes both dataframes contain the same numeric feature columns.

    Per column:
    - ks_statistic: two-sample Kolmogorov-Smirnov statistic (0 to 1). Only the statistic
      is used, so the asymptotic method is requested; it avoids scipy's exact p-value
      computation and its "exact calculation unsuccessful" warning on large samples.
    - wasserstein_distance: raw 1-Wasserstein distance in the column's own units
    - wasserstein_standardised: the raw distance divided by the real column's std
    - js_divergence: Jensen-Shannon distance on real-quantile bins (see quantile_js_distance)
    """

    results = {}

    for col in real_df.columns:
        r = real_df[col].dropna().to_numpy(dtype=float)
        s = synth_df[col].dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(s) == 0:
            continue

        ks_stat = ks_2samp(r, s, method="asymp").statistic

        results[col] = {
            "ks_statistic": float(ks_stat),
            "wasserstein_distance": float(wasserstein_distance(r, s)),
            "wasserstein_standardised": standardised_wasserstein(r, s),
            "js_divergence": quantile_js_distance(r, s),
        }

    if not results:
        return None

    standardised = [
        v["wasserstein_standardised"]
        for v in results.values()
        if v["wasserstein_standardised"] is not None
    ]
    summary = {
        "mean_ks": float(np.mean([v["ks_statistic"] for v in results.values()])),
        "mean_js": float(np.mean([v["js_divergence"] for v in results.values()])),
        "mean_wasserstein_standardised": float(np.mean(standardised)) if standardised else None,
        "worst_feature": max(results, key=lambda k: results[k]["ks_statistic"]),
    }

    return {
        "per_feature": results,
        "summary": summary,
    }
