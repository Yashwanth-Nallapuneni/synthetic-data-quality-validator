import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def _js_divergence(real, synth, bins=30):
    hist_r, bin_edges = np.histogram(real, bins=bins, density=True)
    hist_s, _ = np.histogram(synth, bins=bin_edges, density=True)

    hist_r += 1e-8
    hist_s += 1e-8

    return jensenshannon(hist_r, hist_s)


def compute_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    Computes distribution similarity metrics between real and synthetic data.
    Assumes both dataframes contain the same numeric feature columns (no labels).
    """

    results = {}

    for col in real_df.columns:
        r = real_df[col].dropna().to_numpy(dtype=float)
        s = synth_df[col].dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(s) == 0:
            continue

        ks_stat, ks_p = ks_2samp(r, s)
        w_dist = wasserstein_distance(r, s)
        js_div = _js_divergence(r, s)

        results[col] = {
            "ks_statistic": float(ks_stat),
            "wasserstein_distance": float(w_dist),
            "js_divergence": float(js_div),
        }

    if not results:
        return None

    summary = {
        "mean_ks": float(np.mean([v["ks_statistic"] for v in results.values()])),
        "mean_js": float(np.mean([v["js_divergence"] for v in results.values()])),
        "worst_feature": max(results, key=lambda k: results[k]["ks_statistic"]),
    }

    return {
        "per_feature": results,
        "summary": summary,
    }
