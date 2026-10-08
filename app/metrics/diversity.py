"""Diversity: variance ratio, PCA variance ratio and k-NN coverage of the real data."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from app.utils.preprocessing import build_preprocessor

COVERAGE_K = 5
MAX_COVERAGE_QUERIES = 5000
MAX_PCA_COMPONENTS = 10
MIN_ROWS = 2


def _variance_ratios(real_num: pd.DataFrame, synth_num: pd.DataFrame, cols: list[str]):
    var_real, var_synth = real_num.var(), synth_num.var()
    ratios = {}
    for col in cols:
        if np.isfinite(var_real[col]) and var_real[col] > 0 and np.isfinite(var_synth[col]):
            ratios[col] = float(var_synth[col] / var_real[col])
    return ratios


def _pca_variance_ratio(real_x: np.ndarray, synth_x: np.ndarray) -> float | None:
    """Synthetic variance over real variance in the real data's top principal components."""
    n_comp = min(MAX_PCA_COMPONENTS, real_x.shape[1], real_x.shape[0])
    if n_comp < 1:
        return None
    pca = PCA(n_components=n_comp).fit(real_x)
    real_var = float(np.sum(pca.explained_variance_))
    if real_var <= 0:
        return None
    synth_var = float(np.var(pca.transform(synth_x), axis=0, ddof=1).sum())
    return synth_var / real_var


def knn_coverage(
    real_x: np.ndarray,
    synth_x: np.ndarray,
    k: int = COVERAGE_K,
    max_queries: int = MAX_COVERAGE_QUERIES,
    seed: int = 42,
) -> dict:
    """Fraction of real rows whose nearest synthetic row is inside the row's own k-NN radius.

    For a real row, the radius is its distance to its k-th nearest OTHER real row. The row
    is covered when the nearest synthetic row is at most that far away. Rows are queried
    against the full real and synthetic sets; only the set of query rows is subsampled
    (seeded) above `max_queries`, which keeps the estimate unbiased and memory bounded.
    """
    if len(real_x) <= k or len(synth_x) == 0:
        return {"value": None, "k": k, "n_real_queried": 0, "subsampled": False}
    rng = np.random.default_rng(seed)
    n_q = min(len(real_x), max_queries)
    idx = np.sort(rng.choice(len(real_x), n_q, replace=False)) if n_q < len(real_x) else None
    queries = real_x if idx is None else real_x[idx]

    # k + 1 neighbours because each query row is in the real index (itself at distance 0).
    real_nn = NearestNeighbors(n_neighbors=k + 1).fit(real_x)
    radius = real_nn.kneighbors(queries)[0][:, k]
    synth_nn = NearestNeighbors(n_neighbors=1).fit(synth_x)
    nearest_synth = synth_nn.kneighbors(queries)[0][:, 0]
    return {
        "value": float(np.mean(nearest_synth <= radius)),
        "k": k,
        "n_real_queried": int(n_q),
        "subsampled": bool(idx is not None),
    }


def compute_diversity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Measure how much of the real data's variety the synthetic data reproduces.

    Column types come from the caller (app.utils.column_types.detect_column_types).
    - variance_ratio: per numeric column, synthetic variance over real variance
    - pca_coverage_ratio: synthetic variance over real variance within the real data's top
      principal components (up to 10), after the real-fitted preprocessor (median impute,
      standardise numeric columns, one-hot categoricals)
    - coverage: fraction of real rows with a synthetic row inside their own k-NN radius
      (k=5) in the same preprocessed space; low values mean parts of the real data have no
      synthetic counterpart (for example mode collapse)
    """
    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols or [])
    out = {
        "variance_ratio": None,
        "mean_variance_ratio": None,
        "pca_coverage_ratio": None,
        "coverage": None,
    }
    if not numeric_cols and not categorical_cols:
        return out
    if len(real_df) < MIN_ROWS or len(synth_df) < MIN_ROWS:
        return out

    if numeric_cols:
        ratios = _variance_ratios(
            real_df[numeric_cols].astype(float), synth_df[numeric_cols].astype(float), numeric_cols
        )
        if ratios:
            out["variance_ratio"] = ratios
            out["mean_variance_ratio"] = float(np.mean(list(ratios.values())))

    pre = build_preprocessor(numeric_cols, categorical_cols).fit(real_df)
    real_x = pre.transform(real_df)
    synth_x = pre.transform(synth_df)
    out["pca_coverage_ratio"] = _pca_variance_ratio(real_x, synth_x)
    out["coverage"] = knn_coverage(real_x, synth_x, seed=seed)
    return out
