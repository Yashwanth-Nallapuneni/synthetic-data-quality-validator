"""Detectability: a classifier two-sample test (C2ST) between real and synthetic rows.

A gradient-boosting classifier is trained to tell real rows (label 0) from synthetic rows
(label 1) and scored by cross-validated ROC AUC. An AUC near 0.5 means the classifier
cannot separate them; values near 1 mean the synthetic rows are easy to spot.

Exact copies of real rows are NOT detected by this test (a copied row is indistinguishable
from its original); that failure is covered by the privacy check.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.utils.preprocessing import build_preprocessor, limit_threads_for_small_data

MIN_ROWS_PER_SIDE = 30
MAX_ROWS_PER_SIDE = 10_000
N_SPLITS = 5


def compute_detectability(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
    max_rows_per_side: int = MAX_ROWS_PER_SIDE,
    n_splits: int = N_SPLITS,
) -> dict:
    """Cross-validated real-vs-synthetic classifier AUC.

    Both sides are subsampled (seeded, without replacement) to the same size, which is the
    smaller of the two row counts capped at `max_rows_per_side`, so the classes are
    balanced and the cost is bounded. The preprocessor (median imputation, standardisation,
    one-hot) lives inside the Pipeline so it is refit on each training fold. Returns the
    mean and standard deviation of the per-fold AUC.
    """
    features = list(numeric_cols) + list(categorical_cols)
    if not features:
        return {"status": "insufficient_data", "reason": "No usable feature columns"}

    n = min(len(real_df), len(synth_df), max_rows_per_side)
    if min(len(real_df), len(synth_df)) < MIN_ROWS_PER_SIDE:
        return {
            "status": "insufficient_data",
            "reason": f"Need at least {MIN_ROWS_PER_SIDE} rows in both real and synthetic data",
        }

    rng = np.random.default_rng(seed)
    real = real_df[features].iloc[np.sort(rng.choice(len(real_df), n, replace=False))]
    synth = synth_df[features].iloc[np.sort(rng.choice(len(synth_df), n, replace=False))]
    X = pd.concat([real, synth], ignore_index=True)
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]

    pipe = Pipeline(
        [
            ("pre", build_preprocessor(list(numeric_cols), list(categorical_cols))),
            ("clf", HistGradientBoostingClassifier(max_iter=100, random_state=seed)),
        ]
    )
    k = max(2, min(n_splits, n // 2))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    with limit_threads_for_small_data(len(X)):
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    return {
        "status": "ok",
        "reason": None,
        "model": "HistGradientBoostingClassifier",
        "auc_mean": float(np.mean(scores)),
        "auc_std": float(np.std(scores)),
        "auc_per_fold": [float(s) for s in scores],
        "n_splits": int(k),
        "n_per_side": int(n),
        "subsampled": bool(n < len(real_df) or n < len(synth_df)),
        "n_real_available": int(len(real_df)),
        "n_synthetic_available": int(len(synth_df)),
    }
