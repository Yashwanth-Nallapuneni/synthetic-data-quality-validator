"""Memorisation risk via distance to closest record (DCR) against a holdout baseline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from app.utils.preprocessing import build_preprocessor, coerce_features

MIN_REAL_ROWS = 20
HOLDOUT_FRACTION = 0.5
BASELINE_QUANTILE = 0.05
HIGH_SHARE = 0.15
MEDIUM_SHARE = 0.10
EXACT_MATCH_MARGIN = 0.05
EXACT_TOL = 1e-9


def _nn_distances(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(train)
    dist, _ = nn.kneighbors(query)
    return dist.ravel()


def _describe(d: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "min": float(np.min(d)),
        "p5": float(np.quantile(d, BASELINE_QUANTILE)),
    }


def compute_privacy_risk(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
) -> dict:
    """Distance-to-closest-record (DCR) privacy check with a holdout baseline.

    Real rows are split into a train half and a holdout half (fixed seed). Numeric
    columns are standardised with statistics from the train half and categorical
    columns are one-hot encoded, so every column contributes on a comparable scale.
    We then compare, for the nearest train row:
    - the distance of each synthetic row, and
    - the distance of each genuinely unseen holdout row.
    If many more synthetic rows than the baseline sit closer than the 5th percentile of
    the holdout distances, the generator is likely reproducing training rows.

    The baseline assumes the generator saw only the train half. A generator trained on
    all real rows still gets flagged when it copies train rows, but the holdout half
    dilutes the signal. The risk thresholds are not yet calibrated.
    """
    features = list(numeric_cols) + list(categorical_cols)
    if not features:
        return {
            "status": "insufficient_data",
            "reason": "No usable feature columns",
            "privacy_risk_level": "UNKNOWN",
        }
    if len(real_df) < MIN_REAL_ROWS or len(synth_df) == 0:
        return {
            "status": "insufficient_data",
            "reason": f"Need at least {MIN_REAL_ROWS} real rows and 1 synthetic row",
            "privacy_risk_level": "UNKNOWN",
        }

    real = coerce_features(real_df, numeric_cols, categorical_cols)
    synth = coerce_features(synth_df, numeric_cols, categorical_cols)

    train, holdout = train_test_split(real, test_size=HOLDOUT_FRACTION, random_state=seed)
    pre = build_preprocessor(list(numeric_cols), list(categorical_cols)).fit(train)
    train_x = pre.transform(train)
    holdout_x = pre.transform(holdout)
    synth_x = pre.transform(synth)

    synth_dcr = _nn_distances(train_x, synth_x)
    holdout_dcr = _nn_distances(train_x, holdout_x)

    threshold = float(np.quantile(holdout_dcr, BASELINE_QUANTILE))
    share_closer = float(np.mean(synth_dcr <= threshold))

    # Exact matches against every real row, with the holdout-vs-train rate as baseline.
    all_x = pre.transform(real)
    exact_rate = float(np.mean(_nn_distances(all_x, synth_x) <= EXACT_TOL))
    holdout_exact_rate = float(np.mean(holdout_dcr <= EXACT_TOL))

    if share_closer > HIGH_SHARE or exact_rate > holdout_exact_rate + EXACT_MATCH_MARGIN:
        level = "HIGH"
    elif share_closer > MEDIUM_SHARE:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "status": "ok",
        "reason": None,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "n_synthetic": int(len(synth)),
        "synthetic_dcr": _describe(synth_dcr),
        "holdout_dcr": _describe(holdout_dcr),
        "dcr_threshold_p5": threshold,
        "share_closer_than_holdout_p5": share_closer,
        "expected_share_without_memorisation": BASELINE_QUANTILE,
        "exact_match_rate": exact_rate,
        "holdout_exact_match_rate": holdout_exact_rate,
        "privacy_risk_level": level,
    }
