"""Memorisation risk via distance to closest record (DCR) against a holdout baseline."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from app.thresholds import Thresholds, load_thresholds
from app.utils.preprocessing import build_preprocessor, coerce_features

MIN_REAL_ROWS = 20
MAX_QUERY_ROWS = 20_000
HOLDOUT_FRACTION = 0.5
BASELINE_QUANTILE = 0.05
EXACT_TOL = 1e-9


def _nn_distances(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(train)
    dist, _ = nn.kneighbors(query)
    return dist.ravel()


def closeness_auc(synth_dcr: np.ndarray, holdout_dcr: np.ndarray) -> float:
    """P(synthetic DCR < holdout DCR) + 0.5 * P(equal), from ranks (Mann-Whitney U).

    It is 0.5 when synthetic rows are as far from the training rows as unseen real rows
    are, and above 0.5 when they sit closer. Ties count one half, so the statistic stays
    meaningful when many distances are exactly 0 (duplicate rows), where a quantile cut-off
    on the holdout distances degenerates.
    """
    n1, n2 = len(synth_dcr), len(holdout_dcr)
    ranks = rankdata(np.concatenate([synth_dcr, holdout_dcr]))  # average ranks for ties
    u_synth = ranks[:n1].sum() - n1 * (n1 + 1) / 2  # pairs where synth is farther (ties 0.5)
    return float(1.0 - u_synth / (n1 * n2))


def _describe(d: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "min": float(np.min(d)),
        "p5": float(np.quantile(d, BASELINE_QUANTILE)),
    }


def _cap(x: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded subsample of query rows (no replacement) when above the cap."""
    if len(x) <= max_rows:
        return x
    return x[np.sort(rng.choice(len(x), max_rows, replace=False))]


def compute_privacy_risk(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
    thresholds: Thresholds | None = None,
    holdout_df: pd.DataFrame | None = None,
    max_query_rows: int = MAX_QUERY_ROWS,
) -> dict:
    """Distance-to-closest-record (DCR) privacy check with a holdout baseline.

    Two modes:
    - holdout_df given: `real_df` is the generator's training set and `holdout_df` is real
      data the generator never saw. This is the sharp version of the test.
    - holdout_df omitted: real rows are split 50/50 (seeded) into a train half and a
      holdout half. If the generator was actually trained on all of `real_df`, the holdout
      half is not unseen, which dilutes the signal.

    Numeric columns are standardised with train statistics and categorical columns are
    one-hot encoded, so every column contributes on a comparable scale. We compare, for
    the nearest train row, the distance of each synthetic row (synthetic_dcr) with the
    distance of each genuinely unseen holdout row (holdout_dcr). If many more synthetic
    rows than the baseline 5% sit closer than the 5th percentile of holdout DCR, the
    generator is likely reproducing training rows. The exact-match rate is the share of
    synthetic rows identical (within 1e-9 after preprocessing) to a training row.

    Query sets above `max_query_rows` are subsampled (seeded) and the result records it.
    The risk thresholds are provisional.
    """
    th = thresholds or load_thresholds()
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
    if holdout_df is not None and len(holdout_df) < MIN_REAL_ROWS:
        return {
            "status": "insufficient_data",
            "reason": f"Need at least {MIN_REAL_ROWS} holdout rows, got {len(holdout_df)}",
            "privacy_risk_level": "UNKNOWN",
        }

    real = coerce_features(real_df, numeric_cols, categorical_cols)
    synth = coerce_features(synth_df, numeric_cols, categorical_cols)

    if holdout_df is not None:
        mode = "provided_holdout"
        train = real
        holdout = coerce_features(holdout_df, numeric_cols, categorical_cols)
    else:
        mode = "split_real_50_50"
        train, holdout = train_test_split(real, test_size=HOLDOUT_FRACTION, random_state=seed)

    pre = build_preprocessor(list(numeric_cols), list(categorical_cols)).fit(train)
    train_x = pre.transform(train)
    rng = np.random.default_rng(seed)
    holdout_x = _cap(pre.transform(holdout), max_query_rows, rng)
    synth_x = _cap(pre.transform(synth), max_query_rows, rng)

    synth_dcr = _nn_distances(train_x, synth_x)
    holdout_dcr = _nn_distances(train_x, holdout_x)

    threshold = float(np.quantile(holdout_dcr, BASELINE_QUANTILE))
    share_closer = float(np.mean(synth_dcr <= threshold))

    # Exact matches against every real row (in split mode that includes the holdout half,
    # since the generator may have seen it), with the holdout-vs-train rate as baseline.
    if holdout_df is not None:
        exact_rate = float(np.mean(synth_dcr <= EXACT_TOL))
    else:
        exact_rate = float(np.mean(_nn_distances(pre.transform(real), synth_x) <= EXACT_TOL))
    holdout_exact_rate = float(np.mean(holdout_dcr <= EXACT_TOL))

    closeness = closeness_auc(synth_dcr, holdout_dcr)
    if (
        share_closer > th.privacy_high_share
        or exact_rate > holdout_exact_rate + th.privacy_exact_match_margin
        or closeness > th.privacy_closeness_auc
    ):
        level = "HIGH"
    elif share_closer > th.privacy_medium_share:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "status": "ok",
        "reason": None,
        "mode": mode,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "n_synthetic": int(len(synth)),
        "n_holdout_queried": int(len(holdout_x)),
        "n_synthetic_queried": int(len(synth_x)),
        "subsampled": bool(len(holdout_x) < len(holdout) or len(synth_x) < len(synth)),
        "synthetic_dcr": _describe(synth_dcr),
        "holdout_dcr": _describe(holdout_dcr),
        "dcr_threshold_p5": threshold,
        "share_closer_than_holdout_p5": share_closer,
        "expected_share_without_memorisation": BASELINE_QUANTILE,
        "dcr_closeness_auc": closeness,
        "expected_closeness_auc_without_memorisation": 0.5,
        "exact_match_rate": exact_rate,
        "holdout_exact_match_rate": holdout_exact_rate,
        "privacy_risk_level": level,
    }
