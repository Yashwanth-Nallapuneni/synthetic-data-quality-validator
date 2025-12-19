import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_privacy_risk(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    threshold: float = 0.1,
):
    """
    Estimates memorization risk using nearest-neighbor distance.

    IMPORTANT:
    - Uses NUMERIC FEATURES ONLY
    - Categorical columns are intentionally excluded because
      distance-based privacy is undefined for categories.
    """

    # Keep numeric features only
    real_num = real_df.select_dtypes(include="number")
    synth_num = synth_df.select_dtypes(include="number")

    # If no numeric features exist, privacy risk is undefined
    if real_num.empty or synth_num.empty:
        return {
            "mean_nn_distance": None,
            "min_nn_distance": None,
            "fraction_below_threshold": None,
            "privacy_risk_level": "UNKNOWN",
        }

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(real_num.values)

    distances, _ = nn.kneighbors(synth_num.values)
    distances = distances.flatten()

    risk_fraction = float(np.mean(distances < threshold))

    return {
        "mean_nn_distance": float(np.mean(distances)),
        "min_nn_distance": float(np.min(distances)),
        "fraction_below_threshold": risk_fraction,
        "privacy_risk_level": (
            "HIGH" if risk_fraction > 0.2
            else "MEDIUM" if risk_fraction > 0.05
            else "LOW"
        ),
    }