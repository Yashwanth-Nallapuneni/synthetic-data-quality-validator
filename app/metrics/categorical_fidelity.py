import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def _category_distributions(real_col, synth_col):
    """
    Align category distributions over union of categories.
    Returns two probability vectors with same ordering.
    """
    real_counts = real_col.value_counts(normalize=True)
    synth_counts = synth_col.value_counts(normalize=True)

    categories = sorted(set(real_counts.index) | set(synth_counts.index))

    real_probs = np.array([real_counts.get(cat, 0.0) for cat in categories])
    synth_probs = np.array([synth_counts.get(cat, 0.0) for cat in categories])

    return categories, real_probs, synth_probs


def compute_categorical_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame, categorical_cols):
    """
    Compute categorical fidelity metrics for given columns.
    """

    results = {}

    l1_scores = []
    js_scores = []

    for col in categorical_cols:
        categories, p_real, p_synth = _category_distributions(
            real_df[col], synth_df[col]
        )

        l1 = float(np.sum(np.abs(p_real - p_synth)))
        js = float(jensenshannon(p_real, p_synth))

        worst_idx = int(np.argmax(np.abs(p_real - p_synth)))
        worst_category = categories[worst_idx]
        worst_shift = float(p_real[worst_idx] - p_synth[worst_idx])

        results[col] = {
            "l1_distance": l1,
            "js_divergence": js,
            "worst_category": worst_category,
            "worst_shift": worst_shift,
        }

        l1_scores.append(l1)
        js_scores.append(js)

    summary = {
        "mean_l1": float(np.mean(l1_scores)) if l1_scores else 0.0,
        "mean_js": float(np.mean(js_scores)) if js_scores else 0.0,
        "worst_feature": (
            max(results, key=lambda c: results[c]["l1_distance"])
            if results else None
        ),
    }

    return {
        "per_feature": results,
        "summary": summary,
    }