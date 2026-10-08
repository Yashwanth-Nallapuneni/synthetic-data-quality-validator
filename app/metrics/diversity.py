import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_diversity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict:
    """Measure diversity of synthetic data relative to real data on the numeric columns.

    Column types come from the caller (app.utils.column_types.detect_column_types).
    """
    empty = {
        "variance_ratio": None,
        "mean_variance_ratio": None,
        "pca_coverage_ratio": None,
    }
    if not numeric_cols:
        return empty

    real_num = real_df[numeric_cols].astype(float)
    synth_num = synth_df[numeric_cols].astype(float)
    if len(real_num.dropna()) < 2 or len(synth_num.dropna()) < 2:
        return empty

    var_real = real_num.var()
    var_synth = synth_num.var()
    variance_ratio = {col: float(var_synth[col] / (var_real[col] + 1e-8)) for col in numeric_cols}
    mean_variance_ratio = float(np.nanmean(list(variance_ratio.values())))

    # PCA coverage ratio on complete rows
    real_complete = real_num.dropna()
    synth_complete = synth_num.dropna()
    pca = PCA(n_components=min(len(numeric_cols), len(real_complete)))
    pca.fit(real_complete.to_numpy())
    real_var = float(np.sum(pca.explained_variance_))
    synth_proj = pca.transform(synth_complete.to_numpy())
    synth_var = float(np.var(synth_proj, axis=0, ddof=1).sum())

    return {
        "variance_ratio": variance_ratio,
        "mean_variance_ratio": mean_variance_ratio,
        "pca_coverage_ratio": float(synth_var / (real_var + 1e-8)),
    }
