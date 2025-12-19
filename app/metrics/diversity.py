import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def compute_diversity(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    Measures diversity of synthetic data relative to real data.
    Uses numeric features only.
    """

    # Keep numeric features only
    real_num = real_df.select_dtypes(include="number")
    synth_num = synth_df.select_dtypes(include="number")

    if real_num.empty or synth_num.empty:
        return {
            "variance_ratio": None,
            "mean_variance_ratio": None,
            "pca_coverage_ratio": None,
        }

    # Variance ratio per feature
    var_real = real_num.var()
    var_synth = synth_num.var()

    variance_ratio = (var_synth / (var_real + 1e-8)).to_dict()
    mean_variance_ratio = float(np.mean(list(variance_ratio.values())))

    # PCA coverage ratio
    pca = PCA(n_components=min(len(real_num.columns), len(real_num)))
    pca.fit(real_num.values)

    real_var = np.sum(pca.explained_variance_)
    synth_proj = pca.transform(synth_num.values)
    synth_var = np.var(synth_proj, axis=0).sum()

    pca_coverage_ratio = float(synth_var / (real_var + 1e-8))

    return {
        "variance_ratio": variance_ratio,
        "mean_variance_ratio": mean_variance_ratio,
        "pca_coverage_ratio": pca_coverage_ratio,
    }