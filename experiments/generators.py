"""Synthetic-data generators for the Phase 3 benchmark.

Every generator takes the training frame `train` (features plus the label column), the number
of rows `n` to produce, the label column name, a seed and `fresh` (real rows that are NOT in
`train`), and returns a new DataFrame with the same columns.

Only the memorisation injections (exact_copy, near_copy) and the copula use `train`. The
other four injections damage one property of otherwise genuine unseen rows, so they start
from `fresh` (the benchmark passes H_gold): that way the privacy check has no reason to fire
on them and "any warning fired" is not inflated by copying training rows. The held-out real
generator, `heldout_real`, is handled by the benchmark itself.

Good generator:
- gaussian_copula: a from-scratch Gaussian copula (no SDV).

Deliberately broken generators, one failure mode each:
- exact_copy: rows of train sampled without replacement
- near_copy: rows of train plus Gaussian noise of `level` times each numeric column's std
- column_shuffle: a bootstrap of fresh with every column permuted independently
- mode_collapse: a bootstrap of 5 random fresh rows, with 1% numeric jitter (scaled by the
  std of the training column)
- category_oversample: bootstrap of fresh with 50% of one feature's rows set to the rarest
  category (in train) of the feature with the most categories (in train)
- label_shuffle: a bootstrap of fresh with the label column permuted
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from app.utils.column_types import detect_column_types

MISSING = "__missing__"
NEAR_COPY_LEVELS = {"near_copy": 0.01, "near_copy_5pct": 0.05}
COLLAPSE_ROWS = 5
COLLAPSE_JITTER = 0.01
OVERSAMPLE_SHARE = 0.5


def split_columns(train: pd.DataFrame, label: str) -> tuple[list[str], list[str]]:
    """(numeric feature columns, all other columns including the label)."""
    types = detect_column_types(train.drop(columns=[label]))
    numeric = list(types["numeric"])
    other = [c for c in train.columns if c not in numeric]
    return numeric, other


def _jitter(df: pd.DataFrame, numeric: list[str], level: float, rng) -> pd.DataFrame:
    out = df.copy()
    for c in numeric:
        std = float(out[c].std())
        if std > 0:
            out[c] = out[c].astype(float) + rng.normal(0.0, level * std, len(out))
    return out


def _pool(fresh: pd.DataFrame | None) -> pd.DataFrame:
    if fresh is None:
        raise ValueError("this generator needs `fresh` rows that are not in the training set")
    return fresh


def exact_copy(train: pd.DataFrame, n: int, label: str, seed: int, fresh=None) -> pd.DataFrame:
    return train.sample(n, replace=False, random_state=seed).reset_index(drop=True)


def near_copy(train: pd.DataFrame, n: int, label: str, seed: int, fresh=None, level: float = 0.01):
    rng = np.random.default_rng(seed)
    numeric, _ = split_columns(train, label)
    base = train.sample(n, replace=False, random_state=seed).reset_index(drop=True)
    return _jitter(base, numeric, level, rng)


def column_shuffle(
    train: pd.DataFrame, n: int, label: str, seed: int, fresh: pd.DataFrame | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boot = _pool(fresh).sample(n, replace=True, random_state=seed).reset_index(drop=True)
    for c in boot.columns:
        boot[c] = boot[c].to_numpy()[rng.permutation(n)]
    return boot


def mode_collapse(
    train: pd.DataFrame, n: int, label: str, seed: int, fresh: pd.DataFrame | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric, _ = split_columns(train, label)
    centres = _pool(fresh).sample(COLLAPSE_ROWS, replace=False, random_state=seed)
    boot = centres.iloc[rng.integers(0, COLLAPSE_ROWS, n)].reset_index(drop=True)
    # Jitter is scaled by the std of the full training column, not of the 5 rows.
    out = boot.copy()
    for c in numeric:
        std = float(train[c].std())
        if std > 0:
            out[c] = out[c].astype(float) + rng.normal(0.0, COLLAPSE_JITTER * std, n)
    return out


def oversample_target(train: pd.DataFrame, label: str) -> tuple[str, object]:
    """The categorical feature with the most categories and its rarest category.

    Ties go to the first column and the first category in sorted order. The label is not
    eligible, and missing values do not count as a category.
    """
    _, other = split_columns(train, label)
    cands = [c for c in other if c != label]
    col = max(cands, key=lambda c: (train[c].nunique(), -cands.index(c)))
    counts = train[col].value_counts().sort_index()
    rarest = counts.idxmin()
    return col, rarest


def category_oversample(
    train: pd.DataFrame, n: int, label: str, seed: int, fresh: pd.DataFrame | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    col, rarest = oversample_target(train, label)
    boot = _pool(fresh).sample(n, replace=True, random_state=seed).reset_index(drop=True)
    pick = rng.choice(n, int(round(OVERSAMPLE_SHARE * n)), replace=False)
    values = boot[col].astype(object).to_numpy().copy()
    values[pick] = rarest
    if isinstance(train[col].dtype, pd.CategoricalDtype):
        boot[col] = pd.Categorical(values, categories=train[col].cat.categories)
    else:
        boot[col] = values
    return boot


def label_shuffle(
    train: pd.DataFrame, n: int, label: str, seed: int, fresh: pd.DataFrame | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boot = _pool(fresh).sample(n, replace=True, random_state=seed).reset_index(drop=True)
    boot[label] = boot[label].to_numpy()[rng.permutation(n)]
    return boot


# Gaussian copula ---------------------------------------------------------------------


class GaussianCopula:
    """Gaussian copula with empirical marginals.

    - Numeric column: normal score of its empirical CDF (average ranks, (rank - 0.5) / n).
      Sampling inverts with the empirical quantile function of the training column.
      Columns whose training values are all integers are rounded back to integers. Missing
      values: the normal score of a missing entry is a fresh N(0,1) draw when fitting, and
      at sampling time each value is set missing independently at the column's training
      missing rate.
    - Categorical column (and the label): categories are ordered by frequency and mapped to
      consecutive intervals of [0, 1] whose widths are the category frequencies. A row's
      value is a uniform draw inside its category's interval, then normal-scored. Missing
      values are their own category. Sampling maps the normal score back through the normal
      CDF to the interval.
    - The copula is the correlation matrix of the normal scores (made positive definite),
      sampled as a multivariate normal.
    """

    def __init__(self, label: str, seed: int = 0):
        self.label = label
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "GaussianCopula":
        rng = np.random.default_rng(self.seed)
        numeric, _ = split_columns(train, self.label)
        self.columns_ = list(train.columns)
        self.numeric_ = set(numeric)
        self.info_: dict[str, dict] = {}
        scores = np.empty((len(train), len(self.columns_)))
        for j, col in enumerate(self.columns_):
            s = train[col]
            if col in self.numeric_:
                x = s.to_numpy(dtype=float)
                miss = ~np.isfinite(x)
                valid = np.sort(x[~miss])
                u = np.empty(len(x))
                u[~miss] = (rankdata(x[~miss]) - 0.5) / max(len(valid), 1)
                z = np.empty(len(x))
                z[~miss] = norm.ppf(u[~miss])
                z[miss] = rng.standard_normal(int(miss.sum()))
                self.info_[col] = {
                    "valid": valid,
                    "miss_rate": float(miss.mean()),
                    "integer": bool(len(valid) and np.allclose(valid, np.round(valid))),
                }
            else:
                vals = s.astype(object).where(s.notna(), MISSING)
                freq = vals.value_counts()
                cats = list(freq.index)
                probs = freq.to_numpy(dtype=float) / freq.sum()
                edges = np.r_[0.0, np.cumsum(probs)]
                edges[-1] = 1.0
                idx = pd.Index(cats).get_indexer(vals)
                u = edges[idx] + rng.uniform(size=len(vals)) * (edges[idx + 1] - edges[idx])
                z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
                self.info_[col] = {"cats": cats, "edges": edges}
            scores[:, j] = z
        corr = np.corrcoef(scores, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        self.corr_ = _nearest_correlation(corr)
        self.dtypes_ = train.dtypes.to_dict()
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        z = rng.multivariate_normal(np.zeros(len(self.columns_)), self.corr_, size=n)
        u = norm.cdf(z)
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self.columns_):
            info = self.info_[col]
            if col in self.numeric_:
                v = (
                    np.quantile(info["valid"], u[:, j])
                    if len(info["valid"])
                    else np.full(n, np.nan)
                )
                if info["integer"]:
                    v = np.round(v)
                if info["miss_rate"] > 0:
                    v = np.where(rng.uniform(size=n) < info["miss_rate"], np.nan, v)
                out[col] = v
            else:
                k = np.searchsorted(info["edges"], u[:, j], side="right") - 1
                k = np.clip(k, 0, len(info["cats"]) - 1)
                vals = np.array(info["cats"], dtype=object)[k]
                out[col] = np.where(vals == MISSING, np.nan, vals)
        df = pd.DataFrame(out, columns=self.columns_)
        for col, dt in self.dtypes_.items():
            if isinstance(dt, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col], categories=dt.categories)
            elif not df[col].isna().any() and (
                col not in self.numeric_ or self.info_[col]["integer"]
            ):
                df[col] = df[col].astype(dt)  # restore int, bool and float dtypes
        return df


def _nearest_correlation(corr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip eigenvalues to make the matrix positive definite, then rescale to unit diagonal."""
    vals, vecs = np.linalg.eigh((corr + corr.T) / 2)
    vals = np.clip(vals, eps, None)
    out = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def gaussian_copula(train: pd.DataFrame, n: int, label: str, seed: int, fresh=None) -> pd.DataFrame:
    return GaussianCopula(label, seed).fit(train).sample(n, seed + 1)


GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "gaussian_copula": gaussian_copula,
    "exact_copy": exact_copy,
    "near_copy": partial(near_copy, level=NEAR_COPY_LEVELS["near_copy"]),
    "near_copy_5pct": partial(near_copy, level=NEAR_COPY_LEVELS["near_copy_5pct"]),
    "column_shuffle": column_shuffle,
    "mode_collapse": mode_collapse,
    "category_oversample": category_oversample,
    "label_shuffle": label_shuffle,
}

# Name of the gold-standard null; it is not in GENERATORS because S = H_gold.
HELDOUT = "heldout_real"
GOOD = (HELDOUT, "gaussian_copula")
BROKEN = tuple(g for g in GENERATORS if g != "gaussian_copula")
