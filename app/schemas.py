"""Pydantic response models for the API and the JSON report.

`SCHEMA_VERSION` is the version of the JSON layout and is bumped when a key is removed or
changes meaning (major) or when a key is added (minor). It is separate from the package
version (`app.__version__`), which both the API and the report carry as `app_version`.

Models are strict (`extra="forbid"`) so a metric that grows a key without a schema update
fails the schema test instead of silently changing the contract. The only open-ended
mappings are the ones keyed by data: per-feature statistics are keyed by column name and
per-model utility statistics by model name.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "2.0"

Severity = Literal["clean", "issues"]
Status = Literal["ok", "insufficient_data"]
Matrix = list[list[float | None]]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Model):
    status: str


class Warning(_Model):
    """One threshold check that fired. `metric` is the path of the statistic compared."""

    code: str
    message: str
    metric: str
    value: float | None
    threshold: float | None


# Fidelity


class NumericFeatureFidelity(_Model):
    ks_statistic: float | None
    wasserstein_distance: float | None
    wasserstein_standardised: float | None
    js_divergence: float | None


class NumericFidelitySummary(_Model):
    mean_ks: float | None
    mean_js: float | None
    mean_wasserstein_standardised: float | None
    worst_feature: str | None


class NumericFidelity(_Model):
    per_feature: dict[str, NumericFeatureFidelity]
    summary: NumericFidelitySummary


class CategoricalFeatureFidelity(_Model):
    l1_distance: float | None
    js_divergence: float | None
    worst_category: str | None
    worst_shift: float | None


class CategoricalFidelitySummary(_Model):
    mean_l1: float | None
    mean_js: float | None
    worst_feature: str | None


class CategoricalFidelity(_Model):
    per_feature: dict[str, CategoricalFeatureFidelity]
    summary: CategoricalFidelitySummary


class FidelityCounts(_Model):
    num_numeric_features: int
    num_categorical_features: int


class Fidelity(_Model):
    numeric: NumericFidelity | None
    categorical: CategoricalFidelity | None
    summary: FidelityCounts


# Dependence


class DependenceBlock(_Model):
    measure: str
    rows: list[str]
    columns: list[str]
    n_pairs: int
    mean_abs_diff: float | None
    max_abs_diff: float | None
    matrices_omitted: bool
    real: Matrix | None
    synthetic: Matrix | None
    abs_diff: Matrix | None


class DependenceSummary(_Model):
    n_pairs: int
    mean_abs_diff: float | None
    max_abs_diff: float | None
    worst_pair: list[str]


class DependencePair(_Model):
    a: str
    b: str
    kind: str
    real: float | None
    synthetic: float | None
    abs_diff: float | None


class Dependence(_Model):
    status: Status
    reason: str | None = None
    numeric_numeric: DependenceBlock | None = None
    categorical_categorical: DependenceBlock | None = None
    numeric_categorical: DependenceBlock | None = None
    summary: DependenceSummary | None = None
    worst_pairs: list[DependencePair] | None = None


# Detectability


class Detectability(_Model):
    status: Status
    reason: str | None = None
    model: str | None = None
    auc_mean: float | None = None
    auc_std: float | None = None
    auc_per_fold: list[float | None] | None = None
    n_splits: int | None = None
    n_per_side: int | None = None
    subsampled: bool | None = None
    n_real_available: int | None = None
    n_synthetic_available: int | None = None


# Utility


class ClassificationScores(_Model):
    accuracy: float | None
    balanced_accuracy: float | None
    f1: float | None
    ece: float | None


class RegressionScores(_Model):
    r2: float | None
    mae: float | None


class ClassificationComparison(_Model):
    accuracy_ratio: float | None
    balanced_accuracy_ratio: float | None
    ece_increase: float | None


class RegressionComparison(_Model):
    r2_drop: float | None


class ClassificationModelResult(_Model):
    trtr: ClassificationScores
    tstr: ClassificationScores
    tstr_vs_trtr: ClassificationComparison


class RegressionModelResult(_Model):
    trtr: RegressionScores
    tstr: RegressionScores
    tstr_vs_trtr: RegressionComparison


class Utility(_Model):
    """`models` maps a model name to its result; the scores depend on `task`."""

    status: Status
    reason: str | None = None
    task: Literal["classification", "regression"] | None = None
    n_real_train: int | None = None
    n_real_test: int | None = None
    n_synthetic: int | None = None
    n_classes: int | None = None
    synthetic_single_class: bool | None = None
    models: dict[str, ClassificationModelResult | RegressionModelResult] | None = None


# Privacy


class DcrSummary(_Model):
    mean: float | None
    median: float | None
    min: float | None
    p5: float | None


class Privacy(_Model):
    status: Status
    reason: str | None = None
    privacy_risk_level: Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]
    mode: Literal["provided_holdout", "split_real_50_50"] | None = None
    n_train: int | None = None
    n_holdout: int | None = None
    n_synthetic: int | None = None
    n_holdout_queried: int | None = None
    n_synthetic_queried: int | None = None
    subsampled: bool | None = None
    synthetic_dcr: DcrSummary | None = None
    holdout_dcr: DcrSummary | None = None
    dcr_threshold_p5: float | None = None
    share_closer_than_holdout_p5: float | None = None
    expected_share_without_memorisation: float | None = None
    dcr_closeness_auc: float | None = None
    expected_closeness_auc_without_memorisation: float | None = None
    exact_match_rate: float | None = None
    holdout_exact_match_rate: float | None = None


# Diversity


class Coverage(_Model):
    value: float | None
    k: int
    n_real_queried: int
    subsampled: bool


class Diversity(_Model):
    """All fields are null when the data is too small or has no usable columns."""

    variance_ratio: dict[str, float | None] | None
    mean_variance_ratio: float | None
    pca_coverage_ratio: float | None
    coverage: Coverage | None


# Report


class Thresholds(_Model):
    privacy_high_share: float
    privacy_medium_share: float
    privacy_exact_match_margin: float
    privacy_closeness_auc: float
    variance_ratio_low: float
    variance_ratio_high: float
    coverage_min: float
    numeric_max_ks: float
    utility_balanced_accuracy_ratio: float
    utility_r2_drop: float
    utility_ece_increase: float
    categorical_mean_l1: float
    categorical_mean_js: float
    categorical_worst_shift: float
    dependence_mean_abs_diff: float
    c2st_auc: float


class Report(_Model):
    seed: int
    fidelity: Fidelity
    dependence: Dependence
    detectability: Detectability
    utility: Utility
    privacy: Privacy
    diversity: Diversity
    warnings: list[Warning]
    notices: list[str]
    thresholds: Thresholds


class ValidationResponse(_Model):
    schema_version: str
    app_version: str
    severity: Severity
    report: Report
