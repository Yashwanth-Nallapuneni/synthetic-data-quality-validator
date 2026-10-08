"""Every warning threshold in one place.

PROVISIONAL: these values are hand-picked defaults that have not been calibrated against
known-good and known-bad synthetic data. Phase 3 replaces them with calibrated values
loaded from a JSON file via `load_thresholds(path)`.
"""

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from app.errors import InputError


@dataclass(frozen=True)
class Thresholds:
    # Privacy (share of synthetic rows closer to train than the holdout 5th percentile)
    privacy_high_share: float = 0.15
    privacy_medium_share: float = 0.10
    privacy_exact_match_margin: float = 0.05
    # Diversity (mean per-column variance ratio, synthetic over real)
    variance_ratio_low: float = 0.5
    variance_ratio_high: float = 1.5
    # Diversity (k-NN coverage of the real data; warn when below)
    coverage_min: float = 0.5
    # Numeric marginals (largest per-column KS statistic)
    numeric_max_ks: float = 0.2
    # Utility (TSTR compared with TRTR)
    utility_accuracy_ratio: float = 0.8
    utility_r2_drop: float = 0.2
    utility_ece_increase: float = 0.1
    # Categorical fidelity
    categorical_mean_l1: float = 0.3
    categorical_mean_js: float = 0.2
    categorical_worst_shift: float = 0.25
    # Pairwise dependence (mean absolute difference of association matrices)
    dependence_mean_abs_diff: float = 0.1
    # Detectability (mean cross-validated ROC AUC of the real-vs-synthetic classifier)
    c2st_auc: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Return the default thresholds, optionally overridden by a JSON object in `path`.

    Unknown keys and non-numeric values raise InputError so a typo cannot silently fall
    back to a default.
    """
    base = Thresholds()
    if path is None:
        return base
    try:
        overrides = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"Could not read thresholds file {path}: {e}") from e
    if not isinstance(overrides, dict):
        raise InputError("Thresholds file must contain a JSON object")
    known = {f.name for f in fields(Thresholds)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise InputError(f"Unknown threshold names: {unknown}. Known: {sorted(known)}")
    for k, v in overrides.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InputError(f"Threshold '{k}' must be a number, got {v!r}")
    return replace(base, **{k: float(v) for k, v in overrides.items()})
