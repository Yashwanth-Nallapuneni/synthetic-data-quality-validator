"""Every warning threshold in one place.

`Thresholds()` holds the PROVISIONAL defaults: hand-picked values that were never fitted to
data. Phase 3 calibrates them on held-out real data (experiments/calibrate.py) and ships the
result as app/calibrated_thresholds.json. `load_thresholds()` with no path returns the
calibrated values when that file exists and falls back to the provisional defaults otherwise.
"""

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

from app.errors import InputError


@dataclass(frozen=True)
class Thresholds:
    # Privacy (share of synthetic rows closer to train than the holdout 5th percentile)
    privacy_high_share: float = 0.15
    privacy_medium_share: float = 0.10
    privacy_exact_match_margin: float = 0.05
    privacy_closeness_auc: float = 0.6
    # Diversity (mean per-column variance ratio, synthetic over real)
    variance_ratio_low: float = 0.5
    variance_ratio_high: float = 1.5
    # Diversity (k-NN coverage of the real data; warn when below)
    coverage_min: float = 0.5
    # Numeric marginals (largest per-column KS statistic)
    numeric_max_ks: float = 0.2
    # Utility (TSTR compared with TRTR)
    utility_balanced_accuracy_ratio: float = 0.8
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CALIBRATED_PATH = Path(__file__).resolve().parent / "calibrated_thresholds.json"


def load_thresholds(path: str | Path | None = None) -> Thresholds:
    """Thresholds from a JSON file, the shipped calibrated file or the provisional defaults.

    - With `path`: the provisional defaults overridden by the file's values.
    - Without `path`: the same, using app/calibrated_thresholds.json if it exists, else the
      provisional defaults.

    The file is either a flat JSON object of threshold values or the calibrated format
    {"thresholds": {...}, "calibration": {...}} where only "thresholds" is read. Unknown
    keys and non-numeric values raise InputError so a typo cannot silently fall back to a
    default.
    """
    base = Thresholds()
    if path is None:
        if not CALIBRATED_PATH.exists():
            return base
        path = CALIBRATED_PATH
    try:
        overrides = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise InputError(f"Could not read thresholds file {path}: {e}") from e
    if not isinstance(overrides, dict):
        raise InputError("Thresholds file must contain a JSON object")
    if "thresholds" in overrides:
        overrides = overrides["thresholds"]
        if not isinstance(overrides, dict):
            raise InputError("'thresholds' in the thresholds file must be a JSON object")
    known = {f.name for f in fields(Thresholds)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise InputError(f"Unknown threshold names: {unknown}. Known: {sorted(known)}")
    for k, v in overrides.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InputError(f"Threshold '{k}' must be a number, got {v!r}")
    return replace(base, **{k: float(v) for k, v in overrides.items()})
