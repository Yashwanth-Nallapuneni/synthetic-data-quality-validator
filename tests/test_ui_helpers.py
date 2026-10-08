import numpy as np
import pandas as pd
import pytest

from app.report import generate_trust_report
from app.ui_helpers import (
    categorical_fidelity_table,
    category_share_frame,
    column_kind,
    dependence_heatmap_frame,
    detectability_table,
    dimension_verdicts,
    diversity_table,
    histogram_frame,
    numeric_fidelity_table,
    privacy_table,
    utility_table,
    warnings_table,
)
from tests.conftest import make_dataset


@pytest.fixture(scope="module")
def report():
    return generate_trust_report(make_dataset(300, 1), make_dataset(300, 2), "label")


@pytest.fixture(scope="module")
def tiny_report():
    return generate_trust_report(make_dataset(8, 1), make_dataset(8, 2), "label")


@pytest.fixture(scope="module")
def copy_report():
    real = make_dataset(300, 1)
    return generate_trust_report(real, real.copy(), "label")


def test_fidelity_tables(report):
    num = numeric_fidelity_table(report)
    assert set(num["feature"]) == {"age", "income"}
    assert list(num.columns) == ["feature", "KS", "Wasserstein (standardised)", "JS"]
    cat = categorical_fidelity_table(report)
    assert list(cat["feature"]) == ["city"]


def test_fidelity_tables_empty_when_no_columns(report):
    r = {**report, "fidelity": {**report["fidelity"], "numeric": None, "categorical": None}}
    assert numeric_fidelity_table(r).empty
    assert categorical_fidelity_table(r).empty


def test_utility_table_classification_and_regression(report):
    t = utility_table(report)
    assert set(t["metric"]) == {"accuracy", "balanced_accuracy", "f1", "ece"}
    assert set(t["model"]) == {"logistic_regression", "hist_gradient_boosting"}
    row = t.iloc[0]
    assert row["TSTR minus TRTR"] == pytest.approx(row["TSTR"] - row["TRTR"])

    def reg(n, seed):
        df = make_dataset(n, seed)
        df["label"] = df["income"] * 0.001 + df["age"]
        return df

    rr = generate_trust_report(reg(300, 1), reg(300, 2), "label")
    assert set(utility_table(rr)["metric"]) == {"r2", "mae"}


def test_tables_empty_for_insufficient_data(tiny_report):
    assert utility_table(tiny_report).empty
    assert privacy_table(tiny_report).empty
    assert detectability_table(tiny_report).empty
    assert dependence_heatmap_frame(tiny_report).empty


def test_privacy_diversity_detectability_tables(report):
    assert len(privacy_table(report)) == 5
    div = diversity_table(report)
    assert "Mean variance ratio" in set(div["measure"])
    assert any(m.startswith("Coverage") for m in div["measure"])
    assert "AUC, mean over folds" in set(detectability_table(report)["measure"])


def test_warnings_table(copy_report, report):
    t = warnings_table(copy_report)
    assert list(t.columns) == ["code", "message", "value", "threshold"]
    assert "privacy_memorisation" in set(t["code"])
    assert len(warnings_table({"warnings": []})) == 0


def test_dimension_verdicts(copy_report, tiny_report, report):
    v = dimension_verdicts(copy_report).set_index("dimension")
    assert v.loc["Privacy", "verdict"] == "warn"
    assert "privacy_memorisation" in v.loc["Privacy", "checks fired"]
    t = dimension_verdicts(tiny_report).set_index("dimension")
    assert t.loc["Utility", "verdict"] == "not computed"
    assert t.loc["Detectability", "verdict"] == "not computed"
    assert list(dimension_verdicts(report)["dimension"]) == [
        "Fidelity",
        "Dependence",
        "Detectability",
        "Utility",
        "Privacy",
        "Diversity",
    ]


def test_dimension_verdict_pass_when_no_warnings():
    r = {"warnings": [], "privacy": {"status": "ok"}}
    v = dimension_verdicts(r).set_index("dimension")
    assert v.loc["Privacy", "verdict"] == "pass"
    assert v.loc["Utility", "verdict"] == "pass"


def test_heatmap_frame_is_symmetric(report):
    h = dependence_heatmap_frame(report)
    assert not h.empty
    cells = {(r.x, r.y): r.abs_diff for r in h.itertuples()}
    assert all(cells[(y, x)] == v for (x, y), v in cells.items())
    assert {"age", "income", "city", "label"} <= set(h["x"])
    assert (h["x"] != h["y"]).all()


def test_heatmap_empty_when_matrices_omitted(report):
    dep = {
        k: (dict(v, abs_diff=None) if isinstance(v, dict) and "abs_diff" in v else v)
        for k, v in report["dependence"].items()
    }
    assert dependence_heatmap_frame({"dependence": dep}).empty


def test_column_kind(report):
    real = make_dataset(300, 1)
    assert column_kind(report, "age", real["age"]) == "numeric"
    assert column_kind(report, "city", real["city"]) == "categorical"
    assert column_kind({}, "x", pd.Series(np.arange(100.0))) == "numeric"
    assert column_kind({}, "x", pd.Series([1, 2, 1, 2])) == "categorical"


def test_histogram_frame_uses_shared_bins_and_sums_to_one():
    rng = np.random.default_rng(0)
    f = histogram_frame(pd.Series(rng.normal(0, 1, 500)), pd.Series(rng.normal(3, 1, 400)), 10)
    assert set(f["dataset"]) == {"real", "synthetic"}
    assert f.groupby("dataset")["density"].sum().tolist() == pytest.approx([1.0, 1.0])
    first = f[f["dataset"] == "real"][["bin_start", "bin_end"]].to_numpy()
    second = f[f["dataset"] == "synthetic"][["bin_start", "bin_end"]].to_numpy()
    assert (first == second).all()


def test_histogram_frame_handles_constant_and_empty():
    f = histogram_frame(pd.Series([2.0, 2.0]), pd.Series([2.0]))
    assert f["density"].sum() == pytest.approx(2.0)
    assert histogram_frame(pd.Series([], dtype=float), pd.Series([], dtype=float)).empty


def test_category_share_frame_includes_unseen_synthetic_category():
    f = category_share_frame(pd.Series(["a", "a", "b"]), pd.Series(["a", "c", "c", "c"]))
    shares = f.pivot(index="category", columns="dataset", values="share")
    assert shares.loc["a", "real"] == pytest.approx(2 / 3)
    assert shares.loc["c", "real"] == 0.0
    assert shares.loc["c", "synthetic"] == pytest.approx(0.75)


def test_category_share_frame_limits_categories():
    s = pd.Series([f"c{i}" for i in range(50)])
    f = category_share_frame(s, s, max_categories=5)
    assert f["category"].nunique() == 5


def test_tables_are_arrow_serialisable(report):
    import pyarrow as pa

    for t in (
        numeric_fidelity_table(report),
        categorical_fidelity_table(report),
        utility_table(report),
        privacy_table(report),
        diversity_table(report),
        detectability_table(report),
        warnings_table(report),
        dimension_verdicts(report),
    ):
        pa.Table.from_pandas(t)
