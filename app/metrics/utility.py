import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    Standard definition: sum_k |acc(k) - conf(k)| * P(k)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            acc = np.mean(y_true[mask])      # fraction of positives
            conf = np.mean(y_prob[mask])     # mean confidence
            ece += np.abs(acc - conf) * np.mean(mask)

    return float(ece)


def compute_utility(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    """
    Compares downstream ML utility of real vs synthetic data.
    Assumes:
    - Last column is the label
    - Binary classification
    """

    # Split features / labels
    # Separate features / labels
    X_real_full = real_df.iloc[:, :-1]
    y_real = real_df.iloc[:, -1]

    X_synth_full = synth_df.iloc[:, :-1]
    y_synth = synth_df.iloc[:, -1]

    # Keep only numeric features
    numeric_cols = X_real_full.select_dtypes(include="number").columns

    X_real = X_real_full[numeric_cols]
    X_synth = X_synth_full[numeric_cols]

    # Train / test split on REAL data
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_real,
        y_real,
        test_size=0.3,
        random_state=42,
        stratify=y_real,
    )

    # Train on REAL data
    real_model = LogisticRegression(max_iter=1000)
    real_model.fit(Xr_train, yr_train)
    real_preds = real_model.predict(Xr_test)
    real_probs = real_model.predict_proba(Xr_test)[:, 1]

    # Train on SYNTHETIC data, test on REAL test set
    synth_model = LogisticRegression(max_iter=1000)
    synth_model.fit(X_synth, y_synth)
    synth_preds = synth_model.predict(Xr_test)
    synth_probs = synth_model.predict_proba(Xr_test)[:, 1]

    return {
        "real_train_accuracy": accuracy_score(yr_test, real_preds),
        "synthetic_train_accuracy": accuracy_score(yr_test, synth_preds),
        "real_train_f1": f1_score(yr_test, real_preds, average="weighted"),
        "synthetic_train_f1": f1_score(yr_test, synth_preds, average="weighted"),
        "real_train_ece": expected_calibration_error(
            yr_test.values, real_probs
        ),
        "synthetic_train_ece": expected_calibration_error(
            yr_test.values, synth_probs
        ),
    }

