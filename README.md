# 🧪 Synthetic Data Quality Validator

A comprehensive framework to **evaluate the trustworthiness of synthetic tabular data** across **fidelity, utility, privacy, diversity, and calibration**, with automatic detection of **numeric vs categorical features** and **actionable warnings**.

---

## 🚀 Motivation

Synthetic data is widely used to:
- mitigate privacy risks
- address data scarcity
- balance biased datasets

However, **poor-quality synthetic data can silently fail** by:
- memorizing real samples
- collapsing categorical distributions
- degrading downstream model performance
- producing poorly calibrated predictions

This project provides a **systematic, metric-driven validator** to detect such failures **before deployment**.

---

## ✨ Key Features

### 🔍 Automatic Feature Type Detection
- Distinguishes **numeric** vs **categorical** columns
- Avoids invalid statistical assumptions
- Handles mixed-type tabular datasets robustly

---

### 📊 Fidelity Evaluation
**Numeric features**
- Kolmogorov–Smirnov statistic
- Wasserstein distance
- Jensen–Shannon divergence

**Categorical features**
- Frequency L1 distance
- Jensen–Shannon divergence
- Worst-category shift detection

---

### 🤖 Downstream Utility
- Trains a logistic regression model
- Compares real-trained vs synthetic-trained performance
- Reports:
  - Accuracy
  - F1 score
  - **Expected Calibration Error (ECE)**

> Utility is computed on **numeric features only** to avoid arbitrary categorical encodings.

---

### 🔐 Privacy Risk Estimation
- Nearest-neighbor distance analysis
- Detects potential memorization
- Reports:
  - Mean / minimum NN distance
  - Fraction below risk threshold
  - Privacy risk level (LOW / MEDIUM / HIGH)

> Computed on numeric features only for valid distance metrics.

---

### 🌱 Diversity Analysis
- Per-feature variance ratio
- Mean variance ratio
- PCA coverage ratio

Detects:
- Mode collapse
- Excessive variance
- Loss of support

---

### 🚨 Intelligent Warnings
Automatically flags:
- High memorization risk
- Distribution collapse or instability
- Significant utility degradation
- **Severe categorical distortion**
- **Calibration degradation**

Warnings are **metric-backed**, interpretable, and non-redundant.

---

## 🧱 Project Structure

```
synthetic-validator/
├── app/
│   ├── report.py                 # Central trust report
│   ├── utils/
│   │   └── column_types.py        # Feature type detection
│   └── metrics/
│       ├── fidelity.py            # Numeric fidelity metrics
│       ├── categorical_fidelity.py
│       ├── utility.py             # Downstream ML utility
│       ├── privacy.py             # Privacy risk
│       └── diversity.py           # Diversity analysis
├── venv/
├── README.md
└── requirements.txt
```

---

## 🧪 Example Usage

```python
import pandas as pd
from app.report import generate_trust_report

real = pd.DataFrame({
    "age": [22, 25, 30, 40, 35],
    "gender": ["M", "F", "F", "M", "M"],
    "income": [40000, 52000, 61000, 72000, 68000],
    "label": [0, 1, 1, 0, 1],
})

synth = pd.DataFrame({
    "age": [23, 27, 29, 38, 34],
    "gender": ["M", "M", "M", "M", "M"],
    "income": [38000, 50000, 60000, 70000, 66000],
    "label": [0, 1, 1, 0, 1],
})

report = generate_trust_report(real, synth, label_column="label")

print(report["fidelity"]["summary"])
print(report["utility"])
print(report["privacy"])
print(report["diversity"])
print(report["warnings"])
```

---

## 📌 Design Principles

- **Metric validity over convenience**
- **No arbitrary encodings**
- **Separation of concerns**:
  - distribution fidelity ≠ predictive utility ≠ privacy
- Conservative defaults, interpretable outputs
- Robust to mixed-type real-world datasets

---

## ⚠️ Limitations

- Utility and privacy are evaluated on numeric features only
- Categorical privacy metrics (e.g., k-anonymity) are out of scope
- Designed for tabular data (not images / text)

---

## 📈 Future Work

- Support for regression utility
- Categorical privacy metrics
- CLI interface
- Visualization dashboard
- Dataset-level trust scoring

---

## 🧑‍💻 Author Notes

This project was built with an emphasis on:
- statistical correctness
- robustness to real-world edge cases
- clarity of interpretation

It is intended as a **foundational evaluation layer** for synthetic data pipelines.
