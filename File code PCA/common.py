"""Shared helpers for the PCA case study and Appendices A-F.

Conventions used everywhere in this project
-------------------------------------------
* Dataset : UCI Breast Cancer Wisconsin (Diagnostic), n = 569, p = 30.
            In scikit-learn the target is 0 = malignant, 1 = benign.
* Split   : stratified 80/20, random_state = 42  ->  455 train / 114 test.
* Scaling : StandardScaler fitted on the TRAIN split only (no data leakage).
* Positive clinical class : MALIGNANT (label 0). All Precision / Recall / F1
  values called "Malignant_*" use pos_label = 0.
* Eigenvalues : "lambda_k" always means an eigenvalue of the correlation
  matrix R (so that sum(lambda_k) = p exactly).  scikit-learn's
  PCA.explained_variance_ uses ddof = 1 while StandardScaler uses ddof = 0,
  so explained_variance_ is larger than the eigenvalue of R by n / (n - 1).
  Ratios (EVR, CEV) are identical in both conventions.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from scipy.special import gammaln
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data():
    """Load Breast Cancer Wisconsin (n = 569 samples, p = 30 features)."""
    dataset = load_breast_cancer()
    return (
        dataset.data,
        dataset.target,
        dataset.target_names,
        [str(f) for f in dataset.feature_names],
    )


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified Train/Test split (80/20 -> 455 train / 114 test)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def standardize_data(X_train, X_test):
    """Z-score scaling fitted on the training split only (prevents leakage)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ----------------------------------------------------------------------------
# Statistical prerequisites: Bartlett, KMO
# ----------------------------------------------------------------------------
def log10_chi2_sf(x, k, n_terms=6):
    """log10 of the chi-square survival function for very large x.

    scipy's chi2.sf / chi2.logsf underflow to 0 / -inf when x >> k, so the
    upper incomplete gamma tail is evaluated with its asymptotic series
        Q(a, z) ~ z^(a-1) e^(-z) / Gamma(a) * sum_j (a-1)(a-2)...(a-j) / z^j
    with a = k / 2 and z = x / 2.
    """
    a, z = k / 2.0, x / 2.0
    series, term = 1.0, 1.0
    for j in range(1, n_terms + 1):
        term *= (a - j) / z
        series += term
    ln_q = (a - 1.0) * np.log(z) - z - gammaln(a) + np.log(series)
    return float(ln_q / np.log(10.0))


def check_pca_suitability(X):
    """Bartlett's sphericity test (1951) and Kaiser-Meyer-Olkin index (1970/74).

    Returns chi2, df, log10 p-value, overall KMO and the per-variable
    measures of sampling adequacy (MSA).
    """
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    _, logdet = np.linalg.slogdet(R)  # numerically safe for near-singular R
    chi2 = -(n - 1 - (2 * p + 5) / 6.0) * logdet
    df = p * (p - 1) // 2

    R_inv = np.linalg.inv(R)
    D_inv = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
    P_part = -D_inv @ R_inv @ D_inv  # partial correlations
    off = lambda M: M - np.diag(np.diag(M))
    r_sq, p_sq = off(R) ** 2, off(P_part) ** 2
    kmo = float(r_sq.sum() / (r_sq.sum() + p_sq.sum()))
    msa = r_sq.sum(axis=0) / (r_sq.sum(axis=0) + p_sq.sum(axis=0))
    return {
        "chi2": float(chi2),
        "df": int(df),
        "log10_p": log10_chi2_sf(chi2, df),
        "kmo": kmo,
        "msa": msa,
        "log_det_R": float(logdet),
    }


# ----------------------------------------------------------------------------
# Eigenvalues and number-of-components rules
# ----------------------------------------------------------------------------
def correlation_eigenvalues(X_scaled):
    """Eigenvalues of the correlation matrix R, sorted decreasingly (sum = p)."""
    R = np.corrcoef(X_scaled, rowvar=False)
    return np.sort(np.linalg.eigvalsh(R))[::-1]


def horn_parallel_analysis(n_samples, n_features, n_iter=200, percentile=95, random_state=42):
    """Horn (1965) parallel analysis: 95th-percentile eigenvalues of random R."""
    rng = np.random.default_rng(random_state)
    random_eigs = np.zeros((n_iter, n_features))
    for i in range(n_iter):
        Z_rand = rng.standard_normal((n_samples, n_features))
        R_rand = np.corrcoef(Z_rand, rowvar=False)
        random_eigs[i] = np.linalg.eigvalsh(R_rand)[::-1]
    return np.percentile(random_eigs, percentile, axis=0)


def count_kaiser(eigenvalues):
    """Kaiser-Guttman rule: number of leading eigenvalues with lambda > 1."""
    q = 0
    for lam in eigenvalues:
        if lam > 1.0:
            q += 1
        else:
            break
    return q


def count_horn(eigenvalues, thresholds):
    """Horn rule: retain components until the first one that fails the test."""
    q = 0
    for lam, thr in zip(eigenvalues, thresholds):
        if lam > thr:
            q += 1
        else:
            break
    return q


def smallest_q_for_cev(evr, target):
    """Smallest q whose cumulative explained variance ratio reaches `target`."""
    return int(np.searchsorted(np.cumsum(evr), target) + 1)


# ----------------------------------------------------------------------------
# PCA, reconstruction, models
# ----------------------------------------------------------------------------
def reduce_with_pca(X_train, X_test, n_components=10, random_state=42):
    """Linear orthogonal PCA fitted on the training split only."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def compute_reconstruction_error(X_scaled, X_pca, pca):
    """Relative squared reconstruction error (%) - Eckart-Young low-rank bound."""
    X_reconstructed = pca.inverse_transform(X_pca)
    err = np.sum((X_scaled - X_reconstructed) ** 2) / np.sum(X_scaled ** 2) * 100.0
    return float(err)


def train_logistic_regression(X_train, y_train, random_state=42):
    """Logistic regression classifier."""
    model = LogisticRegression(max_iter=5000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Classification metrics (class 0 = Malignant is the clinical positive class)."""
    preds = model.predict(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Malignant_Precision": float(precision_score(y_test, preds, pos_label=0, zero_division=0)),
        "Malignant_Recall": float(recall_score(y_test, preds, pos_label=0, zero_division=0)),
        "Malignant_F1": float(f1_score(y_test, preds, pos_label=0, zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    }
    return metrics, preds


# ----------------------------------------------------------------------------
# Reproducibility helpers and plotting style
# ----------------------------------------------------------------------------
def get_environment_metadata():
    """Record library versions so that results can be reproduced."""
    return {
        "environment": {
            "Python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "NumPy": np.__version__,
        },
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_json_result(filename, result):
    """Save quantitative results as JSON inside outputs/."""
    path = OUT_DIR / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON: {path}")


def set_plot_style():
    """Common matplotlib style used by all figures of the paper."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.30,
        "figure.dpi": 100,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    })
    return plt
