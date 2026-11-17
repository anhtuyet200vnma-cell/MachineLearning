"""Hàm dùng chung cho Case Study và các Phụ lục A, B, C, D của thuật toán PCA."""
from pathlib import Path
from datetime import datetime, timezone
import json
import platform

import numpy as np
import pandas as pd
import sklearn
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


def load_data():
    """Tải bộ dữ liệu Breast Cancer Wisconsin (n=569 mẫu, p=30 đặc trưng)."""
    dataset = load_breast_cancer()
    return (
        dataset.data,
        dataset.target,
        dataset.target_names,
        list(dataset.feature_names),
    )


def split_data(X, y, test_size=0.2, random_state=42):
    """Chia tập Train/Test phân tầng 80/20 (455 mẫu Train / 114 mẫu Test)."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def standardize_data(X_train, X_test):
    """Chuẩn hóa Z-score (chỉ fit trên Train, sau đó transform Test để chống Data Leakage)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def check_pca_suitability(X_train):
    """Kiểm định cầu Bartlett (1951) và chỉ số KMO (Kaiser, 1970) trên tập Train."""
    n, p = X_train.shape
    R = np.corrcoef(X_train, rowvar=False)
    det_R = max(float(np.linalg.det(R)), 1e-300)
    chi2 = -(n - 1 - (2 * p + 5) / 6.0) * np.log(det_R)
    df = p * (p - 1) // 2

    R_inv = np.linalg.pinv(R)
    D_inv = np.diag(1.0 / np.sqrt(np.diag(R_inv)))
    P_part = -D_inv @ R_inv @ D_inv

    r_sq_sum = float(np.sum((R - np.diag(np.diag(R))) ** 2))
    p_sq_sum = float(np.sum((P_part - np.diag(np.diag(P_part))) ** 2))
    kmo = r_sq_sum / (r_sq_sum + p_sq_sum)
    return {"chi2": float(chi2), "df": int(df), "kmo": float(kmo)}


def horn_parallel_analysis(n_samples, n_features, n_iter=200, percentile=95, random_state=42):
    """Phân tích song song Horn (1965): ước lượng ngưỡng trị riêng nhiễu ngẫu nhiên."""
    rng = np.random.default_rng(random_state)
    random_eigs = np.zeros((n_iter, n_features))
    for i in range(n_iter):
        Z_rand = rng.standard_normal((n_samples, n_features))
        Z_rand = (Z_rand - Z_rand.mean(axis=0)) / Z_rand.std(axis=0, ddof=1)
        R_rand = np.corrcoef(Z_rand, rowvar=False)
        eigs = np.linalg.eigvalsh(R_rand)[::-1]
        random_eigs[i] = eigs
    return np.percentile(random_eigs, percentile, axis=0)


def reduce_with_pca(X_train, X_test, n_components=10, random_state=42):
    """Giảm chiều dữ liệu bằng PCA trực giao tuyến tính."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def compute_reconstruction_error(X_scaled, X_pca, pca):
    """Tính sai số tái tạo tương đối (%) theo định lý xấp xỉ hạng thấp Eckart-Young."""
    X_reconstructed = pca.inverse_transform(X_pca)
    err = np.sum((X_scaled - X_reconstructed) ** 2) / np.sum(X_scaled ** 2) * 100.0
    return float(err)


def train_logistic_regression(X_train, y_train, random_state=42):
    """Huấn luyện mô hình phân loại Logistic Regression."""
    model = LogisticRegression(max_iter=5000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Đánh giá hiệu năng phân loại (lớp 0 = Ác tính/Malignant là lớp dương tính lâm sàng)."""
    preds = model.predict(X_test)
    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Malignant_Precision": float(precision_score(y_test, preds, pos_label=0, zero_division=0)),
        "Malignant_Recall": float(recall_score(y_test, preds, pos_label=0, zero_division=0)),
        "Malignant_F1": float(f1_score(y_test, preds, pos_label=0, zero_division=0)),
        "Confusion_Matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    }
    return metrics, preds


def get_environment_metadata():
    """Ghi nhận thông tin phiên bản môi trường để đảm bảo tính tái lập (Reproducibility)."""
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
    """Lưu kết quả định lượng dạng JSON vào thư mục outputs/."""
    folder = Path(__file__).resolve().parent / "outputs"
    folder.mkdir(exist_ok=True)
    path = folder / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu JSON: {path}")
