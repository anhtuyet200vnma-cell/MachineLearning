"""Phụ lục B: Đánh giá chéo 5-Fold Cross-Validation trên tập Train (tránh Data Leakage)."""
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from common import load_data, split_data, get_environment_metadata, save_json_result


def main():
    base_dir = Path(__file__).resolve().parent

    X, y, _, _ = load_data()
    # Chỉ chạy Cross-Validation trên tập Train (n=455) để chọn số chiều q chuẩn khoa học
    X_train, _, y_train, _ = split_data(X, y, test_size=0.2, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1"]

    results = []
    meta = get_environment_metadata()
    meta["cv_results"] = []

    # 1. Baseline: 30 biến gốc chuẩn hóa
    baseline_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ])
    scores = cross_validate(baseline_pipe, X_train, y_train, cv=cv, scoring=scoring)
    row_base = {
        "Method": "Original",
        "Dimensions": 30,
        "Mean Accuracy": float(scores["test_accuracy"].mean()),
        "Std": float(scores["test_accuracy"].std(ddof=1)),
        "Mean F1": float(scores["test_f1"].mean()),
    }
    results.append(row_base)
    meta["cv_results"].append(row_base)

    # 2. Khảo sát các số chiều PCA: q = [1, 2, 3, 5, 6, 7, 10, 15]
    for q in [1, 2, 3, 5, 6, 7, 10, 15]:
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=q, random_state=42)),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ])
        scores = cross_validate(pca_pipe, X_train, y_train, cv=cv, scoring=scoring)
        row = {
            "Method": "PCA",
            "Dimensions": q,
            "Mean Accuracy": float(scores["test_accuracy"].mean()),
            "Std": float(scores["test_accuracy"].std(ddof=1)),
            "Mean F1": float(scores["test_f1"].mean()),
        }
        results.append(row)
        meta["cv_results"].append(row)

    table = pd.DataFrame(results)
    print("===== 5-FOLD CROSS VALIDATION (TRAIN n=455) =====")
    print(table.to_string(index=False))

    csv_path = base_dir / "cross_validation_results.csv"
    table.to_csv(csv_path, index=False)
    save_json_result("results_B.json", meta)


if __name__ == "__main__":
    main()
