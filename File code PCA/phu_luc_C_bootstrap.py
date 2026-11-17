"""Phụ lục C: Kiểm định Bootstrap (1.000 lần lấy mẫu lại) ước lượng khoảng tin cậy 95% trên tập Test."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from common import (
    load_data,
    split_data,
    standardize_data,
    reduce_with_pca,
    train_logistic_regression,
    get_environment_metadata,
    save_json_result,
)


def main():
    base_dir = Path(__file__).resolve().parent

    X, y, _, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # So sánh Baseline 30D và PCA tối ưu q = 10
    base_clf = train_logistic_regression(X_train_scaled, y_train)
    pred_base = base_clf.predict(X_test_scaled)

    X_tr_pca, X_te_pca, _ = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=10)
    pca_clf = train_logistic_regression(X_tr_pca, y_train)
    pred_pca = pca_clf.predict(X_te_pca)

    rng = np.random.default_rng(42)
    n_bootstrap = 1000
    boot_pca = np.empty(n_bootstrap)
    boot_diff = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.integers(0, len(y_test), size=len(y_test))
        acc_pca = accuracy_score(y_test[indices], pred_pca[indices])
        acc_base = accuracy_score(y_test[indices], pred_base[indices])
        boot_pca[b] = acc_pca
        boot_diff[b] = acc_pca - acc_base

    mean_score = float(np.mean(boot_pca))
    lower, upper = np.percentile(boot_pca, [2.5, 97.5]).tolist()
    diff_lower, diff_upper = np.percentile(boot_diff, [2.5, 97.5]).tolist()

    print("===== BOOTSTRAP RESULTS =====")
    print("Number of Bootstrap samples:", n_bootstrap)
    print("Mean accuracy (PCA q=10):", round(mean_score, 4))
    print("95% interval:", round(lower, 4), "-", round(upper, 4))
    print("Diff (PCA 10D - Original 30D) 95% CI:", round(diff_lower, 4), "-", round(diff_upper, 4))

    pd.DataFrame({
        "Bootstrap Accuracy": boot_pca,
        "Diff_PCA10_vs_Original30": boot_diff,
    }).to_csv(base_dir / "bootstrap_results.csv", index=False)

    meta = get_environment_metadata()
    meta["bootstrap"] = {
        "seed": 42,
        "resamples": n_bootstrap,
        "pca10_mean_accuracy": mean_score,
        "ci": [lower, upper],
        "diff_vs_original_95_ci": [diff_lower, diff_upper],
    }
    save_json_result("results_C.json", meta)


if __name__ == "__main__":
    main()
