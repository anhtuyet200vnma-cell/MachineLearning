"""Case Study chính: Giảm chiều dữ liệu bằng PCA trên tập Breast Cancer Wisconsin."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    load_data,
    split_data,
    standardize_data,
    check_pca_suitability,
    reduce_with_pca,
    compute_reconstruction_error,
    train_logistic_regression,
    evaluate_model,
)


def main():
    base_dir = Path(__file__).resolve().parent

    # 1. LOAD DATA
    print("===== BREAST CANCER WISCONSIN - PCA CASE STUDY =====")
    X, y, target_names, feature_names = load_data()
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    # 2. SPLIT DATA 80/20 (STRATIFIED)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. KIỂM ĐỊNH TIÊN QUYẾT BARTLETT & KMO
    suitability = check_pca_suitability(X_train)
    print(
        f"Bartlett Chi2 = {suitability['chi2']:.1f} (df={suitability['df']}, p < 1e-300) "
        f"| KMO = {suitability['kmo']:.4f}"
    )

    # 4. STANDARDIZE DATA (CHỐNG DATA LEAKAGE)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # 5. BASELINE MODEL: ORIGINAL 30 FEATURES
    print("\n===== BASELINE MODEL (ORIGINAL 30D) =====")
    baseline_model = train_logistic_regression(X_train_scaled, y_train)
    base_metrics, _ = evaluate_model(baseline_model, X_test_scaled, y_test)
    print("Original dimensions:", X_train_scaled.shape[1])
    print("Baseline accuracy:", round(base_metrics["Accuracy"], 4))
    print("Baseline Malignant Recall:", round(base_metrics["Malignant_Recall"], 4))

    # 6. PCA FEATURE REDUCTION ACROSS DIMENSIONS
    dimensions = [1, 2, 3, 5, 6, 7, 10, 15]
    results = [
        {
            "Method": "Original",
            "Dimensions": 30,
            "CEV (%)": 100.00,
            "Recon_Error (%)": 0.00,
            "Accuracy": base_metrics["Accuracy"],
            "Malignant_Precision": base_metrics["Malignant_Precision"],
            "Malignant_Recall": base_metrics["Malignant_Recall"],
            "Malignant_F1": base_metrics["Malignant_F1"],
        }
    ]

    for q in dimensions:
        print(f"\n===== PCA {q} DIMENSIONS =====")
        X_train_pca, X_test_pca, pca = reduce_with_pca(
            X_train_scaled, X_test_scaled, n_components=q
        )
        cev = float(pca.explained_variance_ratio_.sum() * 100.0)
        recon_err = compute_reconstruction_error(X_test_scaled, X_test_pca, pca)

        model = train_logistic_regression(X_train_pca, y_train)
        metrics, _ = evaluate_model(model, X_test_pca, y_test)

        results.append(
            {
                "Method": "PCA",
                "Dimensions": q,
                "CEV (%)": round(cev, 2),
                "Recon_Error (%)": round(recon_err, 2),
                "Accuracy": metrics["Accuracy"],
                "Malignant_Precision": metrics["Malignant_Precision"],
                "Malignant_Recall": metrics["Malignant_Recall"],
                "Malignant_F1": metrics["Malignant_F1"],
            }
        )

        print(f"Reduced dimensions: {q} | CEV: {cev:.2f}% | Recon Error: {recon_err:.2f}%")
        print(f"Test accuracy: {metrics['Accuracy']:.4f} | Malignant Recall: {metrics['Malignant_Recall']:.4f}")

    # 7. MODEL COMPARISON TABLE & CSV EXPORT
    comparison = pd.DataFrame(results)
    print("\n===== FINAL COMPARISON =====")
    print(comparison.to_string(index=False))

    csv_path = base_dir / "pca_results.csv"
    comparison.to_csv(csv_path, index=False)

    # 8. ACCURACY VISUALIZATION
    plot_subset = comparison[comparison["Dimensions"].isin([30, 2, 5, 6, 10])].copy()
    labels = [
        f"Original\n(30D)" if m == "Original" else f"PCA\n(q={d})"
        for m, d in zip(plot_subset["Method"], plot_subset["Dimensions"])
    ]
    accuracies = plot_subset["Accuracy"].tolist()
    colors = ["#1f4e78", "#7fb3d5", "#5499c7", "#2980b9", "#1b7837"]

    plt.figure(figsize=(9.5, 5.2))
    bars = plt.bar(labels, accuracies, color=colors, width=0.55)
    plt.title("Logistic Regression: Original (30D) vs. PCA Dimensionality Reduction", fontweight="bold")
    plt.xlabel("Feature Representation & Dimensions")
    plt.ylabel("Test Accuracy (n = 114)")
    plt.ylim(0.85, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.2%}",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig_path = base_dir / "pca_accuracy_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved {csv_path.name} and {fig_path.name}!")


if __name__ == "__main__":
    main()
