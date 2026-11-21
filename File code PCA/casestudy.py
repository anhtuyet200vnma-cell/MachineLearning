"""Main case study: PCA dimensionality reduction on Breast Cancer Wisconsin."""
import numpy as np
import pandas as pd

from common import (
    BASE_DIR, FIG_DIR,
    load_data, split_data, standardize_data, check_pca_suitability,
    correlation_eigenvalues, horn_parallel_analysis, count_kaiser, count_horn,
    smallest_q_for_cev, reduce_with_pca, compute_reconstruction_error,
    train_logistic_regression, evaluate_model, set_plot_style,
)


def main():
    plt = set_plot_style()

    # 1. LOAD DATA
    print("===== BREAST CANCER WISCONSIN - PCA CASE STUDY =====")
    X, y, target_names, feature_names = load_data()
    print("Number of samples:", X.shape[0], "| class counts (0=malignant, 1=benign):", np.bincount(y).tolist())
    print("Number of features:", X.shape[1])

    # 2. STRATIFIED 80/20 SPLIT
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print("Training samples:", len(X_train), np.bincount(y_train).tolist(),
          "| Testing samples:", len(X_test), np.bincount(y_test).tolist())

    # 3. BARTLETT & KMO ON THE TRAINING SET
    s = check_pca_suitability(X_train)
    print(f"Bartlett chi2 = {s['chi2']:.1f} (df={s['df']}, log10 p = {s['log10_p']:.0f}) | KMO = {s['kmo']:.4f}")
    print(f"Per-variable MSA: min = {s['msa'].min():.3f} ({feature_names[int(s['msa'].argmin())]}), "
          f"max = {s['msa'].max():.3f} ({feature_names[int(s['msa'].argmax())]})")

    # 4. STANDARDIZE (TRAIN-ONLY FIT)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # 5. RULES FOR CHOOSING q
    _, _, pca_full = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=30)
    eig = correlation_eigenvalues(X_train_scaled)
    evr = pca_full.explained_variance_ratio_
    horn = horn_parallel_analysis(*X_train_scaled.shape)
    print("\n===== NUMBER-OF-COMPONENTS RULES =====")
    print(f"Kaiser (lambda > 1): q = {count_kaiser(eig)} | Horn parallel analysis: q = {count_horn(eig, horn)}")
    print(f"CEV >= 90%: q = {smallest_q_for_cev(evr, 0.90)} | CEV >= 95%: q = {smallest_q_for_cev(evr, 0.95)}")

    # 6. BASELINE: ORIGINAL 30 FEATURES
    print("\n===== BASELINE MODEL (ORIGINAL 30D) =====")
    baseline_model = train_logistic_regression(X_train_scaled, y_train)
    base_metrics, _ = evaluate_model(baseline_model, X_test_scaled, y_test)
    print("Baseline accuracy:", round(base_metrics["Accuracy"], 4),
          "| Malignant recall:", round(base_metrics["Malignant_Recall"], 4))

    def cm_cols(m):
        (tp, fn), (fp, tn) = m["Confusion_Matrix"]   # rows = true [malignant, benign]
        return {"Mal_TP": tp, "Mal_FN": fn, "Mal_FP": fp, "Mal_TN": tn}

    results = [{
        "Method": "Original", "Dimensions": 30, "CEV (%)": 100.00, "Recon_Error (%)": 0.00,
        "Accuracy": base_metrics["Accuracy"],
        "Malignant_Precision": base_metrics["Malignant_Precision"],
        "Malignant_Recall": base_metrics["Malignant_Recall"],
        "Malignant_F1": base_metrics["Malignant_F1"], **cm_cols(base_metrics),
    }]

    # 7. PCA ACROSS DIMENSIONS
    for q in [1, 2, 3, 5, 6, 7, 10, 15]:
        X_train_pca, X_test_pca, pca = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=q)
        cev = float(pca.explained_variance_ratio_.sum() * 100.0)
        recon_err = compute_reconstruction_error(X_test_scaled, X_test_pca, pca)
        model = train_logistic_regression(X_train_pca, y_train)
        m, _ = evaluate_model(model, X_test_pca, y_test)
        results.append({
            "Method": "PCA", "Dimensions": q, "CEV (%)": round(cev, 2),
            "Recon_Error (%)": round(recon_err, 2), "Accuracy": m["Accuracy"],
            "Malignant_Precision": m["Malignant_Precision"],
            "Malignant_Recall": m["Malignant_Recall"],
            "Malignant_F1": m["Malignant_F1"], **cm_cols(m),
        })
        print(f"PCA q={q:2d} | CEV {cev:6.2f}% | recon err {recon_err:5.2f}% | "
              f"acc {m['Accuracy']:.4f} | malignant recall {m['Malignant_Recall']:.4f}")

    comparison = pd.DataFrame(results)
    print("\n===== FINAL COMPARISON =====")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(BASE_DIR / "pca_results.csv", index=False)

    # 8. FIGURE: (a) test accuracy, (b) CEV and reconstruction error versus q
    sub = comparison[comparison["Dimensions"].isin([30, 2, 5, 6, 10])]
    labels = ["Original\n(30D)" if m == "Original" else f"PCA\n(q={d})"
              for m, d in zip(sub["Method"], sub["Dimensions"])]
    acc = sub["Accuracy"].tolist()
    colors = ["#1f4e78", "#7fb3d5", "#5499c7", "#2980b9", "#1b7837"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(labels, acc, color=colors, width=0.55)
    axes[0].set_ylim(0.85, 1.02)
    axes[0].set_ylabel("Test accuracy (n = 114)")
    axes[0].set_title("(a) Logistic regression: 30D vs. PCA (y-axis starts at 0.85)")
    for b, v in zip(bars, acc):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2%}", ha="center", fontsize=9, fontweight="bold")
    pca_rows = comparison[comparison["Method"] == "PCA"]
    axes[1].plot(pca_rows["Dimensions"], pca_rows["CEV (%)"], "o-", color="#1b5e20", label="Cumulative explained variance (train)")
    axes[1].plot(pca_rows["Dimensions"], 100 - pca_rows["Recon_Error (%)"], "s--", color="#c62828", label="100 - relative reconstruction error (test)")
    axes[1].axhline(95, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Number of retained components q")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("(b) Information retained versus q")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_accuracy_comparison.png")
    plt.close()
    print("\nSaved pca_results.csv and figures/pca_accuracy_comparison.png")


if __name__ == "__main__":
    main()
