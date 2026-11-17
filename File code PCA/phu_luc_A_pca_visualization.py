"""Phụ lục A: Trực quan hóa hình học PCA (2D Biplot, Scree Plot, Kaiser, Horn PA & Scaling)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import (
    load_data,
    split_data,
    standardize_data,
    reduce_with_pca,
    horn_parallel_analysis,
)


def main():
    base_dir = Path(__file__).resolve().parent

    X, y, target_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # --- HÌNH A1: PCA 2D VISUALIZATION (BIPLOT PC1 vs PC2) ---
    X_train_2d, _, pca_2d = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=2)
    evr_2d = pca_2d.explained_variance_ratio_ * 100.0

    plt.figure(figsize=(8.8, 6.4))
    for cls_idx, (label, color) in enumerate(
        zip(["Malignant (Ác tính)", "Benign (Lành tính)"], ["#d32f2f", "#1976d2"])
    ):
        mask = y_train == cls_idx
        plt.scatter(
            X_train_2d[mask, 0],
            X_train_2d[mask, 1],
            c=color,
            label=label,
            s=30,
            alpha=0.82,
            edgecolors="none",
        )

    # Vẽ top 4 vector trọng số Loadings tiêu biểu
    top_features = [7, 6, 27, 9]
    scale_arrow = 8.2
    for f_idx in top_features:
        vx = pca_2d.components_[0, f_idx] * scale_arrow
        vy = pca_2d.components_[1, f_idx] * scale_arrow
        plt.arrow(0, 0, vx, vy, color="#2e7d32", alpha=0.85, head_width=0.25, linewidth=1.3)
        plt.text(
            vx * 1.12,
            vy * 1.12,
            feature_names[f_idx],
            color="#1b5e20",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.title(
        f"PCA 2D Projection & Feature Loadings - Breast Cancer (CEV = {evr_2d.sum():.2f}%)",
        fontweight="bold",
    )
    plt.xlabel(f"PC1 ({evr_2d[0]:.2f}% Variance - Cell Size & Concavity)")
    plt.ylabel(f"PC2 ({evr_2d[1]:.2f}% Variance - Fractal & Texture Complexity)")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_dir / "pca_2d_visualization.png", dpi=300)
    plt.close()

    # --- HÌNH A2: SCREE PLOT + KAISER + HORN PA + SCALING DISTORTION ---
    _, _, pca_full = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=30)
    eigenvalues = pca_full.explained_variance_
    evr = pca_full.explained_variance_ratio_ * 100.0
    cev = np.cumsum(evr)
    pa_thresh = horn_parallel_analysis(X_train_scaled.shape[0], X_train_scaled.shape[1])

    pca_unscaled = PCA().fit(X_train)
    evr_unscaled = pca_unscaled.explained_variance_ratio_ * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Scree Plot + Kaiser Rule + Horn Parallel Analysis + CEV 95%
    ax1 = axes[0]
    k_range = np.arange(1, 16)
    bar_colors = ["#1976d2" if k <= 6 else ("#2e7d32" if k <= 10 else "#b0bec5") for k in k_range]
    ax1.bar(k_range, eigenvalues[:15], color=bar_colors, alpha=0.85, label="Eigenvalue λk")
    ax1.axhline(1.0, color="#d32f2f", linestyle="--", linewidth=1.3, label="Kaiser Rule (λ = 1.0 → q = 6)")
    ax1.plot(
        k_range,
        pa_thresh[:15],
        color="#ef6c00",
        linestyle="-.",
        marker="^",
        markersize=5,
        linewidth=1.4,
        label="Horn Parallel Analysis (95th pct)",
    )
    ax1.set_title("A. Scree Plot, Kaiser Rule & CEV 95% Threshold", fontweight="bold")
    ax1.set_xlabel("Principal Component k (PC1 → PC15)")
    ax1.set_ylabel("Eigenvalue λk", color="#1976d2")
    ax1.set_xticks(k_range)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_range, cev[:15], color="#1b5e20", marker="o", linewidth=2, label="Cumulative Variance CEV (%)")
    ax1_twin.axhline(95.0, color="#388e3c", linestyle=":", linewidth=1.3, label="95% CEV Threshold (q = 10)")
    ax1_twin.set_ylabel("Cumulative Explained Variance (%)", color="#1b5e20")
    ax1_twin.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    # Subplot 2: Unscaled (Covariance) vs Scaled (Correlation) Variance Distortion
    ax2 = axes[1]
    x_idx = np.arange(5)
    width = 0.36
    ax2.bar(x_idx - width / 2, evr_unscaled[:5], width=width, color="#c62828", alpha=0.85, label="Unscaled (Covariance S)")
    ax2.bar(x_idx + width / 2, evr[:5], width=width, color="#1565c0", alpha=0.85, label="Standardized (Correlation R)")
    ax2.set_title("B. Impact of Z-Score Standardization on Explained Variance", fontweight="bold")
    ax2.set_xlabel("Top 5 Principal Components")
    ax2.set_ylabel("Explained Variance Ratio (%)")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(["PC1", "PC2", "PC3", "PC4", "PC5"])
    ax2.set_ylim(0, 108)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right")

    for i in range(5):
        ax2.text(x_idx[i] - width / 2, evr_unscaled[i] + 1.5, f"{evr_unscaled[i]:.1f}%", ha="center", fontsize=8, color="#b71c1c", fontweight="bold")
        ax2.text(x_idx[i] + width / 2, evr[i] + 1.5, f"{evr[i]:.1f}%", ha="center", fontsize=8, color="#0d47a1", fontweight="bold")

    plt.tight_layout()
    plt.savefig(base_dir / "pca_scree_and_scaling.png", dpi=300)
    plt.close()
    print("Saved pca_2d_visualization.png and pca_scree_and_scaling.png!")


if __name__ == "__main__":
    main()
