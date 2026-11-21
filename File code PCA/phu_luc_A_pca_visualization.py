"""Appendix A: PCA visualisations - biplot, scree/Kaiser/Horn, scaling effect,
loadings heat-map and the 5D -> 2D variance profile."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (
    FIG_DIR, load_data, split_data, standardize_data, reduce_with_pca,
    correlation_eigenvalues, horn_parallel_analysis, count_kaiser, count_horn,
    smallest_q_for_cev, set_plot_style,
)


def main():
    plt = set_plot_style()
    X, y, target_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # ---------------- Figure A1: 2D score plot with loading arrows ----------------
    X_2d, _, pca_2d = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=2)
    evr_2d = pca_2d.explained_variance_ratio_ * 100.0
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for cls, label, color in [(0, "Malignant", "#d32f2f"), (1, "Benign", "#1976d2")]:
        m = y_train == cls
        ax.scatter(X_2d[m, 0], X_2d[m, 1], c=color, s=26, alpha=0.75, edgecolors="none", label=f"{label} (n = {m.sum()})")
    # arrows: the two largest |weights| on PC1 and on PC2 (selected automatically)
    chosen = []
    for k in (0, 1):
        for j in np.argsort(-np.abs(pca_2d.components_[k])):
            if j not in chosen:
                chosen.append(int(j))
            if len(chosen) >= 2 * (k + 1):
                break
    scale_arrow = 0.85 * np.abs(X_2d).max() / np.abs(pca_2d.components_[:2, chosen]).max()
    for j in chosen:
        vx, vy = pca_2d.components_[0, j] * scale_arrow, pca_2d.components_[1, j] * scale_arrow
        ax.arrow(0, 0, vx, vy, color="#2e7d32", alpha=0.9, head_width=0.22, linewidth=1.3, length_includes_head=True)
        ax.text(vx * 1.08, vy * 1.08, feature_names[j], color="#1b5e20", fontsize=8.5, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.axhline(0, color="gray", ls=":", lw=0.8); ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(f"PC1 ({evr_2d[0]:.2f}% of variance)")
    ax.set_ylabel(f"PC2 ({evr_2d[1]:.2f}% of variance)")
    ax.set_title(f"PCA biplot of the training set (CEV = {evr_2d.sum():.2f}%)")
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pca_2d_visualization.png"); plt.close()

    # ---------------- Figure A2: scree plot, Kaiser, Horn, CEV ----------------
    pca_full = PCA().fit(X_train_scaled)
    eig = correlation_eigenvalues(X_train_scaled)
    evr = pca_full.explained_variance_ratio_ * 100.0
    cev = np.cumsum(evr)
    horn = horn_parallel_analysis(*X_train_scaled.shape)
    q_k, q_h = count_kaiser(eig), count_horn(eig, horn)
    q90, q95 = smallest_q_for_cev(evr / 100, 0.90), smallest_q_for_cev(evr / 100, 0.95)
    ks = np.arange(1, 16)
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    cols = ["#1976d2" if k <= q_h else ("#64b5f6" if k <= q_k else ("#2e7d32" if k <= q95 else "#b0bec5")) for k in ks]
    ax.bar(ks, eig[:15], color=cols, alpha=0.9, label="Eigenvalue $\\lambda_k$ of R")
    ax.axhline(1.0, color="#d32f2f", ls="--", lw=1.3, label=f"Kaiser rule ($\\lambda$ = 1) $\\Rightarrow$ q = {q_k}")
    ax.plot(ks, horn[:15], color="#ef6c00", ls="-.", marker="^", ms=4.5, lw=1.4, label=f"Horn parallel analysis (95th pct) $\\Rightarrow$ q = {q_h}")
    ax.set_xlabel("Principal component k"); ax.set_ylabel("Eigenvalue $\\lambda_k$"); ax.set_xticks(ks)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ks, cev[:15], color="#1b5e20", marker="o", lw=2, label="Cumulative variance (CEV, %)")
    ax2.axhline(90, color="#9e9d24", ls=":", lw=1.2, label=f"90% CEV $\\Rightarrow$ q = {q90}")
    ax2.axhline(95, color="#388e3c", ls=":", lw=1.4, label=f"95% CEV $\\Rightarrow$ q = {q95}")
    ax2.set_ylabel("Cumulative explained variance (%)"); ax2.set_ylim(0, 105)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    ax.set_title("Scree plot with Kaiser, Horn and cumulative-variance criteria")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pca_scree_plot.png"); plt.close()

    # ---------------- Figure A3: effect of scaling ----------------
    evr_unscaled = PCA().fit(X_train).explained_variance_ratio_ * 100.0
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xi, w = np.arange(5), 0.36
    ax.bar(xi - w / 2, evr_unscaled[:5], w, color="#c62828", alpha=0.9, label="Unscaled (covariance matrix S)")
    ax.bar(xi + w / 2, evr[:5], w, color="#1565c0", alpha=0.9, label="Standardized (correlation matrix R)")
    for i in range(5):
        ax.text(xi[i] - w / 2, evr_unscaled[i] + 1.2, f"{evr_unscaled[i]:.1f}%", ha="center", fontsize=8, color="#b71c1c", fontweight="bold")
        ax.text(xi[i] + w / 2, evr[i] + 1.2, f"{evr[i]:.1f}%", ha="center", fontsize=8, color="#0d47a1", fontweight="bold")
    ax.set_xticks(xi); ax.set_xticklabels([f"PC{i+1}" for i in xi]); ax.set_ylim(0, 108)
    ax.set_ylabel("Explained variance ratio (%)"); ax.set_title("Impact of z-score standardization on PCA")
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pca_scaling_effect.png"); plt.close()

    # ---------------- Figure A4: loadings heat-map ----------------
    L = pca_full.components_[:3].T * np.sqrt(eig[:3] * 1.0)      # corr(x_j, z_k) for standardized data
    order = np.argsort(-np.max(np.abs(L), axis=1))[:14]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(L[order], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["PC1", "PC2", "PC3"])
    ax.set_yticks(range(len(order))); ax.set_yticklabels([feature_names[i] for i in order], fontsize=8.5)
    for r, i in enumerate(order):
        for c in range(3):
            ax.text(c, r, f"{L[i, c]:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(L[i, c]) > 0.6 else "black")
    ax.grid(False); ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label="Loading = corr($x_j$, $z_k$)")
    ax.set_title("Loadings of the 14 most influential features")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pca_loadings_heatmap.png"); plt.close()

    # ---------------- Figure A5: 5D -> 2D variance profile ----------------
    idx_5d = [0, 2, 3, 5, 4]
    pca5 = PCA().fit(StandardScaler().fit_transform(X_train[:, idx_5d]))
    e5 = pca5.explained_variance_ratio_ * 100.0
    c5 = np.cumsum(e5)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhspan(90, 95, color="#c8e6c9", alpha=0.6, label="90-95% target band")
    colors5 = ["#1976d2", "#1976d2", "#b0bec5", "#b0bec5", "#b0bec5"]
    bars = ax.bar(range(1, 6), e5, color=colors5, width=0.55)
    for b, v in zip(bars, e5):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.plot(range(1, 6), c5, "o-", color="#1b5e20", lw=2, label="Cumulative variance")
    ax.annotate(f"CEV(2) = {c5[1]:.2f}%", (2, c5[1]), textcoords="offset points", xytext=(14, -22), ha="left",
                fontsize=9, color="#1b5e20", fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1b5e20", alpha=0.9))
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f"PC{k}\n({'keep' if k <= 2 else 'drop'})" for k in range(1, 6)])
    ax.set_ylim(0, 108); ax.set_ylabel("Percent of total variance")
    ax.set_title("5D to 2D: variance profile of the five morphology features")
    ax.legend(loc="center right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pca_5d_variance.png"); plt.close()

    # ---------------- Figure A6: three classifiers versus q (needs pca_experiment.py output) ----------------
    import pandas as pd
    from common import BASE_DIR
    csv = BASE_DIR / "multi_classifier_results.csv"
    if csv.exists():
        bench = pd.read_csv(csv)
        order = [1, 2, 3, 5, 6, 7, 10, 15, 30]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for ax, col, title in [(axes[0], "Accuracy", "(a) Test accuracy"), (axes[1], "Malignant_Recall", "(b) Malignant recall (sensitivity)")]:
            for name, mk, c in [("Logistic Regression", "o", "#1565c0"), ("SVM (RBF)", "s", "#2e7d32"), ("Random Forest", "^", "#ef6c00")]:
                sub = bench[bench.Classifier == name].set_index("Dimensions").loc[order]
                ls, lw = ("--", 1.4) if name == "SVM (RBF)" else ("-", 2.4)
                ax.plot(range(len(order)), sub[col] * 100, marker=mk, color=c, lw=lw, ls=ls, label=name, zorder=3 if ls == "--" else 2)
            ax.set_xticks(range(len(order))); ax.set_xticklabels([str(o) if o != 30 else "30\n(orig.)" for o in order])
            ax.set_xlabel("Number of retained components q"); ax.set_ylabel(col.replace("_", " ") + " (%)"); ax.set_title(title)
        axes[0].legend(loc="lower right")
        plt.tight_layout(); plt.savefig(FIG_DIR / "multi_classifier_vs_q.png"); plt.close()

    print(f"Rules: Kaiser q = {q_k}, Horn q = {q_h}, 90% CEV q = {q90}, 95% CEV q = {q95}")
    print("Biplot arrows:", [feature_names[j] for j in chosen])
    print("Top |loadings| (PC1, PC2, PC3):")
    for k in range(3):
        j = np.argsort(-np.abs(L[:, k]))[:4]
        print(f"  PC{k+1}:", [(feature_names[i], round(float(L[i, k]), 3)) for i in j])
    print("Saved figures in figures/")


if __name__ == "__main__":
    main()
