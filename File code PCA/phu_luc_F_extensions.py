"""Appendix F: limits of linear PCA and extensions - PCA vs LDA, Kernel PCA, and an
image-compression demonstration on the scikit-learn digits dataset."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from common import BASE_DIR, FIG_DIR, set_plot_style

BLUE, RED, GREEN, GRAY = "#1565c0", "#c62828", "#2e7d32", "#607d8b"


def pca_vs_lda(plt):
    """Two elongated Gaussian classes whose separation lies along the MINOR axis."""
    rng = np.random.default_rng(7)
    cov = np.array([[9.0, 0.0], [0.0, 0.5]])
    X = np.vstack([rng.multivariate_normal([0, 1.4], cov, 250), rng.multivariate_normal([0, -1.4], cov, 250)])
    y = np.r_[np.zeros(250, int), np.ones(250, int)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    pca = PCA(n_components=1).fit(Xtr)
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xtr, ytr)
    acc_pca = LogisticRegression().fit(pca.transform(Xtr), ytr).score(pca.transform(Xte), yte)
    acc_lda = LogisticRegression().fit(lda.transform(Xtr), ytr).score(lda.transform(Xte), yte)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    for c, col, lab in [(0, BLUE, "class A"), (1, RED, "class B")]:
        ax.scatter(Xtr[ytr == c, 0], Xtr[ytr == c, 1], s=12, color=col, alpha=0.6, label=lab)
    t = np.linspace(-9, 9, 10)
    d_pca = pca.components_[0]; d_lda = lda.scalings_[:, 0] / np.linalg.norm(lda.scalings_[:, 0])
    ax.plot(t * d_pca[0], t * d_pca[1], color="black", lw=2, label="PC1 (max variance)")
    ax.plot(t * d_lda[0] * 0.4, t * d_lda[1] * 0.4, color=GREEN, lw=2.4, label="LDA axis (max separation)")
    ax.set_xlim(-10, 10); ax.set_ylim(-5.5, 5.5); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("(a) PC1 ignores the labels"); ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    for ax, proj, name, acc in [(axes[1], pca.transform(Xtr)[:, 0], "PCA (1 component)", acc_pca), (axes[2], lda.transform(Xtr)[:, 0], "LDA (1 component)", acc_lda)]:
        ax.hist(proj[ytr == 0], bins=30, color=BLUE, alpha=0.65); ax.hist(proj[ytr == 1], bins=30, color=RED, alpha=0.65)
        ax.set_title(f"{name}: test acc = {acc*100:.1f}%"); ax.set_xlabel("1D projection"); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_pca_vs_lda.png"); plt.close()
    return {"pca_1d_test_acc": acc_pca, "lda_1d_test_acc": acc_lda}


def kernel_pca(plt):
    rows = []
    for factor in (0.3, 0.45):
        X, y = make_circles(n_samples=300, factor=factor, noise=0.05, random_state=42)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        for k in (1, 2):
            p = PCA(n_components=k).fit(Xtr)
            rows.append({"factor": factor, "Method": "Linear PCA", "n_components": k, "gamma": np.nan, "cv_acc": np.nan,
                         "test_acc": LogisticRegression().fit(p.transform(Xtr), ytr).score(p.transform(Xte), yte)})
        for k in (1, 2, 3):
            pipe = Pipeline([("kpca", KernelPCA(kernel="rbf")), ("lr", LogisticRegression())])
            gs = GridSearchCV(pipe, {"kpca__gamma": [0.5, 1, 2, 5, 10, 15, 20, 30], "kpca__n_components": [k]},
                              cv=StratifiedKFold(5, shuffle=True, random_state=42)).fit(Xtr, ytr)
            rows.append({"factor": factor, "Method": "RBF Kernel PCA", "n_components": k, "gamma": gs.best_params_["kpca__gamma"],
                         "cv_acc": gs.best_score_, "test_acc": gs.score(Xte, yte)})
    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "kernel_pca_results.csv", index=False)

    X, y = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    g = float(df[(df.factor == 0.3) & (df.Method == "RBF Kernel PCA") & (df.n_components == 2)].gamma.iloc[0])
    lin = PCA(n_components=2).fit(Xtr); kp = KernelPCA(n_components=2, kernel="rbf", gamma=g).fit(Xtr)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (Z, title) in zip(axes, [(Xtr, "(a) Original data (two rings)"), (lin.transform(Xtr), "(b) Linear PCA (2 components)"),
                                     (kp.transform(Xtr), f"(c) RBF Kernel PCA ($\\gamma$ = {g:g})")]):
        for c, col in [(0, BLUE), (1, RED)]:
            ax.scatter(Z[ytr == c, 0], Z[ytr == c, 1], s=14, color=col, alpha=0.7)
        ax.set_title(title)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[2].set_xlabel("Kernel PC1"); axes[2].set_ylabel("Kernel PC2")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_kernel_pca.png"); plt.close()
    return df


def digits_demo(plt):
    d = load_digits()
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pca = PCA().fit(Xtr)                      # pixels share one scale (0-16): covariance PCA is appropriate
    cev = np.cumsum(pca.explained_variance_ratio_) * 100
    q90, q95, q99 = (int(np.searchsorted(cev, t) + 1) for t in (90, 95, 99))
    rows = []
    for q in (2, 5, 10, 20, 30, 40, 64):
        p = PCA(n_components=q).fit(Xtr)
        Ztr, Zte = p.transform(Xtr), p.transform(Xte)
        rec = p.inverse_transform(Zte)
        err = np.sum((Xte - rec) ** 2) / np.sum((Xte - Xtr.mean(axis=0)) ** 2) * 100
        acc = LogisticRegression(max_iter=5000).fit(Ztr, ytr).score(Zte, yte)
        rows.append({"q": q, "CEV (%)": round(float(cev[q - 1]), 2), "Recon_Error_test (%)": round(float(err), 2),
                     "LogReg test acc": round(float(acc), 4), "compression (%)": round(100 * (1 - q / 64), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "digits_pca_results.csv", index=False)

    # ---- figure 1: CEV curve and 2D projection ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.plot(range(1, 65), cev, color="#1b5e20", lw=2)
    for t, q, c in [(90, q90, "#9e9d24"), (95, q95, "#388e3c"), (99, q99, "#00695c")]:
        ax.axhline(t, color=c, ls=":", lw=1)
        ax.scatter([q], [cev[q - 1]], color=c, zorder=5)
        ax.annotate(f"{t}% at q = {q}", (q, cev[q - 1]), textcoords="offset points", xytext=(10, -14), fontsize=8.5, color=c)
    ax.set_xlabel("Number of components q"); ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("(a) CEV on the digits data (p = 64 pixels)")
    ax = axes[1]
    Z2 = PCA(n_components=2).fit(Xtr).transform(Xtr)
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=ytr, cmap="tab10", s=8, alpha=0.8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("(b) 2D projection coloured by digit class")
    ax.legend(*sc.legend_elements(), ncol=1, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), title="digit", title_fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_digits_pca.png"); plt.close()

    # ---- figure 2: reconstructions of two test digits ----
    qs_show = [None, 2, 5, 10, 20, 40]
    models = {q: PCA(n_components=q).fit(Xtr) for q in qs_show if q}
    fig, axes = plt.subplots(2, len(qs_show), figsize=(9.6, 3.6))
    for r, idx in enumerate([7, 21]):
        for j, q in enumerate(qs_show):
            ax = axes[r, j]; ax.grid(False)
            img = Xte[idx] if q is None else models[q].inverse_transform(models[q].transform(Xte[idx:idx + 1]))[0]
            ax.imshow(img.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("original (64D)" if q is None else f"q = {q}", fontsize=9.5)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_digits_reconstruction.png"); plt.close()
    return df, (q90, q95, q99)


def main():
    plt = set_plot_style()
    r1 = pca_vs_lda(plt)
    print("PCA vs LDA (1 component, test accuracy):", {k: round(v, 4) for k, v in r1.items()})
    pd.DataFrame([{"Method": "PCA (1 component)", "Test accuracy": r1["pca_1d_test_acc"]},
                  {"Method": "LDA (1 component)", "Test accuracy": r1["lda_1d_test_acc"]}]).to_csv(BASE_DIR / "pca_vs_lda_results.csv", index=False)
    df_k = kernel_pca(plt)
    print("\nKernel PCA on concentric circles (train/test = 70/30, gamma chosen by 5-fold CV on train):")
    print(df_k.round(4).to_string(index=False))
    df_d, qs = digits_demo(plt)
    print(f"\nDigits dataset: q for 90/95/99% CEV = {qs}")
    print(df_d.to_string(index=False))


if __name__ == "__main__":
    main()
