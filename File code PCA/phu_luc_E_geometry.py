"""Appendix E: geometric illustrations - variance as information, Pythagorean identity,
four-step transformation, hand-worked example, PCA vs OLS and 3D -> 2D projection."""
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import FIG_DIR, load_data, split_data, set_plot_style

BLUE, RED, GREEN, GRAY, ORANGE = "#1565c0", "#c62828", "#2e7d32", "#607d8b", "#ef6c00"


def hand_example():
    X = np.array([[2, 1], [3, 3], [4, 3], [5, 5], [6, 5]], dtype=float)
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, ddof=1)
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if V[0, 0] < 0:
        V[:, 0] = -V[:, 0]
    if V[1, 1] < 0:
        V[:, 1] = -V[:, 1]
    return X, mu, S, w, V


def pipeline_figure(plt):
    """Flow-chart of the leakage-free PCA pipeline."""
    from matplotlib.patches import FancyBboxPatch
    steps = [("1  Raw data\nX (n x p), labels y", "#eceff1"), ("2  Stratified split\n80% train / 20% test", "#eceff1"),
             ("3  Fit StandardScaler\non TRAIN only", "#bbdefb"), ("4  Bartlett + KMO\n(is PCA suitable?)", "#bbdefb"),
             ("5  Fit PCA (SVD)\non TRAIN only", "#bbdefb"), ("6  Choose q: CEV, Kaiser,\nHorn, cross-validation", "#c8e6c9"),
             ("7  Project train and test\nwith frozen A_q", "#c8e6c9"), ("8  Train classifier on Z_train\nevaluate on Z_test", "#ffe0b2"),
             ("9  Interpret loadings,\nreport CI / robustness", "#ffe0b2")]
    fig, ax = plt.subplots(figsize=(11.5, 4.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0.75, 4.2)
    bw, bh, step = 1.8, 0.95, 1.98
    pos = [(0.1 + step * i, 2.7) for i in range(5)] + [(0.1 + step * (4 - j), 0.9) for j in range(4)]
    for (x, y), (txt, col) in zip(pos, steps):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.08", fc=col, ec="#37474f", lw=1.2))
        ax.text(x + bw / 2, y + bh / 2, txt, ha="center", va="center", fontsize=8.1)
    for i in range(len(pos) - 1):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if abs(y0 - y1) < 1e-6:
            right = x1 > x0
            ax.annotate("", xy=(x1 + (0 if right else bw), y1 + bh / 2), xytext=(x0 + (bw if right else 0), y0 + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.4))
        else:
            ax.annotate("", xy=(x1 + bw / 2, y1 + bh), xytext=(x0 + bw / 2, y0), arrowprops=dict(arrowstyle="-|>", color="#37474f", lw=1.4))
    ax.text(5.0, 4.0, "Blue boxes use the TRAINING split only (no data leakage); the test split is touched only in steps 7-8",
            ha="center", fontsize=9, style="italic")
    plt.savefig(FIG_DIR / "fig_pipeline.png"); plt.close()


def main():
    plt = set_plot_style()
    X, mu, S, lam, A = hand_example()
    a1, a2 = A[:, 0], A[:, 1]
    Xc = X - mu
    z = Xc @ A                      # scores in (PC1, PC2)
    Xhat = np.outer(z[:, 0], a1) + mu
    d2 = np.sum((X - Xhat) ** 2, axis=1)

    # ---------- E1: variance = information (synthetic correlated cloud) ----------
    rng = np.random.default_rng(3)
    C = np.array([[3.0, 2.2], [2.2, 2.4]])
    P = rng.multivariate_normal([0, 0], C, 300)
    w0, V0 = np.linalg.eigh(np.cov(P, rowvar=False)); o = np.argsort(w0)[::-1]; w0, V0 = w0[o], V0[:, o]
    V0[:, 0] *= np.sign(V0[0, 0]); V0[:, 1] *= np.sign(V0[1, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.scatter(P[:, 0], P[:, 1], s=14, color=BLUE, alpha=0.55)
    ang = np.degrees(np.arctan2(V0[1, 0], V0[0, 0]))
    ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(w0[0]), 2 * 2 * np.sqrt(w0[1]), angle=ang, fill=False, ec=RED, lw=1.6, ls="--"))
    for k, c, lab in [(0, RED, f"PC1 ($\\lambda_1$ = {w0[0]:.2f})"), (1, GREEN, f"PC2 ($\\lambda_2$ = {w0[1]:.2f})")]:
        v = V0[:, k] * 2 * np.sqrt(w0[k])
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.text(v[0] + (0.15 if k == 0 else 0.12), v[1] + (-0.25 if k == 0 else 0.05), lab, color=c, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_aspect("equal"); ax.set_xlabel("Feature $X_1$"); ax.set_ylabel("Feature $X_2$")
    ax.set_title("(a) Covariance ellipse (2$\\sigma$) and principal axes")
    ax = axes[1]
    s1, s2 = P @ V0[:, 0], P @ V0[:, 1]
    ax.scatter(s1, np.full_like(s1, 1.0) + rng.uniform(-.12, .12, len(s1)), s=12, color=RED, alpha=0.5)
    ax.scatter(s2, np.full_like(s2, 0.0) + rng.uniform(-.12, .12, len(s2)), s=12, color=GREEN, alpha=0.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"PC2\nVar = {s2.var(ddof=1):.2f}", f"PC1\nVar = {s1.var(ddof=1):.2f}"])
    ax.set_xlabel("Projected coordinate"); ax.set_ylim(-0.6, 1.6)
    ax.set_title("(b) Spread of the projected scores"); ax.grid(axis="y", alpha=0)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_variance_information.png"); plt.close()

    # ---------- E2: Pythagorean identity for one centred observation ----------
    i = 4
    x, zi = Xc[i], Xc[i] @ a1
    proj = zi * a1
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    t = np.linspace(-3.3, 3.3, 10)
    ax.plot(t * a1[0], t * a1[1], color=BLUE, lw=2, label="PC1 axis (unit vector $a_1$)")
    ax.plot([0, proj[0]], [0, proj[1]], color=GREEN, lw=4, solid_capstyle="butt")
    ax.plot([proj[0], x[0]], [proj[1], x[1]], color=RED, lw=3.4, solid_capstyle="butt")
    ax.plot([0, x[0]], [0, x[1]], color="black", lw=1.8, ls="--")
    ax.scatter([0], [0], color="black", zorder=5); ax.scatter([x[0]], [x[1]], color="black", s=55, zorder=5); ax.scatter([proj[0]], [proj[1]], color=BLUE, s=45, zorder=5)
    ax.text(-0.22, -0.38, "O", fontsize=10)
    ax.annotate("$x_{c,i}$ = P5 (centred)", xy=x, xytext=(2.35, 2.55), fontsize=9, arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.annotate("$\\hat{x}_{c,i}$ (projection)", xy=proj, xytext=(0.2, 2.45), fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate(f"$d_{{\\perp,i}}^2$ = {d2[i]:.3f} (error part)", xy=((x[0] + proj[0]) / 2, (x[1] + proj[1]) / 2), xytext=(2.45, 0.75),
                color=RED, fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax.text(proj[0] / 2 + 0.2, proj[1] / 2 - 0.5, f"$z_{{i,1}}^2$ = {zi**2:.3f} (variance part)", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(0.02, x[1] * 0.78 + 0.35, f"$\\|x_{{c,i}}\\|^2$ = {x @ x:.3f}", fontsize=9, rotation=np.degrees(np.arctan2(x[1], x[0])) - 8, ha="left")
    ax.set_xlim(-1.5, 4.6); ax.set_ylim(-1.2, 3.4); ax.set_aspect("equal")
    ax.set_title(f"Pythagorean identity: {x@x:.3f} = {zi**2:.3f} + {d2[i]:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_pythagoras.png"); plt.close()

    # ---------- E3: four-step transformation ----------
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2)); axes = axes.ravel()
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], color=BLUE, s=48, zorder=3)
    for k, (px, py) in enumerate(X):
        ax.text(px + 0.12, py - 0.35, f"P{k+1}", fontsize=8)
    ax.scatter(*mu, marker="X", s=110, color=RED, zorder=4, label=f"centroid $\\mu$ = ({mu[0]:.1f}, {mu[1]:.1f})")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-1, 7); ax.set_ylim(-1, 6.5); ax.set_title("Step 1: raw data (off-centre)"); ax.legend(loc="lower right", fontsize=7.5)
    ax = axes[1]
    ax.scatter(Xc[:, 0], Xc[:, 1], color=BLUE, s=48, zorder=3)
    ell = Ellipse((0, 0), 2 * 2 * np.sqrt(lam[0]), 2 * 2 * np.sqrt(lam[1]), angle=np.degrees(np.arctan2(a1[1], a1[0])), fill=False, ec=RED, ls="--", lw=1.5)
    ax.add_patch(ell)
    for v, c, lab in [(a1 * 2 * np.sqrt(lam[0]), RED, "$a_1$"), (a2 * 2 * np.sqrt(lam[0]) * 0.35, GREEN, "$a_2$")]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2)); ax.text(v[0] * 1.08, v[1] * 1.08, lab, color=c, fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8); ax.set_aspect("equal")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_title("Step 2: centred data, ellipse of S")
    ax = axes[2]
    ax.scatter(z[:, 0], z[:, 1], color=BLUE, s=48, zorder=3)
    for k in range(5):
        ax.text(z[k, 0] + 0.1, z[k, 1] + 0.14, f"P{k+1}", fontsize=8)
    ax.axhline(0, color=GREEN, lw=1.4); ax.axvline(0, color=RED, lw=1.4)
    ax.set_xlim(-4, 4); ax.set_ylim(-1.6, 1.6); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Step 3: rotation $Z = X_c A$ (Cov = 0)")
    ax = axes[3]
    ax.scatter(z[:, 0], z[:, 1], color="#b0bec5", s=40, zorder=2, label="dropped PC2 part")
    ax.scatter(z[:, 0], np.zeros(5), color=BLUE, s=50, zorder=3, label="1D scores $z_{i,1}$")
    for k in range(5):
        ax.plot([z[k, 0]] * 2, [z[k, 1], 0], ls=":", color=GRAY, lw=1)
        ax.text(z[k, 0], -0.28 - 0.3 * (k % 2), f"{z[k,0]:+.2f}", ha="center", fontsize=8)
    ax.axhline(0, color=RED, lw=1.6); ax.set_xlim(-4, 4); ax.set_ylim(-1.6, 1.6); ax.set_xlabel("PC1")
    ax.set_title("Step 4: project onto PC1 (2D $\\to$ 1D)"); ax.legend(loc="upper left", fontsize=7.5)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_four_steps.png"); plt.close()

    # ---------- E4: hand-worked example, trace conservation ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    t = np.linspace(-3.8, 3.8, 10)
    ax.plot(mu[0] + t * a1[0], mu[1] + t * a1[1], color=BLUE, lw=2, label=f"PC1 axis $a_1$ = ({a1[0]:.3f}, {a1[1]:.3f})")
    ax.scatter(X[:, 0], X[:, 1], color=BLUE, s=60, zorder=4, label="observations $x_i$")
    ax.scatter(Xhat[:, 0], Xhat[:, 1], color="black", s=34, zorder=5, label="projections $\\hat{x}_i$")
    for k in range(5):
        ax.plot([X[k, 0], Xhat[k, 0]], [X[k, 1], Xhat[k, 1]], "--", color=GRAY, lw=1.2)
        ax.text(X[k, 0] - 0.45, X[k, 1] + 0.15, f"P{k+1}", fontsize=8.5)
    ax.scatter(*mu, marker="X", s=120, color=RED, zorder=6, label="centroid")
    ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("(a) Orthogonal projection onto PC1"); ax.legend(loc="upper left", fontsize=7.8)
    ax = axes[1]
    bars = ax.bar(["$\\lambda_1$\n(kept)", "$\\lambda_2$\n(discarded)", "trace(S)", "MSE$_\\perp$"], [lam[0], lam[1], np.trace(S), d2.sum() / 4],
                  color=[BLUE, "#b0bec5", "#455a64", RED], width=0.6)
    for b, v in zip(bars, [lam[0], lam[1], np.trace(S), d2.sum() / 4]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 6.2); ax.set_title(f"(b) {lam[0]:.4f} + {lam[1]:.4f} = {np.trace(S):.4f};  MSE$_\\perp$ = $\\lambda_2$")
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_hand_example.png"); plt.close()

    # ---------- E5: PCA versus OLS ----------
    b_ols = S[0, 1] / S[0, 0]; c_ols = mu[1] - b_ols * mu[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = np.linspace(1, 7, 10)
    ax.plot(xs, c_ols + b_ols * xs, color=RED, lw=2, label=f"OLS line (slope {b_ols:.3f}): minimises vertical residuals")
    ax.plot(mu[0] + (xs - mu[0]), mu[1] + (xs - mu[0]) * a1[1] / a1[0], color=BLUE, lw=2, label=f"PC1 line (slope {a1[1]/a1[0]:.3f}): minimises orthogonal residuals")
    ax.scatter(X[:, 0], X[:, 1], color="black", s=45, zorder=5)
    for k in range(5):
        ax.plot([X[k, 0]] * 2, [X[k, 1], c_ols + b_ols * X[k, 0]], color=RED, ls=":", lw=1.4)
        ax.plot([X[k, 0], Xhat[k, 0]], [X[k, 1], Xhat[k, 1]], color=BLUE, ls="--", lw=1.2)
    ax.scatter(*mu, marker="X", s=110, color=GREEN, zorder=6, label="centroid (both lines pass through it)")
    ax.set_xlim(1, 7); ax.set_ylim(0, 6.6); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("PCA (Pearson) versus OLS regression")
    ax.legend(loc="upper left", fontsize=7.8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_pca_vs_ols.png"); plt.close()

    # ---------- E6: 3D -> 2D projection with real data ----------
    Xb, yb, _, fn = load_data()
    Xtr, _, ytr, _ = split_data(Xb, yb)
    cols = [fn.index(c) for c in ["mean radius", "mean texture", "mean concavity"]]
    Z3 = StandardScaler().fit_transform(Xtr[:, cols])
    p3 = PCA(n_components=3).fit(Z3)
    sel = np.random.default_rng(0).choice(len(Z3), 90, replace=False)
    Zs = Z3[sel]; Zp = (Zs @ p3.components_[:2].T) @ p3.components_[:2]
    fig = plt.figure(figsize=(6.8, 5.4)); ax = fig.add_subplot(111, projection="3d")
    g = np.linspace(-3.2, 3.2, 2)
    G1, G2 = np.meshgrid(g, g)
    plane = G1[..., None] * p3.components_[0] + G2[..., None] * p3.components_[1]
    ax.plot_surface(plane[..., 0], plane[..., 1], plane[..., 2], alpha=0.18, color=GREEN)
    ax.scatter(*Zs.T, color=RED, s=14, label="original points in $\\mathbb{R}^3$")
    ax.scatter(*Zp.T, color=BLUE, s=14, label="projections on span(PC1, PC2)")
    for a, b in zip(Zs, Zp):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=GRAY, lw=0.5, alpha=0.7)
    ax.set_xlabel("mean radius (z)"); ax.set_ylabel("mean texture (z)"); ax.set_zlabel("mean concavity (z)")
    ax.view_init(elev=18, azim=-58)
    ax.set_title(f"3D $\\to$ 2D: plane keeps {p3.explained_variance_ratio_[:2].sum()*100:.1f}% of variance"); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "fig_projection_3d.png"); plt.close()

    pipeline_figure(plt)

    # ---------- printed verification ----------
    print("S =", S.tolist(), "| eigenvalues =", np.round(lam, 4), "| a1 =", np.round(a1, 4), "| a2 =", np.round(a2, 4))
    print("scores z1 =", np.round(z[:, 0], 2), "| z2 =", np.round(z[:, 1], 2))
    print("squared errors =", np.round(d2, 3), "sum =", round(float(d2.sum()), 3), "MSE =", round(float(d2.sum() / 4), 4))
    print(f"P5: |x|^2 = {x@x:.4f} = z^2 {zi**2:.4f} + d^2 {d2[4]:.4f}")
    print(f"OLS slope = {b_ols:.4f}, intercept = {c_ols:.4f}; PC1 slope = {a1[1]/a1[0]:.4f}")
    print("3D example EVR (%):", np.round(p3.explained_variance_ratio_ * 100, 2).tolist())
    print("Saved geometry figures in figures/")


if __name__ == "__main__":
    main()
