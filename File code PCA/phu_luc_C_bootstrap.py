"""Appendix C: bootstrap (B = 1,000) confidence intervals on the TEST set and an exact
McNemar test for the paired comparison PCA(q=10) vs. original 30 features."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score

from common import (BASE_DIR, FIG_DIR, load_data, split_data, standardize_data, reduce_with_pca,
                    train_logistic_regression, get_environment_metadata, save_json_result, set_plot_style)


def main():
    plt = set_plot_style()
    X, y, _, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    base_clf = train_logistic_regression(X_train_scaled, y_train)
    pred_base = base_clf.predict(X_test_scaled)
    X_tr_pca, X_te_pca, _ = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=10)
    pca_clf = train_logistic_regression(X_tr_pca, y_train)
    pred_pca = pca_clf.predict(X_te_pca)

    rng = np.random.default_rng(42)
    B = 1000
    boot_pca, boot_base, boot_diff = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(y_test), size=len(y_test))
        boot_pca[b] = accuracy_score(y_test[idx], pred_pca[idx])
        boot_base[b] = accuracy_score(y_test[idx], pred_base[idx])
        boot_diff[b] = boot_pca[b] - boot_base[b]

    lo, hi = np.percentile(boot_pca, [2.5, 97.5]).tolist()
    blo, bhi = np.percentile(boot_base, [2.5, 97.5]).tolist()
    dlo, dhi = np.percentile(boot_diff, [2.5, 97.5]).tolist()
    acc_pca, acc_base = accuracy_score(y_test, pred_pca), accuracy_score(y_test, pred_base)

    # exact McNemar test on discordant pairs
    ok_pca, ok_base = pred_pca == y_test, pred_base == y_test
    b01 = int(np.sum(ok_base & ~ok_pca))       # original right, PCA wrong
    b10 = int(np.sum(~ok_base & ok_pca))       # original wrong, PCA right
    p_mc = float(binomtest(b01, b01 + b10, 0.5).pvalue) if (b01 + b10) > 0 else 1.0

    print("===== BOOTSTRAP RESULTS (test set n = 114, B = 1000) =====")
    print(f"Observed accuracy: PCA q=10 = {acc_pca:.4f} | Original 30D = {acc_base:.4f}")
    print(f"Bootstrap mean accuracy (PCA q=10): {boot_pca.mean():.4f} | 95% percentile CI: [{lo:.4f}, {hi:.4f}]")
    print(f"Bootstrap mean accuracy (Original): {boot_base.mean():.4f} | 95% percentile CI: [{blo:.4f}, {bhi:.4f}]")
    print(f"Difference (PCA10 - Original) 95% CI: [{dlo:.4f}, {dhi:.4f}] | mean = {boot_diff.mean():.4f}")
    print(f"Discordant pairs: original-correct/PCA-wrong = {b01}, original-wrong/PCA-correct = {b10} | exact McNemar p = {p_mc:.4f}")

    pd.DataFrame({"Bootstrap Accuracy": boot_pca, "Diff_PCA10_vs_Original30": boot_diff,
                  "Bootstrap Accuracy Original30": boot_base}).to_csv(BASE_DIR / "bootstrap_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].hist(boot_pca * 100, bins=np.arange(90, 100.6, 0.5), color="#1976d2", alpha=0.85, edgecolor="white")
    axes[0].axvline(lo * 100, color="#c62828", ls="--"); axes[0].axvline(hi * 100, color="#c62828", ls="--", label="95% CI")
    axes[0].axvline(acc_pca * 100, color="black", lw=1.6, label=f"Observed ({acc_pca*100:.2f}%)")
    axes[0].set_xlabel("Bootstrap accuracy (%)"); axes[0].set_ylabel("Frequency")
    axes[0].set_title("(a) PCA q = 10: bootstrap accuracy"); axes[0].legend()
    axes[1].hist(boot_diff * 100, bins=np.arange(-6.5, 5.6, 0.5), color="#2e7d32", alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.4)
    axes[1].axvline(dlo * 100, color="#c62828", ls="--"); axes[1].axvline(dhi * 100, color="#c62828", ls="--", label="95% CI")
    axes[1].set_xlabel("Accuracy difference, PCA10 $-$ Original (percentage points)"); axes[1].set_ylabel("Frequency")
    axes[1].set_title("(b) Paired difference"); axes[1].legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "bootstrap_distribution.png"); plt.close()

    meta = get_environment_metadata()
    meta["bootstrap"] = {"seed": 42, "resamples": B, "observed_accuracy_pca10": acc_pca, "observed_accuracy_original": acc_base,
                         "pca10_mean_accuracy": float(boot_pca.mean()), "pca10_ci": [lo, hi],
                         "original_ci": [blo, bhi], "diff_95_ci": [dlo, dhi],
                         "mcnemar": {"orig_right_pca_wrong": b01, "orig_wrong_pca_right": b10, "p_exact": p_mc}}
    save_json_result("results_C.json", meta)


if __name__ == "__main__":
    main()
