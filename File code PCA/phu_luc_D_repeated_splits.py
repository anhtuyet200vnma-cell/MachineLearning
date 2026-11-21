"""Appendix D: stability over 12 random stratified Train/Test splits (seeds 0..11)."""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from common import (BASE_DIR, FIG_DIR, load_data, split_data, standardize_data, reduce_with_pca,
                    train_logistic_regression, evaluate_model, get_environment_metadata,
                    save_json_result, set_plot_style)


def main():
    plt = set_plot_style()
    X, y, _, _ = load_data()
    rows = []
    for seed in range(12):
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, random_state=seed)
        X_tr_s, X_te_s, _ = standardize_data(X_tr, X_te)
        base_m, _ = evaluate_model(train_logistic_regression(X_tr_s, y_tr), X_te_s, y_te)
        X_tr_p, X_te_p, pca = reduce_with_pca(X_tr_s, X_te_s, n_components=10)
        pca_m, _ = evaluate_model(train_logistic_regression(X_tr_p, y_tr), X_te_p, y_te)
        rows.append({
            "Random State": seed,
            "Original_30D_Acc": base_m["Accuracy"], "PCA_10D_Acc": pca_m["Accuracy"],
            "PCA_10D_CEV (%)": round(float(pca.explained_variance_ratio_.sum() * 100.0), 2),
            "Original_30D_Mal_Recall": base_m["Malignant_Recall"],
            "PCA_10D_Mal_Recall": pca_m["Malignant_Recall"],
        })
    df = pd.DataFrame(rows)
    print("===== REPEATED STRATIFIED SPLITS (SEEDS 0..11) =====")
    print(df.round(4).to_string(index=False))

    diff = df["PCA_10D_Acc"] - df["Original_30D_Acc"]
    t_stat, t_p = ttest_rel(df["PCA_10D_Acc"], df["Original_30D_Acc"])
    try:
        w_stat, w_p = wilcoxon(diff[diff != 0])
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    summary = {
        "Original_30D_Mean_Acc": float(df["Original_30D_Acc"].mean()),
        "Original_30D_Std_Acc": float(df["Original_30D_Acc"].std(ddof=1)),
        "PCA_10D_Mean_Acc": float(df["PCA_10D_Acc"].mean()),
        "PCA_10D_Std_Acc": float(df["PCA_10D_Acc"].std(ddof=1)),
        "PCA_10D_Mean_CEV (%)": float(df["PCA_10D_CEV (%)"].mean()),
        "Original_30D_Mean_Mal_Recall": float(df["Original_30D_Mal_Recall"].mean()),
        "PCA_10D_Mean_Mal_Recall": float(df["PCA_10D_Mal_Recall"].mean()),
        "PCA_10D_Min_Mal_Recall": float(df["PCA_10D_Mal_Recall"].min()),
        "Mean_Diff_PCA_minus_Original": float(diff.mean()),
        "Seeds_PCA_better/equal/worse": [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())],
        "paired_t_p": float(t_p), "wilcoxon_p": float(w_p),
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    df.to_csv(BASE_DIR / "repeated_splits_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(df["Random State"], df["Original_30D_Acc"] * 100, "o-", color="#c62828", label="Original 30D")
    axes[0].plot(df["Random State"], df["PCA_10D_Acc"] * 100, "s--", color="#1565c0", label="PCA q = 10")
    axes[0].set_xlabel("Random seed of the split"); axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_xticks(range(12)); axes[0].set_title("(a) Accuracy per split"); axes[0].legend(loc="lower right")
    axes[1].bar(df["Random State"], diff * 100, color=np.where(diff >= 0, "#2e7d32", "#c62828"), width=0.6)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xlabel("Random seed of the split"); axes[1].set_ylabel("PCA10 $-$ Original (percentage points)")
    axes[1].set_xticks(range(12)); axes[1].set_title(f"(b) Paired difference (mean = {diff.mean()*100:+.2f} pp)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "repeated_splits.png"); plt.close()

    meta = get_environment_metadata()
    meta["summary"] = summary; meta["seeds"] = rows
    save_json_result("results_D.json", meta)


if __name__ == "__main__":
    main()
