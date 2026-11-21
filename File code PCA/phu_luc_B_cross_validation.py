"""Appendix B: 5-fold stratified cross-validation on the TRAIN split (no leakage:
scaler + PCA + classifier live inside one Pipeline and are refitted in every fold)."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from common import (BASE_DIR, FIG_DIR, load_data, split_data, get_environment_metadata,
                    save_json_result, set_plot_style)


def main():
    plt = set_plot_style()
    X, y, _, _ = load_data()
    X_train, _, y_train, _ = split_data(X, y, test_size=0.2, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # NOTE: the default "f1" scorer uses pos_label = 1 (benign).  The clinical positive
    # class is MALIGNANT (0), so malignant recall / F1 are requested explicitly.
    scoring = {
        "accuracy": "accuracy",
        "mal_recall": make_scorer(recall_score, pos_label=0),
        "mal_f1": make_scorer(f1_score, pos_label=0),
    }

    def run(pipe, method, q):
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        return {
            "Method": method, "Dimensions": q,
            "Mean Accuracy": float(s["test_accuracy"].mean()),
            "Std": float(s["test_accuracy"].std(ddof=1)),
            "Mean Malignant Recall": float(s["test_mal_recall"].mean()),
            "Mean Malignant F1": float(s["test_mal_f1"].mean()),
        }

    results = [run(Pipeline([("scaler", StandardScaler()),
                             ("classifier", LogisticRegression(max_iter=5000, random_state=42))]), "Original", 30)]
    for q in [1, 2, 3, 5, 6, 7, 10, 15]:
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=q, random_state=42)),
                         ("classifier", LogisticRegression(max_iter=5000, random_state=42))])
        results.append(run(pipe, "PCA", q))

    table = pd.DataFrame(results)
    print("===== 5-FOLD CROSS-VALIDATION (TRAIN n = 455) =====")
    print(table.round(4).to_string(index=False))
    table.to_csv(BASE_DIR / "cross_validation_results.csv", index=False)

    # selection of q: best mean, and the one-standard-error rule
    pca_rows = table[table["Method"] == "PCA"].reset_index(drop=True)
    best = pca_rows.loc[pca_rows["Mean Accuracy"].idxmax()]
    se = best["Std"] / np.sqrt(5)
    within = pca_rows[pca_rows["Mean Accuracy"] >= best["Mean Accuracy"] - se]
    q_1se = int(within["Dimensions"].min())
    print(f"\nBest q by mean CV accuracy: {int(best['Dimensions'])} ({best['Mean Accuracy']:.4f} +/- {best['Std']:.4f})")
    print(f"One-standard-error rule (SE = {se:.4f}): smallest q within 1 SE of the best = {q_1se}")

    # figure
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.errorbar(pca_rows["Dimensions"], pca_rows["Mean Accuracy"] * 100, yerr=pca_rows["Std"] * 100,
                fmt="o-", color="#1565c0", capsize=4, lw=1.8, label="PCA + logistic regression (mean $\\pm$ 1 SD)")
    base = table.iloc[0]
    ax.axhline(base["Mean Accuracy"] * 100, color="#c62828", ls="--", lw=1.4, label=f"Original 30D ({base['Mean Accuracy']*100:.2f}%)")
    ax.axhspan((base["Mean Accuracy"] - base["Std"]) * 100, (base["Mean Accuracy"] + base["Std"]) * 100, color="#c62828", alpha=0.08)
    ax.scatter([best["Dimensions"]], [best["Mean Accuracy"] * 100], s=140, facecolors="none", edgecolors="#1b5e20", lw=2, zorder=5,
               label=f"Best: q = {int(best['Dimensions'])}")
    ax.set_xlabel("Number of retained components q"); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.set_title("Cross-validated accuracy versus q (training set, n = 455)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG_DIR / "cv_accuracy_vs_q.png"); plt.close()

    meta = get_environment_metadata()
    meta["cv_results"] = results
    meta["best_q"] = int(best["Dimensions"]); meta["one_se_q"] = q_1se
    save_json_result("results_B.json", meta)


if __name__ == "__main__":
    main()
