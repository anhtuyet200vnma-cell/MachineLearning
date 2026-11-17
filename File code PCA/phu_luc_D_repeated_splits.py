"""Phụ lục D: Đánh giá độ ổn định qua 12 lần chia ngẫu nhiên Train/Test (Repeated Stratified Splits)."""
from pathlib import Path
import pandas as pd

from common import (
    load_data,
    split_data,
    standardize_data,
    reduce_with_pca,
    train_logistic_regression,
    evaluate_model,
    get_environment_metadata,
    save_json_result,
)


def main():
    base_dir = Path(__file__).resolve().parent

    X, y, _, _ = load_data()
    rows = []

    for seed in range(12):
        X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, random_state=seed)
        X_tr_s, X_te_s, _ = standardize_data(X_tr, X_te)

        # Baseline 30D
        base_clf = train_logistic_regression(X_tr_s, y_tr, random_state=42)
        base_m, _ = evaluate_model(base_clf, X_te_s, y_te)

        # PCA q = 10
        X_tr_p, X_te_p, pca = reduce_with_pca(X_tr_s, X_te_s, n_components=10, random_state=42)
        pca_clf = train_logistic_regression(X_tr_p, y_tr, random_state=42)
        pca_m, _ = evaluate_model(pca_clf, X_te_p, y_te)

        rows.append({
            "Random State": seed,
            "Original_30D_Acc": base_m["Accuracy"],
            "PCA_10D_Acc": pca_m["Accuracy"],
            "PCA_10D_CEV (%)": round(float(pca.explained_variance_ratio_.sum() * 100.0), 2),
            "PCA_10D_Mal_Recall": pca_m["Malignant_Recall"],
        })

    df = pd.DataFrame(rows)
    print("===== REPEATED STRATIFIED SPLITS (12 SEEDS: 0 -> 11) =====")
    print(df.to_string(index=False))

    summary = {
        "Original_30D_Mean_Acc": float(df["Original_30D_Acc"].mean()),
        "Original_30D_Std_Acc": float(df["Original_30D_Acc"].std(ddof=1)),
        "PCA_10D_Mean_Acc": float(df["PCA_10D_Acc"].mean()),
        "PCA_10D_Std_Acc": float(df["PCA_10D_Acc"].std(ddof=1)),
        "PCA_10D_Mean_CEV (%)": float(df["PCA_10D_CEV (%)"].mean()),
    }
    print("\nSummary:", summary)

    df.to_csv(base_dir / "repeated_splits_results.csv", index=False)

    meta = get_environment_metadata()
    meta["summary"] = summary
    meta["seeds"] = rows
    save_json_result("results_D.json", meta)


if __name__ == "__main__":
    main()
