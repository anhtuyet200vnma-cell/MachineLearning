"""In-depth PCA experiments: scaling, rules for q, 5D->2D reduction, hand-worked 2D->1D
example and a multi-classifier benchmark."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from common import (
    BASE_DIR, load_data, split_data, standardize_data, check_pca_suitability,
    correlation_eigenvalues, horn_parallel_analysis, count_kaiser, count_horn,
    reduce_with_pca, evaluate_model, save_json_result, get_environment_metadata,
)


def main():
    X, y, target_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)
    n, p = X_train_scaled.shape

    # 1. BARTLETT, KMO, KAISER AND HORN
    print("===== 1. BARTLETT, KMO, KAISER AND HORN PARALLEL ANALYSIS =====")
    suit = check_pca_suitability(X_train)
    print(f"Bartlett chi2 = {suit['chi2']:.1f} (df={suit['df']}, log10 p = {suit['log10_p']:.0f}) | KMO = {suit['kmo']:.4f}")
    pca_full = PCA().fit(X_train_scaled)
    eig = correlation_eigenvalues(X_train_scaled)          # eigenvalues of R (sum = p)
    pa = horn_parallel_analysis(n, p)
    kaiser_q, horn_q = count_kaiser(eig), count_horn(eig, pa)
    print(f"Kaiser rule (lambda > 1): q = {kaiser_q} | Horn parallel analysis (95th pct): q = {horn_q}")

    # 2. COVARIANCE PCA (UNSCALED) VS CORRELATION PCA (SCALED)
    print("\n===== 2. UNSCALED (COVARIANCE) VS SCALED (CORRELATION) PCA =====")
    pca_unscaled = PCA().fit(X_train)
    top_feat = feature_names[int(np.argmax(np.abs(pca_unscaled.components_[0])))]
    print(f"Feature dominating unscaled PC1: {top_feat} (|weight| = {np.abs(pca_unscaled.components_[0]).max():.4f})")
    rows = []
    for k in range(10):
        rows.append({
            "Component": f"PC{k+1}",
            "Unscaled_EVR (%)": round(float(pca_unscaled.explained_variance_ratio_[k] * 100), 2),
            "Unscaled_CEV (%)": round(float(np.sum(pca_unscaled.explained_variance_ratio_[:k + 1]) * 100), 2),
            "Scaled_Eigenvalue": round(float(eig[k]), 4),
            "Scaled_EVR (%)": round(float(pca_full.explained_variance_ratio_[k] * 100), 2),
            "Scaled_CEV (%)": round(float(np.sum(pca_full.explained_variance_ratio_[:k + 1]) * 100), 2),
            "Horn_95pct_Threshold": round(float(pa[k]), 4),
            "Kaiser_Pass": bool(eig[k] > 1.0),
            "Horn_Pass": bool(eig[k] > pa[k]),
        })
    df_scaling = pd.DataFrame(rows)
    print(df_scaling.to_string(index=False))
    df_scaling.to_csv(BASE_DIR / "pca_scaling_comparison.csv", index=False)

    # 3. FIVE MORPHOLOGICAL FEATURES: 5D -> 2D
    print("\n===== 3. 5D -> 2D REDUCTION (5 MORPHOLOGICAL FEATURES) =====")
    idx_5d = [0, 2, 3, 5, 4]    # mean radius, mean perimeter, mean area, mean compactness, mean smoothness
    names_5d = [feature_names[i] for i in idx_5d]
    Z_5d = StandardScaler().fit_transform(X_train[:, idx_5d])
    pca_5d = PCA().fit(Z_5d)
    eig_5d = correlation_eigenvalues(Z_5d)
    rows_5d = []
    for k in range(1, 6):
        cev_k = float(np.sum(pca_5d.explained_variance_ratio_[:k]) * 100)
        rows_5d.append({
            "Component": f"PC{k}", "Eigenvalue": round(float(eig_5d[k - 1]), 4),
            "EVR (%)": round(float(pca_5d.explained_variance_ratio_[k - 1] * 100), 2),
            "CEV (%)": round(cev_k, 2),
            "Decision": "Retain" if k <= 2 else "Discard",
        })
    df_5d = pd.DataFrame(rows_5d)
    print("Inputs:", names_5d)
    print(df_5d.to_string(index=False))
    df_5d.to_csv(BASE_DIR / "pca_5d_to_2d_results.csv", index=False)
    pd.DataFrame(pca_5d.components_.T, index=names_5d,
                 columns=[f"PC{k}" for k in range(1, 6)]).round(4).to_csv(BASE_DIR / "pca_5d_eigenvectors.csv")
    print("Eigenvectors (columns = PCs):\n", pd.DataFrame(pca_5d.components_.T, index=names_5d,
          columns=[f"PC{k}" for k in range(1, 6)]).round(3).to_string())

    # 4. HAND-WORKED EXAMPLE 2D -> 1D (5 POINTS)
    print("\n===== 4. HAND-WORKED 2D -> 1D EXAMPLE (TRACE AND ECKART-YOUNG CHECK) =====")
    X_hand = np.array([[2, 1], [3, 3], [4, 3], [5, 5], [6, 5]], dtype=float)
    mu = X_hand.mean(axis=0)
    S_hand = np.cov(X_hand, rowvar=False, ddof=1)
    pca_hand = PCA(n_components=1).fit(X_hand)
    a1 = pca_hand.components_[0]
    if a1[0] < 0:
        a1 = -a1
    Z_hand = (X_hand - mu) @ a1
    X_rec = np.outer(Z_hand, a1) + mu
    d2 = np.sum((X_hand - X_rec) ** 2, axis=1)
    lam = np.sort(np.linalg.eigvalsh(S_hand))[::-1]
    mse = float(d2.sum() / (len(X_hand) - 1))
    assert np.isclose(np.trace(S_hand), lam.sum()) and np.isclose(mse, lam[1]), "Eckart-Young check failed"
    print("mean:", mu, "| S =", S_hand.tolist(), "| trace =", round(float(np.trace(S_hand)), 4))
    print(f"lambda1 = {lam[0]:.4f} (EVR {lam[0]/lam.sum()*100:.2f}%), lambda2 = {lam[1]:.4f} (EVR {lam[1]/lam.sum()*100:.2f}%)")
    print("a1 =", np.round(a1, 4), "| slope =", round(float(a1[1] / a1[0]), 4))
    print("scores z =", np.round(Z_hand, 2).tolist())
    print("squared errors =", np.round(d2, 3).tolist(), "| sum =", round(float(d2.sum()), 3))
    print(f"MSE_perp = {mse:.4f}  ==  lambda2 = {lam[1]:.4f}  -> Eckart-Young identity verified")
    pd.DataFrame({
        "Point": [f"P{i+1}" for i in range(5)], "x1": X_hand[:, 0], "x2": X_hand[:, 1],
        "xc1": (X_hand - mu)[:, 0], "xc2": (X_hand - mu)[:, 1], "score_z1": Z_hand,
        "xhat1": X_rec[:, 0], "xhat2": X_rec[:, 1], "sq_error": d2,
    }).round(4).to_csv(BASE_DIR / "hand_example_results.csv", index=False)

    # 5. MULTI-CLASSIFIER BENCHMARK: ORIGINAL 30D VS PCA q
    print("\n===== 5. BENCHMARK: LOGISTIC REGRESSION / RBF-SVM / RANDOM FOREST =====")
    pca_all = PCA(random_state=42).fit(X_train_scaled)
    P_tr, P_te = pca_all.transform(X_train_scaled), pca_all.transform(X_test_scaled)
    make = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=5000, random_state=42),
        "SVM (RBF)": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    }
    bench = []
    for name, ctor in make.items():
        clf = ctor().fit(X_train_scaled, y_train)
        m, _ = evaluate_model(clf, X_test_scaled, y_test)
        bench.append({"Classifier": name, "Dimensions": 30, "Accuracy": m["Accuracy"], "Malignant_Recall": m["Malignant_Recall"]})
        for q in [1, 2, 3, 5, 6, 7, 10, 15]:
            clf = ctor().fit(P_tr[:, :q], y_train)
            m, _ = evaluate_model(clf, P_te[:, :q], y_test)
            bench.append({"Classifier": name, "Dimensions": q, "Accuracy": m["Accuracy"], "Malignant_Recall": m["Malignant_Recall"]})
    df_bench = pd.DataFrame(bench)
    df_bench.to_csv(BASE_DIR / "multi_classifier_results.csv", index=False)
    print(df_bench.pivot(index="Dimensions", columns="Classifier", values="Accuracy").round(4).to_string())
    for name in make:
        a = df_bench[(df_bench.Classifier == name) & (df_bench.Dimensions == 30)].iloc[0]
        b = df_bench[(df_bench.Classifier == name) & (df_bench.Dimensions == 10)].iloc[0]
        print(f"{name:20s} | 30D acc {a.Accuracy:.2%} (recall {a.Malignant_Recall:.2%}) -> PCA q=10 acc {b.Accuracy:.2%} (recall {b.Malignant_Recall:.2%})")

    meta = get_environment_metadata()
    meta.update({"bartlett_chi2": suit["chi2"], "bartlett_df": suit["df"], "bartlett_log10_p": suit["log10_p"],
                 "kmo": suit["kmo"], "kaiser_q": kaiser_q, "horn_q": horn_q,
                 "unscaled_pc1_dominant_feature": top_feat})
    save_json_result("results_experiment.json", meta)


if __name__ == "__main__":
    main()
