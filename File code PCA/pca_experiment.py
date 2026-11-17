"""Thực nghiệm chuyên sâu PCA: Scaling, Giảm chiều 5D->2D, Tính tay 2D->1D & Đa mô hình."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from common import (
    load_data,
    split_data,
    standardize_data,
    check_pca_suitability,
    horn_parallel_analysis,
    reduce_with_pca,
    evaluate_model,
)


def main():
    base_dir = Path(__file__).resolve().parent
    X, y, target_names, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    # 1. KIỂM ĐỊNH TIÊN QUYẾT BARTLETT & KMO + HORN PARALLEL ANALYSIS
    print("===== 1. KIỂM ĐỊNH TIÊN QUYẾT BARTLETT, KMO & HORN PARALLEL ANALYSIS =====")
    suit = check_pca_suitability(X_train)
    print(f"Bartlett Chi2 = {suit['chi2']:.1f} (df={suit['df']}) | KMO = {suit['kmo']:.4f}")

    pca_full = PCA().fit(X_train_scaled)
    pa_thresholds = horn_parallel_analysis(X_train_scaled.shape[0], X_train_scaled.shape[1])
    kaiser_q = int(np.sum(pca_full.explained_variance_ > 1.0))
    horn_q = int(np.sum(pca_full.explained_variance_ > pa_thresholds))
    print(f"Số chiều theo Quy tắc Kaiser (lambda > 1.0): q = {kaiser_q}")
    print(f"Số chiều theo Phân tích song song Horn (95th pct): q = {horn_q}")

    # 2. SO SÁNH COVARIANCE PCA (CHƯA CHUẨN HÓA) VS CORRELATION PCA (ĐÃ CHUẨN HÓA Z-SCORE)
    print("\n===== 2. SO SÁNH COVARIANCE PCA (UNSCALED) VS CORRELATION PCA (SCALED) =====")
    pca_unscaled = PCA().fit(X_train)
    scaling_rows = []
    for k in range(10):
        scaling_rows.append(
            {
                "Component": f"PC{k+1}",
                "Unscaled_EVR (%)": round(float(pca_unscaled.explained_variance_ratio_[k] * 100), 2),
                "Unscaled_CEV (%)": round(float(np.sum(pca_unscaled.explained_variance_ratio_[:k+1]) * 100), 2),
                "Scaled_Eigenvalue": round(float(pca_full.explained_variance_[k]), 4),
                "Scaled_EVR (%)": round(float(pca_full.explained_variance_ratio_[k] * 100), 2),
                "Scaled_CEV (%)": round(float(np.sum(pca_full.explained_variance_ratio_[:k+1]) * 100), 2),
                "Horn_95pct_Threshold": round(float(pa_thresholds[k]), 4),
            }
        )
    df_scaling = pd.DataFrame(scaling_rows)
    print(df_scaling.to_string(index=False))
    df_scaling.to_csv(base_dir / "pca_scaling_comparison.csv", index=False)

    # 3. THỰC NGHIỆM GIẢM CHIỀU 5D -> 2D (5 ĐẶC TRƯNG HÌNH THÁI HỌC ĐẠT NGƯỠNG 90-95%)
    print("\n===== 3. THỰC NGHIỆM GIẢM CHIỀU 5D -> 2D (5 ĐẶC TRƯNG HÌNH THÁI HỌC) =====")
    idx_5d = [0, 2, 3, 5, 4]  # mean radius, perimeter, area, compactness, smoothness
    names_5d = [feature_names[i] for i in idx_5d]
    Z_5d = StandardScaler().fit_transform(X_train[:, idx_5d])
    pca_5d = PCA().fit(Z_5d)
    rows_5d = []
    for k, (eig, ratio) in enumerate(
        zip(pca_5d.explained_variance_, pca_5d.explained_variance_ratio_ * 100), start=1
    ):
        cev_k = float(np.sum(pca_5d.explained_variance_ratio_[:k]) * 100)
        status = "Giữ lại (Cốt lõi)" if k <= 2 else "Loại bỏ (Nhiễu)"
        rows_5d.append(
            {
                "Component": f"PC{k}",
                "Eigenvalue": round(float(eig), 4),
                "EVR (%)": round(float(ratio), 2),
                "CEV (%)": round(cev_k, 2),
                "Decision": status,
            }
        )
    df_5d = pd.DataFrame(rows_5d)
    print("5 biến đầu vào:", names_5d)
    print(df_5d.to_string(index=False))
    df_5d.to_csv(base_dir / "pca_5d_to_2d_results.csv", index=False)

    # 4. KIỂM CHỨNG VÍ DỤ TÍNH TAY 2D -> 1D (5 ĐIỂM DỮ LIỆU)
    print("\n===== 4. KIỂM CHỨNG TÍNH TAY 2D -> 1D (ĐỊNH LÝ TRACE & ECKART-YOUNG) =====")
    X_hand = np.array([[2, 1], [3, 3], [4, 3], [5, 5], [6, 5]], dtype=float)
    S_hand = np.cov(X_hand, rowvar=False, ddof=1)
    pca_hand = PCA(n_components=1).fit(X_hand)
    Z_hand = pca_hand.transform(X_hand).ravel()
    X_rec = pca_hand.inverse_transform(Z_hand.reshape(-1, 1))
    recon_mse = float(np.sum((X_hand - X_rec) ** 2) / (len(X_hand) - 1))
    print(f"Trace(S) = {np.trace(S_hand):.4f} | EVR PC1 = {pca_hand.explained_variance_ratio_[0]*100:.2f}%")
    print(f"Tọa độ chiếu Z (1D) = {np.round(Z_hand, 2).tolist()}")
    print(f"Sai số tái tạo Eckart-Young = {recon_mse:.4f} (Khớp trị riêng bỏ đi lambda_2)")

    # 5. ĐỐI CHIẾU ĐA BỘ PHÂN LOẠI (LOGISTIC REGRESSION, SVM RBF, RANDOM FOREST)
    print("\n===== 5. ĐỐI CHIẾU ĐA BỘ PHÂN LOẠI (ORIGINAL 30D VS PCA q=10) =====")
    X_tr_10, X_te_10, _ = reduce_with_pca(X_train_scaled, X_test_scaled, n_components=10)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        m_orig, _ = evaluate_model(clf, X_test_scaled, y_test)
        clf.fit(X_tr_10, y_train)
        m_pca, _ = evaluate_model(clf, X_te_10, y_test)
        print(
            f"{clf_name:20s} | Orig 30D Acc: {m_orig['Accuracy']:.2%} (Recall: {m_orig['Malignant_Recall']:.2%}) "
            f"-> PCA 10D Acc: {m_pca['Accuracy']:.2%} (Recall: {m_pca['Malignant_Recall']:.2%})"
        )


if __name__ == "__main__":
    main()
