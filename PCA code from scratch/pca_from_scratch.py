"""
PCA (Principal Component Analysis) - Implementation From Scratch
================================================================
Triển khai thuật toán Phân tích Thành phần Chính (PCA) từ đầu chỉ sử dụng NumPy
(không dùng sklearn.decomposition.PCA), bao gồm cả hai phương pháp giải toán học:
  1. Phân rã trị riêng ma trận hiệp phương sai / tương quan (Eigendecomposition: S * a_k = lambda_k * a_k)
  2. Phân rã giá trị suy biến (Singular Value Decomposition - SVD: X_c = U * Sigma * V^T)

Kèm 2 kịch bản kiểm chứng khớp số liệu bài thuyết trình:
  - Demo 1: Ví dụ tính tay 5 điểm 2D -> 1D (Kiểm chứng Định lý Trace & Eckart-Young).
  - Demo 2: Tập dữ liệu Breast Cancer Wisconsin UCI (569 mẫu, 30 đặc trưng -> q = 10),
            tuân thủ quy trình chống rò rỉ dữ liệu (Anti-Data-Leakage Pipeline)
            và đối chiếu độ khớp toán học với SVD & scikit-learn.
"""

from __future__ import annotations

import numpy as np


class StandardScalerFromScratch:
    """Chuẩn hóa Z-score từ đầu: Z = (X - mu) / sigma (chỉ fit trên tập Train)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerFromScratch":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        # Tránh chia cho 0 nếu có đặc trưng hằng số
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerFromScratch chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScalerFromScratch chưa được fit.")
        Z = np.asarray(Z, dtype=np.float64)
        return Z * self.scale_ + self.mean_


class PCAFromScratch:
    """
    Principal Component Analysis (PCA) tự cài đặt bằng NumPy thuần túy.

    Parameters
    ----------
    n_components : int | float | None, default=None
        - Nếu là int (q >= 1): giữ lại q thành phần chính đầu tiên.
        - Nếu là float (0 < q < 1): tự động chọn số thành phần nhỏ nhất sao cho
          phương sai tích lũy CEV >= n_components (ví dụ 0.95 cho 95%).
        - Nếu là None: giữ lại toàn bộ min(n_samples, n_features) thành phần.
    method : {"eig", "svd"}, default="eig"
        - "eig": Phân rã trị riêng ma trận hiệp phương sai mẫu S = (X_c^T X_c) / (n - 1).
        - "svd": Phân rã giá trị suy biến trực tiếp trên ma trận dịch tâm X_c = U Sigma V^T.
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        method: str = "eig",
    ) -> None:
        if method not in {"eig", "svd"}:
            raise ValueError("method phải là 'eig' hoặc 'svd'.")
        self.n_components = n_components
        self.method = method

        # Các thuộc tính học được sau khi gọi fit(X)
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.n_components_: int = 0
        self.mean_: np.ndarray | None = None
        self.cov_matrix_: np.ndarray | None = None
        self.components_: np.ndarray | None = None          # Hình dạng (q, p): mỗi hàng là một eigenvector a_k^T
        self.eigenvectors_: np.ndarray | None = None        # Hình dạng (p, q): ma trận chiếu A_q = [a_1, ..., a_q]
        self.explained_variance_: np.ndarray | None = None  # Trị riêng lambda_1 >= ... >= lambda_q
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_ratio_: np.ndarray | None = None
        self.total_variance_: float = 0.0

    @staticmethod
    def _flip_eigenvector_signs(V: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa dấu của các vector riêng (cột của V) sao cho phần tử có trị tuyệt đối
        lớn nhất trong mỗi cột luôn dương (giống quy ước của scikit-learn).
        """
        max_abs_rows = np.argmax(np.abs(V), axis=0)
        signs = np.sign(V[max_abs_rows, np.arange(V.shape[1])])
        signs = np.where(signs == 0, 1.0, signs)
        return V * signs

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("Dữ liệu đầu vào X phải là ma trận 2 chiều (n_samples, n_features).")

        n, p = X.shape
        if n < 2:
            raise ValueError("Cần ít nhất 2 mẫu dữ liệu để tính hiệp phương sai.")

        self.n_samples_ = n
        self.n_features_ = p

        # Bước 1: Dịch tâm dữ liệu (Centering): X_c = X - 1 * mu^T
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Ma trận hiệp phương sai mẫu S = (X_c^T X_c) / (n - 1)
        S = (X_centered.T @ X_centered) / (n - 1)
        self.cov_matrix_ = S

        if self.method == "eig":
            # Bước 2 & 3 (Eigendecomposition): Giải S * a_k = lambda_k * a_k
            # Dùng eigh vì S là ma trận đối xứng thực (bảo đảm trị riêng thực)
            eigenvalues, eigenvectors = np.linalg.eigh(S)

            # Sắp xếp trị riêng giảm dần: lambda_1 >= lambda_2 >= ... >= lambda_p
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[sorted_idx], 0.0)
            eigenvectors = eigenvectors[:, sorted_idx]
        else:
            # Bước 2 & 3 (SVD): X_c = U * Sigma * V^T => lambda_k = sigma_k^2 / (n - 1)
            _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigenvalues = (singular_values ** 2) / (n - 1)
            eigenvectors = Vt.T

        # Chuẩn hóa dấu vector riêng để kết quả nhất quán
        eigenvectors = self._flip_eigenvector_signs(eigenvectors)

        # Tổng phương sai = trace(S) = sum(lambda_k)
        total_var = float(np.sum(eigenvalues))
        self.total_variance_ = total_var
        evr = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
        cev = np.cumsum(evr)

        # Bước 4: Xác định số thành phần chính q cần giữ lại
        max_k = eigenvectors.shape[1]
        if self.n_components is None:
            q = max_k
        elif isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            q = int(np.searchsorted(cev, self.n_components) + 1)
            q = min(max(q, 1), max_k)
        elif isinstance(self.n_components, (int, np.integer)) and 1 <= self.n_components <= max_k:
            q = int(self.n_components)
        else:
            raise ValueError(f"n_components={self.n_components} không hợp lệ với p={p}.")

        self.n_components_ = q
        self.eigenvectors_ = eigenvectors[:, :q]              # A_q: (p, q)
        self.components_ = self.eigenvectors_.T               # A_q^T: (q, p)
        self.explained_variance_ = eigenvalues[:q]
        self.explained_variance_ratio_ = evr[:q]
        self.cumulative_variance_ratio_ = cev[:q]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Chiếu dữ liệu sang không gian thành phần chính: Z = (X - mu) * A_q."""
        if self.mean_ is None or self.eigenvectors_ is None:
            raise RuntimeError("PCAFromScratch chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.eigenvectors_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Tái tạo xấp xỉ dữ liệu về không gian gốc: X_hat = Z * A_q^T + mu."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCAFromScratch chưa được fit.")
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_

    def relative_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Tính sai số tái tạo tương đối theo chuẩn Frobenius:
            E_rel = ||X_c - X_hat_c||_F^2 / ||X_c||_F^2
        """
        if self.mean_ is None:
            raise RuntimeError("PCAFromScratch chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.mean_
        Z = self.transform(X)
        X_hat_c = Z @ self.components_
        denom = float(np.sum(X_c ** 2))
        if denom == 0.0:
            return 0.0
        return float(np.sum((X_c - X_hat_c) ** 2) / denom)


def demo_worked_example_2d() -> None:
    """Demo 1: Kiểm chứng ví dụ tính tay 5 điểm 2D -> 1D trên Slide thuyết trình."""
    print("=" * 72)
    print("DEMO 1: VÍ DỤ TÍNH TAY 2D -> 1D (5 ĐIỂM DỮ LIỆU)")
    print("=" * 72)

    X_2d = np.array([
        [2.0, 1.0],
        [3.0, 3.0],
        [4.0, 3.0],
        [5.0, 5.0],
        [6.0, 5.0],
    ])

    pca_full = PCAFromScratch(n_components=2, method="eig").fit(X_2d)
    print("Vector trung bình mu:", np.round(pca_full.mean_, 4))
    print("Ma trận hiệp phương sai S:\n", np.round(pca_full.cov_matrix_, 4))
    print(f"Tổng phương sai trace(S): {pca_full.total_variance_:.4f}")
    print("Trị riêng (lambda_1, lambda_2):", np.round(pca_full.explained_variance_, 4))
    print("Tỷ lệ phương sai (EVR %):     ", np.round(pca_full.explained_variance_ratio_ * 100, 2))
    print("Vector riêng chính a_1:       ", np.round(pca_full.eigenvectors_[:, 0], 4))

    pca_1d = PCAFromScratch(n_components=1, method="eig").fit(X_2d)
    Z_1d = pca_1d.transform(X_2d)
    X_rec = pca_1d.inverse_transform(Z_1d)
    # Phương sai mất đi khi chiếu xuống 1D: sum(||x_i - x_hat_i||^2) / (n - 1) == lambda_2
    eckart_young_var_lost = np.sum((X_2d - X_rec) ** 2) / (len(X_2d) - 1)
    print("Tọa độ chiếu 1D (Scores Z):   ", np.round(Z_1d.ravel(), 2))
    print(f"Phương sai mất đi (Eckart-Young) = {eckart_young_var_lost:.4f} (khớp lambda_2 = {pca_full.explained_variance_[1]:.4f})\n")


def demo_breast_cancer_pipeline() -> None:
    """Demo 2: Thực nghiệm trên Breast Cancer Wisconsin UCI (569 x 30 -> q = 10)."""
    print("=" * 72)
    print("DEMO 2: BREAST CANCER WISCONSIN UCI (30D -> 10D, ANTI-LEAKAGE PIPELINE)")
    print("=" * 72)

    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, recall_score
    except ImportError:
        print("Cần cài đặt scikit-learn để tải tập dữ liệu UCI và chạy bộ phân loại kiểm chứng.")
        return

    data = load_breast_cancer()
    X, y = data.data, data.target

    # Chia tập Train/Test phân tầng 80/20 (455 Train / 114 Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit StandardScaler & PCA CHỈ TRÊN TẬP TRAIN (Chống rò rỉ dữ liệu)
    scaler = StandardScalerFromScratch().fit(X_train)
    Z_train = scaler.transform(X_train)
    Z_test = scaler.transform(X_test)

    # Đối chiếu độ khớp giữa Eigendecomposition và SVD
    pca_eig = PCAFromScratch(n_components=10, method="eig").fit(Z_train)
    pca_svd = PCAFromScratch(n_components=10, method="svd").fit(Z_train)

    max_eval_diff = float(np.max(np.abs(pca_eig.explained_variance_ - pca_svd.explained_variance_)))
    print(f"Số mẫu Train: {X_train.shape[0]} | Số mẫu Test: {X_test.shape[0]} | Số chiều gốc: {X.shape[1]}")
    print(f"Tổng phương sai giữ lại (q=10): {pca_eig.cumulative_variance_ratio_[-1] * 100:.2f}%")
    print(f"Sai số tái tạo tương đối trên Test: {pca_eig.relative_reconstruction_error(Z_test) * 100:.2f}%")
    print(f"Độ lệch trị riêng lớn nhất giữa EIG và SVD: {max_eval_diff:.2e}")

    # Huấn luyện Logistic Regression trên không gian nén 10 chiều
    # Lưu ý: Trong Breast Cancer UCI, class 0 là Malignant (Ác tính), class 1 là Benign (Lành tính)
    X_train_pca = pca_eig.transform(Z_train)
    X_test_pca = pca_eig.transform(Z_test)

    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    malignant_recall = recall_score(y_test, y_pred, pos_label=0)
    print(f"Test Accuracy (PCA q=10):        {acc * 100:.2f}%")
    print(f"Malignant Recall (pos_label=0):  {malignant_recall * 100:.2f}%")

    # Kiểm chứng chéo với scikit-learn PCA
    from sklearn.decomposition import PCA as SklearnPCA
    sk_pca = SklearnPCA(n_components=10).fit(Z_train)
    sk_diff = float(np.max(np.abs(pca_eig.explained_variance_ - sk_pca.explained_variance_)))
    print(f"Độ lệch trị riêng so với sklearn.decomposition.PCA: {sk_diff:.2e}\n")


if __name__ == "__main__":
    demo_worked_example_2d()
    demo_breast_cancer_pipeline()
