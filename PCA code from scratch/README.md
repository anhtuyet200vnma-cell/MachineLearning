# PCA Code From Scratch (Phân tích Thành phần Chính)

Thư mục này chứa mã nguồn triển khai thuật toán **PCA (Principal Component Analysis)** **hoàn toàn từ đầu (from scratch)** chỉ sử dụng thư viện tính toán ma trận **NumPy**, không phụ thuộc vào `sklearn.decomposition.PCA`.

## Cấu trúc thư mục

```text
PCA code from scratch/
├── README.md               # Tài liệu hướng dẫn và công thức toán học
└── pca_from_scratch.py     # Mã nguồn cài đặt StandardScalerFromScratch & PCAFromScratch
```

## Cơ sở Toán học được Cài đặt

Cho ma trận dữ liệu đầu vào $X \in \mathbb{R}^{n \times p}$ ($n$ mẫu, $p$ đặc trưng):

1. **Chuẩn hóa & Dịch tâm (Centering / Z-score Standardization):**
   $$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad X_c = X - \mathbf{1}\mu^T$$
   *(Nếu các biến khác đơn vị đo, chuẩn hóa theo độ lệch chuẩn trên tập huấn luyện: $Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$).*

2. **Ma trận Hiệp phương sai Mẫu (Sample Covariance Matrix):**
   $$S = \frac{1}{n - 1} X_c^T X_c \in \mathbb{R}^{p \times p}$$

3. **Tìm Hướng Phương sai Cực đại (Hai phương pháp giải tương đương trong code):**
   - **Eigendecomposition (`method="eig"`):** Giải bài toán trị riêng $S a_k = \lambda_k a_k$ và sắp xếp $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$.
   - **Singular Value Decomposition (`method="svd"`):** Phân rã trực tiếp $X_c = U \Sigma V^T$, trong đó $a_k = v_k$ và $\lambda_k = \frac{\sigma_k^2}{n - 1}$.

4. **Chiếu Giảm chiều & Tái tạo Dữ liệu (Projection & Reconstruction):**
   - Lập ma trận chiếu $A_q = [a_1, a_2, \dots, a_q] \in \mathbb{R}^{p \times q}$.
   - Tọa độ mới (Scores): $Z = X_c A_q \in \mathbb{R}^{n \times q}$.
   - Tái tạo xấp xỉ: $\hat{X} = Z A_q^T + \mathbf{1}\mu^T$.

## Hướng dẫn Chạy Mã nguồn

### 1. Cài đặt thư viện cần thiết
```bash
pip install numpy scikit-learn
```
*(Lưu ý: `numpy` dùng cho toàn bộ lõi thuật toán PCA; `scikit-learn` chỉ dùng để tải bộ dữ liệu chuẩn UCI Breast Cancer và chạy bộ phân loại kiểm chứng).*

### 2. Thực thi chương trình
```bash
python pca_from_scratch.py
```

## Kết quả Kiểm chứng Tiêu biểu

| Kịch bản thực nghiệm | Số chiều gốc ($p$) | Số chiều nén ($q$) | Phương sai giữ lại (CEV) | Kết quả chính |
| :--- | :---: | :---: | :---: | :--- |
| **Ví dụ tính tay 5 điểm** | 2 | 1 | **97.25%** | $\lambda_1 = 5.1545$, sai số mất mát $= \lambda_2 = 0.1455$ |
| **Breast Cancer Wisconsin (UCI)** | 30 | 10 | **95.27%** | Test Accuracy $= \mathbf{97.37\%}$, Malignant Recall $= \mathbf{97.62\%}$ |
