# File code PCA - Phân tích Thành phần Chính (Principal Component Analysis)

Thư mục mã nguồn thực nghiệm và phụ lục thống kê cho thuật toán **PCA** trên bộ dữ liệu **UCI Breast Cancer Wisconsin** ($n = 569$ mẫu, $p = 30$ đặc trưng hình thái học tế bào).

## Cấu trúc thư mục
- `common.py`: Các hàm dùng chung (tải dữ liệu, chia Train/Test phân tầng 80/20, chuẩn hóa Z-score chống Data Leakage, kiểm định cầu Bartlett & chỉ số KMO, phân tích song song Horn, tính sai số tái tạo Eckart-Young, đánh giá độ nhạy lớp Ác tính).
- `casestudy.py`: Thực nghiệm chính so sánh mô hình gốc 30 chiều (`Original`) với PCA tại các mốc số chiều $q \in \{1, 2, 3, 5, 6, 7, 10, 15\}$; xuất `pca_results.csv` và biểu đồ `pca_accuracy_comparison.png`.
- `pca_experiment.py`: Thực nghiệm chuyên sâu giải đáp các tiêu chí đánh giá (kiểm định Bartlett & KMO, so sánh Covariance PCA chưa chuẩn hóa vs Correlation PCA đã chuẩn hóa Z-score, giảm chiều 5D $\to$ 2D đạt 94.44% phương sai tích lũy, kiểm chứng ví dụ tính tay 2D $\to$ 1D, đối chiếu Logistic Regression / SVM RBF / Random Forest).
- `phu_luc_A_pca_visualization.py`: Trực quan hóa mặt phẳng chiếu 2D Biplot kèm vector trọng số Loadings (`pca_2d_visualization.png`) và biểu đồ Scree Plot + quy tắc Kaiser $\lambda = 1$ + Horn Parallel Analysis + đường cong CEV 95% (`pca_scree_and_scaling.png`).
- `phu_luc_B_cross_validation.py`: Đánh giá chéo **5-Fold Stratified Cross-Validation** qua `Pipeline` khép kín trên tập Train ($n = 455$); xuất `cross_validation_results.csv` và `outputs/results_B.json`.
- `phu_luc_C_bootstrap.py`: Kiểm định **Bootstrap** ($B = 1.000$ lần lấy mẫu lại) ước lượng khoảng tin cậy 95% trên tập Test ($n = 114$); xuất `bootstrap_results.csv` và `outputs/results_C.json`.
- `phu_luc_D_repeated_splits.py`: Đánh giá độ ổn định qua **12 lần chia ngẫu nhiên phân tầng** (`seeds = 0..11`); xuất `repeated_splits_results.csv` và `outputs/results_D.json`.
- `requirements.txt`: Danh sách thư viện cần thiết.

## Hướng dẫn chạy mã nguồn
```bash
pip install -r requirements.txt
python casestudy.py
python pca_experiment.py
python phu_luc_A_pca_visualization.py
python phu_luc_B_cross_validation.py
python phu_luc_C_bootstrap.py
python phu_luc_D_repeated_splits.py
```
