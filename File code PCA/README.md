# Code PCA – Principal Component Analysis (Breast Cancer Wisconsin)

Gói mã nguồn đã **rà soát, sửa lỗi và chạy lại từ đầu** cho tiểu luận PCA (n = 569 mẫu, p = 30 đặc trưng).
Mọi con số / bảng / hình trong tiểu luận được sinh trực tiếp từ các script này (xem `CODE_REVIEW.md`).

## Cấu trúc
| File | Nội dung |
|---|---|
| `common.py` | Hàm dùng chung: tải dữ liệu, chia 80/20 phân tầng, chuẩn hóa chống leakage, Bartlett + KMO (slogdet), Horn, Kaiser, PCA, metric |
| `minimal_pipeline.py` | Pipeline 12 dòng (Listing 3.1) |
| `casestudy.py` | Thực nghiệm chính: 30D vs PCA q ∈ {1,2,3,5,6,7,10,15} → `pca_results.csv` |
| `pca_experiment.py` | Scaling, Kaiser/Horn, 5D→2D, ví dụ tay 2D→1D, 3 bộ phân loại |
| `phu_luc_A_pca_visualization.py` | Biplot, scree, scaling, heat-map loadings, 5D, đường cong 3 bộ phân loại |
| `phu_luc_B_cross_validation.py` | 5-fold CV bằng Pipeline + quy tắc 1-SE |
| `phu_luc_C_bootstrap.py` | Bootstrap B = 1000 + kiểm định McNemar chính xác |
| `phu_luc_D_repeated_splits.py` | 12 lần chia ngẫu nhiên + kiểm định cặp (t, Wilcoxon) |
| `phu_luc_E_geometry.py` | Hình hình học: Pythagoras, 4 bước, ví dụ tay, PCA vs OLS, 3D→2D, sơ đồ pipeline |
| `phu_luc_F_extensions.py` | PCA vs LDA, Kernel PCA, nén ảnh chữ số |
| `run_all.py` | Chạy tất cả theo thứ tự, lưu log vào `outputs/logs/` |

## Chạy
```bash
pip install -r requirements.txt
python run_all.py
```
Kết quả: CSV ở thư mục gốc, hình ở `figures/`, JSON + log ở `outputs/`.
Môi trường đã kiểm thử: Python 3.12.3, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.0.6 (hai lần chạy cho kết quả trùng khớp từng byte).
