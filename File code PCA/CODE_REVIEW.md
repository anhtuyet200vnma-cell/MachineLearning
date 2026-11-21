# Nhật ký rà soát code và nội dung PCA

Quy trình: (1) chạy lại **toàn bộ** code gốc ở thư mục sạch; (2) đối chiếu output với CSV, slide, tiểu luận và compendium;
(3) kiểm chứng độc lập các giá trị nghi ngờ; (4) sửa code; (5) chạy lại từ đầu và kiểm tra tái lập.

## A. Kết luận về code gốc
Code gốc **chạy không lỗi** và tái lập đúng các CSV có sẵn. Tuy nhiên có các điểm sai/yếu sau (đã sửa):

| # | Vị trí | Vấn đề | Cách sửa |
|---|---|---|---|
| 1 | `pca_experiment.py`, Phụ lục A | `PCA.explained_variance_` (ddof=1) lớn hơn trị riêng thật của ma trận tương quan R (StandardScaler dùng ddof=0) đúng hệ số n/(n−1): tổng = 30.066 chứ không phải 30; văn bản cũ ghi "trace(R)=30.0000" nhưng lại dùng 13.3533 | Thêm `correlation_eigenvalues()`; mọi λ_k nay là trị riêng của R (tổng = p). Kaiser/Horn so sánh đúng đại lượng |
| 2 | `check_pca_suitability` | `np.linalg.det` bị cắt ngưỡng 1e-300; `pinv` thay `inv`; chuỗi "p < 1e-300" **in cứng**, không tính | Dùng `slogdet`; thêm `log10_chi2_sf()` tính log10 p (≈ −6364); trả thêm MSA từng biến |
| 3 | `pca_experiment.py` | Horn đếm mọi thành phần vượt ngưỡng thay vì dừng ở thành phần đầu tiên không vượt | `count_horn()` dừng ở lần fail đầu tiên; `count_kaiser()` tương tự |
| 4 | Phụ lục B | `scoring="f1"` mặc định là F1 của lớp **Benign** (pos_label=1), trong khi mọi chỗ khác dùng **Malignant** làm lớp dương | Dùng `make_scorer(..., pos_label=0)`; thêm Mean Malignant Recall; thêm quy tắc 1-SE |
| 5 | Phụ lục A | "q = 6", "q = 10", ngưỡng màu cột **gán cứng** trong nhãn biểu đồ; nhãn tiếng Việt; chỉ số đặc trưng biplot gán cứng | Tính từ dữ liệu (Kaiser/Horn/CEV); nhãn tiếng Anh; chọn mũi tên tự động |
| 6 | `load_data` | tên đặc trưng là `np.str_` (in ra `np.str_('mean radius')`) | Ép về `str` |
| 7 | Phụ lục C, D | Chỉ có CI của PCA10; chưa so sánh ghép cặp có kiểm định | Thêm bootstrap baseline, McNemar chính xác, t-test ghép cặp và Wilcoxon |
| 8 | Kernel PCA (compendium) | Chọn thủ công "PC3" bằng nhãn và đo accuracy trên **tập train** → kết quả 100% không hợp lệ | Quy trình đúng: tách train/test, chọn γ bằng CV trên train, báo cáo test (Phụ lục F) |

Thêm mới: `phu_luc_E_geometry.py`, `phu_luc_F_extensions.py`, `minimal_pipeline.py`, `run_all.py`; các CSV mới `multi_classifier_results.csv`,
`hand_example_results.csv`, `pca_5d_eigenvectors.csv`, `kernel_pca_results.csv`, `digits_pca_results.csv`, `pca_vs_lda_results.csv`.

## B. Các số liệu trong slide / tiểu luận cũ KHÔNG khớp output thật (đã thay bằng số thật)

| Nội dung | Văn bản cũ | Giá trị thật (đã kiểm chứng) |
|---|---|---|
| Bartlett χ² (train) | 24,251.4 | **31,590.9** (df = 435; 2 cách tính định thức cho cùng kết quả) |
| KMO | 0.8324 | **0.8321** |
| 5D→2D: λ₁, λ₂ | 3.6475 (72.95%), 1.0720 (21.44%) | **3.4805 (69.61%), 1.2416 (24.83%)** |
| 5D→2D: CEV(2) | 94.39% | **94.44%** (README cũ ghi đúng 94.44%) |
| Eigenvectors 5D | PC1 = (.491, .498, .490, .409, .319) | **(.512, .520, .511, .392, .231)**; các PC khác cũng khác |
| Ngưỡng Horn | 1.6420, 1.5210, 1.4380 … | **1.5768, 1.4975, 1.4275 …** (khớp cạnh Marchenko–Pastur 1.5795) |
| CEV q=4, 5, 6, 7 | 79.58, 85.10, 89.03, 91.21 | **79.63, 85.14, 89.08, 91.26** |
| Trị riêng PC4, PC8–PC10 | 2.0082; 0.482, 0.389, 0.3466 | **2.0174; 0.4750, 0.3836, 0.3436** |
| Unscaled EVR PC2, PC3 | 1.68%, 0.18% | **1.72%, 0.14%** |
| Phương sai worst area | ≈ 324,000 (toàn bộ 569 mẫu) | **305,728** (tập train, n = 455); hai biến diện tích chiếm 99.2% tổng phương sai thô |
| CV 5-fold q=10 / baseline | 97.80% / 98.02% (fit PCA trước CV → leakage nhẹ) | **98.24% ± 0.60% / 97.80% ± 1.10%** (Pipeline, không leakage) |
| PCA q=2 (đoạn code Chương 3) | 93.86% acc, 90.48% recall | **94.74% acc, 95.24% recall** |
| Ví dụ tay: Σd², a₁ | 0.584; (0.686, 0.728); slope 1.0606 (compendium) | **0.582** (MSE = 0.1455 = λ₂); a₁ = **(0.6856, 0.7280)**, slope **1.0618** |
| Loadings PC1/PC2/PC3 | 0.2608; 0.2584; 0.3652; 0.3741 | Fit trên train: **0.2614; 0.2578; 0.3646; 0.3765** (ở dạng corr: 0.954, 0.941, 0.869, 0.637) |
| Communality (q=2) | 97.8%, 96.9%, 95.4%, 93.8% | **91.8%, 90.8%, 82.3%** (texture error chỉ **4.7%**; cần q=10 mới đạt 94.2%) |
| Novembre et al. 2008 | r = 0.87 (vĩ độ), 0.61 (kinh độ), ">90% chính xác" | r² = **0.77 / 0.78** (sau xoay trục); **50% trong ~310 km, 90% trong ~700 km** |
| Turk & Pentland 1991 | q ≈ 7–40 eigenfaces, RMS ≈ 2% | Không tìm thấy trong nguồn → bỏ; giữ **M′ = 7**, 96% / 85% / 64% (đã xác nhận) |
| Litterman & Scheinkman 1991 | "> 98%" | **≈ 97% trung bình (tối thiểu ≈ 94%)** |

## C. Điều mới được trung thực hóa trong kết quả
* PCA q=10 **không cải thiện** accuracy (97.37% vs 98.25% = đúng 1 mẫu); khác biệt **không có ý nghĩa thống kê**
  (bootstrap CI của hiệu: [−2.63, 0.00] điểm %, McNemar p = 1.00, 12 lần chia: 97.95% vs 97.81%, t-test p = 0.64).
* Recall ác tính 97.62% chỉ là giá trị của split seed 42; trung bình 12 split là 96.83% (min 90.48%).
* Biến *texture error* có MSA = 0.469 < 0.5, ít được biểu diễn bởi 2 thành phần đầu.
