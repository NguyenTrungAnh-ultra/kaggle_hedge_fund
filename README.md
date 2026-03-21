# Hedge Fund Kaggle - Time Series Forecasting Pipeline

Hệ thống dự báo chuỗi thời gian tài chính được xây dựng dựa trên các tiêu chuẩn **Advanced Machine Learning in Finance (AFML)** của Marcos López de Prado.

## 🚀 Điểm nhấn Kỹ thuật (Key Highlights)
- **AFML Compliance**: Chống rò rỉ dữ liệu (No Data Leakage) tuyệt đối.
- **Fractional Differencing (FFW)**: Giữ lại "bộ nhớ" chuỗi thời gian trong khi vẫn đạt tính dừng (Stationarity).
- **Purged K-Fold CV**: Chia tập dữ liệu khắt khe, loại bỏ nhiễu tại các ranh giới thời gian.
- **Weighted RMSE Metric**: Tối ưu trực tiếp theo hàm chấm điểm của Kaggle.

---

## 🏗️ Cấu trúc Pipeline

### Step 1: Xử lý dữ liệu khuyết thiếu (`Modules/Preprocess/handle_na.py`)
Quy trình lấp đầy NaN 4 lớp:
1. **Spatial Isolation**: Loại bỏ mã cổ phiếu quá nhiễu.
2. **Missingness Indicator**: Gắn cờ logic `is_nan_*` cho 48 cột dữ liệu. (Sẽ được tự động lọc bỏ tại bước huấn luyện nếu toàn 0 hoặc toàn 1 để tối ưu hóa không gian đặc trưng).
3. **Longitudinal F-Fill**: Điền giá trị từ quá khứ đắp sang theo từng mã cổ phiếu.
4. **Cold-Start Median Fill**: Điền trung vị trục chéo (Cross-sectional Median) cho các mã mới xuất hiện.

### Step 2: Dừng hóa dữ liệu (`Modules/Preprocess/make_stationary.py`)
- **Seasonal Handler (Mới)**: Tự động phát hiện và triệt tiêu tính không dừng mùa vụ bằng Dịch chuyển lùi (Seasonal Backshift) trên phân lớp Cross-sectional, tuân thủ nghiêm ngặt AFML.
- Áp dụng **Fractional Differencing** để khử tính không dừng (Non-Stationary).
- Tham số $d$ được ước tính chỉ dựa trên tập Train để ngăn chặn rò rỉ thông tin tương lai.
- Sử dụng Vectorized Convolution (`np.convolve`) để xử lý 5 triệu dòng dữ liệu trong thời gian ngắn.

### Step 3: Huấn luyện mô hình Hybrid (`07_afml_hybrid.py`)
- **CV Strategy**: Purged K-Fold (5 Folds) + Embargo.
- **Model**: LightGBM GBDT (`num_leaves=30`, `max_depth=8`).
- **Metric**: Weighted RMSE (Kaggle Score with `sqrt`).
- **Hybrid Approach**: Kết hợp tính ổn định của AFML với sự nhạy bén của tham số từ Notebook thực chiến.

### Step 4: Huấn luyện Ensemble (`06_train_ensemble.py`)
- **Trạng thái**: Đang huấn luyện (Giai đoạn cuối: Đã xong Tuning XGB/Cat, đang chạy MLP & Final Training).
- **GPU**: RTX 5060 Utilization ~90-95% (Hoạt động ổn định).
- **Tỉ lệ Tuning**: 30% sample (Gần 1.5 triệu dòng).
- **Kết quả Tuning**:
  - *Best XGB*: `n_estimators: 563, max_depth: 4, lr: 0.02`
  - *Best Cat*: `iterations: 451, depth: 6, lr: 0.06`
  - *Best MLP*: `hidden_dim: 128, lr: 0.0009, dropout: 0.44`
- **Stability**: Đã khắc phục lỗi `ValueError` (String conversion) bằng LabelEncoding.

### Step 5: Dự báo Ensemble & Nộp bài (`07_submit_ensemble.py`)
- **Weighted Average**: Kết hợp dự báo từ 3 mô hình theo trọng số hiệu năng thực tế.
- **AFML Memory Preservation**: Giữ nguyên cơ chế FFW và ffill để đảm bảo tính nhất quán giữa Train và Test.

---

## 📈 Kết quả đạt được (Current Performance)

| Metric | Kết quả (Bản Hybrid) |
| :--- | :--- |
| **Kaggle Final Score (Weighted RMSE)** | **0.1625** |
| **Test Zero-Mean R2** | **0.0264** |

---

## 🛠️ Hướng dẫn cài đặt
1. Tạo môi trường ảo: `python -m venv .venv`
2. Cài đặt thư viện: `pip install -r requirements.txt`
3. Chạy theo thứ tự các file script `handle_na.py` -> `make_stationary.py` -> `07` -> `05`.

---
*Dự án được phát triển bởi sự kết hợp giữa tư duy định lượng AFML và kỹ thuật thực chiến Kaggle.*
