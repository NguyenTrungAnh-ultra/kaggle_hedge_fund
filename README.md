# Hedge Fund Kaggle - Time Series Forecasting Pipeline

Hệ thống dự báo chuỗi thời gian tài chính được xây dựng dựa trên các tiêu chuẩn **Advanced Machine Learning in Finance (AFML)** của Marcos López de Prado.

## 🚀 Điểm nhấn Kỹ thuật (Key Highlights)
- **AFML Compliance**: Chống rò rỉ dữ liệu (No Data Leakage) tuyệt đối.
- **Fractional Differencing (FFW)**: Giữ lại "bộ nhớ" chuỗi thời gian trong khi vẫn đạt tính dừng (Stationarity).
- **Purged K-Fold CV**: Chia tập dữ liệu khắt khe, loại bỏ nhiễu tại các ranh giới thời gian.
- **Weighted RMSE Metric**: Tối ưu trực tiếp theo hàm chấm điểm của Kaggle.

---

## 🏗️ Cấu trúc Pipeline

### Step 1: Xử lý dữ liệu khuyết thiếu (`01_handle_na.py`)
Quy trình lấp đầy NaN 4 lớp:
1. **Spatial Isolation**: Loại bỏ mã cổ phiếu quá nhiễu.
2. **Missingness Indicator**: Gắn cờ logic `is_nan_*` cho 48 cột dữ liệu.
3. **Longitudinal F-Fill**: Điền giá trị từ quá khứ đắp sang theo từng mã cổ phiếu.
4. **Cold-Start Median Fill**: Điền trung vị trục chéo (Cross-sectional Median) cho các mã mới xuất hiện.

### Step 2: Dừng hóa dữ liệu (`02_make_stationary.py`)
- Áp dụng **Fractional Differencing** để khử tính không dừng (Non-Stationary).
- Tham số $d$ được ước tính chỉ dựa trên tập Train để ngăn chặn rò rỉ thông tin tương lai.
- Sử dụng Vectorized Convolution (`np.convolve`) để xử lý 5 triệu dòng dữ liệu trong thời gian ngắn.

### Step 3: Huấn luyện mô hình Hybrid (`07_afml_hybrid.py`)
- **CV Strategy**: Purged K-Fold (5 Folds) + Embargo.
- **Model**: LightGBM GBDT (`num_leaves=30`, `max_depth=8`).
- **Metric**: Weighted RMSE (Kaggle Score with `sqrt`).
- **Hybrid Approach**: Kết hợp tính ổn định của AFML với sự nhạy bén của tham số từ Notebook thực chiến.

### Step 4: Dự báo & Nộp bài (`05_submit.py`)
- **Time-series Memory**: Kết hợp dữ liệu cũ và mới để đảm bảo các đặc trưng FFW và ffill có đủ lịch sử để hoạt động chính xác trên tập Test.
- **Safe Encoding**: Xử lý các nhãn (Code/Sub-code) lạ xuất hiện trong tương lai.
- **Ensembling**: Trung bình cộng dự báo từ 5 Folds đã huấn luyện.

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
3. Chạy theo thứ tự các file script `01` -> `02` -> `07` -> `05`.

---
*Dự án được phát triển bởi sự kết hợp giữa tư duy định lượng AFML và kỹ thuật thực chiến Kaggle.*
