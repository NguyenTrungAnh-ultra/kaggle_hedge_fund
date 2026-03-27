# Hedge Fund Kaggle - Time Series Forecasting Pipeline

Hệ thống dự báo chuỗi thời gian tài chính được xây dựng dựa trên các tiêu chuẩn **Advanced Machine Learning in Finance (AFML)** của Marcos López de Prado.

## 🚀 Điểm nhấn Kỹ thuật (Key Highlights)

- **AFML Compliance**: Chống rò rỉ dữ liệu (No Data Leakage) tuyệt đối. Các tham số được ước lượng chỉ dựa trên 80% thời gian đầu của tập Train.
- **Unified Pipeline**: Tiền xử lý hợp nhất Train/Test để đảm bảo tính liên tục của bộ nhớ (Memory Preservation) cho các phép toán chuỗi thời gian.
- **Fractional Differencing (FFW)**: Giữ lại "bộ nhớ" chuỗi thời gian trong khi vẫn đạt tính dừng (Stationarity).
- **Alpha Injection**: Cấy ghép Trí nhớ Trục dọc (Lags) và Sức mạnh Trục ngang (Sector-Neutral Z-Score) để tách Alpha khỏi Beta thị trường.
- **Quant Security (Brakes)**: Triệt tiêu nhiễu bằng "Structural Brakes" (Trees lùn) và "Penalty Brakes" (L1/L2 cao) thay vì Clipping thủ công, giúp mô hình ổn định trước các Fat-Tails.

---

## 🏗️ Kiến trúc các Modules (`Modules/Preprocess/`)

### 1. `handle_na.py` (NaN Handler - 5 Lớp Bảo vệ)
Quy trình phục hồi dữ liệu khuyết thiếu khép kín:
- **Layer 1: Analysis**: Phân tích tỷ lệ trống ban đầu.
- **Layer 2: Missingness Indicator**: Gắn cờ `is_nan_*` cho các cột quan trọng.
- **Layer 3: Longitudinal Forward Fill**: Điền giá trị từ quá khứ (ffill) theo từng nhóm `code` + `horizon` (giới hạn 50 bước).
- **Layer 4: Cold-Start Median Fill**: Điền Trung vị trục chéo (Cross-sectional Median) cho các mã mới xuất hiện hoặc thiếu lịch sử.
- **Layer 5: Feature Purging**: Loại bỏ các cột hằng số và danh sách đen (Blacklist) các features nhiễu cao.

- **Fractional Differencing (FFW)**: Áp dụng chuẩn AFML với cửa sổ tĩnh (Fixed-Width Window). Đã fix lỗi đảo weights và thêm NaN Padding cho phần warm-up để chống rò rỉ dữ liệu.
- **Stationarity Suite**: Tự động tìm $d$ tối ưu (0.1 - 1.0) để cân bằng giữa tính dừng và tính nhớ (Memory Preservation).

### 3. `feature_builder.py` (Alpha Generator)
- **Lagged Momentum**: Tạo 20 đặc trưng trễ (1, 3, 5, 10, 25) cho các cột Core Features.
- **Sector-Neutral Z-Score**: Chuẩn hóa Z-Score chéo giữa các mã trong cùng một nhóm ngành (`sub_category`). Giúp mô hình tập trung vào "Sức mạnh tương đối" hơn là giá trị tuyệt đối.

### 4. `Regularization Strategy` (Financial Guardrails)
- **Structural Brakes**: Ép cây lùn (max_depth: 3-6) để tránh học vẹt nhiễu thị trường.
- **Penalty Brakes**: L1/L2 mạnh (Lambda L1/L2 cao) để tự động chọn lọc features và kìm hãm ngoại lai thay cho Target Clipping truyền thống.
- **Sampling Brakes**: Giảm `feature_fraction` xuống cực thấp (0.4-0.8) để buộc các cây trong ensemble nhìn vào các góc độ khác nhau của thị trường.

---

## ⚙️ Quy trình thực thi (Execution Roadmap)

### Luồng 1: Tiền xử lý Hợp nhất (Phòng chống Leakage)
Sử dụng script điều phối: `Modules/Preprocess/inference_preprocess.py`
- Tự động nạp Train + Test đồng thời.
- Chạy chuỗi: **NaN Handler** -> **FracDiff** -> **Alpha Generator**.
- Xuất file: `ts-forecasting/inference_processed_full.parquet`.

### Luồng 2: Huấn luyện mô hình (Multi-Model Architecture)
Hệ thống hiện hỗ trợ 3 kiến trúc mô hình với cùng một triết lý AFML:
- **LightGBM** (`Models/lightgbm.py`): CPU-optimized, Fast.
- **XGBoost** (`Models/XGboost.py`): GPU-accelerated (RTX 5060), dùng Hist tree method.
- **CatBoost** (`Models/CatBoost.py`): GPU-accelerated, xử lý tốt nhiễu từ các đặc trưng kỹ thuật.

**Đặc điểm chung:**
- Sử dụng **Leak-Free Purged K-Fold CV** (5 Folds) có Embargo.
- Tự động tối ưu tham số bằng **Optuna** trên 20% khung thời gian cuối.
- Model được lưu tại `models_param/`.

### Luồng 3: Inference & Submission (`FINAL_submit.py`)
- **Transductive Encoding**: Đã chuyển LabelEncoder ra ngoài luồng train/test để đảm bảo tính nhất quán.
- **Metric - `weighted_rmse_score`**: Chú ý đây là metric **ĐIỂM CÀNG CAO CÀNG TỐT** (Max: 1.0).
- **Ensemble Strategy**: Hiện tại sử dụng trung bình cộng dự báo từ 5-Fold LightGBM (đã hỗ trợ auto-retrain bằng cờ `retrain=1`).

---

## 🛠️ Hướng dẫn cài đặt & Chạy nhanh

1. `pip install -r requirements.txt`
2. Tạo dữ liệu Inference: `python Modules/Preprocess/inference_preprocess.py`
3. Huấn luyện (Cần chọn mô hình):
   - `python Models/Lightgbm.py`
   - `python Models/XGboost.py`
   - `python Models/CatBoost.py`
4. Tạo file submission: `python FINAL_submit.py` (Mặc định dùng LightGBM Ensemble)

---
_Cấu trúc được tối ưu cho RTX 5060 và CPU i5-12400F, xử lý 5M+ dòng dữ liệu tài chính._
