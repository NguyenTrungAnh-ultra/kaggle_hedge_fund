# Hedge Fund Kaggle - Time Series Forecasting Pipeline

Hệ thống dự báo chuỗi thời gian tài chính được xây dựng dựa trên các tiêu chuẩn **Advanced Machine Learning in Finance (AFML)** của Marcos López de Prado.

## 🚀 Điểm nhấn Kỹ thuật (Key Highlights)

- **AFML Compliance**: Chống rò rỉ dữ liệu (No Data Leakage) tuyệt đối. Các tham số được ước lượng chỉ dựa trên dữ liệu Train.
- **Unified Pipeline**: Tiền xử lý hợp nhất Train/Test để đảm bảo tính liên tục của bộ nhớ (Memory Preservation) cho các phép toán chuỗi thời gian (shift, lag).
- **Log Return**: Biến đổi log-differencing cho các biến toàn dương, đảm bảo tính dừng. Có guard chống NaN im lặng (clip giá trị âm + báo cáo).
- **Alpha Injection**: Cấy ghép Trí nhớ Trục dọc (Lagged Momentum) để nắm bắt động lượng ngắn/trung hạn.
- **Quant Security (Brakes)**: Triệt tiêu nhiễu bằng "Structural Brakes" (Trees lùn) và "Penalty Brakes" (L1/L2 cao) thay vì Clipping thủ công, giúp mô hình ổn định trước các Fat-Tails.
- **Native Categorical**: Các cột phân loại (`code`, `sub_code`, `sub_category`, `horizon`) được xử lý trực tiếp bởi LightGBM qua `categorical_feature`, không qua LabelEncoder.

---

## 🏗️ Kiến trúc các Modules (`Modules/Preprocess/`)

### 1. `handle_na.py` (NaN Handler)
Quy trình phục hồi dữ liệu khuyết thiếu:
- **Layer 2: Missingness Indicator**: Gắn cờ `is_nan_*` cho từng cột; chỉ tạo nếu cột có ít nhất 1 NaN và không phải toàn NaN.
- **Layer 3: Longitudinal Forward Fill** *(tùy chọn, mặc định tắt)*: Điền từ quá khứ theo nhóm `code` + `horizon` (giới hạn 20 bước).
- **Layer 4: Cross-sectional Median Fill** *(tùy chọn, mặc định tắt)*: Điền Trung vị chéo theo `ts_index` + `horizon`.
- **Layer 5: Feature Purging**: Loại bỏ các cột hằng số được phát hiện **động** (không hardcode blacklist).

> Mặc định chạy: Layer 2 + Layer 5. Layer 3 & 4 bật bằng `impute=True`.

### 2. `LogReturn.py` (Log Return Handler)
Biến đổi log-differencing cho các biến toàn dương:
- **Công thức**: `X'_t = log1p(X_t) - log1p(X_{t-1})`, shift an toàn theo group.
- **Guard chống NaN im lặng**: Clip giá trị âm về 0 trước `log1p`, in cảnh báo nếu có.
- **NaN Report**: Sau mỗi lần chạy, báo cáo tổng NaN và phân biệt NaN ranh giới group (bình thường) vs NaN bất thường.
- **Leakage-free**: Danh sách features toàn dương được xác định **chỉ từ tập Train** (`is_test == 0`).

### 3. `feature_builder.py` (Alpha Generator)
- **Lagged Momentum**: Tạo 15 đặc trưng trễ (lag = 1, 3, 5, 10, 25) cho 3 cột Core Log-Return Features.
- **Shift Engine**: Dùng `groupby.shift()` chuẩn của pandas — đúng toán học, không có rủi ro race condition.
- **NaN Report**: Phân biệt NaN ranh giới group (bình thường) vs NaN bất thường sau mỗi lần tạo lag.
- **Sector-Neutral Z-Score** *(chưa kích hoạt)*: Chuẩn hóa Z-Score chéo giữa các mã cùng ngành. Sẵn sàng bật khi cần.

### 4. `Regularization Strategy` (Financial Guardrails)
- **Structural Brakes**: Ép cây lùn (max_depth: 3-6) để tránh học vẹt nhiễu thị trường.
- **Penalty Brakes**: L1/L2 mạnh (Lambda L1/L2 cao) để tự động chọn lọc features và kìm hãm ngoại lai thay cho Target Clipping truyền thống.
- **Sampling Brakes**: Giảm `feature_fraction` (0.4-0.8) để buộc các cây trong ensemble nhìn vào các góc độ khác nhau của thị trường.

---

## ⚙️ Quy trình thực thi (Execution Roadmap)

### Luồng 1: Tiền xử lý Hợp nhất (Phòng chống Leakage)
Sử dụng script điều phối: `Modules/Preprocess/inference_preprocess.py`
- Tự động nạp Train + Test đồng thời, gán cờ `is_test`.
- Memory Optimization: downcast float64 → float32 (bảo vệ `y_target`, `weight`, `ts_index`).
- Chạy chuỗi: **NaN Handler** → **Log Return** → **Alpha Generator (Lagged Momentum)**.
- Xuất file: `ts-forecasting/inference_processed_full.parquet`.

### Luồng 2: Huấn luyện mô hình (Multi-Model Architecture)
Hệ thống hiện hỗ trợ 3 kiến trúc mô hình với cùng một triết lý AFML:
- **LightGBM** (`Models/Lightgbm.py`): CPU-optimized, Fast.
- **XGBoost** (`Models/XGboost.py`): GPU-accelerated (RTX 5060), dùng Hist tree method.
- **CatBoost** (`Models/CatBoost.py`): GPU-accelerated, xử lý tốt nhiễu từ các đặc trưng kỹ thuật.

**Đặc điểm chung:**
- Sử dụng **Leak-Free Purged K-Fold CV** (5 Folds) có Embargo.
- Tự động tối ưu tham số bằng **Optuna** trên 20% khung thời gian cuối.
- Categorical columns xử lý native qua `categorical_feature` của LightGBM/XGBoost/CatBoost.
- Model được lưu tại `models_param/`.

### Luồng 3: Inference & Submission (`FINAL_submit.py`)
- **Metric - `weighted_rmse_score`**: Chú ý đây là metric **ĐIỂM CÀNG CAO CÀNG TỐT** (Max: 1.0).
- **Ensemble Strategy**: Trung bình cộng dự báo từ 5-Fold LightGBM (hỗ trợ auto-retrain bằng cờ `retrain=1`).

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
_Cấu trúc được tối ưu cho RTX 5060 và CPU i5-12400F, xử lý 6M+ dòng dữ liệu tài chính._
