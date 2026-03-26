# Hedge Fund Kaggle - Time Series Forecasting Pipeline

Hệ thống dự báo chuỗi thời gian tài chính được xây dựng dựa trên các tiêu chuẩn **Advanced Machine Learning in Finance (AFML)** của Marcos López de Prado.

## 🚀 Điểm nhấn Kỹ thuật (Key Highlights)

- **AFML Compliance**: Chống rò rỉ dữ liệu (No Data Leakage) tuyệt đối. Các tham số được ước lượng chỉ dựa trên 80% thời gian đầu của tập Train.
- **Unified Pipeline**: Tiền xử lý hợp nhất Train/Test để đảm bảo tính liên tục của bộ nhớ (Memory Preservation) cho các phép toán chuỗi thời gian.
- **Fractional Differencing (FFW)**: Giữ lại "bộ nhớ" chuỗi thời gian trong khi vẫn đạt tính dừng (Stationarity).
- **Alpha Injection**: Cấy ghép Trí nhớ Trục dọc (Lags) và Sức mạnh Trục ngang (Sector-Neutral Z-Score) để tách Alpha khỏi Beta thị trường.
- **Quant Security (Clipping)**: Bảo vệ hàm mất mát bằng Target Clipping (0.01 - 0.99) để triệt tiêu nhiễu từ các Fat-Tails.

---

## 🏗️ Kiến trúc các Modules (`Modules/Preprocess/`)

### 1. `handle_na.py` (NaN Handler - 5 Lớp Bảo vệ)
Quy trình phục hồi dữ liệu khuyết thiếu khép kín:
- **Layer 1: Analysis**: Phân tích tỷ lệ trống ban đầu.
- **Layer 2: Missingness Indicator**: Gắn cờ `is_nan_*` cho các cột quan trọng.
- **Layer 3: Longitudinal Forward Fill**: Điền giá trị từ quá khứ (ffill) theo từng nhóm `code` + `horizon` (giới hạn 50 bước).
- **Layer 4: Cold-Start Median Fill**: Điền Trung vị trục chéo (Cross-sectional Median) cho các mã mới xuất hiện hoặc thiếu lịch sử.
- **Layer 5: Feature Purging**: Loại bỏ các cột hằng số và danh sách đen (Blacklist) các features nhiễu cao.

### 2. `make_stationary.py` (Stationarity Suite)
- **Fractional Differencing (FFW)**: Sử dụng cửa sổ động (Fixed-Width Window) và tích chập (`np.convolve`) để xử lý dữ liệu lớn. Tự động tìm tham số $d$ tối ưu cho từng đặc trưng.
- **Seasonal Handler**: (Sẵn sàng) Module triệt tiêu tính không dừng mùa vụ bằng Seasonal Backshift.

### 3. `feature_builder.py` (Alpha Generator)
- **Lagged Momentum**: Tạo 20 đặc trưng trễ (1, 3, 5, 10, 25) cho các cột Core Features.
- **Sector-Neutral Z-Score**: Chuẩn hóa Z-Score chéo giữa các mã trong cùng một nhóm ngành (`sub_category`). Giúp mô hình tập trung vào "Sức mạnh tương đối" hơn là giá trị tuyệt đối.

### 4. `OutlierHandler.py` (Model Protection)
- **Target Clipping**: Ép giá trị `y_target` vào khoảng phân vị [1%, 99%] để tránh tối ưu sai lệch do các cú sốc thị trường ngắn hạn.

---

## ⚙️ Quy trình thực thi (Execution Roadmap)

### Luồng 1: Tiền xử lý Hợp nhất (Phòng chống Leakage)
Sử dụng script điều phối: `Modules/Preprocess/inference_preprocess.py`
- Tự động nạp Train + Test đồng thời.
- Chạy chuỗi: **NaN Handler** -> **FracDiff** -> **Alpha Generator**.
- Xuất file: `ts-forecasting/inference_processed_full.parquet`.

### Luồng 2: Huấn luyện mô hình (`Models/lightgbm.py`)
- Sử dụng **Purged K-Fold CV** (5 Folds) có Embargo.
- Tự động tối ưu tham số bằng **Optuna** trên 20% khung thời gian cuối.
- Model được lưu tại `models_param/`.

### Luồng 3: Suy luận & Nộp bài (`FINAL_submit.py`)
- Nạp dữ liệu đã xử lý từ Step 1.
- Thực hiện **Transductive Encoding** (LabelEncoder trên 100% dữ liệu).
- **Exponential Time-Decay Ensemble**: Kết hợp dự báo từ 5 model folds với trọng số giảm dần theo độ cũ của dữ liệu.

---

## 🛠️ Hướng dẫn cài đặt & Chạy nhanh

1. `pip install -r requirements.txt`
2. Tạo dữ liệu Inference: `python Modules/Preprocess/inference_preprocess.py`
3. Training (Nếu cần): `python Models/lightgbm.py` (Set `training=True`)
4. Tạo file submission: `python FINAL_submit.py`

---
_Cấu trúc được tối ưu cho hiệu năng xử lý dữ liệu 5M+ dòng trên CPU/GPU thông thường._
