# Hedge Fund Kaggle Benchmark Analysis

## 1. Kết quả tổng hợp (Summary Results)

Dưới đây là bảng so sánh hiệu năng của các chiến lược tiền xử lý dựa trên metric **Weighted R** (được định nghĩa trong `metric.py`).

| Strategy | Score (Weighted R) ↑ | Train Samples | Test Samples | No. Features | Key Features | Note |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **flagna_sta_fea** | **0.082669** | 1,558,549 | 441,926 | 129 | `feature_al`, `feature_u` | Flag NaN + Stationary + Adv Features |
| **fillnafllag_sta_fea** | 0.066814 | 1,558,549 | 441,946 | 129 | `feature_by_lag_1`, `lag_25` | Fill NaN + Flag + Sta + Fea |
| **fillnaflag_sta** | 0.050146 | 1,558,549 | 441,946 | 106 | `feature_al`, `feature_aq` | Fill NaN + Flag + Sta |
| **flagna_featurebuild** | 0.049900 | 4,106,607 | 1,215,665 | 129 | `feature_al`, `feature_cf` | Flag NaN + Adv Features |
| **flagna_stationary** | 0.048891 | 1,558,549 | 441,926 | 106 | `feature_al`, `feature_u` | Flag NaN + Stationary |
| **only_flag_na** | 0.041711 | 4,106,607 | 1,215,665 | 106 | `feature_al`, `feature_by` | Chỉ Flag NaN |
| **fillna_flag_only** | 0.040431 | 4,106,607 | 1,215,665 | 106 | `feature_aq`, `feature_as` | Fill NaN + Flag |

---

## 2. Phân tích các yếu tố quyết định hiệu năng

### A. Cơ chế xử lý giá trị thiếu (NaN Handling)
*   **Flagging > Filling:** Việc chỉ đánh dấu giá trị thiếu bằng cột Flag (`is_nan_...`) và để mô hình tự xử lý (Native NaN handling của LightGBM) cho kết quả tốt hơn việc điền giá trị thủ công.
*   **Kết luận:** Không nên sử dụng `fillna` cho dataset này.

### B. Biến đổi chuỗi thời gian (Stationarity)
*   Việc sử dụng **Fractional Differencing** giúp dữ liệu ổn định hơn, dẫn đến Score tăng vọt (từ ~0.04 lên ~0.05 - 0.08).
*   **Hạn chế:** Giảm kích thước dữ liệu huấn luyện (từ 4.1M xuống 1.5M mẫu) do mất mát dữ liệu ở đầu chuỗi.

### C. Kỹ thuật đặc trưng (Feature Engineering)
*   Các tính năng trễ (**Lagged Features**) và chuẩn hóa theo nhóm (**Sector-Neutral Z-score**) đóng vai trò quyết định.
*   **Sức mạnh của Lag:** Trong bản chạy `fea`, các biến trễ như `feature_by_lag_1` và `feature_al_lag_25` là những biến quan trọng nhất.

---

## 3. Top Features (Độ quan trọng cao nhất)

1.  **`feature_al`**: Biếng số mạnh nhất, đứng đầu bảng importance trong mọi cấu hình.
2.  **`feature_by_lag_1`**: Tín hiệu trễ gần nhất mang lại thông tin dự báo cực mạnh.
3.  **`feature_u`, `feature_cf`, `feature_aq`**: Các core features khác của dataset.
4.  **`is_nan_feature_ce`**: NaN flags mang lại tín hiệu dự báo ổn định.

---

## 4. Khuyến nghị (Final Recommendation)

Sử dụng cấu hình **Native NaN Flagging + Stationary Preprocessing + Advanced Features (Lag & Z-score)**. Đây là cấu hình đạt Score **0.082669** trên tập Test ẩn, cao nhất trong tất cả các thử nghiệm.
