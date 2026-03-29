# Hedge Fund Kaggle Benchmark Analysis

## 1. Kết quả tổng hợp (Summary Results)

Dưới đây là bảng so sánh hiệu năng của các chiến lược tiền xử lý dựa trên metric **Weighted R** (được định nghĩa trong `metric.py`).

| Strategy | Score (Weighted R) ↑ | Train Samples | Test Samples | No. Features | Key Features | Note |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **nam.py** | **0.136134** | 4,000,000+ | 1,000,000+ | 110 | `horizon`, `sub_category` | RAW Data + fillna(0) + No Purge |
| **raw_sta_fea_noblacklist** | 0.110590 | 4,106,607 | 1,215,665 | 212 | `horizon`, `sub_category` | No Blacklist + Raw + Sta + Fea |
| **withraw_sta_fea** | 0.085471 | 4,106,607 | 1,215,665 | 208 | `feature_al`, `feature_by_sta_lag_1` | Raw + Flag NaN + Sta + Adv Features |
| **flagna_sta_fea** | 0.082669 | 1,558,549 | 441,926 | 129 | `feature_al`, `feature_u` | Flag NaN + Stationary + Adv Features |
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
*   Việc sử dụng **Fractional Differencing** giúp dữ liệu ổn định hơn, dẫn đến Score tăng vọt.
*   **Giải quyết mất mát dữ liệu:** Trải qua thử nghiệm ban đầu (giảm mẫu từ 4.1M xuống 1.5M do hạn chế của các biến trễ), chúng ta đã khắc phục thành công bằng cách gộp (merge) lại cột Stationary và thông tin độ trễ vào bộ dữ liệu thô (Raw). Cách tiếp cận này giúp vừa giữ lại 100% dòng dữ liệu (4.1M mẫu train), vừa tận dụng tối đa sức mạnh của Fractional Differencing trong cấu hình `withraw_sta_fea`.

### C. Kỹ thuật đặc trưng (Feature Engineering) & Tầm quan trọng của Metadata
*   **Sức mạnh của Meta-Features (No Blacklist):** Khi thử nghiệm loại bỏ Blacklist và đưa lại các biến cấu trúc như `horizon`, `sub_category`, `code`, `sub_code` vào mô hình (thông qua Label Encoding), hiệu năng đã nhảy vọt đáng kinh ngạc (Score từ 0.085 lên **0.110**).
*   **Giải mã sức mạnh:** Tính chất động của thị trường tài chính phụ thuộc cực kì mạnh mẽ vào khoảng thời gian dự báo (`horizon`) và nhóm ngành/lĩnh vực hoạt động (`sub_category`). Việc cung cấp thông tin "ngữ cảnh" này giúp cây quyết định (LightGBM) bóc tách quy luật cho từng thị trường ngách tốt hơn thay vì học một hàm chung chung.
*   Các tính năng trễ (**Lagged Features**) và chuẩn hóa theo nhóm (**Sector-Neutral Z-score**) vẫn tiếp tục đem lại giá trị thông tin tín hiệu cao.

---

## 3. Top Features (Độ quan trọng cao nhất)

1.  **`horizon` & `sub_category`**: Xếp hạng #1 và #2 về độ quan trọng trong mô hình no-blacklist. Tầm quan trọng vượt trội này chứng minh "Ngữ cảnh" và "Khung thời gian dự báo" quan trọng hơn cả biến động giá trị của features gốc.
2.  **`feature_al`**: Biếng số biểu diễn trạng thái mạnh nhất, duy trì sức mạnh tuyệt đối trong nhóm numerical features mảng tài chính.
3.  **`feature_bp`, `feature_u`, `feature_cf`**: Các core features gốc (và biến thể dừng) mang lại hiệu quả dự báo mạnh mẽ và ổn định.
4.  **`feature_by_sta_lag_1`**: Vẫn duy trì là một trong những tín hiệu độ trễ mạnh nhất kết hợp chuỗi dừng, đóng vai trò như Momentum dự báo ngắn hạn.

4.  **`feature_by_sta_lag_1`**: Vẫn duy trì là một trong những tín hiệu độ trễ mạnh nhất kết hợp chuỗi dừng, đóng vai trò như Momentum dự báo ngắn hạn.

---

## 4. Các vấn đề cần lưu ý (Critical Pitfalls & Caveats)

Trong quá trình thực nghiệm, mỗi chiến lược đều bộc lộ những vấn đề kỹ thuật cần cân nhắc trước khi sử dụng cho môi trường Production:

### A. Nguy cơ Rò rỉ dữ liệu (Data Leakage) - Trường hợp `nam.py`
*   **Vấn đề:** `nam.py` đạt Score cao nhất (**0.136**) nhưng **KHÔNG sử dụng Purging & Embargo**. Trong dữ liệu chuỗi thời gian, các biến có tính chất trễ (Lags) sẽ khiến thông tin từ tương lai rò rỉ vào quá khứ nếu không xóa bỏ vùng đệm giữa Train và Test.
*   **Đánh giá:** Điểm số này có khả năng cao là "vàng giả" (Optimistic bias). Khi submit lên Private Leaderboard của Kaggle, điểm có thể sụt giảm mạnh.

### B. Mất mát mẫu dữ liệu (Sample Loss) - Các bản `sta_fea` cũ
*   **Vấn đề:** Các bản chỉ sử dụng Stationary features (không merge Raw) bị mất khoảng **60% dữ liệu huấn luyện** (từ 4.1M xuống 1.5M rows) do đặc tính của phép tính Fractional Differencing cần một cửa sổ dữ liệu quá khứ.
*   **Hệ quả:** Mô hình thiếu dữ liệu ở các giai đoạn thị trường quan trọng ở đầu tập dữ liệu.

### C. Vấn đề "Neo" dữ liệu ảo - `fillna(0)`
*   **Vấn đề:** Sử dụng `fillna(0)` trong `nam.py` ép các giá trị thiếu về 0 thay vì để LightGBM tự xử lý. 
*   **Hệ quả:** Có thể tạo ra nhiễu nếu giá trị 0 thực sự cũng tồn tại trong feature đó, làm mô hình hiểu sai về phân phối của dữ liệu "không có sẵn". Cấu hình **Native NaN Handling** luôn được khuyến khích hơn.

### D. Độ tin cậy của Cross-Validation - `nokfold`
*   **Vấn đề:** Các bản chạy nhanh (single split) chỉ đánh giá trên 1 cửa sổ thời gian duy nhất. 
*   **Hệ quả:** Điểm số phụ thuộc rất lớn vào việc giai đoạn Val đó có "dễ đoán" hay không. Bản **5-Fold Purged CV** vẫn là thước đo đáng tin cậy nhất về độ ổn định (Generalization).

---

## 5. Khuyến nghị (Final Recommendation)

Sử dụng cấu hình **Un-blacklisting Context Variables (Horizon, Sub-category) + Raw Data Merge + Native NaN Flagging + Stationary Preprocessing + Advanced Features (Lag & Z-score)** (cấu hình `raw_sta_fea_noblalckist`). Thiết lập này vừa bảo toàn được kích thước dữ liệu (4.1M samples), vừa khai thác trọn vẹn ngữ cảnh (212 features bao gồm Categorical & Metadata). Điểm Weighted RMSE Score đạt dấu mốc kỷ lục là **0.110590**, là Pipeline vững chắc và chính xác nhất hiện nay cho việc sinh ra các dự đoán Submission cuối cùng.
