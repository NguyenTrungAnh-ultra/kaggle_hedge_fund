# Báo Cáo Phân Tích Giá Trị Rỗng (NaN Analysis Report)

Báo cáo này tập trung vào việc bóc tách các đặc tính của các giá trị rỗng trong bộ dữ liệu `train.parquet`, giúp hiểu rõ nguồn gốc và chiến thuật xử lý dữ liệu tối ưu.

## 1. Tổng quan (Overall Statistics)

- **Tổng số dòng:** 5,337,414
- **Tổng số cột:** 94
- **Số cột có giá trị rỗng:** 48 cột
- **Số cột sạch (0% NA):** 46 cột (Các biến định danh: `id`, `code`, `ts_index`, `horizon` và các features `feature_a`...)

## 2. Top 10 Features có tỷ lệ rỗng cao nhất

| Feature | Tỷ lệ rỗng (NaN %) | Số lượng dòng trống |
| :--- | :--- | :--- |
| **feature_at** | **12.47%** | 665,676 |
| **feature_by** | **11.02%** | 588,140 |
| **feature_ay** | **8.54%** | 455,920 |
| **feature_cd** | **7.50%** | 400,114 |
| **feature_ce** | **5.17%** | 275,829 |
| **feature_cf** | **4.43%** | 236,389 |
| **feature_al** | **4.22%** | 225,416 |
| **feature_aw** | **3.84%** | 205,194 |
| **feature_bz** | **2.84%** | 151,722 |
| **feature_bi** | **2.76%** | 147,432 |

## 3. Quy luật "Đi theo nhóm" (Co-missingness Patterns)

Phân tích cho thấy một đặc điểm rất thú vị: các cột rỗng không xuất hiện ngẫu nhiên mà thường **trống cùng nhau** với con số chính xác tuyệt đối. Điều này khẳng định chúng được sinh ra từ cùng một hệ thống tính toán hoặc cùng một cửa số lưu dữ liệu (Lookback window).

### Các cụm điển hình:
- **Cụm 1 (k, i):** `feature_k` và `feature_i` có cùng 59,025 dòng trống.
- **Cụm 2 (h, j):** `feature_h` và `feature_j` có cùng 58,465 dòng trống.
- **Cụm 3 (ad, ae, af, ag, aj, ak):** Các cột này thường có cùng mốc trống quanh ngưỡng 8k-9k dòng.
- **Cụm 4 (x, y, z, w):** Bộ tứ này luôn đi cùng nhau với chính xác 6,184 dòng trống.

## 4. Đặc tính của các cột theo Tên (Naming Characteristics)

1.  **Cụm A (at, au, av, aw, ax, ay):** Có tỷ lệ trống rất đa dạng (từ 0.7% đến 12%). Đây có thể là các chỉ số kỹ thuật (Indicators) có kỳ hạn (lag/window) khác nhau (như SMA 5, SMA 20, SMA 200).
2.  **Cụm B (bx, by, bz, bi):** Tỷ lệ rỗng cao thứ hai. Có thể là dữ liệu cơ bản về công ty (Fundamental) hoặc dữ liệu từ một sàn giao dịch phụ.
3.  **Cụm C (cc, cd, ce, cf, cg):** Có xu hướng trống cùng nhau, liên quan đến các đặc trưng biến động (Volatility) hoặc rủi ro.

## 5. Phân tích Vị trí và Phân phối (Grouped Analysis)

Khi phân tích NaNs theo từng chuỗi thời gian cụ thể (Grouped by `code`, `sub_code`, `horizon`), chúng ta thấy các quy luật phân phối dòng trống rất đặc thù:

| Feature | Trống ở Đầu (Start) | Trống ở Giữa (Middle) | Trống ở Cuối (End) | All-NaN Groups |
| :--- | :--- | :--- | :--- | :--- |
| **feature_at** | **31.4** dòng | **56.2** dòng | 1.6 dòng | 63 |
| **feature_by** | **28.6** dòng | **49.7** dòng | 0.9 dòng | 31 |
| **feature_al** | **22.6** dòng | 7.5 dòng | 0.2 dòng | 18 |
| **feature_k** | 4.2 dòng | 0.2 dòng | 2.2 dòng | 20 |
| **feature_ce** | 2.3 dòng | 3.1 dòng | 3.1 dòng | 230 |

### Quan sát quan trọng:
1.  **Đặc trưng "Lookback" (Trống ở đầu):** Các cột như `feature_at`, `feature_by`, `feature_al` có lượng trống ở đầu rất lớn (20-30 dòng). Đây chính là dấu hiệu của các chỉ số kỹ thuật cần một "cửa sổ lịch sử" (VD: SMA 30 ngày) mới có thể tính toán được giá trị đầu tiên.
2.  **Lỗ hổng dữ liệu (Middle Gaps):** Một số cột (`feature_at`, `feature_by`, `feature_ay`) có lượng rỗng ở giữa chuỗi rất lớn (hơn 40-50 dòng). Điều này cho thấy dữ liệu bị gián đoạn (Gaps) do lỗi hệ thống hoặc do tính chất của cột đó không được cập nhật liên tục.
3.  **Nhóm "Mất tích" (All-NaN):** `feature_ce` và `feature_cf` bị trống hoàn toàn trong 230 nhóm khác nhau. Nghĩa là với một số mã cổ phiếu hoặc kỳ hạn nhất định, hai chỉ số này tuyệt đối không tồn tại.

## 6. Kết luận & Đề xuất hành động (Recommendation)

### Tại sao dữ liệu rỗng xuất hiện?
- **Thời gian khởi động (Cold Start):** Mã mới lên sàn nên chưa có đủ dữ liệu quá khứ cho các công thức tính toán phức tạp.
- **Dữ liệu không đồng bộ:** Một số sàn không cung cấp các chỉ số này 24/7.
- **Lỗi hệ thống thu thập:** Rỗng hàng loạt theo nhóm (Cụm 1, 2, 3...) cho thấy lỗi từ nguồn dữ liệu cụ thể.

### Chiến thuật xử lý:
- **KHÔNG NÊN xóa bỏ (`dropna`):** Việc xóa bỏ các dòng này làm mất đi 30% mẫu dữ liệu (như trong Benchmark của bạn đã xóa 1.5 triệu dòng).
- **LGBM Handling:** LightGBM có cơ chế tự động xử lý giá trị rỗng bằng cách đưa chúng vào một nhánh riêng biệt trong cây quyết định. Đây là cách làm thông minh nhất, vừa giữ được số lượng dòng tối đa, vừa không làm sai lệch quy luật của các dòng có dữ liệu.

---
*Báo cáo được thực hiện bởi Antigravity AI Assistant.*
