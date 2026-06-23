# Nhật ký Thực nghiệm và So sánh Chi tiết (EXP.md)

Tài liệu này ghi nhận so sánh chi tiết từng bước (Step-by-step) giữa mô hình hiện tại của chúng ta (**LB 0.1885**) và giải pháp gốc của tác giả đạt vị trí thứ 5 dựa trên tệp `multiple-lightgbm-approach.ipynb` (đây là tệp giải pháp chính thức mới đúng).

---

## 📌 1. Bảng So sánh Toàn diện từng bước (Chúng ta vs. Tác giả)

| Giai đoạn | Mô hình của chúng ta (LB: 0.1885) | Giải pháp gốc của tác giả (Tệp chuẩn) | Điểm khác biệt cốt lõi |
| :--- | :--- | :--- | :--- |
| **1. Tiền xử lý dữ liệu thô** | • Áp dụng `ffill` và `median`.<br>• Dùng Cross-Sectional Z-Score cho `feature_bd`. | • **KHÔNG làm gì cả**. Dữ liệu được ném thẳng vào để LightGBM tự xử lý `NaN` một cách tự nhiên. Tác giả có ghi chú về chuẩn hóa chéo trong văn bản nhưng trong code lại không hề có. | 🔴 Việc điền khuyết của ta làm mất tín hiệu "bị thiếu" gốc. Tác giả dựa hoàn toàn vào khả năng xử lý NaN của LGBM. |
| **2. Kỹ thuật Đặc trưng (FE)** | • Sử dụng **Causal Lags** cho 4 biến.<br>• Thêm `log_horizon`. | • Tạo **Features cực kỳ sắc bén**: Chỉ lấy 5 đặc trưng quan trọng nhất (`feature_al`, `feature_am`, `feature_cg`, `feature_by`, `feature_s`) để tạo lags, diff, rolling mean/std, EWM.<br>• Thêm đặc trưng **Tương tác**: Tính hiệu số (`al - am`, `cg - by`) và tỷ lệ (`al / am`).<br>• Thêm đặc trưng **Chu kỳ thời gian**: Dùng hàm lượng giác `sin(2*pi*ts_index/100)`. | 🔴 **Đặc trưng của tác giả rất tinh gọn nhưng có tính tương tác cao**, tập trung vào nhóm các đặc trưng lõi. Dùng cả EWM và chu kỳ thời gian để bắt sóng thị trường. |
| **3. Target Encoding (TE)** | • TE được tính toán cẩn thận trong CV để tránh leak. | • Tính TE trung bình trên tập `ts_index <= 3500`. Tập hold-out và tập Test lấy bản đồ này để ánh xạ (map) sang. | 🤝 Cả hai đều **không bị rò rỉ dữ liệu**. Tác giả làm rất cẩn thận ở bước này. |
| **4. Đánh giá chéo (CV)** | • **5-Fold Purged K-Fold CV**. Đánh giá toàn diện. | • Tách dữ liệu tĩnh (Hold-out): Huấn luyện trên `ts_index <= 3500`, Valid trên khúc đuôi `ts_index > 3500`. | 🟡 Của ta dùng K-Fold phức tạp hơn, của tác giả đánh giá đơn giản trên khúc đuôi. |
| **5. Cơ chế Huấn luyện & Dự báo** | • Huấn luyện 20 mô hình trên **100% dữ liệu Train**. | • Chỉ dùng tập **`ts_index <= 3500` để huấn luyện**, dừng sớm (early stopping) dựa trên tập `> 3500`. Sau đó dự đoán thẳng lên tập Test bằng 5 seeds. | 🔴 **Khác biệt LỚN**: Tác giả KHÔNG học trên đoạn dữ liệu từ 3501 trở đi! Khúc đuôi chỉ dùng để kìm hãm mô hình (chống Overfit). |

---

## 🔍 2. Bài học rút ra từ Giải pháp thực sự

Sau khi phân tích đúng tệp giải pháp hạng 5 (`multiple-lightgbm-approach.ipynb`), tôi nhận ra phương pháp của họ tuy đơn giản nhưng rất sắc sảo:

1. **Tin tưởng hoàn toàn vào LightGBM để xử lý NaN:** Họ không dùng ffill hay median. Các cây quyết định của LGBM sẽ tự động coi "thiếu dữ liệu" là một nhánh phân loại đặc biệt (missing=True).
2. **Kỹ nghệ đặc trưng rất thông minh:** Họ không làm tràn lan. Họ nhận ra các đặc trưng như `al` và `am` có quan hệ nội tại, nên đã tạo ra hẳn tỷ lệ phân số (`al/am`) và độ chênh lệch (`al-am`). Tương tự với `cg` và `by`. Ngoài ra, họ thêm sóng chu kỳ thời gian (t_cycle) để bắt biến động theo mùa.
3. **Chống Overfit triệt để:** Họ không học dữ liệu cuối cùng (khúc `> 3500`) mà dùng nó làm bộ định chuẩn (Early Stopping) để chốt số lượng cây. Điều này khiến mô hình an toàn trên tập Private Test (vốn dĩ là một đoạn tương lai dài hơn).

**Kết luận:** FE của chúng ta tuy đồ sộ nhưng ít tính tương tác chéo hơn tác giả. Chúng ta huấn luyện full 100% dữ liệu có thể dẫn đến hiện tượng Overfit cục bộ ở đoạn cuối cùng, trong khi tác giả chủ động hi sinh đoạn cuối đó để tăng cường tính khái quát hóa (Generalization). Mặc dù chúng ta không bị leak, nhưng cách họ thiết kế hệ thống thì thông minh hơn hẳn ở khâu chắt lọc đặc trưng và xử lý NaN.
