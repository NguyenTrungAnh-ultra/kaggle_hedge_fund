# Chiến lược Xử lý Dữ liệu Phân loại (Categorical Data)
**Dự án: Hege Fund Kaggle - Time-series Forecasting**

Trong tập dữ liệu này, các cột `code`, `sub_code`, `sub_category`, và `horizon` đóng vai trò là "bối cảnh" (context). Việc xử lý đúng các cột này quyết định liệu mô hình có thể phân biệt được các thực thể khác nhau hay không. Dưới đây là 4 chiến lược quan trọng:

---

## 1. Chiến lược Mã hóa (Encoding)
Mục tiêu là chuyển đổi các nhãn chữ thành số để mô hình có thể tính toán được.

### Lựa chọn kỹ thuật:
*   **`pd.factorize()` hoặc `.cat.codes`**: Đây là cách nhanh nhất cho dữ liệu lớn (5.3M dòng). Nó chuyển `sub_code` (180 loại) thành các số từ 0-179.
*   **Target Encoding**: Thay thế mã bằng giá trị trung bình của `y_target` của nhóm đó trong quá khứ.
    *   *Ví dụ:* Nếu `sub_code` 'AAPL' luôn có mức sinh lời trung bình cao hơn 'MSFT', con số này sẽ giúp mô hình nhận diện thực thể tốt hơn.

> [!CAUTION]
> **Tránh One-Hot Encoding** cho `sub_code`. Việc tạo ra 180 cột mới sẽ làm dữ liệu cực kỳ thưa thớt (sparse), gây lãng phí bộ nhớ và làm mô hình Gradient Boosting chạy chậm đi đáng kể.

---

## 2. Chiến lược Quản lý Tầm nhìn (Horizon Management)
Cột `horizon` (1, 3, 10, 25) đại diện cho 4 bài toán dự báo khác nhau.

### Hai hướng tiếp cận chính:
1.  **Mô hình Toàn cục (Global Model)**: 
    *   Dùng 1 mô hình duy nhất cho cả 4 horizon.
    *   **Ưu điểm:** Tận dụng được toàn bộ 5.3M dòng dữ liệu để học các pattern chung.
    *   **Nhược điểm:** Mô hình khó học được sự khác biệt tinh tế giữa dự báo ngắn hạn (nhiễu nhiều) và dài hạn (xu hướng rõ hơn).
2.  **Mô hình Phân mảnh (Per-Horizon Models)**:
    *   Huấn luyện 4 mô hình độc lập cho mỗi mức horizon.
    *   **Ưu điểm:** Độ chính xác thường cao hơn vì mỗi mô hình được tối ưu cho một mục tiêu cụ thể.
    *   **Nhược điểm:** Tốn thời gian huấn luyện và bảo trì hơn.

---

## 3. Chiến lược Đặc trưng Nhóm (Group-based Features)
Thay vì chỉ dùng các tính năng ẩn danh (`feature_a`...`ch`), hãy tận dụng các cột phân loại để tạo ra các đặc trưng thông minh.

### Các ý tưởng chính:
*   **Group Statistics**: Tính giá trị trung bình, độ lệch chuẩn, giá trị max/min của các feature khác theo từng `code` hoặc `sub_category`.
    *   *Mục đích:* "Bình thường hóa" dữ liệu. (Ví dụ: Một giá trị feature là 0.5 có thể là cao đối với `code` A nhưng lại là thấp đối với `code` B).
*   **Entity Embedding**: Nếu dùng Neural Networks, hãy sử dụng lớp `Embedding` cho `sub_code`. Điều này giúp mô hình tự "vẽ bản đồ" mối quan hệ giữa các tài sản trong không gian đa chiều.

---

## 4. Chiến lược Đặc trưng Thời gian (Lag & Temporal Features)
Sử dụng `ts_index` kết hợp với các định danh để tạo ra "trí nhớ" cho mô hình.

### Cách triển khai:
*   **Lag Features (Biến trễ)**: Lấy giá trị của `y_target` hoặc các feature quan trọng ở bước thời gian trước đó (t-1, t-2).
    *   *Lưu ý:* Phải `groupby(['code', 'sub_code', 'horizon'])` trước khi dùng `.shift()` để chắc chắn bạn không lấy nhầm dữ liệu của tài sản khác.
*   **Rolling Window**: Tính trung bình trượt (Moving Average) của các tính năng trong 5 hoặc 10 bước thời gian gần nhất cho mỗi thực thể.

---

## 💡 Lời khuyên cuối cùng
Trong các cuộc thi tài chính, **tính ổn định quan trọng hơn độ phức tạp**. Hãy bắt đầu với **Chiến lược 1 (Factorize)** và **Chiến lược 2 (Global Model)** để có kết quả cơ sở (Baseline), sau đó mới thử nghiệm các đặc trưng nhóm và biến trễ phức tạp hơn.
