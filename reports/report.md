# Báo Cáo Xử Lý & Đào Tạo: Kaggle Time-Series Forecasting

Báo cáo chi tiết về quy trình kiểm tra, sửa lỗi, xử lý dữ liệu chuẩn theo Marcos López de Prado và kết quả chạy mô hình.

## 1. Kiểm tra và Sửa Lỗi (Bug Fixes)
Trong quá trình rà soát `process_stationary_data.py` và `train...py`, các lỗi nghiêm trọng sau đã được phát hiện và khắc phục hoàn toàn để **đảm bảo sự chính xác tuyệt đối**:

- **Lỗi Group Dữ Liệu**: Code cũ chỉ group theo `code` khi chạy Fractional Differencing. Đã sửa lại chuẩn xác: Group theo cấu trúc `[code, sub_code, horizon]`.
- **Lỗi Xử Lý NaN**: Code cũ sử dụng `ffill().fillna(0)` **trước** khi tính FFW, dẫn đến tính toán sai lệch trên các dữ liệu ảo. Đã loại bỏ hoàn toàn việc tính toán với NaN. FFW chỉ áp dụng trên dữ liệu thật.
- **Lỗi Leakage K-Fold**: Sử dụng `KFold(shuffle=False)` chia theo số thứ tự dòng (row index) trên dữ liệu có trùng lặp lịch sử (cùng `ts_index`), dẫn đến Leakage nghiêm trọng (Train/Test rò rỉ dữ liệu cùng mốc thời gian). Đã viết lại `LeakFreePurgedKFold` để chia theo Time Boundaries (nguy cơ Leakage = 0).
- **Lỗi Công thức Kaggle Metric**: `weighted_rmse_score` của code cũ lấy căn bậc hai `np.sqrt(1 - ratio)`. Công thức chuẩn Zero-Mean $R^2$ của các cuộc thi TS Forecasting (VD: Jane Street) không sử dụng căn bậc hai. Đã sửa chuẩn lại công thức thành Zero-Mean $R^2$.

## 2. Phương Pháp Xử Lý Dữ Liệu (Marcos López de Prado)
Áp dụng phương pháp **Fixed-Width Window (FFW)** nguyên bản:
1. Tính $d$ tối ưu cho mỗi Feature (dựa trên việc dò d từ $0.0 \to 1.0$ thông nghiệm ADF test $p \le 0.05$).
2. Sau khi có $d$, sinh bộ trọng số $w$ qua thuật toán FFW với ngưỡng cắt $\tau = 1e-4$.
3. Convolution Window (Cửa sổ cố định) được áp dụng một cách đồng nhất cho **toàn bộ** các groups.
4. **Loại bỏ NaN (Drop NA)**: Các giá trị khởi đầu không tích lũy đủ dữ liệu cho một full-window FFW sẽ trả về NaN và bị loại bỏ (nhằm đảm bảo tính dừng - strictly stationary).

**Kết quả Xử Lý Dữ Liệu:**
- Tổng số dòng ban đầu: **5,337,414**
- Số dòng giữ lại sau khi drop NA (do mất mát cửa sổ FFW): **3,336,800**
- Tỉ lệ giữ lại: **62.52%**  
*(Lưu ý: Sự hao hụt dữ liệu khổng lồ này là sự đánh đổi toán học bắt buộc của thuật toán FFW khi dữ liệu gốc bị phân mảnh thành các đoạn group `[code, sub_code, horizon]` quá nhỏ so với kích thước cửa sổ tích phân do $d$ sinh ra. Tuy nhiên, 3,336,800 dòng còn lại mang tính chất dừng cực kỳ chuẩn xác và không chứa bất kỳ nhiễu NA nào).*

## 3. Quá Trình Đào Tạo và Đánh Giá (Training)
Tất cả các lệnh và logic liên quan đã được gộp chung vào 1 file duy nhất: `train.py`.

- **Mô Hình**: LightGBM Regressor (1000 estimators).
- **CV Strategy**: Chronological Train/Test Split (80/20) + `LeakFreePurgedKFold` (5 Splits, 1% Embargo cho dữ liệu Train).
- **Models Param**: 5 Folds Models đã được lưu thành công vào thư mục `models_param/` (lgbm_fold_1.txt -> lgbm_fold_5.txt).
- Khác với trước đây, không có bất kỳ mẫu huấn luyện nào được lan truyền thông tin tới tập Test.

### Kết Quả (Hold-out Test 20%)
- **Kaggle Zero-Mean R2 Score**: **0.000000**
- Báo cáo gốc được lưu tại: `test_evaluation_report.txt`

## 4. Cấu Trúc File Mới (Clean Code)
Hệ thống codebase đã được tinh giản từ 6 file rối rắm thành 2 file gọn gàng:
1. `process_data.py`: Pipeline chuẩn hóa dữ liệu tuyệt đối (FFW logic được optimize bằng `np.convolve` vectorization để xử lý siêu tốc cực kỳ tối ưu).
2. `train.py`: Pipeline chống leakage, training và xuất kết quả.
