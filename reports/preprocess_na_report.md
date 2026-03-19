# Báo Cáo Chi Tiết: Tiền xử lý Vắng khuyết (NaN Imputation)

## 1. Mục tiêu và Nguyên tắc Cốt lõi
- **Mục tiêu**: Làm sạch toàn bộ các giá trị `NaN` trong tập dữ liệu chuỗi thời gian tài chính (`train.parquet`) một cách chuẩn xác, chuẩn bị cho mô hình LightGBM.
- **Nguyên tắc Tối thượng**: Chống rò rỉ dữ liệu (Data Leakage) tuyệt đối. Không được phép để bất kỳ thông tin nào từ tương lai (t+n) rò rỉ về quá khứ (t) trong quá trình nội suy số liệu.

## 2. Đường ống 4 Lớp Xử lý (4-Layer Pipeline)

### Lớp 1: Cách ly Không gian (Spatial Isolation)
- **Hành động**: Tất cả các thao tác thống kê và tập hợp (`groupby`) trên toàn bộ hệ thống đều bắt buộc phải lấy `horizon` làm vách ngăn.
- **Ý nghĩa**: Đảm bảo rằng dữ liệu dự báo cho một chân trời (ví dụ 1 Day) sẽ không bao giờ được dùng để nội suy chéo cho dữ liệu dự báo ở chân trời khác (ví dụ 5 Days), bảo vệ tuyệt đối phân phối thống kê nguyên bản của hệ thống.

### Lớp 2: Đánh dấu Khuyết thiếu (Missingness Indicator / Shadow Matrix)
- **Hành động**: Thống kê tại thời điểm T0 phát hiện 48/86 features vướng `NaN`. Hệ thống lập tức tạo thêm 48 biến cờ định danh nhị phân mang tiền tố `is_nan_` (nhận giá trị 1 nếu ô ban đầu bị rỗng, nhận 0 nếu có số liệu hợp lệ).
- **Ý nghĩa**: Đưa sự kiện "mất kết nối dữ liệu" trở thành một tín hiệu độc lập (Feature Engineering). Thuật toán Cây Quyết Định (LightGBM) sẽ có cơ sở phân nhánh học được mẫu hình đứt gãy từ các cột cờ này, thay vì chỉ là bù đắp con số vô tri.

### Lớp 3: Vá lỗi Trục dọc với Forward-Fill (Longitudinal FFill)
- **Hành động**: Nhóm độc lập dữ liệu theo `['code', 'horizon']` $\rightarrow$ Sắp xếp theo dòng thời gian `ts_index` $\rightarrow$ Sử dụng thuật toán điền tiến `ffill(limit=50)`.
- **Ý nghĩa**: Khắc phục các khoảng đứt gãy ở giữa chuỗi thời gian (Middle Gaps). Con số `limit=50` ngăn chặn việc mang số liệu từ quá khứ kéo căng vượt quá giới hạn. Phương thức dồn lên trước (forward-fill) là thiết lập duy nhất trên trục dọc mà **không** gây rò rỉ dữ liệu tương lai.

### Lớp 4: Lấp khoảng trống Đầu chuỗi (Cold-Start Median Fill)
- **Hành động**: Đối với những mã cổ phiếu bị khuyết dữ liệu ngay từ những `ts_index` đầu tiên (Cold Start - khiến Lớp 3 không thể ffill từ trước tới), hệ thống tự bẻ ngang chiều kích: Nhóm theo `['ts_index', 'horizon']` và lấy giá trị trung vị (`Median`) của **tất cả các mã cổ phiếu khác đang điểm danh tại đúng thời khắc đó** để nội suy ngang vào chỗ trống.
- **Cơ chế Chống Rò Rỉ (Anti-Leakage Fallback)**: Đứng trước kịch bản cực đoan "toàn bộ một mặt cắt thời gian bị rỗng 100%", hệ thống thiết lập luật bảo vệ dữ liệu cực chuẩn. Nếu xảy ra, hệ thống sẽ chối từ việc dùng nội suy tương lai, nhả cảnh báo lỗi log ra màn hình và trả nguyên trị số `NaN`. May mắn thay, phân tích thực địa trên file `train.parquet` cho thấy độ lấp đầy đủ tốt: Không có bất kỳ mặt cắt nào bị rỗng 100%, do đó Layer 4 đã lấp kín 100% các NaNs còn thưa thớt một cách hoàn hảo và hợp pháp.

## 3. Thống kê Kết quả Cuối cùng
- **Khối lượng dữ liệu quét**: 5,337,414 dòng.
- **Features bị thiếu số liệu ban đầu**: 48 cột gốc.
- **Số lượng biến Shadow Indicators tạo sinh**: 48 cột `is_nan_*`.
- **Số lượng NaN Tồn đọng trong Features**: **0** (Sạch bóng).
- **Kích thước Dữ liệu Đầu ra**: 142 cột (Đã định dạng nguyên bản hình dáng tệp + 48 features mới gia tăng lợi thế phân loại).
- **File Output Đích**: `f:\Projects\hege_fund_kaggle\ts-forecasting\train_handled_na.parquet`
- **Thời gian Execute Runtime**: Nhanh, xấp xỉ ~29.9s (Tuỳ IO Ổ cứng).
