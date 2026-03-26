# 🕵️‍♂️ Máy dò Biến Tĩnh vs. Biến Động (Static vs. Dynamic Radar)

Công cụ này được thiết kế để phân loại các tính năng (features) trong tập dữ liệu chuỗi thời gian dựa trên tính ổn định của chúng theo thời gian, được nhóm theo thực thể (`code`).

## 🛠 Cơ chế hoạt động
1. **Phân nhóm (Groupby):** Gom dữ liệu theo cột `code`.
2. **Tính toán (Aggregation):** Tính Độ lệch chuẩn (Standard Deviation - `std`) cho mỗi tính năng trong từng nhóm.
3. **Quy tắc phân loại:**
   - **Biến Tĩnh (Static):** Nếu `std == 0` cho TOÀN BỘ các nhóm `code`. Nghĩa là biến này không bao giờ thay đổi giá trị cho một thực thể cụ thể (ví dụ: Phân loại ngành, Quốc gia, Loại tài sản).
   - **Biến Động (Dynamic):** Chỉ cần một `code` có `std > 0`, biến đó được coi là biến động theo thời gian.

## 📁 File Code
Đã được tạo tại: [EDA/static_vs_dynamic_radar.py](file:///f:/Projects/hege_fund_kaggle/EDA/static_vs_dynamic_radar.py)

## 📊 Kết quả phân tích (Sơ bộ)
Sau khi chạy trên `train.parquet`:
- **Nguồn dữ liệu:** `ts-forecasting\train.parquet`
- **Số lượng Biến Tĩnh:** 0
- **Số lượng Biến Động:** 86
  - Danh sách ví dụ: `feature_a, feature_b, feature_c, ... feature_ch`

### Giải mã kết quả:
- Trong tập dữ liệu này, dường như tất cả các biến từ `feature_a` đến `feature_ch` đều có sự thay đổi theo thời gian nội tại của mỗi mã cổ phiếu/tài sản (`code`).
- Điều này cho thấy tập dữ liệu chủ yếu chứa các chỉ số tài chính/kỹ thuật biến động liên tục chứ không có các thuộc tính phân loại nền tĩnh được mã hóa dưới dạng số.

## 🚀 Cách sử dụng lại
Bạn có thể chạy script trực tiếp từ terminal:
```bash
python f:\Projects\hege_fund_kaggle\EDA\static_vs_dynamic_radar.py --group code
```
Hoặc import vào Notebook:
```python
import sys
sys.path.append('f:/Projects/hege_fund_kaggle/EDA')
from static_vs_dynamic_radar import detect_static_dynamic
# ... gọi hàm
```
