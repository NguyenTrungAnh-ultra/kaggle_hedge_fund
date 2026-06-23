# Nhật ký Thực nghiệm và Phân tích Giải pháp (EXP.md)

Tài liệu này ghi nhận quá trình phân tích và so sánh chi tiết giữa mô hình hiện tại của chúng ta (**LB 0.1885**) với hai biến thể giải pháp Top 5 được tìm thấy trong thư mục:
1. **Giải pháp Top 5 "Fake"** (`lgb-causal-insights-0-1993-0-2048-5th.ipynb`): Chứa các kỹ thuật thú vị như Recency Weighting, Target Clipping nhưng bị rò rỉ dữ liệu (Data Leakage) nghiêm trọng.
2. **Giải pháp Top 5 "Real"** (`multiple-lightgbm-approach.ipynb`): Giải pháp gốc thực sự của tác giả hạng 5, một người hoàn toàn khác, với cách tiếp cận tối giản, không điền khuyết NaN và không bị rò rỉ.

---

## 📌 Phần 1: So sánh Chúng ta vs. Top 5 "Fake" (lgb-causal-insights...)

Đây là giải pháp ban đầu được nhầm tưởng là giải pháp thực sự. Dù vậy, nó chứa một số kỹ thuật (cả hữu ích lẫn có hại) đáng để mổ xẻ.

| Giai đoạn | Mô hình của chúng ta (LB: 0.1885) | Giải pháp "Fake" Top 5 | Đánh giá / Khác biệt |
| :--- | :--- | :--- | :--- |
| **Tiền xử lý (NaN)** | Điền khuyết bằng `ffill` và `median`. | Điền khuyết bằng `ffill` và Cross-Sectional Mean. | Cả hai đều điền khuyết, nhưng ta dùng Median an toàn hơn trước ngoại lai. |
| **Target Encoding** | Ánh xạ TE chuẩn xác bên trong từng Fold của Cross-Validation. | Ánh xạ TE sử dụng giá trị trung bình toàn cục (kết hợp cả Train và Test). | 🔴 **Leakage Nghiêm Trọng**: Giải pháp "fake" này bị rò rỉ dữ liệu từ tập Test ngược về tập Train thông qua Target Encoding toàn cục. Đây là lý do điểm CV của họ bị ảo. |
| **Trọng số Thời gian** | Không dùng. (Chỉ dùng trọng số gốc `weight`). | Dùng **Recency Weighting**: Giảm dần trọng số các mẫu cũ theo quy luật phân rã hàm mũ (Half-life = 0.5). | 🤝 Kỹ thuật này rất hay để ép mô hình tập trung vào xu hướng gần đây nhất. Đáng để chúng ta học hỏi. |
| **Bảo vệ Ngoại lai** | Dùng điều chuẩn L1/L2 mạnh để phanh ngoại lai. | Dùng **Target Clipping**: Cắt cụt target (y_target) ở phân vị 5% và 95% theo từng horizon. | 🤝 Cắt cụt target trước khi huấn luyện giúp giảm độ nhạy của LGBM với nhiễu thị trường. |

---

## 📌 Phần 2: Phân tích Kỹ thuật của Top 5 "Real" (multiple-lightgbm-approach.ipynb)

Sau khi kiểm tra tệp giải pháp thực sự (`multiple-lightgbm-approach.ipynb`) đến từ một tác giả khác hoàn toàn, chúng ta thấy một tư duy thiết kế trái ngược với bản "Fake" và hoàn toàn sạch sẽ (Không bị rò rỉ dữ liệu). 

### Các Kỹ Thuật Đặc Biệt Tác Giả Sử Dụng:
1. **Hoàn toàn KHÔNG điền khuyết (Zero Imputation):**
   Tác giả để nguyên `NaN`. Dữ liệu được ném thẳng vào để LightGBM tự phân nhánh trên các giá trị thiếu. Tác giả ngầm coi "sự vắng mặt của dữ liệu" (missingness) là một tín hiệu giao dịch quan trọng.

2. **Kỹ nghệ Đặc trưng (Feature Engineering) "Sắc như dao cạo":**
   - **Tập trung hóa:** Tác giả không làm lag tràn lan. Họ chỉ dùng đúng 5 đặc trưng quan trọng nhất (`feature_al`, `feature_am`, `feature_cg`, `feature_by`, `feature_s`) để làm phái sinh (Lags, Diff, Rolling Mean/Std, Exponential Weighted Moving Average - EWM).
   - **Tương tác Chéo (Interactions):** Tạo ra tỷ lệ phân số (`al / am`) và hiệu số (`al - am`, `cg - by`). Điều này bắt được sự bất đồng điệu giữa các nhóm tài sản.
   - **Sóng Chu kỳ Lượng giác:** Dùng hàm `sin(2*pi*ts_index/100)` để bắt chu kỳ thời gian (Seasonality) của thị trường.

3. **Chống Overfit Cực Đoan (Tách Hold-out Tĩnh):**
   Thay vì dùng K-Fold phức tạp hay huấn luyện 100% dữ liệu như chúng ta, tác giả cắt đúng một nhát: `ts_index <= 3500` làm tập Huấn luyện và `> 3500` làm tập Xác thực (Early Stopping). Khi dự đoán trên Test Set, mô hình hoàn toàn **chưa từng được học** dữ liệu từ 3501 trở đi. Họ chủ động hi sinh đoạn cuối để đạt tính Generalization tối đa trên Private LB (Đề phòng thị trường tương lai biến đổi lớn).

4. **Target Encoding An Toàn (Leak-Free):**
   Giống với pipeline chuẩn của chúng ta, tác giả tính Target Encoding cực kỳ cẩn thận chỉ trên tập Huấn luyện (`<= 3500`) sau đó map giá trị đó sang Test. Hoàn toàn không bị Leakage.

### 🔍 Kết luận: 
Trong khi giải pháp "Fake" có Target Clipping và Recency Weighting để chống nhiễu (nhưng bị dính Data Leakage), thì giải pháp "Real" lại dựa vào **Tương tác đặc trưng tinh gọn**, **Zero Imputation** và cơ chế **Hold-out tĩnh** để giành chiến thắng một cách đường hoàng. Chúng ta đang đứng ở vị thế rất tốt vì base model của ta sạch 100% như bản "Real", và ta hoàn toàn có thể thử nghiệm nghiệm thu các kỹ thuật Feature Engineering xuất sắc của họ để đẩy điểm LB 0.1885 lên cao hơn nữa.
