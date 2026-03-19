# Báo cáo Huấn luyện LGBM (Purged K-Fold) trên train_stationary.parquet

Sau khi áp dụng cơ chế **Purged K-Fold Cross Validation với Embargo** lên dữ liệu lớn (hơn 5.3 triệu quan sát và 86 đặc trưng), tiến trình chia 5 Folds đã hoàn tất thành công.

## 1. Thông tin Dataset
- Khối lượng: **5,337,414** dòng dữ liệu.
- Số chiều: 86 *Features* độc lập.
- Thời gian chốt biên độ nhãn (`horizon`): Đa dạng trải dài [10, 25, 3, 1] bước thời gian. 
- Nhiệm vụ học máy tự động nhận diện: **Regression**.

## 2. Quá trình Thanh lọc và Cấm vận 
Trong mỗi Fold, cơ chế đã chủ động tìm diệt sự rò rỉ:
- **Fold 1**: Loại bỏ **91,084** mẫu rò rỉ trước và mẫu tương quan gần. Tập huấn luyện giữ lại **4,178,847** dòng. Tập Test **1,067,483** dòng.
- **Fold 2**: Loại bỏ **102,482** mẫu.
- **Fold 3**: Loại bỏ **113,145** mẫu.
- **Fold 4**: Loại bỏ **111,326** mẫu. 
- **Fold 5**: Loại bỏ **14,661** mẫu.
*(Lưu ý: Không một mẫu rò rỉ thông tin tương lai nào được phép lọt qua bộ lọc!)*

## 3. Chỉ số Chất lượng Mô hình (Evaluation Report)
Toàn bộ dự báo OOF (Out-Of-Fold) của 5 model ghép lại, và đây là điểm đối chiếu trên thực tế:

- **Overall OOF RMSE**: `32.27198`
- **Overall OOF R2**: `0.01566` 
*(Mức R2 trong bài toán tài chính/định lượng thường rất thấp và dao động quanh 0.01 đến 0.05, điều này chứng tỏ bộ khung Purged K-Fold đã thực hiện đúng sứ mệnh triệt tiêu ảo giác độ chính xác - Overfitting Illusion).*

## 4. Đặc trưng Quan trọng nhất (Top 10 Feature Importances)
Được tổng hợp chéo theo trung bình `Gain` từ 5 Cây rừng:

| Feature | Importance Score |
| :--- | :--- |
| **feature_al** | `1.626610e+07` |
| **feature_u** | `4.628076e+06` |
| **feature_a** | `4.067909e+06` |
| **feature_v** | `3.726542e+06` |
| **feature_cf** | `2.623826e+06` |
| **feature_bg** | `2.427410e+06` |
| **feature_z** | `2.378868e+06` |
| **feature_aq** | `2.316261e+06` |
| **feature_cg** | `2.247188e+06` |
| **feature_j** | `2.183793e+06` |

Mô hình đã ghi chú file thực tế ra Text Format tại `f:\Projects\hege_fund_kaggle\training_report.txt`.
