# Comprehensive Analysis Report: Fold 4 Performance Discrepancy

## Executive Summary

The performance drop in Fold 4 (R2 ~0.0036 vs Fold 1 R2 ~0.0128) is primarily driven by a **Regime Change** in the financial time series. Key predictive signals have either evaporated or drifted significantly, while the target distribution has become less "fat-tailed," making it harder for the model to capture extreme returns.

---

## 1. Feature Signal Decay (Correlation Collapse)

The most critical finding is the loss of predictive power in the model's top features.

| Feature        | Fold 1 Corr | Fold 4 Corr | Change  | Impact                                     |
| :------------- | :---------- | :---------- | :------ | :----------------------------------------- |
| `sub_category` | 0.0102      | 0.0008      | -92%    | **CRITICAL**: Primary signal lost.         |
| `feature_k`    | -0.1365     | -0.0003     | -99%    | **HIGH**: Massive feature signal vanished. |
| `feature_aq`   | -0.0375     | 0.0051      | Flipped | **HIGH**: Directional change.              |
| `feature_h`    | -0.0654     | 0.0032      | Flipped | Signal turned to noise.                    |

---

## 2. Data Drift Analysis (KS Test)

The Kolmogorov-Smirnov (KS) statistic measures the difference between probability distributions. Values > 0.25 indicate severe drift.Kiến thức Tiên quyết (Prerequisite Knowledge)

Để phân tích Sức mạnh Tương đối, chúng ta cần hiểu rõ bối cảnh không gian của dữ liệu tài chính. Dữ liệu trong bài toán này là Dữ liệu Mảng (Panel Data), nghĩa là nó tồn tại trên hai trục tọa độ đồng thời:
Trục Dọc - Chuỗi thời gian (Time-series): Theo dõi một tài sản duy nhất (ví dụ: sub_code 05) biến động như thế nào qua các ngày (ts_index 1, 2, 3...).
Trục Ngang - Lát cắt ngang (Cross-section): Chụp một bức ảnh của toàn bộ thị trường tại một thời điểm duy nhất (ví dụ: tại ts_index 100, đo lường toàn bộ 180 sub_code cùng lúc).
"Sức mạnh Tương đối" là việc chuyển dịch tư duy phân tích từ Trục Dọc sang Trục Ngang.
Giải phẫu Thông tin (Deep Anatomy): Sức mạnh Tương đối (Cross-sectional Power)
Việc các tác giả trong tập dữ liệu chuyển hướng sang sử dụng Điểm Z chéo (Cross-sectional Z-scores) và Xếp hạng (Rank Features) là một bước tiến quan trọng về mặt bản chất định lượng.

1. Sự kiện Thực tế (Facts)
   Bằng chứng từ mã nguồn: Các mô hình đạt điểm cao (baseline từ 0.2598 trở lên) không đưa trực tiếp giá trị tuyệt đối của biến feature*X vào mô hình. Thay vào đó, tại mỗi mốc thời gian $t$, họ gom toàn bộ các công cụ tài chính lại, sau đó tính hạng (Rank) hoặc chuẩn hóa Z-score.
   Cơ chế Toán học: Z-score chéo được tính bằng công thức $Z*{i,t} = \frac{X*{i,t} - \mu_t}{\sigma_t}$, trong đó $X*{i,t}$ là giá trị của mã $i$ tại thời gian $t$; $\mu_t$ và $\sigma_t$ lần lượt là giá trị trung bình và độ lệch chuẩn của toàn bộ thị trường tại đúng thời điểm $t$ đó.
2. Các Giả định (Assumptions) và Thách thức
   Giả định 1: Thị trường là Trò chơi Tổng bằng không (Zero-sum Game) về mặt dòng tiền thông minh (Smart Money).
   Việc sử dụng các đặc trưng chéo ngầm định rằng: Điều quan trọng không phải là mã của bạn tăng hay giảm bao nhiêu phần trăm, mà là nó đang hút tiền vào hay bị rút tiền ra so với phần còn lại. Lợi nhuận vượt trội (Alpha) chỉ sinh ra từ sự chênh lệch (Spread).
   Hãy thách thức nó: Giả định này sẽ thất bại nặng nề trong các đợt biến động vĩ mô cực đoan (Macro Shocks). Khi toàn bộ thị trường sụp đổ (Market Crash), một tài sản có "Sức mạnh tương đối" cao nhất (Rank 1) thực chất chỉ là "tài sản rớt giá ít nhất". Nó vẫn gây lỗ hụt vốn tuyệt đối (Absolute Drawdown) cho quỹ.
   Giả định 2: Tính Đồng nhất của Rổ Tài sản (Universe Homogeneity).
   Khi bạn xếp hạng chéo toàn bộ 180 sub_code với nhau, bạn đang giả định chúng chạy trên cùng một sân chơi và chịu chung một mức độ rủi ro hệ thống.
   Hãy thách thức nó: Đây là rủi ro so sánh "táo và cam" (Apples-to-oranges). Nếu một mã có bản chất là cổ phiếu công nghệ biến động cực mạnh, còn mã kia là trái phiếu chính phủ cực kỳ tĩnh, việc ném chúng vào cùng một rổ để chuẩn hóa Z-score sẽ làm sai lệch hoàn toàn ý nghĩa của phương sai ($\sigma_t$).
   Điểm nghẽn Nhận thức (Cognitive Bottlenecks / Threshold Concepts)
   Điểm ngưỡng để hiểu sâu chủ đề này nằm ở khái niệm Sự Khử Nhiễu Vĩ mô (Macro-Neutralization).
   Khi bạn nhìn vào đồ thị của một tài sản và thấy nó tăng vọt, theo bản năng, bạn nghĩ rằng tài sản đó đang có tín hiệu tốt (tư duy Time-series). Tuy nhiên, nếu ngày hôm đó Cục Dự trữ Liên bang (FED) hạ lãi suất, làm toàn bộ 180 tài sản đều tăng vọt, thì sự tăng giá của tài sản bạn đang nhìn thực chất là vô nghĩa (nhiễu).
   Bằng cách đưa dữ liệu qua bộ lọc Cross-sectional (trừ đi trung bình $\mu_t$ của toàn thị trường), bạn đã "rửa sạch" hoàn toàn tác động của dòng nước vĩ mô dâng lên, để lộ ra giá trị nội tại thực sự của từng con thuyền. Đây là lý do tại sao các mô hình không có Cross-sectional Features luôn bị sụp đổ khi thị trường chuyển pha (Regime Shift).
   Lỗ hổng Logic (Logical Fallacies) & Thiên kiến Nhận thức (Cognitive Biases)
   Ảo giác Giá trị Tuyệt đối (Absolute Value Illusion):
   Thiên kiến này xảy ra khi nhà phân tích bị ám ảnh bởi một ngưỡng cố định. Ví dụ: "Động lượng (Momentum) đạt mức 10 là tín hiệu mua cực mạnh". Trong không gian Cross-sectional, điều này là ngụy biện. Nếu trung bình động lượng của toàn thị trường ngày hôm đó đang là 20, thì tài sản có mức động lượng 10 thực chất đang cực kỳ suy yếu (Underperforming) và có nguy cơ bị bán khống (Short-selling).
   Lỗ hổng Chuẩn hóa Mù (Blind Normalization Fallacy):
   Như đã phân tích ở phần giả định, lỗi logic phổ biến nhất khi viết code là tính lệnh .groupby('ts_index').transform('rank') trên toàn bộ tập dữ liệu mà không quan tâm đến cột sub_category. Một quy trình đúng đắn đòi hỏi phải thực hiện Khử rủi ro theo nhóm (Sector Neutralization): Phải so sánh sức mạnh của một tài sản với các tài sản cùng loại (cùng sub_category) trước khi so sánh với toàn thị trường.
   Bạn có muốn tôi bóc tách chi tiết hơn về cách triển khai mã code cho kỹ thuật Khử rủi ro theo nhóm (Sector Neutralization) để khắc phục Lỗ hổng Chuẩn hóa Mù này không?

- **`feature_cf` (KS = 0.3639)**: Most drifted feature. The model trained on F1-F3 logic is likely applying the wrong "bins" to this feature in F4.
- **`feature_z` (KS = 0.3355)**: Significant distributional shift.
- **`feature_k` (KS = 0.3214)**: Confirms that its correlation collapse is tied to a fundamental change in the feature's behavior.

---

## 3. Target Characteristics Shift

The nature of the target variable `y_target` has changed over time.

- **Kurtosis Drop**: Fold 1 (383) $\rightarrow$ Fold 4 (174).
  - The market became less "fat-tailed" in the F4 period (fewer extreme outliers proportional to the mean).
  - LightGBM often captures R2 from these "tails." When the distribution normalizes, the signal-to-noise ratio drops.

---

## 4. Leakage Audit Results

- **Status**: **PASS**.
- The [LeakFreePurgedKFold](file:///f:/Projects/hege_fund_kaggle/04_train_lightgbm.py#53-101) logic is correctly clearing overlaps and embargoes.
- **Conclusion**: The performance issue is purely data-driven (regime change), not a technical bug in the CV logic.

---

## 5. Strategic Recommendations

> [!IMPORTANT]
> **Action 1: Feature Re-Stationarity**
> Features with KS > 0.25 (k, z, cf, ar, aq) should be re-examined in `Modules/Preprocess/make_stationary.py`. If fractional differencing 'd' was too low, it may have failed to remove this drift.

> [!TIP]
> **Action 2: Time-Weighted Training**
> Since F4/F5 represent a different regime, consider using a **decaying weight** (recency bias) or training a separate "Recent Regime" model and ensembling it.

> [!CAUTION]
> **Action 3: Feature Pruning**
> Features like `feature_k` that have near-zero correlation in later folds should be evaluated for removal or specific handling, as they may introduce more noise than value in recent periods.
