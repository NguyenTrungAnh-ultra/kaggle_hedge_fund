# Comprehensive Analysis Report: Fold 4 Performance Discrepancy

## Executive Summary
The performance drop in Fold 4 (R2 ~0.0036 vs Fold 1 R2 ~0.0128) is primarily driven by a **Regime Change** in the financial time series. Key predictive signals have either evaporated or drifted significantly, while the target distribution has become less "fat-tailed," making it harder for the model to capture extreme returns.

---

## 1. Feature Signal Decay (Correlation Collapse)
The most critical finding is the loss of predictive power in the model's top features.

| Feature | Fold 1 Corr | Fold 4 Corr | Change | Impact |
| :--- | :--- | :--- | :--- | :--- |
| `sub_category` | 0.0102 | 0.0008 | -92% | **CRITICAL**: Primary signal lost. |
| `feature_k` | -0.1365 | -0.0003 | -99% | **HIGH**: Massive feature signal vanished. |
| `feature_aq` | -0.0375 | 0.0051 | Flipped | **HIGH**: Directional change. |
| `feature_h` | -0.0654 | 0.0032 | Flipped | Signal turned to noise. |

---

## 2. Data Drift Analysis (KS Test)
The Kolmogorov-Smirnov (KS) statistic measures the difference between probability distributions. Values > 0.25 indicate severe drift.

*   **`feature_cf` (KS = 0.3639)**: Most drifted feature. The model trained on F1-F3 logic is likely applying the wrong "bins" to this feature in F4.
*   **`feature_z` (KS = 0.3355)**: Significant distributional shift.
*   **`feature_k` (KS = 0.3214)**: Confirms that its correlation collapse is tied to a fundamental change in the feature's behavior.

---

## 3. Target Characteristics Shift
The nature of the target variable `y_target` has changed over time.

*   **Kurtosis Drop**: Fold 1 (383) $\rightarrow$ Fold 4 (174). 
    *   The market became less "fat-tailed" in the F4 period (fewer extreme outliers proportional to the mean).
    *   LightGBM often captures R2 from these "tails." When the distribution normalizes, the signal-to-noise ratio drops.

---

## 4. Leakage Audit Results
*   **Status**: **PASS**.
*   The [LeakFreePurgedKFold](file:///f:/Projects/hege_fund_kaggle/04_train_lightgbm.py#53-101) logic is correctly clearing overlaps and embargoes. 
*   **Conclusion**: The performance issue is purely data-driven (regime change), not a technical bug in the CV logic.

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
