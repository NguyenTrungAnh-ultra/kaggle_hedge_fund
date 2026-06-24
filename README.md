# 📈 Quantitative Trading Forecast

## 📊 Project Overview

This project focuses on building a robust machine learning pipeline for multivariate financial time-series forecasting. The goal is to predict asset returns (`y_target`) across multiple future time horizons (1, 3, 10, and 25 steps) while handling highly noisy data, missing values (NaNs), regime shifts, and severe outliers.

## 🛠️ Technical Approach

Our primary focus is on **generalization** and **strict leakage prevention** to ensure high stability on the Private Leaderboard.

- **Data Cleansing & Normalization:** We enforce rigorous data preprocessing, including forward-filling (`ffill`) and median imputation to safely handle missing values. We also apply Cross-Sectional Z-Score normalization on key features to eliminate broad market volatility (Market Beta) and isolate pure asset signals (Alpha).
- **Leak-Free Cross-Validation:** We utilize a **5-Fold Purged Time-Series K-Fold CV** with a 1% Embargo. This strictly isolates training sets from validation sets, completely preventing the look-ahead bias commonly found in financial modeling.
- **Causal Lags Feature Engineering:** We generate lagged features, rolling statistics (mean, std), and trend slopes specifically for causal variables to capture temporal dynamics.
- **Target Encoding:** Target encoding is safely mapped within the cross-validation loops to prevent target leakage into the validation folds.
- **Recency Weighting:** We apply exponential decay sample weighting (half-life fraction = 0.5) to prioritize recent training data, as all 9 DAG features showed significant temporal distribution shifts (KS test p ≈ 0).
- **Full-Dataset Inference:** We train an ensemble of 20 LightGBM models (4 Horizons × 5 Random Seeds) on 100% of the training data to ensure the models capture the most recent market regimes before predicting the test set.

## 🚀 Results TOP 7

- **Current Public LB Score:** `0.1885`
- Our model provides an extremely robust foundation. It trades a marginal drop in Public LB score for extreme stability and generalization against future, unseen market conditions.

## ⚙️ System Architecture

The project operates as a sequential time-series forecasting pipeline:

1. **Data Ingestion:** Reads uncompressed parquet files and enforces memory-efficient dtypes (`float32`, `int8`, `category`).
2. **Preprocessing Pipeline:** Applies strict NaN imputation, Cross-Sectional Z-Score normalization, and target encoding.
3. **Feature Engineering:** Computes causal lags, rolling metrics, and horizon indicators (`log_horizon`).
4. **LightGBM Execution:** Trains independent Gradient Boosting models via native LightGBM separated by forecast horizons.
5. **Inference & Aggregation:** Loads full-trained horizon models, averages predictions across seeds, and outputs the final submission file.
