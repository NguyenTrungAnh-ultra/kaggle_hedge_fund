import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import sys

BASE_DIR = r"f:\Projects\hege_fund_kaggle\Models"
PROJECT_ROOT = r"f:\Projects\hege_fund_kaggle"
sys.path.append(PROJECT_ROOT)
sys.path.append(BASE_DIR)

from metric import weighted_rmse_score

DATA_DIR = r"f:\Projects\hege_fund_kaggle\ts-forecasting"
file_path = os.path.join(DATA_DIR, 'train_stationary.parquet')

print("1. Loading data...")
df = pd.read_parquet(file_path)

# Mock the processing in run_lgbm
cat_cols = ['code', 'sub_code', 'sub_category']
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

unique_ts = np.sort(df['ts_index'].unique())
split_ts_idx = int(len(unique_ts) * 0.80)
split_time = unique_ts[split_ts_idx]

test_df = df[df['ts_index'] >= split_time].copy()
blacklist = ['feature_k', 'feature_aq']
exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end'] + blacklist
features = [c for c in df.columns if c not in exclude]

# Drop constant columns as in run_lgbm
const_cols = [col for col in features if df[col].nunique() <= 1]
features = [c for c in features if c not in const_cols]

print(f"Features: {len(features)}")

test_X = test_df[features].values
test_y = test_df['y_target'].values
test_w = test_df['weight'].values

print(f"test_X shape: {test_X.shape}, NaNs: {np.isnan(test_X).sum()}")

test_preds = np.zeros(len(test_df))
for fold in range(1, 6):
    model_filename = os.path.join(PROJECT_ROOT, "models_param", f"lgbm_fold_{fold}.txt")
    if os.path.exists(model_filename):
        print(f"Testing model: {model_filename}")
        model = lgb.Booster(model_file=model_filename)
        preds = model.predict(test_X)
        print(f"  Fold {fold} preds: mean={preds.mean():.6e}, std={preds.std():.6e}")
        if np.isnan(preds).any():
            print(f"  !!! Fold {fold} has {np.isnan(preds).sum()} NaNs")
        test_preds += preds / 5

print(f"\nEnsemble preds: mean={test_preds.mean():.6e}, std={test_preds.std():.6e}")

denom = np.sum(test_w * (test_y ** 2))
num = np.sum(test_w * ((test_y - test_preds) ** 2))
print(f"Denom: {denom:.6e}, Num: {num:.6e}")
ratio = num / denom if denom != 0 else 0
print(f"Ratio: {ratio:.6f}")
r2 = 1.0 - np.minimum(np.maximum(ratio, 0.0), 1.0)
print(f"Final Score R2: {r2:.6f}")
