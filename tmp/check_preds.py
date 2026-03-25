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

df = pd.read_parquet(file_path)
unique_ts = np.sort(df['ts_index'].unique())
split_ts_idx = int(len(unique_ts) * 0.80)
split_time = unique_ts[split_ts_idx]

test_df = df[df['ts_index'] >= split_time].copy()
exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end', 'feature_k', 'feature_aq'] # Assuming blacklist
features = [c for c in test_df.columns if c not in exclude]

test_X = test_df[features].values
test_y = test_df['y_target'].values
test_w = test_df['weight'].values

test_preds = np.zeros(len(test_df))
for fold in range(1, 6):
    model_filename = os.path.join(PROJECT_ROOT, "models_param", f"lgbm_fold_{fold}.txt")
    if os.path.exists(model_filename):
        print(f"Loading model: {model_filename}")
        model = lgb.Booster(model_file=model_filename)
        preds = model.predict(test_X)
        print(f"  Fold {fold} preds: mean={preds.mean():.6f}, std={preds.std():.6f}, min={preds.min():.6f}, max={preds.max():.6f}")
        test_preds += preds / 5

print(f"\nEnsemble preds: mean={test_preds.mean():.6f}, std={test_preds.std():.6f}")

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def zero_mean_r2(y_true, y_pred, w):
    denom = np.sum(w * y_true ** 2)
    if denom == 0: return 0.0
    ratio = np.sum(w * (y_true - y_pred) ** 2) / denom
    print(f"  denom: {denom:.6f}, num: {np.sum(w * (y_true - y_pred) ** 2):.6f}, ratio: {ratio:.6f}")
    return 1.0 - _clip01(ratio)

score = zero_mean_r2(test_y, test_preds, test_w)
print(f"Final Score: {score:.6f}")
