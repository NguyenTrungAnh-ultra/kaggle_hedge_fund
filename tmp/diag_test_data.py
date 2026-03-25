import pandas as pd
import numpy as np
import os

DATA_DIR = r"f:\Projects\hege_fund_kaggle\ts-forecasting"
file_path = os.path.join(DATA_DIR, 'train_stationary.parquet')

if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    print(f"Dataset columns: {df.columns.tolist()[:10]}...")
    print(f"y_target dtype: {df['y_target'].dtype}")
    print(f"weight dtype: {df['weight'].dtype}")
    
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    test_df = df[df['ts_index'] >= split_time].copy()
    test_y = test_df['y_target'].values.astype(np.float64)
    test_w = test_df['weight'].values.astype(np.float64)
    
    denom = np.sum(test_w * (test_y ** 2))
    print(f"Exact denom (sum w*y^2): {denom:.12e}")
    
    test_nans_y = np.isnan(test_y).sum()
    test_nans_w = np.isnan(test_w).sum()
    print(f"NaNs in y: {test_nans_y}, in w: {test_nans_w}")
else:
    print(f"File not found: {file_path}")
