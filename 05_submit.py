import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import time
import sys
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
TRAIN_RAW = os.path.join(DATA_DIR, 'train.parquet')
TEST_RAW = os.path.join(DATA_DIR, 'test.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'report_COMPARISON.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models_param')
SUBMISSION_FILE = os.path.join(BASE_DIR, 'test_result', 'submission.csv')

def get_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def find_optimal_d(series):
    series = series.dropna().values
    if len(series) < 20: return 1.0 
    for d in np.linspace(0, 1, 11):
        if d == 0.0:
            diff_series = series
        else:
            w_full = get_weights(d, len(series))
            w_cumsum = np.cumsum(np.abs(w_full))
            skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2)
            if (len(series) - skip) > 50: skip = len(series) - 50
            w = w_full[skip:]
            if len(w) == 0: w = w_full[-1:]
            w_for_conv = w[::-1].flatten() 
            diff_series = np.convolve(series, w_for_conv, mode='valid')
        if len(diff_series) > 10:
            try:
                if adfuller(diff_series)[1] <= 0.05: return float(d)
            except: continue
    return 1.0

def main():
    print(f"=== KAGGLE INFERENCE PIPELINE ===")
    start_time = time.time()
    
    print("1. Loading raw TRAIN and TEST data directly into contiguous timeline...")
    train = pd.read_parquet(TRAIN_RAW)
    test = pd.read_parquet(TEST_RAW)
    
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")
    
    # Label datasets rigidly before union
    train['is_test'] = 0
    test['is_test'] = 1
    
    # Append test firmly AT THE END of chronological train flows
    df = pd.concat([train, test], ignore_index=True)
    
    meta_cols = ['id', 'y_target', 'weight', 'ts_index', 'code', 'sub_code', 'sub_category', 'horizon', 'is_test']
    features = [c for c in train.columns if c not in meta_cols]
    
    print("\n2. [Layer 1-4] Restoring Memory in NaN Imputation...")
    # Discover exactly the same missing features natively defined in training origin
    na_before = train[features].isna().sum()
    features_with_na = na_before[na_before > 0].index.tolist()
    
    # Create the rigid structural binary flags
    for col in features_with_na:
        df[f'is_nan_{col}'] = df[col].isna().astype(np.int8)
        
    # Temporal Sort is MANDATORY to allow memory vector to naturally bleed from Train to Test
    print("Executing ffill(50) along timeline...")
    df = df.sort_values(by=['code', 'horizon', 'ts_index'])
    df[features_with_na] = df.groupby(['code', 'horizon'])[features_with_na].ffill(limit=50)
    
    # Compute median strictly identical to training baseline rules
    na_after_l3 = train[features_with_na].isna().sum() 
    features_still_na = na_after_l3[na_after_l3 > 0].index.tolist()
    if features_still_na:
        print("Executing temporal cross-sectional median stabilization...")
        medians_ts_horizon = df.groupby(['ts_index', 'horizon'])[features_still_na].transform('median')
        df[features_still_na] = df[features_still_na].fillna(medians_ts_horizon)
        
    df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
    
    print("\n3. Restoring Memory Fractional Differencing (Stationarity Limit)...")
    if not os.path.exists(REPORT_FILE):
        print(f"FAILED: Report {REPORT_FILE} missing.")
        return
        
    report = pd.read_csv(REPORT_FILE)
    to_fix = report[report['Final_Verdict'].str.contains("Non-Stationary|Nghi ngờ", na=False)]
    cols_to_fix = to_fix['feature'].unique().tolist()
    
    train_only = df[df['is_test'] == 0]
    unique_ts = np.sort(train_only['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    train_df_for_est = train_only[train_only['ts_index'] < split_time]
    
    top_group_keys = train_df_for_est.groupby(['code', 'sub_code', 'horizon']).size().idxmax()
    mask_test = (train_df_for_est['code'] == top_group_keys[0]) & (train_df_for_est['sub_code'] == top_group_keys[1]) & (train_df_for_est['horizon'] == top_group_keys[2])
    test_group = train_df_for_est[mask_test]
    
    max_size = df.groupby(['code', 'sub_code', 'horizon'], sort=False).size().max()
    grouped = df.groupby(['code', 'sub_code', 'horizon'], sort=False)
    
    for idx, col in enumerate(cols_to_fix, 1):
        if idx % 5 == 0:
            print(f"FFW Transforming: [{idx}/{len(cols_to_fix)}] {col:15}")
        test_series = test_group.set_index('ts_index')[col]
        opt_d = find_optimal_d(test_series)
        
        if opt_d > 0.0:
            w_full = get_weights(opt_d, max_size)
            w_cumsum = np.cumsum(np.abs(w_full))
            skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2)
            if (max_size - skip) > 50: skip = max_size - 50
            w = w_full[skip:]
            if len(w) == 0: continue
            w_for_conv = w[::-1].flatten()
            window_len = len(w)
            
            def apply_ffw(g_vals):
                if len(g_vals) == 0: return g_vals
                conv = np.convolve(g_vals, w_for_conv, mode='full')[:len(g_vals)]
                if len(conv) >= window_len: conv[:window_len-1] = np.nan
                else: conv[:] = np.nan
                return conv
                
            df[col] = grouped[col].transform(lambda g: apply_ffw(g.values))
            
    # Extract only the test rows out
    print("\n4. Isolating Test Frame for Inference...")
    df = df.sort_values(['ts_index', 'code', 'sub_code', 'horizon']).reset_index(drop=True)
    test_processed = df[df['is_test'] == 1].copy()
    del df # Purge mem
    del train
    del test
    
    print("\n5. Target Feature Engineering & Encodings...")
    test_processed['t1_end'] = test_processed['ts_index'] + test_processed['horizon']
    
    cat_cols = ['code', 'sub_code', 'sub_category']
    # Safely load the categorical boundaries directly from training labels to avoid Unknown Label crash
    # Due to full memory architecture, we can use the label mapping identically as observed on train_only.
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_only[col].astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        test_processed[col] = test_processed[col].astype(str).map(mapping).fillna(-1).astype(int)
            
    print("\n6. Loading Model Ensemble Matrix...")
    models = []
    for i in range(1, 6):
        m_path = os.path.join(MODELS_DIR, f"lgbm_fold_{i}.txt")
        if not os.path.exists(m_path):
            print(f"FATAL: Model {m_path} missing.")
            return
        m = lgb.Booster(model_file=m_path)
        models.append(m)
        
    exclude = ['id', 'y_target', 'weight', 'ts_index', 'is_test']
    submission_features = [c for c in test_processed.columns if c not in exclude]
            
    print(f"Matrix Dimension Checked: {len(submission_features)} aligned features.")
    
    X_test = test_processed[submission_features].values
    
    print(f"\n7. Activating Inference Engine (Yielding over {len(X_test)} entities)...")
    preds = np.zeros(len(X_test))
    for i, m in enumerate(models, 1):
        print(f" > Fold {i} weights deployed.")
        preds += m.predict(X_test) / len(models)
        
    test_processed['y_target'] = preds
    submission = test_processed[['id', 'y_target']]
    
    print("\n8. Packaging Result...")
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n==> SUCCESS: Submission ready at {SUBMISSION_FILE}")
    print(f"Total Inference Time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
