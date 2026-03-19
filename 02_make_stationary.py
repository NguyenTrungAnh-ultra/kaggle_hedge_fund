import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train_handled_na.parquet')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_stationary.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'report_COMPARISON.csv')

def get_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] / k * (d - k + 1)
        w.append(w_k)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def find_optimal_d(series):
    series = series.dropna().values
    if len(series) < 20: 
        return 1.0 
    for d in np.linspace(0, 1, 11):
        if d == 0.0:
            diff_series = series
        else:
            w_full = get_weights(d, len(series))
            w_cumsum = np.cumsum(np.abs(w_full))
            skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2) # Increased tau to 1e-2
            
            # Cap window at 50 to preserve data in fragmented groups
            if (len(series) - skip) > 50:
                skip = len(series) - 50
                
            w = w_full[skip:]
            if len(w) == 0:
                w = w_full[-1:] # Fallback
                
            w_for_conv = w[::-1].flatten() 
            diff_series = np.convolve(series, w_for_conv, mode='valid')
            
        if len(diff_series) > 10:
            try:
                p_val = adfuller(diff_series)[1]
                if p_val <= 0.05:
                    return float(d)
            except:
                continue
    return 1.0

def process_data():
    if not os.path.exists(REPORT_FILE):
        print(f"Error: Could not find {REPORT_FILE}")
        sys.exit(1)

    print(f"STEP 1: Reading raw data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    report = pd.read_csv(REPORT_FILE)
    
    to_fix = report[report['Final_Verdict'].str.contains("Non-Stationary|Nghi ngờ", na=False)]
    cols_to_fix = to_fix['feature'].unique().tolist()
    
    df = df.sort_values(['code', 'sub_code', 'horizon', 'ts_index']).reset_index(drop=True)
    
    print("Ensuring AFML compliance: Estimating d ONLY on the 80% chronological train set to prevent Data Leak...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    train_df_for_est = df[df['ts_index'] < split_time]
    
    print("Finding the largest group [code, sub_code, horizon] in Train set for d estimation...")
    top_group_keys = train_df_for_est.groupby(['code', 'sub_code', 'horizon']).size().idxmax()
    mask_test = (train_df_for_est['code'] == top_group_keys[0]) & (train_df_for_est['sub_code'] == top_group_keys[1]) & (train_df_for_est['horizon'] == top_group_keys[2])
    test_group = train_df_for_est[mask_test]

    print(f"\nSTEP 2: Fast vectorized FFW transform using np.convolve for {len(cols_to_fix)} columns...")
    start_all = time.time()
    
    # Calculate max possible window size across all groups just in case
    # but we can just use the exact group size or the max overall size for weight limits
    max_size = df.groupby(['code', 'sub_code', 'horizon'], sort=False).size().max()
    grouped = df.groupby(['code', 'sub_code', 'horizon'], sort=False)

    for idx, col in enumerate(cols_to_fix, 1):
        start_col = time.time()
        
        test_series = test_group.set_index('ts_index')[col]
        opt_d = find_optimal_d(test_series)
        
        print(f"[{idx}/{len(cols_to_fix)}] {col:15} | d_opt: {opt_d:.1f} | ", end='', flush=True)
        
        if opt_d > 0.0:
            w_full = get_weights(opt_d, max_size)
            w_cumsum = np.cumsum(np.abs(w_full))
            skip = np.searchsorted(w_cumsum / w_cumsum[-1], 1e-2) # Increased tau to 1e-2
            
            # Cap window at 50 to preserve data
            if (max_size - skip) > 50:
                skip = max_size - 50
                
            w = w_full[skip:]
            
            if len(w) == 0:
                print(f"Warning: weights empty, keeping original. Done ({time.time() - start_col:.2f}s)")
                continue

            w_for_conv = w[::-1].flatten()
            window_len = len(w)

            def apply_ffw(g_vals):
                if len(g_vals) == 0:
                    return g_vals
                conv = np.convolve(g_vals, w_for_conv, mode='full')[:len(g_vals)]
                if len(conv) >= window_len:
                    conv[:window_len-1] = np.nan
                else:
                    conv[:] = np.nan
                return conv
                
            df[col] = grouped[col].transform(lambda g: apply_ffw(g.values))
        
        print(f"Done ({time.time() - start_col:.2f}s)")

    print(f"\nSTEP 3: Checking NaNs and dropping invalidated rows due to FFW windowing...")
    initial_rows = len(df)
    features_to_check = [c for c in df.columns if c not in ['id', 'y_target']]
    df = df.dropna(subset=features_to_check)
    
    final_rows = len(df)
    remaining_pct = (final_rows / initial_rows) * 100
    print(f"Original rows: {initial_rows}")
    print(f"Remaining rows after dropping NaNs: {final_rows}")
    print(f"Data Retention: {remaining_pct:.2f}%")

    if df[features_to_check].isna().sum().sum() > 0:
        print("WARNING: There are still NaNs in the features!")
    else:
        print("SUCCESS: No NaNs present in the feature set.")

    print(f"\nSTEP 4: Sorting by ts_index...")
    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)

    print(f"STEP 5: Saving the strictly correct data to: {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"\n==========================================")
    print(f"DATA PROCESSING COMPLETE")
    print(f"Total time: {time.time() - start_all:.2f}s")
    print(f"Output: train_stationary.parquet")
    print(f"==========================================")

if __name__ == "__main__":
    process_data()
