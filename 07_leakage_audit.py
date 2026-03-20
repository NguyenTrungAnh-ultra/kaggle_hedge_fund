import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

FILE_PATH = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'

def main():
    print("Auditing Leak-Free Purged K-Fold logic...")
    df = pd.read_parquet(FILE_PATH, columns=['ts_index', 'horizon'])
    df['t1_end'] = df['ts_index'] + df['horizon']
    
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    train_unique_ts = unique_ts[:split_ts_idx]
    train_df = df[df['ts_index'] < unique_ts[split_ts_idx]].copy()
    
    kf = KFold(n_splits=5, shuffle=False)
    pctEmbargo = 0.01
    embargo = int(len(train_unique_ts) * pctEmbargo)
    
    ts_index_arr = train_df['ts_index'].values
    t1_arr = train_df['t1_end'].values
    
    leak_found = False
    for f_idx, (train_ts_idx, val_ts_idx) in enumerate(kf.split(train_unique_ts)):
        fold = f_idx + 1
        train_times = train_unique_ts[train_ts_idx]
        val_times = train_unique_ts[val_ts_idx]
        
        t0_val = val_times[0]
        t1_val = train_df[train_df['ts_index'].isin(val_times)]['t1_end'].max()
        
        # Current logic for training set in Fold
        keep_before = t1_arr <= t0_val
        keep_after = ts_index_arr > t1_val 
        
        if embargo > 0:
            after_val_times = train_unique_ts[train_unique_ts > t1_val]
            if len(after_val_times) > 0:
                embargo_times = after_val_times[:embargo]
                keep_after = keep_after & (~np.isin(ts_index_arr, embargo_times))
        
        # Check for overlap: Train samples whose (ts_index, t1_end) overlaps with Val block (t0_val, t1_val)
        # Any Train sample (t0_i, t1_i) leaks if:
        # t0_i < t1_val AND t1_i > t0_val
        
        train_mask = (np.isin(ts_index_arr, train_times)) & (keep_before | keep_after)
        train_indices = np.where(train_mask)[0]
        
        # Verify overlap for all kept train samples
        # A sample is safe if: (t1_i <= t0_val) OR (t0_i > t1_val)
        leaks = (t1_arr[train_indices] > t0_val) & (ts_index_arr[train_indices] < t1_val)
        leak_count = np.sum(leaks)
        
        print(f"Fold {fold}: Used Train={len(train_indices)}, Leaks found={leak_count}")
        if leak_count > 0:
            leak_found = True
            
    if not leak_found:
        print("\nSUCCESS: No data leakage detected in the Purged K-Fold logic.")
    else:
        print("\nWARNING: Potential leakage detected! Review Purged K-Fold logic.")

if __name__ == '__main__':
    main()
