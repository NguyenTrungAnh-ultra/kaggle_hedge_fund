import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings('ignore')

FILE_PATH = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'

def main():
    print("Step 1: Calculating Folds...")
    ts_data = pd.read_parquet(FILE_PATH, columns=['ts_index'])
    unique_ts = np.sort(ts_data['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    train_unique_ts = unique_ts[:split_ts_idx]
    
    kf = KFold(n_splits=5, shuffle=False)
    folds = {}
    f_idx = 0
    for _, test_idx in kf.split(train_unique_ts):
        f_idx += 1
        folds[f_idx] = train_unique_ts[test_idx]
    del ts_data
    
    top_features = [
        'sub_category', 'feature_bg', 'feature_aq', 'feature_ar', 'feature_a',
        'feature_as', 'feature_ce', 'feature_cg', 'feature_z', 'feature_cf',
        'horizon', 'feature_ai', 'feature_ad', 'feature_al', 'feature_j',
        'feature_l', 'feature_k', 'feature_y', 'sub_code', 'feature_h'
    ]
    cols_to_load = ['ts_index', 'y_target', 'weight', 'horizon'] + top_features
    
    all_fold_stats = {}
    
    print("\nStep 2: Processing Folds for Drift Analysis...")
    for f_id in range(1, 6):
        print(f"Loading Fold {f_id}...")
        u_ts = folds[f_id]
        chunk = pd.read_parquet(
            FILE_PATH, 
            columns=cols_to_load,
            filters=[('ts_index', 'in', u_ts)]
        ).reset_index(drop=True)
        
        # Deduplicate columns
        if chunk.columns.duplicated().any():
            chunk.columns = [f"{col}_{i}" if chunk.columns.duplicated()[i] else col for i, col in enumerate(chunk.columns)]

        # Label Encode categorical objects
        for cat_f in ['sub_category', 'sub_code']:
            if cat_f in chunk.columns:
                chunk[cat_f] = LabelEncoder().fit_transform(chunk[cat_f].astype(str))
            
        chunk = chunk[chunk['y_target'].notna()].copy()
        
        # Stats
        f_stats = {
            'size': len(chunk),
            'target_std': float(chunk['y_target'].std()),
            'target_kurtosis': float(chunk['y_target'].kurtosis()),
            'means': {f: float(chunk[f].mean()) for f in top_features if f in chunk.columns},
            'stds': {f: float(chunk[f].std()) for f in top_features if f in chunk.columns},
            'corrs': {f: float(chunk[f].corr(chunk['y_target'])) for f in top_features if f in chunk.columns},
            'data': {f: chunk[f].fillna(0).values for f in top_features if f in chunk.columns} 
        }
        all_fold_stats[f_id] = f_stats
        del chunk

    # Step 3: Reporting Results
    print("\n" + "="*85)
    print(f"{'Feature':<20} | {'F1 Corr':<8} | {'F4 Corr':<8} | {'KS Drift (F4 vs F1)':<15}")
    print("-" * 85)
    
    ref = all_fold_stats[1]
    f4 = all_fold_stats[4]
    
    for feat in top_features:
        c1 = ref['corrs'].get(feat, 0)
        c4 = f4['corrs'].get(feat, 0)
        # KS Statistic
        if feat in ref['data'] and feat in f4['data']:
            ks_stat, _ = ks_2samp(ref['data'][feat], f4['data'][feat])
        else:
            ks_stat = 0.0
            
        print(f"{feat:<20} | {c1:<8.4f} | {c4:<8.4f} | {ks_stat:<15.4f}")

    print("\n--- TARGET CHARACTERISTICS ---")
    print(f"{'Metric':<20} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'Fold 5':<8}")
    print("-" * 85)
    for m in ['size', 'target_std', 'target_kurtosis']:
        row = [f"{m:<20}"]
        for i in range(1, 6):
            val = all_fold_stats[i][m]
            row.append(f"{val:<8.3f}" if isinstance(val, float) else f"{val:<8}")
        print(" | ".join(row))

if __name__ == '__main__':
    main()
