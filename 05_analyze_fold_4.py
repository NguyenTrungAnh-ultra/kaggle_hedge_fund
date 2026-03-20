import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

FILE_PATH = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'

def main():
    print("Step 1: Calculating split points...")
    ts_data = pd.read_parquet(FILE_PATH, columns=['ts_index'])
    unique_ts = np.sort(ts_data['ts_index'].unique())
    n_unique = len(unique_ts)
    split_ts_idx = int(n_unique * 0.80)
    train_unique_ts = unique_ts[:split_ts_idx]
    
    kf = KFold(n_splits=5, shuffle=False)
    folds = {}
    f_idx = 0
    for _, test_idx in kf.split(train_unique_ts):
        f_idx += 1
        if f_idx in [1, 4]:
            folds[f_idx] = train_unique_ts[test_idx]
    del ts_data
    
    top_features = ['sub_category', 'feature_bg', 'feature_aq', 'feature_ar', 'feature_a', 'horizon']
    cols_to_load = ['ts_index', 'y_target', 'weight', 'horizon'] + top_features
    
    stats = []
    print("\nStep 2: Comparison Analysis...")
    
    for f_id in [1, 4]:
        print(f"Loading Fold {f_id}...")
        target_ts = folds[f_id]
        fold_df = pd.read_parquet(
            FILE_PATH, 
            columns=cols_to_load,
            filters=[('ts_index', 'in', target_ts)]
        ).reset_index(drop=True)
        
        # Label encode sub_category if it's object type (mimicking main script)
        if 'sub_category' in fold_df.columns and fold_df['sub_category'].dtype == 'object':
            le = LabelEncoder()
            fold_df['sub_category'] = le.fit_transform(fold_df['sub_category'].astype(str))
            
        fold_df = fold_df[fold_df['y_target'].notna()].copy()
        
        # Calculations (be careful with nans in other columns)
        # Use simple mean/var first to ensure stability
        t_var = fold_df['y_target'].var()
        w_denom = np.sum(fold_df['weight'] * (fold_df['y_target']**2))
        avg_h = fold_df['horizon'].mean()
        
        c_dict = {}
        m_dict = {}
        for f in top_features:
            if f in fold_df.columns:
                # Fillna with 0 for correlation stability if needed, or just skip nans
                valid_f = fold_df[f].dropna()
                if len(valid_f) > 0:
                    m_dict[f] = valid_f.mean()
                    # Only correlate overlapping non-na rows
                    temp_df = fold_df[[f, 'y_target']].dropna()
                    if len(temp_df) > 10:
                        c_dict[f] = temp_df[f].corr(temp_df['y_target'])
                    else:
                        c_dict[f] = 0.0
                else:
                    m_dict[f] = 0.0
                    c_dict[f] = 0.0
        
        stats.append({
            'fold': f_id,
            'size': len(fold_df),
            'target_var': t_var,
            'weighted_denom': w_denom,
            'avg_horizon': avg_h,
            'corrs': c_dict,
            'means': m_dict
        })
        del fold_df

    # Output remains same format...
    print("\n" + "="*70)
    print(f"{'Metric':<25} | {'Fold 1':<15} | {'Fold 4':<15}")
    print("-" * 70)
    print(f"{'Samples':<25} | {stats[0]['size']:<15} | {stats[1]['size']:<15}")
    print(f"{'Target Var':<25} | {stats[0]['target_var']:<15.6f} | {stats[1]['target_var']:<15.6f}")
    print(f"{'Weighted Denom (R2 Scale)':<25} | {stats[0]['weighted_denom']:<15.2f} | {stats[1]['weighted_denom']:<15.2f}")
    print(f"{'Avg Horizon':<25} | {stats[0]['avg_horizon']:<15.2f} | {stats[1]['avg_horizon']:<15.2f}")
    
    print("\n--- Feature Correlations (Pearson) ---")
    for f in top_features:
        print(f"{f:<25} | {stats[0]['corrs'].get(f, 0):<15.6f} | {stats[1]['corrs'].get(f, 0):<15.6f}")
        
    print("\n--- Feature Mean ---")
    for f in top_features:
        print(f"{f:<25} | {stats[0]['means'].get(f, 0):<15.4f} | {stats[1]['means'].get(f, 0):<15.4f}")

if __name__ == '__main__':
    main()
