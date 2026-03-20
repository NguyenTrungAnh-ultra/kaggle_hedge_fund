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
    
    stats_list = []
    print("\nStep 2: Comparison Analysis...")
    
    for f_id in [1, 4]:
        print(f"Loading Fold {f_id}...")
        u_ts = folds[f_id]
        
        chunk = pd.read_parquet(
            FILE_PATH, 
            columns=cols_to_load,
            filters=[('ts_index', 'in', u_ts)]
        ).reset_index(drop=True)
        
        # Check for duplicate columns
        if chunk.columns.duplicated().any():
            print(f"Warning: Duplicate columns found in Fold {f_id}: {chunk.columns[chunk.columns.duplicated()].unique().tolist()}")
            # Rename duplicates to allow selection
            chunk.columns = [f"{col}_{i}" if chunk.columns.duplicated()[i] else col for i, col in enumerate(chunk.columns)]

        # Consistent encoding
        target_feat = 'sub_category'
        if target_feat in chunk.columns:
            if chunk[target_feat].dtype == 'object':
                le = LabelEncoder()
                chunk[target_feat] = le.fit_transform(chunk[target_feat].astype(str))
            
        # Valid Target only
        chunk = chunk[chunk['y_target'].notna()].copy()
        
        # Stats
        target_var = float(chunk['y_target'].var())
        weighted_denom = float(np.sum(chunk['weight'] * (chunk['y_target']**2)))
        avg_hor = float(chunk['horizon'].mean())
        
        c_dict = {}
        m_dict = {}
        for feat in top_features:
            if feat in chunk.columns:
                series = chunk[feat]
                m_dict[feat] = float(series.mean())
                # Correlation
                sub = chunk[[feat, 'y_target']].dropna()
                if len(sub) > 100:
                    c_dict[feat] = float(sub[feat].corr(sub['y_target']))
                else:
                    c_dict[feat] = 0.0
        
        stats_list.append({
            'fold': f_id,
            'size': len(chunk),
            'target_var': target_var,
            'weighted_denom': weighted_denom,
            'avg_horizon': avg_hor,
            'corrs': c_dict,
            'means': m_dict
        })
        del chunk

    # Final Summary Table
    print("\n" + "="*75)
    print(f"{'Metric':<25} | {'Fold 1':<15} | {'Fold 4':<15}")
    print("-" * 75)
    f1, f4 = stats_list[0], stats_list[1]
    
    print(f"{'Samples':<25} | {f1['size']:<15} | {f4['size']:<15}")
    print(f"{'Target Var':<25} | {f1['target_var']:<15.6f} | {f4['target_var']:<15.6f}")
    print(f"{'Weighted Denom (Scale)':<25} | {f1['weighted_denom']:<15.2f} | {f4['weighted_denom']:<15.2f}")
    print(f"{'Avg Horizon':<25} | {f1['avg_horizon']:<15.2f} | {f4['avg_horizon']:<15.2f}")
    
    print("\n--- Feature Correlations (Pearson) ---")
    for feat in top_features:
        v1, v4 = f1['corrs'].get(feat, 0), f4['corrs'].get(feat, 0)
        print(f"{feat:<25} | {v1:<15.6f} | {v4:<15.6f}")
        
    print("\n--- Feature Mean ---")
    for feat in top_features:
        m1, m4 = f1['means'].get(feat, 0), f4['means'].get(feat, 0)
        print(f"{feat:<25} | {m1:<15.4f} | {m4:<15.4f}")

if __name__ == '__main__':
    main()
