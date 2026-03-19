import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import sys

warnings.filterwarnings('ignore')

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(LOG_DIR, "benchmark.log"), "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- KAGGLE COMPETITION METRIC ---
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def zero_mean_r2_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    if denom == 0: return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(val)

# --- PURGED K-FOLD (FIXED LEAKAGE) ---
class LeakFreePurgedKFold:
    def __init__(self, n_splits=5, pctEmbargo=0.01):
        self.n_splits = n_splits
        self.pctEmbargo = pctEmbargo
        
    def split(self, df):
        unique_ts = np.sort(df['ts_index'].unique())
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        embargo = int(len(unique_ts) * self.pctEmbargo)
        ts_index_arr = df['ts_index'].values
        t1_arr = (df['ts_index'] + df['horizon']).values
        
        for train_ts_idx, test_ts_idx in kf.split(unique_ts):
            train_times = unique_ts[train_ts_idx]
            test_times = unique_ts[test_ts_idx]
            t0_test = test_times[0]
            t1_test = (df[df['ts_index'].isin(test_times)]['ts_index'] + df[df['ts_index'].isin(test_times)]['horizon']).max()
            
            base_train_mask = np.isin(ts_index_arr, train_times)
            test_indices = np.where(np.isin(ts_index_arr, test_times))[0]
            
            keep_before = t1_arr <= t0_test
            keep_after = ts_index_arr > t1_test
            
            if embargo > 0:
                after_test_times = unique_ts[unique_ts > t1_test]
                if len(after_test_times) > 0:
                    embargoed_times = after_test_times[:embargo]
                    keep_after = keep_after & (~np.isin(ts_index_arr, embargoed_times))
                    
            final_train_mask = base_train_mask & (keep_before | keep_after)
            train_indices = np.where(final_train_mask)[0]
            yield train_indices, test_indices

def main():
    print("1. Loading RAW dataset for BENCHMARK...")
    file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_handled_na.parquet'
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    print(f"Raw dataset shape: {df.shape}")

    # Label Encoding
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # User's dropna()
    print("Dropping NaNs (user's way)...")
    df = df.dropna().reset_index(drop=True)
    print(f"Shape after dropna: {df.shape}")

    # Prepare t1 (Label End Time) column
    df['t1_end'] = df['ts_index'] + df['horizon']
    df = df.sort_values(['ts_index']).reset_index(drop=True)

    # Chronological Split (Train/Test 80/20)
    print("\n2. Splitting Baseline (80/20 Chronological + Purging)...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    train_df = df[df['ts_index'] < split_time].copy()
    test_df = df[df['ts_index'] >= split_time].copy()
    
    # Purge gap for test
    overlap_mask = train_df['t1_end'] > test_df['ts_index'].min()
    if overlap_mask.any():
        print(f"Purging {overlap_mask.sum()} samples overlapping test.")
        train_df = train_df[~overlap_mask].copy().reset_index(drop=True)

    # Feature selection
    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end', 'horizon']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"Scanning {len(features)} features for constant columns...")
    const_cols = []
    for i, col in enumerate(features, 1):
        if i % 10 == 0:
            print(f"Progress: [{i}/{len(features)}] features scanned...")
        if df[col].nunique() <= 1:
            const_cols.append(col)
            
    if const_cols:
        print(f"Dropping {len(const_cols)} constant columns.")
        features = [c for c in features if c not in const_cols]
        
    print(f"Total benchmark features: {len(features)}")

    # Purged K-Fold Training
    print("\n3. Starting Purged K-Fold Training for BENCHMARK (ON CPU)...")
    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    X_train_full = train_df[features].values
    y_train_full = train_df['y_target'].values
    w_train_full = train_df['weight'].values
    
    test_X = test_df[features].values
    test_y = test_df['y_target'].values
    test_w = test_df['weight'].values

    test_preds = np.zeros(len(test_df))
    
    fold = 1
    for tr_idx, va_idx in cv.split(train_df):
        print(f"\n--- Fold {fold} ---")
        X_tr, y_tr, w_tr = X_train_full[tr_idx], y_train_full[tr_idx], w_train_full[tr_idx]
        X_va, y_va, w_va = X_train_full[va_idx], y_train_full[va_idx], w_train_full[va_idx]
        
        # CPU used here for benchmark because RAW data crashes GPU logic
        model = lgb.LGBMRegressor(
            n_estimators=1000, 
            device='cpu',
            num_leaves=127,
            learning_rate=0.05,
            extra_trees=True,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            random_state=42,
            verbose=-1
        )
        
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        v_score = zero_mean_r2_score(y_va, model.predict(X_va), w_va)
        print(f"Fold {fold} Validation R2: {v_score:.5f}")
        
        test_preds += model.predict(test_X) / cv.n_splits
        fold += 1
        
    print("\n4. Final Benchmark Evaluation...")
    final_score = zero_mean_r2_score(test_y, test_preds, test_w)
    print(f"==> BENCHMARK ZERO-MEAN R2 SCORE: {final_score:.6f}")
    
    report_file = r'f:\Projects\hege_fund_kaggle\benchmark_report.txt'
    with open(report_file, 'w') as f:
        f.write("=== SYNCED PURGED BENCHMARK EVALUATION ===\n")
        f.write("Data: RAW (dropna() row-wise)\n")
        f.write("Validation: Leak-Free Purged K-Fold (Synced with train.py)\n")
        f.write(f"Kaggle Zero-Mean R2 Score: {final_score:.6f}\n")
        
    print(f"Benchmark report saved to {report_file}")

if __name__ == '__main__':
    main()