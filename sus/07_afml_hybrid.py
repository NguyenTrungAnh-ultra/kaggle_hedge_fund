import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# --- PATHS & LOGGING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
INPUT_FILE = os.path.join(BASE_DIR, 'ts-forecasting', 'train_stationary.parquet')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(LOG_DIR, "afml_hybrid.log"), "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- HYBRID METRIC (From Notebook) ---
def comp_metric_lgb(y_pred, data):
    y_true = data.get_label()
    w = data.get_weight()
    denom = np.sum(w * (y_true ** 2))
    if denom == 0: denom = 1e-15
    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / denom
    clipped = np.clip(ratio, 0.0, 1.0)
    val = 1.0 - clipped
    score = np.sqrt(val)
    return 'weighted_score', score, True

# --- PURGED K-FOLD (AFML Standard) ---
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
    print("=== HYBRID PIPELINE: AFML STRUCTURE + NOTEBOOK METRICS/PARAMS ===")
    start_time_all = time.time()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"1. Loading Stationary Data: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Shape: {df.shape}")

    # Encoding
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Features
    exclude = ['id', 'y_target', 'weight', 'ts_index', 'horizon']
    features = [c for c in df.columns if c not in exclude]
    
    # 80/20 Chronological Wrap
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts = unique_ts[int(len(unique_ts) * 0.8)]
    train_df = df[df['ts_index'] < split_ts].copy()
    test_df = df[df['ts_index'] >= split_ts].copy()
    
    print(f"Train/Test split at ts_index={split_ts}")

    # Training Matrix
    X_train_full = train_df[features].values
    y_train_full = train_df['y_target'].values
    w_train_full = train_df['weight'].values
    
    test_X = test_df[features].values
    test_y = test_df['y_target'].values
    test_w = test_df['weight'].values

    # Hybrid CV
    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    # Notebook Hyperparams
    params = {
        'objective': 'regression',
        'metric': 'None',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 30,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'device': 'cpu', # Staying on CPU for stability in this run
        'verbose': -1
    }

    test_preds = np.zeros(len(test_df))
    fold_scores = []

    print("\n2. Starting Hybrid Purged K-Fold...")
    for fold, (tr_idx, va_idx) in enumerate(cv.split(train_df), 1):
        print(f"--- Fold {fold} ---")
        X_tr, y_tr, w_tr = X_train_full[tr_idx], y_train_full[tr_idx], w_train_full[tr_idx]
        X_va, y_va, w_va = X_train_full[va_idx], y_train_full[va_idx], w_train_full[va_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            feval=comp_metric_lgb,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        v_score = model.best_score['valid_1']['weighted_score']
        fold_scores.append(v_score)
        print(f"Fold {fold} Weighted Score: {v_score:.6f}")
        
        test_preds += model.predict(test_X) / 5
        
    avg_v = np.mean(fold_scores)
    
    # Final eval on test_df
    def simple_eval(y_true, y_pred, w):
        denom = np.sum(w * (y_true ** 2))
        if denom == 0: denom = 1e-15
        ratio = np.sum(w * ((y_true - y_pred) ** 2)) / denom
        return 1.0 - np.clip(ratio, 0.0, 1.0) # returning R2 version (1-ratio)

    final_r2_test = simple_eval(test_y, test_preds, test_w)
    
    print("\n3. Hybrid Evaluation Results")
    print(f"Average CV Weighted Score (sqrt(1-ratio)): {avg_v:.6f}")
    print(f"Final TEST Zero-Mean R2 (1-ratio): {final_r2_test:.6f}")
    
    print(f"\nExecution finished in {time.time() - start_time_all:.2f}s")

if __name__ == '__main__':
    main()
