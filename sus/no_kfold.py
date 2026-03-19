import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train.parquet')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(LOG_DIR, "no_kfold_train.log"), "a", encoding='utf-8')

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

def main():
    print("=== SIMPLE 80/20 TRAIN PIPELINE (NO K-FOLD, FILLNA=0) ===")
    start_time = time.time()

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"1. Loading RAW dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Raw dataset shape: {df.shape}")

    # Handling Categorical Columns
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        if col in df.columns:
            print(f"Label Encoding: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Simple Fill NA = 0
    print("2. Filling ALL NaNs with 0 (as requested)...")
    df = df.fillna(0)

    # Chronological Split (Train/Test 80/20)
    print("3. Splitting 80/20 Chronologically by ts_index...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    train_df = df[df['ts_index'] < split_time].copy()
    test_df = df[df['ts_index'] >= split_time].copy()
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # Feature selection
    exclude = ['id', 'y_target', 'weight', 'ts_index']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"Total features: {len(features)}")

    # Prepare data arrays
    X_tr = train_df[features].values
    y_tr = train_df['y_target'].values
    w_tr = train_df['weight'].values
    
    X_te = test_df[features].values
    y_te = test_df['y_target'].values
    w_te = test_df['weight'].values

    # Training
    print("\n4. Starting LightGBM Training (Single Pass)...")
    model = lgb.LGBMRegressor(
        n_estimators=1000, 
        device='cpu', 
        num_leaves=63,
        learning_rate=0.05,
        extra_trees=True,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_te, y_te)],
        eval_sample_weight=[w_te],
        callbacks=[lgb.early_stopping(50, verbose=True)]
    )

    print("\n5. Evaluation Results...")
    tr_preds = model.predict(X_tr)
    te_preds = model.predict(X_te)
    
    tr_score = zero_mean_r2_score(y_tr, tr_preds, w_tr)
    te_score = zero_mean_r2_score(y_te, te_preds, w_te)
    
    print(f"==> TRAIN ZERO-MEAN R2 SCORE: {tr_score:.6f}")
    print(f"==> TEST ZERO-MEAN R2 SCORE:  {te_score:.6f}")
    
    print(f"\nPipeline Finished in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
