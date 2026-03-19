import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import time
import sys
import warnings

warnings.filterwarnings('ignore')

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train.parquet')
AUTOGLUON_OUT = os.path.join(BASE_DIR, 'ag_models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(LOG_DIR, "autogluon_train.log"), "a", encoding='utf-8')

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
    print("=== AUTOGLUON 80/20 TRAIN PIPELINE (FILLNA=0) ===")
    start_time = time.time()

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"1. Loading RAW dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Raw dataset shape: {df.shape}")

    # No manual Label Encoding needed for AutoGluon, but we'll do Simple Fill NA = 0
    print("2. Filling ALL NaNs with 0...")
    df = df.fillna(0)

    # Chronological Split (Train/Test 80/20)
    print("3. Splitting 80/20 Chronologically by ts_index...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    train_df = df[df['ts_index'] < split_time].copy()
    test_df = df[df['ts_index'] >= split_time].copy()
    
    # AutoGluon handles Categorical, but let's drop 'id'
    train_df = train_df.drop(columns=['id', 'ts_index'])
    test_actual_y = test_df['y_target'].values
    test_actual_w = test_df['weight'].values
    test_df_for_ag = test_df.drop(columns=['id', 'ts_index']) # Keep y_target for internal eval if needed

    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # AutoGluon TabularPredictor
    print("\n4. Starting AutoGluon Tabular Training (Medium quality, time limited)...")
    time_limit = 600 # Total search time limit: 10 mins
    
    predictor = TabularPredictor(
        label='y_target', 
        path=AUTOGLUON_OUT,
        eval_metric='r2',
        sample_weight='weight' # AutoGluon supports sample weights
    ).fit(
        train_df,
        time_limit=time_limit,
        presets='medium_quality'
    )

    print("\n5. Evaluation Results...")
    # Leaderboard
    print("\nLeaderboard on Train data:")
    print(predictor.leaderboard(train_df, silent=True).to_string())

    # Final Evaluation for Zero-Mean R2
    print("\nCalculating Zero-Mean R2 on Test Data...")
    y_preds = predictor.predict(test_df_for_ag)
    
    final_te_r2_score = zero_mean_r2_score(test_actual_y, y_preds.values, test_actual_w)
    print(f"\n==> AUTOGLUON TEST ZERO-MEAN R2 SCORE: {final_te_r2_score:.6f}")
    
    print(f"\nPipeline Finished in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
