import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import sys
import time

warnings.filterwarnings('ignore')

# --- PATHS & LOGGING SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
INPUT_FILE = os.path.join(DATA_DIR, 'train.parquet')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(LOG_DIR, "nam_execution.log"), "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- OPTIMIZE MEMORY (EXACTLY FROM NOTEBOOK) ---
def reduce_mem_usage(df):
    """ Ép kiểu dữ liệu để giảm RAM """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col not in ['id']: # Bỏ qua cột ID dạng chuỗi
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # Dùng float32 thay vì 16 để tránh lỗi số học
                else:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

# --- COMPETITION METRIC (EXACTLY FROM NOTEBOOK) ---
def comp_metric_lgb(y_pred, data):
    """
    Custom Evaluation Metric cho LightGBM
    Trả về: (tên_metric, giá_trị, is_higher_better)
    """
    y_true = data.get_label()
    w = data.get_weight()
    
    denom = np.sum(w * (y_true ** 2))
    # Tránh chia cho 0
    if denom == 0:
        denom = 1e-15
        
    ratio = np.sum(w * ((y_true - y_pred) ** 2)) / denom
    
    # Clip giá trị trong khoảng [0, 1]
    clipped = np.clip(ratio, 0.0, 1.0)
    
    # Tính score cuối cùng
    val = 1.0 - clipped
    score = np.sqrt(val)
    
    # Điểm càng cao (gần 1) càng tốt
    return 'weighted_score', score, True

def main():
    print("=== STARTING NAM.PY (Notebook logic replication) ===")
    start_time_exec = time.time()
    
    if not os.path.exists(INPUT_FILE):
        print(f"FATAL ERROR: Could not find {INPUT_FILE}")
        return

    print("1. Loading data...")
    df = pd.read_parquet(INPUT_FILE)
    df = reduce_mem_usage(df)

    # Yêu cầu: Điền khuyết bằng 0 cho các cột feature
    num_cols = [col for col in df.columns if col.startswith('feature_')]
    print("2. Filling NaNs with 0...")
    df[num_cols] = df[num_cols].fillna(0)

    # Sắp xếp dữ liệu theo thời gian (BẮT BUỘC để tránh rò rỉ dữ liệu)
    df = df.sort_values('ts_index').reset_index(drop=True)

    # Encode các biến phân loại dạng chuỗi
    categorical_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    print("3. Encoding categorical features...")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Định nghĩa các tập features
    features = [col for col in df.columns if col.startswith('feature_')] + categorical_cols
    target = 'y_target'
    weight_col = 'weight'

    # --- SPLIT ---
    print("4. Splitting 80/20 Chronologically...")
    unique_times = df['ts_index'].unique()
    split_idx = int(len(unique_times) * 0.8)
    split_time = unique_times[split_idx]

    print(f"Splitting data... Train <= {split_time}, Val > {split_time}")
    train_df = df[df['ts_index'] <= split_time]
    val_df = df[df['ts_index'] > split_time]

    X_train, y_train, w_train = train_df[features], train_df[target], train_df[weight_col]
    X_val, y_val, w_val = val_df[features], val_df[target], val_df[weight_col]

    # Khởi tạo LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

    # --- TRAIN ---
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
        'verbose': -1
    }

    print("\n5. Start training model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        feval=comp_metric_lgb,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    print("\n6. Final Evaluation Report")
    best_it = model.best_iteration
    val_score = model.best_score['valid_1']['weighted_score']
    
    # Calculate simple Zero-Mean R2 for comparison with previous runs
    # (The notebook's weighted_score is sqrt(1 - ratio), so val^2 = 1 - ratio)
    zero_mean_r2 = val_score**2
    
    print(f"Best Iteration: {best_it}")
    print(f"Validation Weighted Score (sqrt(1-ratio)): {val_score:.6f}")
    print(f"Equivalent Zero-Mean R2 (1-ratio): {zero_mean_r2:.6f}")
    
    print(f"\nExecution finished in {time.time() - start_time_exec:.2f}s")

if __name__ == '__main__':
    main()
