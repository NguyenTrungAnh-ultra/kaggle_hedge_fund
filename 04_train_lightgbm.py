import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import os
import warnings
import sys
import optuna

warnings.filterwarnings('ignore')

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- KAGGLE COMPETITION METRIC ---
def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def zero_mean_r2_score(y_target, y_pred, w) -> float:
    """
    Standard competition metric for typical 0-mean targets (like Jane Street).
    R2 = 1 - sum(w * (y - y_hat)^2) / sum(w * y^2)
    No sqrt is applied.
    """
    denom = np.sum(w * y_target ** 2)
    if denom == 0:
        return 0.0
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
        """
        df must contain 'ts_index' and 't1_end'
        """
        unique_ts = np.sort(df['ts_index'].unique())
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        embargo = int(len(unique_ts) * self.pctEmbargo)
        
        # Use underlying ndarrays for fast masking
        ts_index_arr = df['ts_index'].values
        t1_arr = df['t1_end'].values
        
        for train_ts_idx, test_ts_idx in kf.split(unique_ts):
            train_times = unique_ts[train_ts_idx]
            test_times = unique_ts[test_ts_idx]
            
            t0_test = test_times[0]
            t1_test = df[df['ts_index'].isin(test_times)]['t1_end'].max()
            
            # Initial boundaries
            base_train_mask = np.isin(ts_index_arr, train_times)
            test_indices = np.where(np.isin(ts_index_arr, test_times))[0]
            
            # 1. Purging (Loại bỏ rò rỉ phía trước và bao trùm)
            # Giữ lại nếu nhãn train kết thúc TRƯỚC test bắt đầu
            # HOẶC mẫu train bắt đầu SAU test kết thúc
            keep_before = t1_arr <= t0_test
            keep_after = ts_index_arr > t1_test
            
            # 2. Embargoing (Cấm vận phía sau)
            # Cấm các thời điểm train ngay sau t1_test
            if embargo > 0:
                after_test_times = unique_ts[unique_ts > t1_test]
                if len(after_test_times) > 0:
                    embargoed_times = after_test_times[:embargo]
                    # Loại bỏ các thời điểm trong embargo
                    keep_after = keep_after & (~np.isin(ts_index_arr, embargoed_times))
                    
            final_train_mask = base_train_mask & (keep_before | keep_after)
            train_indices = np.where(final_train_mask)[0]
            
            yield train_indices, test_indices

def objective(trial, train_df, features):
    # Search space
    param = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_use_dp': True,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 50, 5000),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 500, 3000),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-4, 0.1, log=True),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
    }

    # Use a simple Hold-out (last 20% of the sample for validation) to be extremely fast
    # train_df is already sorted by time (ts_index)
    unique_ts = np.sort(train_df['ts_index'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    split_ts = unique_ts[split_idx]

    tr_mask = train_df['ts_index'] < split_ts
    va_mask = train_df['ts_index'] >= split_ts
    
    # Purge the gap in-between
    va_start = train_df[va_mask]['ts_index'].min()
    overlap_mask = (tr_mask) & (train_df['t1_end'] > va_start)
    tr_mask = tr_mask & (~overlap_mask)

    X_tr = train_df[tr_mask][features].values
    y_tr = train_df[tr_mask]['y_target'].values
    w_tr = train_df[tr_mask]['weight'].values

    X_va = train_df[va_mask][features].values
    y_va = train_df[va_mask]['y_target'].values
    w_va = train_df[va_mask]['weight'].values

    print(f"    [Trial {trial.number}] Training with {len(X_tr)} samples, Val with {len(X_va)}...")

    model = lgb.LGBMRegressor(**param)
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_sample_weight=[w_va],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    v_preds = model.predict(X_va)
    score = zero_mean_r2_score(y_va, v_preds, w_va)
    print(f"    [Trial {trial.number}] Validation R2: {score:.6f}")
    return score

def main():
    print("1. Loading dataset...")
    # Read the data produced by process_data.py
    file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    print(f"Dataset shape: {df.shape}")

    # Prepare t1 (Label End Time) column
    df['t1_end'] = df['ts_index'] + df['horizon']
    
    # Handling Categorical Columns with Label Encoding
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        if col in df.columns:
            print(f"Label Encoding: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Sort firmly by time to ensure absolute order
    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
    
    # Check NA in target
    valid_mask = df['y_target'].notna()
    if not valid_mask.all():
        print(f"Dropping {(~valid_mask).sum()} rows with missing target.")
        df = df[valid_mask].copy().reset_index(drop=True)

    # Note: Chronological Split (Train/Test 80/20)
    print("\n2. Splitting Hold-out Test (Chronological 80/20)...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    train_mask = df['ts_index'] < split_time
    test_mask = df['ts_index'] >= split_time
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    test_start_time = test_df['ts_index'].min()
    
    # Purge the gap between train and test
    overlap_mask = train_df['t1_end'] > test_start_time
    purged_count = overlap_mask.sum()
    if purged_count > 0:
        print(f"Purging {purged_count} leaking samples from Train overlapping the Test set...")
        train_df = train_df[~overlap_mask].copy().reset_index(drop=True)
        
    print(f"Final Train shape: {train_df.shape}")
    print(f"Final Test shape : {test_df.shape}")

    # Definition of features
    # Mở rộng danh sách loại trừ bằng các tính năng Không Dừng (Failed FracDiff)
    # blacklist = ['feature_h', 'feature_j', 'feature_l', 'feature_ah', 'feature_an', 'feature_bd', 
    #             'feature_be', 'feature_bf', 'feature_bh', 'feature_bm', 'feature_bn', 'feature_bo', 
    #             'feature_bq', 'feature_br', 'feature_by', 'feature_bz', 'feature_ca', 'feature_cb', 
    #             'feature_cc', 'feature_cd', 'feature_bg', 'feature_bp']

    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end'] 
    features = [c for c in df.columns if c not in exclude]
    
    # Drop constant columns which can cause GPU split errors
    # or columns that are all 0 or all 1 (specifically flag columns).
    print(f"Scanning {len(features)} features for constant or all-0/all-1 flag columns...")
    const_cols = []
    for col in features:
        # Fast check for constant columns (nunique=1 handles all-0, all-1, or any constant value)
        if df[col].nunique() <= 1:
            const_cols.append(col)
            
    if const_cols:
        flag_const = [c for c in const_cols if 'is_nan' in c]
        if flag_const:
            print(f"Dropping {len(flag_const)} constant flag columns (all-0 or all-1).")
        
        other_const = [c for c in const_cols if 'is_nan' not in c]
        if other_const:
            print(f"Dropping {len(other_const)} other constant columns.")
            
        features = [c for c in features if c not in const_cols]
        
    print(f"Total modeling features: {len(features)}")

    def tune():
        print("\n3. Starting Optuna Hyperparameter Tuning (using 20% sample + Single Split for extreme speed)...")
        # Add Optuna log handler to write to our log file
        import logging
        optuna_log_file = os.path.join(LOG_DIR, "lightgbm_train.log")
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))
        
        # Sample 20% of data for tuning speed
        train_df_sample = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        # Ensure it's sorted by time for the split logic
        train_df_sample = train_df_sample.sort_values('ts_index').reset_index(drop=True)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_df_sample, features), n_trials=20)
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return trial.params

    # 3. Use given Best Hyperparameters (Skipping Optuna)
    best_params = {
        'learning_rate': 0.017495140071357364,
        'n_estimators': 703,
        'num_leaves': 111,
        'max_depth': 9,
        'min_child_samples': 1707,
        'feature_fraction': 0.6691779681001662,
        'bagging_fraction': 0.700543212951226,
        'bagging_freq': 4,
        'lambda_l1': 7.808594229512663e-06,
        'lambda_l2': 8.799790407946162e-08,
        'min_data_in_leaf': 777,
        'min_sum_hessian_in_leaf': 0.0030318444826433625,
        'extra_trees': False
    }
    # Ensure standard parameters are included
    best_params.update({
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'gpu',
        'gpu_use_dp': True,
        'importance_type': 'gain'
    })

    print("\n4. Starting Leak-Free Purged K-Fold Training with Best Parameters...")
    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    X_train_full = train_df[features].values
    y_train_full = train_df['y_target'].values
    w_train_full = train_df['weight'].values
    
    test_X = test_df[features].values
    test_y = test_df['y_target'].values
    test_w = test_df['weight'].values

    models = []
    test_preds = np.zeros(len(test_df))
    
    os.makedirs('models_param', exist_ok=True)
    
    fold = 1
    for train_idx, val_idx in cv.split(train_df):
        print(f"\n--- Fold {fold} ---")
        
        X_tr, y_tr, w_tr = X_train_full[train_idx], y_train_full[train_idx], w_train_full[train_idx]
        X_va, y_va, w_va = X_train_full[val_idx], y_train_full[val_idx], w_train_full[val_idx]
        
        print(f"Inner Train samples: {len(X_tr)} | Inner Val samples: {len(X_va)}")
        print(f"Purged/Embargoed: {len(train_df) - len(X_tr) - len(X_va)}")
        
        model = lgb.LGBMRegressor(**best_params)
            
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[w_va],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        model_filename = f"models_param/lgbm_fold_{fold}.txt"
        model.booster_.save_model(model_filename)
        print(f"Saved model to {model_filename}")
        
        v_preds = model.predict(X_va)
        v_score = zero_mean_r2_score(y_va, v_preds, w_va)
        print(f"Fold {fold} Validation R2: {v_score:.5f}")
        
        test_preds += model.predict(test_X) / cv.n_splits
        models.append(model)
        fold += 1
        
    print("\n5. Evaluating Ensemble on Test Set...")
    final_score = zero_mean_r2_score(test_y, test_preds, test_w)
    
    print(f"==> KAGGLE ZERO-MEAN R2 SCORE (Test Set): {final_score:.6f}")
    
    # Feature importances
    imp_df = pd.DataFrame({'feature': features})
    imp_df['importance'] = np.mean([m.feature_importances_ for m in models], axis=0)
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    report_file = os.path.join(BASE_DIR, 'reports', 'test_evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write("=== FINAL TEST SET EVALUATION ===\n")
        f.write("Method: Leak-Free Purged K-Fold (Chronological Train/Test + Validation)\n")
        f.write(f"Kaggle Zero-Mean R2 Score: {final_score:.6f}\n\n")
        f.write("Note: Models have been saved to 'models_param' directory.\n\n")
        f.write("Top 20 Features:\n")
        f.write(imp_df.to_string(index=False))
        
    print(f"\nReport saved to {report_file}")

if __name__ == '__main__':
    main()
