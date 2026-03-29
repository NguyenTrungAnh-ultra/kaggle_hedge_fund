import os
import sys

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import lightgbm as lgb
from metric import weighted_rmse_score
from sklearn.preprocessing import LabelEncoder
import warnings
import optuna

warnings.filterwarnings('ignore')

LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'ts-forecasting')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def lgb_weighted_rmse(preds, dataset):
    y_true = dataset.get_label()
    w = dataset.get_weight()
    score = weighted_rmse_score(y_true, preds, w)
    return 'weighted_rmse', score, True


def objective(trial, train_df, features):
    # --- BƯỚC 1: TRÍCH XUẤT THAM SỐ ĐỘNG LỰC HỌC ---
    suggested_max_depth = trial.suggest_int('max_depth', 3, 6)
    max_theoretical_leaves = 2 ** suggested_max_depth
    lower_bound = max(8, int(max_theoretical_leaves * 0.4)) 
    upper_bound = max(lower_bound, int(max_theoretical_leaves * 0.75))
    suggested_num_leaves = trial.suggest_int('num_leaves', lower_bound, upper_bound)

    # --- BƯỚC 2: KHAI BÁO DICTIONARY TIÊU CHUẨN ---
    params = {
        'objective': 'regression',
        'metric': 'None',
        'verbosity': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': suggested_max_depth,
        'num_leaves': suggested_num_leaves,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1000, 5000), 
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True), 
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-1, 50.0, log=True), 
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8), 
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9), 
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]), 
        'device': 'cpu' 
    }
    
    unique_ts = np.sort(train_df['ts_index'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    split_ts = unique_ts[split_idx]

    tr_mask = train_df['ts_index'] < split_ts
    va_mask = train_df['ts_index'] >= split_ts
    
    va_start = train_df[va_mask]['ts_index'].min()
    overlap_mask = (tr_mask) & (train_df['t1_end'] > va_start)
    tr_mask = tr_mask & (~overlap_mask)

    X_tr = train_df[tr_mask][features].values
    y_tr = train_df[tr_mask]['y_target'].values
    w_tr = train_df[tr_mask]['weight'].values

    X_va = train_df[va_mask][features].values
    y_va = train_df[va_mask]['y_target'].values
    w_va = train_df[va_mask]['weight'].values

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

    trial_params = params.copy()
    num_boost_round = trial_params.pop('n_estimators', 100)

    model = lgb.train(
        trial_params,
        dtrain,
        valid_sets=[dval],
        valid_names=['valid'],
        feval=lgb_weighted_rmse,
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    v_preds = model.predict(X_va)
    w_score = weighted_rmse_score(y_va, v_preds, w_va)
    return w_score

def run_lgbm(file_path, tuning=False, training=True):
    print("1. Loading dataset...")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    print(f"Dataset shape: {df.shape}")

    df['t1_end'] = df['ts_index'] + df['horizon']
    
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        if col in df.columns:
            print(f"Label Encoding: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
    
    valid_mask = df['y_target'].notna()
    if not valid_mask.all():
        print(f"Dropping {(~valid_mask).sum()} rows with missing target.")
        df = df[valid_mask].copy().reset_index(drop=True)

    print("\n2. Splitting Hold-out Test (Chronological 80/20)...")
    unique_ts = np.sort(df['ts_index'].unique())
    split_ts_idx = int(len(unique_ts) * 0.80)
    split_time = unique_ts[split_ts_idx]
    
    train_mask = df['ts_index'] < split_time
    test_mask = df['ts_index'] >= split_time
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    test_start_time = test_df['ts_index'].min()
    overlap_mask = train_df['t1_end'] > test_start_time
    purged_count = overlap_mask.sum()
    if purged_count > 0:
        print(f"Purging {purged_count} leaking samples from Train overlapping the Test set...")
        train_df = train_df[~overlap_mask].copy().reset_index(drop=True)
        
    print(f"Final Train shape: {train_df.shape}")
    print(f"Final Test shape : {test_df.shape}")

    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end']
    features = [c for c in df.columns if c not in exclude]
    
    print(f"Scanning {len(features)} features for constant or all-0/all-1 flag columns...")
    const_cols = [col for col in features if df[col].nunique() <= 1]
    if const_cols:
        features = [c for c in features if c not in const_cols]
    print(f"Total modeling features: {len(features)}")

    def tune():
        print("\n3. Starting Optuna Hyperparameter Tuning (using 20% sample + Single Split)...")
        import logging
        optuna_log_file = os.path.join(LOG_DIR, "lgbm_tune_nokfold.log")
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))
        
        unique_ts = np.sort(train_df['ts_index'].unique())
        cut_off_idx = int(len(unique_ts) * 0.8)
        cut_off_time = unique_ts[cut_off_idx]

        train_df_sample = train_df[train_df['ts_index'] >= cut_off_time].copy().reset_index(drop=True)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_df_sample, features), n_trials=20)
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items(): print(f"    {key}: {value}")

        import json
        param_path = os.path.join(PROJECT_ROOT, "models_param", "lgb_best_param_nokfold.json")
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        with open(param_path, 'w') as f:
            json.dump(trial.params, f, indent=4)
        return trial.params

    import json
    param_path = os.path.join(PROJECT_ROOT, 'models_param', 'lgb_best_param_nokfold.json')
    
    if tuning:
        best_params = tune()
    elif os.path.exists(param_path):
        with open(param_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded parameters from {param_path}")
    else:
        print("Running tuning as fallback...")
        best_params = tune()

    best_params.update({
        'objective': 'regression',
        'metric': 'None',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'importance_type': 'gain'
    })

    test_X = test_df[features].values
    test_y = test_df['y_target'].values
    test_w = test_df['weight'].values

    os.makedirs(os.path.join(PROJECT_ROOT, 'models_param'), exist_ok=True)
    
    if training:
        print(f"\n4. Starting Single-Split Training with Best Parameters...")
        
        unique_train_ts = np.sort(train_df['ts_index'].unique())
        split_idx = int(len(unique_train_ts) * 0.8)
        split_ts = unique_train_ts[split_idx]
        
        tr_mask = train_df['ts_index'] < split_ts
        va_mask = train_df['ts_index'] >= split_ts
        va_start = train_df[va_mask]['ts_index'].min()
        overlap_mask = (tr_mask) & (train_df['t1_end'] > va_start)
        tr_mask = tr_mask & (~overlap_mask)
        
        X_tr = train_df[tr_mask][features].values
        y_tr = train_df[tr_mask]['y_target'].values
        w_tr = train_df[tr_mask]['weight'].values
        
        X_va = train_df[va_mask][features].values
        y_va = train_df[va_mask]['y_target'].values
        w_va = train_df[va_mask]['weight'].values
        
        print(f"Train samples: {len(X_tr)} | Val samples: {len(X_va)}")
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

        params = best_params.copy()
        num_boost_round = params.pop('n_estimators', 1000)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            valid_names=['valid'],
            feval=lgb_weighted_rmse,
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        model_filename = os.path.join(PROJECT_ROOT, "models_param", "lgbm_single.txt")
        model.save_model(model_filename)
        print(f"Saved model to {model_filename}")
        
        test_preds = model.predict(test_X)
    else:
        print("\n4. Loading model from models_param...")
        model_filename = os.path.join(PROJECT_ROOT, "models_param", "lgbm_single.txt")
        model = lgb.Booster(model_file=model_filename)
        test_preds = model.predict(test_X)
        
    final_wrmse = weighted_rmse_score(test_y, test_preds, test_w)
    print(f"==> WEIGHTED RMSE SCORE (Test Set): {final_wrmse:.6f}")
    
    imp_df = pd.DataFrame({'feature': features, 'importance': model.feature_importance(importance_type='gain')})
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    report_file = os.path.join(LOG_DIR, 'reports', 'feature_important_nokfold.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write("=== FINAL TEST SET EVALUATION (NO K-FOLD) ===\n")
        f.write(f"Weighted RMSE Score: {final_wrmse:.6f}\n\n")
        f.write("Top 20 Features:\n")
        f.write(imp_df.head(20).to_string(index=False))
        
    print(f"\nReport saved to {report_file}")

if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'train_feature_builded.parquet')
    run_lgbm(file_path=file_path, tuning=1, training=1)
