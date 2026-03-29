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
from Modules.PurgedKFold import LeakFreePurgedKFold
from Pseudo_Huber_Loss import create_pseudo_huber_objective

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
    # Phải lấy giá trị max_depth trước để làm cơ sở toán học tính toán giới hạn cho num_leaves
    suggested_max_depth = trial.suggest_int('max_depth', 3, 6)

    # Dò tìm ngưỡng Delta (Độ lệch chuẩn của y_target khoảng 32.5, nên dò quanh vùng 5-100)
    delta_opt = trial.suggest_float('pseudo_huber_delta', 5.0, 100.0)

    # Tính toán sức chứa tối đa theo lý thuyết
    max_theoretical_leaves = 2 ** suggested_max_depth
    
    # Ép Cây Bất đối xứng (Asymmetric Trees): num_leaves chỉ chiếm 40% - 75% sức chứa
    lower_bound = max(8, int(max_theoretical_leaves * 0.4)) 
    upper_bound = max(lower_bound, int(max_theoretical_leaves * 0.75))
    
    suggested_num_leaves = trial.suggest_int('num_leaves', lower_bound, upper_bound)

    # --- BƯỚC 2: KHAI BÁO DICTIONARY TIÊU CHUẨN ---
    params = {
        'objective': 'None',
        'metric': 'None',
        'verbosity': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        
        # 1. PHANH CẤU TRÚC (STRUCTURAL BRAKES) - Đã được gán giá trị động
        'max_depth': suggested_max_depth,
        'num_leaves': suggested_num_leaves,
        
        # 2. PHANH LÁ & ĐIỀU CHUẨN (LEAF & PENALTY BRAKES)
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1000, 5000), 
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True), 
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-1, 50.0, log=True), 
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-3, 10.0, log=True),
        
        # 3. PHANH LẤY MẪU (SAMPLING BRAKES) 
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8), 
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9), 
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]), 
        'device': 'cpu' 
    }
    
    # ... (Phần Hold-out split bên dưới giữ nguyên) ...

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

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

    trial_params = params.copy()
    num_boost_round = trial_params.pop('n_estimators', 100)

    trial_params['objective'] = create_pseudo_huber_objective(delta=delta_opt)

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
    print(f"    [Trial {trial.number}] Weighted RMSE: {w_score:.6f}")
    
    # Giải phóng bộ nhớ RAM nội bộ Trial
    del X_tr, y_tr, w_tr, X_va, y_va, w_va, dtrain, dval
    
    return w_score

def run_lgbm(file_path, tuning=False, training=True):
    print("1. Loading dataset...")
    # Read the data produced by process_data.py
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

    # Note: Using FULL dataset for Cross-Validation (No Hold-out)
    # Use the existing df, avoid redundant copies
    train_df = df 
    print(f"Final Train shape: {train_df.shape}")

    # Definition of features
    # Mở rộng danh sách loại trừ bằng các tính năng Không Dừng (Failed FracDiff)
    blacklist = ['code', 'sub_code', 'sub_category']
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
        
        # Lấy danh sách các mốc thời gian duy nhất đã được sắp xếp
        unique_ts = np.sort(train_df['ts_index'].unique())

        # Lấy 20% số lượng mốc thời gian GẦN NHẤT (không phải 20% số hàng)
        cut_off_idx = int(len(unique_ts) * 0.8)
        cut_off_time = unique_ts[cut_off_idx]

        # Lọc toàn bộ các hàng thuộc 20% khung thời gian cuối này
        train_df_sample = train_df[train_df['ts_index'] >= cut_off_time].copy().reset_index(drop=True)
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

        # Save best parameters to JSON
        import json
        param_dir = os.path.join(PROJECT_ROOT, "models_param")
        os.makedirs(param_dir, exist_ok=True)
        
        param_path = os.path.join(param_dir, "lgb_best_param.json")
        with open(param_path, 'w') as f:
            json.dump(trial.params, f, indent=4)
        print(f"\nBest parameters saved to {param_path}")

        params = trial.params
        del train_df_sample # Giải phóng bộ nhớ RAM
        return params

    # 3. Handle Hyperparameters: Tune or Load from JSON
    import json
    param_path = os.path.join(PROJECT_ROOT, 'models_param', 'lgb_best_param.json')
    
    if tuning:
        best_params = tune()
    elif os.path.exists(param_path):
        with open(param_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded parameters from {param_path}")
    else:
        print(f"Warning: {param_path} not found and tuning=False. Running tuning as fallback...")
        best_params = tune()

    # Standard system-level parameters
    best_params.update({
        'metric': 'None',
        'verbosity': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'gpu_use_dp': True,
        'importance_type': 'gain'
    })

    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    models = []
    fold_scores = []
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'models_param'), exist_ok=True)
    
    if training:
        print(f"\n\n4. Starting Leak-Free Purged K-Fold Training with Best Parameters...")
        

        X_train_full = train_df[features].values
        y_train_full = train_df['y_target'].values
        w_train_full = train_df['weight'].values
        
        # Keep only essential columns for CV splitting to save memory
        # LeakFreePurgedKFold needs ts_index and t1_end
        cv_meta = train_df[['ts_index', 't1_end']].copy()
        del train_df # Giải phóng bộ nhớ RAM
        del df # Giải phóng bộ nhớ RAM
        
        fold = 1
        for train_idx, val_idx in cv.split(cv_meta):
            print(f"\n--- Fold {fold} ---")
            
            X_tr, y_tr, w_tr = X_train_full[train_idx], y_train_full[train_idx], w_train_full[train_idx]
            X_va, y_va, w_va = X_train_full[val_idx], y_train_full[val_idx], w_train_full[val_idx]
            
            print(f"Inner Train samples: {len(X_tr)} | Inner Val samples: {len(X_va)}")
            
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

            params = best_params.copy()
            num_boost_round = params.pop('n_estimators', 1000)
            delta_opt = params.get('pseudo_huber_delta', 15.0)
            params['objective'] = create_pseudo_huber_objective(delta=delta_opt)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval, dtrain],
                valid_names=['valid', 'train'],
                feval=lgb_weighted_rmse,
                num_boost_round=num_boost_round,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=False),
                    lgb.log_evaluation(period=50)]
            )
            
            # Save model
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"lgbm_fold_{fold}.txt")
            model.save_model(model_filename)
            print(f"Saved model to {model_filename}")
            
            v_preds = model.predict(X_va)
            v_wrmse = weighted_rmse_score(y_va, v_preds, w_va)
            print(f"Fold {fold} Weighted RMSE: {v_wrmse:.5f}")
            fold_scores.append(v_wrmse)
            
            models.append(model)
            
            # Giải phóng bộ nhớ từng Fold
            del X_tr, y_tr, w_tr, X_va, y_va, w_va, dtrain, dval
            fold += 1

        print(f"\n==> AVERAGE CV WEIGHTED RMSE: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
    else:
        print("\n4. Skipping training, loading models from models_param...")
        for fold in range(1, 6):
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"lgbm_fold_{fold}.txt")
            if not os.path.exists(model_filename):
                print(f"Error: Model not found at {model_filename}")
                raise FileNotFoundError(f"Missing model for fold {fold}. Run with training=True first.")
            
            print(f"Loading model: {model_filename}")
            model = lgb.Booster(model_file=model_filename)
            models.append(model)
        
    final_wrmse = np.mean(fold_scores) if fold_scores else 0
    print(f"\n5. Final Assessment (CV Mean Score): {final_wrmse:.6f}")
    
    # Feature importances
    imp_df = pd.DataFrame({'feature': features})
    if training:
        imp_df['importance'] = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    else:
        # For Booster objects, we use feature_importance()
        imp_df['importance'] = np.mean([m.feature_importance(importance_type='gain') for m in models], axis=0)
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    report_file = os.path.join(LOG_DIR, 'reports', 'feature_important.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write("=== FINAL CV EVALUATION ===\n")
        f.write("Method: Leak-Free Purged K-Fold (Full Training Set)\n")
        f.write(f"Average CV Weighted RMSE: {final_wrmse:.6f}\n\n")
        f.write("Note: Models have been saved to 'models_param' directory.\n\n")
        f.write("Top 20 Features:\n")
        f.write(imp_df.to_string(index=False))
        
    print(f"\nReport saved to {report_file}")

if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'train_feature_builded.parquet')
    run_lgbm(file_path=file_path, tuning=0, training=1)
