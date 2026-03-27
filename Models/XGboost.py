import os
import sys

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
import xgboost as xgb
from metric import weighted_rmse_score
from sklearn.preprocessing import LabelEncoder
import warnings
import optuna
from Modules.PurgedKFold import LeakFreePurgedKFold


warnings.filterwarnings('ignore')

LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'ts-forecasting')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


def xgb_weighted_rmse(preds, dtrain):
    y_true = dtrain.get_label()
    w = dtrain.get_weight()
    if w is None or len(w) == 0:
        w = np.ones_like(y_true)
    score = weighted_rmse_score(y_true, preds, w)
    return 'weighted_rmse', float(score)


def objective(trial, train_df, features):
    # Search space
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda', # Using GPU (RTX 5060)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        
        # 1. PHANH CẤU TRÚC (STRUCTURAL BRAKES)
        'max_depth': trial.suggest_int('max_depth', 3, 6), # Ép cây lùn
        
        # 2. PHANH LÁ & ĐIỀU CHUẨN (LEAF & PENALTY BRAKES)
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True), # L1
        'lambda': trial.suggest_float('lambda', 1e-1, 50.0, log=True), # L2
        
        # 3. PHANH LẤY MẪU (SAMPLING BRAKES)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8), # Tương đương feature_fraction
        'subsample': trial.suggest_float('subsample', 0.5, 0.9), # Tương đương bagging_fraction
    }

    # Use a simple Hold-out (last 20% of the sample for validation) to be extremely fast
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

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=features)
    dval = xgb.DMatrix(X_va, label=y_va, weight=w_va, feature_names=features)

    trial_params = params.copy()
    num_boost_round = trial_params.pop('n_estimators', 100)

    model = xgb.train(
        trial_params,
        dtrain,
        evals=[(dval, 'valid')],
        custom_metric=xgb_weighted_rmse,
        num_boost_round=num_boost_round,
        callbacks=[xgb.callback.EarlyStopping(rounds=30, maximize=True, metric_name='weighted_rmse')],
        verbose_eval=False
    )

    v_preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    w_score = weighted_rmse_score(y_va, v_preds, w_va)
    print(f"    [Trial {trial.number}] Weighted RMSE: {w_score:.6f}")
    return w_score


def run_xgb(file_path, tuning=False, training=True):
    print("1. Loading dataset...")
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

    blacklist = ['code', 'sub_code', 'sub_category', 'horizon']
    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end'] + blacklist
    features = [c for c in df.columns if c not in exclude]
    
    print(f"Scanning {len(features)} features for constant or all-0/all-1 flag columns...")
    const_cols = []
    for col in features:
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
        print("\n3. Starting Optuna Hyperparameter Tuning...")
        import logging
        optuna_log_file = os.path.join(LOG_DIR, "xgb_train.log")
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))
        
        unique_ts = np.sort(train_df['ts_index'].unique())
        cut_off_idx = int(len(unique_ts) * 0.8)
        cut_off_time = unique_ts[cut_off_idx]

        train_df_sample = train_df[train_df['ts_index'] >= cut_off_time].copy().reset_index(drop=True)
        train_df_sample = train_df_sample.sort_values('ts_index').reset_index(drop=True)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_df_sample, features), n_trials=500)
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        import json
        param_dir = os.path.join(PROJECT_ROOT, "models_param")
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        
        param_path = os.path.join(param_dir, "xgb_best_param.json")
        with open(param_path, 'w') as f:
            json.dump(trial.params, f, indent=4)
        print(f"\nBest parameters saved to {param_path}")

        return trial.params

    import json
    param_path = os.path.join(PROJECT_ROOT, 'models_param', 'xgb_best_param.json')
    
    if tuning:
        best_params = tune()
    elif os.path.exists(param_path):
        with open(param_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded parameters from {param_path}")
    else:
        print(f"Warning: {param_path} not found and tuning=False. Running tuning as fallback...")
        best_params = tune()

    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda'
    })

    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    test_X = test_df[features].values
    test_y = test_df['y_target'].values
    test_w = test_df['weight'].values
    test_dmatrix = xgb.DMatrix(test_X, feature_names=features)

    models = []
    test_preds = np.zeros(len(test_df))
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'models_param'), exist_ok=True)
    
    if training:
        print("\n\n4. Starting Leak-Free Purged K-Fold Training with Best Parameters...")
        
        X_train_full = train_df[features].values
        y_train_full = train_df['y_target'].values
        w_train_full = train_df['weight'].values
        
        fold = 1
        for train_idx, val_idx in cv.split(train_df):
            print(f"\n--- Fold {fold} ---")
            
            X_tr, y_tr, w_tr = X_train_full[train_idx], y_train_full[train_idx], w_train_full[train_idx]
            X_va, y_va, w_va = X_train_full[val_idx], y_train_full[val_idx], w_train_full[val_idx]
            
            print(f"Inner Train samples: {len(X_tr)} | Inner Val samples: {len(X_va)}")
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=features)
            dval = xgb.DMatrix(X_va, label=y_va, weight=w_va, feature_names=features)

            params = best_params.copy()
            num_boost_round = params.pop('n_estimators', 1000)

            model = xgb.train(
                params,
                dtrain,
                evals=[(dval, 'valid')],
                custom_metric=xgb_weighted_rmse,
                num_boost_round=num_boost_round,
                callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=True, metric_name='weighted_rmse')],
                verbose_eval=False
            )
            
            # Save format supports JSON out of the box for XGBoost > 1.0.0
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"xgb_fold_{fold}.json")
            model.save_model(model_filename)
            print(f"Saved model to {model_filename}")
            
            v_preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            v_wrmse = weighted_rmse_score(y_va, v_preds, w_va)
            print(f"Fold {fold} Weighted RMSE: {v_wrmse:.5f}")
            
            test_preds += model.predict(test_dmatrix, iteration_range=(0, model.best_iteration + 1)) / cv.n_splits
            models.append(model)
            fold += 1
    else:
        print("\n4. Skipping training, loading models from models_param...")
        for fold in range(1, 6):
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"xgb_fold_{fold}.json")
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"Missing model for fold {fold}. Run with training=True first.")
            
            print(f"Loading model: {model_filename}")
            model = xgb.Booster()
            model.load_model(model_filename)
            # best_iteration is saved inside the booster when saved as json
            try:
                test_preds += model.predict(test_dmatrix, iteration_range=(0, model.best_iteration + 1)) / cv.n_splits
            except AttributeError:
                test_preds += model.predict(test_dmatrix) / cv.n_splits
            models.append(model)
        
    print("\n5. Evaluating Ensemble on Test Set...")
    final_wrmse = weighted_rmse_score(test_y, test_preds, test_w)

    print(f"==> WEIGHTED RMSE SCORE (Test Set)       : {final_wrmse:.6f}")
    
    imp_df = pd.DataFrame({'feature': features})
    
    val_imports = [m.get_score(importance_type='gain') for m in models]
    avg_imp = []
    for f in features:
        score = np.mean([imp.get(f, 0) for imp in val_imports])
        avg_imp.append(score)
        
    imp_df['importance'] = avg_imp
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    report_file = os.path.join(LOG_DIR, 'reports', 'xgb_feature_important.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write("=== FINAL TEST SET EVALUATION ===\n")
        f.write("Method: Leak-Free Purged K-Fold (Chronological Train/Test + Validation)\n")
        f.write(f"Weighted RMSE Score: {final_wrmse:.6f}\n\n")
        f.write("Note: Models have been saved to 'models_param' directory.\n\n")
        f.write("Top 20 Features:\n")
        f.write(imp_df.head(20).to_string(index=False))
        
    print(f"\nReport saved to {report_file}")

if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'train_feature_builded.parquet')
    run_xgb(file_path=file_path, tuning=1, training=1)
