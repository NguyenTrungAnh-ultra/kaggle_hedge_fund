import os
import sys

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from metric import weighted_rmse_score
import warnings
import optuna
from Modules.PurgedKFold import LeakFreePurgedKFold

warnings.filterwarnings('ignore')

LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'ts-forecasting')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def objective(trial, train_df, features, cat_indices):
    suggested_max_depth = trial.suggest_int('depth', 4, 8) # Sâu hơn để khai thác SymmetricTree
    suggested_border_count = trial.suggest_categorical('border_count', [128, 254])

    params = {
        'loss_function': 'RMSE', 
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'score_function': 'NewtonL2', 
        'border_count': suggested_border_count,
        'depth': suggested_max_depth,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.02, log=True), 
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5.0, 200.0, log=True), 
        'random_strength': trial.suggest_float('random_strength', 1.0, 50.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0), 
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2000, 15000), 
        'max_ctr_complexity': 2,
        'verbose': False,
        'random_seed': 42,
        'bootstrap_type': 'Bayesian',
        'one_hot_max_size': 10
    }

    # Hold-out
    unique_ts = np.sort(train_df['ts_index'].unique())
    split_idx = int(len(unique_ts) * 0.8)
    split_ts = unique_ts[split_idx]

    tr_mask = train_df['ts_index'] < split_ts
    va_mask = train_df['ts_index'] >= split_ts
    va_start = train_df[va_mask]['ts_index'].min()
    overlap_mask = (tr_mask) & (train_df['t1_end'] > va_start)
    tr_mask = tr_mask & (~overlap_mask)

    # Clipping & Scaling
    TARGET_SCALE = 1e5
    y_raw = train_df['y_target'].values
    lower_limit = np.percentile(y_raw, 1.0) # Clip mạnh tay hơn (1%)
    upper_limit = np.percentile(y_raw, 99.0)
    
    X_tr = train_df[tr_mask][features]
    y_tr = np.clip(train_df[tr_mask]['y_target'].values, lower_limit, upper_limit) * TARGET_SCALE
    w_tr = train_df[tr_mask]['weight'].values

    X_va = train_df[va_mask][features]
    y_va_raw = train_df[va_mask]['y_target'].values
    y_va_scaled = np.clip(y_va_raw, lower_limit, upper_limit) * TARGET_SCALE
    w_va = train_df[va_mask]['weight'].values

    print(f"    [Trial {trial.number}] Depth={suggested_max_depth}, LR={params['learning_rate']:.4f}, L2={params['l2_leaf_reg']:.1f}")
    
    train_pool = Pool(X_tr, label=y_tr, weight=w_tr, cat_features=cat_indices)
    val_pool = Pool(X_va, label=y_va_scaled, weight=w_va, cat_features=cat_indices)

    model = CatBoostRegressor(
        iterations=5000, 
        early_stopping_rounds=200,
        **params
    )
    
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    v_preds_raw = model.predict(val_pool) / TARGET_SCALE
    w_score = weighted_rmse_score(y_va_raw, v_preds_raw, w_va)
    print(f"    [Trial {trial.number}] Competition Score (R): {w_score:.6f} | Best Iter: {model.get_best_iteration()}")
    
    del X_tr, y_tr, w_tr, X_va, y_va_scaled, w_va, train_pool, val_pool
    return w_score

def run_cb(file_path, tuning=False, training=True):
    print("1. Loading dataset...")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return
        
    df = pd.read_parquet(file_path)
    df['t1_end'] = df['ts_index'] + df['horizon']
    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
    
    valid_mask = df['y_target'].notna()
    if not valid_mask.all():
        df = df[valid_mask].copy().reset_index(drop=True)

    train_df = df 

    cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end']
    features = [c for c in df.columns if c not in exclude]
    
    cat_cols = [c for c in cat_cols if c in features]
    cat_indices = [features.index(c) for c in cat_cols]

    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)

    const_cols = [col for col in features if df[col].nunique() <= 1]
    if const_cols:
        features = [c for c in features if c not in const_cols]
        cat_indices = [features.index(c) for c in cat_cols if c in features]
        
    print(f"Total modeling features: {len(features)}")

    def tune():
        print("\n3. Starting Optuna Hyperparameter Tuning (Institutional Mode)...")
        import logging
        optuna_log_file = os.path.join(LOG_DIR, "catboost_train.log")
        optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))
        
        unique_ts = np.sort(train_df['ts_index'].unique())
        cut_off_idx = int(len(unique_ts) * 0.8)
        cut_off_time = unique_ts[cut_off_idx]

        train_df_sample = train_df[train_df['ts_index'] >= cut_off_time].copy().reset_index(drop=True)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, train_df_sample, features, cat_indices), n_trials=20)
        
        param_dir = os.path.join(PROJECT_ROOT, "models_param")
        os.makedirs(param_dir, exist_ok=True)
        param_path = os.path.join(param_dir, "cb_best_param.json")
        import json
        with open(param_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        return study.best_params

    import json
    param_path = os.path.join(PROJECT_ROOT, 'models_param', 'cb_best_param.json')
    
    if tuning:
        best_params = tune()
    elif os.path.exists(param_path):
        with open(param_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = tune()

    suggested_depth = best_params.pop('depth', 6)
    suggested_border_count = best_params.pop('border_count', 254)
    # Clean up old params from JSON if they exist
    if 'num_leaves' in best_params: best_params.pop('num_leaves')
    if 'subsample' in best_params: best_params.pop('subsample')

    final_params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'score_function': 'NewtonL2',
        'border_count': suggested_border_count,
        'depth': suggested_depth,
        'bootstrap_type': 'Bayesian',
        'max_ctr_complexity': 2,
        'verbose': False,
        'one_hot_max_size': 10,
        **best_params
    }

    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    models = []
    fold_scores = []
    
    os.makedirs(os.path.join(PROJECT_ROOT, 'models_param'), exist_ok=True)
    
    if training:
        print(f"\n\n4. Starting Leak-Free Purged K-Fold Training (Institutional Mode)...")
        TARGET_SCALE = 1e5
        y_raw_full = train_df['y_target'].values
        lower_limit = np.percentile(y_raw_full, 1.0)
        upper_limit = np.percentile(y_raw_full, 99.0)
        
        X_train_full = train_df[features]
        y_train_full = train_df['y_target'].values
        w_train_full = train_df['weight'].values
        cv_meta = train_df[['ts_index', 't1_end']].copy()
        
        del train_df
        del df

        fold = 1
        for train_idx, val_idx in cv.split(cv_meta):
            print(f"\n--- Fold {fold} ---")
            
            X_tr, y_tr_raw, w_tr = X_train_full.iloc[train_idx], y_train_full[train_idx], w_train_full[train_idx]
            X_va, y_va_raw, w_va = X_train_full.iloc[val_idx], y_train_full[val_idx], w_train_full[val_idx]
            
            print(f"Inner Train samples: {len(X_tr)} | Inner Val samples: {len(X_va)}")
            
            y_tr = np.clip(y_tr_raw, lower_limit, upper_limit) * TARGET_SCALE
            y_va_scaled = np.clip(y_va_raw, lower_limit, upper_limit) * TARGET_SCALE
            
            train_pool = Pool(X_tr, label=y_tr, weight=w_tr, cat_features=cat_indices)
            val_pool = Pool(X_va, label=y_va_scaled, weight=w_va, cat_features=cat_indices)

            model = CatBoostRegressor(
                iterations=8000, 
                early_stopping_rounds=300, # Phóng khoáng hơn với early stopping rãnh rỗi
                **final_params
            )
            
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=200)
            
            # Save model
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"cb_fold_{fold}.cbm")
            model.save_model(model_filename)
            print(f"Saved model to {model_filename}")
            
            v_preds = model.predict(val_pool) / TARGET_SCALE 
            v_score = weighted_rmse_score(y_va_raw, v_preds, w_va)
            print(f"Fold {fold} Competition Score (R): {v_score:.6f} | Best Iter: {model.get_best_iteration()}")
            fold_scores.append(v_score)
            
            models.append(model)
            del X_tr, y_tr, w_tr, X_va, y_va_raw, w_va, train_pool, val_pool
            fold += 1

        print(f"\n==> AVERAGE CV COMPETITION SCORE: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
    else:
        print("\n4. Skipping training, loading models from models_param...")
        for fold in range(1, 6):
            model_filename = os.path.join(PROJECT_ROOT, "models_param", f"cb_fold_{fold}.cbm")
            if not os.path.exists(model_filename):
                print(f"Error: Model not found at {model_filename}")
                raise FileNotFoundError(f"Missing model for fold {fold}. Run with training=True first.")
            
            print(f"Loading model: {model_filename}")
            model = CatBoostRegressor()
            model.load_model(model_filename)
            models.append(model)
        
    final_score = np.mean(fold_scores) if fold_scores else 0
    print(f"\n5. Final Assessment (CV Mean Score): {final_score:.6f}")
    
    # Feature importances
    imp_df = pd.DataFrame({'feature': features})
    imp_df['importance'] = np.mean([m.get_feature_importance() for m in models], axis=0)
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    report_file = os.path.join(LOG_DIR, 'reports', 'feature_important_cb.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        f.write("=== FINAL CV EVALUATION ===\n")
        f.write("Method: Leak-Free Purged K-Fold (Full Training Set)\n")
        f.write(f"Average CV Competition Score (R): {final_score:.6f}\n\n")
        f.write("Note: Models have been saved to 'models_param' directory.\n\n")
        f.write("Top 20 Features:\n")
        f.write(imp_df.to_string(index=False))
        
    print(f"\nReport saved to {report_file}")

if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'train_feature_builded.parquet')
    run_cb(file_path=file_path, tuning=0, training=1)
