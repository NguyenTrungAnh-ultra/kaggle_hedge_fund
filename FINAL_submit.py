import pandas as pd
import numpy as np
import lightgbm as lgb
from Models.metric import weighted_rmse_score
import os
import time
import warnings
import json
from Modules.PurgedKFold import LeakFreePurgedKFold


warnings.filterwarnings('ignore')

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ts-forecasting')
TRAIN_RAW = os.path.join(DATA_DIR, 'train.parquet')
TEST_RAW = os.path.join(DATA_DIR, 'test.parquet')
REPORT_FILE = os.path.join(BASE_DIR, 'reports', 'report_COMPARISON.csv')
PROCESSED_DATA = os.path.join(DATA_DIR, 'inference_processed_full.parquet')
MODELS_DIR = os.path.join(BASE_DIR, 'models_param')
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, 'lgb_best_param.json')
SUBMISSION_FILE = os.path.join(BASE_DIR, 'test_result', 'submission.csv')

def lgb_weighted_rmse(preds, dataset):
    y_true = dataset.get_label()
    w = dataset.get_weight()
    score = weighted_rmse_score(y_true, preds, w)
    return 'weighted_rmse', score, True

def main(retrain=False):
    print(f"=== KAGGLE INFERENCE PIPELINE ===")
    start_time = time.time()
    
    # Phase 1 & 2: Load Preprocessed Data Directly
    print(f"1. Loading preprocessed data from {PROCESSED_DATA}...")
    df = pd.read_parquet(PROCESSED_DATA)
    print(f"Data_shape: {df.shape}")

    print("\n2. Feature Engineering & Set Isolation...")
    # Tính t1_end để dùng cho Purging nếu retrain
    df['t1_end'] = df['ts_index'] + df['horizon']

    # BÂY GIỜ mới tiến hành tách dữ liệu
    train_only = df[df['is_test'] == 0].copy()
    
    print("   Isolating Test Frame for Inference...")
    df = df.sort_values(['ts_index', 'code', 'sub_code', 'horizon']).reset_index(drop=True)
    test_processed = df[df['is_test'] == 1].copy()
    del df # Giải phóng bộ nhớ RAM
            
    print("\n4. Feature Selection & Model Deployment...")
    exclude = ['id', 'y_target', 'weight', 'ts_index', 'is_test', 't1_end']
    features = [c for c in train_only.columns if c not in exclude]
    
    # Drop constant columns
    print(f"Scanning {len(features)} features for constant columns...")
    const_cols = [col for col in features if train_only[col].nunique() <= 1]
            
    if const_cols:
        print(f" > Dropping {len(const_cols)} constant columns.")
        features = [c for c in features if c not in const_cols]
            
    print(f"Matrix Dimension Checked: {len(features)} aligned features.")
    
    models = []
    fold_paths = [os.path.join(MODELS_DIR, f"lgbm_fold_{i}.txt") for i in range(1, 6)]
    ENSEMBLE_EXISTS = all(os.path.exists(p) for p in fold_paths)
    
    if ENSEMBLE_EXISTS and not retrain:
        print(" > Found 5-Fold Ensemble files. Loading models...")
        for p in fold_paths:
            models.append(lgb.Booster(model_file=p))
    elif os.path.exists(BEST_PARAMS_PATH):
        print(f" > Ensemble missing or retrain=True. Retraining...")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        
        cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
        
        X_train_full = train_only[features].values
        y_train_full = train_only['y_target'].values
        w_train_full = train_only['weight'].values
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(train_only), 1):
            print(f"\n   --- Retraining Fold {fold} ---")
            X_tr, y_tr, w_tr = X_train_full[tr_idx], y_train_full[tr_idx], w_train_full[tr_idx]
            X_va, y_va, w_va = X_train_full[va_idx], y_train_full[va_idx], w_train_full[va_idx]
            
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)

            params = best_params.copy()
            params.update({
                'objective': 'regression',
                'metric': 'None',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'device': 'cpu'
            })
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
            
            m_path = fold_paths[fold-1]
            model.save_model(m_path)
            
            v_preds = model.predict(X_va)
            v_wrmse = weighted_rmse_score(y_va, v_preds, w_va)
            print(f"   Fold {fold} Weighted RMSE: {v_wrmse:.5f}")
            models.append(model)
    else:
        print("FATAL: Neither ensemble models nor best parameters found. Cannot proceed.")
        return
            
    X_test = test_processed[features].values
    
    print(f"\n7. Activating Inference Engine (Yielding over {len(X_test)} entities)...")
    preds = np.zeros(len(X_test))
    
    for i, m in enumerate(models, 1):
        print(f" > Deploying Fold {i} model...")
        preds += m.predict(X_test) / len(models)
        
    test_processed['y_target'] = preds
    submission = test_processed[['id', 'y_target']].rename(columns={'y_target': 'prediction'})
    
    print("\n8. Packaging Result...")
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n==> SUCCESS: Submission ready at {SUBMISSION_FILE}")
    print(f"Total Inference Time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main(retrain=1)
