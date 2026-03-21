import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import time
import warnings
import json
from Modules.Preprocess.inference_preprocess import InferencePreprocess
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

# Removed redundant local definitions and replaced with Fractional_Differencer module

def main(retrain=False):
    print(f"=== KAGGLE INFERENCE PIPELINE ===")
    start_time = time.time()
    
    # Phase 1 & 2: Load Preprocessed Data Directly (skipping heavy full_pipeline for faster inference)
    print(f"1. Loading preprocessed data from {PROCESSED_DATA}...")
    df = pd.read_parquet(PROCESSED_DATA)
    # df, features = InferencePreprocess.full_pipeline(TRAIN_RAW, TEST_RAW, REPORT_FILE)
    
    # GLOBAL Feature Engineering (Must apply to both Train and Test)
    print("\n4. Feature Engineering & Set Isolation...")
    # Tính t1_end để dùng cho Purging nếu retrain, nhưng sẽ bị loại khỏi features
    df['t1_end'] = df['ts_index'] + df['horizon']

    # Isolate Train set for Categorical Mapping and potential Retraining
    train_only = df[df['is_test'] == 0].copy()
    
    # Extract only the test rows out
    print("   Isolating Test Frame for Inference...")
    df = df.sort_values(['ts_index', 'code', 'sub_code', 'horizon']).reset_index(drop=True)
    test_processed = df[df['is_test'] == 1].copy()
    del df # Purge mem
    
    print("\n5. Categorical Encodings...")
    cat_cols = ['code', 'sub_code', 'sub_category']
    # Safely load the categorical boundaries directly from training labels to avoid Unknown Label crash
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train_only[col].astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        # Apply to both sets
        test_processed[col] = test_processed[col].astype(str).map(mapping).fillna(-1).astype(int)
        train_only[col] = train_only[col].astype(str).map(mapping).fillna(-1).astype(int)
            
    print("\n6. Feature Selection & Model Deployment...")
    # Cần đồng bộ exclude features giống lúc train
    blacklist = ['feature_k', 'feature_aq']
    exclude = ['id', 'y_target', 'weight', 'ts_index', 'is_test', 't1_end'] + blacklist
    features = [c for c in train_only.columns if c not in exclude]
    
    # Đồng bộ logic drop constant columns từ lightgbm.py
    print(f"Scanning {len(features)} features for constant columns (fitting on train)...")
    const_cols = []
    for col in features:
        if train_only[col].nunique() <= 1:
            const_cols.append(col)
            
    if const_cols:
        print(f" > Dropping {len(const_cols)} constant columns identified in training data.")
        features = [c for c in features if c not in const_cols]
            
    print(f"Matrix Dimension Checked: {len(features)} aligned features.")
    
    models = []
    
    # Define filenames for all folds
    fold_paths = [os.path.join(MODELS_DIR, f"lgbm_fold_{i}.txt") for i in range(1, 6)]
    ENSEMBLE_EXISTS = all(os.path.exists(p) for p in fold_paths)
    
    if ENSEMBLE_EXISTS and not retrain:
        print(" > Found 5-Fold Ensemble files. Loading models...")
        for p in fold_paths:
            # Sửa lỗi lgb.Booster bị ghi đè bởi tham số hàm
            models.append(lgb.Booster(model_file=p))
    elif os.path.exists(BEST_PARAMS_PATH):
        print(f" > Ensemble missing or retrain=True. Retraining 5-Fold Ensemble with Purged-CV and Embargo...")
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        
        # System parameters
        params = best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'device': 'cpu' # Recommended for inference scripts to avoid GPU-specific driver errors
        })
        n_estimators = params.pop('n_estimators', 500)
        
        # Prepare data and CV
        cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
        
        X_train_full = train_only[features].values
        y_train_full = train_only['y_target'].values
        w_train_full = train_only['weight'].values
        
        print(f"   Data Shape for Training: {X_train_full.shape}")
        
        # Identify indices of categorical features for LightGBM (needed because we use numpy arrays)
        cat_indices = [i for i, col in enumerate(features) if col in cat_cols]
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(train_only), 1):
            print(f"\n   --- Retraining Fold {fold} ---")
            X_tr, y_tr, w_tr = X_train_full[tr_idx], y_train_full[tr_idx], w_train_full[tr_idx]
            X_va, y_va, w_va = X_train_full[va_idx], y_train_full[va_idx], w_train_full[va_idx]
            
            print(f"   Train: {len(X_tr)} samples | Val: {len(X_va)} samples (Purged/Embargoed)")
            
            model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_sample_weight=[w_va],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
                categorical_feature=cat_indices
            )
            
            # Save Fold
            m_path = os.path.join(MODELS_DIR, f"lgbm_fold_{fold}.txt")
            model.booster_.save_model(m_path)
            print(f"   SUCCESS: Model saved to {m_path}")
            models.append(model.booster_)
    else:
        print("FATAL: Neither ensemble models nor best parameters found. Cannot proceed.")
        return
            
    X_test = test_processed[features].values
    
    print(f"\n7. Activating Inference Engine (Yielding over {len(X_test)} entities)...")
    preds = np.zeros(len(X_test))
    for i, m in enumerate(models, 1):
        print(f" > Fold {i} weights deployed.")
        preds += m.predict(X_test) / len(models)
        
    test_processed['y_target'] = preds
    submission = test_processed[['id', 'y_target']]
    
    print("\n8. Packaging Result...")
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n==> SUCCESS: Submission ready at {SUBMISSION_FILE}")
    print(f"Total Inference Time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main(retrain=True)
