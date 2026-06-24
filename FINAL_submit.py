import sys
import os
import gc
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ====================================================================
# CONFIGURATION
# ====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / 'models_param'
TEST_DIR = PROJECT_ROOT / 'test_result'
os.makedirs(TEST_DIR, exist_ok=True)

HORIZONS = [1, 3, 10, 25]
SEEDS = [42, 123, 7, 2024, 999]
SIGN_SPLIT_FEATURES = ['feature_n', 'feature_w', 'feature_x', 'feature_y', 'feature_z', 'feature_al', 'feature_ce']
LAG_FEATURES = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
LAGS = [1, 3, 5, 10, 25]

from Models.Lightgbm import (
    apply_advanced_features,
    make_recency_weights,
    compute_target_encoding,
    apply_target_encoding,
    select_top_features,
    LGB_PARAMS_BASE
)

def train_final_full_models():
    print("\n=== RETRAINING MODELS ON 100% TRAINING DATA (FULL DATASET) ===")
    
    train_path = DATA_DIR / 'ts-forecasting' / 'train_feature_builded.parquet'
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    
    print("   -> Applying Causal Insights (Lags only) on FULL dataset...")
    train_df = apply_advanced_features(train_df)
    
    print("   -> Optimizing data types to prevent RAM overflow...")
    exclude_cols = ['y_target', 'weight']
    cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    
    for col in train_df.columns:
        if col in exclude_cols:
            continue
        if col in cat_cols:
            train_df[col] = train_df[col].astype('category')
        elif col.startswith('is_nan_'):
            train_df[col] = train_df[col].astype(np.int8)
            
    # Filter valid targets
    valid_mask = train_df['y_target'].notna()
    if not valid_mask.all():
        train_df = train_df[valid_mask].copy().reset_index(drop=True)
        
    train_df = train_df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
    
    # Calibrated trees per horizon from author's CV
    N_EST_PER_HORIZON = {1: 134, 3: 157, 10: 116, 25: 215}
    
    for h in HORIZONS:
        print(f"\n >> Training Horizon {h} on 100% data...")
        h_mask = train_df['horizon'] == h
        h_train = train_df[h_mask].copy().reset_index(drop=True)
        
        # Apply target encoding
        enc_maps, global_mean = compute_target_encoding(h_train)
        h_train = apply_target_encoding(h_train, enc_maps, global_mean)
        
        # Feature Selection
        exclude_feats = ['id', 'y_target', 'weight', 'ts_index', 't1_end', 'horizon', 'code', 'sub_code', 'sub_category']
        candidate_features = [c for c in h_train.columns if c not in exclude_feats and 'logret' not in c]
        
        top_features = select_top_features(h_train, candidate_features, n_features=50)
        
        # Final features list
        must_keep = ['code', 'sub_code', 'sub_category', 'sub_code_target_enc', 'sub_category_target_enc']
        h_features = list(dict.fromkeys([c for c in must_keep if c in h_train.columns] + top_features))
        
        cat_indices = [h_features.index(c) for c in ['code', 'sub_code', 'sub_category'] if c in h_features]
        
        # Target clipping
        y_all = h_train['y_target'].values.copy()
        lo_val = np.percentile(y_all, 5)
        hi_val = np.percentile(y_all, 95)
        y_all_clipped = np.clip(y_all, lo_val, hi_val)
        
        # Recency weights
        ts_all = h_train['ts_index']
        w_all = h_train['weight'].values.copy()
        w_all = w_all * make_recency_weights(ts_all, half_life_frac=0.5)
        
        n_est = N_EST_PER_HORIZON[h]
        
        for seed in SEEDS:
            print(f"    - Seed {seed}...")
            params = LGB_PARAMS_BASE.copy()
            params['random_state'] = seed
            params['n_estimators'] = n_est
            
            dtrain = lgb.Dataset(
                h_train[h_features], label=y_all_clipped, weight=w_all, categorical_feature=cat_indices
            )
            
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=n_est
            )
            
            # Save final full model
            model_name = MODELS_DIR / f'lgbm_h{h}_full_seed{seed}.txt'
            model.save_model(model_name)
            
            del dtrain, model
            gc.collect()
            
    print("=== FINISHED TRAINING FULL MODELS ===")

def main(retrain=False):
    print("=== KAGGLE INFERENCE PIPELINE (0.2 ADVANCED ARCHITECTURE) ===")
    
    expected_models = len(HORIZONS) * len(SEEDS)  # 4 horizons * 5 seeds = 20 models
    found_models = glob.glob(str(MODELS_DIR / 'lgbm_h*_full_seed*.txt'))
    
    if len(found_models) < expected_models or retrain:
        print(f" > Found {len(found_models)}/{expected_models} models. Retraining is required!")
        train_final_full_models()
    
    print("\n1. Loading preprocessed data from inference_processed_full.parquet...")
    test_df = pd.read_parquet(DATA_DIR / 'ts-forecasting' / 'inference_processed_full.parquet')
    print(f"Data_shape: {test_df.shape}")

    # Generate the Causal Insights Features across the FULL test dataset BEFORE filtering
    test_df = apply_advanced_features(test_df)

    print("\n2. Isolating Test Frame for Inference...")
    test_processed = test_df[test_df['is_test'] == 1].copy()
    
    print("   -> Computing target encoding maps from training set...")
    train_path = DATA_DIR / 'ts-forecasting' / 'train_feature_builded.parquet'
    train_df = pd.read_parquet(train_path, columns=['sub_code', 'sub_category', 'y_target'])
    enc_maps, global_mean = compute_target_encoding(train_df)
    test_processed = apply_target_encoding(test_processed, enc_maps, global_mean)
    
    del test_df, train_df
    gc.collect()
    
    cat_cols = ['code', 'sub_code', 'sub_category']
    for col in cat_cols:
        if col in test_processed.columns:
            test_processed[col] = test_processed[col].astype('category')
            
    print("\n4. Activating 5-Seed Inference Engine (Full Data Models)...")
    preds = np.zeros(len(test_processed))
    
    for h in HORIZONS:
        print(f" > Processing Horizon {h}...")
        mask = (test_processed['horizon'] == h)
        if not mask.any():
            continue
            
        h_test = test_processed[mask].copy()
        
        # Load the first model to get the exact feature names used in training
        first_model_path = str(MODELS_DIR / f'lgbm_h{h}_full_seed42.txt')
        if not os.path.exists(first_model_path):
            raise FileNotFoundError(f"Missing model file: {first_model_path}")
            
        first_booster = lgb.Booster(model_file=first_model_path)
        model_features = first_booster.feature_name()
        
        # Accumulate predictions across all 5 Seeds
        h_preds = np.zeros(len(h_test))
        model_count = 0
        
        for seed in SEEDS:
            model_path = str(MODELS_DIR / f'lgbm_h{h}_full_seed{seed}.txt')
            if os.path.exists(model_path):
                booster = lgb.Booster(model_file=model_path)
                h_preds += booster.predict(h_test[model_features])
                model_count += 1
        
        if model_count == 0:
            raise FileNotFoundError(f"No models found for Horizon {h}")
            
        h_preds /= model_count
        preds[mask.values] = h_preds
        
        del h_test
        gc.collect()

    print("\n5. Packaging Result...")
    submission = test_processed[['id']].copy()
    submission['y_target'] = preds
    
    # Kaggle metrics often expect exact row order and specific column name!
    print(" -> Restoring original row order & column name...")
    root_sub = pd.read_csv(DATA_DIR / 'submission.csv', usecols=['id'])
    submission = submission.set_index('id').loc[root_sub['id']].reset_index()
    submission.rename(columns={'y_target': 'prediction'}, inplace=True)
    
    out_path = TEST_DIR / 'submission.csv'
    submission.to_csv(out_path, index=False)
    print(f"\n==> SUCCESS: Submission ready at {out_path}")

if __name__ == '__main__':
    import time
    start = time.time()
    main(retrain=False)
    print(f"Total Inference Time: {time.time() - start:.2f}s")
