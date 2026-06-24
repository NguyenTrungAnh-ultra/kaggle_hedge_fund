import sys
import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from Models.metric import weighted_rmse_score

# Force UTF-8 encoding for stdout on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ====================================================================
# CONFIGURATION
# ====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / 'models_param'
LOGS_DIR = PROJECT_ROOT / 'Models' / 'logs' / 'reports'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

HORIZONS = [1, 3, 10, 25]
SEEDS = [42, 123, 7, 2024, 999]

LGB_PARAMS_BASE = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.015,
    "num_leaves": 80,
    "max_depth": 6,
    "min_child_samples": 200,
    "subsample": 0.7,
    "subsample_freq": 5,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "n_jobs": -1,
    "verbose": -1,
    "device": "cpu"
}

SIGN_SPLIT_FEATURES = ['feature_n', 'feature_w', 'feature_x', 'feature_y', 'feature_z', 'feature_al', 'feature_ce']
LAG_FEATURES = ['feature_al', 'feature_am', 'feature_cg', 'feature_by']
LAGS = [1, 3, 5, 10, 25]

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
class LeakFreePurgedKFold:
    def __init__(self, n_splits=5, pctEmbargo=0.01):
        self.n_splits = n_splits
        self.pctEmbargo = pctEmbargo

    def split(self, X_meta):
        ts_indices = X_meta['ts_index'].values
        t1_ends = X_meta['t1_end'].values
        
        unique_ts = np.unique(ts_indices)
        unique_ts.sort()
        
        max_ts = unique_ts[-1]
        embargo_size = int(max_ts * self.pctEmbargo)
        
        fold_size = len(unique_ts) // self.n_splits
        
        for i in range(self.n_splits):
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(unique_ts)
            
            test_ts_range = unique_ts[test_start_idx:test_end_idx]
            if len(test_ts_range) == 0:
                continue
                
            test_start_ts = test_ts_range[0]
            test_end_ts = test_ts_range[-1]
            
            test_mask = (ts_indices >= test_start_ts) & (ts_indices <= test_end_ts)
            test_indices = np.where(test_mask)[0]
            
            embargo_end_ts = test_end_ts + embargo_size
            train_mask = (t1_ends < test_start_ts) | (ts_indices > embargo_end_ts)
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices

def make_recency_weights(ts_index, half_life_frac=0.5):
    """
    ts_index: pandas Series of timestamps
    half_life_frac: fraction of the total time window at which weight is 0.5
    """
    if len(ts_index) == 0:
        return np.ones(0)
    min_ts, max_ts = ts_index.min(), ts_index.max()
    if min_ts == max_ts:
        return np.ones(len(ts_index))
        
    scaled_ts = (ts_index - min_ts) / (max_ts - min_ts)
    decay_rate = -np.log(0.5) / half_life_frac
    weights = np.exp(decay_rate * (scaled_ts - 1.0))
    return weights.values

def apply_advanced_features(df):
    # 1. Sign-Splits (DISABLED - proved to hurt CV in author's A/B test)
    # for feat in SIGN_SPLIT_FEATURES:
    #     if feat in df.columns:
    #         df[f'{feat}_pos'] = np.maximum(df[feat].values, 0)
    #         df[f'{feat}_neg'] = np.minimum(df[feat].values, 0)
            
    # 2. Feature Lags
    df = df.sort_values(['code', 'sub_code', 'sub_category', 'ts_index'])
    group_cols = ['code', 'sub_code', 'sub_category']
    for col in LAG_FEATURES:
        if col in df.columns:
            grp = df.groupby(group_cols, sort=False, observed=False)[col]
            for lag in LAGS:
                df[f'{col}_lag_{lag}'] = grp.shift(lag)
    
    # Restore original order just in case
    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)
    return df

def compute_target_encoding(train_df):
    enc_maps = {}
    # Handle categoricals correctly by using observed=False to avoid future warnings
    global_mean = train_df['y_target'].mean()
    for col in ['sub_code', 'sub_category']:
        enc_maps[col] = train_df.groupby(col, observed=False)['y_target'].mean().to_dict()
    return enc_maps, global_mean

def apply_target_encoding(df, enc_maps, global_mean):
    df = df.copy()
    for col in ['sub_code', 'sub_category']:
        df[f'{col}_target_enc'] = df[col].astype('object').map(enc_maps[col]).fillna(global_mean).astype('float32')
    return df

def select_top_features(train_df, feature_cols, target_col='y_target', weight_col='weight', n_features=50):
    ts_max = train_df['ts_index'].max()
    cutoff = ts_max - 600
    
    tr_mask = train_df['ts_index'] <= cutoff
    val_mask = train_df['ts_index'] > cutoff
    
    if tr_mask.sum() < 1000 or val_mask.sum() < 100:
        return feature_cols[:n_features]
        
    X_tr = train_df.loc[tr_mask, feature_cols]
    y_tr = train_df.loc[tr_mask, target_col].values
    w_tr = train_df.loc[tr_mask, weight_col].values
    
    X_va = train_df.loc[val_mask, feature_cols]
    y_va = train_df.loc[val_mask, target_col].values
    w_va = train_df.loc[val_mask, weight_col].values
    
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "n_estimators": 150,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    dval = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    importances = pd.Series(model.feature_importance(importance_type='gain'), index=feature_cols)
    top_n = importances.sort_values(ascending=False).head(n_features).index.tolist()
    return top_n

# ====================================================================
# MAIN TRAIN ROUTINE
# ====================================================================
def run_lgbm(file_path, training=True):
    print("1. Loading dataset...")
    df = pd.read_parquet(file_path)
    print(f"Dataset shape: {df.shape}")
    
    print("   -> Applying Causal Insights (Lags only) on FULL dataset...")
    df = apply_advanced_features(df)
    
    print("   -> Optimizing data types to prevent RAM overflow...")
    exclude_cols = ['y_target', 'weight']
    cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col in cat_cols:
            df[col] = df[col].astype('category')
        elif col.startswith('is_nan_'):
            df[col] = df[col].astype(np.int8)

    # Filter invalid targets early
    valid_mask = df['y_target'].notna()
    if not valid_mask.all():
        df = df[valid_mask].copy().reset_index(drop=True)

    df['t1_end'] = df['ts_index'] + df['horizon'].astype(int)
    df = df.sort_values(['ts_index', 'code', 'sub_code']).reset_index(drop=True)

    val_preds = np.zeros(len(df), dtype=np.float64)
    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    
    print("\n4. Starting Leak-Free Purged K-Fold Training (Per-Horizon, Target Encoding, Feature Selection, Recency, 5-Seed)...")
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df[['ts_index', 't1_end']])):
        print(f"\n--- Fold {fold+1} ---")
        
        y_va_all = []
        p_va_all = []
        w_va_all = []
        
        for h in HORIZONS:
            print(f" >> Horizon {h}")
            h_mask = (df['horizon'] == h)
            
            # Intersection of fold idx and horizon
            h_train_mask = h_mask.copy()
            h_train_mask.iloc[val_idx] = False
            
            h_val_mask = h_mask.copy()
            h_val_mask.iloc[train_idx] = False
            
            h_train = df[h_train_mask].copy()
            h_val = df[h_val_mask].copy()
            
            # === Target Encoding (Compute on train, apply to both) ===
            enc_maps, global_mean = compute_target_encoding(h_train)
            h_train = apply_target_encoding(h_train, enc_maps, global_mean)
            h_val = apply_target_encoding(h_val, enc_maps, global_mean)
            
            # === Feature Selection (Top 50) ===
            exclude_feats = ['id', 'y_target', 'weight', 'ts_index', 't1_end', 'horizon', 'code', 'sub_code', 'sub_category']
            candidate_features = [c for c in h_train.columns if c not in exclude_feats and 'logret' not in c]
            
            top_features = select_top_features(h_train, candidate_features, n_features=50)
            
            # Include categories and target encodings in the final feature list
            must_keep = ['code', 'sub_code', 'sub_category', 'sub_code_target_enc', 'sub_category_target_enc']
            h_features = list(dict.fromkeys([c for c in must_keep if c in h_train.columns] + top_features))
            
            cat_indices = [h_features.index(c) for c in ['code', 'sub_code', 'sub_category'] if c in h_features]
            
            # Target clipping
            y_tr = h_train['y_target'].values.copy()
            lo_val = np.percentile(y_tr, 5)
            hi_val = np.percentile(y_tr, 95)
            y_tr_clipped = np.clip(y_tr, lo_val, hi_val)
            
            # Recency Weights
            ts_tr = h_train['ts_index']
            base_w = h_train['weight'].values
            recency_w = make_recency_weights(ts_tr, half_life_frac=0.5)
            w_tr = base_w * recency_w
            
            y_va = h_val['y_target'].values
            w_va = h_val['weight'].values
            
            dval = lgb.Dataset(
                h_val[h_features], label=y_va, weight=w_va, categorical_feature=cat_indices, free_raw_data=False
            )
            
            # Ensemble predictions
            h_seed_preds = np.zeros(len(h_val))
            
            for seed in SEEDS:
                # Update seed in params
                params = LGB_PARAMS_BASE.copy()
                params['random_state'] = seed
                
                dtrain = lgb.Dataset(
                    h_train[h_features], label=y_tr_clipped, weight=w_tr, categorical_feature=cat_indices
                )
                
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    valid_sets=[dval, dtrain],
                    valid_names=['valid', 'train'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0)
                    ]
                )
                
                # Predict and accumulate
                h_seed_preds += model.predict(h_val[h_features])
                
                # Save model
                model_name = MODELS_DIR / f'lgbm_h{h}_f{fold}_seed{seed}.txt'
                model.save_model(model_name)
                
            # Average predictions over seeds
            h_seed_preds /= len(SEEDS)
            
            # Store fold predictions
            val_preds[h_val_mask.values] = h_seed_preds
            
            y_va_all.extend(y_va)
            p_va_all.extend(h_seed_preds)
            w_va_all.extend(w_va)
            
            # Clear memory
            del h_train, h_val, dtrain, dval, y_tr, w_tr, recency_w
            gc.collect()
 
        score = weighted_rmse_score(np.array(y_va_all), np.array(p_va_all), np.array(w_va_all))
        print(f"Fold {fold+1} Competition Score (R): {score:.5f}")
        fold_scores.append(score)
 
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    oof_mask = df['y_target'].notna()
    oof_score = weighted_rmse_score(df.loc[oof_mask, 'y_target'].values, val_preds[oof_mask], df.loc[oof_mask, 'weight'].values)
    
    print(f"\n==> AVERAGE CV SCORE: {mean_score:.6f} (+/- {std_score:.6f})")
    print(f"==> OVERALL OOF SCORE: {oof_score:.6f}\n")
 
if __name__ == '__main__':
    file_path = os.path.join(DATA_DIR, 'ts-forecasting', 'train_feature_builded.parquet')
    run_lgbm(file_path=file_path, training=True)
