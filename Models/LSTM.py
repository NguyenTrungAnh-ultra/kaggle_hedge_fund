import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. HARDWARE CONFIG (NVIDIA RTX 5060 Optimized)
# ==========================================
DEBUG_MODE = False
BATCH_SIZE = 4096   
NUM_WORKERS = 0     

# ==========================================
# 1. SETUP THƯ MỤC VÀ LOGGING (Cố định Folder)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_param')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
for d in [MODELS_DIR, REPORTS_DIR]: os.makedirs(d, exist_ok=True)

def zero_mean_r2_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    if denom == 0: return 0.0
    return 1.0 - np.sum(w * (y_target - y_pred) ** 2) / (denom + 1e-8)

class FastPanelDataset(Dataset):
    def __init__(self, X, y, w, ts, t1, group_ids, time_steps: int = 20):
        super().__init__()
        self.time_steps = time_steps
        self.X_data = X; self.y_data = y; self.w_data = w
        self.ts_index_arr = ts; self.t1_arr = t1
        
        # Valid indices: sequences within same group
        current_g = group_ids[time_steps - 1:]
        past_g = group_ids[:-time_steps + 1]
        self.valid_indices = np.where(current_g == past_g)[0] + time_steps - 1
        print(f"Dataset: Total Rows={len(X)} | Valid Windows={len(self.valid_indices)}")

    def __len__(self): return len(self.valid_indices)
    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.time_steps + 1
        return torch.tensor(self.X_data[start_idx:end_idx+1]), torch.tensor(self.y_data[end_idx]), torch.tensor(self.w_data[end_idx]), end_idx

class LSTM_LeakFreePurgedKFold:
    def __init__(self, dataset, n_splits=5, pctEmbargo=0.01):
        self.dataset = dataset; self.n_splits = n_splits; self.pctEmbargo = pctEmbargo
    def split(self, train_idx_subset):
        df_indices = self.dataset.valid_indices[train_idx_subset]
        ts_valid_subset = self.dataset.ts_index_arr[df_indices]
        unique_ts = np.sort(np.unique(ts_valid_subset))
        n_ts = len(unique_ts)
        # AFML Style: Purging & Embargoing
        kf = KFold(n_splits=min(self.n_splits, n_ts), shuffle=False)
        embargo = int(n_ts * self.pctEmbargo)
        for tr_ts_idx, te_ts_idx in kf.split(unique_ts):
            te_times = unique_ts[te_ts_idx]
            t0_te = te_times[0]; t1_te = self.dataset.t1_arr[np.isin(self.dataset.ts_index_arr, te_times)].max()
            mask_tr_base = np.isin(ts_valid_subset, unique_ts[tr_ts_idx])
            te_subset = np.where(np.isin(ts_valid_subset, te_times))[0]
            t1_tr = self.dataset.t1_arr[df_indices]; ts_tr = ts_valid_subset
            # Purge: Tr ends before Te window starts (accounting for LSTM window)
            keep_before = t1_tr <= (t0_te - self.dataset.time_steps + 1)
            # Embargo: Tr starts after Te ends
            keep_after = ts_tr > t1_te
            if embargo > 0:
                after_te = unique_ts[unique_ts > t1_te]
                if len(after_te) > 0: keep_after &= (~np.isin(ts_tr, after_te[:embargo]))
            yield np.where(mask_tr_base & (keep_before | keep_after))[0], te_subset

class AFML_QuantLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.4):
        super().__init__()
        self.spatial_dropout = nn.Dropout1d(dropout_rate)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x: (N, T, D) -> spatial dropout on D
        x = x.permute(0, 2, 1); x = self.spatial_dropout(x); x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :]).squeeze()

class QuantLSTMTrainer:
    def __init__(self, input_dim, device):
        self.device = device; self.input_dim = input_dim
    def train_evaluate(self, tr_l, va_l, fold):
        model = AFML_QuantLSTM(input_dim=self.input_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_r2 = -float('inf'); best_state = None; patience_limit = 5; patience = 0
        for epoch in range(30):
            model.train(); tr_loss = 0.0
            for X, y, w, _ in tr_l:
                X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
                optimizer.zero_grad(); p = model(X); l = torch.sum(w*(y-p)**2)/(torch.sum(w)+1e-8)
                l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                tr_loss += l.item()
            model.eval(); v_p, v_t, v_w = [], [], []
            with torch.no_grad():
                for X, y, w, _ in va_l:
                    v_p.extend(model(X.to(self.device)).cpu().numpy()); v_t.extend(y.numpy()); v_w.extend(w.numpy())
            v_r2 = zero_mean_r2_score(np.array(v_t), np.array(v_p), np.array(v_w))
            if epoch % 1 == 0: print(f"      F{fold} E{epoch+1} | TrL: {tr_loss/(len(tr_l)+1e-8):.6f} | ValR2: {v_r2:.6f}")
            if v_r2 > best_r2: best_r2 = v_r2; best_state = model.state_dict(); patience = 0
            else: patience += 1
            if patience >= patience_limit: break
        torch.save(best_state, os.path.join(MODELS_DIR, f"lstm_fold_{fold}.pt"))
        model.load_state_dict(best_state)
        return model, best_r2

def main():
    print("====== LSTM START (POLARS MEMORY OPTIMIZED) ======")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Backend: {device.type.upper()}")
    
    file_path = r'f:\Projects\hege_fund_kaggle\ts-forecasting\train_stationary.parquet'
    
    # POLARS LAZY PATH (Cực nhanh và tiết kiệm RAM)
    pl_lazy = pl.scan_parquet(file_path).sort(["code", "sub_code", "horizon", "ts_index"])
    
    if DEBUG_MODE:
        cutoff = pl_lazy.select("ts_index").unique().collect().get_column("ts_index").sort()[int(422 * 0.1)]
        pl_lazy = pl_lazy.filter(pl.col("ts_index") <= cutoff)
    
    blacklist = ['feature_h', 'feature_j', 'feature_l', 'feature_ah', 'feature_an', 'feature_bd', 
                 'feature_be', 'feature_bf', 'feature_bh', 'feature_bm', 'feature_bn', 'feature_bo', 
                 'feature_bq', 'feature_br', 'feature_by', 'feature_bz', 'feature_ca', 'feature_cb', 
                 'feature_cc', 'feature_cd', 'feature_bg', 'feature_bp', 'feature_cf', 'feature_z', 'feature_k', 'feature_aq']
    exclude = ['id', 'y_target', 'weight', 'ts_index', 't1_end', 't1_arr'] + blacklist + ['code', 'sub_code', 'sub_category']
    features = [c for c in pl_lazy.columns if c not in exclude]
    
    # Scaling Cross-Sectional in Lazy Mode
    print("Standardizing Cross-Sectional Features (Lazy over ts_index)...")
    scale_exprs = [((pl.col(f) - pl.col(f).mean().over("ts_index")) / (pl.col(f).std().over("ts_index") + 1e-8)).cast(pl.Float32).alias(f) for f in features]
    
    y_std = pl_lazy.select(pl.col("y_target")).collect().get_column("y_target").std()
    avg_w = pl_lazy.select(pl.col("weight")).collect().get_column("weight").mean()
    
    processed_pl = pl_lazy.with_columns(scale_exprs + [
        (pl.col("y_target") / (y_std + 1e-8)).cast(pl.Float32),
        (pl.col("weight") / (avg_w + 1e-8)).cast(pl.Float32),
        pl.struct(["code", "sub_code", "horizon"]).rank("dense").cast(pl.Int32).alias("group_id")
    ]).collect()
    
    print("Converting to Numpy (Memory Safe)...")
    X = processed_pl.select(features).to_numpy().astype(np.float32)
    y = processed_pl.get_column("y_target").to_numpy().astype(np.float32)
    w = processed_pl.get_column("weight").to_numpy().astype(np.float32)
    ts = processed_pl.get_column("ts_index").to_numpy().astype(np.int32)
    t1 = (processed_pl.get_column("ts_index") + processed_pl.get_column("horizon")).to_numpy().astype(np.int32)
    gids = processed_pl.get_column("group_id").to_numpy().astype(np.int32)
    del processed_pl, pl_lazy
    
    dataset = FastPanelDataset(X, y, w, ts, t1, gids, time_steps=20)
    indices = dataset.valid_indices
    unique_ts = np.sort(np.unique(ts[indices]))
    split_time = unique_ts[int(len(unique_ts) * 0.8)]
    
    train_subset = np.where(ts[indices] < split_time)[0]
    test_subset = np.where(ts[indices] >= split_time)[0]
    train_subset = train_subset[t1[indices[train_subset]] <= (ts[indices[test_subset]].min() - 20 + 1)]
    
    print(f"Train samples: {len(train_subset)} | Test samples: {len(test_subset)}")
    if len(train_subset) < 2: return

    trainer = QuantLSTMTrainer(input_dim=len(features), device=device)
    cv = LSTM_LeakFreePurgedKFold(dataset, n_splits=5)
    test_ldr = DataLoader(dataset, batch_size=4096, sampler=torch.utils.data.SequentialSampler(test_subset))
    test_preds = np.zeros(len(test_subset))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(train_subset), 1):
        print(f"\n--- Fold {fold} ---")
        tr_ldr = DataLoader(dataset, batch_size=4096, sampler=torch.utils.data.SubsetRandomSampler(train_subset[tr_idx]))
        va_ldr = DataLoader(dataset, batch_size=4096, sampler=torch.utils.data.SubsetRandomSampler(train_subset[va_idx]))
        model, _ = trainer.train_evaluate(tr_ldr, va_ldr, fold)
        model.eval(); f_p = []
        with torch.no_grad():
            for X_b, _, _, _ in test_ldr: f_p.extend(model(X_b.to(device)).cpu().numpy())
        test_preds += np.array(f_p) / 5.0

    final_r2 = zero_mean_r2_score(y[indices[test_subset]], test_preds, w[indices[test_subset]])
    print(f"\nFINAL TEST SET KAGGLE R2: {final_r2:.6f}")

if __name__ == '__main__':
    main()