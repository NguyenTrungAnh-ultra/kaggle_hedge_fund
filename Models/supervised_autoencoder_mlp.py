"""
Institutional-Grade Multi-Task MLP for Hedge Fund Kaggle Competition
====================================================================
Refactored to support Multi-Horizon Output (h1, h3, h10, h25) with safe
Out-Of-Distribution categorical handling, fast tensor slicing, and
hierarchical micro/macro identity dropout.

[RESTORED AUTOENCODER]: Fully integrated the Jane Street Supervised Autoencoder
for Raw Data pattern extraction. Includes the Tri-Loss gradient blending 
(Main + AE + Decoder) with corrected target scale and bias mapping.
"""

import os
import sys
import json
import warnings
import argparse
import time

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from metric import weighted_rmse_score
from Modules.PurgedKFold import LeakFreePurgedKFold

LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "ts-forecasting")
PARAM_DIR = os.path.join(PROJECT_ROOT, "models_param")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MLP] Using device: {DEVICE}")

CAT_COLS = ["code", "sub_code", "sub_category"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FAST TENSOR DATALOADER
# ═══════════════════════════════════════════════════════════════════════════════
class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)

        self.i += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CORE ARCHITECTURE (Supervised Autoencoder)
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class GaussianNoise(nn.Module):
    """Gaussian noise layer for robust bottleneck regularization."""
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class HedgeFundMLP(nn.Module):
    """
    Supervised Autoencoder Multi-Task MLP with hierarchical entity dropout.
    """
    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: list,
        emb_dims: list,
        num_targets: int = 4,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, emb_dims)
        ])

        self.micro_drop = nn.Dropout(0.5) 
        self.macro_drop = nn.Dropout(0.1) 

        total_emb_dim = sum(emb_dims)
        self.bn0 = nn.BatchNorm1d(n_cont_features)

        # ─── AUTOENCODER BRANCH ─────────────────────────
        self.gaussian = GaussianNoise(0.035) 
        
        # Encoder Projector
        ae_hidden = hidden_dim // 2
        self.encoder_proj = nn.Sequential(
            nn.Linear(n_cont_features, ae_hidden),
            nn.BatchNorm1d(ae_hidden),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 
        self.decoder_drop = nn.Dropout(max(0.01, dropout - 0.05))
        self.decoder = nn.Linear(ae_hidden, n_cont_features)
        
        # Auxiliary Classification from AE Bottleneck
        self.ae_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(ae_hidden, ae_hidden // 2),
            nn.BatchNorm1d(ae_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ae_hidden // 2, num_targets)
        )

        # ─── MAIN PREDICTION BRANCH ──────────────────────
        input_dim = n_cont_features + total_emb_dim + ae_hidden
        
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        self.main_head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x_cont, x_cat):
        x0 = self.bn0(x_cont)

        embs = []
        for i, emb in enumerate(self.embeddings):
            idx = torch.clamp(x_cat[:, i], min=0, max=emb.num_embeddings - 1)
            embs.append(emb(idx))

        emb_code = self.micro_drop(embs[0])
        emb_sub_code = self.micro_drop(embs[1])
        emb_sub_cat = self.macro_drop(embs[2])
        emb_concat = torch.cat([emb_code, emb_sub_code, emb_sub_cat], dim=1)

        # AE Path
        enc_input = self.gaussian(x0)
        encoder_out = self.encoder_proj(enc_input)
        
        dec_input = self.decoder_drop(encoder_out)
        x_recon = self.decoder(dec_input)
        
        preds_ae = self.ae_head(encoder_out)

        # Main Path
        x = torch.cat([x0, emb_concat, encoder_out], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        preds_main = self.main_head(x)
        
        return preds_main, preds_ae, x_recon


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOSS: Masked Weighted Pseudo-Huber
# ═══════════════════════════════════════════════════════════════════════════════
class WeightedPseudoHuberLoss(nn.Module):
    def __init__(self, delta=15.0):
        super().__init__()
        self.delta = delta

    def forward(self, preds, targets, weights):
        valid_mask = ~torch.isnan(targets)
        weights_expanded = weights.unsqueeze(1).expand_as(targets)
        
        p_valid = preds[valid_mask]
        t_valid = targets[valid_mask]
        w_valid = weights_expanded[valid_mask]

        d = p_valid - t_valid
        
        z = d / self.delta
        loss = self.delta ** 2 * (torch.sqrt(1.0 + z ** 2) - 1.0)
        weighted_loss = (w_valid * loss).sum() / (w_valid.sum() + 1e-12)
        return weighted_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA PREPARATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_data(file_path):
    print(f"1. Loading RAW dataset from: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"   Dataset shape: {df.shape}")

    print("   Pivoting Long to Wide format...")
    base_cols = ["ts_index", "code", "sub_code", "sub_category"]
    
    if "t1_end" in df.columns:
        df.drop(columns=["t1_end"], inplace=True)
        
    df_targets = df.pivot_table(index=base_cols, columns="horizon", values="y_target").reset_index()
    
    target_names = []
    for h in [1, 3, 10, 25]:
        col_name = h if h in df_targets.columns else str(h)
        if col_name in df_targets.columns:
            df_targets.rename(columns={col_name: f"y_h{h}"}, inplace=True)
            target_names.append(f"y_h{h}")

    valid_mask = df_targets[target_names].notna().any(axis=1)
    if not valid_mask.all():
        df_targets = df_targets[valid_mask].reset_index(drop=True)

    df_base = df.drop(columns=["horizon", "y_target"]).drop_duplicates(subset=base_cols)
    df = df_base.merge(df_targets, on=base_cols, how="inner")

    df["t1_end"] = df["ts_index"] + 25 
    df = df.sort_values(["ts_index", "code", "sub_code"]).reset_index(drop=True)

    exclude = {"id", "weight", "ts_index", "t1_end"} | set(target_names)
    all_features = [c for c in df.columns if c not in exclude]
    cont_features = [c for c in all_features if c not in CAT_COLS]

    new_flags = []
    for col in cont_features:
        if df[col].isna().any():
            flag_col = f"flagna_{col}"
            df[flag_col] = df[col].isna().astype(np.float32)
            new_flags.append(flag_col)
    
    cont_features.extend(new_flags)

    const_cols = [c for c in cont_features if df[c].nunique() <= 1]
    if const_cols:
        cont_features = [c for c in cont_features if c not in const_cols]

    print(f"   Continuous features: {len(cont_features)}")
    print(f"   Categorical features: {len(CAT_COLS)}")

    cat_encoders = {}
    cat_cardinalities = []
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
        cat_encoders[col] = df[col].cat.categories
        df[col] = df[col].cat.codes.astype(np.int32)
        if (df[col] < 0).any():
            df[col] = df[col] + 1
            cat_cardinalities.append(len(cat_encoders[col]) + 1)
        else:
            cat_cardinalities.append(len(cat_encoders[col]))

    return df, cont_features, cat_cardinalities, cat_encoders, target_names


def get_fold_data(df, train_idx, val_idx, cont_features, target_names):
    X_cont_tr = df.iloc[train_idx][cont_features].values.astype(np.float32)
    X_cont_va = df.iloc[val_idx][cont_features].values.astype(np.float32)

    X_cat_tr = df.iloc[train_idx][CAT_COLS].values.astype(np.int64)
    X_cat_va = df.iloc[val_idx][CAT_COLS].values.astype(np.int64)

    y_tr = df.iloc[train_idx][target_names].values.astype(np.float32)
    y_va = df.iloc[val_idx][target_names].values.astype(np.float32)

    w_tr = df.iloc[train_idx]["weight"].values.astype(np.float32)
    w_va = df.iloc[val_idx]["weight"].values.astype(np.float32)

    sub_cat_col_idx = CAT_COLS.index("sub_category")
    sub_cat_tr = X_cat_tr[:, sub_cat_col_idx]
    sub_cat_va = X_cat_va[:, sub_cat_col_idx]

    global_medians = np.nanmedian(X_cont_tr, axis=0)
    global_medians = np.where(np.isnan(global_medians), 0.0, global_medians)

    unique_sub_cats = np.unique(sub_cat_tr)
    group_medians = {}
    for sc in unique_sub_cats:
        mask = sub_cat_tr == sc
        sc_medians = np.nanmedian(X_cont_tr[mask], axis=0)
        sc_medians = np.where(np.isnan(sc_medians), global_medians, sc_medians)
        group_medians[sc] = sc_medians

    nan_mask_tr = np.isnan(X_cont_tr)
    if nan_mask_tr.any():
        for sc in unique_sub_cats:
            row_mask = (sub_cat_tr == sc)
            combined = row_mask[:, np.newaxis] & nan_mask_tr
            if combined.any():
                X_cont_tr = np.where(combined, group_medians[sc][np.newaxis, :], X_cont_tr)

    nan_mask_va = np.isnan(X_cont_va)
    if nan_mask_va.any():
        for sc in np.unique(sub_cat_va):
            row_mask = (sub_cat_va == sc)
            combined = row_mask[:, np.newaxis] & nan_mask_va
            if combined.any():
                fill_vals = group_medians.get(sc, global_medians)
                X_cont_va = np.where(combined, fill_vals[np.newaxis, :], X_cont_va)

    X_cont_tr = np.sign(X_cont_tr) * np.log1p(np.abs(X_cont_tr))
    X_cont_va = np.sign(X_cont_va) * np.log1p(np.abs(X_cont_va))

    tr_mean = X_cont_tr.mean(axis=0)
    tr_std = X_cont_tr.std(axis=0) + 1e-8

    X_cont_tr = (X_cont_tr - tr_mean) / tr_std
    X_cont_va = (X_cont_va - tr_mean) / tr_std
    
    # Clip z-scores to remove massive outliers on fully raw data
    X_cont_tr = np.clip(X_cont_tr, -10.0, 10.0)
    X_cont_va = np.clip(X_cont_va, -10.0, 10.0)

    w_mean = w_tr.mean() + 1e-12
    w_tr = w_tr / w_mean
    w_va = w_va / w_mean

    TARGET_SCALE = 10000.0
    y_tr = y_tr * TARGET_SCALE
    y_va = y_va * TARGET_SCALE

    return (X_cont_tr, X_cat_tr, y_tr, w_tr,
            X_cont_va, X_cat_va, y_va, w_va,
            global_medians, tr_mean, tr_std, w_mean)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_cont, x_cat, y, w in loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        preds_main, preds_ae, x_recon = model(x_cont, x_cat)
        
        loss_main = criterion(preds_main, y, w)
        loss_ae = criterion(preds_ae, y, w)
        
        # 10000x multiplier matches the Target Scale magnitude assigned manually, balancing Autoencoder gradients
        loss_recon = F.mse_loss(x_recon, x_cont) * 10000.0
        
        loss = loss_main + (0.5 * loss_ae) + (0.5 * loss_recon)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate and output 2D target metrics robustly."""
    model.eval()
    all_preds, all_y, all_w = [], [], []

    for x_cont, x_cat, y, w in loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)

        preds_main, preds_ae, _ = model(x_cont, x_cat)
        preds = 0.5 * preds_main + 0.5 * preds_ae
        
        all_preds.append(preds.cpu().numpy().astype(np.float64))
        all_y.append(y.numpy().astype(np.float64))
        all_w.append(w.numpy().astype(np.float64))

    return np.concatenate(all_preds), np.concatenate(all_y), np.concatenate(all_w)


def train_fold(
    fold_num,
    X_cont_tr, X_cat_tr, y_tr, w_tr,
    X_cont_va, X_cat_va, y_va, w_va,
    w_va_raw,  
    n_cont_features, cat_cardinalities, target_names,
    hparams, device,
    trial=None  
):
    train_loader = FastTensorDataLoader(
        torch.as_tensor(X_cont_tr, dtype=torch.float32),
        torch.as_tensor(X_cat_tr, dtype=torch.long),
        torch.as_tensor(y_tr, dtype=torch.float32),
        torch.as_tensor(w_tr, dtype=torch.float32),
        batch_size=hparams["batch_size"],
        shuffle=True
    )
    val_loader = FastTensorDataLoader(
        torch.as_tensor(X_cont_va, dtype=torch.float32),
        torch.as_tensor(X_cat_va, dtype=torch.long),
        torch.as_tensor(y_va, dtype=torch.float32),
        torch.as_tensor(w_va, dtype=torch.float32),
        batch_size=hparams["batch_size"] * 2,
        shuffle=False
    )

    emb_dims = [min(dim, hparams["max_emb_dim"]) for dim in
                [max(4, card // 2) for card in cat_cardinalities]]

    model = HedgeFundMLP(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        emb_dims=emb_dims,
        num_targets=len(target_names),
        hidden_dim=hparams["hidden_dim"],
        n_blocks=hparams["n_blocks"],
        dropout=hparams["dropout"],
    ).to(device)

    TARGET_SCALE = 10000.0
    with torch.no_grad():
        for i in range(len(target_names)):
            y_valid_mask = ~np.isnan(y_tr[:, i])
            if y_valid_mask.any() and w_tr[y_valid_mask].sum() > 0:
                weighted_mean = np.sum(w_tr[y_valid_mask] * y_tr[y_valid_mask, i]) / np.sum(w_tr[y_valid_mask])
                model.main_head[-1].bias[i].fill_(weighted_mean)
                model.ae_head[-1].bias[i].fill_(weighted_mean)
            else:
                model.main_head[-1].bias[i].fill_(0.0)
                model.ae_head[-1].bias[i].fill_(0.0)

    scaled_delta = hparams["delta"] * TARGET_SCALE
    criterion = WeightedPseudoHuberLoss(delta=scaled_delta)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": hparams["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=hparams["lr"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=max(1, hparams["patience"] // 3), min_lr=1e-6
    )

    best_score = -np.inf
    best_state = None
    patience_counter = 0
    patience = hparams["patience"]
    n_epochs = hparams["epochs"]

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=hparams["grad_clip"]
        )

        preds_va, targets_va, weights_va = evaluate(model, val_loader, device)
        
        scores = []
        for i in range(targets_va.shape[1]):
            v_mask = ~np.isnan(targets_va[:, i])
            if v_mask.any():
                t = targets_va[v_mask, i]
                p = preds_va[v_mask, i]
                w = w_va_raw[v_mask]
                scores.append(weighted_rmse_score(t, p, w))
                
        val_score = np.mean(scores) if scores else 0.0
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step(val_score)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Fold {fold_num} | Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {train_loss:.6f} | Val Score: {val_score:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if trial is not None:
            trial.report(val_score, epoch)
            if trial.should_prune():
                print(f"  Fold {fold_num} | Optuna Pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            print(f"  Fold {fold_num} | Early stopping at epoch {epoch} - Pruned!")
            break

    print(f"  Fold {fold_num} | Best Val Score: {best_score:.6f}")
    return best_state, best_score

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def optuna_objective(trial, df_sample, cont_features, cat_cardinalities, target_names, device):
    hparams = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 384, 512, 768]),
        "n_blocks": trial.suggest_int("n_blocks", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "delta": trial.suggest_float("delta", 5.0, 100.0),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "max_emb_dim": trial.suggest_categorical("max_emb_dim", [16, 32, 50]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "epochs": 30,  
        "patience": 8,
    }

    unique_ts = np.sort(df_sample["ts_index"].unique())
    split_idx = int(len(unique_ts) * 0.8)
    split_ts = unique_ts[split_idx]

    tr_mask_base = df_sample["ts_index"] < split_ts
    va_mask = df_sample["ts_index"] >= split_ts

    va_start = df_sample[va_mask]["ts_index"].min()
    overlap_mask = tr_mask_base & (df_sample["t1_end"] > va_start)
    tr_mask = tr_mask_base & (~overlap_mask)

    train_idx = np.where(tr_mask.values)[0]
    val_idx = np.where(va_mask.values)[0]

    (X_cont_tr, X_cat_tr, y_tr, w_tr,
     X_cont_va, X_cat_va, y_va, w_va,
     _, _, _, w_mean) = get_fold_data(df_sample, train_idx, val_idx, cont_features, target_names)

    w_va_raw = df_sample.iloc[val_idx]["weight"].values.astype(np.float64)

    _, val_score = train_fold(
        fold_num=f"T{trial.number}",
        X_cont_tr=X_cont_tr, X_cat_tr=X_cat_tr, y_tr=y_tr, w_tr=w_tr,
        X_cont_va=X_cont_va, X_cat_va=X_cat_va, y_va=y_va, w_va=w_va,
        w_va_raw=w_va_raw,
        n_cont_features=len(cont_features),
        cat_cardinalities=cat_cardinalities,
        target_names=target_names,
        hparams=hparams, device=device,
        trial=trial,
    )

    return val_score


def run_tuning(df, cont_features, cat_cardinalities, target_names, device, n_trials=15):
    import logging

    print("\n3. Starting Optuna Hyperparameter Tuning...")

    optuna_log_file = os.path.join(LOG_DIR, "mlp_optuna.log")
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))

    unique_ts = np.sort(df["ts_index"].unique())
    cut_off_idx = int(len(unique_ts) * 0.8)
    cut_off_time = unique_ts[cut_off_idx]
    # To save time and keep Autoencoder robust, sample heavily
    df_sample = df[df["ts_index"] >= cut_off_time].copy().reset_index(drop=True)
    df_sample = df_sample.sort_values("ts_index").reset_index(drop=True)
    print(f"   Tuning sample: {len(df_sample)} rows ({len(df_sample)/len(df)*100:.1f}%)")

    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=3)
    )
    study.optimize(
        lambda trial: optuna_objective(trial, df_sample, cont_features, cat_cardinalities, target_names, device),
        n_trials=n_trials,
    )

    print(f"\n   Best trial: {study.best_trial.number}")
    print(f"   Best value: {study.best_trial.value:.6f}")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

    param_path = os.path.join(PARAM_DIR, "mlp_best_param.json")
    with open(param_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    return study.best_trial.params


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def run_mlp(file_path, tuning=False, n_trials=15, skip_training=False):
    df, cont_features, cat_cardinalities, cat_encoders, target_names = prepare_data(file_path)
    print(f"   Final aggregated shape: {df.shape}")

    param_path = os.path.join(PARAM_DIR, "mlp_best_param.json")
    
    if tuning:
        best_params = run_tuning(df, cont_features, cat_cardinalities, target_names, DEVICE, n_trials)
    elif os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)
        print(f"\n   Loaded parameters from {param_path}")
    else:
        print(f"\n   Warning: {param_path} not found. Using safe baseline defaults.")
        best_params = {
            "hidden_dim": 512,
            "n_blocks": 3,
            "dropout": 0.15,
            "lr": 3e-4,
            "weight_decay": 1e-5,
            "delta": 15.0,
            "batch_size": 4096,
            "max_emb_dim": 32,
            "grad_clip": 1.0,
        }
    
    hparams = {
        **best_params,
        "epochs": best_params.get("epochs", 80),
        "patience": best_params.get("patience", 15),
    }

    print(f"\n   Using robust hyperparameters: {json.dumps(hparams, indent=4)}")

    if skip_training:
        print("\n   Skipping training phase based on parameters.")
        return

    print(f"\n4. Starting Leak-Free Purged K-Fold Training ({len(target_names)} Targets)...")
    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    cv_meta = df[["ts_index", "t1_end"]].copy()

    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(cv_meta), 1):
        print(f"\n{'='*60}\n  FOLD {fold}\n{'='*60}")
        print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

        w_va_raw = df.iloc[val_idx]["weight"].values.astype(np.float64)

        (X_cont_tr, X_cat_tr, y_tr, w_tr,
         X_cont_va, X_cat_va, y_va, w_va,
         train_medians, tr_mean, tr_std, w_mean) = get_fold_data(
            df, train_idx, val_idx, cont_features, target_names
        )

        best_state, val_score = train_fold(
            fold_num=fold,
            X_cont_tr=X_cont_tr, X_cat_tr=X_cat_tr, y_tr=y_tr, w_tr=w_tr,
            X_cont_va=X_cont_va, X_cat_va=X_cat_va, y_va=y_va, w_va=w_va,
            w_va_raw=w_va_raw,
            n_cont_features=len(cont_features),
            cat_cardinalities=cat_cardinalities,
            target_names=target_names,
            hparams=hparams, device=DEVICE,
        )

        fold_scores.append(val_score)

        save_dict = {
            "model_state_dict": best_state,
            "hparams": hparams,
            "cont_features": cont_features,
            "cat_cardinalities": cat_cardinalities,
            "target_names": target_names,
            "train_medians": train_medians,
            "tr_mean": tr_mean,
            "tr_std": tr_std,
            "w_mean": w_mean,
        }
        
        model_path = os.path.join(PARAM_DIR, f"safe_ae_mlp_fold_{fold}.pt")
        torch.save(save_dict, model_path)
        print(f"  Saved Multi-Task framework to {model_path}")

        del X_cont_tr, X_cat_tr, y_tr, w_tr, X_cont_va, X_cat_va, y_va, w_va, best_state

    cv_weights = np.array([0.0625, 0.0625, 0.125, 0.25, 0.5])
    weighted_mean_score = np.average(fold_scores, weights=cv_weights)
    simple_mean_score = np.mean(fold_scores)

    print(f"\n{'='*60}\n  MULTI-TASK MLP FINAL RESULTS SUMMARY\n{'='*60}")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.6f}  (Time-Decay Weight: {cv_weights[i]})")
    print(f"\n  ==> WEIGHTED CV SCORE: {weighted_mean_score:.6f} (Prioritizing recent bounds)")
    print(f"  ==> SIMPLE CV SCORE:   {simple_mean_score:.6f}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task HedgeFund MLP")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials")
    parser.add_argument("--no-train", action="store_true", help="Tune only, skip training")
    args = parser.parse_args()

    file_path = os.path.join(DATA_DIR, "train_feature_builded.parquet")
    run_mlp(file_path, tuning=args.tune, n_trials=args.n_trials, skip_training=args.no_train)
