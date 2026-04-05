"""
MLP Model for Hedge Fund Kaggle Competition
=============================================
Architecture:
  - Entity Embeddings for categorical features (code, sub_code, sub_category, horizon)
  - BatchNorm + Dropout + SiLU (Swish) for each hidden layer
  - Weighted Pseudo-Huber Loss (smooth L1) – matches the GBDT approach
  - Leak-Free Purged K-Fold Cross-Validation with Embargo
  - Optuna hyperparameter tuning on the last 20% time window

Usage:
  python Models/MLPs.py              # Train with default/loaded params  
  python Models/MLPs.py --tune       # Run Optuna tuning first, then train
"""

import os
import sys
import json
import warnings
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# DataLoader replaced by FastTensorDataLoader (direct tensor slicing)

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

# ── Categorical column definitions ──────────────────────────────────────────
CAT_COLS = ["code", "sub_code", "sub_category", "horizon"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FAST TENSOR DATALOADER
# ═══════════════════════════════════════════════════════════════════════════════
class FastTensorDataLoader:
    """
    DataLoader siêu tốc cho Dữ liệu Bảng (Tabular).
    Thay vì gọi __getitem__ từng dòng rồi ghép lại, nó cắt thẳng một khối ma trận (Slicing).
    Giải phóng 100% nghẽn cổ chai CPU.
    """
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
# 2. MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """Pre-activation residual block: BN → SiLU → Dropout → Linear → BN → SiLU → Dropout → Linear + skip."""

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


class HedgeFundMLP(nn.Module):
    """
    MLP with entity embeddings for categorical features
    and residual connections for deeper networks.
    """

    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: list,  # [n_code, n_sub_code, n_sub_category, n_horizon]
        emb_dims: list,           # embedding dim for each categorical
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Entity embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, emb_dims)
        ])

        # Anti Identity-Overfitting: heavy dropout on embeddings
        self.emb_drop = nn.Dropout(0.5)

        total_emb_dim = sum(emb_dims)
        input_dim = n_cont_features + total_emb_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Init weights
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
        # Embed categoricals với Trạm gác Chống sập (Clamp)
        embs = []
        for i, emb in enumerate(self.embeddings):
            # Lấy index của cột category thứ i
            idx = x_cat[:, i]
            # Ép index KHÔNG ĐƯỢC phép nhỏ hơn 0 và lớn hơn số lượng category tối đa
            idx = torch.clamp(idx, min=0, max=emb.num_embeddings - 1)
            embs.append(emb(idx))

        emb_concat = torch.cat(embs, dim=1)

        # Anti Identity-Overfitting: dropout on embedding BEFORE concat
        emb_concat = self.emb_drop(emb_concat)

        # Concat with continuous
        x = torch.cat([x_cont, emb_concat], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOSS: Weighted Pseudo-Huber
# ═══════════════════════════════════════════════════════════════════════════════
class WeightedPseudoHuberLoss(nn.Module):
    def __init__(self, delta=15.0):
        super().__init__()
        self.delta = delta

    def forward(self, preds, targets, weights):
        d = preds - targets
        z = d / self.delta
        loss = self.delta ** 2 * (torch.sqrt(1.0 + z ** 2) - 1.0)
        weighted_loss = (weights * loss).sum() / (weights.sum() + 1e-12)
        return weighted_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA PREPARATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_data(file_path):
    """Load data, encode categoricals, prepare features."""
    print("1. Loading dataset...")
    df = pd.read_parquet(file_path)
    print(f"   Dataset shape: {df.shape}")

    # t1_end for purged k-fold
    df["t1_end"] = df["ts_index"] + df["horizon"]
    df = df.sort_values(["ts_index", "code", "sub_code"]).reset_index(drop=True)

    # Drop NaN targets
    valid_mask = df["y_target"].notna()
    if not valid_mask.all():
        n_drop = (~valid_mask).sum()
        print(f"   Dropping {n_drop} rows with missing target.")
        df = df[valid_mask].copy().reset_index(drop=True)

    # Identify feature columns
    exclude = {"id", "y_target", "weight", "ts_index", "t1_end"}
    all_features = [c for c in df.columns if c not in exclude]

    # Separate categorical vs continuous features
    cont_features = [c for c in all_features if c not in CAT_COLS]

    # Drop constant columns
    const_cols = [c for c in cont_features if df[c].nunique() <= 1]
    if const_cols:
        print(f"   Dropping {len(const_cols)} constant columns.")
        cont_features = [c for c in cont_features if c not in const_cols]

    print(f"   Continuous features: {len(cont_features)}")
    print(f"   Categorical features: {len(CAT_COLS)}")

    # Encode categoricals to integer codes
    cat_encoders = {}
    cat_cardinalities = []
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
        cat_encoders[col] = df[col].cat.categories
        df[col] = df[col].cat.codes.astype(np.int32)
        # Handle potential -1 (NaN) codes
        if (df[col] < 0).any():
            df[col] = df[col] + 1  # shift so -1 -> 0
            cat_cardinalities.append(len(cat_encoders[col]) + 1)
        else:
            cat_cardinalities.append(len(cat_encoders[col]))

    print(f"   Cat cardinalities: {dict(zip(CAT_COLS, cat_cardinalities))}")

    return df, cont_features, cat_cardinalities, cat_encoders


def get_fold_data(df, train_idx, val_idx, cont_features):
    """
    Extract train/val arrays. Leak-free: compute fill values ONLY from train fold.
    Pipeline: Grouped Imputation → Symmetric Log1p → Z-score → Clip.
    """
    X_cont_tr = df.iloc[train_idx][cont_features].values.astype(np.float32)
    X_cont_va = df.iloc[val_idx][cont_features].values.astype(np.float32)

    X_cat_tr = df.iloc[train_idx][CAT_COLS].values.astype(np.int64)
    X_cat_va = df.iloc[val_idx][CAT_COLS].values.astype(np.int64)

    y_tr = df.iloc[train_idx]["y_target"].values.astype(np.float32)
    y_va = df.iloc[val_idx]["y_target"].values.astype(np.float32)

    w_tr = df.iloc[train_idx]["weight"].values.astype(np.float32)
    w_va = df.iloc[val_idx]["weight"].values.astype(np.float32)

    # ── Leak-free NaN imputation: grouped by sub_category (train fold only) ──
    # 1) Global fallback median (from train fold)
    global_medians = np.nanmedian(X_cont_tr, axis=0)
    global_medians = np.where(np.isnan(global_medians), 0.0, global_medians)

    # 2) Per-sub_category median (from train fold only)
    # sub_category is the 3rd categorical column (index 2 in CAT_COLS)
    sub_cat_col_idx = CAT_COLS.index("sub_category")
    sub_cat_tr = X_cat_tr[:, sub_cat_col_idx]  # integer-encoded sub_category for train
    sub_cat_va = X_cat_va[:, sub_cat_col_idx]  # integer-encoded sub_category for val

    unique_sub_cats = np.unique(sub_cat_tr)
    # Build median lookup: {sub_cat_code -> median_array}
    group_medians = {}
    for sc in unique_sub_cats:
        mask = sub_cat_tr == sc
        sc_medians = np.nanmedian(X_cont_tr[mask], axis=0)
        # Fallback to global if a feature is all-NaN within this group
        sc_medians = np.where(np.isnan(sc_medians), global_medians, sc_medians)
        group_medians[sc] = sc_medians

    # 3) Apply grouped imputation to train
    nan_mask_tr = np.isnan(X_cont_tr)
    if nan_mask_tr.any():
        for sc in unique_sub_cats:
            row_mask = (sub_cat_tr == sc)
            combined = row_mask[:, np.newaxis] & nan_mask_tr
            if combined.any():
                fill_vals = group_medians[sc]
                # Broadcast fill: for each column, fill NaN rows of this group
                X_cont_tr = np.where(combined, fill_vals[np.newaxis, :], X_cont_tr)

    # 4) Apply grouped imputation to val (fallback to global for unseen sub_cats)
    nan_mask_va = np.isnan(X_cont_va)
    if nan_mask_va.any():
        unique_sub_cats_va = np.unique(sub_cat_va)
        for sc in unique_sub_cats_va:
            row_mask = (sub_cat_va == sc)
            combined = row_mask[:, np.newaxis] & nan_mask_va
            if combined.any():
                fill_vals = group_medians.get(sc, global_medians)  # fallback
                X_cont_va = np.where(combined, fill_vals[np.newaxis, :], X_cont_va)

    # ── Symmetric Log1p dampener (compress fat tails before normalization) ──
    X_cont_tr = np.sign(X_cont_tr) * np.log1p(np.abs(X_cont_tr))
    X_cont_va = np.sign(X_cont_va) * np.log1p(np.abs(X_cont_va))

    # ── Leak-free normalization: mean/std from training fold only ──
    tr_mean = X_cont_tr.mean(axis=0)
    tr_std = X_cont_tr.std(axis=0) + 1e-8  # avoid division by zero

    X_cont_tr = (X_cont_tr - tr_mean) / tr_std
    X_cont_va = (X_cont_va - tr_mean) / tr_std

    # Clip extreme values after normalization to prevent gradient explosion
    X_cont_tr = np.clip(X_cont_tr, -10.0, 10.0)
    X_cont_va = np.clip(X_cont_va, -10.0, 10.0)

    # ── Normalize weights to prevent extremely large loss ──
    # Scale so that mean weight = 1.0 (using train weights only)
    w_mean = w_tr.mean() + 1e-12
    w_tr = w_tr / w_mean
    w_va = w_va / w_mean

    # ── Target Scaling: amplify gradient signal ──
    # y_target mean ≈ -0.67, std ≈ 32.5 → too small for NN to learn effectively.
    # Scale up 10,000x to escape the zero-prediction trap.
    # The metric (weighted RMSE ratio) is scale-invariant: S² cancels in num/denom.
    TARGET_SCALE = 10000.0
    y_tr = y_tr * TARGET_SCALE
    y_va = y_va * TARGET_SCALE

    return (X_cont_tr, X_cat_tr, y_tr, w_tr,
            X_cont_va, X_cat_va, y_va, w_va,
            global_medians, tr_mean, tr_std, w_mean)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_cont, x_cat, y, w in loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(x_cont, x_cat)
        loss = criterion(preds, y, w)
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return predictions, targets, weights (as float64 numpy to prevent overflow)."""
    model.eval()
    all_preds, all_y, all_w = [], [], []

    for x_cont, x_cat, y, w in loader:
        x_cont = x_cont.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)

        preds = model(x_cont, x_cat)
        # Cast to float64 immediately to prevent massive sum-of-squares precision loss in metric
        all_preds.append(preds.cpu().numpy().astype(np.float64))
        all_y.append(y.numpy().astype(np.float64))
        all_w.append(w.numpy().astype(np.float64))

    return np.concatenate(all_preds), np.concatenate(all_y), np.concatenate(all_w)


def train_fold(
    fold_num,
    X_cont_tr, X_cat_tr, y_tr, w_tr,
    X_cont_va, X_cat_va, y_va, w_va,
    w_va_raw,  # un-normalized weights for metric
    n_cont_features, cat_cardinalities,
    hparams, device,
):
    """Train a single fold and return (model_state_dict, val_score)."""

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

    # Compute embedding dims
    emb_dims = [min(dim, hparams["max_emb_dim"]) for dim in
                [max(4, card // 2) for card in cat_cardinalities]]

    model = HedgeFundMLP(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        emb_dims=emb_dims,
        hidden_dim=hparams["hidden_dim"],
        n_blocks=hparams["n_blocks"],
        dropout=hparams["dropout"],
    ).to(device)

    # Khởi tạo trung bình của Layer Output đúng với WEIGHTED Mean của Target để diệt "Hố đen số 0"
    TARGET_SCALE = 10000.0
    with torch.no_grad():
        weighted_target_mean = np.sum(w_tr * y_tr) / np.sum(w_tr)
        model.head[-1].bias.fill_(weighted_target_mean)

    # Scale Delta tỉ lệ thuận với TARGET_SCALE để PseudoHuberLoss duy trì vùng Smooth L2
    scaled_delta = hparams["delta"] * TARGET_SCALE
    criterion = WeightedPseudoHuberLoss(delta=scaled_delta)

    # Chia nhóm Parameter để Không áp weight_decay lên các hằng số Bias và BatchNorm
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

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=hparams.get("T_0", 10), T_mult=2, eta_min=1e-6
    )

    best_score = -np.inf
    best_state = None
    patience_counter = 0
    patience = hparams.get("patience", 15)

    n_epochs = hparams["epochs"]
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            grad_clip=hparams.get("grad_clip", 1.0)
        )

        # Evaluate with the competition metric
        preds_va, targets_va, weights_va = evaluate(model, val_loader, device)
        val_score = weighted_rmse_score(targets_va, preds_va, w_va_raw)

        elapsed = time.time() - t0

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

        if patience_counter >= patience:
            print(f"  Fold {fold_num} | Early stopping at epoch {epoch} (patience={patience})")
            break

    print(f"  Fold {fold_num} | Best Val Score: {best_score:.6f}")
    return best_state, best_score


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def optuna_objective(trial, df_sample, cont_features, cat_cardinalities, device):
    """Optuna objective using hold-out split on the sample."""

    hparams = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 384, 512, 768]),
        "n_blocks": trial.suggest_int("n_blocks", 2, 5),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "delta": trial.suggest_float("delta", 5.0, 100.0),
        "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
        "max_emb_dim": trial.suggest_categorical("max_emb_dim", [16, 32, 50]),
        "grad_clip": trial.suggest_float("grad_clip", 0.5, 5.0),
        "T_0": trial.suggest_int("T_0", 5, 15),
        "epochs": 30,  # Short for tuning
        "patience": 8,
    }

    # Hold-out split: train on first 80% of timestamps, val on last 20%
    unique_ts = np.sort(df_sample["ts_index"].unique())
    split_idx = int(len(unique_ts) * 0.8)
    split_ts = unique_ts[split_idx]

    tr_mask_base = df_sample["ts_index"] < split_ts
    va_mask = df_sample["ts_index"] >= split_ts

    # Purge overlap
    va_start = df_sample[va_mask]["ts_index"].min()
    overlap_mask = tr_mask_base & (df_sample["t1_end"] > va_start)
    tr_mask = tr_mask_base & (~overlap_mask)

    train_idx = np.where(tr_mask.values)[0]
    val_idx = np.where(va_mask.values)[0]

    (X_cont_tr, X_cat_tr, y_tr, w_tr,
     X_cont_va, X_cat_va, y_va, w_va,
     _, _, _, w_mean) = get_fold_data(df_sample, train_idx, val_idx, cont_features)

    # Raw weights for metric (un-normalized)
    w_va_raw = df_sample.iloc[val_idx]["weight"].values.astype(np.float64)

    _, val_score = train_fold(
        fold_num=f"T{trial.number}",
        X_cont_tr=X_cont_tr, X_cat_tr=X_cat_tr, y_tr=y_tr, w_tr=w_tr,
        X_cont_va=X_cont_va, X_cat_va=X_cat_va, y_va=y_va, w_va=w_va,
        w_va_raw=w_va_raw,
        n_cont_features=len(cont_features),
        cat_cardinalities=cat_cardinalities,
        hparams=hparams, device=device,
    )

    return val_score


def run_tuning(df, cont_features, cat_cardinalities, device, n_trials=15):
    """Run Optuna tuning on the last 20% of the time window."""
    import optuna
    import logging

    print("\n3. Starting Optuna Hyperparameter Tuning...")

    optuna_log_file = os.path.join(LOG_DIR, "mlp_optuna.log")
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(optuna_log_file))

    # Take last 20% of timestamps
    unique_ts = np.sort(df["ts_index"].unique())
    cut_off_idx = int(len(unique_ts) * 0.8)
    cut_off_time = unique_ts[cut_off_idx]
    df_sample = df[df["ts_index"] >= cut_off_time].copy().reset_index(drop=True)
    df_sample = df_sample.sort_values("ts_index").reset_index(drop=True)
    print(f"   Tuning sample: {len(df_sample)} rows ({len(df_sample)/len(df)*100:.1f}%)")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optuna_objective(trial, df_sample, cont_features, cat_cardinalities, device),
        n_trials=n_trials,
    )

    print(f"\n   Best trial: {study.best_trial.number}")
    print(f"   Best value: {study.best_trial.value:.6f}")
    print("   Best params:")
    for key, value in study.best_trial.params.items():
        print(f"      {key}: {value}")

    # Save
    param_path = os.path.join(PARAM_DIR, "mlp_best_param.json")
    with open(param_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"\n   Best parameters saved to {param_path}")

    del df_sample
    return study.best_trial.params


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def run_mlp(file_path, tuning=False, training=True):
    """Main entry: prepare data → (optional) tune → K-Fold train → evaluate."""

    df, cont_features, cat_cardinalities, cat_encoders = prepare_data(file_path)
    print(f"   Final shape: {df.shape}")

    # ── Hyperparameters ──
    param_path = os.path.join(PARAM_DIR, "mlp_best_param.json")

    if tuning:
        best_params = run_tuning(df, cont_features, cat_cardinalities, DEVICE)
    elif os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)
        print(f"\n   Loaded parameters from {param_path}")
    else:
        print(f"\n   Warning: {param_path} not found. Using sensible defaults.")
        best_params = {
            "hidden_dim": 512,
            "n_blocks": 3,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "delta": 15.0,
            "batch_size": 4096,
            "max_emb_dim": 32,
            "grad_clip": 1.0,
            "T_0": 10,
        }

    # Add training-specific defaults that Optuna doesn't tune
    hparams = {
        **best_params,
        "epochs": best_params.get("epochs", 80),
        "patience": best_params.get("patience", 15),
    }

    print(f"\n   Hyperparameters: {json.dumps(hparams, indent=4)}")

    if not training:
        print("\n   Skipping training.")
        return

    # ── Purged K-Fold CV ──
    print(f"\n4. Starting Leak-Free Purged K-Fold Training (5 folds)...")

    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    cv_meta = df[["ts_index", "t1_end"]].copy()

    fold_scores = []
    fold = 1

    for train_idx, val_idx in cv.split(cv_meta):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

        # Raw weights for the metric (MUST use float64, un-normalized)
        w_va_raw = df.iloc[val_idx]["weight"].values.astype(np.float64)

        (X_cont_tr, X_cat_tr, y_tr, w_tr,
         X_cont_va, X_cat_va, y_va, w_va,
         train_medians, tr_mean, tr_std, w_mean) = get_fold_data(
            df, train_idx, val_idx, cont_features
        )

        best_state, val_score = train_fold(
            fold_num=fold,
            X_cont_tr=X_cont_tr, X_cat_tr=X_cat_tr, y_tr=y_tr, w_tr=w_tr,
            X_cont_va=X_cont_va, X_cat_va=X_cat_va, y_va=y_va, w_va=w_va,
            w_va_raw=w_va_raw,
            n_cont_features=len(cont_features),
            cat_cardinalities=cat_cardinalities,
            hparams=hparams, device=DEVICE,
        )

        fold_scores.append(val_score)

        # Save the model + preprocessing params for each fold
        save_dict = {
            "model_state_dict": best_state,
            "hparams": hparams,
            "cont_features": cont_features,
            "cat_cardinalities": cat_cardinalities,
            "train_medians": train_medians,
            "tr_mean": tr_mean,
            "tr_std": tr_std,
            "w_mean": w_mean,
        }
        model_path = os.path.join(PARAM_DIR, f"mlp_fold_{fold}.pt")
        torch.save(save_dict, model_path)
        print(f"  Saved model to {model_path}")

        # Free memory
        del X_cont_tr, X_cat_tr, y_tr, w_tr
        del X_cont_va, X_cat_va, y_va, w_va
        del best_state

        fold += 1

    # ── Recency-weighted CV Score (later folds = closer to present = more important) ──
    cv_weights = np.array([0.0625, 0.0625, 0.125, 0.25, 0.5])
    weighted_mean_score = np.average(fold_scores, weights=cv_weights)
    simple_mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {s:.6f}  (weight: {cv_weights[i-1]})")
    print(f"\n  ==> SIMPLE MEAN CV SCORE:   {simple_mean_score:.6f} (+/- {std_score:.6f})")
    print(f"  ==> WEIGHTED MEAN CV SCORE: {weighted_mean_score:.6f} (recency-weighted)")
    print(f"{'='*60}")

    # Save report
    report_path = os.path.join(LOG_DIR, "mlp_cv_report.txt")
    with open(report_path, "w") as f:
        f.write("=== MLP CV EVALUATION ===\n")
        f.write("Method: Leak-Free Purged K-Fold (5 Folds)\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"Hyperparameters:\n{json.dumps(hparams, indent=4)}\n\n")
        for i, s in enumerate(fold_scores, 1):
            f.write(f"Fold {i}: {s:.6f}  (weight: {cv_weights[i-1]})\n")
        f.write(f"\nSimple Mean CV Score (R): {simple_mean_score:.6f} (+/- {std_score:.6f})\n")
        f.write(f"Weighted Mean CV Score (R): {weighted_mean_score:.6f} (recency-weighted)\n")
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Model for Hedge Fund Kaggle")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before training")
    parser.add_argument("--no-train", action="store_true", help="Skip training (tune only)")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna trials")
    args = parser.parse_args()

    file_path = os.path.join(DATA_DIR, "train_feature_builded.parquet")
    run_mlp(file_path, tuning=args.tune, training=not args.no_train)
