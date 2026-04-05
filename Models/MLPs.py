"""
MLP Model for Hedge Fund Kaggle Competition (Multi-Task Architecture)
=============================================
Architecture (Multi-Task):
  - Shared input features (code, sub_code, sub_category)
  - Multi-head output for 4 horizons (h1, h3, h10, h25)
  - Masked Weighted Pseudo-Huber Loss (Smooth L1)
  - Leak-Free Purged K-Fold Cross-Validation with Wide Pivot
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
print(f"[MLP-MTA] Using device: {DEVICE}")

# ── Categorical column definitions (horizon is now an Output Task) ──
CAT_COLS = ["code", "sub_code", "sub_category"]
HORIZONS = [1, 3, 10, 25]


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
# 2. MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class HedgeFundMLP(nn.Module):
    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: list,
        emb_dims: list,
        hidden_dim: int = 512,
        n_blocks: int = 3,
        dropout: float = 0.15,
        n_tasks: int = 4,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, dim) for card, dim in zip(cat_cardinalities, emb_dims)
        ])
        self.emb_drop = nn.Dropout(0.3)  # Slightly reduced for MTA stabilization
        total_emb_dim = sum(emb_dims)
        input_dim = n_cont_features + total_emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_tasks),
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
        embs = []
        for i, emb in enumerate(self.embeddings):
            idx = x_cat[:, i]
            idx = torch.clamp(idx, min=0, max=emb.num_embeddings - 1)
            embs.append(emb(idx))

        emb_concat = torch.cat(embs, dim=1)
        emb_concat = self.emb_drop(emb_concat)
        x = torch.cat([x_cont, emb_concat], dim=1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOSS: Masked Weighted Pseudo-Huber (Multi-Task)
# ═══════════════════════════════════════════════════════════════════════════════
class MaskedWeightedPseudoHuberLoss(nn.Module):
    def __init__(self, delta=15.0):
        super().__init__()
        self.delta = delta

    def forward(self, preds, targets, weights):
        mask = ~torch.isnan(targets)
        targets_safe = torch.where(mask, targets, torch.zeros_like(targets))
        weights_safe = torch.where(mask, weights, torch.zeros_like(weights))
        
        d = preds - targets_safe
        z = d / self.delta
        loss = self.delta ** 2 * (torch.sqrt(1.0 + z ** 2) - 1.0)
        
        weighted_loss = (weights_safe * loss)
        total_loss = weighted_loss.sum()
        total_weight = weights_safe.sum() + 1e-12
        return total_loss / total_weight


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA PREPARATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_data(file_path):
    print("1. Loading dataset...")
    df = pd.read_parquet(file_path)
    print(f"   Original shape: {df.shape}")

    exclude = {"id", "y_target", "weight", "ts_index", "horizon", "t1_end"}
    all_features = [c for c in df.columns if c not in exclude]
    cont_features = [c for c in all_features if c not in CAT_COLS]

    print("   Pivoting dataset to Wide (Multi-Task) format...")
    df["t1_end"] = df["ts_index"] + df["horizon"]
    pivot_idx = ["ts_index", "code", "sub_code", "sub_category"]
    
    targets_wide = df.pivot(index=pivot_idx, columns="horizon", values="y_target")
    weights_wide = df.pivot(index=pivot_idx, columns="horizon", values="weight")
    t1_end_wide = df.groupby(pivot_idx)["t1_end"].max().rename("t1_end")
    
    print("   Aggregating features across horizons (Wide Concatenation)...")
    # To preserve signal, we concatenate features from all horizons
    features_wide = df.pivot(index=pivot_idx, columns="horizon", values=cont_features)
    features_wide.columns = [f"{c[0]}_h{c[1]}" for c in features_wide.columns]
    
    df_wide = pd.concat([targets_wide.add_prefix("y_h"), 
                         weights_wide.add_prefix("w_h"), 
                         t1_end_wide], axis=1)
    df_wide = df_wide.join(features_wide).reset_index()
    
    y_cols = [f"y_h{h}" for h in HORIZONS]
    w_cols = [f"w_h{h}" for h in HORIZONS]
    feature_cols = [c for c in df_wide.columns if c not in pivot_idx and c not in y_cols and c not in w_cols and c != "t1_end"]
    
    df_wide = df_wide[df_wide[y_cols].notna().any(axis=1)].reset_index(drop=True)
    
    cat_encoders = {}
    cat_cardinalities = []
    for col in CAT_COLS:
        df_wide[col] = df_wide[col].astype("category")
        cat_encoders[col] = df_wide[col].cat.categories
        df_wide[col] = df_wide[col].cat.codes.astype(np.int32)
        if (df_wide[col] < 0).any():
            df_wide[col] = df_wide[col] + 1
            cat_cardinalities.append(len(cat_encoders[col]) + 1)
        else:
            cat_cardinalities.append(len(cat_encoders[col]))

    print(f"   MTA Wide shape: {df_wide.shape}")
    print(f"   Input Features: {len(feature_cols)}")
    return df_wide, feature_cols, y_cols, w_cols, cat_cardinalities, cat_encoders


def get_fold_data(df, train_idx, val_idx, feature_cols, y_cols, w_cols):
    X_cont_tr = df.iloc[train_idx][feature_cols].values.astype(np.float32)
    X_cont_va = df.iloc[val_idx][feature_cols].values.astype(np.float32)

    X_cat_tr = df.iloc[train_idx][CAT_COLS].values.astype(np.int64)
    X_cat_va = df.iloc[val_idx][CAT_COLS].values.astype(np.int64)

    y_tr = df.iloc[train_idx][y_cols].values.astype(np.float32)
    y_va = df.iloc[val_idx][y_cols].values.astype(np.float32)
    
    # Target Clipping to avoid extreme outliers collapsing the model
    # Financial targets are ~0.003 std. 0.05 is ~16 sigma.
    CLIP_VAL = 0.05 
    y_tr = np.clip(y_tr, -CLIP_VAL, CLIP_VAL)
    y_va = np.clip(y_va, -CLIP_VAL, CLIP_VAL)
    
    TARGET_SCALE = 1000.0 # Reduced from 10000 to avoid FP16/32 overflow during squared loss
    y_tr = y_tr * TARGET_SCALE
    y_va = y_va * TARGET_SCALE

    w_tr = df.iloc[train_idx][w_cols].values.astype(np.float32)
    w_va = df.iloc[val_idx][w_cols].values.astype(np.float32)

    # Simple global median for MTA (grouped is too slow with 760 features)
    global_medians = np.nanmedian(X_cont_tr, axis=0)
    global_medians = np.where(np.isnan(global_medians), 0.0, global_medians)
    
    X_cont_tr = np.where(np.isnan(X_cont_tr), global_medians, X_cont_tr)
    X_cont_va = np.where(np.isnan(X_cont_va), global_medians, X_cont_va)

    X_cont_tr = np.sign(X_cont_tr) * np.log1p(np.abs(X_cont_tr))
    X_cont_va = np.sign(X_cont_va) * np.log1p(np.abs(X_cont_va))

    tr_mean = X_cont_tr.mean(axis=0)
    tr_std = X_cont_tr.std(axis=0) + 1e-8

    X_cont_tr = (X_cont_tr - tr_mean) / tr_std
    X_cont_va = (X_cont_va - tr_mean) / tr_std
    X_cont_tr = np.clip(X_cont_tr, -10.0, 10.0)
    X_cont_va = np.clip(X_cont_va, -10.0, 10.0)

    # Normalize weights per task
    w_means = np.nanmean(w_tr, axis=0) + 1e-12
    w_tr = w_tr / w_means
    w_va = w_va / w_means

    TARGET_SCALE = 10000.0
    y_tr = y_tr * TARGET_SCALE
    y_va = y_va * TARGET_SCALE

    return (X_cont_tr, X_cat_tr, y_tr, w_tr,
            X_cont_va, X_cat_va, y_va, w_va,
            global_medians, tr_mean, tr_std, w_means)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x_cont, x_cat, y, w in loader:
        x_cont, x_cat, y, w = x_cont.to(device), x_cat.to(device), y.to(device), w.to(device)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x_cont, x_cat)
        loss = criterion(preds, y, w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    if scheduler is not None: scheduler.step()
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_mta(model, loader, device):
    model.eval()
    all_preds, all_y, all_w = [], [], []
    for x_cont, x_cat, y, w in loader:
        x_cont, x_cat = x_cont.to(device), x_cat.to(device)
        preds = model(x_cont, x_cat)
        all_preds.append(preds.cpu().numpy().astype(np.float64))
        all_y.append(y.numpy().astype(np.float64))
        all_w.append(w.numpy().astype(np.float64))
    return np.concatenate(all_preds), np.concatenate(all_y), np.concatenate(all_w)


def train_fold(fold_num, X_cont_tr, X_cat_tr, y_tr, w_tr, X_cont_va, X_cat_va, y_va, w_va, w_va_raw, n_cont_features, cat_cardinalities, hparams, device):
    train_loader = FastTensorDataLoader(torch.as_tensor(X_cont_tr), torch.as_tensor(X_cat_tr), torch.as_tensor(y_tr), torch.as_tensor(w_tr), batch_size=hparams["batch_size"], shuffle=True)
    val_loader = FastTensorDataLoader(torch.as_tensor(X_cont_va), torch.as_tensor(X_cat_va), torch.as_tensor(y_va), torch.as_tensor(w_va), batch_size=hparams["batch_size"]*2, shuffle=False)

    emb_dims = [min(dim, hparams["max_emb_dim"]) for dim in [max(4, card // 2) for card in cat_cardinalities]]
    model = HedgeFundMLP(n_cont_features=n_cont_features, cat_cardinalities=cat_cardinalities, emb_dims=emb_dims, hidden_dim=hparams["hidden_dim"], n_blocks=hparams["n_blocks"], dropout=hparams["dropout"]).to(device)

    TARGET_SCALE = 10000.0
    with torch.no_grad():
        # Init bias with task-wise means
        for t in range(4):
            valid_mask = ~np.isnan(y_tr[:, t])
            if valid_mask.any():
                m = np.average(y_tr[valid_mask, t], weights=w_tr[valid_mask, t])
                model.head[-1].bias[t].fill_(m)

    # Fix delta scaling: hparams['delta'] is now absolute in scaled space
    criterion = MaskedWeightedPseudoHuberLoss(delta=hparams["delta"]) 
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hparams.get("T_0", 10), T_mult=2)

    best_score, best_state, patience_counter = -np.inf, None, 0
    patience = hparams.get("patience", 15)

    for epoch in range(1, hparams["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, grad_clip=hparams.get("grad_clip", 1.0))
        preds_va, targets_va, weights_va = evaluate_mta(model, val_loader, device)
        
        # Calculate score per horizon and average
        task_scores = []
        for t in range(4):
            mask = ~np.isnan(targets_va[:, t])
            if mask.any():
                s = weighted_rmse_score(targets_va[mask, t], preds_va[mask, t], w_va_raw[mask, t])
                task_scores.append(s)
        val_score = np.mean(task_scores) if task_scores else 0.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Fold {fold_num} | Ep {epoch:2d} | Loss: {train_loss:.4f} | Val Score: {val_score:.6f} | {time.time()-t0:.1f}s")

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience: break

    print(f"  Fold {fold_num} | Best Val Score: {best_score:.6f}")
    return best_state, best_score


def optuna_objective(trial, df_sample, feature_cols, y_cols, w_cols, cat_cardinalities, device):
    hparams = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 768]),
        "n_blocks": trial.suggest_int("n_blocks", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "delta": trial.suggest_float("delta", 10.0, 50.0),
        "batch_size": trial.suggest_categorical("batch_size", [4096, 8192]),
        "max_emb_dim": 32, "epochs": 25, "patience": 8
    }
    split_ts = np.sort(df_sample["ts_index"].unique())[int(len(df_sample["ts_index"].unique()) * 0.8)]
    tr_mask, va_mask = df_sample["ts_index"] < split_ts, df_sample["ts_index"] >= split_ts
    train_idx, val_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
    (X_cont_tr, X_cat_tr, y_tr, w_tr, X_cont_va, X_cat_va, y_va, w_va, _, _, _, _) = get_fold_data(df_sample, train_idx, val_idx, feature_cols, y_cols, w_cols)
    w_va_raw = df_sample.iloc[val_idx][w_cols].values.astype(np.float64)
    _, val_score = train_fold(f"T{trial.number}", X_cont_tr, X_cat_tr, y_tr, w_tr, X_cont_va, X_cat_va, y_va, w_va, w_va_raw, len(feature_cols), cat_cardinalities, hparams, device)
    return val_score


def run_mlp(file_path, tuning=False, training=True):
    df, feature_cols, y_cols, w_cols, cat_cardinalities, cat_encoders = prepare_data(file_path)
    param_path = os.path.join(PARAM_DIR, "mlp_mta_param.json")

    if tuning:
        import optuna
        cut_off = np.sort(df["ts_index"].unique())[int(len(df["ts_index"].unique()) * 0.8)]
        df_sample = df[df["ts_index"] >= cut_off].copy().reset_index(drop=True)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: optuna_objective(t, df_sample, feature_cols, y_cols, w_cols, cat_cardinalities, DEVICE), n_trials=10)
        best_params = study.best_trial.params
        with open(param_path, "w") as f: json.dump(best_params, f, indent=4)
    elif os.path.exists(param_path):
        with open(param_path, "r") as f: best_params = json.load(f)
    else:
        best_params = {"hidden_dim": 512, "n_blocks": 3, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "delta": 20.0, "batch_size": 4096}

    hparams = {
        "max_emb_dim": 32,  # Default
        "epochs": 80,
        "patience": 15,
        "delta": 20.0,      # Default
        **best_params
    }
    if not training: return

    cv = LeakFreePurgedKFold(n_splits=5, pctEmbargo=0.01)
    fold_scores, fold = [], 1
    for train_idx, val_idx in cv.split(df[["ts_index", "t1_end"]]):
        print(f"\nFOLD {fold}"); w_va_raw = df.iloc[val_idx][w_cols].values.astype(np.float64)
        (X_tr, C_tr, y_tr, w_tr, X_va, C_va, y_va, w_va, _, mean, std, w_m) = get_fold_data(df, train_idx, val_idx, feature_cols, y_cols, w_cols)
        state, score = train_fold(fold, X_tr, C_tr, y_tr, w_tr, X_va, C_va, y_va, w_va, w_va_raw, len(feature_cols), cat_cardinalities, hparams, DEVICE)
        fold_scores.append(score)
        torch.save({"state": state, "mean": mean, "std": std, "feature_cols": feature_cols}, os.path.join(PARAM_DIR, f"mlp_mta_fold_{fold}.pt"))
        fold += 1

    print(f"\nFinal MTA CV Score: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()
    run_mlp(os.path.join(DATA_DIR, "train_feature_builded.parquet"), tuning=args.tune, training=not args.no_train)
