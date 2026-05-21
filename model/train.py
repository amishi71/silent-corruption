"""
train.py
Training loop for TabularAE on clean detector data.

Trains ONLY on train_clean.csv (40k rows).
Splits that into 90% train / 10% val internally.
Saves model checkpoint when val loss is best (early stopping).
Selects anomaly threshold from reconstruction error distribution on training data.

Outputs:
  model/ae_checkpoint.pt       — best model weights
  model/threshold.json         — anomaly threshold at 95th / 99th / 99.5th pct
  model/train_history.json     — per-epoch train/val loss for plotting

Run:
    python model/train.py
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.preprocess import load_and_transform
from model.autoencoder import TabularAE


# ── hyperparameters ────────────────────────────────────────────────────────────

N_FEATURES   = 8
BOTTLENECK   = 4       # n_features // 2
DROPOUT      = 0.1
LR           = 1e-3
BATCH_SIZE   = 256
MAX_EPOCHS   = 150
PATIENCE     = 15      # stop if val loss doesn't improve for this many epochs
VAL_SPLIT    = 0.10
SEED         = 42


def train(
    train_csv:  Path = Path("data/train_clean.csv"),
    out_dir:    Path = Path("model"),
) -> None:
    out_dir = Path(out_dir)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── load + transform ───────────────────────────────────────────────────────
    print("Loading and transforming training data...")
    X = load_and_transform(train_csv)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print(f"  Shape: {X_tensor.shape}  (n_rows × n_features)")

    # ── train / val split ──────────────────────────────────────────────────────
    dataset   = TensorDataset(X_tensor)
    n_val     = int(len(dataset) * VAL_SPLIT)
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    # ── model, loss, optimiser ─────────────────────────────────────────────────
    model     = TabularAE(n_features=N_FEATURES, bottleneck=BOTTLENECK, dropout=DROPOUT)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters  |  bottleneck={BOTTLENECK}")

    # ── training loop ──────────────────────────────────────────────────────────
    print(f"\n{'Epoch':>6}  {'Train loss':>12}  {'Val loss':>12}  {'Status'}")
    print("─" * 55)

    history        = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    patience_count = 0
    best_ckpt_path = out_dir / "ae_checkpoint.pt"
    t_start        = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):

        # train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimiser.zero_grad()
            out  = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                out      = model(batch)
                val_loss += criterion(out, batch).item() * len(batch)
        val_loss /= n_val

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))

        # early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            model.save(best_ckpt_path)
            status = "✓ saved"
        else:
            patience_count += 1
            status = f"patience {patience_count}/{PATIENCE}"

        if epoch % 10 == 0 or epoch <= 5 or patience_count == PATIENCE:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {status}")

        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  (best val loss: {best_val_loss:.6f})")
            break

    elapsed = time.perf_counter() - t_start
    print(f"\n  Training time: {elapsed:.1f}s")

    # ── threshold selection ────────────────────────────────────────────────────
    # Load best checkpoint and compute reconstruction errors on ALL train data.
    # Threshold = percentile of training errors.
    # This is the decision boundary — rows above this are anomaly hypotheses.
    print("\nComputing reconstruction errors on training data for threshold selection...")

    best_model = TabularAE.load(best_ckpt_path)
    best_model.eval()

    with torch.no_grad():
        errors = best_model.reconstruction_error(X_tensor).numpy()

    pcts = {
        "p95":   float(np.percentile(errors, 95)),
        "p99":   float(np.percentile(errors, 99)),
        "p99_5": float(np.percentile(errors, 99.5)),
    }

    print(f"\n  Reconstruction error distribution (training data):")
    print(f"    mean : {errors.mean():.6f}")
    print(f"    std  : {errors.std():.6f}")
    print(f"    min  : {errors.min():.6f}")
    print(f"    max  : {errors.max():.6f}")
    print(f"\n  Thresholds:")
    print(f"    95th  percentile : {pcts['p95']:.6f}  → flags ~5%  of training data")
    print(f"    99th  percentile : {pcts['p99']:.6f}  → flags ~1%  of training data  ← default")
    print(f"    99.5th percentile: {pcts['p99_5']:.6f}  → flags ~0.5% of training data")

    threshold_path = out_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "thresholds":          pcts,
            "default":             "p99",
            "train_error_mean":    float(errors.mean()),
            "train_error_std":     float(errors.std()),
            "best_val_loss":       best_val_loss,
            "bottleneck":          BOTTLENECK,
            "n_features":          N_FEATURES,
        }, f, indent=2)

    history_path = out_dir / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  threshold.json    → {threshold_path}")
    print(f"  train_history.json → {history_path}")
    print(f"  Default threshold (p99): {pcts['p99']:.6f}")
    print(f"  Run model/infer.py next to score eval_corrupted.csv")


if __name__ == "__main__":
    train()