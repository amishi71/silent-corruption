"""
preprocess.py
Fits the feature preprocessor on train_clean.csv and saves it to disk.

CRITICAL: Scaler is fitted on training data ONLY.
At inference time, load and apply — never refit on eval data.

Outputs:
  model/scaler.pkl        — fitted StandardScaler
  model/feature_cols.json — ordered feature column list

Run:
    python model/preprocess.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ── feature configuration ──────────────────────────────────────────────────────

# Columns dropped before preprocessing — not informative for anomaly detection
DROP_COLS = ["event_id", "timestamp_ns", "run_status"]

# Ordinal encoding for run_status (chronological order)
STATUS_MAP = {"init": 0, "active": 1, "closed": 2}

# Final feature order — explicitly fixed so train and infer always agree
FEATURE_COLS = [
    "detector_id",
    "channel_id",
    "energy_deposit_keV",
    "hit_multiplicity",
    "signal_amplitude_mV",
    "noise_floor_mV",
    "temperature_K",
    "run_status_enc",   # ordinal-encoded run_status
]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic encodings that don't require fitting.
    Returns a new DataFrame with encoded columns added.
    """
    out = df.copy()
    out["run_status_enc"] = out["run_status"].map(STATUS_MAP).astype(float)
    return out


def fit_scaler(
    train_csv: Path = Path("data/train_clean.csv"),
    out_dir:   Path = Path("model"),
) -> StandardScaler:
    """
    Fit StandardScaler on training data and save to disk.
    Returns the fitted scaler.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {train_csv}...")
    df = pd.read_csv(train_csv)
    print(f"  {len(df):,} rows")

    df_enc = encode(df)

    # Verify all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df_enc.columns]
    if missing:
        raise ValueError(f"Missing feature columns after encoding: {missing}")

    X = df_enc[FEATURE_COLS].values.astype(float)

    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X)

    # Save scaler and feature list
    scaler_path = out_dir / "scaler.pkl"
    cols_path   = out_dir / "feature_cols.json"

    joblib.dump(scaler, scaler_path)
    with open(cols_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"\nFeature stats (post-scaling should be mean≈0, std≈1):")
    X_scaled = scaler.transform(X)
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<30}  mean={X_scaled[:, i].mean():+.4f}  std={X_scaled[:, i].std():.4f}")

    print(f"\n  scaler.pkl       → {scaler_path}")
    print(f"  feature_cols.json → {cols_path}")

    return scaler


def load_and_transform(
    csv_path:    Path,
    scaler_path: Path = Path("model/scaler.pkl"),
    cols_path:   Path = Path("model/feature_cols.json"),
) -> np.ndarray:
    """
    Load a CSV, apply encoding, and transform using the saved scaler.
    Returns a numpy array of shape (n_rows, n_features).
    Used by train.py and infer.py.
    """
    with open(cols_path) as f:
        feature_cols = json.load(f)

    scaler = joblib.load(scaler_path)
    df     = pd.read_csv(csv_path)
    df_enc = encode(df)

    X = df_enc[feature_cols].values.astype(float)
    return scaler.transform(X)


if __name__ == "__main__":
    fit_scaler()