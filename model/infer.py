"""
infer.py
Runs the trained autoencoder on eval_corrupted.csv.
Produces per-row anomaly scores and flags rows above threshold.

Per-feature reconstruction error is logged for each flagged row —
this is the AE's bridge toward interpretability.

Outputs:
  model/scores_ae.csv — one row per eval row:
      row_index, recon_error, flagged_p95, flagged_p99, flagged_p99_5,
      top_feature (field with highest reconstruction error)

Run:
    python model/infer.py
"""

import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from model.preprocess import load_and_transform, FEATURE_COLS
from model.autoencoder import TabularAE


def infer(
    eval_csv:       Path = Path("data/eval_corrupted.csv"),
    checkpoint:     Path = Path("model/ae_checkpoint.pt"),
    threshold_json: Path = Path("model/threshold.json"),
    out_dir:        Path = Path("model"),
) -> pd.DataFrame:

    out_dir = Path(out_dir)

    # ── load threshold ─────────────────────────────────────────────────────────
    with open(threshold_json) as f:
        tdata = json.load(f)
    thresholds = tdata["thresholds"]

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading model checkpoint...")
    model = TabularAE.load(checkpoint)
    model.eval()
    print(f"  Bottleneck: {model.bottleneck}  |  Features: {model.n_features}")

    # ── load + transform eval data ─────────────────────────────────────────────
    print(f"\nLoading {eval_csv}...")
    X = load_and_transform(eval_csv)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print(f"  {X_tensor.shape[0]:,} rows  ×  {X_tensor.shape[1]} features")

    # ── inference ──────────────────────────────────────────────────────────────
    print("\nRunning inference...")
    t0 = time.perf_counter()

    with torch.no_grad():
        recon_errors    = model.reconstruction_error(X_tensor).numpy()
        per_feat_errors = model.per_feature_error(X_tensor).numpy()  # (n, n_features)

    elapsed      = time.perf_counter() - t0
    rows_per_sec = X_tensor.shape[0] / elapsed

    # ── top contributing feature per row ───────────────────────────────────────
    top_feature_idx = np.argmax(per_feat_errors, axis=1)
    top_feature     = [FEATURE_COLS[i] for i in top_feature_idx]

    # ── assemble scores DataFrame ──────────────────────────────────────────────
    scores = pd.DataFrame({
        "row_index":      np.arange(len(recon_errors)),
        "recon_error":    recon_errors.round(8),
        "flagged_p95":    recon_errors > thresholds["p95"],
        "flagged_p99":    recon_errors > thresholds["p99"],
        "flagged_p99_5":  recon_errors > thresholds["p99_5"],
        "top_feature":    top_feature,
    })

    # Add per-feature error columns for analysis
    for i, col in enumerate(FEATURE_COLS):
        scores[f"err_{col}"] = per_feat_errors[:, i].round(8)

    out_path = out_dir / "scores_ae.csv"
    scores.to_csv(out_path, index=False)

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  AUTOENCODER RESULTS")
    print(f"{'─' * 60}")
    print(f"  Rows scored      : {len(scores):,}")
    print(f"  Elapsed          : {elapsed*1000:.1f} ms")
    print(f"  Throughput       : {rows_per_sec:,.0f} rows/sec")

    print(f"\n  {'Threshold':<12}  {'Cutoff':>10}  {'Flagged':>8}  {'Flag rate':>10}")
    print(f"  {'─'*50}")
    for key, label in [("p95","95th pct"), ("p99","99th pct (default)"), ("p99_5","99.5th pct")]:
        flagged   = int(scores[f"flagged_{key}"].sum())
        flag_rate = flagged / len(scores) * 100
        print(f"  {label:<20}  {thresholds[key]:>10.6f}  {flagged:>8,}  {flag_rate:>9.1f}%")

    print(f"\n  Reconstruction error distribution (eval data):")
    print(f"    mean : {recon_errors.mean():.6f}")
    print(f"    std  : {recon_errors.std():.6f}")
    print(f"    max  : {recon_errors.max():.6f}")

    print(f"\n  Top feature distribution (flagged rows at p99):")
    flagged_mask = scores["flagged_p99"]
    if flagged_mask.any():
        top_counts = scores.loc[flagged_mask, "top_feature"].value_counts()
        for feat, count in top_counts.items():
            print(f"    {feat:<30} {count:>5} flagged rows")

    print(f"\n  scores_ae.csv → {out_path}")

    return scores


if __name__ == "__main__":
    infer()