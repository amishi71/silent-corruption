"""
profile.py
Sanity-check the clean splits before moving to Phase 2.

Checks:
  - Shape and null counts
  - Per-field stats (mean, std, min, max) — train vs eval should be similar
  - Invariant quick-checks on both splits
  - Corruption profile of eval_corrupted.csv (without peeking at labels)

Run:
    python data/profile.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

REGISTERED_DETECTORS  = list(range(1, 9))
CHANNELS_PER_DETECTOR = 16
HIT_THRESHOLD_KEV     = 5.0
TEMP_MAX_RATE_K       = 0.3

NUMERIC_COLS = [
    "energy_deposit_keV",
    "hit_multiplicity",
    "signal_amplitude_mV",
    "noise_floor_mV",
    "temperature_K",
]


def profile(data_dir: Path = Path("data")) -> None:
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train_clean.csv")
    eval_ = pd.read_csv(data_dir / "eval_clean.csv")

    _section("SHAPES")
    print(f"  train_clean : {train.shape[0]:>6,} rows × {train.shape[1]} cols")
    print(f"  eval_clean  : {eval_.shape[0]:>6,} rows × {eval_.shape[1]} cols")

    _section("NULL COUNTS")
    total_nulls = train.isna().sum().sum() + eval_.isna().sum().sum()
    if total_nulls == 0:
        print("  No nulls in either split ✓")
    else:
        for col in train.columns:
            tn = train[col].isna().sum()
            en = eval_[col].isna().sum()
            if tn > 0 or en > 0:
                print(f"  {col:<30} train={tn}  eval={en}")

    _section("NUMERIC STATS (train | eval)")
    fmt = "  {:<28}  mean: {:>10.4f} / {:>10.4f}   std: {:>8.4f} / {:>8.4f}   min: {:>8.4f} / {:>8.4f}"
    for col in NUMERIC_COLS:
        print(fmt.format(
            col,
            train[col].mean(), eval_[col].mean(),
            train[col].std(),  eval_[col].std(),
            train[col].min(),  eval_[col].min(),
        ))

    _section("CATEGORICAL DISTRIBUTIONS")
    for col in ["run_status"]:
        tv = dict(train[col].value_counts().sort_index())
        ev = dict(eval_[col].value_counts().sort_index())
        print(f"  {col}")
        print(f"    train: {tv}")
        print(f"    eval:  {ev}")

    det_train = sorted(train["detector_id"].unique().tolist())
    det_eval  = sorted(eval_["detector_id"].unique().tolist())
    print(f"  detector_id unique values")
    print(f"    train: {det_train}")
    print(f"    eval:  {det_eval}")

    _section("TIMESTAMP RANGES")
    print(f"  train: {train['timestamp_ns'].min():>16,}  →  {train['timestamp_ns'].max():>16,}")
    print(f"  eval:  {eval_['timestamp_ns'].min():>16,}  →  {eval_['timestamp_ns'].max():>16,}")
    print(f"  gap between splits: {eval_['timestamp_ns'].min() - train['timestamp_ns'].max():,} ns")

    _section("INVARIANT CHECKS")
    _check_invariants(train, "train_clean")
    _check_invariants(eval_,  "eval_clean")

    _section("CORRUPTED EVAL SURFACE STATS")
    _profile_corrupted(data_dir)

    print()


def _check_invariants(df: pd.DataFrame, name: str) -> None:
    results = {}

    # 1. Timestamps monotone
    results["timestamp_ns strictly increasing"] = \
        bool((np.diff(df["timestamp_ns"].values) > 0).all())

    # 2. Registered detectors
    results["detector_id all registered"] = \
        bool(df["detector_id"].isin(REGISTERED_DETECTORS).all())

    # 3. Channel IDs valid
    results["channel_id in [0, 15]"] = \
        bool(df["channel_id"].between(0, CHANNELS_PER_DETECTOR - 1).all())

    # 4. Energy positive
    results["energy_deposit_keV > 0"] = \
        bool((df["energy_deposit_keV"] > 0).all())

    # 5. Hit multiplicity rule
    above = df["energy_deposit_keV"] >= HIT_THRESHOLD_KEV
    mult  = df["hit_multiplicity"]
    bad_above = (above & (mult == 0)).sum()
    bad_below = (~above & (mult != 0)).sum()
    results["hit_multiplicity consistent with energy"] = \
        (bad_above == 0 and bad_below == 0)

    # 6. Temperature rate-of-change
    results["temperature_K rate-of-change < 0.3 K"] = \
        bool((np.abs(np.diff(df["temperature_K"].values)) < TEMP_MAX_RATE_K).all())

    all_ok = all(results.values())
    print(f"\n  [{name}]")
    for check, ok in results.items():
        print(f"    {'✓' if ok else '✗ FAIL'} {check}")
    if all_ok:
        print(f"    → All invariants satisfied")


def _profile_corrupted(data_dir: Path) -> None:
    """Show surface-level stats on corrupted eval — no labels peek."""
    corr_path = data_dir / "eval_corrupted.csv"
    if not corr_path.exists():
        print("  eval_corrupted.csv not found — run inject.py first")
        return

    c  = pd.read_csv(corr_path)
    e  = pd.read_csv(data_dir / "eval_clean.csv")

    print(f"  Rows with out-of-range energy  : {(c['energy_deposit_keV'] < 0).sum()} negative, "
          f"{(c['energy_deposit_keV'] > 500).sum()} > 500 keV")
    print(f"  Rows with unregistered det_id  : {(~c['detector_id'].isin(REGISTERED_DETECTORS)).sum()}")

    ts_diffs = np.diff(c["timestamp_ns"].values)
    print(f"  Timestamp non-increases        : {(ts_diffs <= 0).sum()}")

    temp_diffs = np.abs(np.diff(c["temperature_K"].values))
    print(f"  Temp rate-of-change violations : {(temp_diffs > TEMP_MAX_RATE_K).sum()}")

    # Energy distribution shift — mean comparison
    print(f"  Energy mean: clean={e['energy_deposit_keV'].mean():.3f}  "
          f"corrupted={c['energy_deposit_keV'].mean():.3f}")
    print(f"  Energy std:  clean={e['energy_deposit_keV'].std():.3f}  "
          f"corrupted={c['energy_deposit_keV'].std():.3f}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    profile()
