"""
generator.py
Produces 50,000 rows of clean particle detector telemetry.

Domain: simulated particle detector — 8 detectors, 16 channels each.
All physical invariants hold for 100% of rows by construction.
The clean dataset is split BEFORE injection into:
  - train_clean.csv  (rows 0–39,999)
  - eval_clean.csv   (rows 40,000–49,999)

Run:
    python data/generator.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

SEED               = 42
N_ROWS             = 50_000
TRAIN_SIZE         = 40_000

REGISTERED_DETECTORS   = list(range(1, 9))   # IDs 1–8
CHANNELS_PER_DETECTOR  = 16                   # channel IDs 0–15
HIT_THRESHOLD_KEV      = 5.0                  # below this: hit_multiplicity must be 0
ENERGY_MAX_KEV         = 500.0
ENERGY_MIN_KEV         = 0.1
TEMP_NOMINAL_K         = 293.15               # 20°C
TEMP_MAX_RATE_K        = 0.3                  # K per event — rate-of-change invariant
AMP_TOLERANCE_MV       = 3.0                  # signal_amplitude residual tolerance (6-sigma)


# ── main ───────────────────────────────────────────────────────────────────────

def generate(seed: int = SEED, n_rows: int = N_ROWS, out_dir: Path = Path("data")) -> tuple:
    """Generate clean dataset. Returns (train_df, eval_df)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-channel physical parameters (fixed for this run) ──────────────────
    channel_keys = [
        (d, c)
        for d in REGISTERED_DETECTORS
        for c in range(CHANNELS_PER_DETECTOR)
    ]
    gain        = {k: float(rng.uniform(0.85, 1.15))  for k in channel_keys}  # mV/keV
    noise_floor = {k: float(rng.uniform(2.0,  8.0))   for k in channel_keys}  # mV baseline

    # Save for use by rules engine and AE preprocessing
    channel_params = {
        f"{k[0]}_{k[1]}": {"gain": gain[k], "noise_floor": noise_floor[k]}
        for k in channel_keys
    }
    with open(out_dir / "channel_params.json", "w") as f:
        json.dump(channel_params, f, indent=2)

    # ── vectorised event generation ────────────────────────────────────────────

    # detector and channel assignment
    det_ids = rng.choice(REGISTERED_DETECTORS, size=n_rows).astype(np.int32)
    ch_ids  = rng.integers(0, CHANNELS_PER_DETECTOR, size=n_rows).astype(np.int32)

    # timestamps: Poisson inter-arrival, minimum 100 ns gap
    inter_arrival = np.maximum(
        rng.exponential(scale=10_000, size=n_rows).astype(np.int64),
        100,
    )
    timestamps = np.cumsum(inter_arrival)

    # energy: mixture — 15% signal peak at ~120 keV, 85% exponential background
    is_signal   = rng.random(size=n_rows) < 0.15
    bg_energy   = rng.exponential(scale=18.0, size=n_rows)
    sig_energy  = rng.normal(loc=120.0, scale=15.0, size=n_rows)
    energy      = np.where(is_signal, sig_energy, bg_energy)
    energy      = np.clip(energy, ENERGY_MIN_KEV, ENERGY_MAX_KEV)

    # hit multiplicity: 0 below threshold, 1 + Poisson above
    above_thresh    = energy >= HIT_THRESHOLD_KEV
    lam             = np.clip(energy / 50.0, 0.0, None)
    poisson_part    = rng.poisson(lam=lam).astype(np.int32)
    hit_multiplicity = np.where(above_thresh, 1 + poisson_part, 0).astype(np.int32)

    # build lookup arrays for gain and noise_floor (indexed by [detector_id, channel_id])
    gain_arr  = np.zeros((max(REGISTERED_DETECTORS) + 1, CHANNELS_PER_DETECTOR))
    noise_arr = np.zeros_like(gain_arr)
    for k in channel_keys:
        gain_arr[k[0], k[1]]  = gain[k]
        noise_arr[k[0], k[1]] = noise_floor[k]

    channel_gain  = gain_arr[det_ids, ch_ids]
    channel_noise = noise_arr[det_ids, ch_ids]

    # signal amplitude: gain * energy + noise_floor + gaussian readout noise
    readout_noise    = rng.normal(0.0, 0.5, size=n_rows)
    signal_amplitude = channel_gain * energy + channel_noise + readout_noise
    signal_amplitude = np.maximum(signal_amplitude, channel_noise * 0.9)

    # temperature: Ornstein-Uhlenbeck mean-reverting process
    temps      = np.empty(n_rows)
    temps[0]   = TEMP_NOMINAL_K + rng.normal(0.0, 0.5)
    ou_noise   = rng.normal(0.0, 0.05, size=n_rows)
    theta      = 0.01
    for i in range(1, n_rows):
        temps[i] = temps[i - 1] + theta * (TEMP_NOMINAL_K - temps[i - 1]) + ou_noise[i]

    # run status: init (first 10), closed (last 5), active otherwise
    run_status              = np.full(n_rows, "active", dtype=object)
    run_status[:10]         = "init"
    run_status[n_rows - 5:] = "closed"

    # ── assemble dataframe ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "event_id":            np.arange(n_rows, dtype=np.int32),
        "timestamp_ns":        timestamps,
        "detector_id":         det_ids,
        "channel_id":          ch_ids,
        "energy_deposit_keV":  energy.round(4),
        "hit_multiplicity":    hit_multiplicity,
        "signal_amplitude_mV": signal_amplitude.round(4),
        "noise_floor_mV":      channel_noise.round(4),
        "temperature_K":       temps.round(4),
        "run_status":          run_status,
    })

    # ── verify all invariants before saving ────────────────────────────────────
    print("Verifying invariants on generated data...")
    _verify_generator_invariants(df, gain_arr, noise_arr, strict=True)

    # ── split and save ─────────────────────────────────────────────────────────
    train_df = df.iloc[:TRAIN_SIZE].reset_index(drop=True)
    eval_df  = df.iloc[TRAIN_SIZE:].reset_index(drop=True)

    train_df.to_csv(out_dir / "train_clean.csv", index=False)
    eval_df.to_csv(out_dir  / "eval_clean.csv",  index=False)

    print(f"\nGenerated {n_rows:,} rows.")
    print(f"  train_clean.csv : {len(train_df):,} rows")
    print(f"  eval_clean.csv  : {len(eval_df):,} rows")
    print(f"  channel_params.json saved")

    return train_df, eval_df


# ── invariant verification ─────────────────────────────────────────────────────

def _verify_generator_invariants(
    df: pd.DataFrame,
    gain_arr: np.ndarray,
    noise_arr: np.ndarray,
    strict: bool = True,
) -> list[str]:
    """
    Check every physical invariant on df.
    Returns list of violation descriptions (empty = all OK).
    Raises ValueError if strict=True and any violations exist.
    """
    violations = []

    det = df["detector_id"].values.astype(int)
    ch  = df["channel_id"].values.astype(int)

    # 1. Timestamps strictly increasing
    if not (np.diff(df["timestamp_ns"].values) > 0).all():
        n_bad = (np.diff(df["timestamp_ns"].values) <= 0).sum()
        violations.append(f"timestamp_ns not strictly increasing ({n_bad} pairs)")

    # 2. Registered detector IDs
    bad_det = ~np.isin(det, REGISTERED_DETECTORS)
    if bad_det.any():
        violations.append(f"unregistered detector_id: {np.unique(det[bad_det])} ({bad_det.sum()} rows)")

    # 3. Valid channel IDs
    bad_ch = (ch < 0) | (ch >= CHANNELS_PER_DETECTOR)
    if bad_ch.any():
        violations.append(f"invalid channel_id ({bad_ch.sum()} rows)")

    # 4. Energy bounds
    energy = df["energy_deposit_keV"].values
    bad_energy = (energy <= 0) | (energy > ENERGY_MAX_KEV)
    if bad_energy.any():
        violations.append(f"energy_deposit_keV out of (0, {ENERGY_MAX_KEV}] ({bad_energy.sum()} rows)")

    # 5. Hit multiplicity rule
    mult       = df["hit_multiplicity"].values
    above      = energy >= HIT_THRESHOLD_KEV
    bad_mult   = (above & (mult == 0)) | (~above & (mult != 0))
    if bad_mult.any():
        violations.append(f"hit_multiplicity inconsistent with energy ({bad_mult.sum()} rows)")

    # 6. Temperature rate-of-change
    temp_diff = np.abs(np.diff(df["temperature_K"].values))
    bad_temp  = temp_diff > TEMP_MAX_RATE_K
    if bad_temp.any():
        violations.append(f"temperature_K rate-of-change exceeded ({bad_temp.sum()} transitions)")

    # 7. Signal amplitude consistent with energy (within tolerance)
    # Safe indexing: clamp det/ch to array bounds (invalid IDs already caught above)
    safe_det = np.clip(det, 0, gain_arr.shape[0] - 1)
    safe_ch  = np.clip(ch,  0, gain_arr.shape[1] - 1)
    expected  = gain_arr[safe_det, safe_ch] * energy + noise_arr[safe_det, safe_ch]
    residuals = np.abs(df["signal_amplitude_mV"].values - expected)
    bad_amp   = residuals > AMP_TOLERANCE_MV
    if bad_amp.any():
        violations.append(
            f"signal_amplitude_mV vs energy residual > {AMP_TOLERANCE_MV} mV ({bad_amp.sum()} rows)"
        )

    if violations:
        msg = "Invariant violations:\n  " + "\n  ".join(violations)
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
    else:
        print("  All invariants OK ✓")

    return violations


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate()
