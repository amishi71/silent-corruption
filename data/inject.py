"""
inject.py
Injects 6 corruption types into eval_clean.csv.

Outputs:
  eval_corrupted.csv  — the dataset your detectors will run on
  labels.csv          — injected labels for the evaluation set

Each corruption type targets ~0.8% of eval rows (~80 rows / type).
Rows are only corrupted once — no overlap between types.

Corruption types:
  bit_flip             Large value error in energy_deposit (scale factor)
  stale_timestamp      Previous row's timestamp reused
  subtle_drift         3–5% energy shift — within range, statistically detectable
  cross_field          signal_amplitude randomised, uncorrelated with energy
  phantom_detector     detector_id replaced with unregistered ID (99)
  thermal_spike        Temperature spike exceeding rate-of-change limit

Run:
    python data/inject.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────

SEED           = 99           # different from generator seed
RATE_PER_TYPE  = 0.008        # 0.8% of eval rows per corruption type
PHANTOM_DET_ID = 99           # unregistered detector ID

CORRUPTION_TYPES = [
    "bit_flip",
    "stale_timestamp",
    "subtle_drift",
    "cross_field",
    "phantom_detector",
    "thermal_spike",
]


# ── main ───────────────────────────────────────────────────────────────────────

def inject(
    eval_csv: Path  = Path("data/eval_clean.csv"),
    out_dir: Path   = Path("data"),
    seed: int       = SEED,
    rate: float     = RATE_PER_TYPE,
) -> None:
    rng     = np.random.default_rng(seed)
    out_dir = Path(out_dir)

    df = pd.read_csv(eval_csv)
    n  = len(df)

    corrupted = df.copy()

    # labels: one row per eval row, corruption_type column
    labels = pd.DataFrame({
        "row_index":       range(n),
        "corruption_type": ["clean"] * n,
    })

    n_per_type   = max(1, int(n * rate))
    already_used = set()

    for ctype in CORRUPTION_TYPES:
        available   = [i for i in range(n) if i not in already_used]
        n_sample    = min(n_per_type, len(available))
        target_rows = sorted(
            rng.choice(available, size=n_sample, replace=False).tolist()
        )

        for idx in target_rows:
            affected = _apply(corrupted, idx, ctype, rng, len(df))
            if affected:
                for affected_idx in affected:
                    # Only label if still clean — don't overwrite a prior corruption type
                    if labels.loc[affected_idx, "corruption_type"] == "clean":
                        labels.loc[affected_idx, "corruption_type"] = ctype
                already_used.update(affected)

    # ── save outputs ───────────────────────────────────────────────────────────
    corrupted.to_csv(out_dir / "eval_corrupted.csv", index=False)
    labels.to_csv(out_dir    / "labels.csv",          index=False)

    # ── summary ────────────────────────────────────────────────────────────────
    counts         = labels["corruption_type"].value_counts()
    total_corrupted = (labels["corruption_type"] != "clean").sum()

    print(f"Injection complete.")
    print(f"  Total corrupted: {total_corrupted} / {n} ({100*total_corrupted/n:.1f}%)")
    print()
    print(f"  {'Type':<25} Rows")
    print(f"  {'-'*35}")
    for ctype in CORRUPTION_TYPES:
        print(f"  {ctype:<25} {counts.get(ctype, 0)}")
    print()
    print(f"  eval_corrupted.csv → data/")
    print(f"  labels.csv         → data/")


# ── corruption appliers ────────────────────────────────────────────────────────

def _apply(df: pd.DataFrame, idx: int, ctype: str, rng: np.random.Generator, n: int) -> list[int]:
    """
    Mutate df at row idx with the given corruption type.
    Returns a list of row indices that should be labeled as corrupted
    (empty list = could not apply this corruption to this row).
    Corruptions are designed to be plausible-looking, not cartoonish.
    """

    if ctype == "bit_flip":
        # Scale energy by a large factor — simulates a bit flip in the ADC register.
        # Deliberately leaves hit_multiplicity unchanged → cross-field inconsistency.
        # Real ADC bit flips set a high bit and produce large positive values.
        original = float(df.loc[idx, "energy_deposit_keV"])
        factor   = float(rng.choice([12.5, 25.0, 50.0]))
        df.loc[idx, "energy_deposit_keV"] = round(original * factor, 4)
        return [idx]

    elif ctype == "stale_timestamp":
        # Reuse previous row's timestamp — simulates a clock register not incrementing.
        if idx == 0:
            return []
        df.loc[idx, "timestamp_ns"] = int(df.loc[idx - 1, "timestamp_ns"])
        return [idx]

    elif ctype == "subtle_drift":
        # Shift energy by 3–5% in either direction — within range, hard to spot visually.
        # signal_amplitude is NOT updated, so a small cross-field residual also appears.
        # Primary signal is statistical: z-score drift over the window.
        # Skip rows with very low energy where clipping would neutralise the drift.
        original = float(df.loc[idx, "energy_deposit_keV"])
        if original < 1.0:
            return []
        pct         = float(rng.uniform(0.03, 0.05))
        direction   = float(rng.choice([1.0, -1.0]))
        drifted     = original * (1.0 + pct * direction)
        drifted     = round(float(np.clip(drifted, 0.1, 500.0)), 4)
        df.loc[idx, "energy_deposit_keV"] = drifted
        return [idx]

    elif ctype == "cross_field":
        # Signal amplitude set to a random value uncorrelated with energy.
        # Simulates a channel calibration table corruption — each field looks
        # individually plausible but the relationship is broken.
        random_amp = round(float(rng.uniform(5.0, 400.0)), 4)
        df.loc[idx, "signal_amplitude_mV"] = random_amp
        return [idx]

    elif ctype == "phantom_detector":
        # Replace detector_id with an unregistered ID.
        # Simulates a routing error in the DAQ system.
        df.loc[idx, "detector_id"] = PHANTOM_DET_ID
        return [idx]

    elif ctype == "thermal_spike":
        # Single-row temperature spike exceeding the rate-of-change limit.
        # Simulates a bad sensor read or thermal event.
        # A spike at idx creates two anomalous transitions: (idx-1)→idx and idx→(idx+1).
        # Label both idx and idx+1 so the rules engine's flag on the recovery step
        # is not counted as a false positive.
        if idx + 1 >= n:
            return []
        original = float(df.loc[idx, "temperature_K"])
        spike    = float(rng.uniform(3.0, 8.0)) * float(rng.choice([1.0, -1.0]))
        df.loc[idx, "temperature_K"] = round(original + spike, 4)
        return [idx, idx + 1]

    return []


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inject()
