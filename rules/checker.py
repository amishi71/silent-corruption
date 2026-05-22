"""
checker.py
InvariantChecker: rules-based corruption detector for particle detector telemetry.

Loads all rule parameters from config.yaml — no thresholds in code.
Runs all checks in batch over a DataFrame and returns a flags DataFrame.

Output schema (flags DataFrame):
    row_index   int      positional index in the input DataFrame
    field       str      field that triggered the flag
    rule_type   str      which rule fired
    severity    int      1=LOW, 2=MEDIUM, 3=HIGH
    reason      str      human-readable explanation

Run on eval data:
    python rules/checker.py
    → produces rules/flags_rules.csv
"""

import json
import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


# ── Flag dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Flag:
    row_index: int
    field:     str
    rule_type: str
    severity:  int
    reason:    str


# ── InvariantChecker ───────────────────────────────────────────────────────────

class InvariantChecker:
    """
    Loads config from YAML, runs all defined checks, returns flags.

    Usage:
        checker = InvariantChecker("rules/config.yaml")
        flags_df = checker.check(df)
        checker.summary(flags_df)
    """

    def __init__(
        self,
        config_path: str | Path,
        channel_params_path: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Channel params for amplitude check — path from config or override
        cp_path = channel_params_path
        if cp_path is None and "cross_field_amplitude" in self.cfg:
            cp_path = self.cfg["cross_field_amplitude"].get("channel_params_path")

        self._gain:  np.ndarray | None = None
        self._noise: np.ndarray | None = None
        if cp_path is not None:
            self._load_channel_params(Path(cp_path))

    def _load_channel_params(self, path: Path) -> None:
        """
        Build gain_arr[det_id, ch_id] and noise_arr[det_id, ch_id]
        from channel_params.json for fast vectorised amplitude checks.
        """
        with open(path) as f:
            params = json.load(f)

        # Keys are "detectorId_channelId" strings
        max_det = max(int(k.split("_")[0]) for k in params) + 1
        max_ch  = max(int(k.split("_")[1]) for k in params) + 1
        self._gain  = np.zeros((max_det, max_ch))
        self._noise = np.zeros((max_det, max_ch))
        for key, vals in params.items():
            d, c = map(int, key.split("_"))
            self._gain[d, c]  = vals["gain"]
            self._noise[d, c] = vals["noise_floor"]

    # ── public interface ───────────────────────────────────────────────────────

    def check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all configured checks on df.
        df must have a 0-based integer index (use reset_index() if needed).
        Returns a DataFrame of flags — one row per flag raised.
        A single input row may produce multiple flags (different rules).
        """
        flags: list[Flag] = []
        cfg = self.cfg

        if "range_checks" in cfg:
            flags.extend(self._check_ranges(df, cfg["range_checks"]))

        if "registered_detectors" in cfg:
            flags.extend(self._check_registered_detectors(df, cfg["registered_detectors"]))

        if "channel_ids" in cfg:
            flags.extend(self._check_channel_ids(df, cfg["channel_ids"]))

        if "timestamp_monotone" in cfg:
            flags.extend(self._check_timestamp_monotone(df, cfg["timestamp_monotone"]))

        if "hit_multiplicity_rule" in cfg:
            flags.extend(self._check_hit_multiplicity(df, cfg["hit_multiplicity_rule"]))

        if "temperature_rate_of_change" in cfg:
            flags.extend(self._check_temp_rate(df, cfg["temperature_rate_of_change"]))

        if "cross_field_amplitude" in cfg and self._gain is not None:
            flags.extend(self._check_amplitude(df, cfg["cross_field_amplitude"]))

        if "z_score" in cfg:
            flags.extend(self._check_z_score(df, cfg["z_score"]))

        if not flags:
            return pd.DataFrame(
                columns=["row_index", "field", "rule_type", "severity", "reason"]
            )

        return pd.DataFrame([{
            "row_index": f.row_index,
            "field":     f.field,
            "rule_type": f.rule_type,
            "severity":  f.severity,
            "reason":    f.reason,
        } for f in flags])

    def summary(self, flags_df: pd.DataFrame) -> None:
        """Print a human-readable summary of a flags DataFrame."""
        if flags_df.empty:
            print("  No flags raised.")
            return

        total_flags = len(flags_df)
        unique_rows = flags_df["row_index"].nunique()

        print(f"  Total flags      : {total_flags:,}")
        print(f"  Unique rows      : {unique_rows:,}")
        print()

        by_rule = (
            flags_df.groupby("rule_type")
            .agg(flags=("row_index", "count"), rows=("row_index", "nunique"))
            .sort_values("flags", ascending=False)
        )
        print(f"  {'Rule type':<35}  Flags   Rows")
        print(f"  {'─' * 55}")
        for rule, row in by_rule.iterrows():
            print(f"  {rule:<35}  {row['flags']:<8} {row['rows']}")

        print()
        by_sev = flags_df.groupby("severity").size()
        labels = {3: "HIGH  ", 2: "MEDIUM", 1: "LOW   "}
        for sev in sorted(by_sev.index, reverse=True):
            print(f"  Sev {sev} ({labels.get(sev, str(sev))}): {by_sev[sev]:,} flags")

    # ── individual checks ──────────────────────────────────────────────────────

    def _check_ranges(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        flags = []
        for field, params in cfg.items():
            if field not in df.columns:
                continue
            vals = df[field].values.astype(float)
            sev  = int(params.get("severity", 2))

            if "min" in params:
                for idx in np.where(vals < params["min"])[0]:
                    flags.append(Flag(
                        int(idx), field, "range_check", sev,
                        f"{field}={vals[idx]:.4f} below min={params['min']}"
                    ))

            if "max" in params:
                for idx in np.where(vals > params["max"])[0]:
                    flags.append(Flag(
                        int(idx), field, "range_check", sev,
                        f"{field}={vals[idx]:.4f} > max={params['max']}"
                    ))
        return flags

    def _check_registered_detectors(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        valid = set(cfg["valid_ids"])
        sev   = int(cfg.get("severity", 3))
        bad   = np.where(~df["detector_id"].isin(valid).values)[0]
        return [
            Flag(
                int(i), "detector_id", "registered_detector", sev,
                f"detector_id={df['detector_id'].iloc[i]} not in registered set {sorted(valid)}"
            )
            for i in bad
        ]

    def _check_channel_ids(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        lo   = int(cfg.get("min", 0))
        hi   = int(cfg.get("max", 15))
        sev  = int(cfg.get("severity", 3))
        vals = df["channel_id"].values.astype(int)
        bad  = np.where((vals < lo) | (vals > hi))[0]
        return [
            Flag(
                int(i), "channel_id", "channel_id_check", sev,
                f"channel_id={vals[i]} outside [{lo}, {hi}]"
            )
            for i in bad
        ]

    def _check_timestamp_monotone(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        strict = bool(cfg.get("strict", True))
        sev    = int(cfg.get("severity", 3))
        ts     = df["timestamp_ns"].values.astype(np.int64)
        diffs  = np.diff(ts)
        # pair_idx is the earlier of the pair; violation is at pair_idx+1
        bad_pairs = np.where(diffs <= 0)[0] if strict else np.where(diffs < 0)[0]
        flags = []
        for pi in bad_pairs:
            row = int(pi + 1)
            flags.append(Flag(
                row, "timestamp_ns", "timestamp_monotone", sev,
                f"timestamp_ns={ts[row]} ≤ prev={ts[pi]} (delta={diffs[pi]})"
            ))
        return flags

    def _check_hit_multiplicity(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        threshold = float(cfg.get("energy_threshold_keV", 5.0))
        sev       = int(cfg.get("severity", 3))
        energy    = df["energy_deposit_keV"].values.astype(float)
        mult      = df["hit_multiplicity"].values.astype(int)
        above     = energy >= threshold
        # Violation: above threshold with zero hits, OR below threshold with hits
        bad = np.where(above & (mult == 0))[0]  # only: above threshold, zero hits
        return [
            Flag(
                int(i), "hit_multiplicity", "hit_multiplicity_rule", sev,
                f"energy={energy[i]:.4f} keV (threshold={threshold}), "
                f"hit_multiplicity={mult[i]}"
            )
            for i in bad
        ]

    def _check_temp_rate(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        max_delta = float(cfg.get("max_delta_K", 0.3))
        sev       = int(cfg.get("severity", 2))
        temps     = df["temperature_K"].values.astype(float)
        diffs     = np.abs(np.diff(temps))
        # Flag the later row in each violating pair
        bad_pairs = np.where(diffs > max_delta)[0]
        flags = []
        for pi in bad_pairs:
            row = int(pi + 1)
            flags.append(Flag(
                row, "temperature_K", "temperature_rate_of_change", sev,
                f"temp delta={diffs[pi]:.4f} K > max={max_delta} K "
                f"(row {pi}: {temps[pi]:.4f} K → row {row}: {temps[row]:.4f} K)"
            ))
        return flags

    def _check_amplitude(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        tolerance = float(cfg.get("tolerance_mV", 10.0))
        sev       = int(cfg.get("severity", 2))

        det = df["detector_id"].values.astype(int)
        ch  = df["channel_id"].values.astype(int)

        # Only check rows where detector and channel IDs are within array bounds.
        # Rows with invalid IDs are already flagged by other rules.
        valid = (
            (det >= 0) & (det < self._gain.shape[0]) &
            (ch  >= 0) & (ch  < self._gain.shape[1])
        )

        energy    = df["energy_deposit_keV"].values.astype(float)
        amplitude = df["signal_amplitude_mV"].values.astype(float)

        safe_det = np.clip(det, 0, self._gain.shape[0] - 1)
        safe_ch  = np.clip(ch,  0, self._gain.shape[1] - 1)

        expected  = self._gain[safe_det, safe_ch] * energy + self._noise[safe_det, safe_ch]
        residuals = np.abs(amplitude - expected)

        bad = np.where(valid & (residuals > tolerance))[0]
        return [
            Flag(
                int(i), "signal_amplitude_mV", "cross_field_amplitude", sev,
                f"amplitude={amplitude[i]:.4f} mV, expected≈{expected[i]:.4f} mV, "
                f"residual={residuals[i]:.4f} mV > tolerance={tolerance} mV"
            )
            for i in bad
        ]

    def _check_z_score(self, df: pd.DataFrame, cfg: dict) -> list[Flag]:
        window    = int(cfg.get("window_size", 500))
        threshold = float(cfg.get("threshold", 3.5))
        min_samp  = int(cfg.get("min_samples", 30))
        fields    = cfg.get("fields", [])
        sev       = int(cfg.get("severity", 1))

        flags = []
        for field in fields:
            if field not in df.columns:
                continue

            vals   = df[field].astype(float)
            roll   = vals.rolling(window=window, min_periods=min_samp)
            r_mean = roll.mean()
            # Population std (ddof=0): consistent with how we'd compute it in production
            r_std  = roll.std(ddof=0).replace(0.0, np.nan)

            z_arr  = ((vals - r_mean) / r_std).values  # numpy, positionally aligned

            bad = np.where(~np.isnan(z_arr) & (np.abs(z_arr) > threshold))[0]
            for i in bad:
                flags.append(Flag(
                    int(i), field, "z_score", sev,
                    f"{field} z={z_arr[i]:.2f} "
                    f"(window={window}, threshold=±{threshold}, "
                    f"value={vals.iloc[i]:.4f}, "
                    f"window_mean={r_mean.iloc[i]:.4f})"
                ))
        return flags


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir  = Path("data")
    rules_dir = Path("rules")

    eval_path = data_dir / "eval_corrupted.csv"
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run data/inject.py first.")
        sys.exit(1)

    print("Loading checker...")
    checker = InvariantChecker(rules_dir / "config.yaml")

    print(f"Loading {eval_path}...")
    df = pd.read_csv(eval_path)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("\nRunning rules engine...")
    t0       = time.perf_counter()
    flags_df = checker.check(df)
    elapsed  = time.perf_counter() - t0

    out_path = rules_dir / "flags_rules.csv"
    flags_df.to_csv(out_path, index=False)

    print(f"\n{'─' * 60}")
    print(f"  RESULTS")
    print(f"{'─' * 60}")
    checker.summary(flags_df)

    print(f"\n{'─' * 60}")
    print(f"  PERFORMANCE")
    print(f"{'─' * 60}")
    rows_per_sec = len(df) / elapsed
    print(f"  Rows processed   : {len(df):,}")
    print(f"  Elapsed          : {elapsed*1000:.1f} ms")
    print(f"  Throughput       : {rows_per_sec:,.0f} rows/sec")
    print()
    print(f"  Flags saved to   : {out_path}")
