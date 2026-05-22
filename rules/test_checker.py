"""
test_checker.py
Unit tests for InvariantChecker — one test per rule type.

Each test builds a minimal hand-crafted DataFrame, runs the checker,
and asserts on exactly which rows were flagged and why.

Run:
    python rules/test_checker.py
"""

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from rules.checker import InvariantChecker


# ── test helpers ───────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
_results: list[tuple[str, bool, str]] = []


def test(name: str):
    """Decorator that catches assertion errors and records pass/fail."""
    def decorator(fn):
        try:
            fn()
            _results.append((name, True, ""))
        except AssertionError as e:
            _results.append((name, False, str(e)))
        except Exception as e:
            _results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
        return fn
    return decorator


def _checker(extra: dict | None = None, channel_params: dict | None = None) -> InvariantChecker:
    """
    Build an InvariantChecker from a minimal base config.
    extra: additional config keys merged in (e.g. z_score, cross_field_amplitude)
    channel_params: optional dict written to a temp file
    """
    cfg = {
        "range_checks": {
            "energy_deposit_keV": {"min": 0.0, "max": 500.0, "severity": 3},
            "temperature_K":      {"min": 270.0, "max": 320.0, "severity": 2},
        },
        "registered_detectors": {"valid_ids": [1, 2, 3, 4, 5, 6, 7, 8], "severity": 3},
        "channel_ids":           {"min": 0, "max": 15, "severity": 3},
        "timestamp_monotone":    {"strict": True, "severity": 3},
        "hit_multiplicity_rule": {"energy_threshold_keV": 5.0, "severity": 3},
        "temperature_rate_of_change": {"max_delta_K": 0.3, "severity": 2},
    }
    if extra:
        cfg.update(extra)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    ) as f:
        yaml.dump(cfg, f)
        cfg_path = f.name

    # Write channel params to temp file if provided
    cp_path = None
    if channel_params is not None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir="/tmp"
        ) as f:
            json.dump(channel_params, f)
            cp_path = f.name

    return InvariantChecker(cfg_path, cp_path)


def _base_row(n: int = 1, start_ts: int = 1000) -> pd.DataFrame:
    """
    Generate n clean rows. All invariants hold.
    Detector 1, channel 0, gain=1.0, noise=5.0 for amplitude consistency.
    """
    return pd.DataFrame({
        "event_id":            range(n),
        "timestamp_ns":        [start_ts + i * 1000 for i in range(n)],
        "detector_id":         [1] * n,
        "channel_id":          [0] * n,
        "energy_deposit_keV":  [30.0] * n,
        "hit_multiplicity":    [1] * n,
        "signal_amplitude_mV": [35.0] * n,   # ≈ 1.0 * 30 + 5.0
        "noise_floor_mV":      [5.0] * n,
        "temperature_K":       [293.15] * n,
        "run_status":          ["active"] * n,
    })


def _cp_for_detector1_channel0(gain: float = 1.0, noise: float = 5.0) -> dict:
    """Minimal channel_params.json for detector 1, channel 0."""
    return {"1_0": {"gain": gain, "noise_floor": noise}}


# ── tests ──────────────────────────────────────────────────────────────────────

@test("range_check: clean row raises no flags")
def _():
    c  = _checker()
    df = _base_row(5)
    fl = c.check(df)
    assert fl.empty, f"Expected no flags on clean rows, got {len(fl)}"


@test("range_check: energy < 0 is flagged (severity 3)")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[1, "energy_deposit_keV"] = -1.0   # strictly negative — must be flagged
    fl = c.check(df)
    rc = fl[fl["rule_type"] == "range_check"]
    assert len(rc) == 1, f"Expected 1 range flag, got {len(rc)}"
    assert rc.iloc[0]["row_index"] == 1
    assert rc.iloc[0]["severity"]  == 3

@test("range_check: energy > 500 is flagged")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[2, "energy_deposit_keV"] = 501.0
    fl = c.check(df)
    rc = fl[fl["rule_type"] == "range_check"]
    assert any(rc["row_index"] == 2), "Row 2 not flagged for energy > 500"


@test("range_check: temperature out of bounds is flagged (severity 2)")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[0, "temperature_K"] = 269.0      # below 270
    fl = c.check(df)
    rc = fl[(fl["rule_type"] == "range_check") & (fl["row_index"] == 0)]
    assert len(rc) == 1
    assert rc.iloc[0]["severity"] == 2


@test("registered_detector: valid IDs raise no flags")
def _():
    c  = _checker()
    df = _base_row(4)
    df["detector_id"] = [1, 3, 5, 8]
    fl = c.check(df)
    rd = fl[fl["rule_type"] == "registered_detector"]
    assert rd.empty, f"Unexpected registered_detector flags: {rd}"


@test("registered_detector: unregistered ID 99 is flagged (severity 3)")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[1, "detector_id"] = 99
    fl = c.check(df)
    rd = fl[fl["rule_type"] == "registered_detector"]
    assert len(rd) == 1
    assert rd.iloc[0]["row_index"] == 1
    assert rd.iloc[0]["severity"]  == 3


@test("channel_id_check: valid range [0, 15] passes")
def _():
    c  = _checker()
    df = _base_row(4)
    df["channel_id"] = [0, 7, 15, 3]
    fl = c.check(df)
    assert fl[fl["rule_type"] == "channel_id_check"].empty


@test("channel_id_check: channel_id 16 is flagged")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[2, "channel_id"] = 16
    fl = c.check(df)
    ci = fl[fl["rule_type"] == "channel_id_check"]
    assert len(ci) == 1
    assert ci.iloc[0]["row_index"] == 2


@test("timestamp_monotone: strictly increasing passes")
def _():
    c  = _checker()
    df = _base_row(5)    # timestamps: 1000, 2000, 3000, 4000, 5000
    fl = c.check(df)
    assert fl[fl["rule_type"] == "timestamp_monotone"].empty


@test("timestamp_monotone: equal timestamps flagged at later row (stale clock)")
def _():
    c  = _checker()
    df = _base_row(4)
    df.loc[2, "timestamp_ns"] = df.loc[1, "timestamp_ns"]  # stale
    fl = c.check(df)
    ts = fl[fl["rule_type"] == "timestamp_monotone"]
    assert len(ts) == 1, f"Expected 1 timestamp flag, got {len(ts)}"
    assert ts.iloc[0]["row_index"] == 2   # later of the pair


@test("timestamp_monotone: decreasing timestamp flagged")
def _():
    c  = _checker()
    df = _base_row(4)
    df.loc[3, "timestamp_ns"] = df.loc[2, "timestamp_ns"] - 100
    fl = c.check(df)
    ts = fl[fl["rule_type"] == "timestamp_monotone"]
    assert any(ts["row_index"] == 3)


@test("hit_multiplicity_rule: energy=30, mult=1 passes")
def _():
    c  = _checker()
    df = _base_row(2)
    # row 0: energy=30 (above 5 keV), mult=1 — valid
    # row 1: energy=1  (below 5 keV), mult=0 — valid
    df.loc[1, "energy_deposit_keV"] = 1.0
    df.loc[1, "hit_multiplicity"]   = 0
    fl = c.check(df)
    hm = fl[fl["rule_type"] == "hit_multiplicity_rule"]
    assert hm.empty, f"Unexpected hit_mult flags: {hm}"


@test("hit_multiplicity_rule: energy above threshold with zero hits is flagged")
def _():
    c  = _checker()
    df = _base_row(3)
    df.loc[1, "energy_deposit_keV"] = 50.0
    df.loc[1, "hit_multiplicity"]   = 0    # violation
    fl = c.check(df)
    hm = fl[fl["rule_type"] == "hit_multiplicity_rule"]
    assert len(hm) == 1
    assert hm.iloc[0]["row_index"] == 1
    assert hm.iloc[0]["severity"]  == 3


@test("hit_multiplicity_rule: energy below threshold with hits is NOT flagged")
def _():
    # Regression test for fix_hit_multiplicity: low-energy events legitimately
    # register hits. The ~above & (mult != 0) condition was removed because it
    # generated 1,907 false positives on clean data. This behaviour is now correct.
    c  = _checker()
    df = _base_row(3)
    df.loc[0, "energy_deposit_keV"] = 2.0   # below threshold
    df.loc[0, "hit_multiplicity"]   = 1     # valid — low energy can still register hits
    fl = c.check(df)
    hm = fl[fl["rule_type"] == "hit_multiplicity_rule"]
    assert not any(hm["row_index"] == 0), \
        "Low-energy row with hits incorrectly flagged — fix_hit_multiplicity regression"


@test("temperature_rate_of_change: stable temperature passes")
def _():
    c  = _checker()
    df = _base_row(5)     # all 293.15
    fl = c.check(df)
    assert fl[fl["rule_type"] == "temperature_rate_of_change"].empty


@test("temperature_rate_of_change: spike flags the later row")
def _():
    c  = _checker()
    df = _base_row(5)
    df.loc[2, "temperature_K"] = 300.0   # 6.85 K jump from 293.15
    fl = c.check(df)
    tr = fl[fl["rule_type"] == "temperature_rate_of_change"]
    # spike at row 2 → flags row 2 (2 vs 1) and row 3 (3 vs 2)
    flagged_rows = set(tr["row_index"].tolist())
    assert 2 in flagged_rows, f"Row 2 (spike) not flagged. Flagged: {flagged_rows}"
    assert 3 in flagged_rows, f"Row 3 (recovery) not flagged. Flagged: {flagged_rows}"


@test("cross_field_amplitude: consistent amplitude passes")
def _():
    cp = _cp_for_detector1_channel0(gain=1.0, noise=5.0)
    c  = _checker(
        extra={"cross_field_amplitude": {"tolerance_mV": 10.0, "severity": 2}},
        channel_params=cp,
    )
    df = _base_row(3)
    # amplitude = 1.0 * 30 + 5.0 = 35.0 — exact, residual = 0
    fl = c.check(df)
    assert fl[fl["rule_type"] == "cross_field_amplitude"].empty


@test("cross_field_amplitude: random amplitude exceeds tolerance")
def _():
    cp = _cp_for_detector1_channel0(gain=1.0, noise=5.0)
    c  = _checker(
        extra={"cross_field_amplitude": {"tolerance_mV": 10.0, "severity": 2}},
        channel_params=cp,
    )
    df = _base_row(3)
    # energy=30 → expected=35 mV, but set amplitude to 200 mV → residual=165 mV
    df.loc[1, "signal_amplitude_mV"] = 200.0
    fl = c.check(df)
    xf = fl[fl["rule_type"] == "cross_field_amplitude"]
    assert len(xf) == 1
    assert xf.iloc[0]["row_index"] == 1
    assert xf.iloc[0]["severity"]  == 2


@test("cross_field_amplitude: invalid detector_id row skipped (not double-penalised)")
def _():
    cp = _cp_for_detector1_channel0(gain=1.0, noise=5.0)
    c  = _checker(
        extra={"cross_field_amplitude": {"tolerance_mV": 10.0, "severity": 2}},
        channel_params=cp,
    )
    df = _base_row(3)
    # Phantom detector — amplitude check should skip this row
    df.loc[1, "detector_id"] = 99
    fl = c.check(df)
    xf = fl[(fl["rule_type"] == "cross_field_amplitude") & (fl["row_index"] == 1)]
    assert xf.empty, "Amplitude check should not run on rows with invalid detector IDs"


@test("z_score: normal values produce no flags after cold start")
def _():
    c = _checker(extra={
        "z_score": {
            "window_size": 50,
            "threshold":   3.5,
            "min_samples": 10,
            "fields":      ["energy_deposit_keV"],
            "severity":    1,
        }
    })
    rng = np.random.default_rng(0)
    n   = 200
    df  = _base_row(n)
    # Mild random variation — no outliers
    df["energy_deposit_keV"] = np.clip(
        rng.normal(30.0, 2.0, size=n), 5.0, 100.0
    )
    df["timestamp_ns"] = np.arange(1000, 1000 + n * 1000, 1000)
    fl = c.check(df)
    zf = fl[fl["rule_type"] == "z_score"]
    assert zf.empty, f"Expected no z-score flags on normal data, got {len(zf)}"


@test("z_score: large spike is flagged after window fills")
def _():
    c = _checker(extra={
        "z_score": {
            "window_size": 50,
            "threshold":   3.5,
            "min_samples": 10,
            "fields":      ["energy_deposit_keV"],
            "severity":    1,
        }
    })
    n   = 100
    df  = _base_row(n)
    df["timestamp_ns"] = np.arange(1000, 1000 + n * 1000, 1000)
    # Rows 0–49: stable at 30 keV; row 80: extreme spike
    df["energy_deposit_keV"] = 30.0
    df.loc[80, "energy_deposit_keV"] = 500.0   # ~235-sigma
    df.loc[80, "hit_multiplicity"]   = 1        # keep hit_mult consistent
    fl = c.check(df)
    zf = fl[fl["rule_type"] == "z_score"]
    assert any(zf["row_index"] == 80), \
        f"Row 80 spike not flagged. Flagged rows: {zf['row_index'].tolist()}"


@test("z_score: cold start — no flags before min_samples rows")
def _():
    c = _checker(extra={
        "z_score": {
            "window_size": 50,
            "threshold":   3.5,
            "min_samples": 30,
            "fields":      ["energy_deposit_keV"],
            "severity":    1,
        }
    })
    n  = 40
    df = _base_row(n)
    df["timestamp_ns"] = np.arange(1000, 1000 + n * 1000, 1000)
    df["energy_deposit_keV"] = 30.0
    # Spike at row 5 — before the cold-start window fills
    df.loc[5, "energy_deposit_keV"] = 500.0
    df.loc[5, "hit_multiplicity"]   = 1
    fl = c.check(df)
    zf = fl[fl["rule_type"] == "z_score"]
    # Row 5 is at position 5, window requires 30 samples — should NOT be flagged
    assert not any(zf["row_index"] == 5), \
        "Row 5 flagged before cold-start window filled — min_samples not respected"

@test("range_check: hit_multiplicity=0 with low energy is NOT a false positive")
def _():
    # Regression test for the <= vs < bug in _check_ranges.
    # hit_multiplicity=0 is valid when energy is below threshold.
    # The lower-bound check must be strict (value < min), not (value <= min).
    c = _checker()
    df = _base_row(3)
    df["energy_deposit_keV"] = [1.0, 2.0, 3.0]   # all below 5 keV threshold
    df["hit_multiplicity"]   = [0, 0, 0]           # zero hits — valid for low energy
    fl = c.check(df)
    rc = fl[fl["rule_type"] == "range_check"]
    fp_rows = rc[rc["field"] == "hit_multiplicity"]["row_index"].tolist()
    assert len(fp_rows) == 0, (
        f"hit_multiplicity=0 incorrectly flagged as range violation on rows {fp_rows}. "
        f"Lower-bound check must use strict < not <=."
    )
# ── runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Force all decorated functions to run by re-executing the module body
    # (they already ran at import time via the decorator)

    print()
    print("─" * 60)
    print("  InvariantChecker unit tests")
    print("─" * 60)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    for name, ok, msg in _results:
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        if not ok:
            for line in msg.splitlines():
                print(f"       {line}")

    print()
    print(f"  {passed} passed  {failed} failed  ({len(_results)} total)")
    print()

    if failed > 0:
        sys.exit(1)
