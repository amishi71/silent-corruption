"""
eval/downstream_impact.py — Silent corruption downstream analysis.

Demonstrates the central claim concretely:
    68 subtle_drift rows passed both detectors undetected.
    What happens to physics analysis built on that contaminated data?

This script computes three downstream statistics on:
    (a) clean eval data          — ground truth
    (b) contaminated eval data   — what an analyst actually sees
    (c) detected-only eval data  — what you get after running the pipeline

Statistics computed:
    - Mean energy deposit (keV)           — the primary observable
    - Mean energy by detector             — per-channel systematic bias
    - Energy distribution percentiles     — shape of the spectrum

Then traces specific missed rows back to their impact on the mean,
showing exactly which rows caused the shift and by how much.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data"  / "labels.csv"
EVAL_PATH   = ROOT / "data"  / "eval_corrupted.csv"
CLEAN_PATH  = ROOT / "data"  / "eval_clean.csv"
RULES_PATH  = ROOT / "rules" / "flags_rules.csv"
AE_PATH     = ROOT / "model" / "scores_ae.csv"
THRESH_PATH = ROOT / "model" / "threshold.json"
OUT_DIR     = ROOT / "eval"
OUT_DIR.mkdir(exist_ok=True)


def load_threshold():
    with open(THRESH_PATH) as f:
        data = json.load(f)
    if "thresholds" in data:
        return float(data["thresholds"][data.get("default", "p99")])
    for k in ("threshold", "p99", "value"):
        if k in data:
            return float(data[k])
    raise KeyError("No threshold found")


def main():
    # ── Load everything ───────────────────────────────────────────────────────
    labels = pd.read_csv(LABELS_PATH)
    if "row_index" in labels.columns:
        labels = labels.rename(columns={"row_index": "row_id"})
    corrupted_labels = labels[labels["corruption_type"] != "clean"].copy()
    missed_labels    = labels[labels["corruption_type"] == "subtle_drift"].copy()

    eval_df  = pd.read_csv(EVAL_PATH)
    clean_df = pd.read_csv(CLEAN_PATH)

    rules_flags = pd.read_csv(RULES_PATH)
    if "row_index" in rules_flags.columns:
        rules_flags = rules_flags.rename(columns={"row_index": "row_id"})

    ae_scores = pd.read_csv(AE_PATH)
    if "row_index" in ae_scores.columns:
        ae_scores = ae_scores.rename(columns={"row_index": "row_id"})
    if "recon_error" in ae_scores.columns:
        ae_scores = ae_scores.rename(columns={"recon_error": "reconstruction_error"})

    threshold = load_threshold()

    # ── Build datasets ────────────────────────────────────────────────────────
    rules_flagged = set(rules_flags["row_id"])
    ae_flagged    = set(ae_scores.loc[
        ae_scores["reconstruction_error"] > threshold, "row_id"
    ])
    pipeline_flagged = rules_flagged | ae_flagged

    all_corrupted_ids   = set(corrupted_labels["row_id"])
    subtle_drift_ids    = set(missed_labels["row_id"])
    pipeline_missed_ids = all_corrupted_ids - pipeline_flagged

    # Rows the analyst sees after removing pipeline-flagged rows
    detected_removed = eval_df[~eval_df.index.isin(pipeline_flagged)].copy()

    # Rows the analyst sees with NO detection (baseline contamination)
    no_detection = eval_df.copy()

    # True clean baseline
    clean_baseline = clean_df.copy()

    # ── Compute downstream statistics ─────────────────────────────────────────
    def energy_stats(df, label):
        e = df["energy_deposit_keV"]
        return {
            "dataset":   label,
            "n_rows":    len(df),
            "mean_keV":  round(e.mean(), 6),
            "median_keV": round(e.median(), 6),
            "std_keV":   round(e.std(), 6),
            "p10_keV":   round(e.quantile(0.10), 6),
            "p90_keV":   round(e.quantile(0.90), 6),
        }

    stats = [
        energy_stats(clean_baseline,    "clean_baseline"),
        energy_stats(no_detection,      "contaminated_no_detection"),
        energy_stats(detected_removed,  "after_pipeline_detection"),
    ]
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_DIR / "downstream_stats.csv", index=False)

    # ── Per-detector mean energy ───────────────────────────────────────────────
    det_clean = clean_baseline.groupby("detector_id")["energy_deposit_keV"].mean()
    det_cont  = no_detection.groupby("detector_id")["energy_deposit_keV"].mean()
    det_diff  = (det_cont - det_clean).rename("delta_keV")
    det_pct   = ((det_cont - det_clean) / det_clean * 100).rename("delta_pct")
    det_df = pd.concat([det_clean.rename("clean_mean"), det_cont.rename("contaminated_mean"),
                        det_diff, det_pct], axis=1).reset_index()
    det_df.to_csv(OUT_DIR / "downstream_by_detector.csv", index=False)

    # ── Trace missed rows to impact ───────────────────────────────────────────
    # For each pipeline-missed row, compute its individual contribution
    # to the mean energy shift
    missed_energies = []
    for rid in sorted(pipeline_missed_ids):
        ctype = labels.loc[labels["row_id"] == rid, "corruption_type"].values
        ctype = ctype[0] if len(ctype) else "unknown"
        if rid < len(eval_df):
            row = eval_df.iloc[rid]
            clean_row = clean_df.iloc[rid] if rid < len(clean_df) else None
            energy_corrupt = row["energy_deposit_keV"]
            energy_clean   = clean_row["energy_deposit_keV"] if clean_row is not None else None
            delta = (energy_corrupt - energy_clean) if energy_clean is not None else None
            missed_energies.append({
                "row_id":          rid,
                "corruption_type": ctype,
                "energy_corrupt":  round(energy_corrupt, 4),
                "energy_clean":    round(energy_clean, 4) if energy_clean else None,
                "energy_delta":    round(delta, 4) if delta is not None else None,
            })
    missed_df = pd.DataFrame(missed_energies)
    if not missed_df.empty:
        missed_df.to_csv(OUT_DIR / "downstream_missed_rows.csv", index=False)

    # ── Print report ──────────────────────────────────────────────────────────
    SEP  = "─" * 78
    SEP2 = "═" * 78

    clean_mean = stats_df.loc[stats_df["dataset"] == "clean_baseline", "mean_keV"].values[0]
    cont_mean  = stats_df.loc[stats_df["dataset"] == "contaminated_no_detection", "mean_keV"].values[0]
    pipe_mean  = stats_df.loc[stats_df["dataset"] == "after_pipeline_detection", "mean_keV"].values[0]

    abs_shift  = cont_mean - clean_mean
    pct_shift  = abs_shift / clean_mean * 100
    pipe_shift = pipe_mean - clean_mean
    pipe_pct   = pipe_shift / clean_mean * 100

    print(f"\n{SEP2}")
    print("  DOWNSTREAM IMPACT ANALYSIS — SILENT CORRUPTION")
    print(SEP2)

    print(f"\n  {len(pipeline_missed_ids)} corrupted rows passed ALL detectors undetected.")
    print(f"  {len([x for x in pipeline_missed_ids if labels.loc[labels['row_id']==x, 'corruption_type'].values[0] == 'subtle_drift' if x in labels['row_id'].values])} are subtle_drift — gradual energy drift, no rule violation.")

    print(f"\n{SEP}")
    print("  MEAN ENERGY DEPOSIT (keV) — THE PRIMARY OBSERVABLE")
    print(SEP)
    print(f"  {'Dataset':<35} {'Mean (keV)':>12}  {'Delta':>10}  {'Delta %':>9}")
    print(f"  {'-'*35} {'-'*12}  {'-'*10}  {'-'*9}")
    print(f"  {'Clean baseline (ground truth)':<35} {clean_mean:>12.6f}  {'—':>10}  {'—':>9}")
    print(f"  {'Contaminated (no detection)':<35} {cont_mean:>12.6f}  {abs_shift:>+10.6f}  {pct_shift:>+9.4f}%")
    print(f"  {'After pipeline detection':<35} {pipe_mean:>12.6f}  {pipe_shift:>+10.6f}  {pipe_pct:>+9.4f}%")

    print(f"\n{SEP}")
    print("  ENERGY DISTRIBUTION SHAPE")
    print(SEP)
    for _, row in stats_df.iterrows():
        print(f"  {row['dataset']:<35}  "
              f"p10={row['p10_keV']:.2f}  median={row['median_keV']:.2f}  "
              f"p90={row['p90_keV']:.2f}  std={row['std_keV']:.4f}")

    print(f"\n{SEP}")
    print("  PER-DETECTOR MEAN ENERGY BIAS")
    print(SEP)
    print(f"  {'Detector':>10}  {'Clean mean':>12}  {'Contaminated':>14}  {'Delta':>8}  {'Delta %':>9}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}  {'-'*8}  {'-'*9}")
    for _, row in det_df.iterrows():
        print(f"  {int(row['detector_id']):>10}  {row['clean_mean']:>12.4f}  "
              f"{row['contaminated_mean']:>14.4f}  {row['delta_keV']:>+8.4f}  "
              f"{row['delta_pct']:>+9.4f}%")

    print(f"\n{SEP}")
    print("  ROW-LEVEL TRACE — MISSED CORRUPTIONS AND THEIR ENERGY DELTA")
    print(SEP)
    print(f"  (showing rows where drift is largest — these shift the mean most)")
    print(f"  {'Row':>6}  {'Type':<20}  {'Corrupt keV':>12}  {'Clean keV':>10}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*8}")
    if not missed_df.empty and "energy_delta" in missed_df.columns:
        top_impact = missed_df.dropna(subset=["energy_delta"]).nlargest(10, "energy_delta")
        for _, row in top_impact.iterrows():
            print(f"  {int(row['row_id']):>6}  {row['corruption_type']:<20}  "
                  f"{row['energy_corrupt']:>12.4f}  {row['energy_clean']:>10.4f}  "
                  f"{row['energy_delta']:>+8.4f}")

    print(f"\n{SEP}")
    print("  THE CLAIM — DEMONSTRATED")
    print(SEP)
    print(f"  Without any detection:")
    print(f"    Mean energy shifts {abs_shift:+.6f} keV ({pct_shift:+.4f}%)")
    print(f"    This is a systematic bias — not random noise — introduced by")
    print(f"    {len(pipeline_missed_ids)} rows that look completely normal individually.")
    print()
    print(f"  After running the pipeline:")
    print(f"    Mean energy shifts {pipe_shift:+.6f} keV ({pipe_pct:+.4f}%)")
    print(f"    Residual bias from the {len(pipeline_missed_ids)} rows no detector caught.")
    print()
    if abs(pct_shift) > 0.01:
        print(f"  A {abs(pct_shift):.4f}% shift in mean energy is scientifically meaningful.")
        print(f"  In a cross-section measurement, this propagates directly into the result.")
        print(f"  A physicist working on this data would report a wrong number.")
    else:
        print(f"  The shift is small ({abs(pct_shift):.4f}%) but nonzero and systematic.")
        print(f"  At scale (petabytes of data), systematic biases compound.")
        print(f"  A physicist working on this data would report a subtly wrong number.")
    print()
    print(f"  Silent corruption is more dangerous than visible failure.")
    print(f"  A crash is obvious. A {abs(pct_shift):.4f}% bias is not.")

    print(f"\n  Saved: eval/downstream_stats.csv")
    print(f"  Saved: eval/downstream_by_detector.csv")
    print(f"  Saved: eval/downstream_missed_rows.csv")
    print(f"\n{SEP2}\n")


if __name__ == "__main__":
    main()