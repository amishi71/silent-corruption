"""
eval/pipeline.py — Combined cascade detection pipeline.

Architecture:
    Stage 1 — Rules engine (fast, deterministic, high-recall on known types)
        → Rows with rule violations are flagged immediately with reason codes.
        → Clean rows pass to Stage 2.

    Stage 2 — Autoencoder (catches distributional residuals rules miss)
        → Flags rows whose reconstruction error exceeds threshold.
        → AE flags are annotated with top contributing feature.

    Stage 3 — Explanation (AE flags interrogated by rules for human-readable reason)
        → For each AE-only flag, checks which invariants are nearest to violation.
        → Produces a combined explanation where possible.

Output:
    eval/pipeline_flags.csv  — all flags with source (rules / ae / both)
    eval/pipeline_report.csv — per-type precision/recall vs individual detectors
    Terminal report           — cascade summary + comparison table
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data"  / "labels.csv"
RULES_PATH  = ROOT / "rules" / "flags_rules.csv"
AE_PATH     = ROOT / "model" / "scores_ae.csv"
EVAL_PATH   = ROOT / "data"  / "eval_corrupted.csv"
THRESH_PATH = ROOT / "model" / "threshold.json"
OUT_DIR     = ROOT / "eval"
OUT_DIR.mkdir(exist_ok=True)

CORRUPTION_TYPES = [
    "bit_flip", "phantom_detector", "stale_timestamp",
    "thermal_spike", "cross_field", "subtle_drift",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def load_labels():
    df = pd.read_csv(LABELS_PATH)
    if "row_index" in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    return df[df["corruption_type"] != "clean"].copy()


def load_threshold():
    with open(THRESH_PATH) as f:
        data = json.load(f)
    if "thresholds" in data:
        key = data.get("default", "p99")
        return float(data["thresholds"][key])
    for k in ("threshold", "p99", "value"):
        if k in data:
            return float(data[k])
    raise KeyError("No threshold found")


def load_rules_flags():
    df = pd.read_csv(RULES_PATH)
    if "row_index" in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    if "rule_type" in df.columns and "rule_name" not in df.columns:
        df = df.rename(columns={"rule_type": "rule_name"})
    return df


def load_ae_scores():
    df = pd.read_csv(AE_PATH)
    if "row_index" in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    if "recon_error" in df.columns:
        df = df.rename(columns={"recon_error": "reconstruction_error"})
    return df


def evaluate_detector(flagged_ids, labels, name):
    all_labeled = set(labels["row_id"])
    rows = []
    for ctype in CORRUPTION_TYPES:
        type_ids = set(labels.loc[labels["corruption_type"] == ctype, "row_id"])
        tp = len(flagged_ids & type_ids)
        fn = len(type_ids - flagged_ids)
        fp_g = len(flagged_ids - all_labeled)
        w = len(type_ids) / max(len(all_labeled), 1)
        fp = round(fp_g * w)
        p, r, f = prf(tp, fp, fn)
        rows.append({"detector": name, "corruption_type": ctype,
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f})
    tp_a = len(flagged_ids & all_labeled)
    fp_a = len(flagged_ids - all_labeled)
    fn_a = len(all_labeled - flagged_ids)
    p, r, f = prf(tp_a, fp_a, fn_a)
    rows.append({"detector": name, "corruption_type": "AGGREGATE",
                 "tp": tp_a, "fp": fp_a, "fn": fn_a,
                 "precision": p, "recall": r, "f1": f})
    return rows


# ── Stage 3: near-miss explanation for AE-only flags ─────────────────────────
def explain_ae_flag(row_id: int, eval_df: pd.DataFrame,
                    ae_scores: pd.DataFrame) -> str:
    """
    For a row flagged only by the AE, produce a plain-English explanation
    by examining which fields contributed most to reconstruction error.
    """
    ae_row = ae_scores[ae_scores["row_id"] == row_id]
    if ae_row.empty:
        return "no AE detail available"
    ae_row = ae_row.iloc[0]

    recon_err = ae_row.get("reconstruction_error", float("nan"))
    top_feat  = ae_row.get("top_feature", "unknown")

    # Strip err_ prefix for readability
    field_name = str(top_feat).replace("err_", "")

    if row_id < len(eval_df):
        data_row = eval_df.iloc[row_id]
        val = data_row.get(field_name, None)
        val_str = f"={val:.4f}" if val is not None and isinstance(val, float) else ""
    else:
        val_str = ""

    return (f"recon_err={recon_err:.4f} | "
            f"top_field: {field_name}{val_str} | "
            f"hypothesis: distributional anomaly, not a rule violation")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    labels    = load_labels()
    eval_df   = pd.read_csv(EVAL_PATH)
    threshold = load_threshold()

    rules_flags = load_rules_flags()
    ae_scores   = load_ae_scores()

    all_labeled = set(labels["row_id"])

    # ── Stage 1: rules ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    rules_flagged: set[int] = set(rules_flags["row_id"])
    rules_time = time.perf_counter() - t0

    # ── Stage 2: AE on rows NOT caught by rules ───────────────────────────────
    t1 = time.perf_counter()
    ae_flagged_all: set[int] = set(
        ae_scores.loc[ae_scores["reconstruction_error"] > threshold, "row_id"]
    )
    ae_only: set[int] = ae_flagged_all - rules_flagged   # residuals only
    ae_time = time.perf_counter() - t1

    combined: set[int] = rules_flagged | ae_only
    both: set[int]     = rules_flagged & ae_flagged_all  # caught by both

    # ── Stage 3: explain AE-only flags ───────────────────────────────────────
    ae_only_tp = ae_only & all_labeled
    explanations = []
    for rid in sorted(list(ae_only_tp))[:5]:
        ctype = labels.loc[labels["row_id"] == rid, "corruption_type"].values[0]
        expl  = explain_ae_flag(rid, eval_df, ae_scores)
        explanations.append((rid, ctype, expl))

    # ── Build flags output CSV ────────────────────────────────────────────────
    flag_rows = []
    for rid in rules_flagged:
        rule_row = rules_flags[rules_flags["row_id"] == rid].iloc[0]
        source = "both" if rid in ae_flagged_all else "rules"
        flag_rows.append({
            "row_id":   rid,
            "source":   source,
            "rule":     rule_row.get("rule_name", ""),
            "severity": rule_row.get("severity", ""),
            "recon_err": ae_scores.loc[ae_scores["row_id"] == rid, "reconstruction_error"].values[0]
                         if rid in ae_flagged_all else None,
            "top_feature": ae_scores.loc[ae_scores["row_id"] == rid, "top_feature"].values[0]
                           if rid in ae_flagged_all else None,
        })
    for rid in ae_only:
        ae_row = ae_scores[ae_scores["row_id"] == rid].iloc[0]
        flag_rows.append({
            "row_id":    rid,
            "source":    "ae",
            "rule":      None,
            "severity":  None,
            "recon_err": ae_row.get("reconstruction_error"),
            "top_feature": ae_row.get("top_feature"),
        })

    flags_df = pd.DataFrame(flag_rows).sort_values("row_id").reset_index(drop=True)
    flags_df.to_csv(OUT_DIR / "pipeline_flags.csv", index=False)

    # ── Evaluate all three systems ────────────────────────────────────────────
    results = []
    results.extend(evaluate_detector(rules_flagged,  labels, "rules"))
    results.extend(evaluate_detector(ae_flagged_all, labels, "ae"))
    results.extend(evaluate_detector(combined,       labels, "pipeline"))
    report_df = pd.DataFrame(results)
    report_df.to_csv(OUT_DIR / "pipeline_report.csv", index=False)

    # ── Terminal report ───────────────────────────────────────────────────────
    SEP  = "─" * 78
    SEP2 = "═" * 78
    n_total = len(eval_df)

    print(f"\n{SEP2}")
    print("  COMBINED CASCADE PIPELINE — EVALUATION REPORT")
    print(SEP2)

    print(f"\n  Dataset:         {n_total:,} rows")
    print(f"  Ground truth:    {len(all_labeled)} corrupted rows")
    print(f"  Threshold (p99): {threshold:.6f}")

    print(f"\n  Stage 1 — Rules engine")
    print(f"    Flagged:    {len(rules_flagged):>5} rows  ({len(rules_flagged)/n_total*100:.1f}%)")
    print(f"    TP:         {len(rules_flagged & all_labeled):>5}")
    print(f"    FP:         {len(rules_flagged - all_labeled):>5}")

    print(f"\n  Stage 2 — AE on rules-residuals only")
    print(f"    Input:      {n_total - len(rules_flagged):>5} rows (rules-clean)")
    print(f"    AE flagged: {len(ae_only):>5} additional rows")
    print(f"    TP gained:  {len(ae_only & all_labeled):>5}")
    print(f"    FP added:   {len(ae_only - all_labeled):>5}")

    print(f"\n  Combined pipeline total")
    print(f"    Flagged:    {len(combined):>5} rows  ({len(combined)/n_total*100:.1f}%)")
    print(f"    TP:         {len(combined & all_labeled):>5}")
    print(f"    FP:         {len(combined - all_labeled):>5}")
    print(f"    Missed:     {len(all_labeled - combined):>5}")

    print(f"\n{SEP}")
    print("  PRECISION / RECALL / F1 — RULES vs AE vs PIPELINE")
    print(SEP)
    agg = report_df[report_df["corruption_type"] == "AGGREGATE"]
    print(f"  {'Detector':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>9}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for _, r in agg.iterrows():
        n_flags = (len(rules_flagged) if r["detector"] == "rules"
                   else len(ae_flagged_all) if r["detector"] == "ae"
                   else len(combined))
        print(f"  {r['detector']:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {n_flags:>9}")

    print(f"\n{SEP}")
    print("  PER-TYPE RECALL — RULES vs AE vs PIPELINE")
    print(SEP)
    print(f"  {'Corruption Type':<22} {'rules':>8} {'ae':>8} {'pipeline':>10}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10}")
    for ctype in CORRUPTION_TYPES:
        vals = {}
        for det in ["rules", "ae", "pipeline"]:
            row = report_df[(report_df["detector"] == det) &
                            (report_df["corruption_type"] == ctype)]
            vals[det] = row["recall"].values[0] if len(row) else 0.0
        print(f"  {ctype:<22} {vals['rules']:>8.3f} {vals['ae']:>8.3f} {vals['pipeline']:>10.3f}")

    print(f"\n{SEP}")
    print("  STAGE 3 — AE-ONLY FLAGS EXPLAINED (sample TPs)")
    print(SEP)
    if explanations:
        for rid, ctype, expl in explanations:
            print(f"  row {rid:>6} | actual: {ctype:<20} | {expl}")
    else:
        print("  No AE-only TPs to explain.")

    # Missed by pipeline
    pipeline_missed = all_labeled - combined
    print(f"\n{SEP}")
    print(f"  STILL MISSED BY PIPELINE ({len(pipeline_missed)} rows)")
    print(SEP)
    for rid in sorted(list(pipeline_missed))[:5]:
        ctype = labels.loc[labels["row_id"] == rid, "corruption_type"].values[0]
        if rid < len(eval_df):
            row = eval_df.iloc[rid]
            energy = row.get("energy_deposit_keV", "N/A")
            temp   = row.get("temperature_K", "N/A")
            print(f"  row {rid:>6} | {ctype:<20} | energy={energy} keV  temp={temp} K")
    if len(pipeline_missed) > 5:
        print(f"  ... and {len(pipeline_missed)-5} more.")

    print(f"\n  Saved: eval/pipeline_flags.csv")
    print(f"  Saved: eval/pipeline_report.csv")
    print(f"\n{SEP2}\n")


if __name__ == "__main__":
    main()