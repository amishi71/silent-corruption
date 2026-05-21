"""
eval/compare.py — Ground Truth Evaluation

Uses labels.csv to compute the full precision/recall/F1 matrix:
    - Per corruption type × per detector
    - Aggregate per detector
    - Compute cost (rows/sec)
    - Interpretability examples

Matching logic: EXACT row-level match only.
A flag is a TP iff its row_id appears in labels.csv for that corruption type.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data"   / "labels.csv"
RULES_PATH  = ROOT / "rules"  / "flags_rules.csv"
AE_PATH     = ROOT / "model"  / "scores_ae.csv"
EVAL_PATH   = ROOT / "data"   / "eval_corrupted.csv"
THRESH_PATH = ROOT / "model"  / "threshold.json"
OUT_DIR     = ROOT / "eval"
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1). Safe against zero denominators."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def load_labels() -> pd.DataFrame:
    """Load ground-truth labels. Normalises row_index → row_id."""
    df = pd.read_csv(LABELS_PATH)
    if "row_index" in df.columns and "row_id" not in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    assert {"row_id", "corruption_type"}.issubset(df.columns), (
        f"labels.csv columns found: {list(df.columns)}"
    )
    return df


def load_rules_flags() -> pd.DataFrame:
    """
    Load rules engine output. Normalises row_index → row_id, rule_type → rule_name.
    Returns one row per (row_id, rule_name) flag.
    """
    df = pd.read_csv(RULES_PATH)
    if "row_index" in df.columns and "row_id" not in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    if "rule_type" in df.columns and "rule_name" not in df.columns:
        df = df.rename(columns={"rule_type": "rule_name"})
    assert "row_id" in df.columns, (
        f"flags_rules.csv columns found: {list(df.columns)}"
    )
    return df


def load_ae_scores() -> pd.DataFrame:
    """
    Load AE scores. Normalises row_index → row_id, recon_error → reconstruction_error.
    """
    df = pd.read_csv(AE_PATH)
    if "row_index" in df.columns and "row_id" not in df.columns:
        df = df.rename(columns={"row_index": "row_id"})
    if "recon_error" in df.columns and "reconstruction_error" not in df.columns:
        df = df.rename(columns={"recon_error": "reconstruction_error"})
    assert {"row_id", "reconstruction_error"}.issubset(df.columns), (
        f"scores_ae.csv columns found: {list(df.columns)}"
    )
    return df


def load_threshold() -> float:
    with open(THRESH_PATH) as f:
        data = json.load(f)
    # handle nested structure: {"thresholds": {"p99": x}, "default": "p99"}
    if "thresholds" in data:
        key = data.get("default", "p99")
        return float(data["thresholds"][key])
    # flat fallbacks
    for key in ("threshold", "p99", "value"):
        if key in data:
            return float(data[key])
    raise KeyError(f"Cannot find threshold value in {THRESH_PATH}: {data}")


# ── Corruption-type mapping ───────────────────────────────────────────────────
# Maps corruption_type label → which rules are expected to catch it.
# Used for the per-type rules breakdown (informational only —
# precision/recall is computed purely from row_id matches).
RULES_AFFINITY = {
    "bit_flip":         ["range_check", "hit_multiplicity_rule"],
    "phantom_detector": ["registered_detector"],
    "stale_timestamp":  ["timestamp_monotone"],
    "thermal_spike":    ["temperature_rate_of_change"],
    "cross_field":      ["cross_field_amplitude"],
    "subtle_drift":     ["z_score"],          # weakest — expected low recall
}

CORRUPTION_TYPES = list(RULES_AFFINITY.keys())


# ── Per-type evaluation ───────────────────────────────────────────────────────
def evaluate_detector(
    flagged_row_ids: set[int],
    labels: pd.DataFrame,
    detector_name: str,
) -> list[dict]:
    """
    Compute TP/FP/FN/precision/recall/F1 per corruption type and in aggregate.

    A flag is a TP iff its row_id appears in labels for ANY corruption type.
    Per-type breakdown: TP = flagged row_id ∈ labels rows of that type.
    FP (global) = flagged row_id ∉ labels at all.
    FN (per type) = labeled rows of that type NOT in flagged set.
    """
    all_labeled_ids = set(labels.loc[labels["corruption_type"] != "clean", "row_id"].tolist())
    rows = []

    for ctype in CORRUPTION_TYPES:
        type_ids = set(labels.loc[labels["corruption_type"] == ctype, "row_id"].tolist())
        tp = len(flagged_row_ids & type_ids)
        fn = len(type_ids - flagged_row_ids)
        # FP: flags that hit no label of ANY type (shared across types)
        fp_global = len(flagged_row_ids - all_labeled_ids)
        # Apportion FP proportionally by type weight for per-type breakdown
        # (honest approximation — exact FP per type is undefined without
        #  knowing which corruption a false flag "intended" to catch)
        type_weight = len(type_ids) / max(len(all_labeled_ids), 1)
        fp_approx = round(fp_global * type_weight)

        precision, recall, f1 = prf(tp, fp_approx, fn)
        rows.append({
            "detector":        detector_name,
            "corruption_type": ctype,
            "n_labeled":       len(type_ids),
            "tp":              tp,
            "fp_approx":       fp_approx,
            "fn":              fn,
            "precision":       precision,
            "recall":          recall,
            "f1":              f1,
        })

    # Aggregate row
    tp_agg  = len(flagged_row_ids & all_labeled_ids)
    fp_agg  = len(flagged_row_ids - all_labeled_ids)
    fn_agg  = len(all_labeled_ids - flagged_row_ids)
    p, r, f = prf(tp_agg, fp_agg, fn_agg)
    rows.append({
        "detector":        detector_name,
        "corruption_type": "AGGREGATE",
        "n_labeled":       len(all_labeled_ids),
        "tp":              tp_agg,
        "fp_approx":       fp_agg,
        "fn":              fn_agg,
        "precision":       p,
        "recall":          r,
        "f1":              f,
    })

    return rows


# ── Compute cost measurement ──────────────────────────────────────────────────
def measure_rules_throughput() -> float:
    """Re-run checker on eval data and time it. Returns rows/sec."""
    import sys
    sys.path.insert(0, str(ROOT))
    from rules.checker import InvariantChecker

    eval_df  = pd.read_csv(EVAL_PATH)
    checker  = InvariantChecker(
        config_path=ROOT / "rules" / "config.yaml",
        channel_params_path=ROOT / "data" / "channel_params.json",
    )
    n_rows   = len(eval_df)

    start = time.perf_counter()
    checker.check(eval_df)
    elapsed = time.perf_counter() - start

    return round(n_rows / elapsed) if elapsed > 0 else 0


def measure_ae_throughput() -> float:
    """Re-run AE inference on eval data and time it. Returns rows/sec."""
    import sys
    sys.path.insert(0, str(ROOT))

    import torch
    from model.autoencoder import TabularAE
    from model.preprocess   import load_and_transform

    CKPT = ROOT / "model" / "ae_checkpoint.pt"
    if not CKPT.exists():
        return 0.0

    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    input_dim  = checkpoint["n_features"]
    model      = TabularAE(n_features=input_dim, bottleneck=checkpoint["bottleneck"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X      = load_and_transform(str(EVAL_PATH))
    tensor = torch.tensor(X, dtype=torch.float32)
    n_rows = len(X)

    start = time.perf_counter()
    with torch.no_grad():
        recon = model(tensor)
        _     = torch.mean((tensor - recon) ** 2, dim=1).numpy()
    elapsed = time.perf_counter() - start

    return round(n_rows / elapsed) if elapsed > 0 else 0


# ── Interpretability examples ─────────────────────────────────────────────────
def get_rules_examples(
    rules_flags: pd.DataFrame,
    labels: pd.DataFrame,
    n: int = 3,
) -> list[str]:
    """Return n example explanations from the rules engine (TP rows only)."""
    labeled_ids = set(labels["row_id"].tolist())
    tp_flags = rules_flags[rules_flags["row_id"].isin(labeled_ids)]
    if tp_flags.empty:
        return ["No TP examples found in rules flags."]

    examples = []
    for _, row in tp_flags.head(n).iterrows():
        ctype = labels.loc[labels["row_id"] == row["row_id"], "corruption_type"]
        ctype_str = ctype.values[0] if len(ctype) else "unknown"
        rule_col  = "rule_name" if "rule_name" in row.index else (
                    "rule"      if "rule"      in row.index else None)
        rule_str  = row[rule_col] if rule_col else "unknown_rule"
        sev_str   = f"severity={int(row['severity'])}" if "severity" in row.index else ""
        examples.append(
            f"  row {int(row['row_id']):>6} | rule: {rule_str:<30} "
            f"| actual: {ctype_str:<20} | {sev_str}"
        )
    return examples


def get_ae_examples(
    ae_scores: pd.DataFrame,
    labels: pd.DataFrame,
    threshold: float,
    n: int = 3,
) -> list[str]:
    """Return n example AE flags with reconstruction error (TP rows only)."""
    labeled_ids = set(labels["row_id"].tolist())
    tp_ae = ae_scores[
        (ae_scores["reconstruction_error"] > threshold) &
        (ae_scores["row_id"].isin(labeled_ids))
    ].sort_values("reconstruction_error", ascending=False)

    if tp_ae.empty:
        return ["No TP examples found in AE scores."]

    examples = []
    for _, row in tp_ae.head(n).iterrows():
        ctype = labels.loc[labels["row_id"] == row["row_id"], "corruption_type"]
        ctype_str = ctype.values[0] if len(ctype) else "unknown"
        err_str   = f"recon_err={row['reconstruction_error']:.6f}"
        top_feat  = ""
        feat_cols = [c for c in ae_scores.columns
                     if c not in ("row_id", "reconstruction_error",
                                  "flagged", "flagged_p95", "flagged_p99",
                                  "flagged_p99_5", "top_feature")]
        if feat_cols:
            feat_errs = row[feat_cols]
            top       = feat_errs.idxmax()
            top_feat  = f"| top_feature: {top}"
        examples.append(
            f"  row {int(row['row_id']):>6} | {err_str:<28} "
            f"| actual: {ctype_str:<20} {top_feat}"
        )
    return examples


# ── Silent propagation example ────────────────────────────────────────────────
def find_missed_corruptions(
    rules_flagged: set[int],
    ae_flagged: set[int],
    labels: pd.DataFrame,
    eval_df: pd.DataFrame,
    n: int = 3,
) -> list[str]:
    """
    Find corrupted rows missed by BOTH detectors — the visceral proof of the
    central claim: silent corruption propagates undetected.
    """
    all_labeled  = set(labels.loc[labels["corruption_type"] != "clean", "row_id"].tolist())
    both_missed  = all_labeled - rules_flagged - ae_flagged

    if not both_missed:
        return ["  All corrupted rows caught by at least one detector."]

    lines = []
    sample_ids = list(both_missed)[:n]
    for rid in sample_ids:
        ctype = labels.loc[labels["row_id"] == rid, "corruption_type"].values[0]
        if rid not in eval_df.index:
            lines.append(f"  row {rid:>6} | type: {ctype} | (row not found in eval data)")
            continue
        row    = eval_df.iloc[rid] if rid < len(eval_df) else None
        if row is None:
            lines.append(f"  row {rid:>6} | type: {ctype} | (index out of range)")
            continue
        energy = row.get("energy_deposit_keV", "N/A")
        temp   = row.get("temperature_K",      "N/A")
        lines.append(
            f"  row {rid:>6} | type: {ctype:<20} "
            f"| energy={energy} keV  temp={temp} K  "
            f"← passed both detectors undetected"
        )
    remaining = len(both_missed) - len(sample_ids)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more rows missed by both detectors.")
    return lines


# ── Terminal report ───────────────────────────────────────────────────────────
def print_report(
    results_df: pd.DataFrame,
    rules_throughput: float,
    ae_throughput: float,
    rules_examples: list[str],
    ae_examples: list[str],
    missed_examples: list[str],
    threshold: float,
    n_rules_flags: int,
    n_ae_flags: int,
    n_labeled: int,
) -> None:
    SEP  = "─" * 78
    SEP2 = "═" * 78

    print(f"\n{SEP2}")
    print("  SILENT CORRUPTION DETECTION — PHASE 4 EVALUATION REPORT")
    print(SEP2)

    print(f"\n  Ground truth: {n_labeled} corrupted rows")
    print(f"  Rules engine flags: {n_rules_flags}")
    print(f"  AE flags (threshold={threshold:.6f}): {n_ae_flags}")

    # ── Per-type table ──
    print(f"\n{SEP}")
    print("  PER CORRUPTION TYPE × DETECTOR  (exact row-id match)")
    print(SEP)
    hdr = f"  {'Corruption Type':<22} {'Detector':<10} {'Labeled':>7} {'TP':>5} {'FP~':>5} {'FN':>5}  {'Prec':>6} {'Rec':>6} {'F1':>6}"
    print(hdr)
    print(f"  {'-'*22} {'-'*10} {'-'*7} {'-'*5} {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")

    type_rows = results_df[results_df["corruption_type"] != "AGGREGATE"]
    for _, r in type_rows.sort_values(["corruption_type", "detector"]).iterrows():
        print(
            f"  {r['corruption_type']:<22} {r['detector']:<10} "
            f"{int(r['n_labeled']):>7} {int(r['tp']):>5} {int(r['fp_approx']):>5} "
            f"{int(r['fn']):>5}  {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}"
        )

    # ── Aggregate table ──
    print(f"\n{SEP}")
    print("  AGGREGATE")
    print(SEP)
    agg_rows = results_df[results_df["corruption_type"] == "AGGREGATE"]
    for _, r in agg_rows.iterrows():
        print(
            f"  {r['detector']:<10}  TP={int(r['tp'])}  FP={int(r['fp_approx'])}  "
            f"FN={int(r['fn'])}  Precision={r['precision']:.3f}  "
            f"Recall={r['recall']:.3f}  F1={r['f1']:.3f}"
        )

    # ── Compute cost ──
    print(f"\n{SEP}")
    print("  COMPUTE COST")
    print(SEP)
    print(f"  Rules engine : {rules_throughput:>10,} rows/sec")
    print(f"  AE inference : {ae_throughput:>10,} rows/sec")
    if ae_throughput > 0:
        ratio = rules_throughput / ae_throughput
        print(f"  Speedup      : {ratio:.1f}× faster (rules vs AE)")

    # ── Interpretability ──
    print(f"\n{SEP}")
    print("  INTERPRETABILITY — RULES ENGINE (sample TP flags)")
    print(SEP)
    for line in rules_examples:
        print(line)

    print(f"\n{SEP}")
    print("  INTERPRETABILITY — AUTOENCODER (sample TP flags)")
    print(SEP)
    for line in ae_examples:
        print(line)

    # ── Silent propagation proof ──
    print(f"\n{SEP}")
    print("  SILENT PROPAGATION — ROWS MISSED BY BOTH DETECTORS")
    print(SEP)
    for line in missed_examples:
        print(line)

    print(f"\n{SEP2}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading ground truth labels...")
    labels    = load_labels()
    eval_df   = pd.read_csv(EVAL_PATH)
    threshold = load_threshold()

    print("Loading detector outputs...")
    rules_flags = load_rules_flags()
    ae_scores   = load_ae_scores()

    # Unique flagged row IDs per detector
    rules_flagged_ids: set[int] = set(rules_flags["row_id"].tolist())
    ae_flagged_ids:    set[int] = set(
        ae_scores.loc[ae_scores["reconstruction_error"] > threshold, "row_id"].tolist()
    )

    print("Evaluating rules engine...")
    rules_results = evaluate_detector(rules_flagged_ids, labels, "rules")

    print("Evaluating autoencoder...")
    ae_results = evaluate_detector(ae_flagged_ids, labels, "ae")

    results_df = pd.DataFrame(rules_results + ae_results)

    # Save CSVs
    by_type_path  = OUT_DIR / "results_by_type.csv"
    summary_path  = OUT_DIR / "results_summary.csv"
    results_df.to_csv(by_type_path, index=False)
    results_df[results_df["corruption_type"] == "AGGREGATE"].to_csv(
        summary_path, index=False
    )
    print(f"Saved: {by_type_path}")
    print(f"Saved: {summary_path}")

    print("Measuring compute throughput...")
    rules_throughput = measure_rules_throughput()
    ae_throughput    = measure_ae_throughput()

    # Examples
    rules_examples  = get_rules_examples(rules_flags, labels)
    ae_examples     = get_ae_examples(ae_scores, labels, threshold)
    missed_examples = find_missed_corruptions(
        rules_flagged_ids, ae_flagged_ids, labels, eval_df
    )

    # Print full report
    print_report(
        results_df        = results_df,
        rules_throughput  = rules_throughput,
        ae_throughput     = ae_throughput,
        rules_examples    = rules_examples,
        ae_examples       = ae_examples,
        missed_examples   = missed_examples,
        threshold         = threshold,
        n_rules_flags     = len(rules_flagged_ids),
        n_ae_flags        = len(ae_flagged_ids),
        n_labeled         = len(labels[labels["corruption_type"] != "clean"]),
    )


if __name__ == "__main__":
    main()