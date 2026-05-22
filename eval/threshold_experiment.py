"""
eval/threshold_experiment.py — AE threshold sensitivity analysis.

Runs evaluation at p95 / p99 / p99.5 and prints a comparison table
showing how subtle_drift recall (and overall precision) changes across
thresholds. Demonstrates that threshold selection is a hyperparameter,
not a discovery.
"""

import json
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data"  / "labels.csv"
AE_PATH     = ROOT / "model" / "scores_ae.csv"
THRESH_PATH = ROOT / "model" / "threshold.json"
OUT_DIR     = ROOT / "eval"
OUT_DIR.mkdir(exist_ok=True)

CORRUPTION_TYPES = [
    "bit_flip", "phantom_detector", "stale_timestamp",
    "thermal_spike", "cross_field", "subtle_drift",
]


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


def evaluate_at_threshold(ae_scores, labels, threshold, name):
    flagged = set(ae_scores.loc[ae_scores["reconstruction_error"] > threshold, "row_id"])
    all_labeled = set(labels["row_id"])
    rows = []
    for ctype in CORRUPTION_TYPES:
        type_ids = set(labels.loc[labels["corruption_type"] == ctype, "row_id"])
        tp = len(flagged & type_ids)
        fn = len(type_ids - flagged)
        fp_g = len(flagged - all_labeled)
        w = len(type_ids) / max(len(all_labeled), 1)
        fp = round(fp_g * w)
        p, r, f = prf(tp, fp, fn)
        rows.append({"threshold": name, "corruption_type": ctype,
                     "tp": tp, "fn": fn, "recall": r, "precision": p, "f1": f})
    # aggregate
    tp_a = len(flagged & all_labeled)
    fp_a = len(flagged - all_labeled)
    fn_a = len(all_labeled - flagged)
    p, r, f = prf(tp_a, fp_a, fn_a)
    rows.append({"threshold": name, "corruption_type": "AGGREGATE",
                 "tp": tp_a, "fn": fn_a, "recall": r, "precision": p, "f1": f})
    return rows


def main():
    labels = load_labels()
    ae_scores = pd.read_csv(AE_PATH)
    if "row_index" in ae_scores.columns:
        ae_scores = ae_scores.rename(columns={"row_index": "row_id"})
    if "recon_error" in ae_scores.columns:
        ae_scores = ae_scores.rename(columns={"recon_error": "reconstruction_error"})

    with open(THRESH_PATH) as f:
        tdata = json.load(f)
    thresholds = {
        "p95":   tdata["thresholds"]["p95"],
        "p99":   tdata["thresholds"]["p99"],
        "p99.5": tdata["thresholds"]["p99_5"],
    }

    all_rows = []
    for name, val in thresholds.items():
        all_rows.extend(evaluate_at_threshold(ae_scores, labels, val, name))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "threshold_experiment.csv", index=False)

    SEP  = "─" * 72
    SEP2 = "═" * 72

    print(f"\n{SEP2}")
    print("  AE THRESHOLD SENSITIVITY ANALYSIS")
    print(f"  p95={thresholds['p95']:.4f}  p99={thresholds['p99']:.4f}  p99.5={thresholds['p99.5']:.4f}")
    print(SEP2)

    # subtle_drift focus
    print(f"\n  {'SUBTLE DRIFT recall by threshold — the hard case':}")
    print(SEP)
    print(f"  {'Threshold':<10} {'Cutoff':>10}  {'Flagged':>8}  {'TP':>5}  {'FN':>5}  {'Recall':>7}  {'Precision':>10}")
    print(f"  {'-'*10} {'-'*10}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*10}")
    for name, val in thresholds.items():
        n_flagged = int((ae_scores["reconstruction_error"] > val).sum())
        row = df[(df["threshold"] == name) & (df["corruption_type"] == "subtle_drift")].iloc[0]
        print(f"  {name:<10} {val:>10.4f}  {n_flagged:>8}  {int(row['tp']):>5}  "
              f"{int(row['fn']):>5}  {row['recall']:>7.3f}  {row['precision']:>10.3f}")

    # full recall table
    print(f"\n  {'RECALL BY CORRUPTION TYPE × THRESHOLD':}")
    print(SEP)
    ctypes_show = CORRUPTION_TYPES + ["AGGREGATE"]
    header = f"  {'Corruption Type':<22}"
    for name in thresholds:
        header += f"  {name:>8}"
    print(header + "   (recall)")
    print(f"  {'-'*22}" + "  " + "  ".join(["-"*8]*3))
    for ctype in ctypes_show:
        row_str = f"  {ctype:<22}"
        for name in thresholds:
            r = df[(df["threshold"] == name) & (df["corruption_type"] == ctype)]["recall"].values[0]
            row_str += f"  {r:>8.3f}"
        print(row_str)

    # precision table
    print(f"\n  {'PRECISION BY CORRUPTION TYPE × THRESHOLD':}")
    print(SEP)
    header = f"  {'Corruption Type':<22}"
    for name in thresholds:
        header += f"  {name:>8}"
    print(header + "   (precision)")
    print(f"  {'-'*22}" + "  " + "  ".join(["-"*8]*3))
    for ctype in ctypes_show:
        row_str = f"  {ctype:<22}"
        for name in thresholds:
            p = df[(df["threshold"] == name) & (df["corruption_type"] == ctype)]["precision"].values[0]
            row_str += f"  {p:>8.3f}"
        print(row_str)

    print(f"\n  Saved: eval/threshold_experiment.csv")
    print(f"\n{SEP2}\n")

    # Key insight summary
    print("  KEY INSIGHT:")
    sd_p95 = df[(df["threshold"]=="p95") & (df["corruption_type"]=="subtle_drift")]["recall"].values[0]
    sd_p99 = df[(df["threshold"]=="p99") & (df["corruption_type"]=="subtle_drift")]["recall"].values[0]
    agg_p95_prec = df[(df["threshold"]=="p95") & (df["corruption_type"]=="AGGREGATE")]["precision"].values[0]
    agg_p99_prec = df[(df["threshold"]=="p99") & (df["corruption_type"]=="AGGREGATE")]["precision"].values[0]
    print(f"  subtle_drift recall:  p95={sd_p95:.3f}  vs  p99={sd_p99:.3f}  "
          f"(+{sd_p95-sd_p99:.3f} at cost of precision)")
    print(f"  aggregate precision:  p95={agg_p95_prec:.3f}  vs  p99={agg_p99_prec:.3f}")
    print(f"  → Threshold is a business decision: recall vs precision tradeoff,")
    print(f"    not a model property. The model is the same. The cutoff changes everything.\n")


if __name__ == "__main__":
    main()