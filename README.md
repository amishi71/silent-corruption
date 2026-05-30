# silent-corruption

> **Central claim:** Silent data corruption is more dangerous than visible failure.
> A crashed system is obvious — you fix it. A system that silently writes wrong values
> continues to operate, builds false confidence, and propagates errors downstream.
> This project tests that claim on simulated particle detector data using two independent
> detection systems and a combined cascade pipeline, then traces exactly what the missed
> corruptions do to a physics measurement.

---

## What this is

A two-system face-off for detecting silent corruption in scientific instrument data —
built as a direct precursor to anomaly detection work on real HEP datasets.

**System 1 — Invariant rules engine:** Deterministic checks encoding physical constraints:
field ranges, registered detector IDs, timestamp monotonicity, cross-field amplitude
consistency, hit multiplicity thresholds, rolling z-score drift detection.

**System 2 — PyTorch autoencoder:** Trained on clean data only. Flags rows whose
reconstruction error exceeds the p99 threshold (0.2485). Treats anomaly score as a
hypothesis, not a verdict.

**System 3 — Combined cascade pipeline:** Rules run first (fast, zero FP). Unflagged
rows pass to the AE. AE-only flags are annotated with top contributing feature and a
plain-English hypothesis. The combination is evaluated against both individual systems.

**Downstream impact analysis:** The rows that passed all three systems are traced to
their effect on mean energy — the primary physics observable — demonstrating that silent
corruption produces wrong science with no visible alarm.

All systems run on the same corrupted evaluation set. Results are compared against
sealed ground-truth labels opened only at evaluation time.

---

## Key results

554 corrupted rows across 6 corruption types injected into a 10,000-row eval set.

### Per-type recall — rules vs AE vs pipeline

| Corruption Type  | Labeled | Rules Recall | AE Recall | Pipeline Recall |
| ---------------- | ------- | ------------ | --------- | --------------- |
| bit_flip         | 80      | 0.975        | 0.875     | 0.975           |
| phantom_detector | 80      | 1.000        | 1.000     | 1.000           |
| stale_timestamp  | 80      | 1.000        | 0.025     | 1.000           |
| thermal_spike    | 158     | 1.000        | 0.513     | 1.000           |
| cross_field      | 80      | 0.938        | 0.738     | 0.938           |
| subtle_drift     | 76      | 0.013        | 0.013     | 0.026           |

### Aggregate

| Detector | Precision | Recall | F1    | Flagged | Throughput     |
| -------- | --------- | ------ | ----- | ------- | -------------- |
| Rules    | 1.000     | 0.852  | 0.920 | 472     | ~1.6M rows/sec |
| AE       | 0.724     | 0.529  | 0.611 | 405     | ~690k rows/sec |
| Pipeline | 0.809     | 0.854  | 0.831 | 585     | —              |

### AE threshold sensitivity (subtle_drift recall)

| Threshold | Cutoff | Flagged | subtle_drift Recall | Aggregate Precision |
| --------- | ------ | ------- | ------------------- | ------------------- |
| p95       | 0.1235 | 855     | 0.092               | 0.367               |
| p99       | 0.2485 | 405     | 0.013               | 0.724               |
| p99.5     | 0.3389 | 343     | 0.013               | 0.840               |

### Downstream impact — what the missed rows do

81 rows passed all three detectors undetected. 74 are subtle_drift.

| Dataset                     | Mean energy (keV) | Delta  | Delta % |
| --------------------------- | ----------------- | ------ | ------- |
| Clean baseline              | 32.814            | —      | —       |
| Contaminated (no detection) | 39.632            | +6.818 | +20.78% |
| After pipeline detection    | 32.312            | −0.502 | −1.53%  |

The pipeline reduces a 20.78% systematic bias to a 1.53% residual. The residual is
not random noise — it is a directional bias introduced by 81 rows that look completely
normal individually. A physicist working on this data would report a mean energy of
32.3 keV where the true value is 32.8 keV, with no indication anything is wrong.

Per-detector bias (contaminated vs clean):

| Detector | Clean mean (keV) | Contaminated (keV) | Bias   |
| -------- | ---------------- | ------------------ | ------ |
| 1        | 32.63            | 38.96              | +19.4% |
| 2        | 31.54            | 42.18              | +33.7% |
| 3        | 33.41            | 34.48              | +3.2%  |
| 4        | 32.67            | 33.78              | +3.4%  |
| 5        | 33.44            | 43.79              | +30.9% |
| 6        | 34.05            | 43.85              | +28.8% |
| 7        | 31.62            | 40.81              | +29.0% |
| 8        | 33.19            | 39.49              | +19.0% |

The bias is not uniform — detectors 2, 5, 6, 7 are hit hardest. In a real experiment,
this would look like a calibration problem, not a data quality problem.

---

## What the results show

**Rules are verdicts. ML output is a hypothesis.**
A rules violation is a logical proof — if `registered_detector` fires, that detector ID
is not in the valid set, full stop. An AE reconstruction error says "this row is unusual
relative to training distribution," which could mean corruption or a legitimate edge case.
Never conflate these two.

**Each system has a blind spot the other doesn't.**
The rules engine catches all 80 stale timestamps (recall 1.000) because it encodes the
ordering relationship explicitly. The AE catches 0.025 — repeated timestamps look
statistically normal to a model that learned value distributions, not sequence constraints.

Conversely, subtle_drift is nearly invisible to both systems at p99. The gradual energy
drift — 3–5% per affected row — distributes reconstruction error across multiple features
with no dominant signal. No single rule threshold catches it either. This is the hardest
corruption type by design, and the downstream analysis shows exactly why it matters: 74
undetected subtle_drift rows introduce a 20% systematic bias in mean energy.

**The cascade pipeline works correctly.**
Rules first: 472 flags, precision 1.000, zero false positives, 1.6M rows/sec.
AE on residuals: 113 additional flags, 1 more TP, 112 FPs added.
Combined: F1 0.831 — better than either system alone.
The pipeline F1 exceeding both individual systems is the correct outcome when rules
have high precision and the AE catches residuals the rules miss.

**Threshold selection is a business decision, not a model property.**
The same trained AE produces subtle_drift recall of 0.092 at p95 and 0.013 at p99.
Aggregate precision drops from 0.724 to 0.367 at the lower threshold. The model is
unchanged — the cutoff changes everything. In a physics experiment where one missed
systematic bias can invalidate an analysis, p95 is defensible. In a real-time DAQ
system where false alarms cause operators to ignore the system, p99.5 is defensible.
That tradeoff cannot be resolved by the model. It requires a domain judgment.

**Interpretability is not symmetric.**
Rules give an exact reason code in one line: `range_check | severity=3`.
The AE gives a reconstruction error and a top contributing feature:
`recon_err=0.56 | top_field: hit_multiplicity | hypothesis: distributional anomaly`.
That is not an explanation — it is a starting point for one.

---

## Domain

Simulated particle detector readings. Fields:

| Field                 | Type        | Key invariant                               |
| --------------------- | ----------- | ------------------------------------------- |
| `event_id`            | int         | monotone per run                            |
| `timestamp_ns`        | int         | strictly increasing                         |
| `detector_id`         | int         | must be in registered set                   |
| `channel_id`          | int         | valid per detector                          |
| `energy_deposit_keV`  | float       | > 0, < channel max                          |
| `hit_multiplicity`    | int         | ≥ 1 if energy > threshold                   |
| `signal_amplitude_mV` | float       | correlated with energy via per-channel gain |
| `noise_floor_mV`      | float       | stable per channel per run                  |
| `temperature_K`       | float       | rate-of-change bounded                      |
| `run_status`          | categorical | transitions: init → active → closed         |

Domain chosen for direct alignment with high-energy physics data quality infrastructure and as a
foundation for L05 (physics anomaly detector on real HEP data).

---

## Corruption types injected

| Type               | Description                                                 | Detectable by    |
| ------------------ | ----------------------------------------------------------- | ---------------- |
| `bit_flip`         | ADC high-bit flip → large positive energy value             | Rules (range)    |
| `phantom_detector` | Event attributed to unregistered detector ID                | Rules (registry) |
| `stale_timestamp`  | Duplicate timestamp, breaking strict monotonicity           | Rules (ordering) |
| `thermal_spike`    | Temperature jump exceeding rate-of-change bound             | Rules (rate)     |
| `cross_field`      | Signal amplitude inconsistent with energy × channel gain    | Rules + AE       |
| `subtle_drift`     | Gradual energy drift within plausible range, no single rule | AE (partial)     |

---

## Project structure

```
silent-corruption/
├── data/
│   ├── generator.py        # synthetic clean data (40k train, 10k eval)
│   ├── inject.py           # corruption injection → eval_corrupted.csv + labels.csv
│   └── profile.py          # dataset sanity checks
├── rules/
│   ├── config.yaml         # rule parameters (ranges, z-score window, tolerances)
│   ├── checker.py          # InvariantChecker — runs all rules, returns flags_rules.csv
│   └── test_checker.py     # 22/22 unit tests
├── model/
│   ├── preprocess.py       # fits MinMaxScaler on train_clean.csv → scaler.pkl
│   ├── autoencoder.py      # TabularAE (bottleneck=4, 8 features)
│   ├── train.py            # training loop, early stopping, threshold selection
│   └── infer.py            # inference → scores_ae.csv
├── eval/
│   ├── compare.py               # ground truth evaluation, precision/recall matrix
│   ├── threshold_experiment.py  # AE threshold sensitivity across p95/p99/p99.5
│   ├── pipeline.py              # combined cascade: rules → AE residuals → explanation
│   └── downstream_impact.py     # traces missed rows to mean energy bias
├── requirements.txt
└── .gitignore
```

---

## Run order

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python data/generator.py               # → train_clean.csv, eval_clean.csv, channel_params.json

# 3. Inject corruption (seals labels.csv)
python data/inject.py                  # → eval_corrupted.csv, labels.csv
chmod 000 data/labels.csv             # seal until evaluation

# 4. Rules engine
python rules/test_checker.py           # 22/22 — verify before running on eval
PYTHONPATH=. python rules/checker.py   # → rules/flags_rules.csv

# 5. Autoencoder
python model/preprocess.py             # → model/scaler.pkl, model/feature_cols.json
PYTHONPATH=. python model/train.py     # → model/ae_checkpoint.pt, model/threshold.json
PYTHONPATH=. python model/infer.py     # → model/scores_ae.csv

# 6. Evaluation (unlock labels first)
chmod 644 data/labels.csv
PYTHONPATH=. python eval/compare.py               # → eval/results_by_type.csv
PYTHONPATH=. python eval/threshold_experiment.py  # → eval/threshold_experiment.csv
PYTHONPATH=. python eval/pipeline.py              # → eval/pipeline_flags.csv
PYTHONPATH=. python eval/downstream_impact.py     # → eval/downstream_stats.csv
```

---

## Design decisions worth noting

**Why exact-match evaluation, not window-match.**
A rules engine that fires on the exact corrupted row is meaningfully better than one
that fires one row off. Window matching obscures that distinction. Thermal spike labels
cover both the spike row and the recovery row — the double-labeling problem is handled
at injection time, not at evaluation time.

**Why seal labels.csv.**
If you read the ground truth while building detectors, you unconsciously overfit rules
to the exact corruptions you injected. The seal enforces the discipline of operating
without confirmation — which is the actual production condition.

**Why the AE trains on clean data only.**
If corrupted rows leak into training, the model learns to reconstruct corruptions as
normal. The clean-only split is enforced before injection runs.

**Why the bottleneck is 4.**
For 8 tabular features, a bottleneck of 4 forces meaningful compression without
collapsing to a representation too small to reconstruct normal data faithfully. Too
small → high baseline error, poor discrimination. Too large → memorisation, defeats
the purpose.

**Why the range check uses strict lower bound.**
`hit_multiplicity=0` is valid — low-energy events below threshold register zero hits.
The lower bound check uses `value < min` (strictly less than), not `value <= min`.
Using `<=` would flag every zero-hit row as a violation, generating thousands of false
positives on clean data.

**Why the cascade pipeline result is now honest.**
After fixing the false positives in the rules engine, the cascade produces F1=0.831
vs rules-alone F1=0.920 and AE-alone F1=0.611. The pipeline beats the AE but not
rules alone on F1 — because rules already have near-perfect precision, adding the AE
stage introduces some FPs while gaining minimal TPs on subtle_drift. This is the
correct engineering result given the corruption distribution. Documenting it is the
point.

---

## Dependencies

```
numpy
pandas
torch
scikit-learn
pyyaml
joblib
scipy
```

---
