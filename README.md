# silent-corruption

> **Central claim:** Silent data corruption is more dangerous than visible failure.
> A crashed system is obvious — you fix it. A system that silently writes wrong values
> continues to operate, builds false confidence, and propagates errors downstream.
> This project tests that claim on simulated particle detector data using two independent
> detection systems, then measures exactly how much corruption slips through both.

---

## What this is

A two-system face-off for detecting silent corruption in scientific instrument data —
built as a direct precursor to anomaly detection work on real HEP datasets (CERN track).

**System 1 — Invariant rules engine:** Deterministic checks encoding physical constraints:
field ranges, registered detector IDs, timestamp monotonicity, cross-field amplitude
consistency, hit multiplicity thresholds, rolling z-score drift detection.

**System 2 — PyTorch autoencoder:** Trained on clean data only. Flags rows whose
reconstruction error exceeds the p99 threshold (0.2485). Treats anomaly score as a
hypothesis, not a verdict.

Both systems run on the same corrupted evaluation set. Results are compared against
sealed ground-truth labels opened only at evaluation time.

---

## Key results

554 corrupted rows across 6 corruption types injected into a 10,000-row eval set.

| Corruption Type  | Labeled | Rules TP | Rules Recall | AE TP | AE Recall |
| ---------------- | ------- | -------- | ------------ | ----- | --------- |
| bit_flip         | 80      | 80       | 1.000        | 70    | 0.875     |
| phantom_detector | 80      | 80       | 1.000        | 80    | 1.000     |
| stale_timestamp  | 80      | 80       | 1.000        | 2     | 0.025     |
| thermal_spike    | 158     | 158      | 1.000        | 81    | 0.513     |
| cross_field      | 80      | 76       | 0.950        | 59    | 0.738     |
| subtle_drift     | 76      | 11       | 0.145        | 1     | 0.013     |

**Aggregate:**

| Detector | Precision | Recall | F1    | Throughput     |
| -------- | --------- | ------ | ----- | -------------- |
| Rules    | 0.203     | 0.875  | 0.329 | ~550k rows/sec |
| AE       | 0.724     | 0.529  | 0.611 | ~490k rows/sec |

**68 subtle_drift rows passed both detectors completely undetected.** Energy readings
of 6.8–141.4 keV with stable temperatures — indistinguishable from clean data at the
row level. No alarm fires. Those values propagate into downstream analysis. That is
the proof of the central claim.

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

Conversely, the AE catches 3× more subtle_drift than rules (1.3% vs 14.5%), because
gradual energy drift distributes reconstruction error across multiple features in a way
no single rule threshold catches cleanly.

**The combination is the point.**
Rules run first: fast, cheap, zero false negatives on known violation types. Unresolved
rows pass to the AE: catches distributional residuals rules miss. AE flags get interrogated
by rules to produce a human-readable explanation. This is how production systems at
CERN, financial exchanges, and data-heavy science actually work.

**Interpretability is not symmetric.**
Rules give an exact reason code in one line: `range_check | severity=3`.
The AE gives a reconstruction error and a top contributing feature: `recon_err=2831.3 | top_feature: err_energy_deposit_keV`.
That's not an explanation — it's a starting point for one.

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

Domain chosen for direct alignment with CERN data quality infrastructure and as a
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
│   └── compare.py          # Ground truth evaluation, computes full precision/recall matrix
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
python data/generator.py          # → train_clean.csv, eval_clean.csv, channel_params.json

# 3. Inject corruption (seals labels.csv)
python data/inject.py             # → eval_corrupted.csv, labels.csv
chmod 000 data/labels.csv         # seal until evaluation

# 4. Rules engine
python rules/test_checker.py      # 22/22 — verify before running on eval
PYTHONPATH=. python rules/checker.py   # → rules/flags_rules.csv

# 5. Autoencoder
python model/preprocess.py        # → model/scaler.pkl, model/feature_cols.json
PYTHONPATH=. python model/train.py     # → model/ae_checkpoint.pt, model/threshold.json
PYTHONPATH=. python model/infer.py     # → model/scores_ae.csv

# 6. Evaluation
chmod 644 data/labels.csv
PYTHONPATH=. python eval/compare.py    # → eval/results_by_type.csv, eval/results_summary.csv
```

---

## Design decisions worth noting

**Why exact-match evaluation, not window-match.**
A rules engine that fires on the exact corrupted row is meaningfully better than one that
fires one row off. Window matching would obscure that distinction. Thermal spike labels
cover both the spike row and the recovery row — the double-labeling problem is handled
at injection time, not at evaluation time.

**Why seal labels.csv.**
If you read the ground truth while building detectors, you unconsciously overfit your
rules to the exact corruptions you injected. The seal enforces the discipline of operating
without confirmation — which is the actual production condition.

**Why the AE trains on clean data only.**
If corrupted rows leak into training, the model learns to reconstruct corruptions as
normal. The clean-only split is enforced before injection runs.

**Why the bottleneck is 4.**
For 8 tabular features, a bottleneck of 4 forces meaningful compression without
collapsing to a representation too small to reconstruct normal data faithfully. Too
small → high baseline error, poor discrimination. Too large → memorisation, defeats
the purpose.

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
