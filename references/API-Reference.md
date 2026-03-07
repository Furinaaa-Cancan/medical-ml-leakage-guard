# CLI API Reference / CLI API 参考

All scripts live under `scripts/`. Exit codes: **0** = pass, **2** = fail (gate blocked).

---

## Unified CLI — `mlgg.py`

**Purpose:** Single entry point for all ML Leakage Guard commands.

```
python3 scripts/mlgg.py <subcommand> [options] [-- <forwarded-args>]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `subcommand` | choice | ✅ | — | One of: `init`, `split`, `train`, `workflow`, `strict`, `preflight`, `doctor`, `onboarding`, `interactive`, `play`, `summary`, `authority`, `authority-release`, `authority-research-heart`, `benchmark-suite`, `adversarial`, `scan-diabetes` |
| `--python` | str | | `python3` | Python interpreter path |
| `--cwd` | str | | `.` | Working directory |
| `--dry-run` | flag | | `false` | Print commands without executing |
| `--interactive` | flag | | `false` | Enable interactive prompts |
| `--profile-name` | str | | — | Named profile for saving/loading configs |
| `--profile-dir` | str | | — | Directory for profile storage |
| `--save-profile` | flag | | `false` | Save current args as profile |
| `--load-profile` | flag | | `false` | Load args from profile |
| `--print-only` | flag | | `false` | Print resolved command only |
| `--accept-defaults` | flag | | `false` | Accept all defaults in interactive mode |
| `--error-json` | flag | | `false` | Output errors as JSON |

**Example:**
```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/demo --mode guided --yes
python3 scripts/mlgg.py split -- --input data.csv --output-dir out/ --patient-id-col pid
```

---

## Data Preparation

### `schema_preflight.py`

**Purpose:** Preflight schema checks for medical train/valid/test splits or a single CSV.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | | `""` | Path to train CSV |
| `--valid` | str | | `""` | Path to valid CSV |
| `--test` | str | | `""` | Path to test CSV |
| `--input-csv` | str | | `""` | Path to single complete CSV for pre-split checks |
| `--target-col` | str | | `y` | Preferred target column name |
| `--patient-id-col` | str | | `patient_id` | Preferred patient ID column name |
| `--time-col` | str | | `event_time` | Preferred index time column name |
| `--mapping-out` | str | | — | Optional output JSON for resolved field mapping |
| `--report` | str | | — | Optional output JSON report path |
| `--strict` | flag | | `false` | Fail when required columns need auto-mapping |

### `split_data.py`

**Purpose:** Split a single CSV into train/valid/test with medical safety guarantees (patient-level disjoint, temporal ordering, prevalence checks).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--input` | str | ✅ | — | Path to complete CSV file |
| `--output-dir` | str | ✅ | — | Directory to write train/valid/test CSVs |
| `--patient-id-col` | str | ✅ | — | Column for patient/entity ID (group-disjoint) |
| `--target-col` | str | | `y` | Binary target column name |
| `--time-col` | str | | — | Index time column for temporal splitting |
| `--strategy` | choice | | `grouped_temporal` | `grouped_temporal` / `grouped_random` / `stratified_grouped` |
| `--train-ratio` | float | | `0.6` | Train set ratio |
| `--valid-ratio` | float | | `0.2` | Valid set ratio |
| `--test-ratio` | float | | `0.2` | Test set ratio |
| `--seed` | int | | `20260228` | Random seed |
| `--report` | str | | — | Optional output JSON report path |
| `--split-protocol-out` | str | | — | Optional split protocol JSON output |
| `--min-rows-per-split` | int | | `10` | Minimum rows per split |

### `env_doctor.py`

**Purpose:** Check Python/runtime dependencies for ml-leakage-guard.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--require-optional-models` | str | | — | Comma-separated optional backends (xgboost,catboost,lightgbm,tabpfn,optuna) |
| `--strict` | flag | | `false` | Treat optional warnings as failures |
| `--report` | str | | — | Optional output JSON report path |

---

## Training & Evaluation

### `train_select_evaluate.py`

**Purpose:** Train/select/evaluate leakage-safe medical binary models.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Path to train CSV |
| `--valid` | str | ✅ | — | Path to valid CSV |
| `--test` | str | ✅ | — | Path to test CSV |
| `--target-col` | str | | `y` | Target column |
| `--patient-id-col` | str | | `patient_id` | Patient ID column for trace hashing |
| `--ignore-cols` | str | | `patient_id,event_time` | Comma-separated non-feature columns |
| `--performance-policy` | str | | — | Optional performance policy JSON path |
| `--missingness-policy` | str | | — | Optional missingness policy JSON path |
| `--selection-data` | str | | `cv_inner` | Model selection source (valid/cv_inner/nested_cv) |
| `--threshold-selection-split` | str | | `valid` | Split used for threshold selection |
| `--calibration-method` | choice | | `sigmoid` | `sigmoid` / `isotonic` / `power` / `beta` / `none` |
| `--cv-splits` | int | | `5` | CV folds for candidate scoring |
| `--model-pool` | str | | — | Comma-separated model families |
| `--include-optional-models` | flag | | `false` | Include optional backends (XGBoost, CatBoost, etc.) |
| `--ensemble-top-k` | int | | `3` | Top-k models for ensemble |
| `--device` | choice | | `cpu` | `cpu` / `gpu` / `mps` / `auto` |
| `--max-trials-per-family` | int | | `20` | Max hyperparameter trials per family |
| `--hyperparam-search` | choice | | `fixed_grid` | `random_subsample` / `fixed_grid` / `optuna` |
| `--optuna-trials` | int | | `50` | Optuna trials per family |
| `--n-jobs` | int | | `1` | Parallel workers |
| `--beta` | float | | `2.0` | Beta for F-beta threshold objective |
| `--sensitivity-floor` | float | | `0.85` | Minimum sensitivity |
| `--npv-floor` | float | | `0.90` | Minimum NPV |
| `--specificity-floor` | float | | `0.40` | Minimum specificity |
| `--ppv-floor` | float | | `0.55` | Minimum PPV |
| `--random-seed` | int | | `20260225` | Random seed |
| `--primary-metric` | str | | `pr_auc` | Primary optimization metric |
| `--bootstrap-resamples` | int | | `500` | Bootstrap samples for CI |
| `--ci-bootstrap-resamples` | int | | `2000` | Bootstrap samples for CI matrix |
| `--permutation-resamples` | int | | `1000` | Permutation resamples |
| `--fast-diagnostic-mode` | flag | | `false` | Skip expensive computations |
| `--model-selection-report-out` | str | ✅ | — | Output model selection report JSON |
| `--evaluation-report-out` | str | ✅ | — | Output evaluation report JSON |
| `--model-out` | str | | — | Output model artifact path |

**Output files:** `model_selection_report.json`, `evaluation_report.json`, `prediction_trace.csv.gz`, model pickle.

---

## Safety Gates

All gates share `--report` (optional JSON output) and `--strict` (fail on warnings) parameters.

### `calibration_dca_gate.py`

**Purpose:** Validate calibration ECE and decision curve net benefit.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--prediction-trace` | str | ✅ | — | Path to prediction_trace CSV/CSV.GZ |
| `--evaluation-report` | str | ✅ | — | Path to evaluation_report JSON |
| `--external-validation-report` | str | ✅ | — | Path to external_validation_report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `ci_matrix_gate.py`

**Purpose:** Validate CI width and bootstrap resampling adequacy.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Path to evaluation_report JSON |
| `--prediction-trace` | str | ✅ | — | Path to prediction_trace CSV/CSV.GZ |
| `--external-validation-report` | str | ✅ | — | Path to external_validation_report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |
| `--ci-matrix-report` | str | | — | CI matrix report JSON |

### `clinical_metrics_gate.py`

**Purpose:** Validate clinical metric floors (sensitivity, specificity, PPV, NPV).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Path to evaluation_report JSON |
| `--external-validation-report` | str | | — | External validation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `covariate_shift_gate.py`

**Purpose:** Detect covariate shift between splits using statistical tests.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Path to train CSV |
| `--test` | str | ✅ | — | Path to test CSV |
| `--target-col` | str | | `y` | Target column |
| `--ignore-cols` | str | | — | Comma-separated non-feature columns |

### `definition_variable_guard.py`

**Purpose:** Guard against disease-definition variable leakage in features.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--target` | str | ✅ | — | Target disease name |
| `--definition-spec` | str | ✅ | — | Phenotype definition JSON |
| `--train` | str | ✅ | — | Training CSV path |
| `--target-col` | str | | `y` | Target column |
| `--ignore-cols` | str | | — | Non-feature columns |

### `distribution_generalization_gate.py`

**Purpose:** Validate distribution shift and split separability.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Train CSV |
| `--valid` | str | ✅ | — | Valid CSV |
| `--test` | str | ✅ | — | Test CSV |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--external-validation-report` | str | | — | External validation report JSON |
| `--feature-group-spec` | str | | — | Feature group spec JSON |
| `--target-col` | str | | `y` | Target column |
| `--ignore-cols` | str | | — | Non-feature columns |
| `--performance-policy` | str | | — | Performance policy JSON |
| `--distribution-report` | str | | — | Distribution report JSON output |

### `evaluation_quality_gate.py`

**Purpose:** Validate CI and baseline comparison in evaluation report.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--ci-matrix-report` | str | | — | CI matrix report JSON |
| `--metric-name` | str | ✅ | — | Primary metric name (e.g. roc_auc) |
| `--metric-path` | str | | — | Dot path to metric value |
| `--primary-metric` | str | | — | Primary metric for selection |
| `--tolerance` | float | | — | Tolerance for metric comparison |
| `--min-resamples` | int | | — | Minimum bootstrap resamples |
| `--min-baseline-delta` | float | | — | Minimum improvement over baseline |
| `--max-ci-width` | float | | — | Maximum acceptable CI width |

### `execution_attestation_gate.py`

**Purpose:** Verify signed execution attestation and artifact integrity.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--attestation-spec` | str | ✅ | — | Execution attestation spec JSON |
| `--evaluation-report` | str | ✅ | — | Canonical evaluation report JSON |
| `--study-id` | str | | — | Expected study_id from request contract |
| `--run-id` | str | | — | Expected run_id |

### `external_validation_gate.py`

**Purpose:** Validate external cohort report with replayed trace metrics.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--external-validation-report` | str | ✅ | — | External validation report JSON |
| `--prediction-trace` | str | ✅ | — | Prediction trace CSV/CSV.GZ |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `feature_engineering_audit_gate.py`

**Purpose:** Validate feature engineering provenance/stability/reproducibility.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--feature-group-spec` | str | ✅ | — | Feature group spec JSON |
| `--feature-engineering-report` | str | ✅ | — | Feature engineering report JSON |
| `--lineage-spec` | str | ✅ | — | Feature lineage spec JSON |
| `--tuning-spec` | str | ✅ | — | Tuning protocol spec JSON |

### `feature_lineage_gate.py`

**Purpose:** Detect lineage-level leakage from disease-definition variables.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--target` | str | ✅ | — | Target name in definition spec |
| `--definition-spec` | str | ✅ | — | Phenotype definition JSON |
| `--lineage-spec` | str | ✅ | — | Feature lineage JSON |
| `--train` | str | ✅ | — | Training CSV |
| `--valid` | str | | — | Validation CSV |
| `--test` | str | | — | Test CSV |
| `--target-col` | str | | `y` | Target column |
| `--ignore-cols` | str | | — | Non-feature columns |
| `--allow-missing-lineage` | flag | | `false` | Allow missing lineage entries |

### `generalization_gap_gate.py`

**Purpose:** Validate train/valid/test metric gaps for overfitting risk.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `imbalance_policy_gate.py`

**Purpose:** Validate imbalance handling policy and split label distributions.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--policy-spec` | str | ✅ | — | Imbalance policy JSON |
| `--train` | str | ✅ | — | Train CSV |
| `--valid` | str | | — | Valid CSV |
| `--test` | str | ✅ | — | Test CSV |
| `--target-col` | str | | `y` | Label column |

### `leakage_gate.py`

**Purpose:** Detect data leakage between train/valid/test splits.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Train CSV |
| `--valid` | str | ✅ | — | Valid CSV |
| `--test` | str | ✅ | — | Test CSV |
| `--id-col` | str | | — | Entity ID column |
| `--time-col` | str | | — | Index/prediction time column |
| `--target-col` | str | | `y` | Target column |

### `metric_consistency_gate.py`

**Purpose:** Cross-validate metric consistency across report artifacts.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--prediction-trace` | str | ✅ | — | Prediction trace CSV/CSV.GZ |
| `--performance-policy` | str | | — | Performance policy JSON |

### `missingness_policy_gate.py`

**Purpose:** Validate missingness handling policy and imputation scope.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Train CSV |
| `--valid` | str | | — | Valid CSV |
| `--test` | str | ✅ | — | Test CSV |
| `--target-col` | str | | `y` | Target column |
| `--ignore-cols` | str | | — | Non-feature columns |
| `--policy-spec` | str | ✅ | — | Missingness policy JSON |
| `--evaluation-report` | str | | — | Evaluation report JSON |

### `model_selection_audit_gate.py`

**Purpose:** Audit model selection for leakage-safe practices.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--model-selection-report` | str | ✅ | — | Model selection report JSON |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `permutation_significance_gate.py`

**Purpose:** Validate permutation significance testing.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `prediction_replay_gate.py`

**Purpose:** Replay predictions from trace and verify metric consistency.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--prediction-trace` | str | ✅ | — | Prediction trace CSV/CSV.GZ |
| `--performance-policy` | str | | — | Performance policy JSON |

### `publication_gate.py`

**Purpose:** Final publication-grade gate aggregating all evidence.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--request` | str | ✅ | — | Request contract JSON |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `reporting_bias_gate.py`

**Purpose:** Detect selective reporting bias in metrics.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `request_contract_gate.py`

**Purpose:** Validate request contract paths, schemas, and policy compliance.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--request` | str | ✅ | — | Request contract JSON |

### `robustness_gate.py`

**Purpose:** Validate subgroup robustness metrics (PR-AUC drop/range).

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--robustness-report` | str | ✅ | — | Robustness report JSON |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `seed_stability_gate.py`

**Purpose:** Validate seed sensitivity and training stability.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--seed-sensitivity-report` | str | ✅ | — | Seed sensitivity report JSON |
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `self_critique_gate.py`

**Purpose:** Self-critique gate for model limitations and assumptions.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evaluation-report` | str | ✅ | — | Evaluation report JSON |
| `--performance-policy` | str | | — | Performance policy JSON |

### `split_protocol_gate.py`

**Purpose:** Validate split protocol and patient-level disjointness.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--train` | str | ✅ | — | Train CSV |
| `--valid` | str | ✅ | — | Valid CSV |
| `--test` | str | ✅ | — | Test CSV |
| `--id-col` | str | | — | Entity ID column |
| `--time-col` | str | | — | Index time column |
| `--target-col` | str | | `y` | Target column |

### `tuning_leakage_gate.py`

**Purpose:** Validate tuning protocol against leakage-safe requirements.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--tuning-spec` | str | ✅ | — | Tuning protocol JSON |
| `--id-col` | str | | — | Runtime ID column for grouped CV |
| `--has-valid-split` | flag | | `false` | Indicate dedicated validation split exists |

---

## Interactive Tools

### `mlgg_pixel.py`

**Purpose:** Pixel-art interactive CLI wizard for guided pipeline setup and execution.

```
python3 scripts/mlgg_pixel.py [--lang {en,zh}] [--dry-run]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--lang` | choice | | — | Set language directly (`en`/`zh`), skipping selection step |
| `--dry-run` | flag | | `false` | Print commands without executing |

### `mlgg_interactive.py`

**Purpose:** Interactive wizard for core commands (init/workflow/train/authority).

### `mlgg_onboarding.py`

**Purpose:** Guided novice onboarding (demo data → train → attestation → strict workflow).

---

## Analysis & Utility Tools

### `policy_generator.py`

**Purpose:** Generate a recommended `performance_policy.json` from evidence reports.

```
python3 scripts/policy_generator.py --evidence-dir <dir> [--margin 0.15] [--preset {lenient,standard,strict}] [--text] [--output <path>]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evidence-dir` | str | ✅ | — | Path to evidence directory |
| `--margin` | float | | `0.15` | Headroom margin fraction |
| `--preset` | choice | | — | Named preset (`lenient`/`standard`/`strict`), overrides `--margin` |
| `--text` | flag | | `false` | Output human-readable text instead of JSON |
| `--output` | str | | — | Write output to file (default: stdout) |

### `gate_timeline.py`

**Purpose:** Analyze gate execution timeline from an evidence directory.

```
python3 scripts/gate_timeline.py --evidence-dir <dir> [--json] [--top 5] [--output <path>]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evidence-dir` | str | ✅ | — | Path to evidence directory |
| `--json` | flag | | `false` | Output JSON instead of text |
| `--top` | int | | `5` | Number of bottleneck gates to show |
| `--output` | str | | — | Write output to file (default: stdout) |

### `gate_coverage_matrix.py`

**Purpose:** Generate a gate coverage matrix from an evidence directory against the full gate registry.

```
python3 scripts/gate_coverage_matrix.py --evidence-dir <dir> [--json] [--output <path>]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--evidence-dir` | str | ✅ | — | Path to evidence directory |
| `--json` | flag | | `false` | Output JSON instead of text |
| `--output` | str | | — | Write output to file (default: stdout) |

### `evidence_comparator.py`

**Purpose:** Compare two evidence directories (baseline vs current) and show gate-level diffs.

```
python3 scripts/evidence_comparator.py --baseline <dir> --current <dir> [--json] [--output <path>]
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `--baseline` | str | ✅ | — | Path to baseline evidence directory |
| `--current` | str | ✅ | — | Path to current evidence directory |
| `--json` | flag | | `false` | Output JSON instead of text |
| `--output` | str | | — | Write output to file (default: stdout) |

---

## Notes

- All gates exit **0** on pass and **2** on fail (fail-closed design).
- `--strict` promotes warnings to failures.
- `--report` writes structured JSON for programmatic consumption.
- Use `mlgg.py` as the unified entry point; forwarded args after `--` are passed to the underlying script.
