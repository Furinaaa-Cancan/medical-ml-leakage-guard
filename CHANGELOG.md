# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Documentation**
  - System architecture document with Mermaid flowchart (`references/Architecture.md`) (#71)
  - Contributing guide (`CONTRIBUTING.md`) (#72)
  - CLI API Reference for all scripts (`references/API-Reference.md`) (#67)
  - Complete Google-style docstrings for `train_select_evaluate.py` (69 functions) (#68)
  - Complete Google-style docstrings for `split_data.py` (17 functions) (#69)
  - Expanded Troubleshooting with 12 new failure codes (#66)
  - Bilingual README with detailed usage guide (sections 0–11)
  - Beginner quickstart guide (`references/Beginner-Quickstart.md`)
  - PolyForm Noncommercial 1.0.0 license

- **Pixel CLI (`mlgg play`)**
  - Pixel-art interactive CLI launcher with arrow-key navigation (#56–#65)
  - Phased progress bar with percentage display (#56)
  - Export CLI Command option (#57)
  - Training results display after run (#58)
  - Page Up/Down support for select and multi_select (#59)
  - `--lang {en,zh}` bilingual support (#60)
  - Advanced settings for ignore-cols, n-jobs, max-trials (#62)
  - Friendly error messages (#63)
  - Run history recording and "Repeat last run" (#64)
  - `--dry-run` mode (#65)

- **Test Suites (1,100+ test cases)**
  - `request_contract_gate` (107 cases), `split_protocol_gate` (75),
    `covariate_shift_gate` (80), `reporting_bias_gate` (34),
    `model_selection_audit_gate` (69), `clinical_metrics_gate` (54),
    `prediction_replay_gate` (24), `distribution_generalization_gate` (24),
    `generalization_gap_gate` (18), `robustness_gate` (30),
    `seed_stability_gate` (27), `external_validation_gate` (20),
    `ci_matrix_gate` (25), `metric_consistency_gate` (45),
    `self_critique_gate` (19), `execution_attestation_gate` (41),
    `generate_execution_attestation` (27), `render_user_summary` (24),
    `env_doctor` (13), `train_select_evaluate` (67),
    `mlgg.py` routing (25), `mlgg_onboarding` (29),
    `run_strict_pipeline` (11), `run_productized_workflow` (11),
    `init_project` (12), `generate_demo_medical_dataset` (16),
    `mlgg_interactive` (33), wizard + download (16)

- **Single-CSV Auto-Split Workflow**
  - `split_data.py` with 3 strategies: grouped_temporal, grouped_random,
    stratified_grouped
  - Patient-level disjoint splits, temporal ordering, prevalence checks
  - NaN patient_id/target exclusion, row count preservation, SHA-256
    input fingerprint, atomic file writes

- **Real Data Download**
  - `examples/download_real_data.py` for UCI heart disease and breast
    cancer datasets with streaming progress display (#61)

- **Model Pool Expansion**
  - LightGBM, SVM (linear/RBF), TabPFN backends
  - Ensemble methods: soft voting, hard voting, stacking
  - Optuna hyperparameter optimization
  - Device selection (CPU/GPU/MPS)

- **Release Benchmark Suite**
  - `benchmark-suite --profile release` with multi-dataset stability matrix
  - Repeat consistency gate, JUnit output, suite timeout budget
  - Frozen benchmark registry (`benchmark_registry.v1`)
  - Observational diagnostics for non-blocking failures

- **Gate Pipeline (28 gates)**
  - Request contract validation with publication-policy anti-downgrade
  - Manifest fingerprint locking with baseline comparison
  - Signed execution attestation with witness quorum, timestamp trust,
    transparency log, execution receipt, and execution log
  - Split/temporal/ID leakage detection
  - Split protocol enforcement
  - Covariate shift and split separability risk gate
  - TRIPOD+AI / PROBAST+AI / STARD-AI reporting checklist gate
  - Disease-definition variable leakage guard
  - Feature lineage leakage gate
  - Class-imbalance policy gate (train-only resampling)
  - Missingness policy gate with MICE scale guard
  - Tuning leakage isolation gate
  - Model selection audit with one-SE replay
  - Feature engineering audit with stability evidence
  - Clinical metrics completeness gate
  - Prediction replay gate for metric reproducibility
  - Distribution generalization and transport readiness gate
  - Generalization gap overfitting detection
  - Subgroup robustness gate
  - Multi-seed stability gate
  - External validation gate (cross-period, cross-institution)
  - Calibration and decision curve analysis gate
  - Bootstrap CI matrix gate
  - Metric consistency gate
  - Evaluation quality gate with baseline improvement check
  - Permutation falsification significance gate
  - Aggregate publication gate
  - Self-critique scoring gate

- **Orchestration**
  - `run_strict_pipeline.py`: sequential 28-gate orchestrator
  - `run_productized_workflow.py`: doctor → preflight → pipeline → summary
  - `mlgg_onboarding.py`: guided 8-step novice flow with preview mode
  - `mlgg.py`: unified CLI entry point
  - `mlgg_interactive.py`: terminal wizard with profiles and command preview

- **Authority E2E**
  - CKD benchmark (stable publication-grade stress path)
  - Heart disease stress search (advanced research mode)
  - Diabetes130 large-cohort integration
  - Adversarial fail-closed validation harness
  - `authority-release` and `authority-research-heart` preset wrappers
  - Machine-readable error payloads (`--error-json`)

- **Infrastructure**
  - `pyproject.toml` with pip install support and `mlgg` console entry point
  - `_gate_utils.py` shared utilities for all gate scripts
  - `schema_preflight.py` for train/valid/test schema validation
  - `env_doctor.py` for dependency diagnostics
  - CI workflows: smoke (push/PR), full (nightly), extended (weekly)

### Fixed

- Docstring accuracy: `impute_numeric_frame` Returns, `build_imputer`
  Args, `prepare_xy` Raises, `apply_probability_calibrator` Raises
- `generalization_gap_gate.py` / `robustness_gate.py` / `seed_stability_gate.py`:
  `finish()` ignored `--strict` for warning escalation
- `feature_engineering_audit_gate.py`: wrong error code for report parse
  failure; `to_float` missing `math.isfinite` guard
- `request_contract_gate.py`: wrong error code in
  `validate_feature_engineering_report_shape`
- `train_select_evaluate.py`: misleading hard-coded CI bounds in
  `transport_drop_ci` replaced with `null`
- `distribution_generalization_gate.py`: missing `sys` import
- `feature_engineering_audit_gate.py`: missing `Set` import and constants
- Metric-name spoofing and validation-metric spoofing blocked
- Lineage normalization hardened
- Profile key leak in interactive wizard
- Onboarding hardcoded demo columns break `--input-csv` mode
- `validate_binary_target` bug in split_data.py
- Collision-safe intermediate column names in temporal split
- CKD ARFF parser, categorical encoding, lambda closure fixes
- Download error handling and temp file cleanup

### Changed

- Shared gate utilities (`add_issue`, `load_json`, `write_json`, `to_float`)
  extracted to `_gate_utils.py` and imported by all 27 gate scripts
- All `to_float` implementations reject `inf`/`nan` with `math.isfinite`
- All gate `finish()` functions use uniform strict-mode warning escalation
- README Chinese sections aligned with English (commands, bullets, terms)
- Atomic file writes for all CSV and JSON outputs
- Exit style unified across all scripts
