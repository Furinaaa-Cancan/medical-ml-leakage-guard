# Prediction Study Report Template (Leakage-Safe)

## 1. Problem Statement
- Prediction task:
- Decision use case:
- Prediction unit:
- `t_index` definition:
- Target definition:
- Target horizon:

## 2. Data Sources and Cohorts
- Data sources and acquisition windows:
- Inclusion/exclusion criteria:
- Entity definition (`patient_id`, `user_id`, etc.):
- Missingness profile:

## 3. Medical Phenotype Definition and Forbidden Predictors
- Endpoint phenotype definition source:
- Disease-defining variables:
- Forbidden proxy patterns:
- Adjudication log for ambiguous variables:
- Definition-variable exclusion evidence artifact (`definition_guard_report.json`):
- Lineage leakage evidence artifact (`lineage_report.json`):

## 4. Split Protocol (Frozen Before Modeling)
- Split strategy:
- Time boundaries and rationale:
- Group disjointness constraints:
- Train/validation/test sample counts:
- Event prevalence by split:
- Split protocol artifact (`split_protocol_report.json`):
- Covariate-shift artifact (`covariate_shift_report.json`):

## 5. Leakage Controls
- Leakage risk register summary:
- Feature timestamp legality process:
- Preprocessing isolation strategy:
- Missing-data mechanism assessment (MCAR/MAR/MNAR assumptions):
- Imputation strategy and fit scope:
- Hyperparameter tuning isolation strategy:
- Final test access policy:

## 6. Modeling Pipeline
- Candidate models and search space:
- Model-pool policy (`models`, `required_models`, `max_trials_per_family`, `search_strategy`, `n_jobs`):
- Model-selection evidence artifact (`model_selection_report.json`):
- One-SE + simplicity replay result (`model_selection_audit_report.json`):
- Preprocessing pipeline:
- Hyperparameter optimization procedure:
- Regularization/complexity controls by model family (L1/L2/ElasticNet/tree depth/L2 boosting penalties):
- Calibration and threshold selection protocol:
- Threshold constraint semantics (must report three fields): `constraints_satisfied_selection_split`, `constraints_satisfied_guard_split`, `constraints_satisfied_overall`.
- If `selection_split=cv_inner`, publish guard-split constraint evidence and confirm `constraints_satisfied_guard_split=true`.
- Class-imbalance strategy and train-only scope proof (`imbalance_policy_report.json`):
- Missingness policy and large-data method suitability proof (`missingness_policy_report.json`, include `mice_with_scale_guard` fallback evidence):
- Tuning leakage isolation proof (`tuning_leakage_report.json`):

## 7. Evaluation Plan
- Primary metrics:
- Secondary metrics:
- Required clinical metrics panel (accuracy, precision/PPV, NPV, sensitivity, specificity, F1, F2-beta, ROC-AUC, PR-AUC, Brier):
- Split-level metrics and confusion matrices (`split_metrics.train/valid/test`):
- Clinical metrics consistency artifact (`clinical_metrics_report.json`):
- Prediction replay artifact (`prediction_replay_report.json`) and row-level trace (`prediction_trace.csv.gz`):
- Confidence interval procedure:
- Baselines:
- Generalization gap policy (`performance_policy.json -> gap_thresholds`):
- Generalization gap artifact (`generalization_gap_report.json`):
- Seed sensitivity artifact (`seed_sensitivity_report.json`) and seed-stability gate (`seed_stability_report.json`):
- External validation artifact (`external_validation_report.json`) and gate (`external_validation_gate_report.json`):
- Calibration + DCA gate artifact (`calibration_dca_report.json`) on internal test + all external cohorts:
- Evaluation report artifact and explicit metric extraction path (`evaluation_metric_path`):
- Evaluation report split declaration (`split=test`) for final claim metrics:
- Metric-source consistency check (no conflicting duplicate metric values):

## 8. Robustness and Falsification
- Label permutation test results:
- Split separability/covariate-shift risk assessment:
- Time-slice robustness:
- Group-holdout robustness:
- Seed sensitivity:
- Suspect-feature ablation:

## 9. Main Results
- Train vs valid vs test metric table (include gap columns):
- Selected model and hyperparameters (with regularization profile):
- Validation performance:
- Final test performance:
- Calibration results (ECE / slope / intercept on internal test + all external cohorts):
- Decision-curve analysis results (net benefit vs treat-all/treat-none across policy threshold grid):
- Error analysis:

## 10. Reproducibility Artifacts
- Code version/commit:
- Data snapshot IDs:
- Environment details:
- Config files and random seeds:
- Signed execution attestation spec (`execution_attestation.json`):
- Signed payload (`attestation_payload.json`) and detached signature (`attestation.sig`):
- Revocation list evidence (`key_revocations.json`) and signing key lifecycle evidence:
- Trusted timestamp record + signature (`attestation_timestamp_record.json`, `attestation_timestamp_record.sig`):
- Transparency-log record + signature (`attestation_transparency_record.json`, `attestation_transparency_record.sig`):
- Execution-receipt record + signature (`attestation_execution_receipt_record.json`, `attestation_execution_receipt_record.sig`):
- Execution-log attestation record + signature (`attestation_execution_log_record.json`, `attestation_execution_log_record.sig`):
- Witness-quorum records + signatures (`attestation_witness_record_1.json/.sig`, `attestation_witness_record_2.json/.sig`, ...):
- Execution-log related hash bindings include `prediction_trace_sha256` and `external_validation_report_sha256`:
- Witness minimum count and key-independence policy outcome (`execution_attestation_report.json -> summary.witness_quorum`):
- Re-run reproducibility check:
- Baseline manifest comparison result (`manifest.comparison.matched`):
- Stress seed-search evidence (`stress_seed_search_report.json`) and selected seed/profile freeze record (`stress_seed_selection.json`) when stress mode is used:
- Stress report contract fields (`contract_version`, `run_tag`, `policy_sha256`, `search_profile_set`, `selected_profile`, `dataset_fingerprint`, `code_revision_hint`) are present and consistent:
- Evidence manifest (`manifest.json`):
- Gate script fingerprint lock (all strict gate scripts included in manifest):
- Split protocol gate artifact (`split_protocol_report.json`):
- Covariate-shift gate artifact (`covariate_shift_report.json`):
- Imbalance policy gate artifact (`imbalance_policy_report.json`):
- Missingness policy gate artifact (`missingness_policy_report.json`):
- Tuning leakage gate artifact (`tuning_leakage_report.json`):
- Model-selection audit artifact (`model_selection_audit_report.json`):
- Clinical metrics gate artifact (`clinical_metrics_report.json`):
- Prediction replay gate artifact (`prediction_replay_report.json`):
- Generalization gap gate artifact (`generalization_gap_report.json`):
- Robustness gate artifact (`robustness_report.json`):
- External validation gate artifact (`external_validation_gate_report.json`):
- Calibration/DCA gate artifact (`calibration_dca_report.json`):
- Metric consistency artifact (`metric_consistency_report.json`):
- Publication gate artifact (`publication_gate_report.json`):

## 11. Limitations and Scope
- Potential residual leakage risks:
- External validity limits:
- Non-causal interpretation statement:

## 12. Final Claim Tier
Choose one:
- Preliminary exploratory result.
- Leakage-audited internal validation.
- Publication-grade claim (all hard gates passed).
