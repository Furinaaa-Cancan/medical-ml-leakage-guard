---
name: ml-leakage-guard
description: "Publication-grade medical prediction workflow with strict anti-data-leakage controls, phenotype-definition safeguards, lineage-based leakage detection, split-protocol verification, class-imbalance policy validation, hyperparameter-tuning isolation checks, falsification tests, and reproducibility gates. Use when building, reviewing, or debugging disease risk or prognosis models in EHR/claims/registry data, especially when target definitions, diagnosis codes, lab criteria, medications, temporal windows, and derived features can leak target information."
---

# ML Leakage Guard

## Objective (Goal Clarity)
Solve one narrow problem: produce leakage-safe, publication-grade medical prediction evidence.

Success is binary:
- `pass`: all hard gates pass and self-critique score reaches threshold.
- `fail`: any hard gate fails or strict review conditions are not met.

Never produce publication-grade claims without machine-checkable evidence artifacts.

## Input Contract (Structured Input)
Accept a structured request JSON, not free-form text.

Required fields:
- `study_id`
- `run_id`
- `target_name`
- `prediction_unit`
- `index_time_col`
- `label_col`
- `patient_id_col`
- `primary_metric`
- `claim_tier_target` (`leakage-audited` or `publication-grade`)
- `phenotype_definition_spec`
- `split_paths.train`
- `split_paths.test`

Publication-grade required fields:
- `feature_lineage_spec`
- `feature_group_spec`
- `split_protocol_spec`
- `imbalance_policy_spec`
- `missingness_policy_spec`
- `tuning_protocol_spec`
- `performance_policy_spec`
- `reporting_bias_checklist_spec`
- `execution_attestation_spec`
- `model_selection_report_file`
- `feature_engineering_report_file`
- `distribution_report_file`
- `robustness_report_file`
- `seed_sensitivity_report_file`
- `evaluation_report_file`
- `prediction_trace_file`
- `external_cohort_spec`
- `external_validation_report_file`
- `ci_matrix_report_file`
- `evaluation_metric_path`
- `permutation_null_metrics_file`
- `actual_primary_metric`
- `primary_metric` must be `pr_auc` for publication-grade strict mode.
- `evaluation_metric_path` terminal token must match `primary_metric` (after normalization).

Optional threshold keys under `thresholds`:
- `alpha` and `min_delta` for permutation significance gate.
- `min_baseline_delta`, `ci_min_resamples`, and `ci_max_width` for evaluation quality gate.

Path semantics:
- All relative paths in request JSON are resolved relative to the request file directory.

Template:
- `references/request-schema.example.json`
- `references/feature-lineage.example.json`
- `references/split-protocol.example.json`
- `references/imbalance-policy.example.json`
- `references/missingness-policy.example.json`
- `references/tuning-protocol.example.json`
- `references/performance-policy.example.json`
- `references/external-cohort-spec.example.json`
- `references/reporting-bias-checklist.example.json`
- `references/execution-attestation.example.json`
- `references/attestation-payload.example.json`
- `references/key-revocations.example.json`
- `references/attestation-timestamp-record.example.json`
- `references/attestation-transparency-record.example.json`
- `references/attestation-execution-receipt-record.example.json`
- `references/attestation-execution-log-record.example.json`
- `references/attestation-witness-record.example.json`
- `references/evaluation-report.example.json`
- `references/external-validation-report.example.json`
- `references/prediction-trace.example.csv`

Validate request first:

```bash
python3 scripts/request_contract_gate.py \
  --request configs/request.json \
  --report evidence/request_contract_report.json \
  --strict
```

## Hidden Workflow (Internal, Fail-Closed)
Use this internal sequence in order:
1. Validate request contract.
2. Lock data/config fingerprints (`manifest_lock.py`).
3. Run execution attestation gate (`execution_attestation_gate.py`).
4. Run split/time leakage gate (`leakage_gate.py`).
5. Run split protocol gate (`split_protocol_gate.py`).
6. Run covariate-shift gate (`covariate_shift_gate.py`).
7. Run reporting/bias checklist gate (`reporting_bias_gate.py`).
8. Run phenotype-definition leakage gate (`definition_variable_guard.py`).
9. Run lineage leakage gate (`feature_lineage_gate.py`).
10. Run imbalance policy gate (`imbalance_policy_gate.py`).
11. Run missingness policy gate (`missingness_policy_gate.py`).
12. Run tuning leakage gate (`tuning_leakage_gate.py`).
13. Run model-selection audit gate (`model_selection_audit_gate.py`).
14. Run feature-engineering audit gate (`feature_engineering_audit_gate.py`).
15. Run clinical-metrics gate (`clinical_metrics_gate.py`).
16. Run prediction-replay gate (`prediction_replay_gate.py`).
17. Run distribution-generalization gate (`distribution_generalization_gate.py`).
18. Run generalization-gap gate (`generalization_gap_gate.py`).
19. Run robustness gate (`robustness_gate.py`).
20. Run seed-stability gate (`seed_stability_gate.py`).
21. Run external-validation gate (`external_validation_gate.py`).
22. Run calibration+DCA gate (`calibration_dca_gate.py`).
23. Run CI-matrix gate (`ci_matrix_gate.py`).
24. Run metric consistency gate (`metric_consistency_gate.py`).
25. Run evaluation quality gate (`evaluation_quality_gate.py`).
26. Run permutation falsification gate (`permutation_significance_gate.py`).
27. Aggregate publication gate (`publication_gate.py`).
28. Run self-critique scoring gate (`self_critique_gate.py`).
29. Emit final report only if all strict gates pass.

Treat execution-attestation failures (signature/fingerprint/key-revocation/timestamp/transparency/execution-receipt/execution-log/witness-quorum/cross-role-authority-distinctness), disease-definition leakage, lineage ambiguity, metric-source ambiguity, split protocol violations, covariate-shift anomalies, class-imbalance misuse, missingness/imputation misuse, and tuning/test leakage as critical failures in strict mode.

## Output Contract (Machine-Parseable)
Produce these deterministic artifacts:
1. `evidence/request_contract_report.json`
2. `evidence/manifest.json`
3. `evidence/execution_attestation_report.json`
4. `evidence/reporting_bias_report.json`
5. `evidence/leakage_report.json`
6. `evidence/split_protocol_report.json`
7. `evidence/covariate_shift_report.json`
8. `evidence/definition_guard_report.json`
9. `evidence/lineage_report.json`
10. `evidence/imbalance_policy_report.json`
11. `evidence/missingness_policy_report.json`
12. `evidence/tuning_leakage_report.json`
13. `evidence/model_selection_audit_report.json`
14. `evidence/feature_engineering_audit_report.json`
15. `evidence/clinical_metrics_report.json`
16. `evidence/prediction_replay_report.json`
17. `evidence/distribution_generalization_report.json`
18. `evidence/generalization_gap_report.json`
19. `evidence/robustness_gate_report.json`
20. `evidence/seed_stability_report.json`
21. `evidence/external_validation_gate_report.json`
22. `evidence/calibration_dca_report.json`
23. `evidence/ci_matrix_gate_report.json`
24. `evidence/metric_consistency_report.json`
25. `evidence/evaluation_quality_report.json`
26. `evidence/permutation_report.json`
27. `evidence/publication_gate_report.json`
28. `evidence/self_critique_report.json`
29. `evidence/strict_pipeline_report.json`

Report status from each file must be machine-readable (`pass` or `fail`) with issue codes.

## Quality Control (Self-Critique)
Do not stop at initial gate pass.
Run `self_critique_gate.py` to score evidence quality and produce recommendations.

Publication-grade readiness requires:
- Strict-mode component reports.
- No blocking failures.
- Self-critique score at or above threshold (default `95`).

## Composability (Workflow Node Ready)
Each script is a composable node:
- Deterministic CLI interface.
- Deterministic JSON output.
- Deterministic exit code (`0` pass, `2` fail).

Use one-command orchestration for production use:

```bash
python3 scripts/run_strict_pipeline.py \
  --request configs/request.json \
  --evidence-dir evidence \
  --compare-manifest evidence/manifest_baseline.json \
  --strict
```

Productized one-command wrapper:

```bash
python3 scripts/run_productized_workflow.py \
  --request configs/request.json \
  --evidence-dir evidence \
  --allow-missing-compare \
  --strict
```

Novice onboarding wrapper (guided 8-step flow):

```bash
python3 scripts/mlgg.py onboarding \
  --project-root /tmp/mlgg_demo \
  --mode guided \
  --yes
```

Onboarding contract:
- `scripts/mlgg_onboarding.py` is strict-only (no policy downgrade path).
- Failure behavior:
  - default `--stop-on-fail` (fail-fast)
  - optional `--no-stop-on-fail` (collect full diagnostics while keeping fail-closed result)
  - wrapper route-conflict failure code: `authority_preset_route_override_forbidden`
- Modes:
  - `guided`: step-by-step command preview + confirmation.
  - `preview`: print the full 8-step command plan only.
  - `auto`: execute all steps non-interactively.
- Step order is fixed:
  1. `env_doctor.py`
  2. `init_project.py`
  3. `generate_demo_medical_dataset.py`
  4. config alignment to demo schema (`request/lineage/group/external spec`)
  5. `train_select_evaluate.py`
  6. `generate_execution_attestation.py` (+ keypair bootstrap if needed)
  7. `run_productized_workflow.py --strict --allow-missing-compare`
  8. `run_productized_workflow.py --strict --compare-manifest ...`
- Required report:
  - `evidence/onboarding_report.json` (`contract_version=onboarding_report.v2`)
  - report fields include `stop_on_fail`, `termination_reason`, `failure_codes`, `next_actions`, `copy_ready_commands`
- Offline demo data artifacts:
  - `data/train.csv`, `data/valid.csv`, `data/test.csv`
  - `data/external_2025_q4.csv` (`cross_period`)
  - `data/external_site_b.csv` (`cross_institution`)

This wrapper runs:
1. `env_doctor.py`
2. `schema_preflight.py`
3. `run_strict_pipeline.py`
4. `render_user_summary.py`

For first-run baseline bootstrap, you may omit `--compare-manifest` only with:
- `--allow-missing-compare`
- `run_strict_pipeline.py` always enforces `--strict` for publication-grade execution.
- `--allow-missing-compare` is bootstrap-only for artifact generation; publication-grade readiness still fails until baseline manifest comparison exists.
- `run_strict_pipeline.py` is publication-grade only; non-publication claim tiers are rejected.

## Personal UX Quickstart (Signed Attestation)
Create keypair once:

```bash
mkdir -p keys
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/attestation_priv.pem
openssl pkey -in keys/attestation_priv.pem -pubout -out keys/attestation_pub.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/timestamp_priv.pem
openssl pkey -in keys/timestamp_priv.pem -pubout -out keys/timestamp_pub.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/execution_priv.pem
openssl pkey -in keys/execution_priv.pem -pubout -out keys/execution_pub.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/execution_log_priv.pem
openssl pkey -in keys/execution_log_priv.pem -pubout -out keys/execution_log_pub.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/witness_a_priv.pem
openssl pkey -in keys/witness_a_priv.pem -pubout -out keys/witness_a_pub.pem
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out keys/witness_b_priv.pem
openssl pkey -in keys/witness_b_priv.pem -pubout -out keys/witness_b_pub.pem
```

Generate payload + signature + spec in one command:

```bash
python3 scripts/generate_execution_attestation.py \
  --study-id sepsis-risk-icu-v1 \
  --run-id sepsis-risk-icu-v1-train-2026-02-24-001 \
  --payload-out evidence/attestation_payload.json \
  --signature-out evidence/attestation.sig \
  --spec-out configs/execution_attestation.json \
  --private-key-file keys/attestation_priv.pem \
  --public-key-file keys/attestation_pub.pem \
  --timestamp-private-key-file keys/timestamp_priv.pem \
  --timestamp-public-key-file keys/timestamp_pub.pem \
  --execution-private-key-file keys/execution_priv.pem \
  --execution-public-key-file keys/execution_pub.pem \
  --execution-log-private-key-file keys/execution_log_priv.pem \
  --execution-log-public-key-file keys/execution_log_pub.pem \
  --require-independent-timestamp-authority \
  --require-independent-execution-authority \
  --require-independent-log-authority \
  --require-witness-quorum \
  --min-witness-count 2 \
  --require-independent-witness-keys \
  --require-witness-independence-from-signing \
  --witness "witness-a|keys/witness_a_pub.pem|keys/witness_a_priv.pem" \
  --witness "witness-b|keys/witness_b_pub.pem|keys/witness_b_priv.pem" \
  --command "python train.py --config configs/train_config.json --seed 42" \
  --artifact training_log=evidence/train.log \
  --artifact training_config=configs/train_config.json \
  --artifact model_artifact=models/model_v1.bin \
  --artifact evaluation_report=evidence/evaluation_report.json \
  --artifact prediction_trace=evidence/prediction_trace.csv.gz \
  --artifact external_validation_report=evidence/external_validation_report.json
```

This command also creates:
- `configs/key_revocations.json` (bootstrapped if missing)
- `evidence/attestation_timestamp_record.json` + `.sig`
- `evidence/attestation_transparency_record.json` + `.sig`
- `evidence/attestation_execution_receipt_record.json` + `.sig`
- `evidence/attestation_execution_log_record.json` + `.sig`
- `evidence/attestation_witness_record_1.json` + `.sig`
- `evidence/attestation_witness_record_2.json` + `.sig`

## Manual Strict Execution Order
If orchestration is unavailable, run in this exact order:
1. `request_contract_gate.py`
2. `manifest_lock.py` (with optional `--compare-with`)
3. `execution_attestation_gate.py`
4. `leakage_gate.py`
5. `split_protocol_gate.py`
6. `covariate_shift_gate.py`
7. `reporting_bias_gate.py`
8. `definition_variable_guard.py`
9. `feature_lineage_gate.py`
10. `imbalance_policy_gate.py`
11. `missingness_policy_gate.py`
12. `tuning_leakage_gate.py`
13. `model_selection_audit_gate.py`
14. `feature_engineering_audit_gate.py`
15. `clinical_metrics_gate.py`
16. `prediction_replay_gate.py`
17. `distribution_generalization_gate.py`
18. `generalization_gap_gate.py`
19. `robustness_gate.py`
20. `seed_stability_gate.py`
21. `external_validation_gate.py`
22. `calibration_dca_gate.py`
23. `ci_matrix_gate.py`
24. `metric_consistency_gate.py`
25. `evaluation_quality_gate.py`
26. `permutation_significance_gate.py`
27. `publication_gate.py`
28. `self_critique_gate.py`

If any step returns non-zero, stop and block claim release.

## Medical Non-Negotiable Rules
- Never tune on test data.
- Never fit preprocessors on combined train+validation+test.
- Never apply resampling/SMOTE on validation or test splits.
- Never select thresholds or calibrate probabilities on test split.
- Never fit imputers on validation/test distributions.
- Never use target/outcome information for feature imputation.
- Never run MICE at oversized scale without audited fallback evidence (`mice_with_scale_guard`).
- Never ignore severe train-vs-holdout distribution separability without explicit mitigation and downgrade.
- Never perform model ranking/selection with any test-derived signal.
- Never release without full split-level clinical metrics (accuracy/precision/PPV/NPV/sensitivity/specificity/F1/F2-beta/ROC-AUC/PR-AUC/Brier).
- Never ignore train/valid/test gap breaches beyond configured fail thresholds.
- Never claim publication-grade without signed execution attestation proving run command, timing, and artifact hashes.
- Never reuse revoked/expired/over-age signing keys for publication-grade claims.
- Never omit trusted timestamp or transparency-log records for publication-grade claims.
- Never omit signed execution-receipt proof (with exit code and timing consistency) for publication-grade claims.
- Never omit signed execution-log attestation binding `training_log` to payload hash for publication-grade claims.
- Never omit witness-quorum evidence with independent witness keys and minimum validated witness count for publication-grade claims.
- Never claim publication-grade if TRIPOD+AI/PROBAST+AI checklist has unmet required items.
- Never accept publication-grade primary metrics from non-test evaluation splits; evaluation report must explicitly declare `split=test`.
- Never claim publication-grade without valid primary-metric confidence interval and explicit baseline comparison in the evaluation artifact.
- Never include variables used to define the disease label as model predictors.
- Never include derived features whose lineage contains disease-defining variables.
- Never include post-index features for pre-index prediction tasks.
- Never report point estimates without uncertainty and robustness checks.
- Never claim causality from predictive associations.

## Resources

### scripts/
- `scripts/run_strict_pipeline.py`: single-entry strict orchestrator.
- `scripts/request_contract_gate.py`: request schema/path validation and publication-policy anti-downgrade checks.
- `scripts/mlgg.py`: unified command entrypoint (`onboarding`, `interactive`, `init`, `train`, `workflow`, ...).
- `scripts/mlgg_onboarding.py`: novice-guided strict onboarding flow and report emitter.
- `scripts/generate_demo_medical_dataset.py`: offline reproducible demo dataset generator.

### references/
- `references/Beginner-Quickstart.md`: bilingual novice quickstart (minimal loop + publication-grade loop).
- `references/Troubleshooting-Top20.md`: high-frequency failure code to diagnosis/fix/verify mapping.
- `scripts/manifest_lock.py`: dataset/protocol/evaluation/gate-script fingerprint and baseline comparison.
- `scripts/execution_attestation_gate.py`: signed run-attestation and artifact-hash verification gate.
- `scripts/generate_execution_attestation.py`: one-command payload/signature/spec/timestamp/transparency/execution-receipt/execution-log/witness-quorum generator for personal users.
- `scripts/reporting_bias_gate.py`: TRIPOD+AI / PROBAST+AI / STARD-AI checklist hard gate.
- `scripts/leakage_gate.py`: split contamination, ID overlap, and temporal boundary checks.
- `scripts/split_protocol_gate.py`: enforce split protocol consistency and temporal/group safeguards.
- `scripts/covariate_shift_gate.py`: train-vs-holdout covariate-shift and split separability risk gate.
- `scripts/definition_variable_guard.py`: hard gate against disease-definition variable leakage.
- `scripts/feature_lineage_gate.py`: hard gate against lineage-derived leakage.
- `scripts/imbalance_policy_gate.py`: validate class-imbalance strategy and train-only resampling policy.
- `scripts/missingness_policy_gate.py`: validate missing-data strategy, large-scale method suitability, and imputer isolation policy.
- `scripts/tuning_leakage_gate.py`: validate hyperparameter tuning/test-isolation protocol.
- `scripts/model_selection_audit_gate.py`: validate candidate pool, one-SE replay, and test-isolated model selection.
- `scripts/feature_engineering_audit_gate.py`: validate feature-group provenance, train-only engineering scope, stability evidence, and reproducibility fields.
- `scripts/clinical_metrics_gate.py`: validate clinical metric completeness and confusion-matrix consistency per split.
- `scripts/distribution_generalization_gate.py`: train-vs-holdout distribution shift, split separability, and transport-readiness gate.
- `scripts/generalization_gap_gate.py`: fail-closed overfitting gap checks across train/valid/test.
- `scripts/ci_matrix_gate.py`: bootstrap CI matrix gate for primary metric and transport-drop CI on internal and external cohorts.
- `scripts/metric_consistency_gate.py`: extract and validate metric from evaluation report.
- `scripts/evaluation_quality_gate.py`: enforce primary-metric CI quality and baseline improvement checks.
- `scripts/permutation_significance_gate.py`: falsification significance gate.
- `scripts/publication_gate.py`: aggregate fail-closed publication gate.
- `scripts/self_critique_gate.py`: quality scoring and reviewer-grade self-critique gate.
- `scripts/train_select_evaluate.py`: terminal-ready training, model selection, threshold selection, and evaluation artifact generator.
- `scripts/train_select_evaluate.py` model-pool controls: `--model-pool`, `--include-optional-models`, `--max-trials-per-family`, `--hyperparam-search`, `--n-jobs`.
- `scripts/train_select_evaluate.py` optional model backends: `xgboost` and `catboost` are auto-detected and fail-closed when explicitly requested but unavailable.
- `scripts/init_project.py`: one-command initialization for `configs/`, `data/`, `evidence/`, `models/`, `keys/`, plus `configs/request.json`.
- `scripts/schema_preflight.py`: train/valid/test schema checks with semantic column auto-mapping report.
- `scripts/env_doctor.py`: dependency and environment diagnostics with optional-backend checks.
- `scripts/render_user_summary.py`: user-facing markdown/json summary from strict evidence artifacts.
- `scripts/run_productized_workflow.py`: full UX wrapper (doctor -> preflight -> strict pipeline -> user summary).
- `scripts/mlgg.py`: unified terminal entrypoint for common commands (`interactive/init/doctor/preflight/workflow/strict/summary/train/authority/benchmark-suite/scan-diabetes/adversarial`).
- `scripts/mlgg_interactive.py`: terminal interactive wizard for core commands (`init/workflow/train/authority`) with command preview, confirm-before-run, and profile save/load.
- `experiments/authority-e2e/scan_stress_diabetes_feasibility.py`: stress-case diabetes feasibility scanner across target modes and row caps; outputs a fail-closed feasibility report.

### references/
- `references/request-schema.example.json`: structured request template.
- `references/feature-lineage.example.json`: lineage map template.
- `references/split-protocol.example.json`: split protocol template.
- `references/imbalance-policy.example.json`: class-imbalance policy template.
- `references/missingness-policy.example.json`: missing-data/imputation policy template.
- `references/tuning-protocol.example.json`: hyperparameter tuning protocol template.
- `references/performance-policy.example.json`: metric panel/threshold/gap policy template.
- `references/reporting-bias-checklist.example.json`: TRIPOD+AI / PROBAST+AI / STARD-AI checklist template.
- `references/execution-attestation.example.json`: signed execution-attestation spec template.
- `references/attestation-payload.example.json`: signed payload template with artifact hashes.
- `references/key-revocations.example.json`: key revocation list template.
- `references/attestation-timestamp-record.example.json`: trusted timestamp record template.
- `references/attestation-transparency-record.example.json`: transparency log record template.
- `references/attestation-execution-receipt-record.example.json`: execution receipt record template.
- `references/attestation-execution-log-record.example.json`: execution-log attestation record template.
- `references/attestation-witness-record.example.json`: witness attestation record template.
- `references/feature-group-spec.example.json`: feature group specification template (groups, train-only scope).
- `references/feature-engineering-report.example.json`: feature-engineering audit report template.
- `references/distribution-report.example.json`: distribution/shift report template.
- `references/ci-matrix-report.example.json`: CI matrix report template.
- `references/external-validation-report.example.json`: external validation report template.
- `references/evaluation-report.example.json`: evaluation metrics report template.
- `references/interactive-profile.example.json`: interactive CLI profile contract example (`contract_version/command/saved_at_utc/argument_values/python/cwd`).
- `references/benchmark-registry.json`: frozen benchmark dataset registry (contract `benchmark_registry.v1`).
- `references/stress-seed-search-report.v2.example.json`: stress seed/profile search contract template.
- `references/medical-disease-leakage.md`: medical phenotype leakage patterns and controls.
- `references/leakage-taxonomy.md`: leakage classes, red flags, and mitigations.
- `references/top-tier-rigor-checklist.md`: submission-grade hard gates.
- `references/external-benchmark-comparison.md`: external tool/guideline comparison and gap map.
- `references/release-benchmark-suite.md`: structured benchmark profile matrix and pass contract.
- `references/report-template.md`: reporting template for methods/results/robustness.

## Authority E2E Execution Notes
- Recommended single-entry CLI:
  - `python3 scripts/mlgg.py <command> [command-args]`
  - Examples:
    - `python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo`
    - `python3 scripts/mlgg.py train --interactive`
    - `python3 scripts/mlgg.py interactive --command workflow --profile-name demo --save-profile`
    - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --allow-missing-compare`
    - `python3 scripts/mlgg.py authority --include-stress-cases`
    - `python3 scripts/mlgg.py benchmark-suite --profile release` (recommended multi-dataset stability verdict)
    - `python3 scripts/mlgg.py benchmark-suite --profile release --repeat 3 --registry-file references/benchmark-registry.json`
    - `python3 scripts/mlgg.py authority-release` (recommended release stress path)
    - `python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060` (research/high-pressure mode)
    - preset wrappers are fixed-route; conflicting route flags are rejected fail-closed
    - add `--error-json` for machine-readable failures (`contract_version=mlgg_error.v1`)

- New-user order of operations:
  - `init` -> place split CSVs -> `train` (emit required evidence artifacts) -> `workflow --strict --allow-missing-compare`.
  - Follow-up reproducible runs should pass `--compare-manifest <project>/evidence/manifest_baseline.bootstrap.json`.

- Interactive wizard defaults:
  - Supports `init/workflow/train/authority`.
  - Preview command before execution, then require one confirm step.
  - Train wizard defaults `--include-optional-models` to off; enable manually only when optional backends are installed.
  - Train wizard defaults `--n-jobs` to `1` for cross-platform stability; increase manually for multi-core runs.
  - Train wizard emits `--external-validation-report-out` only when `external_cohort_spec` is provided.
  - Train wizard emits `--feature-engineering-report-out` only when `feature_group_spec` is provided.
  - Profile reuse:
    - `--profile-name <name> --save-profile`
    - `--profile-name <name> --load-profile`
    - `--accept-defaults` for non-blocking execution with defaults/profile values
  - Profile path defaults to `~/.mlgg/profiles` (override with `--profile-dir`).
  - For workflow wizard, `--strict` is always injected and cannot be bypassed by interactive mode.
  - Workflow wizard now auto-suggests evidence output under request project base (`<project>/evidence` when request is under `configs/`).
  - Authority wizard now defaults to release-grade stress path (`--include-stress-cases --stress-case-id uci-chronic-kidney-disease`);
    selecting `uci-heart-disease` is treated as advanced research/high-pressure mode.

- Use isolated output paths in concurrent runs:
  - `--summary-file`
  - `--stress-seed-cache-file`
  - `--stress-selection-file`
- Optional benchmark case switches:
  - `--include-ckd-case` (UCI Chronic Kidney Disease)
  - `--include-large-cases` (Diabetes130 large-cohort path)
  - `--diabetes-target-mode {lt30,gt30,any}` and `--diabetes-max-rows`
- Stress dataset selection:
  - `--stress-case-id {uci-diabetes-130-readmission,uci-heart-disease,uci-chronic-kidney-disease,uci-breast-cancer-wdbc}`
  - default is `uci-chronic-kidney-disease` (most stable publication-grade stress path in current benchmark set)
- Release benchmark blocking suites are `authority_release_core` + `adversarial_fail_closed`; `authority_release_extended` (Diabetes130) is kept as observational/non-blocking in release profile.
- Case-specific training configuration is enabled in authority E2E:
  - larger cohorts (e.g., Diabetes130) use expanded model pool (includes `xgboost` when installed), higher `max-trials-per-family`, and multi-core `--n-jobs`.
- Use `--run-tag` to bind all generated stress artifacts to a unique execution token.
- Stress seed-search profile bundles are selected with `--stress-profile-set` (default `strict_v1`).
- `--stress-seed-search` applies only to `--stress-case-id uci-heart-disease`; other stress cases run without seed search.
- CI coverage:
  - `.github/workflows/ci-smoke.yml` (push/PR/workflow_dispatch)
  - `.github/workflows/ci-full.yml` (nightly/workflow_dispatch release blocking benchmark-suite)
  - `.github/workflows/ci-extended.yml` (weekly/workflow_dispatch extended observational benchmark-suite)
- Optional diabetes feasibility auto-scan on failure:
  - `--auto-scan-diabetes-feasibility`
  - `--diabetes-feasibility-target-modes`
  - `--diabetes-feasibility-max-rows-options`
  - `--diabetes-feasibility-summary-dir`
  - `--diabetes-feasibility-report-file`
- Summary rows now include strict-pipeline root-cause fields for failed cases:
  - `root_failure_code_primary`
  - `root_failure_codes`
  - `failed_steps`
- Summary rows now also include `clinical_floor_gap_summary` with internal/external floor margins
  (`observed - required_min`) for `sensitivity/npv/specificity/ppv`.
- `stress_seed_search_report` v2 contract requires:
  - `contract_version`
  - `run_tag`
  - `policy_sha256`
  - `search_profile_set`
  - `selected_profile`
  - `dataset_fingerprint`
  - `code_revision_hint`

## Deep Review Fix Log

### Session 1 (Fixes applied to request_contract_gate.py, train_select_evaluate.py)

**Fix 1 — `request_contract_gate.py`: wrong error code in `validate_feature_engineering_report_shape`**
- The `except` block for JSON parse failure used `feature_group_spec_missing_or_invalid` instead of `feature_engineering_report_invalid`.
- Fixed: error code now correctly reflects `feature_engineering_report_invalid`.

**Fix 2 — `train_select_evaluate.py`: misleading hard-coded CI bounds in `transport_drop_ci`**
- `ci_95` and `ci_width` in the transport drop block were hard-coded to `[0.0, 0.0]` / `0.0`, falsely implying CIs were bootstrapped.
- Fixed: replaced with `null` and added `ci_note: "not_computed_point_estimate_only"`.
- Verified: `ci_matrix_gate.py` independently recomputes these CIs from prediction traces; downstream not affected.

### Session 2 (Fixes applied to feature_engineering_audit_gate.py, generalization_gap_gate.py, robustness_gate.py, seed_stability_gate.py)

**Fix 3 — `feature_engineering_audit_gate.py`: wrong error code for `feature_engineering_report` parse failure**
- Mirror of Fix 1: the `except` block used `feature_group_spec_missing_or_invalid` when parsing `feature_engineering_report` JSON.
- Fixed: error code now correctly set to `feature_engineering_report_invalid`.

**Fix 4 — `feature_engineering_audit_gate.py`: `to_float` missing `math.isfinite` guard**
- `to_float` accepted `inf` and `nan` as valid float values, inconsistent with all other gate scripts.
- Fixed: added `math.isfinite` guard and added `import math`.

**Fix 5 — `generalization_gap_gate.py`: `finish()` ignored `--strict` for warning escalation**
- `should_fail = bool(failures)` silently swallowed warnings even in strict mode.
- Fixed: `should_fail = bool(failures) or (args.strict and bool(warnings))`.

**Fix 6 — `robustness_gate.py`: same strict-mode bug as Fix 5**
- Fixed: `should_fail = bool(failures) or (args.strict and bool(warnings))`.

**Fix 7 — `seed_stability_gate.py`: same strict-mode bug as Fix 5**
- Fixed: `should_fail = bool(failures) or (args.strict and bool(warnings))`.

### Verified clean (no bugs found)
- `execution_attestation_gate.py`: `finish()` already correct; all validation logic and key/timestamp/transparency/receipt/log/witness-quorum checks are robust.
- `generalization_gap_gate.py`: `to_float` already had `math.isfinite`.
- All 27 gate scripts now uniformly use `bool(failures) or (args.strict and bool(warnings))` in `finish()`.
- All 11 `to_float` implementations across gate scripts now reject `inf`/`nan`.
