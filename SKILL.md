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
- `split_protocol_spec`
- `imbalance_policy_spec`
- `missingness_policy_spec`
- `tuning_protocol_spec`
- `execution_attestation_spec`
- `evaluation_report_file`
- `evaluation_metric_path`
- `permutation_null_metrics_file`
- `actual_primary_metric`
- `evaluation_metric_path` terminal token must match `primary_metric` (after normalization).

Path semantics:
- All relative paths in request JSON are resolved relative to the request file directory.

Template:
- `references/request-schema.example.json`
- `references/feature-lineage.example.json`
- `references/split-protocol.example.json`
- `references/imbalance-policy.example.json`
- `references/missingness-policy.example.json`
- `references/tuning-protocol.example.json`
- `references/execution-attestation.example.json`
- `references/attestation-payload.example.json`
- `references/evaluation-report.example.json`

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
7. Run phenotype-definition leakage gate (`definition_variable_guard.py`).
8. Run lineage leakage gate (`feature_lineage_gate.py`).
9. Run imbalance policy gate (`imbalance_policy_gate.py`).
10. Run missingness policy gate (`missingness_policy_gate.py`).
11. Run tuning leakage gate (`tuning_leakage_gate.py`).
12. Run metric consistency gate (`metric_consistency_gate.py`).
13. Run permutation falsification gate (`permutation_significance_gate.py`).
14. Aggregate publication gate (`publication_gate.py`).
15. Run self-critique scoring gate (`self_critique_gate.py`).
16. Emit final report only if all strict gates pass.

Treat execution-attestation failures, disease-definition leakage, lineage ambiguity, metric-source ambiguity, split protocol violations, covariate-shift anomalies, class-imbalance misuse, missingness/imputation misuse, and tuning/test leakage as critical failures in strict mode.

## Output Contract (Machine-Parseable)
Produce these deterministic artifacts:
1. `evidence/request_contract_report.json`
2. `evidence/manifest.json`
3. `evidence/execution_attestation_report.json`
4. `evidence/leakage_report.json`
5. `evidence/split_protocol_report.json`
6. `evidence/covariate_shift_report.json`
7. `evidence/definition_guard_report.json`
8. `evidence/lineage_report.json`
9. `evidence/imbalance_policy_report.json`
10. `evidence/missingness_policy_report.json`
11. `evidence/tuning_leakage_report.json`
12. `evidence/metric_consistency_report.json`
13. `evidence/permutation_report.json`
14. `evidence/publication_gate_report.json`
15. `evidence/self_critique_report.json`
16. `evidence/strict_pipeline_report.json`

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
  --command "python train.py --config configs/train_config.json --seed 42" \
  --artifact training_log=evidence/train.log \
  --artifact training_config=configs/train_config.json \
  --artifact model_artifact=models/model_v1.bin \
  --artifact evaluation_report=evidence/evaluation_report.json
```

## Manual Strict Execution Order
If orchestration is unavailable, run in this exact order:
1. `request_contract_gate.py`
2. `manifest_lock.py` (with optional `--compare-with`)
3. `execution_attestation_gate.py`
4. `leakage_gate.py`
5. `split_protocol_gate.py`
6. `covariate_shift_gate.py`
7. `definition_variable_guard.py`
8. `feature_lineage_gate.py`
9. `imbalance_policy_gate.py`
10. `missingness_policy_gate.py`
11. `tuning_leakage_gate.py`
12. `metric_consistency_gate.py`
13. `permutation_significance_gate.py`
14. `publication_gate.py`
15. `self_critique_gate.py`

If any step returns non-zero, stop and block claim release.

## Medical Non-Negotiable Rules
- Never tune on test data.
- Never fit preprocessors on combined train+validation+test.
- Never apply resampling/SMOTE on validation or test splits.
- Never select thresholds or calibrate probabilities on test split.
- Never fit imputers on validation/test distributions.
- Never use target/outcome information for feature imputation.
- Never ignore severe train-vs-holdout distribution separability without explicit mitigation and downgrade.
- Never claim publication-grade without signed execution attestation proving run command, timing, and artifact hashes.
- Never accept publication-grade primary metrics from non-test evaluation splits; evaluation report must explicitly declare `split=test`.
- Never include variables used to define the disease label as model predictors.
- Never include derived features whose lineage contains disease-defining variables.
- Never include post-index features for pre-index prediction tasks.
- Never report point estimates without uncertainty and robustness checks.
- Never claim causality from predictive associations.

## Resources

### scripts/
- `scripts/run_strict_pipeline.py`: single-entry strict orchestrator.
- `scripts/request_contract_gate.py`: request schema and path validation.
- `scripts/manifest_lock.py`: dataset/protocol/evaluation/gate-script fingerprint and baseline comparison.
- `scripts/execution_attestation_gate.py`: signed run-attestation and artifact-hash verification gate.
- `scripts/generate_execution_attestation.py`: one-command payload/signature/spec generator for personal users.
- `scripts/leakage_gate.py`: split contamination, ID overlap, and temporal boundary checks.
- `scripts/split_protocol_gate.py`: enforce split protocol consistency and temporal/group safeguards.
- `scripts/covariate_shift_gate.py`: train-vs-holdout covariate-shift and split separability risk gate.
- `scripts/definition_variable_guard.py`: hard gate against disease-definition variable leakage.
- `scripts/feature_lineage_gate.py`: hard gate against lineage-derived leakage.
- `scripts/imbalance_policy_gate.py`: validate class-imbalance strategy and train-only resampling policy.
- `scripts/missingness_policy_gate.py`: validate missing-data strategy, large-scale method suitability, and imputer isolation policy.
- `scripts/tuning_leakage_gate.py`: validate hyperparameter tuning/test-isolation protocol.
- `scripts/metric_consistency_gate.py`: extract and validate metric from evaluation report.
- `scripts/permutation_significance_gate.py`: falsification significance gate.
- `scripts/publication_gate.py`: aggregate fail-closed publication gate.
- `scripts/self_critique_gate.py`: quality scoring and reviewer-grade self-critique gate.

### references/
- `references/request-schema.example.json`: structured request template.
- `references/feature-lineage.example.json`: lineage map template.
- `references/split-protocol.example.json`: split protocol template.
- `references/imbalance-policy.example.json`: class-imbalance policy template.
- `references/missingness-policy.example.json`: missing-data/imputation policy template.
- `references/tuning-protocol.example.json`: hyperparameter tuning protocol template.
- `references/execution-attestation.example.json`: signed execution-attestation spec template.
- `references/attestation-payload.example.json`: signed payload template with artifact hashes.
- `references/evaluation-report.example.json`: evaluation metrics report template.
- `references/medical-disease-leakage.md`: medical phenotype leakage patterns and controls.
- `references/leakage-taxonomy.md`: leakage classes, red flags, and mitigations.
- `references/top-tier-rigor-checklist.md`: submission-grade hard gates.
- `references/external-benchmark-comparison.md`: external tool/guideline comparison and gap map.
- `references/report-template.md`: reporting template for methods/results/robustness.
