---
name: ml-leakage-guard
description: "Publication-grade medical prediction workflow with strict anti-data-leakage controls, phenotype-definition safeguards, lineage-based leakage detection, falsification tests, and reproducibility gates. Use when building, reviewing, or debugging disease risk or prognosis models in EHR/claims/registry data, especially when target definitions, diagnosis codes, lab criteria, medications, temporal windows, and derived features can leak target information."
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
- `evaluation_report_file`
- `evaluation_metric_path`
- `permutation_null_metrics_file`
- `actual_primary_metric`

Path semantics:
- All relative paths in request JSON are resolved relative to the request file directory.

Template:
- `references/request-schema.example.json`
- `references/feature-lineage.example.json`
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
3. Run split/time leakage gate (`leakage_gate.py`).
4. Run phenotype-definition leakage gate (`definition_variable_guard.py`).
5. Run lineage leakage gate (`feature_lineage_gate.py`).
6. Run metric consistency gate (`metric_consistency_gate.py`).
7. Run permutation falsification gate (`permutation_significance_gate.py`).
8. Aggregate publication gate (`publication_gate.py`).
9. Run self-critique scoring gate (`self_critique_gate.py`).
10. Emit final report only if all strict gates pass.

Treat disease-definition leakage, lineage ambiguity, and metric-source ambiguity as critical failure in strict mode.

## Output Contract (Machine-Parseable)
Produce these deterministic artifacts:
1. `evidence/request_contract_report.json`
2. `evidence/manifest.json`
3. `evidence/leakage_report.json`
4. `evidence/definition_guard_report.json`
5. `evidence/lineage_report.json`
6. `evidence/metric_consistency_report.json`
7. `evidence/permutation_report.json`
8. `evidence/publication_gate_report.json`
9. `evidence/self_critique_report.json`
10. `evidence/strict_pipeline_report.json`

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
- In strict mode this downgrades only `manifest_not_comparable` to non-blocking warning; other warnings still block.

## Manual Strict Execution Order
If orchestration is unavailable, run in this exact order:
1. `request_contract_gate.py`
2. `manifest_lock.py` (with optional `--compare-with`)
3. `leakage_gate.py`
4. `definition_variable_guard.py`
5. `feature_lineage_gate.py`
6. `metric_consistency_gate.py`
7. `permutation_significance_gate.py`
8. `publication_gate.py`
9. `self_critique_gate.py`

If any step returns non-zero, stop and block claim release.

## Medical Non-Negotiable Rules
- Never tune on test data.
- Never fit preprocessors on combined train+validation+test.
- Never include variables used to define the disease label as model predictors.
- Never include derived features whose lineage contains disease-defining variables.
- Never include post-index features for pre-index prediction tasks.
- Never report point estimates without uncertainty and robustness checks.
- Never claim causality from predictive associations.

## Resources

### scripts/
- `scripts/run_strict_pipeline.py`: single-entry strict orchestrator.
- `scripts/request_contract_gate.py`: request schema and path validation.
- `scripts/manifest_lock.py`: dataset/protocol fingerprint and baseline comparison.
- `scripts/leakage_gate.py`: split contamination, ID overlap, and temporal boundary checks.
- `scripts/definition_variable_guard.py`: hard gate against disease-definition variable leakage.
- `scripts/feature_lineage_gate.py`: hard gate against lineage-derived leakage.
- `scripts/metric_consistency_gate.py`: extract and validate metric from evaluation report.
- `scripts/permutation_significance_gate.py`: falsification significance gate.
- `scripts/publication_gate.py`: aggregate fail-closed publication gate.
- `scripts/self_critique_gate.py`: quality scoring and reviewer-grade self-critique gate.

### references/
- `references/request-schema.example.json`: structured request template.
- `references/feature-lineage.example.json`: lineage map template.
- `references/evaluation-report.example.json`: evaluation metrics report template.
- `references/medical-disease-leakage.md`: medical phenotype leakage patterns and controls.
- `references/leakage-taxonomy.md`: leakage classes, red flags, and mitigations.
- `references/top-tier-rigor-checklist.md`: submission-grade hard gates.
- `references/report-template.md`: reporting template for methods/results/robustness.
