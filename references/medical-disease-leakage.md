# Medical Disease Leakage Guardrails

Use this reference for medical prediction tasks where target definitions can leak into predictors.

## 1. Core Principle
Do not use any variable that participates in disease definition to predict the same disease endpoint.

Examples:
- If sepsis label includes SOFA criteria, do not include SOFA score or its defining components as predictors in the same prediction window.
- If diabetes label uses HbA1c threshold, do not include the threshold-triggering measurement at or after label index for pre-index prediction.
- If label uses diagnosis codes, do not include those same codes (or direct derivatives) as predictors.

## 2. Common Medical Leakage Channels
1. Phenotype-definition leakage:
- Label created from ICD/procedure/lab/medication rules.
- Predictor set includes any of those rule variables.

2. Care-pathway leakage:
- Orders, interventions, or medication starts triggered by clinician suspicion shortly before confirmed diagnosis.
- These are near-outcome proxies and may invalidate early prediction claims.

3. Post-index chart abstraction leakage:
- Features extracted from notes or registry fields documented after index time.

4. Outcome coding latency leakage:
- Billing/diagnosis codes backfilled after encounter but merged as if available earlier.

## 3. Required Data Design for Medical Studies
- Define index time (`t_index`) per prediction unit.
- Define label window and adjudication process.
- Record availability timestamp for each feature family.
- Enforce as-of joins for all longitudinal tables.
- Keep patient-level disjointness across splits unless protocol explicitly differs.

## 4. Disease Definition Spec Requirements
Maintain a JSON spec with:
- `targets.<target>.defining_variables`: exact variable names.
- `targets.<target>.forbidden_patterns`: regex for aliases/proxies.
- `targets.<target>.notes`: optional rationale text.
- Optional `global_forbidden_variables` and `global_forbidden_patterns`.

Also maintain a feature-lineage spec:
- `features.<feature>.ancestors`: raw variables contributing to each derived feature.
- Include transitive ancestry where derived features depend on intermediate derived features.
- Require near-complete lineage coverage for publication-grade claims.

For publication-grade metric integrity:
- Pin one canonical metric location with `evaluation_metric_path`.
- Reject reports with conflicting duplicated metric values or non-finite metrics (`NaN`, `Inf`).

For publication-grade model development integrity:
- Freeze split protocol and keep patient-disjoint temporal boundaries.
- Apply imbalance mitigation (class weights/resampling) only on train folds.
- Fit imputers only on train folds; for large-scale data prefer simple train-fitted imputation with missing indicators unless strongly justified.
- Avoid direct MICE use for very large/high-dimensional tables unless scale limits and computational stability are documented.
- Keep hyperparameter tuning, thresholding, and calibration fully isolated from final test labels.

Minimal example:

```json
{
  "global_forbidden_patterns": ["(?i)target", "(?i)label"],
  "targets": {
    "sepsis": {
      "defining_variables": [
        "sepsis_icd10_code",
        "sofa_total",
        "suspected_infection_flag"
      ],
      "forbidden_patterns": [
        "(?i)sepsis",
        "(?i)sofa",
        "(?i)suspected[_ ]?infection"
      ],
      "notes": "Do not predict sepsis with variables used in sepsis phenotype construction."
    }
  }
}
```

## 5. Adjudication Rule
If a feature is ambiguous:
1. Mark as forbidden until lineage proves pre-index availability and non-definition status.
2. Require manual review with clinical rationale.
3. Document decision in audit log.

## 6. Claim Tiers
- Preliminary: leakage checks incomplete.
- Leakage-audited: deterministic leakage gates pass.
- Publication-grade: deterministic gates + phenotype-definition guard + falsification + reproducibility gates pass.
