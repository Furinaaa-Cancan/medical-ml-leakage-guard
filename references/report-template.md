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
- Preprocessing pipeline:
- Hyperparameter optimization procedure:
- Calibration and threshold selection protocol:
- Class-imbalance strategy and train-only scope proof (`imbalance_policy_report.json`):
- Missingness policy and large-data method suitability proof (`missingness_policy_report.json`):
- Tuning leakage isolation proof (`tuning_leakage_report.json`):

## 7. Evaluation Plan
- Primary metrics:
- Secondary metrics:
- Confidence interval procedure:
- Baselines:
- Evaluation report artifact and explicit metric extraction path (`evaluation_metric_path`):
- Metric-source consistency check (no conflicting duplicate metric values):

## 8. Robustness and Falsification
- Label permutation test results:
- Time-slice robustness:
- Group-holdout robustness:
- Seed sensitivity:
- Suspect-feature ablation:

## 9. Main Results
- Validation performance:
- Final test performance:
- Calibration results:
- Error analysis:

## 10. Reproducibility Artifacts
- Code version/commit:
- Data snapshot IDs:
- Environment details:
- Config files and random seeds:
- Re-run reproducibility check:
- Evidence manifest (`manifest.json`):
- Split protocol gate artifact (`split_protocol_report.json`):
- Imbalance policy gate artifact (`imbalance_policy_report.json`):
- Missingness policy gate artifact (`missingness_policy_report.json`):
- Tuning leakage gate artifact (`tuning_leakage_report.json`):
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
