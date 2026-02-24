# Top-Tier Rigor Checklist

Use this checklist as a hard gate before claiming publication-grade predictive performance.

## A. Problem Definition
- [ ] Define prediction unit, `t_index`, and target horizon.
- [ ] Define target construction without post-outcome ambiguity.
- [ ] Define intended deployment setting and latency constraints.

## B. Data and Splits
- [ ] Freeze split protocol before model development.
- [ ] Keep machine-readable split protocol spec and gate report.
- [ ] Use out-of-time test for temporal settings.
- [ ] Enforce patient-level disjointness for repeated entities.
- [ ] Document sample counts and event rates per split.
- [ ] Audit row overlap and entity overlap across splits.

## C. Feature Legality
- [ ] Confirm each feature is available at or before `t_index`.
- [ ] Block post-outcome variables and future-derived aggregates.
- [ ] Use as-of joins for external/relational tables.
- [ ] Version feature definitions and generation code.

## D. Pipeline Isolation
- [ ] Fit preprocessors on training folds only.
- [ ] Keep preprocessing + model in one fold-aware pipeline.
- [ ] Run feature selection inside CV loops only.
- [ ] Apply any resampling/SMOTE only on train folds (never valid/test).
- [ ] Fit imputers only on train folds and apply transform forward to valid/test.
- [ ] Do not use target/outcome values in feature imputation.
- [ ] Avoid manual transformations that touch validation/test outcomes.

## E. Model Selection and Tuning
- [ ] Tune hyperparameters using inner CV or validation only.
- [ ] Select architecture without peeking at final test labels.
- [ ] Select threshold/calibration without final test outcomes.
- [ ] Keep tuning protocol spec and prove all `test_used_*` flags are false.
- [ ] Keep final test unopened until design freeze.

## F. Evaluation Quality
- [ ] Report discrimination metrics (for example AUC/PR-AUC where relevant).
- [ ] Report calibration metrics/plots when probabilities are used.
- [ ] Report confidence intervals (bootstrap or repeated resampling).
- [ ] Report baseline comparisons (naive, linear/simple models).
- [ ] Report class imbalance handling and prevalence context.
- [ ] Extract primary metric from evaluation report artifact (no manual metric injection).
- [ ] Pin explicit metric path and confirm no conflicting duplicate metric values in the artifact.
- [ ] Ensure metric-path leaf is consistent with declared primary metric name.
- [ ] Reject non-finite metric values (`NaN`, `Inf`) in all evidence files.

## G. Robustness and Falsification
- [ ] Run label permutation test (expect chance-level metrics).
- [ ] Run time-slice robustness analysis.
- [ ] Run group holdout robustness analysis.
- [ ] Run seed sensitivity analysis across multiple seeds.
- [ ] Run ablation for suspect/high-impact features.

## H. Reproducibility
- [ ] Fix and log random seeds.
- [ ] Log software versions and hardware constraints.
- [ ] Save data snapshot/version identifiers.
- [ ] Save full training/evaluation config files.
- [ ] Ensure end-to-end rerun reproduces results within tolerance.

## I. Reporting and Transparency
- [ ] Document exclusion criteria and missing-data handling.
- [ ] Document all explored models or state search policy.
- [ ] Document all tested subgroup analyses or correction strategy.
- [ ] Include failure modes and known limitations.
- [ ] Separate predictive claims from causal claims.

## J. Medical Phenotype Integrity
- [ ] Freeze disease definition spec before model fitting.
- [ ] List all disease-defining variables and derived proxies.
- [ ] Prove disease-defining variables are excluded from predictors for the same endpoint.
- [ ] Provide feature-lineage mapping and verify no forbidden ancestors in derived features.
- [ ] Audit diagnosis/lab/medication coding timestamps against `t_index`.
- [ ] Document ambiguous variables and adjudication rationale.

## Decision Rule
Apply fail-closed policy:
1. If any item in sections B-E or J fails, block performance claims.
2. If sections F-I have unresolved gaps, downgrade claim from "publication-grade" to "preliminary".
