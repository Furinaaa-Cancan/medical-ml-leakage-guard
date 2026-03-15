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
- [ ] Ensure train/valid/test split files are distinct physical datasets (no path reuse).
- [ ] Document sample counts and event rates per split.
- [ ] Audit row overlap and entity overlap across splits.
- [ ] Audit train-vs-holdout covariate shift (feature distribution drift + missingness drift).

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
- [ ] If using MICE-first strategy, enforce `mice_with_scale_guard` and audited fallback evidence on oversized data.

## E. Model Selection and Tuning
- [ ] Tune hyperparameters using inner CV or validation only.
- [ ] Select architecture without peeking at final test labels.
- [ ] Keep candidate model pool >= 3 and include interpretable logistic baseline.
- [ ] Declare model-pool policy explicitly (families, required baseline, max trials/family, search strategy, CPU parallelism).
- [ ] Keep model-selection report and audit one-SE + simplicity replay.
- [ ] Ensure strict publication-grade objective metric is PR-AUC.
- [ ] Keep family-specific regularization evidence (L1/L2/ElasticNet or tree/boosting complexity penalties) in model selection artifacts.
- [ ] Select threshold/calibration without final test outcomes.
- [ ] Restrict threshold/calibration split to validation or inner-CV scopes (never train/test).
- [ ] Record threshold feasibility with three explicit flags: selection split, guard split, and overall.
- [ ] If `selection_split=cv_inner`, require `constraints_satisfied_guard_split=true` (fail-closed).
- [ ] Keep tuning protocol spec and prove all `test_used_*` flags are false.
- [ ] Enforce anti-downgrade policy constraints: threshold/floor configs must not be weaker than publication baseline.
- [ ] Keep final test unopened until design freeze.

## F. Evaluation Quality
- [ ] Report discrimination metrics (for example AUC/PR-AUC where relevant).
- [ ] Report calibration metrics/plots when probabilities are used.
- [ ] Report confidence intervals (bootstrap or repeated resampling).
- [ ] Report baseline comparisons (naive, linear/simple models).
- [ ] Report class imbalance handling and prevalence context.
- [ ] Report full clinical metric panel: accuracy, precision/PPV, NPV, sensitivity, specificity, F1, F2-beta, ROC-AUC, PR-AUC, Brier.
- [ ] Provide split-level metric panel with confusion matrix for train/valid/test.
- [ ] Validate precision==PPV and metric formulas against confusion matrix.
- [ ] Keep row-level de-identified `prediction_trace` and replay PR-AUC/ROC-AUC/Brier + threshold metrics from raw `y_true/y_score`.
- [ ] Verify replayed metrics exactly match evaluation-report metrics within policy tolerance.
- [ ] Enforce train-valid, valid-test, and train-test gap thresholds with fail-closed policy.
- [ ] Extract primary metric from evaluation report artifact (no manual metric injection).
- [ ] Require evaluation report to declare `split=test` for final primary metric claims.
- [ ] Pin explicit metric path and confirm no conflicting duplicate metric values in the artifact.
- [ ] Ensure metric-path leaf is consistent with declared primary metric name.
- [ ] Reject non-finite metric values (`NaN`, `Inf`) in all evidence files.
- [ ] Include at least one publication-grade external cohort (`cross_period` or `cross_institution`) with transport-gap checks.
- [ ] Hard-gate calibration quality on internal test + all external cohorts (ECE/slope/intercept).
- [ ] Hard-gate decision-curve net benefit on internal test + all external cohorts.

## G. Robustness and Falsification
- [ ] Run label permutation test (expect chance-level metrics).
- [ ] Run split separability (adversarial validation/proxy) and justify if holdout is highly separable.
- [ ] Run time-slice robustness analysis.
- [ ] Run group holdout robustness analysis.
- [ ] Keep machine-checkable robustness artifact and pass `robustness_gate` thresholds.
- [ ] Run seed sensitivity analysis across multiple seeds.
- [ ] Keep machine-checkable seed sensitivity artifact and pass `seed_stability_gate` thresholds.
- [ ] Run ablation for suspect/high-impact features.

## H. Reproducibility
- [ ] Fix and log random seeds.
- [ ] Log software versions and hardware constraints.
- [ ] Save data snapshot/version identifiers.
- [ ] Save full training/evaluation config files.
- [ ] If interactive wizard was used, archive generated command, `run_tag`, and profile contract/version; verify interactive mode did not bypass strict gate evidence requirements.
- [ ] If onboarding flow was used, archive `onboarding_report.json` and confirm the recorded 8-step sequence matches executed evidence artifacts.
- [ ] Require signed execution attestation (detached signature + public-key verification).
- [ ] Ensure signed payload covers training command, start/finish timestamps, and critical artifact hashes.
- [ ] Enforce key-rotation/expiry policy and check revocation list for signing key ID and fingerprint.
- [ ] Verify trusted timestamp record binds to signed payload hash and run identity.
- [ ] Verify transparency-log record binds to signed payload hash and run identity.
- [ ] Verify signed execution-log attestation binds to the `training_log` artifact hash and run identity.
- [ ] Verify execution-log attestation also binds `prediction_trace` and `external_validation_report` hashes.
- [ ] Verify witness-quorum records bind to signed payload hash and run identity.
- [ ] Enforce minimum validated witness count with independent witness authorities/keys.
- [ ] Require independent key custody for payload signer, execution receipt, and execution-log authorities.
- [ ] Require witness keys to be independent from the payload signing key.
- [ ] Enforce cross-role authority/key distinctness across timestamp/transparency/execution-receipt/execution-log/witness roles.
- [ ] Compare current manifest against a locked baseline manifest and require exact match before publication-grade claim.
- [ ] Ensure end-to-end rerun reproduces results within tolerance.
- [ ] For stress-mode datasets, keep `stress_seed_search_report.json` and `stress_seed_selection.json` as reproducibility evidence.
- [ ] For stress-mode cache reuse, require contract match on `contract_version`, `policy_sha256`, `search_profile_set`, `dataset_fingerprint`, and selected seed/profile range.

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

## K. Fairness and Equity (TRIPOD+AI 2024 / PROBAST+AI 2025)
- [ ] Report subgroup performance across key demographic/clinical variables.
- [ ] Compute equalized odds gap across subgroups (threshold: <0.15).
- [ ] Compute disparate impact ratio (four-fifths rule: >0.80).
- [ ] Document minimum subgroup size and flag underpowered subgroups (<20).
- [ ] If disparity detected, document mitigation strategy and results.
- [ ] Report fairness assessment in TRIPOD+AI checklist.

## L. Sample Size Adequacy (Riley et al. 2019/2025)
- [ ] Compute events per variable (EPV) and justify EPV ≥ 10 (minimum) or ≥ 20 (recommended).
- [ ] Estimate shrinkage factor (target ≥ 0.90) or cite pmsampsize calculation.
- [ ] Ensure minimum 100 events and 100 non-events for model development.
- [ ] Ensure test set has ≥ 50 events for reliable performance estimation.
- [ ] For ML models, note that higher EPV may be needed vs regression (Tsegaye et al. 2025).

## M. Model Comparison and Improvement Metrics
- [ ] Report Net Reclassification Improvement (NRI) vs clinical baseline.
- [ ] Report Integrated Discrimination Improvement (IDI) vs baseline.
- [ ] Report DeLong test for AUC comparison.
- [ ] Report McNemar test for classification comparison.
- [ ] Include both prevalence baseline and logistic regression baseline comparisons.

## Decision Rule
Apply fail-closed policy:
1. If any item in sections B-E or J fails, block performance claims.
2. If sections F-I or K-M have unresolved gaps, downgrade claim from "publication-grade" to "preliminary".
