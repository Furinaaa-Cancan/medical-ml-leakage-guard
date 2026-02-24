# Leakage Taxonomy

Use this file when identifying and classifying leakage risk in prediction pipelines.

## 1. Split Contamination
Definition:
Reuse identical or near-identical samples across train/validation/test.

Red flags:
- Same row appears in multiple splits.
- Near-duplicate records differ only in non-causal metadata.

Mitigation:
- Deduplicate before splitting.
- Lock split assignment by immutable sample ID.
- Audit overlap at row-hash level and entity-ID level.

## 2. Group Leakage
Definition:
Allow the same entity (patient/user/account/device) to appear across splits.

Red flags:
- High score but poor performance on new entities.
- Cross-validation folds are not group-aware.

Mitigation:
- Use group-aware split (`GroupKFold`, grouped holdout).
- Enforce entity disjointness between train/validation/test.

## 3. Temporal Look-Ahead
Definition:
Use information that becomes available only after prediction time.

Red flags:
- Features use future windows, forward fill from future rows, or post-event logs.
- Validation/test periods overlap training horizon in a way that breaks deployment realism.

Mitigation:
- Define `t_pred` and `t_target` explicitly.
- Keep only features observed at or before `t_pred`.
- Use out-of-time holdout and forward-chaining CV.

## 4. Target Proxy Leakage
Definition:
Include features that directly encode target or outcome-adjacent variables.

Red flags:
- Feature names include patterns like `target`, `label`, `outcome`, `post`, `future`, `next`.
- Abruptly extreme model performance with weak domain plausibility.
- For medical tasks, predictors include diagnosis/lab/medication variables used to define the same disease label.

Mitigation:
- Maintain an allowlist of admissible features.
- Run feature lineage review for each high-importance feature.
- Check transitive lineage, not only direct ancestors.
- Force ablation of suspect columns and compare metric drop.
- Enforce `scripts/definition_variable_guard.py` against phenotype definition spec.

## 5. Preprocessing Leakage
Definition:
Fit preprocessing transforms on full data before split/fold boundaries.

Red flags:
- Scaler/imputer/encoder/PCA fit once on full dataset.
- Feature selection executed before CV loop.
- Resampling/SMOTE applied on validation/test or before split.

Mitigation:
- Fit transforms only on training partition per fold.
- Wrap preprocessing and model into a single pipeline object.

## 5b. Missingness/Imputation Leakage
Definition:
Use validation/test/target information while fitting imputation logic.

Red flags:
- Imputer fit on full dataset before split.
- Imputation model consumes target or outcome-adjacent variables.
- Heavy iterative imputers (for example MICE) used on very large/high-dimensional data without scale controls.

Mitigation:
- Fit imputers on train folds only and forward-apply to valid/test.
- Exclude target/outcome from imputation features.
- For very large tables, prefer scalable train-fitted simple imputers plus missing indicators.

## 6. Hyperparameter and Model Selection Leakage
Definition:
Use test set repeatedly during tuning or architecture selection.

Red flags:
- Test metrics reported many times during iteration.
- Checkpoint selection references test results.

Mitigation:
- Use nested CV or dedicated validation for tuning.
- Open final test once after design freeze.

## 7. Threshold and Calibration Leakage
Definition:
Tune threshold/calibration using final test outcomes.

Red flags:
- Decision threshold chosen to maximize test F1/AUC.
- Calibration map fit on test data.

Mitigation:
- Tune threshold/calibration on validation or inner folds only.
- Reserve test strictly for final estimation.

## 8. Post-Hoc Subgroup Fishing
Definition:
Search many subgroups on test data and report only favorable ones.

Red flags:
- Many subgroup slices explored, few reported.
- No correction for multiple comparisons.

Mitigation:
- Pre-specify subgroup hypotheses.
- Report all tested subgroup analyses or apply multiplicity control.

## 9. Data Merge Leakage
Definition:
Join external tables with keys or timestamps that inject future knowledge.

Red flags:
- Feature table built with latest snapshot rather than as-of snapshot.
- Slowly changing dimensions joined without temporal validity intervals.

Mitigation:
- Use as-of joins with validity windows.
- Keep data versioning and snapshot timestamps in metadata.

## 10. Leakage Diagnosis Checklist
Run all checks:
1. Verify row-level overlap across splits.
2. Verify entity-level overlap across splits.
3. Verify temporal order (`train_max_time < valid_min_time < test_min_time` when applicable).
4. Verify every feature timestamp is `<= t_pred`.
5. Verify preprocessing is fit only on training partitions.
6. Verify tuning and thresholding avoid final test labels.
7. Verify negative controls collapse to chance-level performance.
8. Verify reported primary metric is extracted from a pinned path and is finite.
