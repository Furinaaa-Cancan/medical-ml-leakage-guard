# Development Notes & Error Tracking

## Common Errors Encountered & Fixes

### 1. SUPPORTED_MODEL_FAMILIES whitelist not updated
**Error**: `Unsupported model family in model-pool: knn`
**Root cause**: Added new model families to `_family_grid()` and `_build_estimator_for_family()` but forgot to add them to `SUPPORTED_MODEL_FAMILIES` set in `train_select_evaluate.py` (line ~107).
**Fix**: Always update ALL 5 places when adding a new model family:
1. `SUPPORTED_MODEL_FAMILIES` set
2. `_family_grid()` — hyperparameter grid
3. `_build_estimator_for_family()` — Pipeline builder
4. `_family_base_complexity()` — complexity ranking
5. `_family_friendly_name()` — display name
Plus in `mlgg_pixel.py`:
6. `MODEL_POOL` list
7. `BASE_FAMILY_GRID_SIZES` dict
8. i18n `_T` strings
9. `MODEL_PROFILE_PRESETS` (balanced/comprehensive)

### 2. NaN assignment to integer numpy array
**Error**: `ValueError: cannot convert float NaN to integer`
**Root cause**: Tried to assign `np.nan` directly to an integer numpy array.
**Fix**: Use `DataFrame.loc[mask, col] = np.nan` instead of `arr[mask] = np.nan`, or convert array to float first.

### 3. GateSpec attribute name mismatch
**Error**: `AttributeError: 'GateSpec' object has no attribute 'output_report'`
**Root cause**: The field is `report_output` in `_gate_registry.py`, not `output_report`.
**Fix**: Always check the actual dataclass field name in `_gate_registry.py:GateSpec`.

### 4. Thyroid dataset extreme class imbalance
**Error**: 97.7% positive rate made binary classification meaningless.
**Root cause**: Used "any abnormal" (class 2 or 3) as positive, but class 3 (hypothyroid) is the majority.
**Fix**: Use hyperthyroid (class 2, ~5.1%) as positive class — clinically relevant and properly imbalanced.

### 5. Unused import cleanup
**Issue**: `Tuple` imported but not used in `gate_coverage_matrix.py`.
**Fix**: Remove unused imports. Always check after refactoring.

## Checklist for Adding New Datasets

1. Add URL to `URLS` dict in `download_real_data.py`
2. Create `prepare_<name>()` function with proper column handling
3. Always call `add_patient_id_and_time(df, seed=N)` with unique seed
4. Always output columns in order: `patient_id, event_time, y, features...`
5. Verify: `y` is 0/1, `patient_id` is unique, positive rate is 5-50%
6. Add to `PREPARE` dict and CLI `choices`
7. Add to `mlgg_pixel.py`: i18n strings + `PLAY_DOWNLOAD_DATASETS`
8. Test download: `python3 examples/download_real_data.py <name> --output /tmp/test.csv`
9. Verify CSV: `pd.read_csv()` loads correctly with expected columns

## Checklist for Adding New Gate Tests

1. Import module directly: `import <gate_module> as <alias>`
2. Use `monkeypatch.setattr("sys.argv", [...])` to set CLI args
3. Call `<alias>.main()` directly
4. Assert return code (0=pass, 2=fail)
5. Read and parse the output report JSON
6. Assert failure codes in `out["failures"]`

### 6. prediction_replay_gate error code naming
**Error**: Test expected `prediction_trace_score_out_of_range` but actual code is `prediction_score_out_of_range`.
**Root cause**: Gate error codes don't always have `trace_` prefix consistently.
**Fix**: Always check the actual error code in the gate source before writing test assertions.

## Key File Locations

- Model families: `scripts/train_select_evaluate.py` (SUPPORTED_MODEL_FAMILIES, _family_grid, _build_estimator_for_family)
- Play UI: `scripts/mlgg_pixel.py` (MODEL_POOL, PLAY_DOWNLOAD_DATASETS, _T i18n dict)
- Gate registry: `scripts/_gate_registry.py` (GateSpec, GATE_REGISTRY)
- Dataset download: `examples/download_real_data.py`
- Tests: `tests/test_*.py`
