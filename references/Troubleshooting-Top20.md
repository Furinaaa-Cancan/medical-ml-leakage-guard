# Troubleshooting Top20 / 高频报错 Top20

Use this list with strict pipeline failure codes.

使用方式：先在 `evidence/strict_pipeline_report.json` 找 `failures[].code`，再按下表执行。

---

## Quick Flow / 快速流程

1. Find failure code in strict pipeline report.
2. Run the diagnose command.
3. Apply one-step fix.
4. Run verify command.

1. 在 strict report 里找 failure code。  
2. 跑诊断命令。  
3. 执行一步修复。  
4. 跑复验命令。  

---

## Top20 Mapping

| Code | Diagnose | One-step Fix | Verify |
|---|---|---|---|
| `onboarding_step_cancelled` | Check `<project>/evidence/onboarding_report.json` for the first cancelled step (`stderr_tail=step_cancelled_by_user`). / 查看 onboarding 报告中首个取消步骤 | Re-run with confirmation bypass: `python3 scripts/mlgg.py onboarding --project-root <project> --mode guided --yes` (or collect full diagnostics with `--no-stop-on-fail`). / 使用 `--yes` 复跑（或加 `--no-stop-on-fail` 收集全量诊断） | `python3 scripts/mlgg.py onboarding --project-root <project> --mode auto --no-stop-on-fail` |
| `onboarding_interactive_input_unavailable` | Check onboarding step stderr for `guided_mode_requires_interactive_stdin` or `guided_mode_stdin_eof`. / 检查 onboarding 步骤 stderr 是否是交互输入不可用 | Run guided with auto-confirm (`--yes`) or use auto mode. / 使用 `--yes` 跳过交互确认，或改用 auto 模式 | `python3 scripts/mlgg.py onboarding --project-root <project> --mode auto --no-stop-on-fail` |
| `authority_preset_route_override_forbidden` | Run the failing wrapper command and inspect stderr. / 复现报错并查看 stderr | Do not pass route flags (`--stress-case-id/--stress-seed-search`) to fixed wrappers; use plain wrapper command. / 固定封装命令不要再传路线参数，直接使用封装命令 | `python3 scripts/mlgg.py authority-release --dry-run` |
| `benchmark_registry_missing` | `python3 scripts/mlgg.py benchmark-suite --profile release --registry-file <path>` | Restore or re-point to valid registry file (`references/benchmark-registry.json`). / 修复或回指到有效 registry 文件 | `python3 scripts/mlgg.py benchmark-suite --profile release --registry-file references/benchmark-registry.json --repeat 1` |
| `benchmark_registry_mismatch` | `python3 scripts/mlgg.py benchmark-suite --profile release --registry-file references/benchmark-registry.json --repeat 1` | Re-sync registry fingerprints with frozen benchmark data; do not bypass hash checks. / 同步 registry 指纹与冻结数据，禁止绕过 hash 校验 | `python3 scripts/mlgg.py benchmark-suite --profile release --registry-file references/benchmark-registry.json --repeat 1` |
| `benchmark_repeat_inconsistent` | Re-run single repeat to inspect unstable suite: `python3 scripts/mlgg.py benchmark-suite --profile release --repeat 1` | Stabilize split/seed/calibration route until repeat conclusions match. / 修复不稳定因素，直到重复运行结论一致 | `python3 scripts/mlgg.py benchmark-suite --profile release --repeat 3` |
| `benchmark_blocking_suite_failed` | Open `release_benchmark_matrix_summary.json` and inspect `blocking_failures/failure_codes`. / 检查矩阵报告中的阻断失败列表 | Execute one-step verify command printed by benchmark-suite, then fix upstream gate failures. / 先执行工具打印的一步复验命令，再修复上游 gate 失败 | `python3 scripts/mlgg.py benchmark-suite --profile release --repeat 3` |
| `missing_required_path` | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` | Ensure all `*_spec`/`*_report_file` paths exist in request. / 补齐 request 中全部路径 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare` |
| `path_not_found` | same as above | Re-run training to regenerate missing evidence, or fix request paths. / 重跑训练或修正路径 | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` |
| `invalid_evaluation_report` | `python3 scripts/mlgg.py train -- --help` | Re-run `train_select_evaluate.py` and ensure `evaluation_report.json` is complete. / 重新产出完整评估报告 | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` |
| `invalid_model_selection_report` | `python3 scripts/mlgg.py train -- --help` | Ensure `--model-selection-report-out` path is writable, then retrain. / 修正输出路径后重训 | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` |
| `invalid_external_validation_report` | `python3 scripts/mlgg.py train -- --help` | Check `external_cohort_spec` and regenerate external validation report. / 校验外部队列配置并重产出报告 | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` |
| `external_validation_cross_period_not_met` | `python3 scripts/external_validation_gate.py --external-validation-report <project>/evidence/external_validation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --performance-policy <project>/configs/performance_policy.json --strict` | Add/fix cross-period external cohort until it passes size/event/metric thresholds. / 补齐并通过 cross_period 队列 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json` |
| `external_validation_cross_institution_not_met` | same gate command as above | Add/fix cross-institution cohort until thresholds pass. / 补齐并通过 cross_institution 队列 | same verify command as above |
| `calibration_ece_exceeds_threshold` | `python3 scripts/calibration_dca_gate.py --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict` | Improve calibration strategy and retrain. / 改进校准后重训 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json` |
| `decision_curve_net_benefit_insufficient` | same calibration gate command | Improve model/threshold so net benefit exceeds treat-all/treat-none baseline. / 提升决策曲线净获益 | same verify command |
| `distribution_shift_exceeds_threshold` | `python3 scripts/distribution_generalization_gate.py --train <project>/data/train.csv --valid <project>/data/valid.csv --test <project>/data/test.csv --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --feature-group-spec <project>/configs/feature_group_spec.json --target-col y --ignore-cols patient_id,event_time --performance-policy <project>/configs/performance_policy.json --distribution-report <project>/evidence/distribution_report.json --strict` | Reduce shift via split redesign/feature rework. / 通过重切分或特征重构降低漂移 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json` |
| `split_separability_exceeds_threshold` | same distribution gate command | Investigate holdout separability roots (time/site/missingness leakage). / 排查可分性来源并修复 | same verify command |
| `overfit_gap_exceeds_threshold` | `python3 scripts/generalization_gap_gate.py --evaluation-report <project>/evidence/evaluation_report.json --performance-policy <project>/configs/performance_policy.json --strict` | Increase regularization/simplify model pool/adjust feature selection. / 强化正则化并简化模型与特征 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json` |
| `clinical_floor_specificity_not_met` | `python3 scripts/clinical_metrics_gate.py --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict` | Re-select threshold/model to satisfy specificity floor while preserving sensitivity/NPV floors. / 调阈值与模型满足 specificity 下限 | same verify command |
| `clinical_floor_ppv_not_met` | same clinical gate command | Improve PPV by threshold/model/features while keeping clinical floors. / 提升 PPV 且不破坏其他临床下限 | same verify command |
| `prediction_metric_replay_mismatch` | `python3 scripts/prediction_replay_gate.py --evaluation-report <project>/evidence/evaluation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --performance-policy <project>/configs/performance_policy.json --strict` | Ensure evaluation and trace come from same run; regenerate both if needed. / 保证评估与 trace 同源并重产出 | same workflow verify command |
| `signature_verification_failed` | `python3 scripts/execution_attestation_gate.py --attestation-spec <project>/configs/execution_attestation.json --evaluation-report <project>/evidence/evaluation_report.json --study-id <study_id> --run-id <run_id> --strict` | Regenerate payload/signature/public-key artifacts without mutation. / 重签名并保持工件不被改写 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare` |
| `missing_execution_attestation_required_artifact` | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` | Ensure all mandatory names are listed in `execution_attestation.required_artifact_names`. / 补齐 required artifact 名单 | same request_contract verify |
| `manifest_comparison_missing` | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare` | Run bootstrap once to create baseline manifest. / 先跑 bootstrap 生成基线 | `python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json` |
| `performance_policy_downgrade_new_blocks` | `python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict` | Restore policy blocks/thresholds to publication baseline (no downgrade). / 恢复发布级阈值，不允许放宽 | same request_contract verify |
| `ci_width_exceeds_threshold` | `python3 scripts/ci_matrix_gate.py --evaluation-report <project>/evidence/evaluation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --ci-matrix-report <project>/evidence/ci_matrix_report.json --strict` | Increase effective sample size and model stability, then recompute CI. / 提高样本与稳定性后重算 CI | same workflow verify command |
| `feature_selection_data_leakage` | `python3 scripts/feature_engineering_audit_gate.py --feature-group-spec <project>/configs/feature_group_spec.json --feature-engineering-report <project>/evidence/feature_engineering_report.json --lineage-spec <project>/configs/feature_lineage.json --tuning-spec <project>/configs/tuning_protocol.json --strict` | Restrict feature selection to train/cv-inner-train scope only. / 强制特征筛选仅在训练域 | same workflow verify command |

---

## Notes / 说明

- All publication-grade gates are fail-closed; do not bypass by editing report status manually.
- For first run without baseline, use `--allow-missing-compare` only for bootstrap.

- 发布级 gate 都是 fail-closed，不能手工改 report status 绕过。
- 首跑无 baseline 时，仅用于 bootstrap 使用 `--allow-missing-compare`。
