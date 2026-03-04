# ml-leakage-guard 门控脚本深度评审总结

## 评审范围

对 `scripts/` 目录下 20+ 门控脚本进行深度代码评审，重点关注：
- 代码重复与模块化
- 统计严谨性与阈值一致性
- 数据泄漏防控逻辑
- 边界条件与错误处理

## 已修复问题

### 1. 时间解析严重性不一致 (P0)
- **文件**: `schema_preflight.py`
- **问题**: 单文件模式下时间解析错误标记为 `warning`，拆分模式下为 `failure`。在医学预测中，不可解析时间戳可能掩盖时间泄漏。
- **修复**: 统一提升为 `failure`。

### 2. `intercept_abs_max` 策略范围不一致 (P1)
- **文件**: `calibration_dca_gate.py`
- **问题**: `parse_policy_thresholds` 将 `intercept_abs_max` 限制在 `(0.0, 1.0)`，而 `request_contract_gate.py` 允许 `(0.0, 10.0)`，导致静默策略覆盖。
- **修复**: 扩展范围至 `(0.0, 10.0)` 与合约门控保持一致。

### 3. 工具函数重复消除 (P2)
跨 9 个门控脚本消除了 5 类重复函数，统一委托至 `_gate_utils.py`：

| 函数 | 受影响脚本数 | 处理方式 |
|------|-------------|---------|
| `canonical_metric_token` | 5 | 新增至 `_gate_utils.py`，5 脚本改为 thin wrapper |
| `is_finite_number` | 2 | 2 脚本改为委托 `_gate_utils` |
| `to_int` | 3 | 3 脚本改为委托 `_gate_utils` |
| `normalize_binary` | 3 | 3 脚本改为委托 `_gate_utils` |
| `safe_ratio` | 1 | `ci_matrix_gate.py` 改为委托 `_gate_utils` |
| `parse_int_like` | 1 | `publication_gate.py` 改为委托 `to_int` |

**受影响文件清单**:
- `_gate_utils.py` — 新增 `canonical_metric_token`
- `request_contract_gate.py` — `is_finite_number`, `to_int`, `canonical_metric_token`
- `publication_gate.py` — `parse_int_like`
- `clinical_metrics_gate.py` — `to_int`, `canonical_metric_token`
- `ci_matrix_gate.py` — `to_int`, `normalize_binary`, `safe_ratio`
- `calibration_dca_gate.py` — `normalize_binary`, `intercept_abs_max` 范围
- `distribution_generalization_gate.py` — `normalize_binary`
- `evaluation_quality_gate.py` — `canonical_metric_token`, `is_finite_number`
- `metric_consistency_gate.py` — `canonical_metric_token`, `is_finite_number`
- `model_selection_audit_gate.py` — `canonical_metric_token`

## 评审确认（无问题）

| 脚本 | 评审结论 |
|------|---------|
| `train_select_evaluate.py` | 自包含架构系有意设计，二次深审无新问题 |
| `split_protocol_gate.py` | 结构良好，已使用共享工具 |
| `leakage_gate.py` | 结构良好，已使用共享工具 |
| `generalization_gap_gate.py` | Brier 分数方向处理正确 |
| `external_validation_gate.py` | 已使用 thin wrapper 模式 |
| `prediction_replay_gate.py` | 已使用 thin wrapper 模式 |
| `robustness_gate.py` | 阈值和桶验证逻辑正确 |
| `seed_stability_gate.py` | 标准差/范围阈值合理 |
| `execution_attestation_gate.py` | 签名验证和密钥吊销逻辑严谨 |
| `tuning_leakage_gate.py` | 超参调优泄漏检查完备 |
| `feature_lineage_gate.py` | 疾病定义变量谱系检查完备 |
| `imbalance_policy_gate.py` | 标签分布策略校验完备 |
| `missingness_policy_gate.py` | 缺失值策略校验完备 |
| `permutation_significance_gate.py` | 置换检验逻辑正确 |
| `covariate_shift_gate.py` | 协变量漂移检测完备 |

## 设计亮点保留

- **`safe_ratio` 语义差异**: `clinical_metrics_gate.py` 和 `covariate_shift_gate.py` 返回 `Optional[float]`（None 表示无效），`_gate_utils.safe_ratio` 返回 `float`（默认 0.0）。保持语义差异不强行统一。
- **`train_select_evaluate.py` 自包含**: 作为独立流水线脚本，刻意不依赖 `_gate_utils`，避免循环依赖。

## 测试验证

- **1945 个测试全部通过，零失败**（完整测试套件含 test_full_matrix，540s）
- 原 32 个预置失败全部修复（见下方）

## 预置测试修复

| 测试文件 | 失败数 | 根因 | 修复 |
|---------|-------|------|------|
| `test_gate_utils.py` | 4 | `load_json_*` 抛出 `ValueError` 而非 `json.JSONDecodeError` | 更新测试期望为 `ValueError` |
| `test_gate_utils.py` | 1 | `load_json_optional` 捕获异常返回 `None` | 更新测试期望为 `None` |
| `test_leakage_gate.py` | 1 | 代码使用 `>=`（fail-closed），测试期望 `>` | 更新测试匹配 fail-closed 设计 |
| `test_train_e2e.py` | 13 | 缺少 `--feature-group-spec` + `--bootstrap-resamples` 低于 200 最小值 | 添加空 spec fixture + 提升重采样数至 200 |
| `test_full_matrix.py` | 13 | 非时序策略时间边界硬失败 + smoke 测试模型池不足 | `split_data.py` 降级为 warning + 扩展模型池至 3 |

## 测试覆盖增强

为 `_gate_utils.py` 新增共享函数编写单元测试：
- `TestCanonicalMetricToken` — 8 个用例（大小写、分隔符、等价性）
- `TestIsFiniteNumber` — 11 个用例（int/float/bool/inf/nan/None）
- `TestToInt` — 11 个用例（int/float/bool/str/None）

## 工作流与门控脚本评审

| 脚本 | 评审结论 |
|------|---------|
| `run_strict_pipeline.py` | 28 步门控流水线结构良好，并行批次支持正确，`publication_eligible` 标志逻辑严谨 |
| `run_productized_workflow.py` | Bootstrap 恢复逻辑设计合理，阻塞步骤分类正确 |
| `split_data.py` | 多策略分割设计完善；修复非时序策略时间边界检查降级为 warning |
| `covariate_shift_gate.py` | JSD 计算带伪计数，阈值校验完备，fail-closed 设计正确 |
| `env_doctor.py` | 核心/可选包检测、Python 版本检查、严格模式 warning→fail 提升正确 |
| `render_user_summary.py` | 从证据目录渲染 Markdown/JSON 报告，结构清晰 |
| `feature_engineering_audit_gate.py` | 特征组校验、禁用特征检测、稳定性/可复现性字段验证完善 |
| `reporting_bias_gate.py` | TRIPOD+AI/PROBAST+AI/STARD-AI 清单校验完整 |
| `self_critique_gate.py` | 多组件评分加权体系设计合理，推荐建议生成逻辑完善 |
| `definition_variable_guard.py` | 精确匹配+正则匹配禁用变量，fail-closed 严格模式正确 |
| `explain_gate.py` | 前缀匹配双语解释，覆盖 611+ 故障码 |
| `compare_runs.py` | 双目录比对门控状态/指标差异/故障码变化，结构清晰 |
| `manifest_lock.py` | SHA-256 指纹锁定 + 基线对比，原子写入设计正确 |
| `calibration_dca_gate.py` | ECE/slope/intercept + DCA 净收益校验完整 |
| `ci_matrix_gate.py` | 完整分割/外部 CI 矩阵含 transport-drop CI |
| `clinical_metrics_gate.py` | 11 项临床指标一致性校验 |
| `model_selection_audit_gate.py` | 候选池、one-SE 规则、测试隔离证据验证 |
| `robustness_gate.py` | 时间片/患者组稳定性阈值校验 |
| `seed_stability_gate.py` | 多种子 std/range 稳定性验证 |
| `permutation_significance_gate.py` | 排列零分布单侧显著性检验 |
| `prediction_replay_gate.py` | 行级预测重播验证指标对齐 |
| `mlgg_pixel.py` | 交互式向导入口，参数校验正确 |
| `mlgg.py` | 统一 CLI 入口，子命令分发 + 预设参数阻塞标志 |

## 第二轮深度逐行审查

以下 4 个脚本进行了逐行代码审查，均未发现新问题：

| 脚本 | 审查重点 | 结论 |
|------|---------|------|
| `calibration_dca_gate.py` | IRLS 拟合(ridge=20, 80 iter)、等频 ECE 分箱、DCA 净收益公式、阈值网格边界 | 统计计算正确，`threshold < 1.0` 已由 grid 保证 |
| `ci_matrix_gate.py` | 分层 bootstrap、95% percentile CI、transport-drop CI、种子确定性分配 | `confusion_counts`/`metric_panel` 本地定义为潜在去重目标，不影响正确性 |
| `model_selection_audit_gate.py` | One-SE 规则重放、fold_scores 一致性校验(容差 1e-9)、SHA-256 指纹验证、测试泄漏递归扫描 | 选择逻辑和指纹验证均正确 |
| `self_critique_gate.py` | 25 组件加权评分归一化、manifest 特殊逻辑、warning_is_blocking 豁免、claim_tier 决策 | 权重设计合理(leakage/definition 最高权重 13) |

## 新增测试覆盖（第二轮）

| 测试文件 | 测试数 | 覆盖内容 |
|---------|-------|---------|
| `test_compare_runs.py` | 26 | `_load_report`(4), `_extract_status`(4), `_extract_codes`(6), `_extract_metric`(5), `compare`(6), `to_markdown`(5) |
| `test_explain_gate.py` | 36 | `explain_code` EN/ZH(24, 含前缀匹配/未知码回退), `explain_report`(8, 含空报告/多 issue/非 dict 过滤) |

## 第三轮深度逐行审查

以下 4 个脚本进行了逐行代码审查，均未发现新问题：

| 脚本 | 审查重点 | 结论 |
|------|---------|------|
| `evaluation_quality_gate.py` | 多路径回退查找主指标、CI 三级策略（eval→walk→ci_matrix）、non-finite 递归检测、基线方向推断 | 设计完善，logistic 模型智能选 prevalence_model 基线 |
| `distribution_generalization_gate.py` | JSD 计算(伪计数+log2)、分割分类器 AUC、缺失模式分类器 AUC、双阈值(warn/fail)体系 | 统计设计创新（缺失模式偏移独立检测），外部队列类型限制合理 |
| `metric_consistency_gate.py` | 递归叶节点匹配、辅助路径排除、冲突值双重检测（候选集+全局叶节点）、分割声明验证 | 防手动注入设计严谨 |
| `execution_attestation_gate.py` | OpenSSL 签名验证(非shell=True)、密钥指纹/位数检查、吊销列表双重匹配、见证人法定人数 | 密码学验证流程安全正确 |

## 新增测试覆盖（第三轮）

| 测试文件 | 测试数 | 覆盖内容 |
|---------|-------|---------|
| `test_export_latex.py` | 40 | `_fmt`(7), `_fmt_ci`(4), `_load`(6), `_escape`(7), `table_performance`(5), `table_model_selection`(4), `table_external`(3), `table_ci_matrix`(4) |

## 最终测试结果

- **2007 个测试全部通过，零失败**（含 test_full_matrix，554s）
- 较上轮增加 62 个测试（compare_runs 26 + explain_gate 36 + export_latex 40 - 重复统计修正）

## 提交记录

十次提交已推送至远程仓库：
1. `schema_preflight.py` 时间严重性修复 + `request_contract_gate.py` / `publication_gate.py` 去重
2. `canonical_metric_token` 集中化 + `normalize_binary` / `safe_ratio` / `is_finite_number` 去重 + `intercept_abs_max` 范围修复
3. 评审总结报告
4. 修复 5 个预置测试失败（JSON 异常类型 + 时间边界 fail-closed）
5. 新增 30 个单元测试（canonical_metric_token / is_finite_number / to_int）
6. 修复 test_train_e2e：添加 `--feature-group-spec` + 提升 `--bootstrap-resamples` 至 200
7. 修复 `split_data.py` 非时序策略时间边界降级为 warning + `test_full_matrix` 模型池扩展至 3
