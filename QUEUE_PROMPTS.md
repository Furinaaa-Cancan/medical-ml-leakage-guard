# ML Leakage Guard — 批量任务提示词清单（严格评审版）

> 共 100 条，每条都包含完整的「实现 → 测试 → 严格评审 → 再检查 → 提交」闭环流程。
> 可直接逐条放入 queue 自动执行。

---

## A. 单元测试补全（1-15）

### 1
为 `scripts/_gate_utils.py` 编写完整单元测试。

**第一步：通读源码。** 用 code_search 或 read_file 完整阅读 `scripts/_gate_utils.py`，理解每个导出函数（`add_issue`、`load_json`、`write_json`、`to_float` 等）的签名、参数类型、返回值、异常路径。记录所有分支和边界条件。

**第二步：设计测试矩阵。** 列出每个函数的所有路径，制定 test case 清单。最低要求：
- `add_issue`: 正常添加 / 重复 key / 空 dict / issue 为 None
- `load_json`: 正常文件 / 不存在的路径 / 格式错误 JSON / 空文件 / 编码问题
- `write_json`: 正常写入 / 目录不存在 / 只读路径
- `to_float`: 正常 int/float / 字符串数字 / "inf" / "nan" / None / 空字符串 / bool / list
每个函数至少 5 个 test case，总计至少 20 个。

**第三步：编写 `tests/test_gate_utils.py`。** 使用 pytest 风格，每个 test 函数名清晰表达意图。用 `tmp_path` fixture 做文件操作。不要 mock 标准库，用真实文件。

**第四步：运行测试。** 执行 `python3 -m pytest tests/test_gate_utils.py -v`，确保全部通过。

**第五步：严格评审。** 重新完整阅读测试文件和源码，逐行检查：
- 是否有遗漏的分支路径未测试？
- 是否有边界条件（空输入、超大输入、特殊字符路径）未覆盖？
- assert 消息是否足够诊断失败原因？
- 是否有测试之间的隐式依赖（共享状态）？
记录发现的遗漏，补充测试。

**第六步：再次运行测试，确认全部通过后提交。** `git add tests/test_gate_utils.py && git commit -m "test: comprehensive unit tests for _gate_utils.py (N cases)" && git push`

---

### 2
为 `scripts/split_data.py` 编写完整单元测试。

**第一步：通读源码。** 完整阅读 `scripts/split_data.py`（每一行都要看），理解：
- argparse 参数定义（input, output-dir, patient-id-col, target-col, time-col, strategy, train-ratio, valid-ratio, test-ratio）
- 三种 split 策略的具体算法（grouped_temporal, grouped_random, stratified_grouped）
- 安全检查：NaN patient_id 排除、NaN target 排除、min 10 pos/neg per split、min 5 patients per split、prevalence shift warning、row count preservation、SHA256 fingerprint
- 输出文件格式（train.csv, valid.csv, test.csv）
- 边界处理：只有 1 个患者、全部 positive、全部 negative、刚好在阈值边界

**第二步：创建 mock 数据生成器。** 在测试文件顶部写一个 `_make_csv(tmp_path, n_patients, n_rows_per_patient, pos_rate, ...)` helper，可以生成各种特征的测试 CSV。

**第三步：设计测试矩阵（至少 15 个 case）：**
- 3 种策略 × 正常数据 = 3 cases
- grouped_temporal + 无 time_col = 1 case（应该报错或 fallback）
- 全 positive target = 1 case（应该失败或 warning）
- NaN patient_id 行 = 1 case（应该被排除）
- NaN target 行 = 1 case（应该被排除）
- 只有 2 个患者 = 1 case（应该失败 min_patients 检查）
- 行数保存验证：split 后行数之和 = 原始行数（减 NaN 排除）= 1 case
- 患者不交叉验证 = 1 case
- SHA256 fingerprint 存在验证 = 1 case
- 比例验证（60/20/20 近似） = 1 case
- 重复执行幂等性 = 1 case

**第四步：编写 `tests/test_split_data.py`。** 用 subprocess 调用 split_data.py（与实际使用一致），或直接 import 内部函数。

**第五步：运行测试，确保全部通过。**

**第六步：严格评审。** 重新阅读 split_data.py 的每个分支，检查是否有未覆盖的路径。特别关注：
- `--strategy grouped_temporal` 但 `--time-col` 未指定时的行为
- 所有 sys.exit() 路径是否都被测试到
- 输出 CSV 的列名是否与输入完全一致

**第七步：补充遗漏测试，再次运行，全部通过后提交。**

---

### 3
为 `mlgg_pixel.py` 中的 `detect_columns` 和 `_hint_match` 编写独立单元测试。

**第一步：阅读 `_hint_match` 和 `detect_columns` 源码。** 理解匹配逻辑：单词 hint（无 `_`）用 `split("_")` 词级匹配；多词 hint（含 `_`）用子串匹配。理解 `_PID_HINTS`、`_TGT_HINTS`、`_TIME_HINTS` 的完整列表。

**第二步：设计测试矩阵（至少 50 个 assert）：**

对 `_hint_match`：
- 每个 _PID_HINTS 条目至少 2 个测试（正确匹配 + 常见误匹配）
- 每个 _TGT_HINTS 条目至少 2 个测试
- 每个 _TIME_HINTS 条目至少 2 个测试
- 特殊边界：空字符串 hint、空字符串 col、只有下划线的 col

对 `detect_columns`：
- 典型医学数据集列名组合（patient_id, target, event_time, age, gender, ...）
- 无任何匹配的列名组合
- 只匹配 pid 不匹配 target 的情况
- 列名含空格（应被替换为下划线后匹配）
- 混合大小写列名

**第三步：编写 `tests/test_detect_columns.py`。**

**第四步：运行测试。**

**第五步：严格评审。** 重新对照 `_PID_HINTS` / `_TGT_HINTS` / `_TIME_HINTS` 列表，逐条验证是否每个 hint 都有正确匹配和误匹配测试。检查是否遗漏了 "status" in _TGT_HINTS 可能匹配 "patient_status" 这类跨类别冲突。

**第六步：补充后提交。**

---

### 4
为 `scripts/schema_preflight.py` 编写完整单元测试。

**第一步：完整阅读 `scripts/schema_preflight.py` 源码。** 理解两种模式（单文件 pre-split 检查 vs split 后三文件检查）、所有验证项（列存在性、类型检查、target 值域、patient_id NaN、time 格式、语义列自动映射）、报告 JSON 格式和所有可能的 issue codes。

**第二步：设计测试矩阵（至少 10 个 case）：**
- 正常 CSV → pass
- target 列缺失 → fail + 正确 issue code
- target 值不是 0/1（含 2, -1, 字符串）→ fail
- patient_id 全部为 NaN → fail
- patient_id 部分为 NaN → warning
- event_time 格式不一致（混合 YYYY-MM-DD 和 timestamp）→ warning
- 空 CSV（只有 header）→ fail
- 3 个 split 文件列名不一致 → fail
- 超大列数（100+ 列）→ pass 但性能检查

**第三步：编写 `tests/test_schema_preflight.py`，用 subprocess 调用或 import。**

**第四步：运行测试。**

**第五步：严格评审。** 逐行对照 schema_preflight.py 的每个 if/elif/else 分支，确认每个分支至少被一个测试覆盖。检查 issue code 字符串是否与源码完全匹配（拼写错误会导致断言失败但不会 catch 逻辑错误）。

**第六步：补充遗漏后提交。**

---

### 5
为 `scripts/leakage_gate.py` 编写完整单元测试。

**第一步：完整阅读 `scripts/leakage_gate.py`。** 理解所有泄漏检测类型：
- patient ID overlap between train/valid/test
- temporal boundary violation (test 样本的时间早于 train)
- 列级信息泄漏检查
- strict 模式行为（warning → failure 升级）
记录所有 issue codes 和 exit code 逻辑。

**第二步：构造 mock 数据：**
- 正常数据（无泄漏）
- patient overlap（train 和 test 有相同 patient_id）
- temporal violation（test 最早时间 < train 最晚时间）
- 同时存在多种泄漏
- 边界情况：train/test 时间刚好相等

**第三步：编写 `tests/test_leakage_gate.py`，至少 8 个 test case。**

**第四步：运行测试。**

**第五步：严格评审。** 检查：
- 是否测试了 --strict 和非 strict 模式的差异？
- 是否验证了报告 JSON 中 issues 列表的具体内容（不只是 pass/fail）？
- 是否测试了输入文件路径不存在的情况？
- 是否测试了空 CSV（只有 header）的情况？

**第六步：补充遗漏后提交。**

---

### 6
为 `scripts/manifest_lock.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 manifest 指纹计算逻辑（SHA256 of 数据文件 + 配置文件 + gate 脚本）、baseline 比较逻辑（匹配/不匹配/缺失文件处理）、输出格式。

**第二步：设计测试矩阵（至少 8 个 case）：**
- 首次生成 manifest → 所有 fingerprint 非空
- 二次生成 manifest（文件不变）→ fingerprint 相同
- 修改数据文件后生成 → fingerprint 变化
- baseline 比较：匹配 → pass
- baseline 比较：不匹配 → fail + 变化列表
- 缺少 baseline 文件 → fail
- 数据文件不存在 → 报错处理
- 空目录 → 处理方式

**第三步：编写 `tests/test_manifest_lock.py`。**

**第四步-第六步：运行 → 严格评审 → 补充 → 提交。**

---

### 7
为 `scripts/definition_variable_guard.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 phenotype_definition_spec 的格式、疾病定义变量列表提取、与训练特征的交集检测逻辑、issue codes。

**第二步：设计测试矩阵（至少 6 个 case）：**
- 无泄漏（定义变量全部在 ignore-cols 中）→ pass
- 定义变量出现在预测特征中 → fail
- phenotype spec 格式错误 → fail
- phenotype spec 缺失 → fail
- 定义变量名大小写不一致 → 是否检测到？
- 空 phenotype spec（无定义变量）→ pass

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 8
为 `scripts/feature_lineage_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 lineage map 格式、post-index 特征检测、疾病定义变量衍生特征检测、ambiguity 处理逻辑。

**第二步：设计测试矩阵（至少 6 个 case）：**
- 正常 lineage（所有特征 pre-index）→ pass
- post-index 特征存在 → fail
- 特征衍生自疾病定义变量 → fail
- lineage 含 ambiguous 来源 → fail (strict) / warning (non-strict)
- lineage map 格式错误 → fail
- lineage map 缺失 → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 9
为 `scripts/imbalance_policy_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 policy JSON 格式、SMOTE/over-sampling/under-sampling 的 scope 验证（必须 train-only）、class_weight 处理、issue codes。

**第二步：设计测试矩阵（至少 6 个 case）：**
- SMOTE scope = train_only → pass
- SMOTE scope = all → fail
- SMOTE scope 缺失 → fail
- class_weight = balanced → pass
- policy 缺失 → fail
- policy 格式错误 → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 10
为 `scripts/missingness_policy_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 policy 格式、imputer isolation 验证（train-only fit）、MICE scale guard、target-info imputation 禁令。

**第二步：设计测试矩阵（至少 6 个 case）：**
- imputer fit = train_only → pass
- imputer fit = all_splits → fail
- MICE + large scale + 无 guard → fail
- MICE + large scale + guard 存在 → pass
- target-info imputation → fail
- policy 缺失 → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 11
为 `scripts/tuning_leakage_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 tuning protocol 格式、test isolation 验证、valid-only hyperparameter search 验证、issue codes。

**第二步：设计测试矩阵（至少 6 个 case）：**
- tuning scope = train+valid → pass
- tuning scope = train+valid+test → fail
- tuning scope 缺失 → fail
- protocol 格式错误 → fail
- test_used_in_tuning = true → fail
- nested CV 配置验证

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 12
为 `scripts/calibration_dca_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解校准方法验证、DCA net benefit 计算、calibration slope/intercept 检查、issue codes。

**第二步：设计测试矩阵（至少 5 个 case）：**
- 正常校准报告 → pass
- calibration slope 超出合理范围 → fail
- DCA 数据缺失 → fail
- 校准在 test set 上做（而非 validation set）→ fail
- 报告格式错误 → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 13
为 `scripts/evaluation_quality_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 CI width 阈值、baseline delta 要求、min_resamples 检查、primary metric 来源验证。

**第二步：设计测试矩阵（至少 6 个 case）：**
- CI width OK + baseline delta OK → pass
- CI width 超标 → fail
- baseline delta 不足 → fail
- CI resamples 数量不足 → fail
- primary metric 来自非 test split → fail
- 正常边界值测试（刚好在阈值上）

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 14
为 `scripts/permutation_significance_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解 permutation null 分布读取、p-value 计算、alpha 阈值、min_delta 检查。

**第二步：设计测试矩阵（至少 5 个 case）：**
- p-value < alpha → pass
- p-value >= alpha → fail
- null 文件缺失 → fail
- null 文件格式错误 → fail
- actual_metric < max(null) + min_delta → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

### 15
为 `scripts/publication_gate.py` 编写完整单元测试。

**第一步：完整阅读源码。** 理解聚合逻辑：扫描所有子报告 → 全部 pass 才聚合为 pass、任一 fail 聚合为 fail、缺少必要报告 → fail。

**第二步：设计测试矩阵（至少 6 个 case）：**
- 全部 27 个子报告 pass → pass
- 1 个 fail → fail + 正确标识哪个 gate 失败
- 多个 fail → fail + 全部列出
- 子报告文件缺失 → fail
- 子报告 JSON 格式错误 → fail
- 子报告 status 字段缺失 → fail

**第三步-第六步：编写 → 运行 → 严格评审 → 补充 → 提交。**

---

## B. 集成测试（16-22）

### 16
创建完整的 onboarding E2E 测试。

**第一步：阅读 `scripts/mlgg_onboarding.py` 和 `scripts/mlgg.py` 中 onboarding 子命令的完整代码。** 理解 8 步流程、每一步的成功/失败条件、报告字段。

**第二步：编写 `tests/test_onboarding_e2e.py`。** 用 subprocess 运行：
```python
subprocess.run([sys.executable, "scripts/mlgg.py", "onboarding",
                "--project-root", str(tmp_dir),
                "--mode", "auto", "--yes"],
               timeout=600, capture_output=True, text=True)
```
验证：
- exit code = 0
- `evidence/onboarding_report.json` 存在
- report 中 `status` = "pass"
- report 中 `termination_reason` = "completed_successfully"
- report 中 `contract_version` = "onboarding_report.v2"
- `data/train.csv`, `data/valid.csv`, `data/test.csv` 存在
- `evidence/dag_pipeline_report.json` 存在
- `models/` 目录非空

**第三步：运行测试（需要较长时间）。**

**第四步：严格评审。** 检查：
- 是否验证了 copy_ready_commands 中的路径是绝对路径？
- 是否验证了 next_actions 非空？
- 是否验证了 failure_codes 为空列表（因为全部通过）？
- 测试失败时是否输出足够的诊断信息（stdout/stderr capture）？

**第五步：补充遗漏后提交。**

---

### 17
创建完整的 split E2E 测试。

**第一步：阅读 `scripts/split_data.py` 的 argparse 和主流程。** 理解输入输出约定。

**第二步：编写 `tests/test_split_e2e.py`。** 使用 `examples/heart_disease.csv` 真实数据，通过 subprocess 调用 split：
```
mlgg.py split -- --input examples/heart_disease.csv --output-dir <tmp>/data
  --patient-id-col patient_id --target-col y --time-col event_time
  --strategy grouped_temporal
```
验证：
- exit code = 0
- train.csv, valid.csv, test.csv 存在
- 用 pandas 读取后：行数之和 = 原始行数（减 NaN 排除）
- patient_id 集合在三个文件间无交叉
- test.csv 的 event_time 最小值 >= train.csv 的 event_time 最大值（temporal ordering）
- 每个 split 中 positive 和 negative 样本数 >= 10
- 对三种策略都运行一遍

**第三步：运行测试。**

**第四步：严格评审。** 检查：是否存在行缺失？列名是否完全保留？临时文件是否清理？

**第五步：补充后提交。**

---

### 18
创建完整的 train E2E 测试。

**第一步：阅读 `scripts/train_select_evaluate.py` 的参数和输出。**

**第二步：编写 `tests/test_train_e2e.py`。** 流程：
1. 用 subprocess 生成 demo 数据（或使用 examples/ 数据 split 后的结果）
2. 用 subprocess 调用 train_select_evaluate.py，使用最小配置（2 个模型、fixed_grid）
3. 验证所有输出文件存在且 JSON 格式正确
4. 验证 model_selection_report.json 中 candidate_count >= 2
5. 验证 evaluation_report.json 中 split = "test"
6. 验证 model.pkl 或 model.joblib 存在
7. 超时设为 300 秒

**第三步-第五步：运行 → 严格评审 → 补充 → 提交。**

---

### 19
创建完整的 workflow E2E 测试。

**第一步：阅读 `scripts/run_productized_workflow.py` 和 `scripts/run_strict_pipeline.py`。**

**第二步：编写 `tests/test_workflow_e2e.py`。** 流程：
1. 先用 onboarding 在 /tmp 下生成完整项目
2. 运行 bootstrap workflow（--allow-missing-compare）
3. 验证 manifest_baseline.bootstrap.json 存在
4. 运行 compare workflow（--compare-manifest）
5. 验证 dag_pipeline_report.json 存在且 status 正确
6. 验证 productized_workflow_report.json 的 contract_version、status、status_reason
7. 超时 600 秒

**第三步-第五步：运行 → 严格评审 → 补充 → 提交。**

---

### 20
创建 adversarial gate 测试。

**第一步：阅读 `experiments/authority-e2e/run_adversarial_gate_checks.py` 和 adversarial/ 目录。** 理解每个对抗测试的注入方式和预期结果。

**第二步：编写 `tests/test_adversarial_e2e.py`。** 对每个对抗案例：
- 验证 gate 正确检测到泄漏并返回 fail
- 验证 issue codes 包含正确的泄漏类型
- 验证没有 false negative（所有注入都被检测到）

**第三步-第五步：运行 → 严格评审 → 补充 → 提交。**

---

### 21
扩展 `tests/test_wizard_interactive.py` 的 pty 交互测试。

**第一步：阅读现有 12 个测试。** 理解 PTYSession 使用方式和 marker 同步策略。

**第二步：新增至少 4 个测试：**
- **T13 CSV 手动输入**：English → CSV → Enter path manually → 输入路径 → 验证到达 Step 4 或 Step 5
- **T14 列数不足错误**：构造 1 列 CSV → 选择该 CSV → 验证出现错误提示且不死循环
- **T15 step_split 完整流程**：Download → Heart → default name → temporal → select time → select ratio → 验证到达 Step 6
- **T16 step_confirm 导出信息**：走完整 download 流程到 Step 8 → 验证确认框中包含所有关键信息（file, pid, target, strategy, ratio, models）

**第三步：运行全部测试（16 个），确保全部通过。**

**第四步：严格评审。** 检查新测试：
- marker 是否足够独特不会误匹配？
- timeout 是否充足？
- 是否正确处理了 send_keys 的时序？

**第五步：补充后提交。**

---

### 22
创建数据集下载测试。

**第一步：阅读 `examples/download_real_data.py`。** 理解下载 URL、保存路径、数据转换逻辑。

**第二步：编写 `tests/test_download_real_data.py`。** 标记为 `@pytest.mark.network`。对 heart/breast/ckd 三个数据集分别：
- 调用下载脚本
- 验证 CSV 存在
- 验证列名包含 patient_id, y, event_time
- 验证行数在合理范围（heart: 250-350, breast: 500-600, ckd: 350-450）
- 验证 target 列只有 0/1

**第三步-第五步：运行 → 严格评审 → 补充 → 提交。**

---

## C. Gate 脚本代码审查与修复（23-42）

### 23
严格审查 `scripts/request_contract_gate.py`。

**第一步：完整阅读文件（从第 1 行到最后一行），不跳过任何代码。** 对每个函数，记录其职责和所有分支。

**第二步：逐行检查以下维度：**
- 所有 JSON 字段验证：required 字段缺失是否都报错？字段类型错误是否检测？
- 路径解析：相对路径是否正确解析为 request 文件所在目录的相对路径？
- strict 模式：是否所有 warning 在 strict 模式下都升级为 failure？
- 错误码一致性：每个 `add_issue` 的 code 是否与文档/SKILL.md 一致？
- finish() 函数：`should_fail = bool(failures) or (args.strict and bool(warnings))` 是否正确？
- Exit code：pass=0, fail=2 是否正确？

**第三步：检查边界条件：**
- request JSON 为空 {}
- request JSON 含多余字段（是否静默忽略？）
- 路径含中文/空格
- 路径不存在
- publication-grade 模式下所有额外必填字段是否都检查？

**第四步：修复发现的 bug（如果有）。每个修复都要说明 root cause。**

**第五步：运行 `python3 scripts/test_gate_smoke.py` 确保无回归。**

**第六步：再次严格审查修复后的代码，确保修复没有引入新问题。**

**第七步：提交。**

---

### 24
严格审查 `scripts/split_protocol_gate.py`。

**第一步：完整阅读全文。** 理解 split protocol JSON 的 schema（strategy, patient_disjoint, temporal_ordering, prevalence_check 等）和所有验证规则。

**第二步：逐行检查：**
- protocol JSON 的每个字段是否都有验证？
- temporal ordering 检查逻辑是否正确（比较方式、时区处理、NaT 处理）？
- group disjoint 检查是否覆盖所有 split pair（train-valid, train-test, valid-test）？
- prevalence check 的阈值和比较逻辑？
- strict 模式下 finish() 行为？
- to_float 使用是否正确（是否有 math.isfinite guard）？

**第三步：检查与 `references/split-protocol.example.json` 的一致性。** gate 是否验证了 example 中定义的所有字段？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 25
严格审查 `scripts/covariate_shift_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 分布漂移检测算法（KS test? Chi-squared? classifier-based?）的正确性
- separability 检测的阈值和判断逻辑
- 多列并行检测时的 multiple testing correction 是否应用？
- 报告中的 shift 指标是否有 confidence interval？
- strict 模式 finish() 行为
- 大数据集时的性能问题（是否有 sampling？）

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 26
严格审查 `scripts/reporting_bias_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- TRIPOD+AI checklist 的所有 required items 是否被强制检查？
- PROBAST+AI checklist 的必填项验证？
- STARD-AI checklist 验证？
- 每个 checklist item 的 status 值域（met/not_met/not_applicable）处理？
- not_applicable 是否对 required items 允许？
- strict 模式行为？

**第三步：对照 `references/reporting-bias-checklist.example.json` 验证覆盖完整性。**

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 27
严格审查 `scripts/model_selection_audit_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- candidate pool 完整性验证（是否检查 min candidate count？）
- one-SE rule replay 逻辑（验证选中模型的 metric 在 best - 1SE 范围内？）
- test isolation：选择过程中是否有任何 test 数据参与？
- 报告中 ranking/scores 的正确性
- 与 `references/evaluation-report.example.json` 的 model_selection 字段对齐

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 28
严格审查 `scripts/feature_engineering_audit_gate.py`。

**第一步：完整阅读全文。** 特别关注之前 Fix 3（error code 修正）和 Fix 4（to_float isfinite guard）是否仍正确。

**第二步：逐行检查：**
- feature group spec 的格式验证
- train-only engineering scope 验证
- stability evidence 检查
- reproducibility fields 验证
- JSON parse 错误的 error code 是否为 `feature_engineering_report_invalid`（Fix 3）
- to_float 是否有 `math.isfinite` guard（Fix 4）

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 29
严格审查 `scripts/clinical_metrics_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 11 个临床指标的完整性验证（accuracy/precision/PPV/NPV/sensitivity/specificity/F1/F2-beta/ROC-AUC/PR-AUC/Brier）
- confusion matrix 一致性（TP+FP+TN+FN = total, precision = TP/(TP+FP) 等）
- 每个 split（train/valid/test）的指标是否都检查？
- 指标值域验证（0-1 范围，Brier score 方向）
- 缺失指标的处理（fail vs warning）

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 30
严格审查 `scripts/prediction_replay_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 预测重放的完整流程（加载模型 → 对 test 数据预测 → 与 trace 对比）
- hash 一致性验证（prediction_trace 的 hash 与 attestation 中的 hash 匹配）
- 容差处理（浮点数比较的 atol/rtol）
- 模型文件缺失处理
- trace 文件格式验证（CSV vs CSV.gz）

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 31
严格审查 `scripts/distribution_generalization_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- train-vs-holdout 分布漂移检测方法和阈值
- split separability 检测（分类器能否区分 train vs test？）
- transport-readiness 判断逻辑
- 与 covariate_shift_gate 的区别和互补性
- strict 模式行为

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 32
严格审查 `scripts/generalization_gap_gate.py`。

**第一步：完整阅读全文。** 确认 Fix 5（`should_fail = bool(failures) or (args.strict and bool(warnings))`）仍正确。

**第二步：逐行检查：**
- gap 阈值的来源（硬编码 vs 来自 performance_policy）
- train-valid gap、valid-test gap、train-test gap 的分别处理
- 各指标的 gap 计算方向（train高test低 = 过拟合）
- warning vs failure 的阈值区分
- strict 模式 finish() 行为

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 33
严格审查 `scripts/robustness_gate.py`。

**第一步：完整阅读全文。** 确认 Fix 6 仍正确。

**第二步：逐行检查：**
- 鲁棒性报告的格式和验证规则
- perturbation 类型和影响评估
- 阈值判断逻辑
- strict 模式行为

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 34
严格审查 `scripts/seed_stability_gate.py`。

**第一步：完整阅读全文。** 确认 Fix 7 仍正确。

**第二步：逐行检查：**
- 多种子运行结果的读取和聚合
- stability 指标计算（std, cv, range）
- 阈值判断（cv 超标 → fail）
- 种子数量不足的处理
- strict 模式行为

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 35
严格审查 `scripts/external_validation_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- cross-period 和 cross-institution 两种外部验证的区分处理
- transport-drop 计算（external metric - internal metric）
- transport-drop 阈值判断
- 外部队列最小样本量检查
- 外部 cohort spec 格式验证
- external_validation_report 格式验证

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 36
严格审查 `scripts/ci_matrix_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- bootstrap CI 计算的正确性（n_resamples、percentile method、seed）
- CI 矩阵覆盖所有 split × metric 组合
- transport-drop CI 计算
- internal 和 external cohort 的分别处理
- CI width 阈值判断
- 与 `references/ci-matrix-report.example.json` 的 schema 一致性

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 37
严格审查 `scripts/metric_consistency_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- evaluation_metric_path 的解析逻辑（JSONPath-like traversal）
- primary_metric 归一化（pr_auc vs PR_AUC vs pr-auc）
- 提取值与 actual_primary_metric 的一致性验证
- path 不存在或值类型错误的处理

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 38
严格审查 `scripts/self_critique_gate.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 评分算法：每个维度的权重、分数计算公式
- 阈值判断（default 95）
- 各维度的评分输入来源
- reviewer-grade 建议生成逻辑
- 浮点精度问题（评分是否可能因 floating point 错误导致 94.999... < 95？）
- strict 模式行为

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 39
严格审查 `scripts/execution_attestation_gate.py`（最复杂 gate）。

**第一步：完整阅读全文（预计 500+ 行）。** 这是最复杂的 gate，需要格外仔细。

**第二步：逐一检查每个验证模块：**
1. 主签名验证（RSA, public key, payload hash）
2. artifact hash 校验（每个 artifact 的 SHA256）
3. key revocation 检查（revocation list, 过期时间, key age）
4. timestamp record 验证（签名, 时间一致性）
5. transparency record 验证
6. execution receipt 验证（exit code, timing）
7. execution log 验证（log binding to payload hash）
8. witness quorum 验证（min count, key independence, role independence）
9. cross-role authority distinctness（signing key ≠ timestamp key ≠ execution key ≠ witness keys）

**第三步：特别检查安全相关逻辑：**
- 签名验证是否使用正确的 padding（PKCS1v15 vs PSS）？
- hash 算法是否为 SHA256？
- 是否防止了 key reuse across roles？
- 时间窗口验证是否有 clock skew 容忍？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 40
严格审查 `scripts/generate_execution_attestation.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- keypair 生成参数（RSA 3072 bits）
- payload 构造（study_id, run_id, timestamps, artifact hashes）
- 签名流程（private key → SHA256 → PKCS1v15 签名）
- 所有附属 artifact 生成（timestamp, transparency, receipt, log, witness records）
- witness quorum 处理（min count, key pair generation/validation）
- bootstrap 逻辑（key 不存在时自动创建 + key_revocations.json 自动创建）

**第三步：验证与 gate（#39）的对称性** — 所有 generate 产生的 artifact 格式是否被 gate 正确验证？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 41
严格审查 `scripts/run_strict_pipeline.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 28 步的执行顺序是否与 SKILL.md 中定义的一致？
- 每步的参数传递是否正确？
- 错误传播：任何 gate 返回非 0 exit code 是否正确停止或记录？
- 报告聚合：dag_pipeline_report.json 的格式、所有字段
- --strict 标志是否传递到每个 gate？
- --compare-manifest 参数的传递

**第三步：验证每步的输入文件是否在前面步骤中已生成。** 画一个简单的依赖图。

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 42
严格审查 `scripts/run_productized_workflow.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 四步流程：doctor → preflight → strict pipeline → user summary
- bootstrap recovery 逻辑（什么条件触发？恢复后重新运行哪些步骤？）
- 报告格式 v2 的所有字段（status, status_reason, blocking_failure_count, recovered_failure_count, bootstrap_recovery_applied, bootstrap_recovery_source, steps[]）
- first-run bootstrap（--allow-missing-compare）的处理
- 与 run_strict_pipeline.py 的参数传递

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

## D. 训练核心代码审查（43-48）

### 43
严格审查 `scripts/train_select_evaluate.py` — 模型训练部分。

**第一步：定位并阅读模型训练相关代码。** 用 code_search 找到 fit/train 相关函数。

**第二步：逐行检查：**
- train/valid/test 数据是否严格隔离？（fit 只在 train 上？preprocessing 只 fit on train？）
- ignore-cols 是否从 train/valid/test 中都正确移除？
- model pool 初始化：是否正确处理了 xgboost/catboost 不可用的情况？
- class_weight 处理：是否对 tree-based 和 logistic 分别正确设置？
- feature engineering scope：任何 scaling/encoding/imputation 是否都是 train-only fit？
- 随机种子控制

**第三步：检查医学 non-negotiable 规则：**
- 是否有任何代码路径会导致 SMOTE 应用到 valid/test？
- 是否有任何代码路径会导致 imputer fit on valid/test？
- 是否有任何 feature selection 步骤使用了 test 数据？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 44
严格审查 `scripts/train_select_evaluate.py` — 模型选择部分。

**第一步：定位模型选择/排名代码。**

**第二步：逐行检查：**
- one-SE rule 实现：是否正确计算 mean ± SE？选择的模型是否在 best-SE 范围内最简单？
- 排名使用的 metric 是否来自 validation set（不是 test）？
- 是否存在任何从 test set 反馈到选择过程的路径？
- report 输出的 candidates 列表是否包含所有候选模型的 metric？
- 选择结果是否可复现（种子控制）？

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 45
严格审查 `scripts/train_select_evaluate.py` — 评估部分。

**第一步：定位评估代码。**

**第二步：逐行检查：**
- 评估是否仅在 test split 上执行？报告中 `split` 字段是否 = "test"？
- CI 计算方法（bootstrap percentile? BCa?），n_resamples 数量
- confusion matrix 计算（TP/FP/TN/FN）
- 11 个临床指标的计算公式：
  - accuracy = (TP+TN)/(TP+FP+TN+FN)
  - precision = TP/(TP+FP)，除零处理？
  - recall/sensitivity = TP/(TP+FN)
  - specificity = TN/(TN+FP)
  - NPV = TN/(TN+FN)
  - F1 = 2*precision*recall/(precision+recall)
  - F2-beta = (1+4)*precision*recall/(4*precision+recall)
  - ROC-AUC = sklearn.metrics.roc_auc_score
  - PR-AUC = sklearn.metrics.average_precision_score
  - Brier = sklearn.metrics.brier_score_loss
- threshold 选择逻辑（Youden's J? F1 optimal?）
- threshold 是否在 validation set 上选择（不是 test）？

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 46
严格审查 `scripts/train_select_evaluate.py` — 超参搜索部分。

**第一步：定位超参搜索代码。**

**第二步：逐行检查：**
- fixed_grid: 网格定义是否合理？是否只在 train+valid 上搜索？
- random_subsample: 采样空间定义？max_trials 限制？
- optuna: study 创建参数？pruning？objective 是否只用 valid metric？
- 所有策略是否都确保 test 数据未参与？
- cross-validation 如果使用，是否在 train 内部 fold？
- 超参结果保存格式

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 47
严格审查 `scripts/train_select_evaluate.py` — 外部验证部分。

**第一步：定位外部验证代码。**

**第二步：逐行检查：**
- external cohort spec 解析（cross_period, cross_institution）
- external 数据加载和预处理（是否使用 train 的 preprocessor？）
- transport-drop 计算：external_metric - internal_test_metric
- 分别处理多个外部队列
- 报告格式与 `references/external-validation-report.example.json` 一致性

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 48
严格审查 `scripts/train_select_evaluate.py` — 输出工件部分。

**第一步：定位所有 report/artifact 写入代码。**

**第二步：逐一验证输出格式：**
- model_selection_report.json vs `references/evaluation-report.example.json`
- evaluation_report.json 的所有必填字段
- prediction_trace.csv(.gz) 的列名和格式
- distribution_report.json 的格式
- ci_matrix_report.json 的格式
- robustness_report.json 的格式
- seed_sensitivity_report.json 的格式
- model.joblib/model.pkl 的序列化方式
- permutation_null 文件格式

**第三步：检查文件写入的原子性** — 是否使用 temp file + rename 避免部分写入？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

## E. CLI 入口与编排审查（49-55）

### 49
严格审查 `scripts/mlgg.py` 命令路由。

**第一步：完整阅读 `scripts/mlgg.py`。**

**第二步：逐行检查所有子命令的 dispatch：**
- onboarding: 参数传递到 mlgg_onboarding.py 是否完整？
- interactive: 参数传递到 mlgg_interactive.py？
- init: 传递到 init_project.py？
- train: 支持 --interactive 标志？
- workflow: 传递到 run_productized_workflow.py？
- play: 调用 mlgg_pixel.py？
- split: 参数 "--" 分隔后正确传递？
- doctor: 调用 env_doctor.py？
- preflight: 调用 schema_preflight.py？
- authority/benchmark-suite/adversarial/authority-release/authority-research-heart: 正确路由？
- --help: 每个子命令的 help 是否正确？
- --error-json: 结构化错误输出？
- 未知子命令的处理？

**第三步：检查 cli_main 函数（pyproject.toml 入口点）是否正确定义和导出。**

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 50
严格审查 `scripts/mlgg_onboarding.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查 8 步流程：**
1. env_doctor: 调用参数？失败处理？
2. init_project: 调用参数？已有项目处理？
3. generate_demo_medical_dataset: 输出路径？数据格式？
4. config alignment: 哪些 config 文件被生成/修改？模板来源？
5. train_select_evaluate: 完整参数列表？输出路径？
6. generate_execution_attestation: keypair bootstrap？
7. run_productized_workflow (bootstrap): --allow-missing-compare？
8. run_productized_workflow (compare): --compare-manifest 路径？

**第三步：检查特殊模式：**
- --mode preview: 是否真的不执行任何命令？
- --mode guided + 无 stdin: 是否 fail-closed？
- --no-stop-on-fail: 是否正确收集所有失败？
- copy_ready_commands 中的路径是否都是绝对路径？

**第四步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 51
严格审查 `scripts/mlgg_interactive.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查四个 wizard：**
- init wizard: 参数收集？默认值？
- workflow wizard: --strict 是否强制注入？first-run 检测？
- train wizard: 所有训练参数的交互收集？可选模型 backend 检测？
- authority wizard: release vs research path 选择？stress case ID 默认？
- 通用逻辑：command preview → confirm → execute 流程
- profile save/load: 文件格式？路径？版本兼容？
- --accept-defaults: 是否跳过所有 prompt？
- --print-only: 是否不执行？

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 52
严格审查 `scripts/init_project.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 创建的目录列表（configs/, data/, evidence/, models/, keys/）
- 生成的 request.json 模板内容是否与 `references/request-schema.example.json` 一致？
- 幂等性：重复执行是否安全？是否覆盖已有文件？
- 路径处理：相对 vs 绝对？含空格？含中文？
- 权限问题处理

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 53
严格审查 `scripts/env_doctor.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- Python 版本检查（>= 3.10）
- 核心依赖检测和版本验证（numpy, pandas, scikit-learn, joblib）
- 可选 backend 检测（xgboost, catboost, lightgbm, tabpfn, optuna）
- openssl 检测（PATH 中可执行？版本？）
- 报告格式和输出
- 检测失败时的行为（warning vs error）

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 54
严格审查 `scripts/render_user_summary.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- evidence 工件的读取和解析逻辑
- Markdown 生成的格式正确性
- 所有 gate 状态的汇总显示
- 中英文支持是否完整？
- 缺少某些 evidence 文件时的处理

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

### 55
严格审查 `scripts/generate_demo_medical_dataset.py`。

**第一步：完整阅读全文。**

**第二步：逐行检查：**
- 合成数据的统计特性（class balance, feature distribution）
- temporal consistency（event_time 是否有序？）
- external cohort 生成（cross_period: 时间范围不同；cross_institution: 分布漂移）
- 输出列名（patient_id, event_time, y, features）
- 随机种子控制（可复现）
- 输出文件路径

**第三步-第七步：修复 → 验证 → 再审查 → 提交。**

---

## F. 像素风 Wizard 增强（56-65）

### 56
在 `mlgg_pixel.py` 的 `step_run` 中添加分阶段进度显示。

**第一步：阅读当前 `step_run` 全部代码。** 理解三个阶段（Download/Split/Train）的流程。

**第二步：设计进度 UI。** 在每个阶段完成时更新进度百分比（Download: 33%, Split: 66%, Train: 100%），用 ANSI 进度条替代当前的 spinner-only 方式。在各阶段间显示已完成步骤的 ✓ 标记。

**第三步：实现代码修改。** 修改 step_run 中三个 phase 之间的 _clear + _progress 调用，添加百分比显示。

**第四步：用 `python3 -c "import py_compile; py_compile.compile(...)"` 验证语法。**

**第五步：运行 `python3 tests/test_wizard_interactive.py` 确保现有 12 个测试不回归。**

**第六步：严格评审。** 重新阅读修改后的代码，检查：
- 进度百分比是否在所有代码路径中正确更新（包括某个阶段失败时）？
- ANSI 进度条的宽度是否适配终端宽度？
- 中英文文本是否都正确？

**第七步：补充后提交。**

---

### 57
在 `step_confirm` 中添加"导出命令行"功能。

**第一步：阅读 step_confirm 和 step_run 的代码，理解所有配置参数如何映射到 CLI 命令。**

**第二步：实现 export_cli 函数。** 将 state 字典转换为等效的 `mlgg.py split` + `train_select_evaluate.py` 命令行字符串。

**第三步：在 step_confirm 的 select 中添加第三个选项 "Export CLI Command"。** 选择后打印命令并等待 Enter。

**第四步：验证导出的命令是否可以直接复制执行。** 用 subprocess 运行导出的 split 命令验证。

**第五步：运行交互测试确保不回归。**

**第六步：严格评审。** 检查：
- 命令中的路径是否使用绝对路径？
- 含空格/中文的路径是否正确引用？
- 所有参数是否都包含在导出命令中？

**第七步：提交。**

---

### 58
添加训练结果展示步骤。

**第一步：了解 model_selection_report.json 和 evaluation_report.json 的格式。** 阅读 `references/evaluation-report.example.json`。

**第二步：在 step_run 成功完成后，添加结果解析和展示逻辑。** 读取报告 JSON，提取关键指标（AUC, PR-AUC, sensitivity, specificity, accuracy），用 box() 在终端展示。

**第三步：处理报告不存在或格式异常的情况。** 用 try-except 保护。

**第四步：验证语法和交互测试。**

**第五步：严格评审。** 检查指标提取路径是否与实际报告格式匹配、中英文标签是否正确。

**第六步：提交。**

---

### 59
为 select/multi_select 添加 Page Up/Down 支持。

**第一步：阅读 _getch 和 select/multi_select 的导航逻辑。**

**第二步：在 _getch 中添加 Page Up (`\x1b[5~`) 和 Page Down (`\x1b[6~`) 的识别。** 注意这些序列以 `~` 结尾，需要在 CSI 解析中处理。

**第三步：在 select 和 multi_select 的导航逻辑中添加 PAGE_UP 和 PAGE_DOWN 处理。** 每次移动 max_vis 行，正确更新 sel 和 offset。

**第四步：编写 2 个新的 pty 测试验证 Page Up/Down 行为。**

**第五步：运行全部交互测试。**

**第六步：严格评审。** 检查：
- Page Up 在顶部时是否正确处理？
- Page Down 在底部时是否正确处理？
- CSI 序列 `\x1b[5~` 中的 `~` 是否在 drain 逻辑中正确处理？
- sel 和 offset 更新后是否保持一致（sel 在可见范围内）？

**第七步：提交。**

---

### 60
添加 `--lang` 命令行参数支持。

**第一步：阅读 main() 和 wizard() 函数。**

**第二步：在 main() 中使用 argparse 添加 `--lang {en,zh}` 参数。**

**第三步：在 wizard() 中，如果指定了 --lang，跳过 step_lang 并直接设置 LANG。**

**第四步：验证语法和测试。**

**第五步：严格评审。** 检查 --lang 与 detect_lang() 的交互、BACK 导航时 --lang 步骤的跳过逻辑。

**第六步：提交。**

---

### 61
为 download 数据集添加下载进度显示。

**第一步：阅读 `examples/download_real_data.py` 的下载逻辑。**

**第二步：修改 download_real_data.py，在下载过程中输出进度信息到 stderr。** 使用 urllib 的回调或 stream reading。

**第三步：修改 mlgg_pixel.py 的 run_spinner 或 step_run，实时读取子进程 stderr 并更新进度条。**

**第四步：验证语法和测试。**

**第五步：严格评审。** 检查：网络超时处理？进度条回退？无网络时的行为？

**第六步：提交。**

---

### 62
添加高级配置步骤。

**第一步：确定需要暴露的高级参数：** `--ignore-cols`、`--n-jobs`、`--max-trials-per-family`、`--include-optional-models`。

**第二步：在 wizard steps 列表中插入 step_advanced（step 7.5，在 tuning 之后 confirm 之前），默认 SKIP。** 在 step_tuning 结尾添加 "Advanced settings?" 选项，选是则进入 step_advanced，否则 SKIP。

**第三步：实现 step_advanced 的 UI（每个参数一个子步骤）。**

**第四步：验证配置正确传递到 step_run。**

**第五步：运行交互测试。**

**第六步：严格评审。** 检查默认跳过路径、BACK 导航、state 一致性。

**第七步：提交。**

---

### 63
添加友好的错误提示。

**第一步：收集 split 和 train 常见的 stderr 错误模式。** 阅读 split_data.py 和 train_select_evaluate.py 的所有 sys.exit 和 raise 路径。

**第二步：在 step_run 中添加错误分类逻辑。** 匹配常见模式（"ValueError: not enough positive"、"FileNotFoundError"、"column .* not found"）并映射到中英文友好提示。

**第三步：在错误 box 中显示友好提示 + 原始错误摘要（最后 3 行）。**

**第四步：验证语法和测试。**

**第五步：严格评审。** 检查模式匹配是否可能误匹配？中英文翻译是否准确？

**第六步：提交。**

---

### 64
实现运行历史记录功能。

**第一步：设计 history.json 格式。** 包含 timestamp、state snapshot（所有配置项）、结果状态。

**第二步：在 step_run 成功完成后保存历史到 `~/.mlgg/history.json`。**

**第三步：在 step_source 中添加 "Repeat last run" 选项（仅当历史存在时显示）。** 选择后加载上次配置到 state 并跳到 step_confirm。

**第四步：验证语法和测试。**

**第五步：严格评审。** 检查：历史文件损坏时的处理？多次运行的历史堆叠？隐私（路径信息）？

**第六步：提交。**

---

### 65
添加 --dry-run 模式。

**第一步：阅读 main() 和 step_run()。**

**第二步：添加 `--dry-run` argparse 参数。** 在 step_run 中检查此标志，如果 dry-run 则打印命令但不执行。

**第三步：在 step_confirm 中标注 [DRY RUN] 模式。**

**第四步：验证语法和测试。**

**第五步：严格评审。** 检查 dry-run 模式下是否真的没有任何副作用（不创建目录、不写文件）。

**第六步：提交。**

---

## G. 文档完善（66-73）

### 66
扩展 Troubleshooting 文档。

**第一步：阅读现有 `references/Troubleshooting-Top20.md`。** 理解格式和已有条目。

**第二步：阅读所有 gate 脚本中的 `add_issue` 调用，收集所有 issue codes。**

**第三步：识别 Top20 之外未覆盖的高频 failure codes，至少补充 10 条。** 覆盖 imbalance、missingness、tuning、calibration、external validation、seed stability、robustness 相关错误码。

**第四步：为每条新增条目编写：failure code → 诊断步骤 → 修复命令 → 验证方法。**

**第五步：严格审查。** 检查修复命令是否可直接复制执行、中英文是否准确。

**第六步：提交。**

---

### 67
创建 CLI API 参考文档。

**第一步：遍历所有 29 个 gate 脚本，提取 argparse 参数定义。**

**第二步：为每个脚本编写标准化文档：** 用途（一句话）、参数列表（名称/类型/必填/默认值/说明）、输入文件、输出文件、exit code（0=pass, 2=fail）、示例命令。

**第三步：编写 `references/API-Reference.md`。**

**第四步：严格审查。** 逐个对照源码确认参数列表完整、默认值正确。

**第五步：提交。**

---

### 68
为 train_select_evaluate.py 添加完整 docstring。

**第一步：阅读全部函数。**

**第二步：为每个函数添加 Google style docstring：** Args（每个参数名、类型、含义）、Returns（类型、含义）、Raises（异常类型、条件）。

**第三步：添加模块级 docstring：** 用途、使用示例、输出文件列表。

**第四步：严格审查。** 对照代码确认 docstring 与实际行为一致。

**第五步：提交。**

---

### 69
为 split_data.py 添加完整 docstring。

**第一步：阅读全部代码。**

**第二步：添加模块级 docstring：** 三种策略的详细算法描述、安全检查清单、输入/输出格式。

**第三步：为每个函数添加 docstring。**

**第四步：严格审查，确认与代码行为一致。**

**第五步：提交。**

---

### 70
补全 README.md 中文部分。

**第一步：阅读 README.md 全文。** 确认中文版缺失的具体章节。

**第二步：对照英文版的第 5-10 节内容，编写完整的中文翻译。** 包含所有命令示例、说明文本、注意事项。

**第三步：确保中文命令示例中的路径、参数与英文版完全一致。**

**第四步：严格审查。** 检查专业术语翻译（leakage→泄漏, gate→门控, attestation→证明）是否前后一致、代码块是否完整。

**第五步：提交。**

---

### 71
创建系统架构文档。

**第一步：阅读 SKILL.md 的 Hidden Workflow 和 Manual Strict Execution Order。**

**第二步：用 Mermaid 语法绘制：** 29 gate 的执行顺序流程图、每个 gate 的输入/输出文件、数据流方向。

**第三步：编写 `references/Architecture.md`。** 包含流程图、gate 简介表格、依赖关系说明。

**第四步：严格审查。** 检查流程图是否与 SKILL.md 一致、文件名是否正确。

**第五步：提交。**

---

### 72
创建贡献指南。

**第一步：了解项目的代码风格和测试规范。**

**第二步：编写 `CONTRIBUTING.md`：** 开发环境搭建、代码风格（Python 格式化工具）、测试要求（新功能必须有测试、gate 脚本必须有 --report 和 --strict）、PR 流程、commit message 规范。

**第三步：严格审查，确保指南与实际项目结构一致。**

**第四步：提交。**

---

### 73
创建 CHANGELOG。

**第一步：运行 `git log --oneline -50` 查看最近的提交历史。**

**第二步：按 [Keep a Changelog](https://keepachangelog.com/) 格式编写 `CHANGELOG.md`。** 分 Added/Changed/Fixed 类别。

**第三步：严格审查，确认每个条目描述准确。**

**第四步：提交。**

---

## H. CI/CD 改进（74-80）

### 74
为 CI 添加 Python 版本矩阵。

**第一步：阅读当前 `ci-smoke.yml`。**

**第二步：修改为 strategy.matrix，添加 Python 3.10 和 3.12。**

**第三步：验证 YAML 语法。**

**第四步：严格审查。** 检查 matrix 语法是否正确、每个版本的依赖是否兼容。

**第五步：提交。**

---

### 75
创建 CI full 流程。

**第一步：检查 `.github/workflows/` 是否已有 ci-full.yml。**

**第二步：如不存在，创建 `.github/workflows/ci-full.yml`。** 配置 nightly schedule、完整 onboarding + benchmark-suite --profile release、超时 90 分钟、失败通知。

**第三步：严格审查 YAML 语法和步骤依赖。**

**第四步：提交。**

---

### 76
创建 CI extended 流程。

**第一步：检查是否已有 ci-extended.yml。**

**第二步：如不存在，创建。** Weekly schedule、extended benchmark、超时 6 小时、结果存 artifact。

**第三步：严格审查。**

**第四步：提交。**

---

### 77
在 CI 中添加交互测试。

**第一步：阅读 `tests/test_wizard_interactive.py` 的依赖（pty 模块，仅 Linux/macOS）。**

**第二步：在 ci-smoke.yml 中添加步骤，条件执行（仅 Linux/macOS）。**

**第三步：严格审查。** 确认 pty 在 GitHub Actions ubuntu-latest 上可用。

**第四步：提交。**

---

### 78
添加代码覆盖率收集。

**第一步：创建 `.coveragerc` 配置。** 排除 test 文件、__pycache__、experiments/。

**第二步：修改 ci-smoke.yml，添加 `pip install pytest-cov` 和 `pytest --cov=scripts --cov-report=xml`。**

**第三步：（可选）配置 Codecov 上传。**

**第四步：严格审查覆盖率配置。**

**第五步：提交。**

---

### 79
添加 lint 检查。

**第一步：创建 `ruff.toml`。** 选择合理规则（E, F, W，忽略 E501 行长度）。

**第二步：在 ci-smoke.yml 中添加 `pip install ruff && ruff check scripts/`。**

**第三步：本地运行 ruff check，修复所有 error（不修复 warning）。**

**第四步：严格审查。** 确认规则不会误报现有代码。

**第五步：提交。**

---

### 80
添加类型检查。

**第一步：创建 `mypy.ini`。** 设定 strict 检查核心模块（_gate_utils.py, split_data.py）。

**第二步：为核心模块添加类型注解。**

**第三步：在 CI 中添加 mypy 步骤。**

**第四步：严格审查类型注解正确性。**

**第五步：提交。**

---

## I. 性能与健壮性（81-88）

### 81
优化 train_select_evaluate.py 内存使用。

**第一步：阅读数据加载和训练代码，识别内存瓶颈。**

**第二步：添加 `--low-memory` 参数。** 启用时：chunked CSV reading、及时 del DataFrame、gc.collect()。

**第三步：用大数据集（>100k 行）测试内存峰值。**

**第四步：严格审查。** 检查 low-memory 模式是否影响结果正确性。

**第五步：提交。**

---

### 82
为 split_data.py 添加大文件支持。

**第一步：阅读 SHA256 fingerprint 计算代码。**

**第二步：改为 chunked reading（每次读 64KB）计算 SHA256。**

**第三步：用 1GB+ 文件测试。**

**第四步：严格审查。** 验证 fingerprint 结果与一次性读取一致。

**第五步：提交。**

---

### 83
为 gate 脚本添加统一超时机制。

**第一步：阅读 `_gate_utils.py`，设计 timeout 装饰器或参数。**

**第二步：在 _gate_utils.py 中添加 `--timeout` 参数支持和信号处理。**

**第三步：在 2-3 个 gate 脚本中试用，验证超时后正确输出 timeout 报告。**

**第四步：严格审查。** 检查信号处理是否安全、子线程是否被正确清理。

**第五步：提交。**

---

### 84
为 train_select_evaluate.py 添加 checkpoint 功能。

**第一步：设计 checkpoint 格式（已完成的模型列表、当前最佳模型、partial results）。**

**第二步：在每个模型训练完成后保存 checkpoint。**

**第三步：添加 `--resume-from-checkpoint` 参数，加载已有结果并跳过已完成模型。**

**第四步：严格审查。** 检查 checkpoint 恢复后结果是否与全新运行一致、checkpoint 文件损坏时的处理。

**第五步：提交。**

---

### 85
优化 scan_csv 函数。

**第一步：阅读 mlgg_pixel.py 的 scan_csv()。**

**第二步：添加约束：** 文件大小 > 100MB 跳过、单目录扫描超过 2 秒跳过、总文件数上限 30。

**第三步：运行交互测试。**

**第四步：严格审查。** 检查超时逻辑是否正确、是否有 edge case 导致永久阻塞。

**第五步：提交。**

---

### 86
为 run_strict_pipeline.py 添加并行 gate 支持。

**第一步：分析 29 gate 的依赖关系。** 用 SKILL.md 中的 Hidden Workflow 确认哪些 gate 之间无依赖。

**第二步：实现 `--parallel` 模式。** 使用 concurrent.futures.ProcessPoolExecutor 并行执行无依赖 gate。

**第三步：验证并行结果与串行一致。**

**第四步：严格审查。** 检查文件写入冲突、日志混乱、错误传播。

**第五步：提交。**

---

### 87
为 gate 输出添加执行时间记录。

**第一步：修改 _gate_utils.py，添加 start_time/end_time 记录和 execution_time_seconds 字段写入。**

**第二步：在 run_strict_pipeline.py 中聚合显示总耗时和 top-5 慢 gate。**

**第三步：运行完整流程验证。**

**第四步：严格审查。** 确认时间记录不影响 gate 逻辑。

**第五步：提交。**

---

### 88
为 run_spinner 添加超时保护。

**第一步：阅读 mlgg_pixel.py 的 run_spinner 和 Spinner 类。**

**第二步：修改 run_spinner，添加 timeout 参数（默认 1800 秒）。** 超时后 kill 子进程并返回错误。

**第三步：运行交互测试。**

**第四步：严格审查。** 检查 kill 后资源清理、zombie 进程处理。

**第五步：提交。**

---

## J. 新功能开发（89-100）

### 89
创建本地 Web UI。

**第一步：设计 Web wizard 的页面结构（9 步对应 9 个页面）。**

**第二步：选择框架（Flask + Jinja2 或 FastAPI + 前端），创建 `scripts/mlgg_web.py`。** 实现：步骤进度条、配置表单（对应 wizard 的每个 step）、文件上传（CSV）、实时日志流（SSE/WebSocket）、结果展示。

**第三步：绑定到 127.0.0.1:8501，不对外暴露。**

**第四步：实现完整 9 步流程。**

**第五步：严格审查。** 检查安全性（无 CSRF？输入验证？路径遍历？）、端口冲突处理、进程管理。

**第六步：编写基本的请求测试。**

**第七步：提交。**

---

### 90
添加 LightGBM 模型支持。

**第一步：阅读 train_select_evaluate.py 中 xgboost/catboost 的集成方式。**

**第二步：按相同模式添加 `lightgbm_balanced`。** Auto-detect lightgbm 包、fail-closed when requested but unavailable。

**第三步：更新 mlgg_pixel.py 的 MODEL_POOL。**

**第四步：运行训练测试验证。**

**第五步：严格审查。** 检查 class_weight 处理、超参搜索空间、与 requirements-optional.txt 的一致性。

**第六步：提交。**

---

### 91
添加 TabPFN 模型支持。

**第一步：了解 TabPFN 的限制（< 10000 行、< 100 特征）。**

**第二步：添加 `tabpfn` 到 model pool，带自动尺寸检查。**

**第三步：更新 mlgg_pixel.py 的 MODEL_POOL。**

**第四步：严格审查尺寸限制的 guard 逻辑。**

**第五步：提交。**

---

### 92
创建 gate 报告解释器。

**第一步：收集所有 failure code 到 code→explanation 映射。** 阅读每个 gate 脚本的 `add_issue` 调用。

**第二步：创建 `scripts/explain_gate.py`。** 输入一个 gate 报告 JSON，输出人类可读的中英文解释。每个 failure code 映射到诊断步骤和修复建议。

**第三步：验证对所有已知 failure codes 的覆盖。**

**第四步：严格审查。** 检查是否有新增的 failure code 未覆盖。

**第五步：提交。**

---

### 93
创建结果可视化工具。

**第一步：了解 evaluation_report.json 和 prediction_trace.csv 的格式。**

**第二步：创建 `scripts/visualize_results.py`。** 使用 matplotlib 生成：ROC 曲线（带 AUC）、PR 曲线（带 AP）、Calibration 曲线、DCA 曲线、Feature Importance Top-20。

**第三步：输出 PNG 到 evidence/ 目录。**

**第四步：严格审查。** 检查图表标签、颜色方案、DPI、中文字体支持。

**第五步：提交。**

---

### 94
在 wizard 中添加完整 29 gate 管线模式。

**第一步：了解 publication-grade 路径需要的额外配置（lineage spec, phenotype spec, feature group spec 等）。**

**第二步：在 step_source 中添加第 4 选项 "Full Publication-Grade Pipeline"。**

**第三步：添加额外步骤引导用户提供/生成所需配置文件。**

**第四步：在 step_run 中调用 run_productized_workflow.py。**

**第五步：严格审查完整流程的 state 一致性和 BACK 导航。**

**第六步：提交。**

---

### 95
创建运行比较工具。

**第一步：设计 diff report 格式（变化的 gate 状态、指标变化、failure code 变化）。**

**第二步：创建 `scripts/compare_runs.py`。** 输入两个 evidence 目录，输出 JSON diff 和 Markdown 摘要。

**第三步：用两次 onboarding 运行的结果测试。**

**第四步：严格审查。** 检查所有 gate 报告的比较逻辑、缺失文件处理。

**第五步：提交。**

---

### 96
扩展 download_real_data.py 数据集支持。

**第一步：阅读现有下载逻辑。**

**第二步：添加数据集：** Diabetes 130-US Hospitals、Hepatitis、SPECT Heart、Dermatology。每个数据集的 URL、列映射、标准化处理。

**第三步：验证每个数据集的下载和格式化。**

**第四步：严格审查。** 检查 URL 可用性、数据转换正确性、edge case（缺失值处理）。

**第五步：提交。**

---

### 97
创建 LaTeX 导出工具。

**第一步：了解医学论文常见的表格格式（Table 1: Baseline、Table 2: Performance、Table 3: External）。**

**第二步：创建 `scripts/export_latex.py`。** 从 evidence JSON 生成 LaTeX tabular 代码。

**第三步：验证生成的 LaTeX 可编译。**

**第四步：严格审查。** 检查数字格式（小数位）、CI 显示格式、表格对齐。

**第五步：提交。**

---

### 98
创建多 LLM agent 接口。

**第一步：阅读 `agents/openai.yaml`，理解格式。**

**第二步：创建 `agents/claude.yaml` 和 `agents/gemini.yaml`。** 适配各平台 API 格式。

**第三步：严格审查。** 对照 OpenAI 版本确认功能对等。

**第四步：提交。**

---

### 99
添加 pyproject.toml 入口点。

**第一步：阅读 pyproject.toml 的 `[project.scripts]` 部分。**

**第二步：添加 `mlgg-pixel = "scripts.mlgg_pixel:main"`。**

**第三步：验证 `mlgg play` 子命令正确调用 mlgg_pixel。**

**第四步：用 `pip install -e .` 安装后测试 `mlgg-pixel` 命令。**

**第五步：严格审查。** 确认入口点函数签名正确、import 路径正确。

**第六步：提交。**

---

### 100
创建全矩阵集成测试。

**第一步：设计测试矩阵：** 3 数据集（heart, breast, ckd）× 3 split 策略 × 2 调优策略 = 18 组合。

**第二步：编写 `tests/test_full_matrix.py`。** 使用 `pytest.mark.parametrize` 参数化。每个组合：下载数据 → split → train（最小模型 2 个）→ 验证 evaluation_report.json 存在且格式正确。标记 `@pytest.mark.slow`。超时 3600 秒。

**第三步：运行 1-2 个组合验证测试框架正确。**

**第四步：严格审查。** 检查参数组合是否完整、临时目录清理、测试隔离性。

**第五步：提交。**
