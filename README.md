# medical-ml-leakage-guard

Publication-grade medical prediction workflow with strict anti-data-leakage gates, reproducibility evidence, and fail-closed review logic.

面向医学预测任务的发布级防泄漏工作流，提供严格门控、可复现实验工件与 fail-closed 审核机制。

---

## English

### What This Project Is
- A strict workflow for **medical binary prediction** (risk/prognosis/readmission style tasks).
- Built to block common leakage paths:
  - definition-variable leakage
  - lineage leakage
  - split/time contamination
  - tuning/model-selection leakage
  - threshold/calibration misuse
- Produces machine-checkable evidence artifacts and aggregated release gates.

### Core CLI
- Unified entrypoint:
  - `python3 scripts/mlgg.py <subcommand> [args]`
- Core subcommands:
  - `interactive`, `init`, `doctor`, `preflight`, `workflow`, `strict`, `summary`, `train`, `authority`, `adversarial`

### Interactive Wizard (V7)
- New terminal wizard for core commands: `init / workflow / train / authority`
- Two trigger modes:
  - `python3 scripts/mlgg.py interactive --command train`
  - `python3 scripts/mlgg.py train --interactive`
- Wizard behavior:
  - collect options in terminal
  - preview final command
  - execute only after one confirmation
- Train wizard safety defaults:
  - optional model backends are **off by default** (avoid hard-fail on missing `xgboost/catboost`)
  - default `n_jobs` is `1` for maximum cross-platform stability (increase manually when needed)
  - `external_validation_report_out` is emitted only when `external_cohort_spec` is provided
  - `feature_engineering_report_out` is emitted only when `feature_group_spec` is provided
- Reusable profiles:
  - save: `--profile-name <name> --save-profile`
  - load: `--profile-name <name> --load-profile`
  - profile dir default: `~/.mlgg/profiles` (override with `--profile-dir`)

### Quick Start
1. Environment check:
   - `python3 scripts/mlgg.py doctor`
2. Initialize a project:
   - non-interactive: `python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo`
   - interactive: `python3 scripts/mlgg.py init --interactive`
3. Put split files into `data/train.csv`, `data/valid.csv`, `data/test.csv`.
4. Run productized strict flow:
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict`

### Strict Validation and Benchmarks
- Gate smoke tests:
  - `python3 scripts/test_gate_smoke.py`
- Authority E2E:
  - `python3 scripts/mlgg.py authority`
- Adversarial fail-closed checks:
  - `python3 scripts/mlgg.py adversarial`

### Repository Map
- `scripts/`: all gates, trainers, wrappers, and CLI tools.
- `references/`: schema/policy/report examples and rigor checklists.
- `experiments/authority-e2e/`: benchmark datasets, E2E runners, adversarial scenarios.
- `SKILL.md`: full workflow contract and gate ordering.

### Notes
- This project is for predictive modeling rigor, not causal inference claims.
- Publication-grade claims require all strict gates to pass.

---

## 中文说明

### 这个项目解决什么问题
- 用于**医学二分类预测**（风险预测、预后、再入院等）的严格流程。
- 重点阻断常见数据泄漏路径：
  - 疾病定义变量泄漏
  - 特征血缘泄漏
  - 数据划分/时间污染
  - 调参与模型选择泄漏
  - 阈值与校准不合规
- 输出可机器校验的证据工件，并通过总发布门汇总判断。

### 统一终端入口
- 统一命令：
  - `python3 scripts/mlgg.py <subcommand> [args]`
- 常用子命令：
  - `interactive`, `init`, `doctor`, `preflight`, `workflow`, `strict`, `summary`, `train`, `authority`, `adversarial`

### 交互式终端向导（V7）
- 新增交互向导，覆盖核心命令：`init / workflow / train / authority`
- 两种进入方式：
  - `python3 scripts/mlgg.py interactive --command train`
  - `python3 scripts/mlgg.py train --interactive`
- 交互行为：
  - 在终端逐项选择参数
  - 先展示最终命令
  - 二次确认后才执行
- `train` 向导安全默认：
  - 可选模型后端默认关闭（避免本机未安装 `xgboost/catboost` 时直接失败）
  - `n_jobs` 默认是 `1`（优先跨平台稳定，需并行时可手动调大）
  - 只有提供 `external_cohort_spec` 才会生成 `external_validation_report_out`
  - 只有提供 `feature_group_spec` 才会生成 `feature_engineering_report_out`
- 支持配置复用（profile）：
  - 保存：`--profile-name <name> --save-profile`
  - 加载：`--profile-name <name> --load-profile`
  - 默认目录：`~/.mlgg/profiles`（可用 `--profile-dir` 覆盖）

### 快速开始
1. 先做环境检查：
   - `python3 scripts/mlgg.py doctor`
2. 初始化项目：
   - 非交互：`python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo`
   - 交互：`python3 scripts/mlgg.py init --interactive`
3. 将分割数据放入 `data/train.csv`, `data/valid.csv`, `data/test.csv`。
4. 运行产品化严格流程：
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict`

### 严格验证与基准测试
- Gate 冒烟测试：
  - `python3 scripts/test_gate_smoke.py`
- Authority E2E：
  - `python3 scripts/mlgg.py authority`
- 对抗 fail-closed 检查：
  - `python3 scripts/mlgg.py adversarial`

### 目录说明
- `scripts/`：所有 gate、训练器、封装器与 CLI。
- `references/`：schema/policy/report 示例与顶刊级检查清单。
- `experiments/authority-e2e/`：权威数据集实验、E2E 与对抗脚本。
- `SKILL.md`：完整流程契约与 gate 顺序。

### 说明
- 该项目面向预测建模严谨性，不直接支持因果结论声明。
- 若要声明 publication-grade，必须严格门全部通过。
