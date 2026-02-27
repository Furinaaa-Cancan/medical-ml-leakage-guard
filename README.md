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
  - `onboarding`, `interactive`, `init`, `doctor`, `preflight`, `workflow`, `strict`, `summary`, `train`, `authority`, `adversarial`

### Novice Onboarding (V8)
- One command (recommended for first-time users):
  - `python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes`
- Modes:
  - `guided`: step-by-step confirmation
  - `preview`: print full 8-step command plan only
  - `auto`: execute full flow non-interactively
- Failure control:
  - default: `--stop-on-fail` (enabled by default)
  - full diagnosis run: `--no-stop-on-fail`
- Fixed flow:
  - doctor -> init -> demo data -> config alignment -> train -> attestation -> workflow bootstrap -> workflow compare
- Outputs:
  - onboarding report: `<project>/evidence/onboarding_report.json`
  - user summary: `<project>/evidence/user_summary.md`
  - onboarding report contract: `onboarding_report.v2` (`stop_on_fail`, `termination_reason`, `failure_codes`, `next_actions`)

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
  - non-blocking run with profile/defaults: `--accept-defaults`

### Quick Start
1. Fastest path (novice):
   - `python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes`
2. Manual path (advanced):
   - `python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo`
   - `python3 scripts/mlgg.py train --interactive`
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --allow-missing-compare`
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --compare-manifest /tmp/mlgg_demo/evidence/manifest_baseline.bootstrap.json`

Note:
- `workflow` now resolves relative `--evidence-dir` against the request project base (if request is under `configs/`, evidence defaults to `<project>/evidence`).

### Strict Validation and Benchmarks
- Gate smoke tests:
  - `python3 scripts/test_gate_smoke.py`
- Onboarding smoke tests:
  - `python3 scripts/test_onboarding_smoke.py`
- Authority E2E:
  - `python3 scripts/mlgg.py authority`
- Adversarial fail-closed checks:
  - `python3 scripts/mlgg.py adversarial`

### Repository Map
- `scripts/`: all gates, trainers, wrappers, and CLI tools.
- `references/`: schema/policy/report examples and rigor checklists.
- `experiments/authority-e2e/`: benchmark datasets, E2E runners, adversarial scenarios.
- `SKILL.md`: full workflow contract and gate ordering.
- `references/Beginner-Quickstart.md`: bilingual novice tutorial.
- `references/Troubleshooting-Top20.md`: high-frequency failure-code remediation guide.

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
  - `onboarding`, `interactive`, `init`, `doctor`, `preflight`, `workflow`, `strict`, `summary`, `train`, `authority`, `adversarial`

### 新手引导（V8）
- 首次使用推荐一条命令：
  - `python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes`
- 模式说明：
  - `guided`：逐步确认执行
  - `preview`：仅输出完整 8 步命令计划
  - `auto`：非交互串行执行全部步骤
- 失败控制：
  - 默认：`--stop-on-fail`（默认开启）
  - 全量诊断：`--no-stop-on-fail`
- 固定流程：
  - doctor -> init -> demo 数据 -> 配置对齐 -> train -> attestation -> workflow 首跑 -> workflow 基线对比复跑
- 关键产物：
  - 引导报告：`<project>/evidence/onboarding_report.json`
  - 用户摘要：`<project>/evidence/user_summary.md`
  - 引导报告契约：`onboarding_report.v2`（含 `stop_on_fail`、`termination_reason`、`failure_codes`、`next_actions`）

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
  - 使用默认值免交互执行：`--accept-defaults`

### 快速开始
1. 新手最快路径：
   - `python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes`
2. 手动路径（进阶）：
   - `python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo`
   - `python3 scripts/mlgg.py train --interactive`
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --allow-missing-compare`
   - `python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --compare-manifest /tmp/mlgg_demo/evidence/manifest_baseline.bootstrap.json`

说明：
- `workflow` 的相对 `--evidence-dir` 现在按 request 项目根目录解析（当 request 位于 `configs/` 下时，默认是 `<project>/evidence`）。

### 严格验证与基准测试
- Gate 冒烟测试：
  - `python3 scripts/test_gate_smoke.py`
- Onboarding 冒烟测试：
  - `python3 scripts/test_onboarding_smoke.py`
- Authority E2E：
  - `python3 scripts/mlgg.py authority`
- 对抗 fail-closed 检查：
  - `python3 scripts/mlgg.py adversarial`

### 目录说明
- `scripts/`：所有 gate、训练器、封装器与 CLI。
- `references/`：schema/policy/report 示例与顶刊级检查清单。
- `experiments/authority-e2e/`：权威数据集实验、E2E 与对抗脚本。
- `SKILL.md`：完整流程契约与 gate 顺序。
- `references/Beginner-Quickstart.md`：双语新手教程。
- `references/Troubleshooting-Top20.md`：高频失败码修复手册。

### 说明
- 该项目面向预测建模严谨性，不直接支持因果结论声明。
- 若要声明 publication-grade，必须严格门全部通过。
