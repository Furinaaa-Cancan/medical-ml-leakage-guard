# medical-ml-leakage-guard

Publication-grade medical prediction workflow with strict anti-data-leakage gates, reproducibility evidence, and fail-closed review logic.

面向医学预测任务的发布级防泄漏工作流，提供严格门控、可复现实验工件与 fail-closed 审核机制。

---

## English Guide

### 1. What This Repository Does
- Builds and reviews **medical binary prediction** pipelines under strict leakage controls.
- Enforces publication-grade checks for:
  - definition-variable leakage
  - feature lineage leakage
  - split/time contamination
  - model-selection/tuning leakage
  - threshold/calibration misuse
  - external cohort transport robustness
- Outputs machine-checkable evidence and gate reports for release decisions.

---

### 2. Requirements
- Python `3.10+`
- `openssl` in PATH (required for execution attestation)
- Python packages: `numpy`, `pandas`, `scikit-learn`, `joblib`
- Optional model backends: `xgboost`, `catboost`

Install core dependencies:

```bash
python3 -m pip install -U numpy pandas scikit-learn joblib
```

Check runtime environment:

```bash
python3 scripts/mlgg.py doctor
```

---

### 3. Fastest First Run (Recommended for New Users)

#### 3.1 One-command onboarding

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes
```

This runs a fixed 8-step strict flow:
1. `doctor`
2. `init`
3. generate offline demo medical data
4. align configs
5. train/select/evaluate
6. generate attestation artifacts
7. strict workflow bootstrap (`--allow-missing-compare`)
8. strict workflow compare rerun

#### 3.2 Preview commands only (no execution)

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode preview
```

#### 3.3 Continue after failures for full diagnosis

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode auto --no-stop-on-fail
```

---

### 4. Key Outputs and How To Read Them

After onboarding (or manual workflow), check:
- `<project>/evidence/onboarding_report.json`
- `<project>/evidence/strict_pipeline_report.json`
- `<project>/evidence/productized_workflow_report.json`
- `<project>/evidence/user_summary.md`

`onboarding_report.json` contract is `onboarding_report.v2`:
- `status`: `pass` or `fail`
- `stop_on_fail`: run-time behavior (`true` or `false`)
- `termination_reason`:
  - `completed_successfully`
  - `stopped_on_failure`
  - `completed_with_failures`
  - `cancelled_by_user`
- `failure_codes`: merged codes from gate reports + onboarding step-level codes
- `next_actions`: remediation commands (includes recommended release benchmark and advanced heart research route when onboarding passes)
- `copy_ready_commands`: copy/paste-ready command block (`workflow_bootstrap/workflow_compare/authority_release/authority_research_heart/adversarial`)

`productized_workflow_report.json` contract is `productized_workflow_report.v2`:
- `status`: `pass` or `fail`
- `status_reason`:
  - `all_blocking_steps_passed`
  - `blocking_step_failed`
  - `bootstrap_recovered`
- `blocking_failure_count`: count of blocking steps still failed at final state
- `recovered_failure_count`: count of steps marked as `recovered`
- `bootstrap_recovery_applied`: whether bootstrap retry recovery was applied
- `bootstrap_recovery_source`: bootstrap trigger evidence source (or `null`)
- `steps[]` now includes:
  - `status`: `pass|fail|recovered`
  - `blocking`: `true|false`
  - `recovered_by_step`: retry step name or `null`

Bootstrap isolation rule:
- Bootstrap retry is triggered only by evidence generated in the current strict run.
- Historical `publication_gate_report.json` / `manifest.json` are ignored.

Quick inspect with Python:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/tmp/mlgg_demo/evidence/onboarding_report.json")
r = json.loads(p.read_text(encoding="utf-8"))
print("status:", r["status"])
print("termination_reason:", r.get("termination_reason"))
print("failure_codes:", r.get("failure_codes", []))
print("copy_ready_commands:", sorted(r.get("copy_ready_commands", {}).keys()))
PY
```

---

### 5. Use Your Own Data (Manual Publication-Grade Path)

#### Step A: Initialize project skeleton

```bash
python3 scripts/mlgg.py init --project-root /tmp/mlgg_project
```

#### Step B: Prepare dataset files

Put your files here:
- `/tmp/mlgg_project/data/train.csv`
- `/tmp/mlgg_project/data/valid.csv`
- `/tmp/mlgg_project/data/test.csv`

For publication-grade external validation, prepare both:
- cross-period external cohort CSV
- cross-institution external cohort CSV

Minimum column contract (recommended):
- `patient_id`: patient/entity ID
- `event_time`: index/event time
- `y`: binary label (`0/1`)
- plus leakage-safe predictors

#### Step C: Run schema preflight

```bash
python3 scripts/mlgg.py preflight \
  --train /tmp/mlgg_project/data/train.csv \
  --valid /tmp/mlgg_project/data/valid.csv \
  --test /tmp/mlgg_project/data/test.csv \
  --target-col y \
  --patient-id-col patient_id \
  --time-col event_time \
  --mapping-out /tmp/mlgg_project/evidence/schema_mapping.json \
  --report /tmp/mlgg_project/evidence/schema_preflight_report.json
```

#### Step D: Train/select/evaluate

Option 1 (recommended for most users): interactive wizard

```bash
python3 scripts/mlgg.py train --interactive
```

Option 2: direct CLI template

```bash
python3 scripts/train_select_evaluate.py \
  --train /tmp/mlgg_project/data/train.csv \
  --valid /tmp/mlgg_project/data/valid.csv \
  --test /tmp/mlgg_project/data/test.csv \
  --target-col y \
  --patient-id-col patient_id \
  --ignore-cols patient_id,event_time \
  --performance-policy /tmp/mlgg_project/configs/performance_policy.json \
  --missingness-policy /tmp/mlgg_project/configs/missingness_policy.json \
  --feature-group-spec /tmp/mlgg_project/configs/feature_group_spec.json \
  --external-cohort-spec /tmp/mlgg_project/configs/external_cohort_spec.json \
  --model-selection-report-out /tmp/mlgg_project/evidence/model_selection_report.json \
  --evaluation-report-out /tmp/mlgg_project/evidence/evaluation_report.json \
  --prediction-trace-out /tmp/mlgg_project/evidence/prediction_trace.csv.gz \
  --external-validation-report-out /tmp/mlgg_project/evidence/external_validation_report.json \
  --feature-engineering-report-out /tmp/mlgg_project/evidence/feature_engineering_report.json \
  --distribution-report-out /tmp/mlgg_project/evidence/distribution_report.json \
  --ci-matrix-report-out /tmp/mlgg_project/evidence/ci_matrix_report.json \
  --robustness-report-out /tmp/mlgg_project/evidence/robustness_report.json \
  --seed-sensitivity-out /tmp/mlgg_project/evidence/seed_sensitivity_report.json \
  --model-out /tmp/mlgg_project/models/model.joblib \
  --permutation-null-out /tmp/mlgg_project/evidence/permutation_null_pr_auc.txt
```

#### Step E: Run strict workflow (bootstrap + compare)

First strict run (bootstrap baseline manifest):

```bash
python3 scripts/mlgg.py workflow \
  --request /tmp/mlgg_project/configs/request.json \
  --strict \
  --allow-missing-compare
```

Second strict run (compare against baseline):

```bash
python3 scripts/mlgg.py workflow \
  --request /tmp/mlgg_project/configs/request.json \
  --strict \
  --compare-manifest /tmp/mlgg_project/evidence/manifest_baseline.bootstrap.json
```

---

### 6. Interactive Wizard (Terminal UX)

Core interactive targets:
- `init`
- `workflow`
- `train`
- `authority`

Entry methods:

```bash
python3 scripts/mlgg.py interactive --command train
python3 scripts/mlgg.py train --interactive
python3 scripts/mlgg.py interactive --command train -- --help
```

Reusable profiles:

```bash
# save
python3 scripts/mlgg.py interactive --command train --profile-name demo --save-profile

# load
python3 scripts/mlgg.py interactive --command train --profile-name demo --load-profile
```

Print generated command only:

```bash
python3 scripts/mlgg.py interactive --command workflow --print-only --accept-defaults
```

---

### 7. Validation and Benchmark Commands

```bash
# unified help
python3 scripts/mlgg.py --help

# gate smoke tests
python3 scripts/test_gate_smoke.py

# onboarding smoke tests
python3 scripts/test_onboarding_smoke.py

# authority benchmark suite
python3 scripts/mlgg.py authority

# authority release-grade stress path (recommended wrapper)
python3 scripts/mlgg.py authority-release
# equivalent explicit form:
python3 scripts/mlgg.py authority --include-stress-cases --stress-case-id uci-chronic-kidney-disease

# heart stress is advanced research/high-pressure mode (can fail by design)
python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060
# equivalent explicit form:
python3 scripts/mlgg.py authority --include-stress-cases --stress-case-id uci-heart-disease --stress-seed-search --stress-seed-min 20250003 --stress-seed-max 20250060

# adversarial fail-closed checks
python3 scripts/mlgg.py adversarial

# machine-readable failure payload (JSON on stderr)
python3 scripts/mlgg.py authority-release --dry-run --stress-case-id uci-heart-disease --error-json
```

Notes:
- Default stress case is `uci-chronic-kidney-disease` for a stable publication-grade path.
- `uci-heart-disease` stress search is an advanced research/high-pressure benchmark; seed ranges may have no release-ready candidate under fixed strict floors.
- Interactive `authority` wizard now defaults to the CKD release path; heart is presented as an advanced option with explicit warning.
- `authority-release` and `authority-research-heart` are fixed-route wrappers; conflicting route flags are rejected fail-closed.
- Use `--error-json` to emit structured failure payloads (`contract_version=mlgg_error.v1`) for automation.
- CI pipelines:
  - `.github/workflows/ci-smoke.yml` (push/PR fast checks)
  - `.github/workflows/ci-full.yml` (nightly/manual full regression: authority-release + adversarial)

---

### 8. Troubleshooting (New User Focus)

If guided mode is cancelled, onboarding now fails closed with:
- failure code: `onboarding_step_cancelled`
- actionable `next_actions` in onboarding report
- wrapper conflict failure code: `authority_preset_route_override_forbidden`

Use this mapping for top failures:
- `references/Troubleshooting-Top20.md`

Typical diagnosis commands:

```bash
python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict
python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare
python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json
python3 scripts/mlgg.py authority-release --dry-run --stress-case-id uci-heart-disease --error-json
```

---

### 9. Repository Map
- `scripts/`: gates, trainers, wrappers, and CLI tools
- `references/`: schema/policy/report examples and checklists
- `experiments/authority-e2e/`: authority and adversarial runners
- `SKILL.md`: full workflow contract and gate ordering
- `references/Beginner-Quickstart.md`: bilingual beginner tutorial
- `references/Troubleshooting-Top20.md`: top failure-code remediation

---

### 10. Scope Notes
- This repository is for predictive modeling rigor, not causal inference claims.
- Publication-grade claim is valid only when strict gates all pass.

---

## 中文指南

### 1. 这个仓库是做什么的
- 用于**医学二分类预测**的严格工程化流程。
- 专门阻断高风险泄漏路径：
  - 疾病定义变量泄漏
  - 特征血缘泄漏
  - 划分/时间污染
  - 调参与模型选择泄漏
  - 阈值与校准误用
  - 外部队列迁移鲁棒性不足
- 输出可机器校验的证据工件和发布门结果。

---

### 2. 环境要求
- Python `3.10+`
- PATH 中可用 `openssl`（执行证明必需）
- Python 包：`numpy`、`pandas`、`scikit-learn`、`joblib`
- 可选模型后端：`xgboost`、`catboost`

安装核心依赖：

```bash
python3 -m pip install -U numpy pandas scikit-learn joblib
```

环境体检：

```bash
python3 scripts/mlgg.py doctor
```

---

### 3. 新手最快上手（推荐）

#### 3.1 一条命令跑完整引导

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes
```

固定 8 步流程：
1. `doctor`
2. `init`
3. 生成离线 demo 医学数据
4. 对齐配置
5. 训练/选择/评估
6. 生成 attestation 工件
7. 严格流程首跑（`--allow-missing-compare`）
8. 严格流程基线对比复跑

#### 3.2 只看命令不执行

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode preview
```

#### 3.3 失败后继续收集完整诊断

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode auto --no-stop-on-fail
```

---

### 4. 关键产物和判读方式

重点看：
- `<project>/evidence/onboarding_report.json`
- `<project>/evidence/strict_pipeline_report.json`
- `<project>/evidence/productized_workflow_report.json`
- `<project>/evidence/user_summary.md`

`onboarding_report.json` 目前契约是 `onboarding_report.v2`：
- `status`: `pass` 或 `fail`
- `stop_on_fail`: 是否遇到失败立即停止
- `termination_reason`:
  - `completed_successfully`
  - `stopped_on_failure`
  - `completed_with_failures`
  - `cancelled_by_user`
- `failure_codes`: 汇总 gate 报告和 onboarding 步骤级失败码
- `next_actions`: 直接可执行的修复动作（通过时会给出发布级推荐基准和高级 heart 研究路径）
- `copy_ready_commands`: 可直接复制执行的命令块（`workflow_bootstrap/workflow_compare/authority_release/authority_research_heart/adversarial`）

`productized_workflow_report.json` 目前契约是 `productized_workflow_report.v2`：
- `status`: `pass` 或 `fail`
- `status_reason`:
  - `all_blocking_steps_passed`
  - `blocking_step_failed`
  - `bootstrap_recovered`
- `blocking_failure_count`: 最终仍失败的阻断步骤数量
- `recovered_failure_count`: 被标记为 `recovered` 的步骤数量
- `bootstrap_recovery_applied`: 是否触发并应用了 bootstrap 恢复
- `bootstrap_recovery_source`: 触发恢复的证据来源（或 `null`）
- `steps[]` 新增字段：
  - `status`: `pass|fail|recovered`
  - `blocking`: `true|false`
  - `recovered_by_step`: 恢复它的重试步骤名或 `null`

bootstrap 隔离规则：
- 仅本次 strict 运行中产生的证据允许触发 bootstrap 重试。
- 历史遗留的 `publication_gate_report.json` / `manifest.json` 会被忽略。

快速查看：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/tmp/mlgg_demo/evidence/onboarding_report.json")
r = json.loads(p.read_text(encoding="utf-8"))
print("status:", r["status"])
print("termination_reason:", r.get("termination_reason"))
print("failure_codes:", r.get("failure_codes", []))
print("copy_ready_commands:", sorted(r.get("copy_ready_commands", {}).keys()))
PY
```

---

### 5. 用你自己的数据（发布级手动路径）

#### 步骤 A：初始化项目

```bash
python3 scripts/mlgg.py init --project-root /tmp/mlgg_project
```

#### 步骤 B：准备数据文件

至少放置：
- `/tmp/mlgg_project/data/train.csv`
- `/tmp/mlgg_project/data/valid.csv`
- `/tmp/mlgg_project/data/test.csv`

发布级外部验证要求两类 external：
- `cross_period`
- `cross_institution`

推荐最小字段：
- `patient_id`: 患者/实体 ID
- `event_time`: 索引时间
- `y`: 二分类标签（`0/1`）
- 其余为泄漏安全特征

#### 步骤 C：先做 schema 预检

```bash
python3 scripts/mlgg.py preflight \
  --train /tmp/mlgg_project/data/train.csv \
  --valid /tmp/mlgg_project/data/valid.csv \
  --test /tmp/mlgg_project/data/test.csv \
  --target-col y \
  --patient-id-col patient_id \
  --time-col event_time \
  --mapping-out /tmp/mlgg_project/evidence/schema_mapping.json \
  --report /tmp/mlgg_project/evidence/schema_preflight_report.json
```

#### 步骤 D：训练评估

方式 1（推荐）：交互式

```bash
python3 scripts/mlgg.py train --interactive
```

方式 2：直跑命令模板（按需改路径）

```bash
python3 scripts/train_select_evaluate.py \
  --train /tmp/mlgg_project/data/train.csv \
  --valid /tmp/mlgg_project/data/valid.csv \
  --test /tmp/mlgg_project/data/test.csv \
  --target-col y \
  --patient-id-col patient_id \
  --ignore-cols patient_id,event_time \
  --performance-policy /tmp/mlgg_project/configs/performance_policy.json \
  --missingness-policy /tmp/mlgg_project/configs/missingness_policy.json \
  --feature-group-spec /tmp/mlgg_project/configs/feature_group_spec.json \
  --external-cohort-spec /tmp/mlgg_project/configs/external_cohort_spec.json \
  --model-selection-report-out /tmp/mlgg_project/evidence/model_selection_report.json \
  --evaluation-report-out /tmp/mlgg_project/evidence/evaluation_report.json \
  --prediction-trace-out /tmp/mlgg_project/evidence/prediction_trace.csv.gz \
  --external-validation-report-out /tmp/mlgg_project/evidence/external_validation_report.json \
  --feature-engineering-report-out /tmp/mlgg_project/evidence/feature_engineering_report.json \
  --distribution-report-out /tmp/mlgg_project/evidence/distribution_report.json \
  --ci-matrix-report-out /tmp/mlgg_project/evidence/ci_matrix_report.json \
  --robustness-report-out /tmp/mlgg_project/evidence/robustness_report.json \
  --seed-sensitivity-out /tmp/mlgg_project/evidence/seed_sensitivity_report.json \
  --model-out /tmp/mlgg_project/models/model.joblib \
  --permutation-null-out /tmp/mlgg_project/evidence/permutation_null_pr_auc.txt
```

#### 步骤 E：严格流程（先 bootstrap，再 compare）

首跑（生成 baseline manifest）：

```bash
python3 scripts/mlgg.py workflow \
  --request /tmp/mlgg_project/configs/request.json \
  --strict \
  --allow-missing-compare
```

复跑（与 baseline 比较）：

```bash
python3 scripts/mlgg.py workflow \
  --request /tmp/mlgg_project/configs/request.json \
  --strict \
  --compare-manifest /tmp/mlgg_project/evidence/manifest_baseline.bootstrap.json
```

---

### 6. 交互式终端向导（易用层）

支持核心命令：
- `init`
- `workflow`
- `train`
- `authority`

进入方式：

```bash
python3 scripts/mlgg.py interactive --command train
python3 scripts/mlgg.py train --interactive
python3 scripts/mlgg.py interactive --command train -- --help
```

profile 复用：

```bash
# 保存
python3 scripts/mlgg.py interactive --command train --profile-name demo --save-profile

# 加载
python3 scripts/mlgg.py interactive --command train --profile-name demo --load-profile
```

只输出命令：

```bash
python3 scripts/mlgg.py interactive --command workflow --print-only --accept-defaults
```

---

### 7. 验证与基准命令

```bash
# 统一帮助
python3 scripts/mlgg.py --help

# gate 冒烟测试
python3 scripts/test_gate_smoke.py

# onboarding 冒烟测试
python3 scripts/test_onboarding_smoke.py

# authority 基准
python3 scripts/mlgg.py authority

# authority 发布级 stress 路径（推荐封装）
python3 scripts/mlgg.py authority-release
# 等价显式命令：
python3 scripts/mlgg.py authority --include-stress-cases --stress-case-id uci-chronic-kidney-disease

# heart stress 属于高级研究/高压模式（允许失败）
python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060
# 等价显式命令：
python3 scripts/mlgg.py authority --include-stress-cases --stress-case-id uci-heart-disease --stress-seed-search --stress-seed-min 20250003 --stress-seed-max 20250060

# 对抗 fail-closed 检查
python3 scripts/mlgg.py adversarial

# 机器可解析失败输出（stderr JSON）
python3 scripts/mlgg.py authority-release --dry-run --stress-case-id uci-heart-disease --error-json
```

说明：
- 默认 stress case 是 `uci-chronic-kidney-disease`，作为稳定的发布级路径。
- `uci-heart-disease` stress-search 是高级研究型高压基准；在固定严格 floor 下，某些 seed 区间可能不存在 release-ready 候选。
- 交互式 `authority` 向导默认走 CKD 发布路径，heart 会以“高级选项”显示并提示风险。
- 自动化场景可加 `--error-json`，输出结构化失败载荷（`contract_version=mlgg_error.v1`）。
- CI 流水线：
  - `.github/workflows/ci-smoke.yml`（push/PR 快速检查）
  - `.github/workflows/ci-full.yml`（nightly/手动全量回归：authority-release + adversarial）
- `authority-release` 与 `authority-research-heart` 是固定路线封装；若传入冲突路线参数会 fail-closed 拒绝执行。

---

### 8. 新手排障入口

guided 模式取消后现在会 fail-closed，失败码为：
- `onboarding_step_cancelled`
- `authority_preset_route_override_forbidden`（固定封装命令上传入冲突路线参数）

并在 onboarding 报告中给出 `next_actions`。

高频失败码映射文档：
- `references/Troubleshooting-Top20.md`

常用诊断命令：

```bash
python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict
python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare
python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json
python3 scripts/mlgg.py authority-release --dry-run --stress-case-id uci-heart-disease --error-json
```

---

### 9. 目录结构
- `scripts/`: gate、训练器、封装器、CLI
- `references/`: schema/policy/report 示例与检查清单
- `experiments/authority-e2e/`: authority/adversarial 实验脚本
- `SKILL.md`: 完整流程契约与 gate 顺序
- `references/Beginner-Quickstart.md`: 双语新手教程
- `references/Troubleshooting-Top20.md`: 高频失败码修复手册

---

### 10. 范围说明
- 本项目是预测建模严谨性系统，不直接支持因果推断声明。
- 要宣称 publication-grade，必须严格门全部通过。
