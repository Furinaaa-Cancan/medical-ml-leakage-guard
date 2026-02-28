# Beginner Quickstart / 新手快速开始

This guide gives a reproducible first run for users who are not familiar with the full gate stack.

本指南给新手提供可复现的首跑路径，不需要先理解全部 gate 细节。

---

## 1. Prerequisites / 环境前置

English:
- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `joblib`
- `openssl` in PATH (required for attestation generation)

中文：
- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `joblib`
- PATH 中可用 `openssl`（生成 attestation 必需）

Check environment:

```bash
python3 scripts/mlgg.py doctor
```

---

## 2. Fastest First Run (Recommended) / 推荐首跑（最快）

English:
- Run onboarding once; it will execute an 8-step strict flow with offline synthetic medical data.

中文：
- 直接跑 onboarding；它会使用离线合成医学数据执行固定 8 步严格流程。

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode guided --yes
```

Need full diagnostics without early stop:

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode auto --no-stop-on-fail
```

Expected key outputs:
- `/tmp/mlgg_demo/evidence/onboarding_report.json`
- `/tmp/mlgg_demo/evidence/user_summary.md`
- `/tmp/mlgg_demo/evidence/strict_pipeline_report.json`

---

## 3. Preview-Only Mode / 仅预览命令模式

English:
- If you want to inspect commands without running anything:

中文：
- 如果只想看完整命令而不执行：

```bash
python3 scripts/mlgg.py onboarding --project-root /tmp/mlgg_demo --mode preview
```

Preview semantics:
- `onboarding_report.json` records `preview_only=true` and `display_status=preview`.
- No training/gate execution happens in preview mode.

预览语义：
- `onboarding_report.json` 会记录 `preview_only=true` 与 `display_status=preview`。
- preview 模式不会执行训练和 gate。

---

## 4. Manual Advanced Path / 手动进阶路径

English:
1. Initialize project template.
2. Train/evaluate and generate evidence.
3. Bootstrap manifest baseline.
4. Re-run against baseline manifest.

中文：
1. 初始化项目模板。
2. 训练评估并生成证据工件。
3. 首跑 bootstrap 生成 baseline manifest。
4. 使用 baseline manifest 复跑对比。

```bash
python3 scripts/mlgg.py init --project-root /tmp/mlgg_demo
python3 scripts/mlgg.py train --interactive
python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --allow-missing-compare
python3 scripts/mlgg.py workflow --request /tmp/mlgg_demo/configs/request.json --strict --compare-manifest /tmp/mlgg_demo/evidence/manifest_baseline.bootstrap.json
```

---

## 5. How To Read Result Status / 如何判断结果是否通过

English:
- `strict_pipeline_report.json`:
  - `status=pass`: all strict gates passed.
  - `status=fail`: at least one hard gate blocked release.
- `onboarding_report.json`:
  - Contract `onboarding_report.v2`.
  - Step-level command, exit code, and error tails.
  - `display_status=preview` for preview mode.
  - `preview_only=true` indicates no execution.
  - `stop_on_fail` and `termination_reason` show run-stop semantics.
  - `failure_codes` and `next_actions` provide fail-closed diagnosis guidance.
  - `copy_ready_commands` gives direct rerun/benchmark commands with absolute `mlgg.py` path.
- `user_summary.md`:
  - Human-readable pass/fail summary and key evidence links.

中文：
- `strict_pipeline_report.json`：
  - `status=pass`：所有严格门通过。
  - `status=fail`：至少一个硬门阻断发布。
- `onboarding_report.json`：
  - 契约版本为 `onboarding_report.v2`。
  - 包含逐步命令、退出码和错误尾部信息。
  - preview 模式下 `display_status=preview`。
  - `preview_only=true` 表示仅预览未执行。
  - `stop_on_fail` 与 `termination_reason` 反映终止语义。
  - `failure_codes` 与 `next_actions` 提供 fail-closed 诊断动作。
  - `copy_ready_commands` 提供带绝对 `mlgg.py` 路径的复跑/基准命令。
- `user_summary.md`：
  - 人类可读的通过/失败摘要与关键证据路径。

---

## 6. Common Next Commands / 常用后续命令

```bash
# Unified CLI help
python3 scripts/mlgg.py --help
python3 scripts/mlgg.py onboarding --help
python3 scripts/mlgg.py train --interactive --help

# Recommended release-grade authority benchmark
python3 scripts/mlgg.py authority-release

# Advanced heart research/high-pressure benchmark
python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060

# Interactive wizard (init/workflow/train/authority)
python3 scripts/mlgg.py interactive --command train

# Gate smoke tests
python3 scripts/test_gate_smoke.py

# Onboarding smoke tests
python3 scripts/test_onboarding_smoke.py
```

Notes:
- `authority-release` and `authority-research-heart` are fixed-route wrappers; conflicting route flags are rejected fail-closed.
- `authority-research-heart` is advanced mode and may fail by design under strict fixed floors.
- If guided onboarding runs in non-interactive shell, use `--yes` or `--mode auto`; otherwise it fails closed with `onboarding_interactive_input_unavailable`.

说明：
- `authority-release` 与 `authority-research-heart` 是固定路线封装；冲突路线参数会被 fail-closed 拒绝。
- `authority-research-heart` 是高级研究模式，在固定严格阈值下可能按设计失败。
- 若 guided onboarding 在无交互 shell 下运行，请加 `--yes` 或改用 `--mode auto`；否则会以 `onboarding_interactive_input_unavailable` fail-closed。

---

## 7. Troubleshooting Entry / 故障入口

English:
- Use failure codes from `strict_pipeline_report.json` and map them in:
  - `references/Troubleshooting-Top20.md`

中文：
- 从 `strict_pipeline_report.json` 读取 failure code，再到下列文档定位修复动作：
  - `references/Troubleshooting-Top20.md`
