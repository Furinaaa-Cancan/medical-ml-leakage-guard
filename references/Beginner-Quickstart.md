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
  - Step-level command, exit code, and error tails.
- `user_summary.md`:
  - Human-readable pass/fail summary and key evidence links.

中文：
- `strict_pipeline_report.json`：
  - `status=pass`：所有严格门通过。
  - `status=fail`：至少一个硬门阻断发布。
- `onboarding_report.json`：
  - 包含逐步命令、退出码和错误尾部信息。
- `user_summary.md`：
  - 人类可读的通过/失败摘要与关键证据路径。

---

## 6. Common Next Commands / 常用后续命令

```bash
# Unified CLI help
python3 scripts/mlgg.py --help

# Interactive wizard (init/workflow/train/authority)
python3 scripts/mlgg.py interactive --command train

# Gate smoke tests
python3 scripts/test_gate_smoke.py

# Onboarding smoke tests
python3 scripts/test_onboarding_smoke.py
```

---

## 7. Troubleshooting Entry / 故障入口

English:
- Use failure codes from `strict_pipeline_report.json` and map them in:
  - `references/Troubleshooting-Top20.md`

中文：
- 从 `strict_pipeline_report.json` 读取 failure code，再到下列文档定位修复动作：
  - `references/Troubleshooting-Top20.md`
