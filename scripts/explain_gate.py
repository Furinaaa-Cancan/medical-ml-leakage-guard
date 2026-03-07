#!/usr/bin/env python3
"""
Gate Report Explainer — human-readable interpretation of gate failure codes.

Reads a gate report JSON and outputs structured explanations with
diagnostic steps and fix suggestions in English and Chinese.

Usage:
    python3 scripts/explain_gate.py --report evidence/leakage_report.json
    python3 scripts/explain_gate.py --report evidence/leakage_report.json --lang zh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Failure code → (EN explanation, ZH explanation, fix suggestion EN, fix suggestion ZH) ──
# Codes are matched by prefix to handle the 611+ unique codes efficiently.
# More specific patterns are checked first.

_EXPLANATIONS: List[Tuple[str, str, str, str, str]] = [
    # (prefix, en_explanation, zh_explanation, en_fix, zh_fix)

    # Leakage
    ("patient_overlap", "Patient IDs found in multiple splits.", "患者 ID 出现在多个分割中。",
     "Re-split data ensuring patient-level separation.", "重新按患者级别分割数据。"),
    ("row_overlap", "Duplicate rows detected across splits.", "跨分割检测到重复行。",
     "Remove duplicate rows before splitting.", "分割前移除重复行。"),
    ("temporal_overlap", "Temporal leakage: future data in training set.", "时间泄漏：训练集包含未来数据。",
     "Enforce temporal ordering in split strategy.", "在分割策略中强制时间排序。"),
    ("temporal_boundary", "Temporal boundary violated between splits.", "分割间时间边界被违反。",
     "Ensure strict chronological split boundaries.", "确保严格的时间顺序分割边界。"),
    ("test_data_usage", "Test data was used during training or tuning.", "测试数据在训练或调优中被使用。",
     "Isolate test split completely from training pipeline.", "将测试分割完全隔离于训练流程之外。"),
    ("test_split_used", "Test split used for model selection or calibration.", "测试分割被用于模型选择或校准。",
     "Use validation split for selection/calibration, never test.", "使用验证集进行选择/校准，永远不要用测试集。"),
    ("leakage", "Data leakage detected.", "检测到数据泄漏。",
     "Review split strategy and feature pipeline for information leakage.", "检查分割策略和特征流水线的信息泄漏。"),
    ("resampling_scope_leakage", "Resampling scope crosses split boundaries.", "重采样范围跨越分割边界。",
     "Ensure resampling is confined within training split.", "确保重采样限制在训练分割内。"),

    # Missing files/artifacts
    ("missing_required", "Required file or field is missing.", "缺少必需的文件或字段。",
     "Ensure all required artifacts are generated before running this gate.", "确保运行此 gate 前所有必需的产出物已生成。"),
    ("missing_split", "Split file not found.", "分割文件未找到。",
     "Run split_data.py first to generate train/valid/test CSVs.", "先运行 split_data.py 生成 train/valid/test CSV 文件。"),
    ("missing_evaluation", "Evaluation report not found.", "评估报告未找到。",
     "Run train_select_evaluate.py first.", "先运行 train_select_evaluate.py。"),
    ("missing_prediction", "Prediction trace file not found.", "预测轨迹文件未找到。",
     "Ensure --prediction-trace-out was used during training.", "确保训练时使用了 --prediction-trace-out。"),
    ("missing_", "Required artifact or field is missing.", "缺少必需的产出物或字段。",
     "Check that all upstream pipeline steps completed successfully.", "检查所有上游流水线步骤是否成功完成。"),

    # Split protocol
    ("split_seed_not_locked", "Random seed not locked in split protocol.", "分割协议中随机种子未锁定。",
     "Specify a fixed seed in the split protocol spec.", "在分割协议规格中指定固定种子。"),
    ("split_not_frozen", "Split is not frozen/immutable.", "分割未冻结/不可变。",
     "Freeze splits after generation, do not modify.", "分割生成后冻结，不要修改。"),
    ("split_", "Split-related issue detected.", "检测到分割相关问题。",
     "Review split_protocol_spec and re-run split_data.py.", "检查 split_protocol_spec 并重新运行 split_data.py。"),

    # Covariate shift
    ("covariate_shift", "Significant covariate shift between splits.", "分割间存在显著的协变量偏移。",
     "Check feature distributions; consider stratified splitting.", "检查特征分布；考虑分层分割。"),
    ("top_feature_shift", "Top features show high distributional shift.", "主要特征显示高分布偏移。",
     "Investigate shifted features for temporal or selection bias.", "调查偏移特征的时间或选择偏差。"),
    ("prevalence_shift", "Label prevalence differs across splits.", "标签流行率在分割间不同。",
     "Use stratified splitting to maintain prevalence balance.", "使用分层分割维持流行率平衡。"),

    # Model selection
    ("model_selection", "Model selection audit issue.", "模型选择审计问题。",
     "Review model_selection_report.json for candidate pool issues.", "检查 model_selection_report.json 的候选池问题。"),
    ("selected_model", "Selected model validation issue.", "所选模型验证问题。",
     "Verify the selected model exists in candidate pool.", "验证所选模型存在于候选池中。"),
    ("selection_", "Model selection process issue.", "模型选择过程问题。",
     "Review selection criteria and one-SE rule application.", "检查选择标准和 one-SE 规则的应用。"),

    # Clinical metrics
    ("clinical_", "Clinical metric threshold not met.", "临床指标阈值未达标。",
     "Adjust model or thresholds per performance policy.", "按性能策略调整模型或阈值。"),
    ("threshold_", "Threshold selection issue.", "阈值选择问题。",
     "Review threshold_policy in performance_policy.json.", "检查 performance_policy.json 中的 threshold_policy。"),

    # Calibration
    ("calibration", "Calibration issue detected.", "检测到校准问题。",
     "Try different calibration methods (sigmoid, isotonic).", "尝试不同的校准方法（sigmoid, isotonic）。"),

    # Robustness
    ("robustness_", "Robustness check failed.", "鲁棒性检查失败。",
     "Investigate metric stability across data subgroups.", "调查指标在数据子组间的稳定性。"),

    # Seed stability
    ("seed_stability", "Seed sensitivity exceeds threshold.", "种子敏感性超过阈值。",
     "Results vary too much across seeds; consider more regularization.", "结果在不同种子间变化过大；考虑更多正则化。"),
    ("seed_", "Seed-related issue.", "种子相关问题。",
     "Ensure consistent seed usage across pipeline.", "确保流水线中一致使用种子。"),

    # Distribution/generalization
    ("distribution", "Distribution shift detected.", "检测到分布偏移。",
     "Review feature distributions between internal and external cohorts.", "检查内部和外部队列间的特征分布。"),
    ("generalization", "Generalization gap exceeds threshold.", "泛化差距超过阈值。",
     "Model may be overfitting; try simpler models or more regularization.", "模型可能过拟合；尝试更简单的模型或更多正则化。"),
    ("overfit_", "Overfitting detected.", "检测到过拟合。",
     "Reduce model complexity or increase regularization.", "降低模型复杂度或增加正则化。"),

    # External validation
    ("external_", "External validation issue.", "外部验证问题。",
     "Check external cohort data quality and compatibility.", "检查外部队列数据质量和兼容性。"),

    # Permutation test
    ("permutation_", "Permutation significance test issue.", "排列显著性检验问题。",
     "Model may not be better than random; check features.", "模型可能不优于随机；检查特征。"),

    # CI matrix
    ("ci_", "Confidence interval issue.", "置信区间问题。",
     "Increase bootstrap resamples or review CI width thresholds.", "增加 bootstrap 重采样次数或检查 CI 宽度阈值。"),

    # Metric consistency
    ("metric_", "Metric consistency check failed.", "指标一致性检查失败。",
     "Verify reported metrics match computed values.", "验证报告指标与计算值匹配。"),

    # Attestation/signing
    ("signature_", "Cryptographic signature issue.", "密码签名问题。",
     "Regenerate signatures with valid keys.", "使用有效密钥重新生成签名。"),
    ("signing_", "Signing key issue.", "签名密钥问题。",
     "Check key validity, expiration, and revocation status.", "检查密钥有效性、过期和撤销状态。"),
    ("witness_", "Witness quorum issue.", "见证人法定人数问题。",
     "Ensure sufficient independent witnesses for attestation.", "确保有足够的独立见证人进行认证。"),
    ("timestamp_", "Timestamp trust issue.", "时间戳信任问题。",
     "Verify timestamp authority and ordering.", "验证时间戳权威和排序。"),
    ("transparency_", "Transparency log issue.", "透明日志问题。",
     "Check transparency log entries and signatures.", "检查透明日志条目和签名。"),

    # Tuning
    ("tuning_", "Tuning protocol issue.", "调优协议问题。",
     "Review tuning_protocol_spec for leakage-safe configuration.", "检查 tuning_protocol_spec 的无泄漏配置。"),

    # Feature engineering
    ("feature_", "Feature engineering issue.", "特征工程问题。",
     "Review feature group spec and lineage for correctness.", "检查特征组规格和血缘的正确性。"),

    # Missingness
    ("missingness_", "Missingness handling issue.", "缺失值处理问题。",
     "Review imputation strategy in missingness_policy.", "检查 missingness_policy 中的插补策略。"),

    # Imbalance
    ("imbalance", "Class imbalance handling issue.", "类别不平衡处理问题。",
     "Review imbalance_policy_spec for appropriate strategy.", "检查 imbalance_policy_spec 的适当策略。"),

    # Definition guard
    ("definition_", "Variable definition issue.", "变量定义问题。",
     "Review phenotype_definition_spec for target variable.", "检查 phenotype_definition_spec 的目标变量。"),
    ("target_", "Target variable issue.", "目标变量问题。",
     "Ensure target column is binary with no missing values.", "确保目标列是二元且无缺失值。"),

    # Publication
    ("publication_", "Publication gate requirement not met.", "出版 gate 要求未满足。",
     "Address all upstream gate failures before publication claim.", "在出版声明前解决所有上游 gate 失败。"),

    # Gate timeout
    ("gate_timeout", "Gate exceeded its configured timeout.", "Gate 超过了配置的超时时间。",
     "Increase --timeout or optimize data/model complexity.", "增加 --timeout 或优化数据/模型复杂度。"),

    # Generic fallbacks
    ("input_error", "Invalid input provided.", "提供了无效输入。",
     "Check input file paths and formats.", "检查输入文件路径和格式。"),
    ("io_error", "File I/O error.", "文件 I/O 错误。",
     "Check file permissions and disk space.", "检查文件权限和磁盘空间。"),
    ("path_not_found", "File path not found.", "文件路径未找到。",
     "Verify the file exists at the specified path.", "验证文件存在于指定路径。"),
]


def explain_code(code: str, lang: str = "en") -> Dict[str, str]:
    """Look up explanation for a failure code."""
    for prefix, en_exp, zh_exp, en_fix, zh_fix in _EXPLANATIONS:
        if code.startswith(prefix):
            if lang == "zh":
                return {"code": code, "explanation": zh_exp, "fix": zh_fix}
            return {"code": code, "explanation": en_exp, "fix": en_fix}
    if lang == "zh":
        return {"code": code, "explanation": f"未知故障代码: {code}", "fix": "查看对应 gate 脚本源码了解详情。"}
    return {"code": code, "explanation": f"Unknown failure code: {code}", "fix": "Review the gate script source for details."}


def explain_report(report: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
    """Explain all issues in a gate report."""
    status = report.get("status", "unknown")
    gate = report.get("gate", report.get("gate_name", "unknown"))
    issues = report.get("issues", [])

    explained = []
    for issue in issues:
        if isinstance(issue, dict):
            code = str(issue.get("code", ""))
            message = str(issue.get("message", ""))
            exp = explain_code(code, lang)
            exp["original_message"] = message
            explained.append(exp)

    return {
        "gate": gate,
        "status": status,
        "issue_count": len(explained),
        "explanations": explained,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Explain gate report failure codes.")
    parser.add_argument("--report", required=True, help="Path to gate report JSON.")
    parser.add_argument("--lang", default="en", choices=["en", "zh"], help="Output language.")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    path = Path(args.report).expanduser().resolve()
    if not path.exists():
        print(f"Report not found: {path}", file=sys.stderr)
        return 1

    try:
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in report: {exc}", file=sys.stderr)
        return 1

    if not isinstance(report, dict):
        print("Report is not a JSON object.", file=sys.stderr)
        return 1

    result = explain_report(report, args.lang)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Gate: {result['gate']}")
        print(f"Status: {result['status']}")
        print(f"Issues: {result['issue_count']}")
        print()
        for i, exp in enumerate(result["explanations"], 1):
            print(f"  [{i}] {exp['code']}")
            print(f"      {exp['explanation']}")
            print(f"      Original: {exp['original_message']}")
            print(f"      Fix: {exp['fix']}")
            print()

    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
