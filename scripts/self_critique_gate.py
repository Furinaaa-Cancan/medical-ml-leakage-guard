#!/usr/bin/env python3
"""
Self-critique gate for publication-grade medical prediction workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from _gate_utils import add_issue, load_json_from_str as load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-critique scoring on evidence artifacts.")
    parser.add_argument("--request-report", required=True, help="Path to request contract report JSON.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument(
        "--execution-attestation-report",
        required=True,
        help="Path to execution attestation report JSON.",
    )
    parser.add_argument(
        "--reporting-bias-report",
        required=True,
        help="Path to reporting/bias checklist report JSON.",
    )
    parser.add_argument("--leakage-report", required=True, help="Path to leakage report JSON.")
    parser.add_argument("--split-protocol-report", required=True, help="Path to split protocol report JSON.")
    parser.add_argument("--covariate-shift-report", required=True, help="Path to covariate-shift report JSON.")
    parser.add_argument("--definition-report", required=True, help="Path to definition guard report JSON.")
    parser.add_argument("--lineage-report", required=True, help="Path to lineage gate report JSON.")
    parser.add_argument("--imbalance-report", required=True, help="Path to imbalance policy report JSON.")
    parser.add_argument("--missingness-report", required=True, help="Path to missingness policy report JSON.")
    parser.add_argument("--tuning-report", required=True, help="Path to tuning leakage report JSON.")
    parser.add_argument("--model-selection-audit-report", required=True, help="Path to model selection audit report JSON.")
    parser.add_argument("--feature-engineering-audit-report", required=True, help="Path to feature engineering audit report JSON.")
    parser.add_argument("--clinical-metrics-report", required=True, help="Path to clinical metrics report JSON.")
    parser.add_argument("--prediction-replay-report", required=True, help="Path to prediction replay gate report JSON.")
    parser.add_argument("--distribution-generalization-report", required=True, help="Path to distribution generalization report JSON.")
    parser.add_argument("--generalization-gap-report", required=True, help="Path to generalization gap report JSON.")
    parser.add_argument("--robustness-report", required=True, help="Path to robustness report JSON.")
    parser.add_argument("--seed-stability-report", required=True, help="Path to seed stability report JSON.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external validation gate report JSON.")
    parser.add_argument("--calibration-dca-report", required=True, help="Path to calibration/DCA gate report JSON.")
    parser.add_argument("--ci-matrix-report", required=True, help="Path to CI matrix gate report JSON.")
    parser.add_argument("--metric-report", required=True, help="Path to metric consistency report JSON.")
    parser.add_argument("--evaluation-quality-report", required=True, help="Path to evaluation quality report JSON.")
    parser.add_argument("--permutation-report", required=True, help="Path to permutation report JSON.")
    parser.add_argument("--publication-report", required=True, help="Path to publication gate report JSON.")
    parser.add_argument("--min-score", type=float, default=95.0, help="Minimum score for publication-grade readiness.")
    parser.add_argument(
        "--allow-missing-comparison",
        action="store_true",
        help="Allow missing manifest comparison section without strict failure.",
    )
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail when warnings exist or score below threshold.")
    return parser.parse_args()


def score_component(report: Dict[str, Any], hard_weight: float, warn_penalty: float) -> float:
    status = str(report.get("status", "")).lower()
    failure_count = int(report.get("failure_count", 0) or 0)
    warning_count = int(report.get("warning_count", 0) or 0)
    base = hard_weight if status == "pass" and failure_count == 0 else 0.0
    penalty = warn_penalty * float(max(warning_count, 0))
    return max(base - penalty, 0.0)


def summarize_recommendations(issues: List[Dict[str, Any]]) -> List[str]:
    recs: List[str] = []
    codes = {i["code"] for i in issues}
    components = {
        str(i.get("details", {}).get("component"))
        for i in issues
        if isinstance(i.get("details"), dict) and i.get("details", {}).get("component") is not None
    }
    if "component_not_passed" in codes or "component_has_failures" in codes:
        recs.append("Resolve all failed component gates before interpreting model metrics.")
    if "execution_attestation_report" in components:
        recs.append("Regenerate signed execution attestation and verify detached signature against public key.")
    if "reporting_bias_report" in components:
        recs.append("Complete TRIPOD+AI/PROBAST+AI/STARD-AI checklist items and rerun reporting_bias_gate.")
    if "model_selection_audit_report" in components:
        recs.append("Repair model-selection evidence (candidate pool, one-SE replay, and test-isolation proof).")
    if "feature_engineering_audit_report" in components:
        recs.append("Repair feature-engineering provenance (explicit groups, train-only scope, stability evidence, reproducibility fields).")
    if "clinical_metrics_report" in components:
        recs.append("Regenerate split metrics with full clinical panel and confusion-matrix-consistent values.")
    if "prediction_replay_report" in components:
        recs.append("Rebuild prediction_trace from model outputs and verify replayed metrics/hash alignment.")
    if "distribution_generalization_report" in components:
        recs.append("Address distribution shift and split separability before asserting transport-ready generalization.")
    if "generalization_gap_report" in components:
        recs.append("Reduce overfitting gap or tighten regularization before publication-grade claims.")
    if "robustness_report" in components:
        recs.append("Investigate temporal/group robustness failures before publication-grade claims.")
    if "seed_stability_report" in components:
        recs.append("Improve multi-seed robustness or simplify model/hyperparameters to reduce instability.")
    if "external_validation_report" in components:
        recs.append("Strengthen external cohort transport performance and ensure cross-period/cross-institution coverage.")
    if "calibration_dca_report" in components:
        recs.append("Improve probability calibration and decision-curve net benefit on test and external cohorts.")
    if "ci_matrix_report" in components:
        recs.append("Increase bootstrap precision and complete full-matrix CI evidence (including transport-drop CI).")
    if "component_not_strict" in codes:
        recs.append("Regenerate component reports with --strict for publication-grade claims.")
    if "evaluation_quality_report" in components:
        recs.append("Provide valid primary-metric CI + baseline comparison in evaluation artifact and rerun evaluation_quality_gate.")
    if "insufficient_quality_score" in codes:
        recs.append("Increase robustness evidence and reduce warnings to lift quality score.")
    if "manifest_not_comparable" in codes:
        recs.append("Provide baseline manifest comparison to satisfy strict reproducibility gate.")
    if "publication_claim_without_repro_comparison" in codes:
        recs.append("Do not claim publication-grade readiness until manifest baseline comparison is present.")
    if not recs:
        recs.append("No blocking critique findings detected.")
    return recs


def warning_is_blocking(issue: Dict[str, Any], args: argparse.Namespace) -> bool:
    code = str(issue.get("code", ""))
    if args.strict and args.allow_missing_comparison and code == "manifest_not_comparable":
        return False
    return True


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    artifact_paths = {
        "request_report": args.request_report,
        "manifest": args.manifest,
        "execution_attestation_report": args.execution_attestation_report,
        "reporting_bias_report": args.reporting_bias_report,
        "leakage_report": args.leakage_report,
        "split_protocol_report": args.split_protocol_report,
        "covariate_shift_report": args.covariate_shift_report,
        "definition_report": args.definition_report,
        "lineage_report": args.lineage_report,
        "imbalance_report": args.imbalance_report,
        "missingness_report": args.missingness_report,
        "tuning_report": args.tuning_report,
        "model_selection_audit_report": args.model_selection_audit_report,
        "feature_engineering_audit_report": args.feature_engineering_audit_report,
        "clinical_metrics_report": args.clinical_metrics_report,
        "prediction_replay_report": args.prediction_replay_report,
        "distribution_generalization_report": args.distribution_generalization_report,
        "generalization_gap_report": args.generalization_gap_report,
        "robustness_report": args.robustness_report,
        "seed_stability_report": args.seed_stability_report,
        "external_validation_report": args.external_validation_report,
        "calibration_dca_report": args.calibration_dca_report,
        "ci_matrix_report": args.ci_matrix_report,
        "metric_report": args.metric_report,
        "evaluation_quality_report": args.evaluation_quality_report,
        "permutation_report": args.permutation_report,
        "publication_report": args.publication_report,
    }

    loaded: Dict[str, Dict[str, Any]] = {}
    for name, path in artifact_paths.items():
        try:
            loaded[name] = load_json(path)
        except Exception as exc:
            add_issue(
                failures,
                "missing_or_invalid_artifact",
                "Failed to load required artifact.",
                {"artifact": name, "path": str(Path(path).expanduser()), "error": str(exc)},
            )

    requested_claim_tier: str = ""
    request_report = loaded.get("request_report")
    if isinstance(request_report, dict):
        normalized_request = request_report.get("normalized_request")
        if isinstance(normalized_request, dict):
            claim_tier_value = normalized_request.get("claim_tier_target")
            if isinstance(claim_tier_value, str):
                requested_claim_tier = claim_tier_value.strip()

    def require_pass(name: str, strict_mode_required: bool = True) -> None:
        report = loaded.get(name)
        if report is None:
            return

        status = str(report.get("status", "")).lower()
        if status != "pass":
            add_issue(
                failures,
                "component_not_passed",
                "Component report status is not pass.",
                {"component": name, "status": report.get("status")},
            )

        failure_count = int(report.get("failure_count", 0) or 0)
        if failure_count > 0:
            add_issue(
                failures,
                "component_has_failures",
                "Component report contains failures.",
                {"component": name, "failure_count": failure_count},
            )

        if strict_mode_required and report.get("strict_mode") is not True:
            target_bucket = failures if args.strict else warnings
            add_issue(
                target_bucket,
                "component_not_strict",
                "Component report was not generated in strict mode.",
                {"component": name},
            )

    strict_optional_components = set()
    for component in (
        "request_report",
        "execution_attestation_report",
        "reporting_bias_report",
        "leakage_report",
        "split_protocol_report",
        "covariate_shift_report",
        "definition_report",
        "lineage_report",
        "imbalance_report",
        "missingness_report",
        "tuning_report",
        "model_selection_audit_report",
        "feature_engineering_audit_report",
        "clinical_metrics_report",
        "prediction_replay_report",
        "distribution_generalization_report",
        "generalization_gap_report",
        "robustness_report",
        "seed_stability_report",
        "external_validation_report",
        "calibration_dca_report",
        "ci_matrix_report",
        "metric_report",
        "evaluation_quality_report",
        "permutation_report",
        "publication_report",
    ):
        require_pass(component, strict_mode_required=(component not in strict_optional_components))

    manifest = loaded.get("manifest")
    reproducibility_comparison_evaluated = False
    if manifest is not None:
        if str(manifest.get("status", "")).lower() != "pass":
            add_issue(
                failures,
                "manifest_not_passed",
                "Manifest status is not pass.",
                {"status": manifest.get("status")},
            )

        files = manifest.get("files")
        if not isinstance(files, list) or len(files) < 2:
            add_issue(
                failures,
                "manifest_insufficient_files",
                "Manifest must lock at least two files (data + config/spec).",
                {"file_count": len(files) if isinstance(files, list) else None},
            )

        comparison = manifest.get("comparison")
        if isinstance(comparison, dict):
            reproducibility_comparison_evaluated = True
            if comparison.get("matched") is False:
                add_issue(
                    failures,
                    "manifest_comparison_mismatch",
                    "Manifest comparison indicates differences from baseline.",
                    {
                        "missing_in_current": comparison.get("missing_in_current", []),
                        "missing_in_baseline": comparison.get("missing_in_baseline", []),
                        "hash_mismatches": comparison.get("hash_mismatches", []),
                    },
                )
        else:
            reproducibility_comparison_evaluated = False

    if manifest is not None and reproducibility_comparison_evaluated is False:
        add_issue(
            failures if (args.strict and not args.allow_missing_comparison) else warnings,
            "manifest_not_comparable",
            "Manifest has no comparison section; rerun consistency not evaluated.",
            {},
        )

    if args.strict and requested_claim_tier == "publication-grade" and reproducibility_comparison_evaluated is False:
        add_issue(
            failures,
            "publication_claim_without_repro_comparison",
            "Publication-grade readiness requires baseline manifest comparison; bootstrap mode cannot be claim-ready.",
            {"allow_missing_comparison": bool(args.allow_missing_comparison)},
        )

    # Weighted score emphasizes phenotype integrity and lineage coverage.
    _raw_weights = {
        "request_report": 7.0,
        "manifest": 10.0,
        "execution_attestation_report": 8.0,
        "reporting_bias_report": 8.0,
        "leakage_report": 13.0,
        "split_protocol_report": 8.0,
        "covariate_shift_report": 7.0,
        "definition_report": 13.0,
        "lineage_report": 11.0,
        "imbalance_report": 8.0,
        "missingness_report": 8.0,
        "tuning_report": 8.0,
        "model_selection_audit_report": 8.0,
        "feature_engineering_audit_report": 8.0,
        "clinical_metrics_report": 8.0,
        "prediction_replay_report": 8.0,
        "distribution_generalization_report": 8.0,
        "generalization_gap_report": 8.0,
        "robustness_report": 8.0,
        "seed_stability_report": 7.0,
        "external_validation_report": 8.0,
        "calibration_dca_report": 8.0,
        "ci_matrix_report": 8.0,
        "metric_report": 7.0,
        "evaluation_quality_report": 8.0,
        "permutation_report": 7.0,
        "publication_report": 8.0,
    }
    _total_raw = sum(_raw_weights.values())
    weights = {k: v * 100.0 / _total_raw for k, v in _raw_weights.items()}
    warn_penalty = 1.0
    quality_score = 0.0
    for key, weight in weights.items():
        report = loaded.get(key)
        if report is None:
            continue
        if key == "manifest":
            comparison = report.get("comparison")
            comparable_bonus = weight if str(report.get("status", "")).lower() == "pass" else 0.0
            if isinstance(comparison, dict) and comparison.get("matched") is True:
                quality_score += comparable_bonus
            else:
                quality_score += max(comparable_bonus - 5.0, 0.0)
            continue
        quality_score += score_component(report, hard_weight=weight, warn_penalty=warn_penalty)

    quality_score = float(max(min(quality_score, 100.0), 0.0))
    if quality_score < args.min_score:
        add_issue(
            failures if args.strict else warnings,
            "insufficient_quality_score",
            "Overall quality score below threshold.",
            {"score": quality_score, "min_score": args.min_score},
        )

    blocking_warning_count = (
        sum(1 for issue in warnings if warning_is_blocking(issue, args))
        if args.strict
        else 0
    )
    should_fail = bool(failures) or (args.strict and blocking_warning_count > 0)
    claim_tier = "publication-grade-ready" if (not should_fail and quality_score >= args.min_score) else "not-ready"
    decision = "pass" if not should_fail else "fail"

    report = {
        "status": decision,
        "strict_mode": bool(args.strict),
        "quality_score": round(quality_score, 2),
        "min_score": args.min_score,
        "claim_tier_decision": claim_tier,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "blocking_warning_count": blocking_warning_count,
        "failures": failures,
        "warnings": warnings,
        "recommendations": summarize_recommendations(failures + warnings),
        "reproducibility_comparison_evaluated": reproducibility_comparison_evaluated,
        "artifacts": {
            key: {
                "path": str(Path(path).expanduser().resolve()),
                "loaded": key in loaded,
                "status": loaded.get(key, {}).get("status"),
            }
            for key, path in artifact_paths.items()
        },
    }

    if args.report:
        from _gate_utils import write_json as _write_report
        _write_report(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(
        f"QualityScore: {report['quality_score']:.2f} | MinScore: {args.min_score:.2f} | "
        f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}"
    )
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
