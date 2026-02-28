#!/usr/bin/env python3
"""
Aggregate fail-closed publication gate for medical prediction evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_utils import add_issue, load_json_from_str as load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run aggregate publication-grade evidence gate.")
    parser.add_argument("--request-report", required=True, help="Path to request contract report JSON.")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument(
        "--execution-attestation-report",
        required=True,
        help="Path to execution attestation gate report JSON.",
    )
    parser.add_argument(
        "--reporting-bias-report",
        required=True,
        help="Path to reporting/bias checklist gate report JSON.",
    )
    parser.add_argument("--leakage-report", required=True, help="Path to leakage report JSON.")
    parser.add_argument("--split-protocol-report", required=True, help="Path to split protocol gate report JSON.")
    parser.add_argument(
        "--covariate-shift-report",
        required=True,
        help="Path to covariate-shift gate report JSON.",
    )
    parser.add_argument("--definition-report", required=True, help="Path to definition-variable guard report JSON.")
    parser.add_argument("--lineage-report", required=True, help="Path to lineage gate report JSON.")
    parser.add_argument("--imbalance-report", required=True, help="Path to imbalance policy gate report JSON.")
    parser.add_argument("--missingness-report", required=True, help="Path to missingness policy gate report JSON.")
    parser.add_argument("--tuning-report", required=True, help="Path to tuning leakage gate report JSON.")
    parser.add_argument("--model-selection-audit-report", required=True, help="Path to model selection audit report JSON.")
    parser.add_argument("--feature-engineering-audit-report", required=True, help="Path to feature engineering audit report JSON.")
    parser.add_argument("--clinical-metrics-report", required=True, help="Path to clinical metrics gate report JSON.")
    parser.add_argument("--prediction-replay-report", required=True, help="Path to prediction replay gate report JSON.")
    parser.add_argument("--distribution-generalization-report", required=True, help="Path to distribution generalization gate report JSON.")
    parser.add_argument("--generalization-gap-report", required=True, help="Path to generalization gap gate report JSON.")
    parser.add_argument("--robustness-report", required=True, help="Path to robustness gate report JSON.")
    parser.add_argument("--seed-stability-report", required=True, help="Path to seed stability gate report JSON.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external validation gate report JSON.")
    parser.add_argument("--calibration-dca-report", required=True, help="Path to calibration/DCA gate report JSON.")
    parser.add_argument("--ci-matrix-report", required=True, help="Path to CI matrix gate report JSON.")
    parser.add_argument("--metric-report", required=True, help="Path to metric consistency report JSON.")
    parser.add_argument("--evaluation-quality-report", required=True, help="Path to evaluation quality gate report JSON.")
    parser.add_argument("--permutation-report", required=True, help="Path to permutation gate report JSON.")
    parser.add_argument("--report", help="Optional output publication gate report path.")
    parser.add_argument("--strict", action="store_true", help="Require strict-mode component reports.")
    return parser.parse_args()


def validate_component_status(
    name: str,
    report: Optional[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    strict: bool,
    strict_mode_required: bool = True,
) -> None:
    if report is None:
        add_issue(failures, "missing_component_report", "Missing required component report.", {"component": name})
        return

    status = str(report.get("status", "")).lower()
    if status != "pass":
        add_issue(
            failures,
            "component_not_passed",
            "Required component report did not pass.",
            {"component": name, "status": report.get("status")},
        )

    if strict_mode_required:
        if strict and report.get("strict_mode") is not True:
            add_issue(
                failures,
                "component_not_strict",
                "Required component report was not generated in strict mode.",
                {"component": name},
            )
        elif report.get("strict_mode") is not True:
            add_issue(
                warnings,
                "component_not_strict",
                "Component report was not generated in strict mode.",
                {"component": name},
            )

    failure_count = report.get("failure_count")
    if isinstance(failure_count, int) and failure_count > 0:
        add_issue(
            failures,
            "component_has_failures",
            "Component report contains failures.",
            {"component": name, "failure_count": failure_count},
        )


def parse_int_like(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def enforce_execution_attestation_publication_contract(
    execution_attestation_report: Optional[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    if not isinstance(execution_attestation_report, dict):
        add_issue(
            failures,
            "execution_attestation_summary_missing",
            "Execution attestation report is missing or invalid.",
            {},
        )
        return

    summary = execution_attestation_report.get("summary")
    if not isinstance(summary, dict):
        add_issue(
            failures,
            "execution_attestation_summary_missing",
            "Execution attestation report must contain summary object.",
            {},
        )
        return

    key_assurance = summary.get("key_assurance")
    policy = key_assurance.get("policy") if isinstance(key_assurance, dict) else None
    if not isinstance(policy, dict):
        add_issue(
            failures,
            "execution_attestation_policy_missing",
            "Execution attestation summary must include key_assurance.policy.",
            {},
        )
        return

    required_true_flags = (
        "require_revocation_list",
        "require_timestamp_trust",
        "require_transparency_log",
        "require_transparency_log_signature",
        "require_execution_receipt",
        "require_execution_log_attestation",
        "require_independent_timestamp_authority",
        "require_independent_execution_authority",
        "require_independent_log_authority",
        "require_distinct_authority_roles",
        "require_witness_quorum",
        "require_independent_witness_keys",
        "require_witness_independence_from_signing",
    )
    for field in required_true_flags:
        if policy.get(field) is not True:
            add_issue(
                failures,
                "execution_attestation_policy_disabled",
                "Publication gate requires execution attestation policy flag to be true.",
                {"field": field, "value": policy.get(field)},
            )

    min_witness_count = parse_int_like(policy.get("min_witness_count"))
    if min_witness_count is None:
        add_issue(
            failures,
            "execution_attestation_min_witness_count_invalid",
            "Publication gate requires numeric key_assurance.policy.min_witness_count.",
            {"value": policy.get("min_witness_count")},
        )
        min_witness_count = 0
    elif min_witness_count < 2:
        add_issue(
            failures,
            "execution_attestation_min_witness_count_too_low",
            "Publication gate requires key_assurance.policy.min_witness_count >= 2.",
            {"min_witness_count": min_witness_count},
        )

    for block_name in ("timestamp_trust", "transparency_log", "execution_receipt", "execution_log_attestation"):
        block = summary.get(block_name)
        if not isinstance(block, dict) or block.get("present") is not True:
            add_issue(
                failures,
                "execution_attestation_required_block_missing",
                "Publication gate requires execution attestation block to be present.",
                {"block": block_name, "present": block.get("present") if isinstance(block, dict) else None},
            )

    witness_quorum = summary.get("witness_quorum")
    if not isinstance(witness_quorum, dict):
        add_issue(
            failures,
            "execution_attestation_witness_summary_missing",
            "Publication gate requires witness_quorum summary block.",
            {},
        )
        return

    if witness_quorum.get("present") is not True:
        add_issue(
            failures,
            "execution_attestation_witness_not_present",
            "Publication gate requires witness quorum block to be present.",
            {"present": witness_quorum.get("present")},
        )
    if witness_quorum.get("required") is not True:
        add_issue(
            failures,
            "execution_attestation_witness_not_required",
            "Publication gate requires witness quorum to be required.",
            {"required": witness_quorum.get("required")},
        )
    validated_witnesses = parse_int_like(witness_quorum.get("validated_witnesses"))
    reported_min_count = parse_int_like(witness_quorum.get("min_witness_count"))
    if validated_witnesses is None:
        add_issue(
            failures,
            "execution_attestation_validated_witnesses_invalid",
            "Publication gate requires numeric witness_quorum.validated_witnesses.",
            {"value": witness_quorum.get("validated_witnesses")},
        )
        validated_witnesses = 0
    if reported_min_count is None:
        add_issue(
            failures,
            "execution_attestation_witness_min_count_invalid",
            "Publication gate requires numeric witness_quorum.min_witness_count.",
            {"value": witness_quorum.get("min_witness_count")},
        )
        reported_min_count = 0
    if validated_witnesses < max(min_witness_count, reported_min_count):
        add_issue(
            failures,
            "execution_attestation_witness_quorum_not_met",
            "Publication gate requires validated witness count to meet quorum minimum.",
            {
                "validated_witnesses": validated_witnesses,
                "policy_min_witness_count": min_witness_count,
                "reported_min_witness_count": reported_min_count,
            },
        )

    role_distinctness = summary.get("authority_role_distinctness")
    if not isinstance(role_distinctness, dict):
        add_issue(
            failures,
            "execution_attestation_role_distinctness_missing",
            "Publication gate requires authority_role_distinctness summary block.",
            {},
        )
        return
    if role_distinctness.get("enforced") is not True:
        add_issue(
            failures,
            "execution_attestation_role_distinctness_not_enforced",
            "Publication gate requires cross-role authority distinctness enforcement.",
            {"enforced": role_distinctness.get("enforced")},
        )
    if str(role_distinctness.get("status", "")).lower() != "pass":
        add_issue(
            failures,
            "execution_attestation_role_distinctness_failed",
            "Publication gate requires authority_role_distinctness status to be pass.",
            {"status": role_distinctness.get("status")},
        )


def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    loaded: Dict[str, Dict[str, Any]] = {}

    files = {
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
    }

    for name, path in files.items():
        try:
            loaded[name] = load_json(path)
        except Exception as exc:
            add_issue(
                failures,
                "invalid_or_missing_json",
                "Failed to load required JSON artifact.",
                {"artifact": name, "path": str(Path(path).expanduser()), "error": str(exc)},
            )

    manifest = loaded.get("manifest")
    if manifest is not None:
        if str(manifest.get("status", "")).lower() != "pass":
            add_issue(
                failures,
                "manifest_not_passed",
                "Manifest status is not pass.",
                {"status": manifest.get("status")},
            )

        files_meta = manifest.get("files")
        if not isinstance(files_meta, list) or not files_meta:
            add_issue(
                failures,
                "manifest_missing_files",
                "Manifest does not contain locked file entries.",
                {},
            )

        errors = manifest.get("errors", [])
        if isinstance(errors, list) and errors:
            add_issue(
                failures,
                "manifest_has_errors",
                "Manifest contains errors.",
                {"error_count": len(errors)},
            )

        comparison = manifest.get("comparison")
        if not isinstance(comparison, dict):
            add_issue(
                failures,
                "manifest_comparison_missing",
                "Publication gate requires manifest baseline comparison result.",
                {},
            )
        else:
            if comparison.get("matched") is not True:
                add_issue(
                    failures,
                    "manifest_comparison_mismatch",
                    "Manifest comparison against baseline did not match.",
                    {
                        "matched": comparison.get("matched"),
                        "missing_in_current": comparison.get("missing_in_current", []),
                        "missing_in_baseline": comparison.get("missing_in_baseline", []),
                        "hash_mismatches": comparison.get("hash_mismatches", []),
                    },
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
    ):
        validate_component_status(
            component,
            loaded.get(component),
            failures,
            warnings,
            args.strict,
            strict_mode_required=(component not in strict_optional_components),
        )

    enforce_execution_attestation_publication_contract(
        execution_attestation_report=loaded.get("execution_attestation_report"),
        failures=failures,
    )

    metric_report = loaded.get("metric_report")
    if isinstance(metric_report, dict):
        actual_metric = metric_report.get("actual_metric")
        if not (
            isinstance(actual_metric, (int, float))
            and not isinstance(actual_metric, bool)
            and math.isfinite(float(actual_metric))
        ):
            add_issue(
                failures,
                "metric_report_missing_actual",
                "Metric consistency report must contain finite numeric actual_metric.",
                {"actual_metric_type": type(actual_metric).__name__ if actual_metric is not None else None},
            )

    should_fail = bool(failures) or (args.strict and bool(warnings))
    quality_score = max(0.0, min(100.0, 100.0 - 20.0 * len(failures) - 2.5 * len(warnings)))

    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "quality_score": round(quality_score, 2),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "artifacts": {
                name: {
                    "path": str(Path(path).expanduser().resolve()),
                    "loaded": name in loaded,
                    "status": loaded.get(name, {}).get("status"),
                }
                for name, path in files.items()
            }
        },
    }

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
