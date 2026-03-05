#!/usr/bin/env python3
"""
Fail-closed reporting and bias checklist gate for publication-grade medical AI prediction.

This gate enforces machine-checkable checklist completion aligned with:
- TRIPOD+AI reporting expectations
- PROBAST+AI risk-of-bias domains
- STARD-AI diagnostic reporting items (when applicable)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_utils import add_issue, load_json_from_str as load_json
from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)


register_remediations({
    "invalid_checklist_spec": "Fix JSON syntax in the checklist spec file. Use a JSON linter.",
    "missing_tripod_ai": "Add a 'tripod_ai' object to the checklist spec with all required TRIPOD+AI items set to true.",
    "tripod_ai_item_not_true": "Set the missing TRIPOD+AI checklist item to true in the checklist spec.",
    "missing_probast_ai": "Add a 'probast_ai' object to the checklist spec with all required PROBAST+AI domains set to true.",
    "probast_ai_item_not_true": "Set the missing PROBAST+AI domain to true in the checklist spec.",
    "stard_ai_item_not_true": "Set the missing STARD-AI item to true in the checklist spec (if diagnostic model).",
    "bias_risk_not_low": "Address bias risk factors until overall_risk_of_bias is 'low'. Review PROBAST+AI domains.",
    "claim_level_not_publication": "Set claim_level to 'publication-grade' in the checklist spec.",
})

TRIPOD_REQUIRED_TRUE = [
    "title_identifies_prediction_model",
    "target_population_defined",
    "outcome_definition_prespecified",
    "predictor_definition_prespecified",
    "sample_size_justification_reported",
    "missing_data_handling_reported",
    "model_building_procedure_reported",
    "internal_validation_strategy_reported",
    "full_model_specification_reported",
    "performance_measures_with_ci_reported",
    "limitations_and_clinical_use_reported",
]

PROBAST_REQUIRED_TRUE = [
    "participants_domain_low_risk",
    "predictors_domain_low_risk",
    "outcome_domain_low_risk",
    "analysis_domain_low_risk",
    "no_data_leakage_signals",
    "no_test_set_peeking",
]

STARD_REQUIRED_TRUE = [
    "participant_flow_reported",
    "index_test_details_reported",
    "reference_standard_reported",
    "blinding_status_reported",
    "diagnostic_accuracy_with_ci_reported",
]

ALLOWED_OVERALL_BIAS = {"low", "unclear", "high"}
ALLOWED_CLAIM_LEVEL = {"publication-grade", "preliminary", "not-claim-ready"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reporting and risk-of-bias checklist JSON.")
    parser.add_argument("--checklist-spec", required=True, help="Path to reporting/bias checklist JSON.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def load_json(path: Path, failures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        add_issue(
            failures,
            "missing_checklist_spec",
            "Checklist spec file not found.",
            {"path": str(path)},
        )
        return None
    if not path.is_file():
        add_issue(
            failures,
            "invalid_checklist_spec_path",
            "Checklist spec path must point to file.",
            {"path": str(path)},
        )
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_checklist_spec_json",
            "Failed to parse checklist spec JSON.",
            {"path": str(path), "error": str(exc)},
        )
        return None
    if not isinstance(payload, dict):
        add_issue(
            failures,
            "invalid_checklist_spec_root",
            "Checklist spec JSON root must be object.",
            {"actual_type": type(payload).__name__},
        )
        return None
    return payload


def require_section(
    spec: Dict[str, Any],
    key: str,
    failures: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    value = spec.get(key)
    if not isinstance(value, dict):
        add_issue(
            failures,
            "invalid_checklist_section",
            "Checklist section must be object.",
            {"section": key, "actual_type": type(value).__name__ if value is not None else None},
        )
        return None
    return value


def require_true_fields(
    section: Optional[Dict[str, Any]],
    section_name: str,
    required_keys: List[str],
    failures: List[Dict[str, Any]],
) -> int:
    if section is None:
        return 0
    true_count = 0
    for key in required_keys:
        value = section.get(key)
        if value is True:
            true_count += 1
            continue
        add_issue(
            failures,
            "checklist_item_not_satisfied",
            "Required checklist item must be true for publication-grade.",
            {"section": section_name, "item": key, "actual_value": value},
        )
    return true_count


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    checklist_path = Path(args.checklist_spec).expanduser().resolve()
    spec = load_json(checklist_path, failures)
    if spec is None:
        return finish(args, failures, warnings, {"path": str(checklist_path)})

    tripod = require_section(spec, "tripod_ai", failures)
    probast = require_section(spec, "probast_ai", failures)
    stard = require_section(spec, "stard_ai", failures)

    tripod_true = require_true_fields(tripod, "tripod_ai", TRIPOD_REQUIRED_TRUE, failures)
    probast_true = require_true_fields(probast, "probast_ai", PROBAST_REQUIRED_TRUE, failures)

    # STARD-AI can be marked as not applicable for non-diagnostic contexts.
    stard_applicable = None
    if stard is not None:
        stard_applicable = stard.get("applicable")
        if not isinstance(stard_applicable, bool):
            add_issue(
                failures,
                "invalid_stard_applicable",
                "stard_ai.applicable must be boolean.",
                {"actual_value": stard_applicable},
            )
            stard_applicable = None
    stard_not_applicable_justification: Optional[str] = None
    if stard_applicable is True:
        stard_true = require_true_fields(stard, "stard_ai", STARD_REQUIRED_TRUE, failures)
    elif stard_applicable is False:
        stard_true = 0
        if stard is not None:
            justification = stard.get("not_applicable_justification")
            if not isinstance(justification, str) or not justification.strip():
                add_issue(
                    failures,
                    "missing_stard_not_applicable_justification",
                    "When stard_ai.applicable=false, not_applicable_justification must be non-empty string.",
                    {"actual_value": justification},
                )
            else:
                stard_not_applicable_justification = justification.strip()
    else:
        stard_true = 0

    overall_bias = spec.get("overall_risk_of_bias")
    if not isinstance(overall_bias, str) or overall_bias.strip().lower() not in ALLOWED_OVERALL_BIAS:
        add_issue(
            failures,
            "invalid_overall_risk_of_bias",
            "overall_risk_of_bias must be one of: low/unclear/high.",
            {"actual_value": overall_bias, "allowed": sorted(ALLOWED_OVERALL_BIAS)},
        )
    elif overall_bias.strip().lower() != "low":
        add_issue(
            failures,
            "overall_risk_not_low",
            "Publication-grade requires overall_risk_of_bias=low.",
            {"overall_risk_of_bias": overall_bias},
        )

    claim_level = spec.get("claim_level")
    if not isinstance(claim_level, str) or claim_level.strip().lower() not in ALLOWED_CLAIM_LEVEL:
        add_issue(
            failures,
            "invalid_claim_level",
            "claim_level must be one of publication-grade/preliminary/not-claim-ready.",
            {"actual_value": claim_level, "allowed": sorted(ALLOWED_CLAIM_LEVEL)},
        )
    elif claim_level.strip().lower() != "publication-grade":
        add_issue(
            failures,
            "claim_level_not_publication_grade",
            "Checklist claim_level must be publication-grade for strict pipeline.",
            {"claim_level": claim_level},
        )

    summary = {
        "checklist_spec": str(checklist_path),
        "tripod_required_count": len(TRIPOD_REQUIRED_TRUE),
        "tripod_true_count": tripod_true,
        "probast_required_count": len(PROBAST_REQUIRED_TRUE),
        "probast_true_count": probast_true,
        "stard_required_count": len(STARD_REQUIRED_TRUE) if stard_applicable else 0,
        "stard_true_count": stard_true,
        "stard_applicable": stard_applicable,
        "stard_not_applicable_justification": stard_not_applicable_justification,
        "overall_risk_of_bias": overall_bias,
        "claim_level": claim_level,
    }
    return finish(args, failures, warnings, summary)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> int:
    from _gate_utils import get_gate_elapsed, write_json as _write_report

    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"

    fi = [GateIssue.from_legacy(f, Severity.ERROR) for f in failures]
    wi = [GateIssue.from_legacy(w, Severity.WARNING) for w in warnings]
    for issue in fi + wi:
        if not issue.remediation:
            issue.remediation = get_remediation(issue.code)

    report = build_report_envelope(
        gate_name="reporting_bias_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary=summary,
        input_files={
            "checklist_spec": str(Path(args.checklist_spec).expanduser().resolve()),
        },
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="reporting_bias_gate",
        status=status,
        failures=fi,
        warnings=wi,
        strict=bool(args.strict),
        elapsed=get_gate_elapsed(),
    )

    return 2 if should_fail else 0


if __name__ == "__main__":
    from _gate_utils import start_gate_timer
    start_gate_timer()
    raise SystemExit(main())
