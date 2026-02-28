#!/usr/bin/env python3
"""
One-command project bootstrap for ml-leakage-guard publication-grade workflow.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _gate_utils import load_json_from_path as load_json, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_ROOT = REPO_ROOT / "references"

TEMPLATE_COPY_MAP = {
    "feature-lineage.example.json": "feature_lineage.json",
    "feature-group-spec.example.json": "feature_group_spec.json",
    "split-protocol.example.json": "split_protocol.json",
    "imbalance-policy.example.json": "imbalance_policy.json",
    "missingness-policy.example.json": "missingness_policy.json",
    "tuning-protocol.example.json": "tuning_protocol.json",
    "performance-policy.example.json": "performance_policy.json",
    "external-cohort-spec.example.json": "external_cohort_spec.json",
    "reporting-bias-checklist.example.json": "reporting_bias_checklist.json",
    "execution-attestation.example.json": "execution_attestation.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a publication-grade ml-leakage-guard project.")
    parser.add_argument("--project-root", required=True, help="Project folder to initialize.")
    parser.add_argument("--study-id", default="medical-prediction-v1", help="study_id for request.json.")
    parser.add_argument("--run-id", default="", help="run_id for request.json (defaults to UTC timestamp token).")
    parser.add_argument("--target-name", default="disease_risk", help="target_name for request.json.")
    parser.add_argument("--prediction-unit", default="patient-episode", help="prediction_unit for request.json.")
    parser.add_argument("--index-time-col", default="event_time", help="index_time_col for request.json.")
    parser.add_argument("--label-col", default="y", help="label_col for request.json.")
    parser.add_argument("--patient-id-col", default="patient_id", help="patient_id_col for request.json.")
    parser.add_argument("--claim-tier", default="publication-grade", choices=["publication-grade"], help="Claim tier.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config files.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    return parser.parse_args()


def copy_template_json(src_name: str, dst_path: Path, force: bool) -> str:
    if dst_path.exists() and not force:
        return "preserved"
    src_path = REFERENCES_ROOT / src_name
    payload = load_json(src_path)
    write_json(dst_path, payload)
    return "written"


def make_phenotype_template(target_name: str) -> Dict[str, Any]:
    return {
        "global_forbidden_patterns": ["(?i)target", "(?i)label", "(?i)outcome"],
        "targets": {
            target_name: {
                "defining_variables": ["confirmed_diagnosis_code", "reference_standard_positive"],
                "forbidden_patterns": ["(?i)diagnosis_code", "(?i)reference_standard"],
                "notes": "Replace with project-specific disease-defining variables.",
            }
        },
    }


def build_request_payload(
    template: Dict[str, Any],
    study_id: str,
    run_id: str,
    target_name: str,
    prediction_unit: str,
    index_time_col: str,
    label_col: str,
    patient_id_col: str,
    claim_tier: str,
) -> Dict[str, Any]:
    payload = dict(template)
    payload["study_id"] = study_id
    payload["run_id"] = run_id
    payload["target_name"] = target_name
    payload["prediction_unit"] = prediction_unit
    payload["index_time_col"] = index_time_col
    payload["label_col"] = label_col
    payload["patient_id_col"] = patient_id_col
    payload["claim_tier_target"] = claim_tier
    payload["actual_primary_metric"] = 0.0
    payload["phenotype_definition_spec"] = "phenotype_definitions.json"
    payload["feature_lineage_spec"] = "feature_lineage.json"
    payload["feature_group_spec"] = "feature_group_spec.json"
    payload["split_protocol_spec"] = "split_protocol.json"
    payload["imbalance_policy_spec"] = "imbalance_policy.json"
    payload["missingness_policy_spec"] = "missingness_policy.json"
    payload["tuning_protocol_spec"] = "tuning_protocol.json"
    payload["performance_policy_spec"] = "performance_policy.json"
    payload["external_cohort_spec"] = "external_cohort_spec.json"
    payload["reporting_bias_checklist_spec"] = "reporting_bias_checklist.json"
    payload["execution_attestation_spec"] = "execution_attestation.json"
    payload["split_paths"] = {
        "train": "../data/train.csv",
        "valid": "../data/valid.csv",
        "test": "../data/test.csv",
    }
    payload["model_selection_report_file"] = "../evidence/model_selection_report.json"
    payload["feature_engineering_report_file"] = "../evidence/feature_engineering_report.json"
    payload["distribution_report_file"] = "../evidence/distribution_report.json"
    payload["robustness_report_file"] = "../evidence/robustness_report.json"
    payload["seed_sensitivity_report_file"] = "../evidence/seed_sensitivity_report.json"
    payload["evaluation_report_file"] = "../evidence/evaluation_report.json"
    payload["prediction_trace_file"] = "../evidence/prediction_trace.csv.gz"
    payload["external_validation_report_file"] = "../evidence/external_validation_report.json"
    payload["ci_matrix_report_file"] = "../evidence/ci_matrix_report.json"
    payload["permutation_null_metrics_file"] = "../evidence/permutation_null_pr_auc.txt"
    payload["context"] = {
        "notes": "Fill dataset-specific medical context and confirm disease-definition variables are excluded.",
    }
    return payload


def ensure_dirs(project_root: Path) -> List[str]:
    created: List[str] = []
    for rel in ("configs", "data", "evidence", "models", "keys"):
        p = project_root / rel
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))
        else:
            p.mkdir(parents=True, exist_ok=True)
    return created


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip()
    if not run_id:
        run_id = datetime.now(tz=timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")

    created_dirs = ensure_dirs(project_root)
    file_status: Dict[str, str] = {}

    for src_name, dst_name in TEMPLATE_COPY_MAP.items():
        dst_path = configs_dir / dst_name
        result = copy_template_json(src_name, dst_path, force=bool(args.force))
        file_status[str(dst_path)] = result

    phenotype_path = configs_dir / "phenotype_definitions.json"
    if phenotype_path.exists() and not args.force:
        file_status[str(phenotype_path)] = "preserved"
    else:
        write_json(phenotype_path, make_phenotype_template(str(args.target_name)))
        file_status[str(phenotype_path)] = "written"

    request_template = load_json(REFERENCES_ROOT / "request-schema.example.json")
    request_payload = build_request_payload(
        template=request_template,
        study_id=str(args.study_id),
        run_id=run_id,
        target_name=str(args.target_name),
        prediction_unit=str(args.prediction_unit),
        index_time_col=str(args.index_time_col),
        label_col=str(args.label_col),
        patient_id_col=str(args.patient_id_col),
        claim_tier=str(args.claim_tier),
    )
    request_path = configs_dir / "request.json"
    if request_path.exists() and not args.force:
        file_status[str(request_path)] = "preserved"
    else:
        write_json(request_path, request_payload)
        file_status[str(request_path)] = "written"

    report = {
        "status": "pass",
        "project_root": str(project_root),
        "created_directories": created_dirs,
        "file_status": file_status,
        "request_file": str(request_path),
        "next_steps": [
            "Place split CSV files at data/train.csv, data/valid.csv, data/test.csv.",
            "Run train_select_evaluate.py (or `mlgg.py train --interactive`) to generate required evidence artifacts.",
            "Run productized strict workflow with `--allow-missing-compare` on first run to bootstrap manifest baseline.",
        ],
    }

    if args.report:
        write_json(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"ProjectRoot: {project_root}")
    print(f"RequestFile: {request_path}")
    print("Next:")
    for line in report["next_steps"]:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
