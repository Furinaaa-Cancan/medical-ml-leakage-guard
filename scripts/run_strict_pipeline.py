#!/usr/bin/env python3
"""
Single-entry strict pipeline runner for medical leakage-safe prediction review.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full strict publication-grade gate pipeline.")
    parser.add_argument("--request", required=True, help="Path to request JSON.")
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="Directory for gate artifacts and reports (default: evidence).",
    )
    parser.add_argument(
        "--compare-manifest",
        help="Optional baseline manifest JSON path for reproducibility comparison.",
    )
    parser.add_argument(
        "--allow-missing-compare",
        action="store_true",
        help="Allow strict run without manifest baseline comparison (first-run bootstrap).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for running gate scripts.",
    )
    parser.add_argument("--report", help="Optional pipeline summary report JSON path.")
    parser.add_argument("--strict", action="store_true", help="Run all gates in strict mode.")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Diagnostic mode: keep executing subsequent gates after failures (never publication-valid).",
    )
    return parser.parse_args()


def run_step(name: str, cmd: List[str]) -> Tuple[int, str, str]:
    print(f"\n== Step: {name} ==")
    print(f"$ {shlex.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode, proc.stdout, proc.stderr


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def ensure_number(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Missing or invalid numeric field: {name}")


def main() -> int:
    args = parse_args()
    if not args.strict:
        print(
            "[FAIL] run_strict_pipeline.py enforces publication-grade strict mode. Re-run with --strict.",
            file=sys.stderr,
        )
        return 2
    strict_flag = ["--strict"]
    continue_on_fail = bool(args.continue_on_fail)

    if args.strict and not args.compare_manifest and not args.allow_missing_compare:
        print(
            "[FAIL] Strict mode requires --compare-manifest, or explicitly set --allow-missing-compare for bootstrap.",
            file=sys.stderr,
        )
        return 2

    request_path = Path(args.request).expanduser().resolve()
    if not request_path.exists():
        print(f"[FAIL] Request file not found: {request_path}", file=sys.stderr)
        return 2

    cwd = Path.cwd()
    evidence_dir = resolve_path(cwd, args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    reports = {
        "request_report": evidence_dir / "request_contract_report.json",
        "manifest": evidence_dir / "manifest.json",
        "execution_attestation_report": evidence_dir / "execution_attestation_report.json",
        "reporting_bias_report": evidence_dir / "reporting_bias_report.json",
        "leakage_report": evidence_dir / "leakage_report.json",
        "split_protocol_report": evidence_dir / "split_protocol_report.json",
        "covariate_shift_report": evidence_dir / "covariate_shift_report.json",
        "definition_report": evidence_dir / "definition_guard_report.json",
        "lineage_report": evidence_dir / "lineage_report.json",
        "imbalance_report": evidence_dir / "imbalance_policy_report.json",
        "missingness_report": evidence_dir / "missingness_policy_report.json",
        "tuning_report": evidence_dir / "tuning_leakage_report.json",
        "model_selection_audit_report": evidence_dir / "model_selection_audit_report.json",
        "feature_engineering_audit_report": evidence_dir / "feature_engineering_audit_report.json",
        "clinical_metrics_report": evidence_dir / "clinical_metrics_report.json",
        "prediction_replay_report": evidence_dir / "prediction_replay_report.json",
        "distribution_generalization_report": evidence_dir / "distribution_generalization_report.json",
        "generalization_gap_report": evidence_dir / "generalization_gap_report.json",
        "robustness_gate_report": evidence_dir / "robustness_gate_report.json",
        "seed_stability_report": evidence_dir / "seed_stability_report.json",
        "external_validation_gate_report": evidence_dir / "external_validation_gate_report.json",
        "calibration_dca_report": evidence_dir / "calibration_dca_report.json",
        "ci_matrix_gate_report": evidence_dir / "ci_matrix_gate_report.json",
        "metric_consistency_report": evidence_dir / "metric_consistency_report.json",
        "evaluation_quality_report": evidence_dir / "evaluation_quality_report.json",
        "permutation_report": evidence_dir / "permutation_report.json",
        "publication_report": evidence_dir / "publication_gate_report.json",
        "self_critique_report": evidence_dir / "self_critique_report.json",
    }

    steps: List[Dict[str, Any]] = []
    had_failure = False

    def execute(name: str, cmd: List[str]) -> bool:
        nonlocal had_failure
        code, stdout, stderr = run_step(name, cmd)
        steps.append(
            {
                "name": name,
                "command": shlex.join(cmd),
                "exit_code": code,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        ok = code == 0
        if not ok:
            had_failure = True
            if continue_on_fail:
                print(
                    f"[DIAGNOSTIC] Step failed but pipeline continues (--continue-on-fail): {name}",
                    file=sys.stderr,
                )
                return True
        return ok

    # Step 1: request contract
    if not execute(
        "request_contract_gate",
        [
            args.python,
            str(scripts_dir / "request_contract_gate.py"),
            "--request",
            str(request_path),
            "--report",
            str(reports["request_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    request_report = load_json(reports["request_report"])
    normalized = request_report.get("normalized_request", {})
    if not isinstance(normalized, dict):
        print("[FAIL] request_contract_report missing normalized_request.", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    split_paths = normalized.get("split_paths", {})
    if not isinstance(split_paths, dict):
        print("[FAIL] normalized_request.split_paths missing.", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    try:
        claim_tier_target = str(normalized["claim_tier_target"])
    except Exception as exc:
        print(f"[FAIL] Missing required normalized request fields: {exc}", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    if claim_tier_target != "publication-grade":
        print(
            f"[FAIL] run_strict_pipeline.py only supports publication-grade requests (got: {claim_tier_target}).",
            file=sys.stderr,
        )
        return finalize(args, reports, steps, success=False)

    try:
        train = str(split_paths["train"])
        test = str(split_paths["test"])
        valid = split_paths.get("valid")
        target_name = str(normalized["target_name"])
        id_col = str(normalized["patient_id_col"])
        time_col = str(normalized["index_time_col"])
        label_col = str(normalized["label_col"])
        metric_name = str(normalized["primary_metric"])
        study_id = str(normalized["study_id"])
        run_id = str(normalized["run_id"])
        phenotype_spec = str(normalized["phenotype_definition_spec"])
        lineage_spec = str(normalized["feature_lineage_spec"])
        split_protocol_spec = str(normalized["split_protocol_spec"])
        imbalance_policy_spec = str(normalized["imbalance_policy_spec"])
        missingness_policy_spec = str(normalized["missingness_policy_spec"])
        tuning_protocol_spec = str(normalized["tuning_protocol_spec"])
        reporting_bias_checklist_spec = str(normalized["reporting_bias_checklist_spec"])
        execution_attestation_spec = str(normalized["execution_attestation_spec"])
        performance_policy_spec = str(normalized["performance_policy_spec"])
        feature_group_spec = str(normalized["feature_group_spec"])
        model_selection_report_file = str(normalized["model_selection_report_file"])
        feature_engineering_report_file = str(normalized["feature_engineering_report_file"])
        distribution_report_file = str(normalized["distribution_report_file"])
        robustness_report_file = str(normalized["robustness_report_file"])
        seed_sensitivity_report_file = str(normalized["seed_sensitivity_report_file"])
        evaluation_report_file = str(normalized["evaluation_report_file"])
        prediction_trace_file = str(normalized["prediction_trace_file"])
        external_cohort_spec = str(normalized["external_cohort_spec"])
        external_validation_report_file = str(normalized["external_validation_report_file"])
        ci_matrix_report_file = str(normalized["ci_matrix_report_file"])
        evaluation_metric_path = normalized.get("evaluation_metric_path")
        null_metrics_file = str(normalized["permutation_null_metrics_file"])
        expected_metric = ensure_number(normalized.get("actual_primary_metric"), "actual_primary_metric")
        thresholds = normalized.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        alpha = float(thresholds.get("alpha", 0.01))
        min_delta = float(thresholds.get("min_delta", 0.03))
        min_baseline_delta = float(thresholds.get("min_baseline_delta", 0.0))
        ci_min_resamples = int(float(thresholds.get("ci_min_resamples", 200)))
        ci_max_width = float(thresholds.get("ci_max_width", 0.50))
    except Exception as exc:
        print(f"[FAIL] Missing required publication-grade fields: {exc}", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    split_args: List[str] = ["--train", train, "--test", test]
    if isinstance(valid, str) and valid:
        split_args += ["--valid", valid]
    valid_for_required_gates = valid if isinstance(valid, str) and valid else test

    compare_args: List[str] = []
    if args.compare_manifest:
        compare_args = ["--compare-with", str(resolve_path(cwd, args.compare_manifest))]

    manifest_inputs = [train]
    if isinstance(valid, str) and valid:
        manifest_inputs.append(valid)

    attestation_spec_path = resolve_path(cwd, execution_attestation_spec)
    attestation_extra_inputs: List[str] = [str(attestation_spec_path)]
    try:
        attestation_spec_payload = load_json(attestation_spec_path)
        signing_base = attestation_spec_path.parent

        def append_attestation_path(raw_value: Any) -> None:
            if isinstance(raw_value, str) and raw_value.strip():
                attestation_extra_inputs.append(str(resolve_path(signing_base, raw_value.strip())))

        signing_block = attestation_spec_payload.get("signing")
        if isinstance(signing_block, dict):
            for key in (
                "signed_payload_file",
                "signature_file",
                "public_key_file",
                "revocation_list_file",
            ):
                append_attestation_path(signing_block.get(key))

        timestamp_block = attestation_spec_payload.get("timestamp_trust")
        if isinstance(timestamp_block, dict):
            for key in ("record_file", "signature_file", "public_key_file"):
                append_attestation_path(timestamp_block.get(key))

        transparency_block = attestation_spec_payload.get("transparency_log")
        if isinstance(transparency_block, dict):
            for key in ("record_file", "signature_file", "public_key_file"):
                append_attestation_path(transparency_block.get(key))

        execution_receipt_block = attestation_spec_payload.get("execution_receipt")
        if isinstance(execution_receipt_block, dict):
            for key in ("record_file", "signature_file", "public_key_file"):
                append_attestation_path(execution_receipt_block.get(key))

        execution_log_block = attestation_spec_payload.get("execution_log_attestation")
        if isinstance(execution_log_block, dict):
            for key in ("record_file", "signature_file", "public_key_file"):
                append_attestation_path(execution_log_block.get(key))

        witness_block = attestation_spec_payload.get("witness_quorum")
        if isinstance(witness_block, dict):
            witness_records = witness_block.get("records")
            if isinstance(witness_records, list):
                for record in witness_records:
                    if isinstance(record, dict):
                        for key in ("record_file", "signature_file", "public_key_file"):
                            append_attestation_path(record.get(key))
    except Exception as exc:
        print(f"[FAIL] Failed to parse execution_attestation_spec for manifest lock: {exc}", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    # Keep deterministic order while deduplicating.
    attestation_extra_inputs = list(dict.fromkeys(attestation_extra_inputs))

    gate_script_inputs = [
        str(scripts_dir / "run_strict_pipeline.py"),
        str(scripts_dir / "request_contract_gate.py"),
        str(scripts_dir / "manifest_lock.py"),
        str(scripts_dir / "execution_attestation_gate.py"),
        str(scripts_dir / "generate_execution_attestation.py"),
        str(scripts_dir / "reporting_bias_gate.py"),
        str(scripts_dir / "leakage_gate.py"),
        str(scripts_dir / "split_protocol_gate.py"),
        str(scripts_dir / "covariate_shift_gate.py"),
        str(scripts_dir / "definition_variable_guard.py"),
        str(scripts_dir / "feature_lineage_gate.py"),
        str(scripts_dir / "imbalance_policy_gate.py"),
        str(scripts_dir / "missingness_policy_gate.py"),
        str(scripts_dir / "tuning_leakage_gate.py"),
        str(scripts_dir / "model_selection_audit_gate.py"),
        str(scripts_dir / "feature_engineering_audit_gate.py"),
        str(scripts_dir / "clinical_metrics_gate.py"),
        str(scripts_dir / "prediction_replay_gate.py"),
        str(scripts_dir / "distribution_generalization_gate.py"),
        str(scripts_dir / "generalization_gap_gate.py"),
        str(scripts_dir / "robustness_gate.py"),
        str(scripts_dir / "seed_stability_gate.py"),
        str(scripts_dir / "external_validation_gate.py"),
        str(scripts_dir / "calibration_dca_gate.py"),
        str(scripts_dir / "ci_matrix_gate.py"),
        str(scripts_dir / "metric_consistency_gate.py"),
        str(scripts_dir / "evaluation_quality_gate.py"),
        str(scripts_dir / "permutation_significance_gate.py"),
        str(scripts_dir / "publication_gate.py"),
        str(scripts_dir / "self_critique_gate.py"),
    ]

    manifest_inputs.extend(
        [
            test,
            phenotype_spec,
            lineage_spec,
            split_protocol_spec,
            imbalance_policy_spec,
            missingness_policy_spec,
            tuning_protocol_spec,
            reporting_bias_checklist_spec,
            performance_policy_spec,
            feature_group_spec,
            model_selection_report_file,
            feature_engineering_report_file,
            distribution_report_file,
            robustness_report_file,
            seed_sensitivity_report_file,
            *attestation_extra_inputs,
            evaluation_report_file,
            prediction_trace_file,
            external_cohort_spec,
            external_validation_report_file,
            ci_matrix_report_file,
            null_metrics_file,
            str(request_path),
            *gate_script_inputs,
        ]
    )

    # Step 2: manifest lock
    if not execute(
        "manifest_lock",
        [
            args.python,
            str(scripts_dir / "manifest_lock.py"),
            "--inputs",
            *manifest_inputs,
            "--output",
            str(reports["manifest"]),
            *compare_args,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 3: execution attestation gate
    if not execute(
        "execution_attestation_gate",
        [
            args.python,
            str(scripts_dir / "execution_attestation_gate.py"),
            "--attestation-spec",
            execution_attestation_spec,
            "--evaluation-report",
            evaluation_report_file,
            "--study-id",
            study_id,
            "--run-id",
            run_id,
            "--report",
            str(reports["execution_attestation_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 4: leakage gate
    if not execute(
        "leakage_gate",
        [
            args.python,
            str(scripts_dir / "leakage_gate.py"),
            *split_args,
            "--id-cols",
            id_col,
            "--time-col",
            time_col,
            "--target-col",
            label_col,
            "--report",
            str(reports["leakage_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 5: split protocol gate
    if not execute(
        "split_protocol_gate",
        [
            args.python,
            str(scripts_dir / "split_protocol_gate.py"),
            "--protocol-spec",
            split_protocol_spec,
            *split_args,
            "--id-col",
            id_col,
            "--time-col",
            time_col,
            "--target-col",
            label_col,
            "--report",
            str(reports["split_protocol_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 6: covariate shift gate
    if not execute(
        "covariate_shift_gate",
        [
            args.python,
            str(scripts_dir / "covariate_shift_gate.py"),
            *split_args,
            "--target-col",
            label_col,
            "--ignore-cols",
            f"{id_col},{time_col}",
            "--report",
            str(reports["covariate_shift_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 7: reporting and bias checklist gate
    if not execute(
        "reporting_bias_gate",
        [
            args.python,
            str(scripts_dir / "reporting_bias_gate.py"),
            "--checklist-spec",
            reporting_bias_checklist_spec,
            "--report",
            str(reports["reporting_bias_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 8: definition variable guard
    if not execute(
        "definition_variable_guard",
        [
            args.python,
            str(scripts_dir / "definition_variable_guard.py"),
            "--target",
            target_name,
            "--definition-spec",
            phenotype_spec,
            *split_args,
            "--target-col",
            label_col,
            "--ignore-cols",
            f"{id_col},{time_col}",
            "--report",
            str(reports["definition_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 9: lineage gate
    if not execute(
        "feature_lineage_gate",
        [
            args.python,
            str(scripts_dir / "feature_lineage_gate.py"),
            "--target",
            target_name,
            "--definition-spec",
            phenotype_spec,
            "--lineage-spec",
            lineage_spec,
            *split_args,
            "--target-col",
            label_col,
            "--ignore-cols",
            f"{id_col},{time_col}",
            "--report",
            str(reports["lineage_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 10: imbalance policy gate
    if not execute(
        "imbalance_policy_gate",
        [
            args.python,
            str(scripts_dir / "imbalance_policy_gate.py"),
            "--policy-spec",
            imbalance_policy_spec,
            *split_args,
            "--target-col",
            label_col,
            "--report",
            str(reports["imbalance_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 11: missingness policy gate
    if not execute(
        "missingness_policy_gate",
        [
            args.python,
            str(scripts_dir / "missingness_policy_gate.py"),
            "--policy-spec",
            missingness_policy_spec,
            *split_args,
            "--target-col",
            label_col,
            "--ignore-cols",
            f"{id_col},{time_col}",
            "--evaluation-report",
            evaluation_report_file,
            "--report",
            str(reports["missingness_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 12: tuning leakage gate
    tuning_cmd = [
        args.python,
        str(scripts_dir / "tuning_leakage_gate.py"),
        "--tuning-spec",
        tuning_protocol_spec,
        "--id-col",
        id_col,
        "--report",
        str(reports["tuning_report"]),
        *strict_flag,
    ]
    if isinstance(valid, str) and valid:
        tuning_cmd.append("--has-valid-split")

    if not execute(
        "tuning_leakage_gate",
        tuning_cmd,
    ):
        return finalize(args, reports, steps, success=False)

    # Step 13: model selection audit gate
    if not execute(
        "model_selection_audit_gate",
        [
            args.python,
            str(scripts_dir / "model_selection_audit_gate.py"),
            "--model-selection-report",
            model_selection_report_file,
            "--tuning-spec",
            tuning_protocol_spec,
            "--train",
            train,
            *(["--valid", valid] if isinstance(valid, str) and valid else []),
            "--test",
            test,
            "--expected-primary-metric",
            metric_name,
            "--report",
            str(reports["model_selection_audit_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 14: feature engineering audit gate
    if not execute(
        "feature_engineering_audit_gate",
        [
            args.python,
            str(scripts_dir / "feature_engineering_audit_gate.py"),
            "--feature-group-spec",
            feature_group_spec,
            "--feature-engineering-report",
            feature_engineering_report_file,
            "--lineage-spec",
            lineage_spec,
            "--tuning-spec",
            tuning_protocol_spec,
            "--report",
            str(reports["feature_engineering_audit_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 15: clinical metrics gate
    if not execute(
        "clinical_metrics_gate",
        [
            args.python,
            str(scripts_dir / "clinical_metrics_gate.py"),
            "--evaluation-report",
            evaluation_report_file,
            "--external-validation-report",
            external_validation_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["clinical_metrics_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 16: prediction replay gate
    if not execute(
        "prediction_replay_gate",
        [
            args.python,
            str(scripts_dir / "prediction_replay_gate.py"),
            "--evaluation-report",
            evaluation_report_file,
            "--prediction-trace",
            prediction_trace_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["prediction_replay_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 17: distribution generalization gate
    if not execute(
        "distribution_generalization_gate",
        [
            args.python,
            str(scripts_dir / "distribution_generalization_gate.py"),
            "--train",
            train,
            "--valid",
            valid_for_required_gates,
            "--test",
            test,
            "--evaluation-report",
            evaluation_report_file,
            "--external-validation-report",
            external_validation_report_file,
            "--feature-group-spec",
            feature_group_spec,
            "--target-col",
            label_col,
            "--ignore-cols",
            f"{id_col},{time_col}",
            "--performance-policy",
            performance_policy_spec,
            "--distribution-report",
            distribution_report_file,
            "--report",
            str(reports["distribution_generalization_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 18: generalization gap gate
    if not execute(
        "generalization_gap_gate",
        [
            args.python,
            str(scripts_dir / "generalization_gap_gate.py"),
            "--evaluation-report",
            evaluation_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["generalization_gap_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 19: robustness gate
    if not execute(
        "robustness_gate",
        [
            args.python,
            str(scripts_dir / "robustness_gate.py"),
            "--robustness-report",
            robustness_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["robustness_gate_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 20: seed stability gate
    if not execute(
        "seed_stability_gate",
        [
            args.python,
            str(scripts_dir / "seed_stability_gate.py"),
            "--seed-sensitivity-report",
            seed_sensitivity_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["seed_stability_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 21: external validation gate
    if not execute(
        "external_validation_gate",
        [
            args.python,
            str(scripts_dir / "external_validation_gate.py"),
            "--external-validation-report",
            external_validation_report_file,
            "--prediction-trace",
            prediction_trace_file,
            "--evaluation-report",
            evaluation_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["external_validation_gate_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 22: calibration + decision curve gate
    if not execute(
        "calibration_dca_gate",
        [
            args.python,
            str(scripts_dir / "calibration_dca_gate.py"),
            "--prediction-trace",
            prediction_trace_file,
            "--evaluation-report",
            evaluation_report_file,
            "--external-validation-report",
            external_validation_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--report",
            str(reports["calibration_dca_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 23: CI matrix gate
    if not execute(
        "ci_matrix_gate",
        [
            args.python,
            str(scripts_dir / "ci_matrix_gate.py"),
            "--evaluation-report",
            evaluation_report_file,
            "--prediction-trace",
            prediction_trace_file,
            "--external-validation-report",
            external_validation_report_file,
            "--performance-policy",
            performance_policy_spec,
            "--ci-matrix-report",
            ci_matrix_report_file,
            "--report",
            str(reports["ci_matrix_gate_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 24: metric consistency gate
    metric_consistency_cmd = [
        args.python,
        str(scripts_dir / "metric_consistency_gate.py"),
        "--evaluation-report",
        evaluation_report_file,
        "--required-evaluation-split",
        "test",
        "--metric-name",
        metric_name,
        "--expected",
        str(expected_metric),
        "--report",
        str(reports["metric_consistency_report"]),
        *strict_flag,
    ]
    if isinstance(evaluation_metric_path, str) and evaluation_metric_path:
        metric_consistency_cmd.extend(["--metric-path", evaluation_metric_path])

    if not execute(
        "metric_consistency_gate",
        metric_consistency_cmd,
    ):
        return finalize(args, reports, steps, success=False)

    metric_consistency_report = load_json(reports["metric_consistency_report"])
    try:
        actual_metric = ensure_number(
            metric_consistency_report.get("actual_metric"), "metric_consistency_report.actual_metric"
        )
    except Exception as exc:
        print(f"[FAIL] Metric consistency report missing actual metric: {exc}", file=sys.stderr)
        return finalize(args, reports, steps, success=False)

    # Step 25: evaluation quality gate
    evaluation_quality_cmd = [
        args.python,
        str(scripts_dir / "evaluation_quality_gate.py"),
        "--evaluation-report",
        evaluation_report_file,
        "--ci-matrix-report",
        ci_matrix_report_file,
        "--metric-name",
        metric_name,
        "--primary-metric",
        str(actual_metric),
        "--min-resamples",
        str(ci_min_resamples),
        "--min-baseline-delta",
        str(min_baseline_delta),
        "--max-ci-width",
        str(ci_max_width),
        "--report",
        str(reports["evaluation_quality_report"]),
        *strict_flag,
    ]
    if isinstance(evaluation_metric_path, str) and evaluation_metric_path:
        evaluation_quality_cmd.extend(["--metric-path", evaluation_metric_path])

    if not execute(
        "evaluation_quality_gate",
        evaluation_quality_cmd,
    ):
        return finalize(args, reports, steps, success=False)

    # Step 26: permutation significance gate
    if not execute(
        "permutation_significance_gate",
        [
            args.python,
            str(scripts_dir / "permutation_significance_gate.py"),
            "--metric-name",
            metric_name,
            "--actual",
            str(actual_metric),
            "--null-metrics-file",
            null_metrics_file,
            "--alpha",
            str(alpha),
            "--min-delta",
            str(min_delta),
            "--report",
            str(reports["permutation_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 27: publication gate
    if not execute(
        "publication_gate",
        [
            args.python,
            str(scripts_dir / "publication_gate.py"),
            "--request-report",
            str(reports["request_report"]),
            "--manifest",
            str(reports["manifest"]),
            "--execution-attestation-report",
            str(reports["execution_attestation_report"]),
            "--reporting-bias-report",
            str(reports["reporting_bias_report"]),
            "--leakage-report",
            str(reports["leakage_report"]),
            "--split-protocol-report",
            str(reports["split_protocol_report"]),
            "--covariate-shift-report",
            str(reports["covariate_shift_report"]),
            "--definition-report",
            str(reports["definition_report"]),
            "--lineage-report",
            str(reports["lineage_report"]),
            "--imbalance-report",
            str(reports["imbalance_report"]),
            "--missingness-report",
            str(reports["missingness_report"]),
            "--tuning-report",
            str(reports["tuning_report"]),
            "--model-selection-audit-report",
            str(reports["model_selection_audit_report"]),
            "--feature-engineering-audit-report",
            str(reports["feature_engineering_audit_report"]),
            "--clinical-metrics-report",
            str(reports["clinical_metrics_report"]),
            "--prediction-replay-report",
            str(reports["prediction_replay_report"]),
            "--distribution-generalization-report",
            str(reports["distribution_generalization_report"]),
            "--generalization-gap-report",
            str(reports["generalization_gap_report"]),
            "--robustness-report",
            str(reports["robustness_gate_report"]),
            "--seed-stability-report",
            str(reports["seed_stability_report"]),
            "--external-validation-report",
            str(reports["external_validation_gate_report"]),
            "--calibration-dca-report",
            str(reports["calibration_dca_report"]),
            "--ci-matrix-report",
            str(reports["ci_matrix_gate_report"]),
            "--metric-report",
            str(reports["metric_consistency_report"]),
            "--evaluation-quality-report",
            str(reports["evaluation_quality_report"]),
            "--permutation-report",
            str(reports["permutation_report"]),
            "--report",
            str(reports["publication_report"]),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    # Step 28: self critique
    if not execute(
        "self_critique_gate",
        [
            args.python,
            str(scripts_dir / "self_critique_gate.py"),
            "--request-report",
            str(reports["request_report"]),
            "--manifest",
            str(reports["manifest"]),
            "--execution-attestation-report",
            str(reports["execution_attestation_report"]),
            "--reporting-bias-report",
            str(reports["reporting_bias_report"]),
            "--leakage-report",
            str(reports["leakage_report"]),
            "--split-protocol-report",
            str(reports["split_protocol_report"]),
            "--covariate-shift-report",
            str(reports["covariate_shift_report"]),
            "--definition-report",
            str(reports["definition_report"]),
            "--lineage-report",
            str(reports["lineage_report"]),
            "--imbalance-report",
            str(reports["imbalance_report"]),
            "--missingness-report",
            str(reports["missingness_report"]),
            "--tuning-report",
            str(reports["tuning_report"]),
            "--model-selection-audit-report",
            str(reports["model_selection_audit_report"]),
            "--feature-engineering-audit-report",
            str(reports["feature_engineering_audit_report"]),
            "--clinical-metrics-report",
            str(reports["clinical_metrics_report"]),
            "--prediction-replay-report",
            str(reports["prediction_replay_report"]),
            "--distribution-generalization-report",
            str(reports["distribution_generalization_report"]),
            "--generalization-gap-report",
            str(reports["generalization_gap_report"]),
            "--robustness-report",
            str(reports["robustness_gate_report"]),
            "--seed-stability-report",
            str(reports["seed_stability_report"]),
            "--external-validation-report",
            str(reports["external_validation_gate_report"]),
            "--calibration-dca-report",
            str(reports["calibration_dca_report"]),
            "--ci-matrix-report",
            str(reports["ci_matrix_gate_report"]),
            "--metric-report",
            str(reports["metric_consistency_report"]),
            "--evaluation-quality-report",
            str(reports["evaluation_quality_report"]),
            "--permutation-report",
            str(reports["permutation_report"]),
            "--publication-report",
            str(reports["publication_report"]),
            "--min-score",
            "95",
            "--report",
            str(reports["self_critique_report"]),
            *(["--allow-missing-comparison"] if args.allow_missing_compare else []),
            *strict_flag,
        ],
    ):
        return finalize(args, reports, steps, success=False)

    return finalize(args, reports, steps, success=(not had_failure))


def finalize(
    args: argparse.Namespace,
    reports: Dict[str, Path],
    steps: List[Dict[str, Any]],
    success: bool,
) -> int:
    summary = {
        "status": "pass" if success else "fail",
        "strict_mode": bool(args.strict),
        "diagnostic_only": bool(args.continue_on_fail),
        "publication_eligible": bool(args.strict and (not args.continue_on_fail) and success),
        "failure_count": sum(1 for s in steps if s.get("exit_code") != 0),
        "steps": steps,
        "artifacts": {k: str(v) for k, v in reports.items()},
    }

    out_path = Path(args.report).expanduser().resolve() if args.report else reports["self_critique_report"].parent / "strict_pipeline_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=True, indent=2)

    print(f"\nPipeline status: {summary['status']}")
    print(f"Pipeline report: {out_path}")
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
