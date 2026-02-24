#!/usr/bin/env python3
"""
Adversarial strict-gate validation for ml-leakage-guard.

This script intentionally injects protocol violations and verifies that
corresponding gates fail with expected failure codes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


@dataclass
class ScenarioResult:
    name: str
    expected_codes: List[str]
    exit_code: int
    observed_codes: List[str]
    passed: bool
    report_path: str
    command: List[str]
    stderr_tail: str
    stdout_tail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adversarial fail-closed checks for strict gates.")
    parser.add_argument(
        "--case-id",
        default="uci-heart-disease",
        help="Case directory under experiments/authority-e2e (default: uci-heart-disease).",
    )
    parser.add_argument(
        "--output",
        default="adversarial/adversarial_summary.json",
        help="Output summary path relative to experiments/authority-e2e.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def absolutize_attestation_paths(spec: Dict[str, Any], spec_dir: Path) -> Dict[str, Any]:
    cloned = copy.deepcopy(spec)
    for block_name, keys in (
        ("signing", ["signed_payload_file", "signature_file", "public_key_file", "revocation_list_file"]),
        ("timestamp_trust", ["record_file", "signature_file", "public_key_file"]),
        ("transparency_log", ["record_file", "signature_file", "public_key_file"]),
        ("execution_receipt", ["record_file", "signature_file", "public_key_file"]),
        ("execution_log_attestation", ["record_file", "signature_file", "public_key_file"]),
    ):
        block = cloned.get(block_name)
        if not isinstance(block, dict):
            continue
        for key in keys:
            raw = block.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (spec_dir / p).resolve()
            else:
                p = p.resolve()
            block[key] = str(p)

    witness_block = cloned.get("witness_quorum")
    if isinstance(witness_block, dict):
        records = witness_block.get("records")
        if isinstance(records, list):
            for entry in records:
                if not isinstance(entry, dict):
                    continue
                for key in ("record_file", "signature_file", "public_key_file"):
                    raw = entry.get(key)
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    p = Path(raw).expanduser()
                    if not p.is_absolute():
                        p = (spec_dir / p).resolve()
                    else:
                        p = p.resolve()
                    entry[key] = str(p)
    return cloned


def execute_scenario(
    name: str,
    expected_codes: List[str],
    cmd: List[str],
    report_path: Path,
) -> ScenarioResult:
    proc = run_cmd(cmd)
    observed_codes: List[str] = []
    if report_path.exists():
        report = load_json(report_path)
        failures = report.get("failures", [])
        if isinstance(failures, list):
            for issue in failures:
                if isinstance(issue, dict):
                    code = issue.get("code")
                    if isinstance(code, str) and code:
                        observed_codes.append(code)

    matched = any(code in observed_codes for code in expected_codes)
    passed = (proc.returncode != 0) and matched
    return ScenarioResult(
        name=name,
        expected_codes=expected_codes,
        exit_code=int(proc.returncode),
        observed_codes=sorted(set(observed_codes)),
        passed=passed,
        report_path=str(report_path),
        command=cmd,
        stderr_tail=proc.stderr[-2000:],
        stdout_tail=proc.stdout[-2000:],
    )


def main() -> int:
    args = parse_args()

    case_root = EXPERIMENT_ROOT / args.case_id
    cfg_dir = case_root / "configs"
    data_dir = case_root / "data"
    evidence_dir = case_root / "evidence"
    request_path = cfg_dir / "request.json"
    if not request_path.exists():
        raise SystemExit(
            f"Case artifacts not found: {request_path}\n"
            "Run experiments/authority-e2e/run_authority_e2e.py first."
        )

    request = load_json(request_path)
    study_id = str(request["study_id"])
    run_id = str(request["run_id"])
    target_name = str(request["target_name"])
    id_col = str(request["patient_id_col"])
    time_col = str(request["index_time_col"])
    label_col = str(request["label_col"])
    train_csv_path = data_dir / "train.csv"
    with train_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    if header is None:
        raise RuntimeError(f"Missing header in {train_csv_path}")
    feature_cols = [h.strip() for h in header if h.strip() and h.strip() not in {id_col, time_col, label_col}]
    if not feature_cols:
        raise RuntimeError(f"No predictor feature columns detected in {train_csv_path}")
    first_feature_col = feature_cols[0]

    tmp_root = EXPERIMENT_ROOT / "adversarial" / args.case_id
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    scenarios: List[ScenarioResult] = []

    # 1) Definition variable leakage.
    phenotype = load_json(cfg_dir / "phenotype_definitions.json")
    target_block = phenotype.get("targets", {}).get(target_name, {})
    if not isinstance(target_block, dict):
        raise RuntimeError("Invalid phenotype definition target block.")
    target_block["defining_variables"] = [str(target_block.get("defining_variables", [""])[0]), first_feature_col]
    pheno_bad = tmp_root / "phenotype_definitions.bad.json"
    write_json(pheno_bad, phenotype)
    report1 = tmp_root / "definition_guard.bad.report.json"
    cmd1 = [
        sys.executable,
        str(SCRIPTS_ROOT / "definition_variable_guard.py"),
        "--target",
        target_name,
        "--definition-spec",
        str(pheno_bad),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--target-col",
        label_col,
        "--ignore-cols",
        f"{id_col},{time_col}",
        "--strict",
        "--report",
        str(report1),
    ]
    scenarios.append(execute_scenario("definition_variable_leakage", ["definition_variable_leakage"], cmd1, report1))

    # 2) Lineage leakage.
    lineage = load_json(cfg_dir / "feature_lineage.json")
    features = lineage.get("features")
    if not isinstance(features, dict) or not features:
        raise RuntimeError("Invalid feature lineage payload.")
    first_feature = sorted(features.keys())[0]
    feature_block = features[first_feature]
    if not isinstance(feature_block, dict):
        feature_block = {"ancestors": []}
        features[first_feature] = feature_block
    feature_block["ancestors"] = ["confirmed_diagnosis_code"]
    lineage_bad = tmp_root / "feature_lineage.bad.json"
    write_json(lineage_bad, lineage)
    report2 = tmp_root / "lineage_guard.bad.report.json"
    cmd2 = [
        sys.executable,
        str(SCRIPTS_ROOT / "feature_lineage_gate.py"),
        "--target",
        target_name,
        "--definition-spec",
        str(cfg_dir / "phenotype_definitions.json"),
        "--lineage-spec",
        str(lineage_bad),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--target-col",
        label_col,
        "--ignore-cols",
        f"{id_col},{time_col}",
        "--strict",
        "--report",
        str(report2),
    ]
    scenarios.append(execute_scenario("lineage_definition_leakage", ["lineage_definition_leakage"], cmd2, report2))

    # 3) Tuning leakage via explicit test usage.
    tuning = load_json(cfg_dir / "tuning_protocol.json")
    tuning["test_used_for_model_selection"] = True
    tuning_bad = tmp_root / "tuning_protocol.bad.json"
    write_json(tuning_bad, tuning)
    report3 = tmp_root / "tuning_guard.bad.report.json"
    cmd3 = [
        sys.executable,
        str(SCRIPTS_ROOT / "tuning_leakage_gate.py"),
        "--tuning-spec",
        str(tuning_bad),
        "--id-col",
        id_col,
        "--has-valid-split",
        "--strict",
        "--report",
        str(report3),
    ]
    scenarios.append(execute_scenario("tuning_test_usage", ["explicit_test_usage"], cmd3, report3))

    # 4) Imbalance policy misuse on test split.
    imbalance = load_json(cfg_dir / "imbalance_policy.json")
    imbalance["threshold_selection_split"] = "test"
    imbalance_bad = tmp_root / "imbalance_policy.bad.json"
    write_json(imbalance_bad, imbalance)
    report4 = tmp_root / "imbalance_guard.bad.report.json"
    cmd4 = [
        sys.executable,
        str(SCRIPTS_ROOT / "imbalance_policy_gate.py"),
        "--policy-spec",
        str(imbalance_bad),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--target-col",
        label_col,
        "--strict",
        "--report",
        str(report4),
    ]
    scenarios.append(execute_scenario("imbalance_postprocessing_on_test", ["test_split_used_for_postprocessing"], cmd4, report4))

    # 5) Missingness leakage via target in imputation.
    missingness = load_json(cfg_dir / "missingness_policy.json")
    missingness["use_target_in_imputation"] = True
    missing_bad = tmp_root / "missingness_policy.bad.json"
    write_json(missing_bad, missingness)
    report5 = tmp_root / "missingness_guard.bad.report.json"
    cmd5 = [
        sys.executable,
        str(SCRIPTS_ROOT / "missingness_policy_gate.py"),
        "--policy-spec",
        str(missing_bad),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--target-col",
        label_col,
        "--ignore-cols",
        f"{id_col},{time_col}",
        "--strict",
        "--report",
        str(report5),
    ]
    scenarios.append(execute_scenario("missingness_target_leakage", ["target_used_in_imputation"], cmd5, report5))

    # 6) Reporting checklist not satisfied.
    checklist = load_json(cfg_dir / "reporting_bias_checklist.json")
    tripod = checklist.get("tripod_ai")
    if not isinstance(tripod, dict):
        raise RuntimeError("Invalid reporting checklist tripod_ai block.")
    tripod["performance_measures_with_ci_reported"] = False
    checklist_bad = tmp_root / "reporting_bias_checklist.bad.json"
    write_json(checklist_bad, checklist)
    report6 = tmp_root / "reporting_bias.bad.report.json"
    cmd6 = [
        sys.executable,
        str(SCRIPTS_ROOT / "reporting_bias_gate.py"),
        "--checklist-spec",
        str(checklist_bad),
        "--strict",
        "--report",
        str(report6),
    ]
    scenarios.append(execute_scenario("reporting_checklist_failure", ["checklist_item_not_satisfied"], cmd6, report6))

    # 7) Execution receipt tampering.
    attestation_spec = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs = absolutize_attestation_paths(attestation_spec, cfg_dir)
    exec_block = attestation_abs.get("execution_receipt")
    if not isinstance(exec_block, dict):
        raise RuntimeError("Execution receipt block missing in attestation spec.")
    record_file = Path(str(exec_block["record_file"]))
    tampered_record = tmp_root / "attestation_execution_receipt_record.tampered.json"
    tampered_payload = load_json(record_file)
    tampered_payload["exit_code"] = 1
    write_json(tampered_record, tampered_payload)
    exec_block["record_file"] = str(tampered_record)
    attestation_bad = tmp_root / "execution_attestation.bad.json"
    write_json(attestation_bad, attestation_abs)
    report7 = tmp_root / "execution_attestation.bad.report.json"
    cmd7 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report7),
    ]
    scenarios.append(
        execute_scenario(
            "execution_receipt_tamper",
            ["signature_verification_failed", "execution_receipt_exit_code_mismatch", "execution_receipt_nonzero_exit_code"],
            cmd7,
            report7,
        )
    )

    # 8) Split protocol allows overlap.
    split_protocol = load_json(cfg_dir / "split_protocol.json")
    split_protocol["allow_patient_overlap"] = True
    split_protocol_bad = tmp_root / "split_protocol.bad.json"
    write_json(split_protocol_bad, split_protocol)
    report8 = tmp_root / "split_protocol.bad.report.json"
    cmd8 = [
        sys.executable,
        str(SCRIPTS_ROOT / "split_protocol_gate.py"),
        "--protocol-spec",
        str(split_protocol_bad),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--id-col",
        id_col,
        "--time-col",
        time_col,
        "--target-col",
        label_col,
        "--strict",
        "--report",
        str(report8),
    ]
    scenarios.append(execute_scenario("split_protocol_overlap_allowed", ["patient_overlap_allowed"], cmd8, report8))

    # 9) Evaluation quality: remove CI evidence.
    eval_report = load_json(evidence_dir / "evaluation_report.json")
    eval_no_ci = copy.deepcopy(eval_report)
    eval_no_ci.pop("uncertainty", None)
    eval_no_ci_path = tmp_root / "evaluation_report.no_ci.json"
    write_json(eval_no_ci_path, eval_no_ci)
    report9 = tmp_root / "evaluation_quality.no_ci.report.json"
    cmd9 = [
        sys.executable,
        str(SCRIPTS_ROOT / "evaluation_quality_gate.py"),
        "--evaluation-report",
        str(eval_no_ci_path),
        "--metric-name",
        "roc_auc",
        "--metric-path",
        "metrics.roc_auc",
        "--primary-metric",
        str(float(eval_report.get("metrics", {}).get("roc_auc", 0.0))),
        "--strict",
        "--report",
        str(report9),
    ]
    scenarios.append(execute_scenario("evaluation_quality_missing_ci", ["missing_primary_metric_ci"], cmd9, report9))

    # 10) Evaluation quality: remove baselines.
    eval_no_baseline = copy.deepcopy(eval_report)
    eval_no_baseline.pop("baselines", None)
    eval_no_baseline_path = tmp_root / "evaluation_report.no_baseline.json"
    write_json(eval_no_baseline_path, eval_no_baseline)
    report10 = tmp_root / "evaluation_quality.no_baseline.report.json"
    cmd10 = [
        sys.executable,
        str(SCRIPTS_ROOT / "evaluation_quality_gate.py"),
        "--evaluation-report",
        str(eval_no_baseline_path),
        "--metric-name",
        "roc_auc",
        "--metric-path",
        "metrics.roc_auc",
        "--primary-metric",
        str(float(eval_report.get("metrics", {}).get("roc_auc", 0.0))),
        "--strict",
        "--report",
        str(report10),
    ]
    scenarios.append(
        execute_scenario(
            "evaluation_quality_missing_baseline",
            ["missing_baseline_metrics"],
            cmd10,
            report10,
        )
    )

    # 11) Execution log attestation tampering.
    attestation_spec_log = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_log = absolutize_attestation_paths(attestation_spec_log, cfg_dir)
    log_block = attestation_abs_log.get("execution_log_attestation")
    if not isinstance(log_block, dict):
        raise RuntimeError("execution_log_attestation block missing in attestation spec.")
    log_record_file = Path(str(log_block["record_file"]))
    tampered_log_record = tmp_root / "attestation_execution_log_record.tampered.json"
    tampered_log_payload = load_json(log_record_file)
    tampered_log_payload["artifact_sha256"] = "0" * 64
    write_json(tampered_log_record, tampered_log_payload)
    log_block["record_file"] = str(tampered_log_record)
    attestation_log_bad = tmp_root / "execution_attestation.log.bad.json"
    write_json(attestation_log_bad, attestation_abs_log)
    report11 = tmp_root / "execution_attestation.log.bad.report.json"
    cmd11 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_log_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report11),
    ]
    scenarios.append(
        execute_scenario(
            "execution_log_attestation_tamper",
            ["signature_verification_failed", "execution_log_artifact_hash_mismatch"],
            cmd11,
            report11,
        )
    )

    # 12) Witness quorum tampering.
    attestation_spec_witness = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_witness = absolutize_attestation_paths(attestation_spec_witness, cfg_dir)
    witness_block = attestation_abs_witness.get("witness_quorum")
    if not isinstance(witness_block, dict):
        raise RuntimeError("witness_quorum block missing in attestation spec.")
    witness_records = witness_block.get("records")
    if not isinstance(witness_records, list) or not witness_records or not isinstance(witness_records[0], dict):
        raise RuntimeError("witness_quorum.records missing in attestation spec.")
    witness_record_file = Path(str(witness_records[0]["record_file"]))
    tampered_witness_record = tmp_root / "attestation_witness_record_1.tampered.json"
    tampered_witness_payload = load_json(witness_record_file)
    tampered_witness_payload["payload_sha256"] = "1" * 64
    write_json(tampered_witness_record, tampered_witness_payload)
    witness_records[0]["record_file"] = str(tampered_witness_record)
    attestation_witness_bad = tmp_root / "execution_attestation.witness.bad.json"
    write_json(attestation_witness_bad, attestation_abs_witness)
    report12 = tmp_root / "execution_attestation.witness.bad.report.json"
    cmd12 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_witness_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report12),
    ]
    scenarios.append(
        execute_scenario(
            "witness_quorum_tamper",
            ["signature_verification_failed", "witness_quorum_not_met"],
            cmd12,
            report12,
        )
    )

    # 13) Witness quorum policy disabled while quorum block exists.
    attestation_spec_witness_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_witness_policy = absolutize_attestation_paths(attestation_spec_witness_policy, cfg_dir)
    assurance_policy = attestation_abs_witness_policy.get("assurance_policy")
    if not isinstance(assurance_policy, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy["require_witness_quorum"] = False
    attestation_witness_policy_bad = tmp_root / "execution_attestation.witness.policy.bad.json"
    write_json(attestation_witness_policy_bad, attestation_abs_witness_policy)
    report13 = tmp_root / "execution_attestation.witness.policy.bad.report.json"
    cmd13 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_witness_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report13),
    ]
    scenarios.append(
        execute_scenario(
            "witness_quorum_policy_disabled",
            ["witness_quorum_policy_disabled"],
            cmd13,
            report13,
        )
    )

    # 14) Witness min-count mismatch between policy and quorum block.
    attestation_spec_witness_min = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_witness_min = absolutize_attestation_paths(attestation_spec_witness_min, cfg_dir)
    assurance_policy_min = attestation_abs_witness_min.get("assurance_policy")
    witness_block_min = attestation_abs_witness_min.get("witness_quorum")
    if not isinstance(assurance_policy_min, dict) or not isinstance(witness_block_min, dict):
        raise RuntimeError("witness quorum/policy blocks missing in attestation spec.")
    assurance_policy_min["require_witness_quorum"] = True
    assurance_policy_min["min_witness_count"] = 2
    witness_block_min["min_witness_count"] = 1
    attestation_witness_min_bad = tmp_root / "execution_attestation.witness.min.bad.json"
    write_json(attestation_witness_min_bad, attestation_abs_witness_min)
    report14 = tmp_root / "execution_attestation.witness.min.bad.report.json"
    cmd14 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_witness_min_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report14),
    ]
    scenarios.append(
        execute_scenario(
            "witness_quorum_min_count_mismatch",
            ["witness_min_count_mismatch"],
            cmd14,
            report14,
        )
    )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_id": args.case_id,
        "overall_status": "pass" if all(s.passed for s in scenarios) else "fail",
        "scenario_count": len(scenarios),
        "passed_count": sum(1 for s in scenarios if s.passed),
        "results": [
            {
                "name": s.name,
                "passed": s.passed,
                "expected_codes": s.expected_codes,
                "observed_codes": s.observed_codes,
                "exit_code": s.exit_code,
                "report_path": s.report_path,
                "command": s.command,
                "stdout_tail": s.stdout_tail,
                "stderr_tail": s.stderr_tail,
            }
            for s in scenarios
        ],
    }

    output_path = (EXPERIMENT_ROOT / args.output).resolve()
    write_json(output_path, summary)
    print(f"Adversarial summary: {output_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if summary["overall_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
