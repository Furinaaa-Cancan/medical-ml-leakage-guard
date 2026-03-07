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
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


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


def sign_file(private_key: Path, input_file: Path, signature_file: Path) -> None:
    signature_file.parent.mkdir(parents=True, exist_ok=True)
    proc = run_cmd(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            str(private_key),
            "-out",
            str(signature_file),
            str(input_file),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"openssl signing failed for {input_file}: returncode={proc.returncode}, stderr={proc.stderr[-400:]}"
        )


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


def absolutize_request_paths(request: Dict[str, Any], request_dir: Path) -> Dict[str, Any]:
    cloned = copy.deepcopy(request)

    def _abs(raw: Any) -> Any:
        if not isinstance(raw, str) or not raw.strip():
            return raw
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (request_dir / p).resolve()
        else:
            p = p.resolve()
        return str(p)

    for key in (
        "phenotype_definition_spec",
        "feature_lineage_spec",
        "feature_group_spec",
        "split_protocol_spec",
        "imbalance_policy_spec",
        "missingness_policy_spec",
        "tuning_protocol_spec",
        "performance_policy_spec",
        "reporting_bias_checklist_spec",
        "execution_attestation_spec",
        "model_selection_report_file",
        "feature_engineering_report_file",
        "distribution_report_file",
        "robustness_report_file",
        "seed_sensitivity_report_file",
        "evaluation_report_file",
        "prediction_trace_file",
        "external_cohort_spec",
        "external_validation_report_file",
        "ci_matrix_report_file",
        "permutation_null_metrics_file",
    ):
        if key in cloned:
            cloned[key] = _abs(cloned.get(key))

    split_paths = cloned.get("split_paths")
    if isinstance(split_paths, dict):
        for split in ("train", "valid", "test"):
            if split in split_paths:
                split_paths[split] = _abs(split_paths.get(split))
        cloned["split_paths"] = split_paths
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

    matched = all(code in observed_codes for code in expected_codes)
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


def summarize_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "n": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    lo = float(min(values))
    hi = float(max(values))
    return {
        "mean": mean,
        "std": std,
        "min": lo,
        "max": hi,
        "range": float(hi - lo),
        "n": int(len(values)),
    }


def main() -> int:
    args = parse_args()

    case_root = EXPERIMENT_ROOT / args.case_id
    cfg_dir = case_root / "configs"
    data_dir = case_root / "data"
    evidence_dir = case_root / "evidence"
    keys_dir = case_root / "keys"
    request_path = cfg_dir / "request.json"
    if not request_path.exists():
        raise SystemExit(
            f"Case artifacts not found: {request_path}\n"
            "Run experiments/authority-e2e/run_authority_e2e.py first."
        )

    request = load_json(request_path)
    request_abs = absolutize_request_paths(request, cfg_dir)
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
    prediction_trace_path = evidence_dir / "prediction_trace.csv.gz"
    external_validation_report_path = evidence_dir / "external_validation_report.json"
    feature_engineering_report_path = evidence_dir / "feature_engineering_report.json"
    distribution_report_path = evidence_dir / "distribution_report.json"
    ci_matrix_report_path = evidence_dir / "ci_matrix_report.json"

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

    # 15) Timestamp trust policy downgrade.
    attestation_spec_ts_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_ts_policy = absolutize_attestation_paths(attestation_spec_ts_policy, cfg_dir)
    assurance_policy_ts = attestation_abs_ts_policy.get("assurance_policy")
    if not isinstance(assurance_policy_ts, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_ts["require_timestamp_trust"] = False
    attestation_abs_ts_policy.pop("timestamp_trust", None)
    attestation_ts_policy_bad = tmp_root / "execution_attestation.timestamp.policy.bad.json"
    write_json(attestation_ts_policy_bad, attestation_abs_ts_policy)
    report15 = tmp_root / "execution_attestation.timestamp.policy.bad.report.json"
    cmd15 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_ts_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report15),
    ]
    scenarios.append(
        execute_scenario(
            "timestamp_policy_disabled",
            ["publication_policy_disabled"],
            cmd15,
            report15,
        )
    )

    # 16) Transparency log policy downgrade.
    attestation_spec_tr_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_tr_policy = absolutize_attestation_paths(attestation_spec_tr_policy, cfg_dir)
    assurance_policy_tr = attestation_abs_tr_policy.get("assurance_policy")
    if not isinstance(assurance_policy_tr, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_tr["require_transparency_log"] = False
    assurance_policy_tr["require_transparency_log_signature"] = False
    attestation_abs_tr_policy.pop("transparency_log", None)
    attestation_tr_policy_bad = tmp_root / "execution_attestation.transparency.policy.bad.json"
    write_json(attestation_tr_policy_bad, attestation_abs_tr_policy)
    report16 = tmp_root / "execution_attestation.transparency.policy.bad.report.json"
    cmd16 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_tr_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report16),
    ]
    scenarios.append(
        execute_scenario(
            "transparency_policy_disabled",
            ["publication_policy_disabled"],
            cmd16,
            report16,
        )
    )

    # 17) Execution receipt policy downgrade.
    attestation_spec_er_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_er_policy = absolutize_attestation_paths(attestation_spec_er_policy, cfg_dir)
    assurance_policy_er = attestation_abs_er_policy.get("assurance_policy")
    if not isinstance(assurance_policy_er, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_er["require_execution_receipt"] = False
    attestation_abs_er_policy.pop("execution_receipt", None)
    attestation_er_policy_bad = tmp_root / "execution_attestation.execution_receipt.policy.bad.json"
    write_json(attestation_er_policy_bad, attestation_abs_er_policy)
    report17 = tmp_root / "execution_attestation.execution_receipt.policy.bad.report.json"
    cmd17 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_er_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report17),
    ]
    scenarios.append(
        execute_scenario(
            "execution_receipt_policy_disabled",
            ["publication_policy_disabled"],
            cmd17,
            report17,
        )
    )

    # 18) Execution-log policy downgrade.
    attestation_spec_log_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_log_policy = absolutize_attestation_paths(attestation_spec_log_policy, cfg_dir)
    assurance_policy_log = attestation_abs_log_policy.get("assurance_policy")
    if not isinstance(assurance_policy_log, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_log["require_execution_log_attestation"] = False
    attestation_abs_log_policy.pop("execution_log_attestation", None)
    attestation_log_policy_bad = tmp_root / "execution_attestation.execution_log.policy.bad.json"
    write_json(attestation_log_policy_bad, attestation_abs_log_policy)
    report18 = tmp_root / "execution_attestation.execution_log.policy.bad.report.json"
    cmd18 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_log_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report18),
    ]
    scenarios.append(
        execute_scenario(
            "execution_log_policy_disabled",
            ["publication_policy_disabled"],
            cmd18,
            report18,
        )
    )

    # 19) Witness independence policy downgrade.
    attestation_spec_wi_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_wi_policy = absolutize_attestation_paths(attestation_spec_wi_policy, cfg_dir)
    assurance_policy_wi = attestation_abs_wi_policy.get("assurance_policy")
    if not isinstance(assurance_policy_wi, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_wi["require_independent_witness_keys"] = False
    assurance_policy_wi["require_witness_independence_from_signing"] = False
    attestation_wi_policy_bad = tmp_root / "execution_attestation.witness_independence.policy.bad.json"
    write_json(attestation_wi_policy_bad, attestation_abs_wi_policy)
    report19 = tmp_root / "execution_attestation.witness_independence.policy.bad.report.json"
    cmd19 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_wi_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report19),
    ]
    scenarios.append(
        execute_scenario(
            "witness_independence_policy_disabled",
            ["publication_policy_disabled"],
            cmd19,
            report19,
        )
    )

    # 20) Cross-role distinctness policy downgrade.
    attestation_spec_distinct_policy = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_distinct_policy = absolutize_attestation_paths(attestation_spec_distinct_policy, cfg_dir)
    assurance_policy_distinct = attestation_abs_distinct_policy.get("assurance_policy")
    if not isinstance(assurance_policy_distinct, dict):
        raise RuntimeError("assurance_policy block missing in attestation spec.")
    assurance_policy_distinct["require_distinct_authority_roles"] = False
    attestation_distinct_policy_bad = tmp_root / "execution_attestation.distinct_roles.policy.bad.json"
    write_json(attestation_distinct_policy_bad, attestation_abs_distinct_policy)
    report20 = tmp_root / "execution_attestation.distinct_roles.policy.bad.report.json"
    cmd20 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_distinct_policy_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report20),
    ]
    scenarios.append(
        execute_scenario(
            "distinct_authority_roles_policy_disabled",
            ["publication_policy_disabled"],
            cmd20,
            report20,
        )
    )

    # 21) Cross-role authority/key overlap (timestamp and execution receipt share the same authority/key).
    attestation_spec_roles = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_roles = absolutize_attestation_paths(attestation_spec_roles, cfg_dir)
    timestamp_block_roles = attestation_abs_roles.get("timestamp_trust")
    execution_block_roles = attestation_abs_roles.get("execution_receipt")
    if not isinstance(timestamp_block_roles, dict) or not isinstance(execution_block_roles, dict):
        raise RuntimeError("timestamp_trust/execution_receipt block missing in attestation spec.")

    timestamp_authority_id = str(timestamp_block_roles.get("authority_id", "")).strip()
    timestamp_pub = Path(str(timestamp_block_roles.get("public_key_file", "")))
    timestamp_fp = str(timestamp_block_roles.get("public_key_fingerprint_sha256", "")).strip()
    timestamp_priv = (keys_dir / "timestamp_priv.pem").resolve()
    if not timestamp_authority_id or not timestamp_fp or not timestamp_pub.exists() or not timestamp_priv.exists():
        raise RuntimeError("timestamp authority key material missing for cross-role overlap scenario.")

    execution_record_src = Path(str(execution_block_roles.get("record_file", "")))
    if not execution_record_src.exists():
        raise RuntimeError("execution_receipt record_file missing in attestation spec.")
    execution_record_tampered = load_json(execution_record_src)
    execution_record_tampered["authority_id"] = timestamp_authority_id
    execution_record_overlap = tmp_root / "attestation_execution_receipt_record.role_overlap.json"
    write_json(execution_record_overlap, execution_record_tampered)
    execution_sig_overlap = tmp_root / "attestation_execution_receipt_record.role_overlap.sig"
    sign_file(timestamp_priv, execution_record_overlap, execution_sig_overlap)

    execution_block_roles["authority_id"] = timestamp_authority_id
    execution_block_roles["record_file"] = str(execution_record_overlap)
    execution_block_roles["signature_file"] = str(execution_sig_overlap)
    execution_block_roles["public_key_file"] = str(timestamp_pub)
    execution_block_roles["public_key_fingerprint_sha256"] = timestamp_fp

    attestation_roles_bad = tmp_root / "execution_attestation.role_overlap.bad.json"
    write_json(attestation_roles_bad, attestation_abs_roles)
    report21 = tmp_root / "execution_attestation.role_overlap.bad.report.json"
    cmd21 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_roles_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report21),
    ]
    scenarios.append(
        execute_scenario(
            "cross_role_authority_overlap",
            ["authority_roles_not_distinct"],
            cmd21,
            report21,
        )
    )

    # 22) Model-selection report leaks test-derived ranking fields.
    model_selection = load_json(evidence_dir / "model_selection_report.json")
    model_selection_bad = copy.deepcopy(model_selection)
    candidates_bad = model_selection_bad.get("candidates")
    if isinstance(candidates_bad, list) and candidates_bad and isinstance(candidates_bad[0], dict):
        candidates_bad[0]["test_rank"] = 1
    else:
        model_selection_bad["test_rank"] = 1
    model_selection_bad_path = tmp_root / "model_selection.bad.json"
    write_json(model_selection_bad_path, model_selection_bad)
    report22 = tmp_root / "model_selection_audit.bad.report.json"
    cmd22 = [
        sys.executable,
        str(SCRIPTS_ROOT / "model_selection_audit_gate.py"),
        "--model-selection-report",
        str(model_selection_bad_path),
        "--tuning-spec",
        str(cfg_dir / "tuning_protocol.json"),
        "--expected-primary-metric",
        "pr_auc",
        "--strict",
        "--report",
        str(report22),
    ]
    scenarios.append(execute_scenario("model_selection_test_ranking", ["test_data_usage_detected"], cmd22, report22))

    # 23) Clinical metrics: missing required metric.
    eval_report_missing_metric = load_json(evidence_dir / "evaluation_report.json")
    eval_report_missing_metric_bad = copy.deepcopy(eval_report_missing_metric)
    test_metrics_block = (
        eval_report_missing_metric_bad.get("split_metrics", {})
        .get("test", {})
        .get("metrics")
    )
    if isinstance(test_metrics_block, dict):
        test_metrics_block.pop("npv", None)
    eval_missing_metric_path = tmp_root / "evaluation_report.missing_metric.json"
    write_json(eval_missing_metric_path, eval_report_missing_metric_bad)
    report23 = tmp_root / "clinical_metrics.missing_metric.report.json"
    cmd23 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_missing_metric_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report23),
    ]
    scenarios.append(execute_scenario("clinical_metrics_missing_required", ["missing_required_metric"], cmd23, report23))

    # 24) Clinical metrics: precision/ppv mismatch.
    eval_report_ppv_bad = load_json(evidence_dir / "evaluation_report.json")
    eval_report_ppv_bad_payload = copy.deepcopy(eval_report_ppv_bad)
    test_metrics_ppv = (
        eval_report_ppv_bad_payload.get("split_metrics", {})
        .get("test", {})
        .get("metrics")
    )
    if isinstance(test_metrics_ppv, dict):
        test_metrics_ppv["ppv"] = float(test_metrics_ppv.get("precision", 0.5)) + 0.1
    eval_ppv_bad_path = tmp_root / "evaluation_report.ppv_mismatch.json"
    write_json(eval_ppv_bad_path, eval_report_ppv_bad_payload)
    report24 = tmp_root / "clinical_metrics.ppv_mismatch.report.json"
    cmd24 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_ppv_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report24),
    ]
    scenarios.append(execute_scenario("clinical_metrics_ppv_precision_mismatch", ["metric_formula_mismatch"], cmd24, report24))

    # 25) Threshold selection uses forbidden test split.
    eval_report_threshold_bad = load_json(evidence_dir / "evaluation_report.json")
    eval_report_threshold_bad_payload = copy.deepcopy(eval_report_threshold_bad)
    threshold_block = eval_report_threshold_bad_payload.get("threshold_selection")
    if isinstance(threshold_block, dict):
        threshold_block["selection_split"] = "test"
    eval_threshold_bad_path = tmp_root / "evaluation_report.threshold_test_split.json"
    write_json(eval_threshold_bad_path, eval_report_threshold_bad_payload)
    report25 = tmp_root / "clinical_metrics.threshold_test_split.report.json"
    cmd25 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_threshold_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report25),
    ]
    scenarios.append(
        execute_scenario(
            "threshold_selection_on_test_split",
            ["test_split_used_for_threshold_selection"],
            cmd25,
            report25,
        )
    )

    # 26) Generalization gap exceeds fail threshold.
    eval_report_gap_bad = load_json(evidence_dir / "evaluation_report.json")
    eval_report_gap_bad_payload = copy.deepcopy(eval_report_gap_bad)
    split_metrics_gap = eval_report_gap_bad_payload.get("split_metrics")
    if isinstance(split_metrics_gap, dict):
        train_metrics_gap = split_metrics_gap.get("train", {}).get("metrics")
        valid_metrics_gap = split_metrics_gap.get("valid", {}).get("metrics")
        if isinstance(train_metrics_gap, dict):
            train_metrics_gap["pr_auc"] = 0.95
        if isinstance(valid_metrics_gap, dict):
            valid_metrics_gap["pr_auc"] = 0.70
    eval_gap_bad_path = tmp_root / "evaluation_report.gap_exceeds.json"
    write_json(eval_gap_bad_path, eval_report_gap_bad_payload)
    report26 = tmp_root / "generalization_gap.exceeds.report.json"
    cmd26 = [
        sys.executable,
        str(SCRIPTS_ROOT / "generalization_gap_gate.py"),
        "--evaluation-report",
        str(eval_gap_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report26),
    ]
    scenarios.append(execute_scenario("generalization_gap_exceeds", ["overfit_gap_exceeds_threshold"], cmd26, report26))

    # 27) MICE scale guard violation (oversized with missing fallback evidence).
    missingness_bad = load_json(cfg_dir / "missingness_policy.json")
    missingness_bad_payload = copy.deepcopy(missingness_bad)
    missingness_bad_payload["strategy"] = "mice_with_scale_guard"
    missingness_bad_payload["mice_max_rows"] = 10
    missingness_bad_payload["mice_max_cols"] = 5
    missingness_bad_payload["scale_guard_evidence"] = {
        "fallback_triggered": False,
        "fallback_strategy": "mice",
        "train_rows_seen": 1,
        "feature_count_seen": 1,
    }
    missingness_bad_path = tmp_root / "missingness_policy.mice_guard.bad.json"
    write_json(missingness_bad_path, missingness_bad_payload)
    report27 = tmp_root / "missingness_policy.mice_guard.bad.report.json"
    cmd27 = [
        sys.executable,
        str(SCRIPTS_ROOT / "missingness_policy_gate.py"),
        "--policy-spec",
        str(missingness_bad_path),
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
        str(report27),
    ]
    scenarios.append(execute_scenario("mice_scale_guard_violation", ["mice_scale_guard_violation"], cmd27, report27))

    # 28) Execution-log run nonce mismatch with valid signature (non-repudiation binding check).
    attestation_spec_nonce = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_nonce = absolutize_attestation_paths(attestation_spec_nonce, cfg_dir)
    log_block_nonce = attestation_abs_nonce.get("execution_log_attestation")
    if not isinstance(log_block_nonce, dict):
        raise RuntimeError("execution_log_attestation block missing in attestation spec.")
    log_record_nonce_src = Path(str(log_block_nonce.get("record_file", "")))
    if not log_record_nonce_src.exists():
        raise RuntimeError("execution_log_attestation.record_file missing in attestation spec.")
    log_record_nonce_tampered = load_json(log_record_nonce_src)
    log_record_nonce_tampered["run_nonce"] = "0" * 32
    tampered_nonce_record = tmp_root / "attestation_execution_log_record.nonce_mismatch.json"
    write_json(tampered_nonce_record, log_record_nonce_tampered)
    tampered_nonce_sig = tmp_root / "attestation_execution_log_record.nonce_mismatch.sig"
    sign_file((keys_dir / "execution_log_priv.pem").resolve(), tampered_nonce_record, tampered_nonce_sig)
    log_block_nonce["record_file"] = str(tampered_nonce_record)
    log_block_nonce["signature_file"] = str(tampered_nonce_sig)
    attestation_nonce_bad = tmp_root / "execution_attestation.log.nonce_mismatch.bad.json"
    write_json(attestation_nonce_bad, attestation_abs_nonce)
    report28 = tmp_root / "execution_attestation.log.nonce_mismatch.report.json"
    cmd28 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_nonce_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report28),
    ]
    scenarios.append(
        execute_scenario(
            "execution_log_run_nonce_mismatch",
            ["execution_log_run_nonce_mismatch"],
            cmd28,
            report28,
        )
    )

    # 29) Performance policy omits mandatory clinical metrics.
    performance_policy_bad = load_json(cfg_dir / "performance_policy.json")
    performance_policy_bad_payload = copy.deepcopy(performance_policy_bad)
    performance_policy_bad_payload["required_metrics"] = ["pr_auc"]
    performance_policy_bad_path = tmp_root / "performance_policy.missing_required_metrics.json"
    write_json(performance_policy_bad_path, performance_policy_bad_payload)
    report29 = tmp_root / "clinical_metrics.performance_policy_missing_required.report.json"
    cmd29 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--performance-policy",
        str(performance_policy_bad_path),
        "--strict",
        "--report",
        str(report29),
    ]
    scenarios.append(
        execute_scenario(
            "performance_policy_missing_required_metrics",
            ["performance_policy_missing_required_metric"],
            cmd29,
            report29,
        )
    )

    # 30) Execution-log bound artifact hash mismatch (signature still valid).
    attestation_spec_bound = load_json(cfg_dir / "execution_attestation.json")
    attestation_abs_bound = absolutize_attestation_paths(attestation_spec_bound, cfg_dir)
    log_block_bound = attestation_abs_bound.get("execution_log_attestation")
    if not isinstance(log_block_bound, dict):
        raise RuntimeError("execution_log_attestation block missing in attestation spec.")
    log_record_bound_src = Path(str(log_block_bound.get("record_file", "")))
    if not log_record_bound_src.exists():
        raise RuntimeError("execution_log_attestation.record_file missing in attestation spec.")
    log_record_bound_tampered = load_json(log_record_bound_src)
    log_record_bound_tampered["model_selection_report_sha256"] = "f" * 64
    tampered_bound_record = tmp_root / "attestation_execution_log_record.bound_hash_mismatch.json"
    write_json(tampered_bound_record, log_record_bound_tampered)
    tampered_bound_sig = tmp_root / "attestation_execution_log_record.bound_hash_mismatch.sig"
    sign_file((keys_dir / "execution_log_priv.pem").resolve(), tampered_bound_record, tampered_bound_sig)
    log_block_bound["record_file"] = str(tampered_bound_record)
    log_block_bound["signature_file"] = str(tampered_bound_sig)
    attestation_bound_bad = tmp_root / "execution_attestation.log.bound_hash_mismatch.bad.json"
    write_json(attestation_bound_bad, attestation_abs_bound)
    report30 = tmp_root / "execution_attestation.log.bound_hash_mismatch.report.json"
    cmd30 = [
        sys.executable,
        str(SCRIPTS_ROOT / "execution_attestation_gate.py"),
        "--attestation-spec",
        str(attestation_bound_bad),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--study-id",
        study_id,
        "--run-id",
        run_id,
        "--strict",
        "--report",
        str(report30),
    ]
    scenarios.append(
        execute_scenario(
            "execution_log_related_hash_mismatch",
            ["execution_log_related_hash_mismatch"],
            cmd30,
            report30,
        )
    )

    # 31) Selection-data mismatch between model-selection report and tuning spec.
    model_selection_mismatch = load_json(evidence_dir / "model_selection_report.json")
    model_selection_mismatch_payload = copy.deepcopy(model_selection_mismatch)
    selection_policy_mismatch = model_selection_mismatch_payload.get("selection_policy")
    if not isinstance(selection_policy_mismatch, dict):
        selection_policy_mismatch = {}
        model_selection_mismatch_payload["selection_policy"] = selection_policy_mismatch
    tuning_spec_current = load_json(cfg_dir / "tuning_protocol.json")
    configured_selection_data = str(tuning_spec_current.get("model_selection_data", "cv_inner")).strip().lower()
    mismatch_selection_data = "valid" if configured_selection_data != "valid" else "cv_inner"
    selection_policy_mismatch["selection_data"] = mismatch_selection_data
    model_selection_mismatch_path = tmp_root / "model_selection.selection_data_mismatch.json"
    write_json(model_selection_mismatch_path, model_selection_mismatch_payload)
    report31 = tmp_root / "model_selection_audit.selection_data_mismatch.report.json"
    cmd31 = [
        sys.executable,
        str(SCRIPTS_ROOT / "model_selection_audit_gate.py"),
        "--model-selection-report",
        str(model_selection_mismatch_path),
        "--tuning-spec",
        str(cfg_dir / "tuning_protocol.json"),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--expected-primary-metric",
        "pr_auc",
        "--strict",
        "--report",
        str(report31),
    ]
    scenarios.append(
        execute_scenario(
            "selection_data_spec_mismatch",
            ["selection_data_spec_mismatch"],
            cmd31,
            report31,
        )
    )

    # 32) Threshold selection split mismatch between evaluation report and performance policy.
    eval_threshold_policy_bad = load_json(evidence_dir / "evaluation_report.json")
    eval_threshold_policy_bad_payload = copy.deepcopy(eval_threshold_policy_bad)
    performance_policy_payload = load_json(cfg_dir / "performance_policy.json")
    policy_selection_split = "valid"
    threshold_policy_cfg = performance_policy_payload.get("threshold_policy")
    if isinstance(threshold_policy_cfg, dict):
        cfg_split = str(threshold_policy_cfg.get("selection_split", "")).strip().lower()
        if cfg_split in {"valid", "cv_inner", "nested_cv"}:
            policy_selection_split = cfg_split
    mismatch_selection_split = "valid" if policy_selection_split != "valid" else "cv_inner"
    threshold_block_policy = eval_threshold_policy_bad_payload.get("threshold_selection")
    if isinstance(threshold_block_policy, dict):
        threshold_block_policy["selection_split"] = mismatch_selection_split
    eval_threshold_policy_bad_path = tmp_root / "evaluation_report.threshold_policy_mismatch.json"
    write_json(eval_threshold_policy_bad_path, eval_threshold_policy_bad_payload)
    report32 = tmp_root / "clinical_metrics.threshold_policy_mismatch.report.json"
    cmd32 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_threshold_policy_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report32),
    ]
    scenarios.append(
        execute_scenario(
            "threshold_selection_policy_mismatch",
            ["threshold_selection_policy_mismatch"],
            cmd32,
            report32,
        )
    )

    # 33) Request contract: model ID mismatch between selection and evaluation artifacts.
    eval_model_id_bad = copy.deepcopy(eval_report)
    eval_model_id_bad["model_id"] = "non_selected_model_for_adversarial_check"
    eval_model_id_bad_path = tmp_root / "evaluation_report.model_id_mismatch.json"
    write_json(eval_model_id_bad_path, eval_model_id_bad)
    request_model_id_bad = copy.deepcopy(request_abs)
    request_model_id_bad["evaluation_report_file"] = str(eval_model_id_bad_path)
    request_model_id_bad_path = tmp_root / "request.model_id_mismatch.json"
    write_json(request_model_id_bad_path, request_model_id_bad)
    report33 = tmp_root / "request_contract.model_id_mismatch.report.json"
    cmd33 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_model_id_bad_path),
        "--strict",
        "--report",
        str(report33),
    ]
    scenarios.append(
        execute_scenario(
            "request_contract_model_id_mismatch",
            ["model_id_cross_artifact_mismatch"],
            cmd33,
            report33,
        )
    )

    # 34) Request contract: split fingerprint mismatch across artifacts.
    eval_fingerprint_bad = copy.deepcopy(eval_report)
    eval_meta_bad = eval_fingerprint_bad.get("metadata")
    eval_fp_bad = eval_meta_bad.get("data_fingerprints") if isinstance(eval_meta_bad, dict) else None
    if isinstance(eval_fp_bad, dict) and isinstance(eval_fp_bad.get("train"), dict):
        eval_fp_bad["train"]["sha256"] = "e" * 64
    eval_fingerprint_bad_path = tmp_root / "evaluation_report.fingerprint_mismatch.json"
    write_json(eval_fingerprint_bad_path, eval_fingerprint_bad)
    request_fingerprint_bad = copy.deepcopy(request_abs)
    request_fingerprint_bad["evaluation_report_file"] = str(eval_fingerprint_bad_path)
    request_fingerprint_bad_path = tmp_root / "request.fingerprint_mismatch.json"
    write_json(request_fingerprint_bad_path, request_fingerprint_bad)
    report34 = tmp_root / "request_contract.fingerprint_mismatch.report.json"
    cmd34 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_fingerprint_bad_path),
        "--strict",
        "--report",
        str(report34),
    ]
    scenarios.append(
        execute_scenario(
            "request_contract_fingerprint_mismatch",
            ["data_fingerprint_cross_artifact_mismatch"],
            cmd34,
            report34,
        )
    )

    # 35) Request contract: declared final refit scope mismatches evaluated fit split.
    tuning_refit_bad = load_json(cfg_dir / "tuning_protocol.json")
    tuning_refit_bad_payload = copy.deepcopy(tuning_refit_bad)
    tuning_refit_bad_payload["final_model_refit_scope"] = "train_plus_valid_no_test"
    tuning_refit_bad_path = tmp_root / "tuning_protocol.refit_scope_mismatch.json"
    write_json(tuning_refit_bad_path, tuning_refit_bad_payload)
    request_refit_bad = copy.deepcopy(request_abs)
    request_refit_bad["tuning_protocol_spec"] = str(tuning_refit_bad_path)
    request_refit_bad_path = tmp_root / "request.refit_scope_mismatch.json"
    write_json(request_refit_bad_path, request_refit_bad)
    report35 = tmp_root / "request_contract.refit_scope_mismatch.report.json"
    cmd35 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_refit_bad_path),
        "--strict",
        "--report",
        str(report35),
    ]
    scenarios.append(
        execute_scenario(
            "request_contract_refit_scope_mismatch",
            ["final_model_refit_scope_mismatch"],
            cmd35,
            report35,
        )
    )

    # 36) Seed stability exceeds thresholds.
    seed_report = load_json(evidence_dir / "seed_sensitivity_report.json")
    seed_report_bad = copy.deepcopy(seed_report)
    bad_rows = seed_report_bad.get("per_seed_results")
    if isinstance(bad_rows, list) and len(bad_rows) >= 5:
        forced_pr_auc = [0.99, 0.92, 0.84, 0.75, 0.64]
        forced_f2 = [0.97, 0.91, 0.83, 0.74, 0.62]
        forced_brier = [0.02, 0.06, 0.11, 0.17, 0.24]
        for i, row in enumerate(bad_rows[:5]):
            if not isinstance(row, dict):
                continue
            metrics = row.get("test_metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                row["test_metrics"] = metrics
            metrics["pr_auc"] = float(forced_pr_auc[i])
            metrics["f2_beta"] = float(forced_f2[i])
            metrics["brier"] = float(forced_brier[i])
        pr_auc_values = [
            float(r.get("test_metrics", {}).get("pr_auc"))
            for r in bad_rows
            if isinstance(r, dict)
            and isinstance(r.get("test_metrics"), dict)
            and isinstance(r.get("test_metrics", {}).get("pr_auc"), (int, float))
        ]
        f2_values = [
            float(r.get("test_metrics", {}).get("f2_beta"))
            for r in bad_rows
            if isinstance(r, dict)
            and isinstance(r.get("test_metrics"), dict)
            and isinstance(r.get("test_metrics", {}).get("f2_beta"), (int, float))
        ]
        brier_values = [
            float(r.get("test_metrics", {}).get("brier"))
            for r in bad_rows
            if isinstance(r, dict)
            and isinstance(r.get("test_metrics"), dict)
            and isinstance(r.get("test_metrics", {}).get("brier"), (int, float))
        ]
        seed_report_bad["summary"] = {
            "pr_auc": summarize_metric(pr_auc_values),
            "f2_beta": summarize_metric(f2_values),
            "brier": summarize_metric(brier_values),
        }
    seed_report_bad_path = tmp_root / "seed_sensitivity.unstable.json"
    write_json(seed_report_bad_path, seed_report_bad)
    report36 = tmp_root / "seed_stability.unstable.report.json"
    cmd36 = [
        sys.executable,
        str(SCRIPTS_ROOT / "seed_stability_gate.py"),
        "--seed-sensitivity-report",
        str(seed_report_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report36),
    ]
    scenarios.append(
        execute_scenario(
            "seed_stability_exceeds",
            ["seed_stability_exceeds_threshold"],
            cmd36,
            report36,
        )
    )

    # 37) Robustness gate: severe time-slice/group degradation beyond policy thresholds.
    robustness_report = load_json(evidence_dir / "robustness_report.json")
    robustness_bad = copy.deepcopy(robustness_report)
    time_rows = robustness_bad.get("time_slices", {}).get("slices")
    group_rows = robustness_bad.get("patient_hash_groups", {}).get("groups")
    if isinstance(time_rows, list) and time_rows:
        for i, row in enumerate(time_rows):
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                row["metrics"] = metrics
            metrics["pr_auc"] = 0.05
    if isinstance(group_rows, list) and group_rows:
        for i, row in enumerate(group_rows):
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                row["metrics"] = metrics
            metrics["pr_auc"] = 0.05

    def _summ(rows: Any, overall: float) -> Dict[str, float]:
        vals = [
            float(r.get("metrics", {}).get("pr_auc"))
            for r in rows
            if isinstance(r, dict)
            and isinstance(r.get("metrics"), dict)
            and isinstance(r.get("metrics", {}).get("pr_auc"), (int, float))
        ]
        if not vals:
            return {"pr_auc_min": 0.0, "pr_auc_max": 0.0, "pr_auc_range": 0.0, "pr_auc_worst_drop_from_overall": 0.0, "n_rows": 0}
        mn = min(vals)
        mx = max(vals)
        return {
            "pr_auc_min": float(mn),
            "pr_auc_max": float(mx),
            "pr_auc_range": float(mx - mn),
            "pr_auc_worst_drop_from_overall": float(overall - mn),
            "n_rows": int(len(vals)),
        }

    overall_pr_auc = float(
        robustness_bad.get("overall_test_metrics", {}).get("pr_auc", 0.0)
        if isinstance(robustness_bad.get("overall_test_metrics"), dict)
        else 0.0
    )
    robustness_bad["summary"] = {
        "time_slices": _summ(time_rows if isinstance(time_rows, list) else [], overall_pr_auc),
        "patient_hash_groups": _summ(group_rows if isinstance(group_rows, list) else [], overall_pr_auc),
    }
    robustness_bad_path = tmp_root / "robustness.degraded.json"
    write_json(robustness_bad_path, robustness_bad)
    report37 = tmp_root / "robustness.degraded.report.json"
    cmd37 = [
        sys.executable,
        str(SCRIPTS_ROOT / "robustness_gate.py"),
        "--robustness-report",
        str(robustness_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report37),
    ]
    scenarios.append(
        execute_scenario(
            "robustness_degradation_exceeds",
            ["robustness_pr_auc_drop_exceeds_threshold"],
            cmd37,
            report37,
        )
    )

    # 38) Request contract: performance policy downgrade should fail (anti-relaxation guard).
    policy_downgrade = load_json(cfg_dir / "performance_policy.json")
    policy_downgrade_bad = copy.deepcopy(policy_downgrade)
    if isinstance(policy_downgrade_bad.get("clinical_floors"), dict):
        policy_downgrade_bad["clinical_floors"]["sensitivity_min"] = 0.70
        policy_downgrade_bad["clinical_floors"]["npv_min"] = 0.80
    if isinstance(policy_downgrade_bad.get("gap_thresholds"), dict):
        vt = policy_downgrade_bad["gap_thresholds"].get("valid_test")
        if isinstance(vt, dict) and isinstance(vt.get("pr_auc"), dict):
            vt["pr_auc"]["fail"] = 0.20
    policy_downgrade_bad["beta"] = 1.0
    policy_downgrade_bad_path = tmp_root / "performance_policy.downgrade.json"
    write_json(policy_downgrade_bad_path, policy_downgrade_bad)
    request_policy_downgrade = copy.deepcopy(request_abs)
    request_policy_downgrade["performance_policy_spec"] = str(policy_downgrade_bad_path)
    request_policy_downgrade_path = tmp_root / "request.performance_policy_downgrade.json"
    write_json(request_policy_downgrade_path, request_policy_downgrade)
    report38 = tmp_root / "request_contract.performance_policy_downgrade.report.json"
    cmd38 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_policy_downgrade_path),
        "--strict",
        "--report",
        str(report38),
    ]
    scenarios.append(
        execute_scenario(
            "performance_policy_downgrade_guard",
            ["performance_policy_downgrade"],
            cmd38,
            report38,
        )
    )

    # 39) Clinical metrics gate: confusion totals must match fingerprint row_count.
    eval_confusion_bad = copy.deepcopy(eval_report)
    split_test_bad = eval_confusion_bad.get("split_metrics", {}).get("test")
    if isinstance(split_test_bad, dict):
        cm_bad = split_test_bad.get("confusion_matrix")
        if isinstance(cm_bad, dict) and isinstance(cm_bad.get("tn"), int):
            cm_bad["tn"] = int(cm_bad["tn"]) + 1
    eval_confusion_bad_path = tmp_root / "evaluation_report.confusion_row_count_mismatch.json"
    write_json(eval_confusion_bad_path, eval_confusion_bad)
    report39 = tmp_root / "clinical_metrics.confusion_row_count_mismatch.report.json"
    cmd39 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_confusion_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report39),
    ]
    scenarios.append(
        execute_scenario(
            "clinical_metrics_confusion_row_count_mismatch",
            ["confusion_matrix_row_count_mismatch"],
            cmd39,
            report39,
        )
    )

    # 40) Prediction replay mismatch: tamper evaluation metric value.
    eval_replay_bad = copy.deepcopy(eval_report)
    split_test_replay = eval_replay_bad.get("split_metrics", {}).get("test")
    if isinstance(split_test_replay, dict):
        test_metrics_replay = split_test_replay.get("metrics")
        if isinstance(test_metrics_replay, dict) and isinstance(test_metrics_replay.get("pr_auc"), (int, float)):
            test_metrics_replay["pr_auc"] = float(min(0.999, float(test_metrics_replay["pr_auc"]) + 0.20))
    eval_replay_bad_path = tmp_root / "evaluation_report.prediction_replay_mismatch.json"
    write_json(eval_replay_bad_path, eval_replay_bad)
    report40 = tmp_root / "prediction_replay.metric_mismatch.report.json"
    cmd40 = [
        sys.executable,
        str(SCRIPTS_ROOT / "prediction_replay_gate.py"),
        "--evaluation-report",
        str(eval_replay_bad_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report40),
    ]
    scenarios.append(
        execute_scenario(
            "prediction_replay_metric_mismatch",
            ["prediction_metric_replay_mismatch"],
            cmd40,
            report40,
        )
    )

    # 41) Prediction replay row-count mismatch in evaluation fingerprints.
    eval_rowcount_bad = copy.deepcopy(eval_report)
    eval_rowcount_meta = eval_rowcount_bad.get("metadata")
    eval_rowcount_fp = eval_rowcount_meta.get("data_fingerprints") if isinstance(eval_rowcount_meta, dict) else None
    if isinstance(eval_rowcount_fp, dict) and isinstance(eval_rowcount_fp.get("test"), dict):
        test_rc = eval_rowcount_fp["test"].get("row_count")
        if isinstance(test_rc, (int, float)):
            eval_rowcount_fp["test"]["row_count"] = int(float(test_rc)) + 1
    eval_rowcount_bad_path = tmp_root / "evaluation_report.prediction_replay_rowcount_mismatch.json"
    write_json(eval_rowcount_bad_path, eval_rowcount_bad)
    report41 = tmp_root / "prediction_replay.rowcount_mismatch.report.json"
    cmd41 = [
        sys.executable,
        str(SCRIPTS_ROOT / "prediction_replay_gate.py"),
        "--evaluation-report",
        str(eval_rowcount_bad_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report41),
    ]
    scenarios.append(
        execute_scenario(
            "prediction_trace_rowcount_mismatch",
            ["prediction_trace_rowcount_mismatch"],
            cmd41,
            report41,
        )
    )

    # 42) Prediction replay score out of range.
    trace_out_of_range = pd.read_csv(prediction_trace_path)
    if not trace_out_of_range.empty:
        trace_out_of_range.loc[0, "y_score"] = 1.5
    trace_out_of_range_path = tmp_root / "prediction_trace.out_of_range.csv"
    trace_out_of_range.to_csv(trace_out_of_range_path, index=False)
    report42 = tmp_root / "prediction_replay.score_out_of_range.report.json"
    cmd42 = [
        sys.executable,
        str(SCRIPTS_ROOT / "prediction_replay_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--prediction-trace",
        str(trace_out_of_range_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report42),
    ]
    scenarios.append(
        execute_scenario(
            "prediction_score_out_of_range",
            ["prediction_score_out_of_range"],
            cmd42,
            report42,
        )
    )

    # 43) External validation missing file.
    report43 = tmp_root / "external_validation.missing.report.json"
    cmd43 = [
        sys.executable,
        str(SCRIPTS_ROOT / "external_validation_gate.py"),
        "--external-validation-report",
        str(tmp_root / "missing_external_validation_report.json"),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report43),
    ]
    scenarios.append(
        execute_scenario(
            "external_validation_missing",
            ["external_validation_missing"],
            cmd43,
            report43,
        )
    )

    # 44) External validation type coverage not met.
    external_type_bad = load_json(external_validation_report_path)
    external_type_bad_payload = copy.deepcopy(external_type_bad)
    ext_type_rows = external_type_bad_payload.get("cohorts")
    if isinstance(ext_type_rows, list):
        for row in ext_type_rows:
            if isinstance(row, dict):
                row["cohort_type"] = "unsupported_type"
    external_type_bad_path = tmp_root / "external_validation.type_coverage_missing.json"
    write_json(external_type_bad_path, external_type_bad_payload)
    report44 = tmp_root / "external_validation.type_coverage_missing.report.json"
    cmd44 = [
        sys.executable,
        str(SCRIPTS_ROOT / "external_validation_gate.py"),
        "--external-validation-report",
        str(external_type_bad_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report44),
    ]
    scenarios.append(
        execute_scenario(
            "external_validation_type_coverage_not_met",
            ["external_validation_type_coverage_not_met"],
            cmd44,
            report44,
        )
    )

    # 45) External validation transport drop exceeds threshold via inflated internal benchmark.
    eval_transport_bad = copy.deepcopy(eval_report)
    eval_transport_metrics = eval_transport_bad.get("metrics")
    if isinstance(eval_transport_metrics, dict):
        eval_transport_metrics["pr_auc"] = 1.0
        eval_transport_metrics["f2_beta"] = 1.0
        eval_transport_metrics["brier"] = 0.0
    eval_transport_bad_path = tmp_root / "evaluation_report.external_transport_drop.json"
    write_json(eval_transport_bad_path, eval_transport_bad)
    report45 = tmp_root / "external_validation.transport_drop.report.json"
    cmd45 = [
        sys.executable,
        str(SCRIPTS_ROOT / "external_validation_gate.py"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(eval_transport_bad_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report45),
    ]
    scenarios.append(
        execute_scenario(
            "external_validation_transport_drop_exceeds_threshold",
            ["external_validation_transport_drop_exceeds_threshold"],
            cmd45,
            report45,
        )
    )

    # 46) Calibration ECE exceeds threshold with adversarially inverted scores.
    trace_ece_bad = pd.read_csv(prediction_trace_path)
    if not trace_ece_bad.empty:
        trace_ece_bad["y_score"] = 1.0 - trace_ece_bad["y_true"].astype(float)
        trace_ece_bad["y_pred"] = (trace_ece_bad["y_score"] >= trace_ece_bad["selected_threshold"].astype(float)).astype(int)
    trace_ece_bad_path = tmp_root / "prediction_trace.calibration_ece_bad.csv"
    trace_ece_bad.to_csv(trace_ece_bad_path, index=False)
    report46 = tmp_root / "calibration_dca.ece_exceeds.report.json"
    cmd46 = [
        sys.executable,
        str(SCRIPTS_ROOT / "calibration_dca_gate.py"),
        "--prediction-trace",
        str(trace_ece_bad_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report46),
    ]
    scenarios.append(
        execute_scenario(
            "calibration_ece_exceeds_threshold",
            ["calibration_ece_exceeds_threshold"],
            cmd46,
            report46,
        )
    )

    # 47) Calibration slope out of range via strict slope window.
    policy_slope_bad = load_json(cfg_dir / "performance_policy.json")
    policy_slope_bad_payload = copy.deepcopy(policy_slope_bad)
    cal_block_slope = policy_slope_bad_payload.get("calibration_dca_thresholds")
    if isinstance(cal_block_slope, dict):
        cal_block_slope["slope_min"] = 1.5
        cal_block_slope["slope_max"] = 2.0
    policy_slope_bad_path = tmp_root / "performance_policy.calibration_slope_tight.json"
    write_json(policy_slope_bad_path, policy_slope_bad_payload)
    report47 = tmp_root / "calibration_dca.slope_out_of_range.report.json"
    cmd47 = [
        sys.executable,
        str(SCRIPTS_ROOT / "calibration_dca_gate.py"),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_slope_bad_path),
        "--strict",
        "--report",
        str(report47),
    ]
    scenarios.append(
        execute_scenario(
            "calibration_slope_out_of_range",
            ["calibration_slope_out_of_range"],
            cmd47,
            report47,
        )
    )

    # 48) Decision-curve net benefit insufficient via aggressive minimum advantage targets.
    policy_dca_bad = load_json(cfg_dir / "performance_policy.json")
    policy_dca_bad_payload = copy.deepcopy(policy_dca_bad)
    cal_block_dca = policy_dca_bad_payload.get("calibration_dca_thresholds")
    if isinstance(cal_block_dca, dict):
        cal_block_dca["min_advantage_coverage"] = 1.0
        cal_block_dca["min_average_advantage"] = 0.50
        cal_block_dca["min_net_benefit_advantage"] = 0.10
    policy_dca_bad_path = tmp_root / "performance_policy.dca_net_benefit_strict.json"
    write_json(policy_dca_bad_path, policy_dca_bad_payload)
    report48 = tmp_root / "calibration_dca.net_benefit_insufficient.report.json"
    cmd48 = [
        sys.executable,
        str(SCRIPTS_ROOT / "calibration_dca_gate.py"),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_dca_bad_path),
        "--strict",
        "--report",
        str(report48),
    ]
    scenarios.append(
        execute_scenario(
            "decision_curve_net_benefit_insufficient",
            ["decision_curve_net_benefit_insufficient"],
            cmd48,
            report48,
        )
    )

    # 49) Request contract: V3 policy-block downgrade guard.
    policy_new_block_downgrade = load_json(cfg_dir / "performance_policy.json")
    policy_new_block_downgrade_payload = copy.deepcopy(policy_new_block_downgrade)
    replay_block_bad = policy_new_block_downgrade_payload.get("prediction_replay_thresholds")
    if isinstance(replay_block_bad, dict):
        replay_block_bad["metric_tolerance"] = 0.0001
    request_new_block_bad = copy.deepcopy(request_abs)
    policy_new_block_downgrade_path = tmp_root / "performance_policy.v3_block_downgrade.json"
    write_json(policy_new_block_downgrade_path, policy_new_block_downgrade_payload)
    request_new_block_bad["performance_policy_spec"] = str(policy_new_block_downgrade_path)
    request_new_block_bad_path = tmp_root / "request.performance_policy_v3_block_downgrade.json"
    write_json(request_new_block_bad_path, request_new_block_bad)
    report49 = tmp_root / "request_contract.performance_policy_v3_block_downgrade.report.json"
    cmd49 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_new_block_bad_path),
        "--strict",
        "--report",
        str(report49),
    ]
    scenarios.append(
        execute_scenario(
            "performance_policy_downgrade_new_blocks",
            ["performance_policy_downgrade_new_blocks"],
            cmd49,
            report49,
        )
    )

    # 50) Request contract: invalid feature_group_spec should fail.
    feature_group_bad = {"groups": {}}
    feature_group_bad_path = tmp_root / "feature_group_spec.invalid.json"
    write_json(feature_group_bad_path, feature_group_bad)
    request_feature_group_bad = copy.deepcopy(request_abs)
    request_feature_group_bad["feature_group_spec"] = str(feature_group_bad_path)
    request_feature_group_bad_path = tmp_root / "request.feature_group_invalid.json"
    write_json(request_feature_group_bad_path, request_feature_group_bad)
    report50 = tmp_root / "request_contract.feature_group_invalid.report.json"
    cmd50 = [
        sys.executable,
        str(SCRIPTS_ROOT / "request_contract_gate.py"),
        "--request",
        str(request_feature_group_bad_path),
        "--strict",
        "--report",
        str(report50),
    ]
    scenarios.append(
        execute_scenario(
            "feature_group_spec_missing_or_invalid",
            ["feature_group_spec_missing_or_invalid"],
            cmd50,
            report50,
        )
    )

    # 51) Feature engineering audit: explicit leakage scope should fail.
    fe_report_bad = load_json(feature_engineering_report_path)
    fe_report_bad_payload = copy.deepcopy(fe_report_bad)
    fe_report_bad_payload["data_scopes_used"] = ["train_only", "test"]
    fe_report_bad_path = tmp_root / "feature_engineering_report.data_scope_leakage.json"
    write_json(fe_report_bad_path, fe_report_bad_payload)
    report51 = tmp_root / "feature_engineering_audit.data_scope_leakage.report.json"
    cmd51 = [
        sys.executable,
        str(SCRIPTS_ROOT / "feature_engineering_audit_gate.py"),
        "--feature-group-spec",
        str(cfg_dir / "feature_group_spec.json"),
        "--feature-engineering-report",
        str(fe_report_bad_path),
        "--lineage-spec",
        str(cfg_dir / "feature_lineage.json"),
        "--tuning-spec",
        str(cfg_dir / "tuning_protocol.json"),
        "--strict",
        "--report",
        str(report51),
    ]
    scenarios.append(
        execute_scenario(
            "feature_selection_data_leakage",
            ["feature_selection_data_leakage"],
            cmd51,
            report51,
        )
    )

    # 52) Distribution gate: severe shifted valid split should fail.
    valid_shifted = pd.read_csv(data_dir / "valid.csv")
    if not valid_shifted.empty:
        numeric_cols = [c for c in valid_shifted.columns if c not in {id_col, time_col, label_col}]
        for col in numeric_cols:
            vals = pd.to_numeric(valid_shifted[col], errors="coerce")
            if vals.notna().any():
                valid_shifted[col] = vals.fillna(vals.median()) + 1000.0
    valid_shifted_path = tmp_root / "valid.shifted.csv"
    valid_shifted.to_csv(valid_shifted_path, index=False)
    policy_shift = load_json(cfg_dir / "performance_policy.json")
    policy_shift_payload = copy.deepcopy(policy_shift)
    dist_shift_block = policy_shift_payload.get("distribution_thresholds_v2")
    if isinstance(dist_shift_block, dict):
        # Disable split-separability failure in this scenario so we isolate distribution-shift fail code.
        dist_shift_block["split_classifier_auc_fail"] = 1.0
        dist_shift_block["split_classifier_auc_warn"] = 1.0
        dist_shift_block["top_feature_jsd_fail"] = 0.05
        dist_shift_block["high_shift_feature_fraction_fail"] = 0.05
    policy_shift_path = tmp_root / "performance_policy.distribution_shift_strict.json"
    write_json(policy_shift_path, policy_shift_payload)
    report52 = tmp_root / "distribution_generalization.shift_exceeds.report.json"
    cmd52 = [
        sys.executable,
        str(SCRIPTS_ROOT / "distribution_generalization_gate.py"),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(valid_shifted_path),
        "--test",
        str(data_dir / "test.csv"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--feature-group-spec",
        str(cfg_dir / "feature_group_spec.json"),
        "--target-col",
        label_col,
        "--ignore-cols",
        f"{id_col},{time_col}",
        "--performance-policy",
        str(policy_shift_path),
        "--distribution-report",
        str(distribution_report_path),
        "--strict",
        "--report",
        str(report52),
    ]
    scenarios.append(
        execute_scenario(
            "distribution_shift_exceeds_threshold",
            ["distribution_shift_exceeds_threshold"],
            cmd52,
            report52,
        )
    )

    # 53) Distribution gate: strict split-separability threshold should fail.
    policy_sep = load_json(cfg_dir / "performance_policy.json")
    policy_sep_payload = copy.deepcopy(policy_sep)
    dist_block = policy_sep_payload.get("distribution_thresholds_v2")
    if isinstance(dist_block, dict):
        dist_block["split_classifier_auc_fail"] = 0.55
        dist_block["split_classifier_auc_warn"] = 0.54
        # Isolate split-separability failure from other distribution-shift fail codes.
        dist_block["top_feature_jsd_fail"] = 1.0
        dist_block["high_shift_feature_fraction_fail"] = 1.0
        dist_block["missing_ratio_delta_fail"] = 1.0
        dist_block["prevalence_delta_fail"] = 1.0
    policy_sep_path = tmp_root / "performance_policy.split_separability_strict.json"
    write_json(policy_sep_path, policy_sep_payload)
    report53 = tmp_root / "distribution_generalization.split_separability.report.json"
    cmd53 = [
        sys.executable,
        str(SCRIPTS_ROOT / "distribution_generalization_gate.py"),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(valid_shifted_path),
        "--test",
        str(data_dir / "test.csv"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--feature-group-spec",
        str(cfg_dir / "feature_group_spec.json"),
        "--target-col",
        label_col,
        "--ignore-cols",
        f"{id_col},{time_col}",
        "--performance-policy",
        str(policy_sep_path),
        "--distribution-report",
        str(distribution_report_path),
        "--strict",
        "--report",
        str(report53),
    ]
    scenarios.append(
        execute_scenario(
            "split_separability_exceeds_threshold",
            ["split_separability_exceeds_threshold"],
            cmd53,
            report53,
        )
    )

    # 54) CI matrix gate: required resamples too high should fail.
    policy_ci_resample = load_json(cfg_dir / "performance_policy.json")
    policy_ci_resample_payload = copy.deepcopy(policy_ci_resample)
    ci_block_bad = policy_ci_resample_payload.get("ci_policy")
    if isinstance(ci_block_bad, dict):
        ci_block_bad["n_resamples"] = 5000
        ci_block_bad["max_width"] = 1.0
    policy_ci_resample_path = tmp_root / "performance_policy.ci_resample_strict.json"
    write_json(policy_ci_resample_path, policy_ci_resample_payload)
    report54 = tmp_root / "ci_matrix.resamples_insufficient.report.json"
    cmd54 = [
        sys.executable,
        str(SCRIPTS_ROOT / "ci_matrix_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--prediction-trace",
        str(prediction_trace_path),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_ci_resample_path),
        "--ci-matrix-report",
        str(tmp_root / "ci_matrix.resamples_insufficient.json"),
        "--strict",
        "--report",
        str(report54),
    ]
    scenarios.append(
        execute_scenario(
            "ci_resamples_insufficient",
            ["ci_resamples_insufficient"],
            cmd54,
            report54,
        )
    )

    # 55) CI matrix gate: CI width threshold too strict should fail.
    policy_ci_width = load_json(cfg_dir / "performance_policy.json")
    policy_ci_width_payload = copy.deepcopy(policy_ci_width)
    ci_block_width = policy_ci_width_payload.get("ci_policy")
    if isinstance(ci_block_width, dict):
        ci_block_width["max_width"] = 0.01
    policy_ci_width_path = tmp_root / "performance_policy.ci_width_strict.json"
    write_json(policy_ci_width_path, policy_ci_width_payload)
    report55 = tmp_root / "ci_matrix.width_exceeds.report.json"
    cmd55 = [
        sys.executable,
        str(SCRIPTS_ROOT / "ci_matrix_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--prediction-trace",
        str(prediction_trace_path),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_ci_width_path),
        "--ci-matrix-report",
        str(tmp_root / "ci_matrix.width_exceeds.json"),
        "--strict",
        "--report",
        str(report55),
    ]
    scenarios.append(
        execute_scenario(
            "ci_width_exceeds_threshold",
            ["ci_width_exceeds_threshold"],
            cmd55,
            report55,
        )
    )

    # 56) Clinical gate: strict specificity floor should fail.
    policy_specificity = load_json(cfg_dir / "performance_policy.json")
    policy_specificity_payload = copy.deepcopy(policy_specificity)
    if isinstance(policy_specificity_payload.get("clinical_floors"), dict):
        policy_specificity_payload["clinical_floors"]["specificity_min"] = 0.99
    if isinstance(policy_specificity_payload.get("threshold_policy"), dict) and isinstance(
        policy_specificity_payload["threshold_policy"].get("clinical_floors"),
        dict,
    ):
        policy_specificity_payload["threshold_policy"]["clinical_floors"]["specificity_min"] = 0.99
    policy_specificity_path = tmp_root / "performance_policy.clinical_specificity_strict.json"
    write_json(policy_specificity_path, policy_specificity_payload)
    report56 = tmp_root / "clinical_metrics.specificity_floor.report.json"
    cmd56 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_specificity_path),
        "--strict",
        "--report",
        str(report56),
    ]
    scenarios.append(
        execute_scenario(
            "clinical_floor_specificity_not_met",
            ["clinical_floor_specificity_not_met"],
            cmd56,
            report56,
        )
    )

    # 57) Clinical gate: strict PPV floor should fail.
    policy_ppv = load_json(cfg_dir / "performance_policy.json")
    policy_ppv_payload = copy.deepcopy(policy_ppv)
    if isinstance(policy_ppv_payload.get("clinical_floors"), dict):
        policy_ppv_payload["clinical_floors"]["ppv_min"] = 0.99
    if isinstance(policy_ppv_payload.get("threshold_policy"), dict) and isinstance(
        policy_ppv_payload["threshold_policy"].get("clinical_floors"),
        dict,
    ):
        policy_ppv_payload["threshold_policy"]["clinical_floors"]["ppv_min"] = 0.99
    policy_ppv_path = tmp_root / "performance_policy.clinical_ppv_strict.json"
    write_json(policy_ppv_path, policy_ppv_payload)
    report57 = tmp_root / "clinical_metrics.ppv_floor.report.json"
    cmd57 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(policy_ppv_path),
        "--strict",
        "--report",
        str(report57),
    ]
    scenarios.append(
        execute_scenario(
            "clinical_floor_ppv_not_met",
            ["clinical_floor_ppv_not_met"],
            cmd57,
            report57,
        )
    )

    # 58) External gate: missing passing cross_institution coverage should fail.
    external_no_inst = load_json(external_validation_report_path)
    external_no_inst_payload = copy.deepcopy(external_no_inst)
    cohorts_no_inst = external_no_inst_payload.get("cohorts")
    if isinstance(cohorts_no_inst, list):
        external_no_inst_payload["cohorts"] = [row for row in cohorts_no_inst if isinstance(row, dict) and row.get("cohort_type") == "cross_period"]
        external_no_inst_payload["cohort_count"] = len(external_no_inst_payload["cohorts"])
    external_no_inst_path = tmp_root / "external_validation.no_cross_institution.json"
    write_json(external_no_inst_path, external_no_inst_payload)
    report58 = tmp_root / "external_validation.cross_institution_missing.report.json"
    cmd58 = [
        sys.executable,
        str(SCRIPTS_ROOT / "external_validation_gate.py"),
        "--external-validation-report",
        str(external_no_inst_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report58),
    ]
    scenarios.append(
        execute_scenario(
            "external_validation_cross_institution_not_met",
            ["external_validation_cross_institution_not_met"],
            cmd58,
            report58,
        )
    )

    # 59) External gate: missing passing cross_period coverage should fail.
    external_no_period = load_json(external_validation_report_path)
    external_no_period_payload = copy.deepcopy(external_no_period)
    cohorts_no_period = external_no_period_payload.get("cohorts")
    if isinstance(cohorts_no_period, list):
        external_no_period_payload["cohorts"] = [row for row in cohorts_no_period if isinstance(row, dict) and row.get("cohort_type") == "cross_institution"]
        external_no_period_payload["cohort_count"] = len(external_no_period_payload["cohorts"])
    external_no_period_path = tmp_root / "external_validation.no_cross_period.json"
    write_json(external_no_period_path, external_no_period_payload)
    report59 = tmp_root / "external_validation.cross_period_missing.report.json"
    cmd59 = [
        sys.executable,
        str(SCRIPTS_ROOT / "external_validation_gate.py"),
        "--external-validation-report",
        str(external_no_period_path),
        "--prediction-trace",
        str(prediction_trace_path),
        "--evaluation-report",
        str(evidence_dir / "evaluation_report.json"),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report59),
    ]
    scenarios.append(
        execute_scenario(
            "external_validation_cross_period_not_met",
            ["external_validation_cross_period_not_met"],
            cmd59,
            report59,
        )
    )

    # 60) Clinical gate: cv_inner selection must satisfy guard split constraints.
    eval_threshold_guard_bad = load_json(evidence_dir / "evaluation_report.json")
    eval_threshold_guard_bad_payload = copy.deepcopy(eval_threshold_guard_bad)
    threshold_guard_block = eval_threshold_guard_bad_payload.get("threshold_selection")
    if isinstance(threshold_guard_block, dict):
        threshold_guard_block["selection_split"] = "cv_inner"
        threshold_guard_block["constraints_satisfied_selection_split"] = True
        threshold_guard_block["constraints_satisfied_guard_split"] = False
        threshold_guard_block["constraints_satisfied_overall"] = False
        threshold_guard_block["constraints_satisfied"] = False
    eval_threshold_guard_bad_path = tmp_root / "evaluation_report.threshold_guard_not_met.json"
    write_json(eval_threshold_guard_bad_path, eval_threshold_guard_bad_payload)
    report60 = tmp_root / "clinical_metrics.threshold_guard_not_met.report.json"
    cmd60 = [
        sys.executable,
        str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
        "--evaluation-report",
        str(eval_threshold_guard_bad_path),
        "--external-validation-report",
        str(external_validation_report_path),
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--strict",
        "--report",
        str(report60),
    ]
    scenarios.append(
        execute_scenario(
            "threshold_guard_constraints_not_met",
            ["threshold_guard_constraints_not_met"],
            cmd60,
            report60,
        )
    )

    # 61) Stress seed-search: invalid range must report stress_seed_feasibility_not_found.
    report61 = tmp_root / "stress_seed_search.not_found.report.json"
    cmd61 = [
        sys.executable,
        "-c",
        (
            "import subprocess\n"
            "import json\n"
            "import pathlib\n"
            "import sys\n"
            "repo_root=pathlib.Path(sys.argv[1]).resolve()\n"
            "out=pathlib.Path(sys.argv[2]).resolve()\n"
            "summary_path=out.parent / 'authority_e2e_summary.invalid_range.json'\n"
            "cache_path=out.parent / 'stress_seed_search.not_found.cache.json'\n"
            "selection_path=out.parent / 'stress_seed_selection.not_found.json'\n"
            "if summary_path.exists():\n"
            "    summary_path.unlink()\n"
            "cmd=[\n"
            "    sys.executable,\n"
            "    str(repo_root / 'experiments' / 'authority-e2e' / 'run_authority_e2e.py'),\n"
            "    '--include-stress-cases',\n"
            "    '--stress-case-id',\n"
            "    'uci-heart-disease',\n"
            "    '--stress-seed-search',\n"
            "    '--stress-seed-min',\n"
            "    '10',\n"
            "    '--stress-seed-max',\n"
            "    '9',\n"
            "    '--summary-file',\n"
            "    str(summary_path),\n"
            "    '--stress-seed-cache-file',\n"
            "    str(cache_path),\n"
            "    '--stress-selection-file',\n"
            "    str(selection_path),\n"
            "]\n"
            "proc=subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)\n"
            "heart_failure_code=None\n"
            "if summary_path.exists():\n"
            "    try:\n"
            "        summary_payload=json.loads(summary_path.read_text(encoding='utf-8'))\n"
            "        for row in summary_payload.get('results', []):\n"
            "            if isinstance(row, dict) and str(row.get('case_id')) == 'uci-heart-disease':\n"
            "                failure_code=row.get('failure_code')\n"
            "                heart_failure_code=str(failure_code) if isinstance(failure_code, str) else None\n"
            "                break\n"
            "    except Exception:\n"
            "        heart_failure_code=None\n"
            "failed=bool(proc.returncode != 0 and heart_failure_code == 'stress_seed_feasibility_not_found')\n"
            "payload={\n"
            "    'status': 'fail' if failed else 'unexpected_pass',\n"
            "    'failures': ([{'code':'stress_seed_feasibility_not_found'}] if failed else []),\n"
            "    'details': {\n"
            "        'subprocess_returncode': int(proc.returncode),\n"
            "        'heart_failure_code': heart_failure_code,\n"
            "        'stdout_tail': proc.stdout[-1200:],\n"
            "        'stderr_tail': proc.stderr[-1200:],\n"
            "    },\n"
            "}\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "out.write_text(json.dumps(payload), encoding='utf-8')\n"
            "sys.exit(2 if failed else 0)\n"
        ),
        str(REPO_ROOT),
        str(report61),
    ]
    scenarios.append(
        execute_scenario(
            "stress_seed_feasibility_not_found",
            ["stress_seed_feasibility_not_found"],
            cmd61,
            report61,
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
