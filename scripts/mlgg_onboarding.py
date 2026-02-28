#!/usr/bin/env python3
"""
Novice onboarding runner for ml-leakage-guard.

Flow:
1) doctor
2) init
3) generate demo data
4) align config files to demo schema
5) train and emit required evidence artifacts
6) generate execution attestation artifacts
7) strict workflow bootstrap (--allow-missing-compare)
8) strict workflow compare-manifest rerun
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
CONTRACT_VERSION = "onboarding_report.v2"

TROUBLESHOOTING_TOP20: Dict[str, Dict[str, str]] = {
    "authority_preset_route_override_forbidden": {
        "diagnose": "python3 scripts/mlgg.py authority-release --dry-run --stress-case-id uci-heart-disease",
        "fix": "authority-release/authority-research-heart 是固定路线封装；移除冲突的 route 参数（如 --stress-case-id/--stress-seed-search）。",
        "verify": "python3 scripts/mlgg.py authority-release --dry-run",
    },
    "missing_required_path": {
        "diagnose": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
        "fix": "确认 request.json 中所有 *_spec / *_report_file 路径存在且可读。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare",
    },
    "path_not_found": {
        "diagnose": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
        "fix": "运行 onboarding 第 5 步 train 产出证据文件，或修正 request 路径。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "invalid_evaluation_report": {
        "diagnose": "python3 scripts/mlgg.py train -- --help",
        "fix": "重跑 train_select_evaluate.py 并确保 evaluation_report.json 完整生成。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "invalid_model_selection_report": {
        "diagnose": "python3 scripts/mlgg.py train -- --help",
        "fix": "确认 --model-selection-report-out 输出路径正确且可写，再重跑 train。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "invalid_external_validation_report": {
        "diagnose": "python3 scripts/mlgg.py train -- --help",
        "fix": "检查 external_cohort_spec 与 external_validation_report_out 路径，重跑 train。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "external_validation_cross_period_not_met": {
        "diagnose": "python3 scripts/external_validation_gate.py --external-validation-report <project>/evidence/external_validation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "补齐并通过 cross_period 外部队列（样本量/事件数/指标阈值）。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "external_validation_cross_institution_not_met": {
        "diagnose": "python3 scripts/external_validation_gate.py --external-validation-report <project>/evidence/external_validation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "补齐并通过 cross_institution 外部队列（样本量/事件数/指标阈值）。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "calibration_ece_exceeds_threshold": {
        "diagnose": "python3 scripts/calibration_dca_gate.py --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "检查校准方法/外部分布漂移，必要时重训并改进 calibration 质量。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "decision_curve_net_benefit_insufficient": {
        "diagnose": "python3 scripts/calibration_dca_gate.py --prediction-trace <project>/evidence/prediction_trace.csv.gz --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "提升模型在决策阈值网格上的净获益（阈值策略/模型泛化）。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "distribution_shift_exceeds_threshold": {
        "diagnose": "python3 scripts/distribution_generalization_gate.py --train <project>/data/train.csv --valid <project>/data/valid.csv --test <project>/data/test.csv --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --feature-group-spec <project>/configs/feature_group_spec.json --target-col y --ignore-cols patient_id,event_time --performance-policy <project>/configs/performance_policy.json --distribution-report <project>/evidence/distribution_report.json --strict",
        "fix": "降低分布漂移：重建特征工程或改进数据切分/外部队列构成。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "split_separability_exceeds_threshold": {
        "diagnose": "python3 scripts/distribution_generalization_gate.py --train <project>/data/train.csv --valid <project>/data/valid.csv --test <project>/data/test.csv --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --feature-group-spec <project>/configs/feature_group_spec.json --target-col y --ignore-cols patient_id,event_time --performance-policy <project>/configs/performance_policy.json --distribution-report <project>/evidence/distribution_report.json --strict",
        "fix": "检查 train/test 可分性来源（时间泄漏、中心差异、缺失模式差异）。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "overfit_gap_exceeds_threshold": {
        "diagnose": "python3 scripts/generalization_gap_gate.py --evaluation-report <project>/evidence/evaluation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "提高正则化/简化模型池/改进特征筛选，降低 train-valid/test gap。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "clinical_floor_specificity_not_met": {
        "diagnose": "python3 scripts/clinical_metrics_gate.py --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "重设阈值策略或模型，满足 specificity floor。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "clinical_floor_ppv_not_met": {
        "diagnose": "python3 scripts/clinical_metrics_gate.py --evaluation-report <project>/evidence/evaluation_report.json --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "提升 PPV（阈值/模型/特征），并保持敏感性与NPV下限。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "prediction_metric_replay_mismatch": {
        "diagnose": "python3 scripts/prediction_replay_gate.py --evaluation-report <project>/evidence/evaluation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --performance-policy <project>/configs/performance_policy.json --strict",
        "fix": "确保 evaluation_report 与 prediction_trace 同一训练产物且未被覆盖。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "signature_verification_failed": {
        "diagnose": "python3 scripts/execution_attestation_gate.py --attestation-spec <project>/configs/execution_attestation.json --evaluation-report <project>/evidence/evaluation_report.json --study-id <study_id> --run-id <run_id> --strict",
        "fix": "重生成 attestation（payload/signature/public key）并保持文件未被改写。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare",
    },
    "missing_execution_attestation_required_artifact": {
        "diagnose": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
        "fix": "确保 execution_attestation.required_artifact_names 包含全部 mandatory 名称。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "manifest_comparison_missing": {
        "diagnose": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --allow-missing-compare",
        "fix": "执行首跑 bootstrap，生成 manifest_baseline.bootstrap.json。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "performance_policy_downgrade_new_blocks": {
        "diagnose": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
        "fix": "恢复 performance_policy 到 publication baseline，不放宽阈值。",
        "verify": "python3 scripts/request_contract_gate.py --request <project>/configs/request.json --strict",
    },
    "ci_width_exceeds_threshold": {
        "diagnose": "python3 scripts/ci_matrix_gate.py --evaluation-report <project>/evidence/evaluation_report.json --prediction-trace <project>/evidence/prediction_trace.csv.gz --external-validation-report <project>/evidence/external_validation_report.json --performance-policy <project>/configs/performance_policy.json --ci-matrix-report <project>/evidence/ci_matrix_report.json --strict",
        "fix": "提高样本量或模型稳定性，收紧指标方差并重算 CI。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
    "feature_selection_data_leakage": {
        "diagnose": "python3 scripts/feature_engineering_audit_gate.py --feature-group-spec <project>/configs/feature_group_spec.json --feature-engineering-report <project>/evidence/feature_engineering_report.json --lineage-spec <project>/configs/feature_lineage.json --tuning-spec <project>/configs/tuning_protocol.json --strict",
        "fix": "确保特征筛选仅在 train/cv_inner_train 范围拟合，禁用 valid/test/external 信息。",
        "verify": "python3 scripts/mlgg.py workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided novice onboarding for ml-leakage-guard.")
    parser.add_argument("--project-root", required=True, help="Target project root.")
    parser.add_argument("--mode", choices=["guided", "preview", "auto"], default="guided", help="Onboarding execution mode.")
    parser.add_argument("--lang", choices=["bilingual", "zh", "en"], default="bilingual", help="Prompt language mode.")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm all steps in guided mode.")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Deprecated flag; onboarding is strict-only and this setting is always enforced.",
    )
    parser.set_defaults(stop_on_fail=True)
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-fail", dest="stop_on_fail", action="store_true", help="Stop at first failing step (default).")
    stop_group.add_argument(
        "--no-stop-on-fail",
        dest="stop_on_fail",
        action="store_false",
        help="Continue after failures to collect full diagnostics.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--seed", type=int, default=20260227, help="Demo dataset random seed.")
    parser.add_argument("--run-id", default="", help="Optional fixed run_id; defaults to UTC token.")
    parser.add_argument("--report", help="Optional onboarding report path.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def maybe_prompt_confirm(
    mode: str,
    auto_yes: bool,
    title: str,
    cmd: str,
    expected: Sequence[str],
) -> Tuple[bool, str]:
    if mode != "guided" or auto_yes:
        return True, ""
    if not sys.stdin:
        print(
            "[WARN] guided mode requires interactive stdin; rerun with --yes or --mode auto.",
            file=sys.stderr,
        )
        return False, "guided_mode_requires_interactive_stdin"
    print(f"\n[STEP] {title}")
    print(f"$ {cmd}")
    if expected:
        print("Expected artifacts:")
        for item in expected:
            print(f"- {item}")
    try:
        raw = input("Execute this step now? [Y/n]: ").strip().lower()
    except EOFError:
        print(
            "[WARN] guided mode input stream closed; rerun with --yes or --mode auto.",
            file=sys.stderr,
        )
        return False, "guided_mode_stdin_eof"
    if raw in {"", "y", "yes"}:
        return True, ""
    return False, "step_cancelled_by_user"


def run_command_step(
    *,
    name: str,
    description: str,
    command: List[str],
    cwd: Path,
    mode: str,
    auto_yes: bool,
    steps: List[Dict[str, Any]],
    expected_artifacts: Optional[Sequence[Path]] = None,
    output_log: Optional[Path] = None,
) -> bool:
    cmd_str = shlex.join(command)
    started = utc_now()
    expected_text = [str(p) for p in (expected_artifacts or [])]

    if mode == "preview":
        print(f"\n[PREVIEW] {name}: {description}")
        print(f"$ {cmd_str}")
        if expected_text:
            print("Expected artifacts:")
            for item in expected_text:
                print(f"- {item}")
        steps.append(
            {
                "name": name,
                "description": description,
                "command": cmd_str,
                "start_utc": started,
                "end_utc": started,
                "exit_code": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "expected_artifacts": expected_text,
                "status": "preview",
            }
        )
        return True

    proceed, cancel_reason = maybe_prompt_confirm(
        mode=mode,
        auto_yes=auto_yes,
        title=f"{name}: {description}",
        cmd=cmd_str,
        expected=expected_text,
    )
    if not proceed:
        failure_code = (
            "onboarding_interactive_input_unavailable"
            if cancel_reason in {"guided_mode_requires_interactive_stdin", "guided_mode_stdin_eof"}
            else "onboarding_step_cancelled"
        )
        steps.append(
            {
                "name": name,
                "description": description,
                "command": cmd_str,
                "start_utc": started,
                "end_utc": utc_now(),
                "exit_code": 2,
                "stdout_tail": "",
                "stderr_tail": cancel_reason or "step_cancelled_by_user",
                "failure_code": failure_code,
                "expected_artifacts": expected_text,
                "status": "fail",
            }
        )
        return False

    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True)
    stdout_tail = proc.stdout[-4000:]
    stderr_tail = proc.stderr[-4000:]
    if output_log:
        ensure_parent(output_log)
        output_log.write_text((proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or ""), encoding="utf-8")

    code = int(proc.returncode)
    if code == 0 and expected_artifacts:
        missing = [str(p) for p in expected_artifacts if not p.exists()]
        if missing:
            code = 2
            stderr_tail = (stderr_tail + "\nmissing_expected_artifacts: " + ", ".join(missing)).strip()

    steps.append(
        {
            "name": name,
            "description": description,
            "command": cmd_str,
            "start_utc": started,
            "end_utc": utc_now(),
            "exit_code": code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "expected_artifacts": expected_text,
            "status": "pass" if code == 0 else "fail",
        }
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return code == 0


def run_internal_step(
    *,
    name: str,
    description: str,
    internal_command: str,
    mode: str,
    auto_yes: bool,
    steps: List[Dict[str, Any]],
    fn: Callable[[], None],
    expected_artifacts: Optional[Sequence[Path]] = None,
) -> bool:
    started = utc_now()
    expected_text = [str(p) for p in (expected_artifacts or [])]

    if mode == "preview":
        print(f"\n[PREVIEW] {name}: {description}")
        print(f"$ {internal_command}")
        if expected_text:
            print("Expected artifacts:")
            for item in expected_text:
                print(f"- {item}")
        steps.append(
            {
                "name": name,
                "description": description,
                "command": internal_command,
                "start_utc": started,
                "end_utc": started,
                "exit_code": 0,
                "stdout_tail": "",
                "stderr_tail": "",
                "expected_artifacts": expected_text,
                "status": "preview",
            }
        )
        return True

    proceed, cancel_reason = maybe_prompt_confirm(
        mode=mode,
        auto_yes=auto_yes,
        title=f"{name}: {description}",
        cmd=internal_command,
        expected=expected_text,
    )
    if not proceed:
        failure_code = (
            "onboarding_interactive_input_unavailable"
            if cancel_reason in {"guided_mode_requires_interactive_stdin", "guided_mode_stdin_eof"}
            else "onboarding_step_cancelled"
        )
        steps.append(
            {
                "name": name,
                "description": description,
                "command": internal_command,
                "start_utc": started,
                "end_utc": utc_now(),
                "exit_code": 2,
                "stdout_tail": "",
                "stderr_tail": cancel_reason or "step_cancelled_by_user",
                "failure_code": failure_code,
                "expected_artifacts": expected_text,
                "status": "fail",
            }
        )
        return False

    code = 0
    stdout_tail = ""
    stderr_tail = ""
    try:
        fn()
        if expected_artifacts:
            missing = [str(p) for p in expected_artifacts if not p.exists()]
            if missing:
                code = 2
                stderr_tail = "missing_expected_artifacts: " + ", ".join(missing)
    except Exception as exc:
        code = 2
        stderr_tail = str(exc)

    steps.append(
        {
            "name": name,
            "description": description,
            "command": internal_command,
            "start_utc": started,
            "end_utc": utc_now(),
            "exit_code": code,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "expected_artifacts": expected_text,
            "status": "pass" if code == 0 else "fail",
        }
    )
    if code != 0 and stderr_tail:
        print(f"[FAIL] {name}: {stderr_tail}", file=sys.stderr)
    return code == 0


def align_demo_configs(project_root: Path, run_id: str) -> None:
    configs = project_root / "configs"
    request_path = configs / "request.json"
    if not request_path.exists():
        raise FileNotFoundError(f"Missing request config: {request_path}")

    request = load_json(request_path)
    request["study_id"] = "demo-medical-leakage-guard"
    request["run_id"] = run_id
    request["target_name"] = "disease_risk"
    request["prediction_unit"] = "patient-episode"
    request["index_time_col"] = "event_time"
    request["label_col"] = "y"
    request["patient_id_col"] = "patient_id"
    request["primary_metric"] = "pr_auc"
    request["claim_tier_target"] = "publication-grade"
    request["split_paths"] = {
        "train": "../data/train.csv",
        "valid": "../data/valid.csv",
        "test": "../data/test.csv",
    }
    request["phenotype_definition_spec"] = "phenotype_definitions.json"
    request["feature_lineage_spec"] = "feature_lineage.json"
    request["feature_group_spec"] = "feature_group_spec.json"
    request["split_protocol_spec"] = "split_protocol.json"
    request["imbalance_policy_spec"] = "imbalance_policy.json"
    request["missingness_policy_spec"] = "missingness_policy.json"
    request["tuning_protocol_spec"] = "tuning_protocol.json"
    request["performance_policy_spec"] = "performance_policy.json"
    request["external_cohort_spec"] = "external_cohort_spec.json"
    request["reporting_bias_checklist_spec"] = "reporting_bias_checklist.json"
    request["execution_attestation_spec"] = "execution_attestation.json"
    request["model_selection_report_file"] = "../evidence/model_selection_report.json"
    request["feature_engineering_report_file"] = "../evidence/feature_engineering_report.json"
    request["distribution_report_file"] = "../evidence/distribution_report.json"
    request["robustness_report_file"] = "../evidence/robustness_report.json"
    request["seed_sensitivity_report_file"] = "../evidence/seed_sensitivity_report.json"
    request["evaluation_report_file"] = "../evidence/evaluation_report.json"
    request["prediction_trace_file"] = "../evidence/prediction_trace.csv.gz"
    request["external_validation_report_file"] = "../evidence/external_validation_report.json"
    request["ci_matrix_report_file"] = "../evidence/ci_matrix_report.json"
    request["evaluation_metric_path"] = "metrics.pr_auc"
    request["permutation_null_metrics_file"] = "../evidence/permutation_null_pr_auc.txt"
    request["actual_primary_metric"] = float(request.get("actual_primary_metric", 0.0) or 0.0)
    request["context"] = {
        "notes": "Onboarding demo dataset. All predictors are available at/before index time and exclude disease definition variables."
    }
    write_json(request_path, request)

    feature_group = {
        "groups": {
            "demographics": ["age", "sex_male", "bmi"],
            "vitals": ["systolic_bp", "heart_rate"],
            "labs": ["wbc", "creatinine", "lactate", "crp"],
            "history": ["comorbidity_index", "smoke_status"],
        },
        "forbidden_features": ["confirmed_diagnosis_code", "reference_standard_positive"],
        "notes": "Demo feature groups for onboarding; each feature appears in exactly one group.",
    }
    write_json(configs / "feature_group_spec.json", feature_group)

    feature_lineage = {
        "features": {
            "age": {"ancestors": ["raw_age"]},
            "sex_male": {"ancestors": ["raw_sex"]},
            "bmi": {"ancestors": ["raw_bmi"]},
            "systolic_bp": {"ancestors": ["raw_systolic_bp"]},
            "heart_rate": {"ancestors": ["raw_heart_rate"]},
            "wbc": {"ancestors": ["raw_wbc"]},
            "creatinine": {"ancestors": ["raw_creatinine"]},
            "lactate": {"ancestors": ["raw_lactate"]},
            "crp": {"ancestors": ["raw_crp"]},
            "comorbidity_index": {"ancestors": ["raw_comorbidity_index"]},
            "smoke_status": {"ancestors": ["raw_smoke_status"]},
        }
    }
    write_json(configs / "feature_lineage.json", feature_lineage)

    external_spec = {
        "cohorts": [
            {
                "cohort_id": "hospital_a_2025_q4",
                "cohort_type": "cross_period",
                "path": "../data/external_2025_q4.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            },
            {
                "cohort_id": "hospital_b_site",
                "cohort_type": "cross_institution",
                "path": "../data/external_site_b.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            },
        ]
    }
    write_json(configs / "external_cohort_spec.json", external_spec)

    tuning_path = configs / "tuning_protocol.json"
    tuning = load_json(tuning_path)
    # Align demo trainer behavior with strict request-contract checks.
    # Onboarding uses selection_data=valid for beginner-stable thresholding.
    tuning["model_selection_data"] = "valid"
    tuning["final_model_refit_scope"] = "train_only"
    cv_block = tuning.get("cv")
    if isinstance(cv_block, dict):
        cv_block["nested"] = False
        tuning["cv"] = cv_block
    write_json(tuning_path, tuning)

    missingness_path = configs / "missingness_policy.json"
    missingness = load_json(missingness_path)
    train_csv = project_root / "data" / "train.csv"
    if train_csv.exists():
        with train_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        ignored_cols = {"patient_id", "event_time", "y"}
        feature_count = sum(1 for col in header if str(col).strip() and str(col).strip() not in ignored_cols)
        scale_guard = missingness.get("scale_guard_evidence")
        if not isinstance(scale_guard, dict):
            scale_guard = {}
        scale_guard["fallback_triggered"] = False
        scale_guard["fallback_strategy"] = "simple_with_indicator"
        scale_guard["train_rows_seen"] = int(row_count)
        scale_guard["feature_count_seen"] = int(feature_count)
        missingness["scale_guard_evidence"] = scale_guard
        write_json(missingness_path, missingness)

    phenotype = load_json(configs / "phenotype_definitions.json")
    targets = phenotype.get("targets")
    if not isinstance(targets, dict):
        targets = {}
    targets["disease_risk"] = {
        "defining_variables": ["confirmed_diagnosis_code", "reference_standard_positive"],
        "forbidden_patterns": ["(?i)diagnosis_code", "(?i)reference_standard"],
        "notes": "Demo target definition; these variables are never present in predictors.",
    }
    phenotype["targets"] = targets
    if "global_forbidden_patterns" not in phenotype:
        phenotype["global_forbidden_patterns"] = ["(?i)target", "(?i)label", "(?i)outcome"]
    write_json(configs / "phenotype_definitions.json", phenotype)


def build_train_command(project_root: Path, python_bin: str) -> Tuple[List[str], Dict[str, Path]]:
    data = project_root / "data"
    cfg = project_root / "configs"
    evidence = project_root / "evidence"
    models = project_root / "models"
    evidence.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    out = {
        "model_selection_report": evidence / "model_selection_report.json",
        "evaluation_report": evidence / "evaluation_report.json",
        "prediction_trace": evidence / "prediction_trace.csv.gz",
        "external_validation_report": evidence / "external_validation_report.json",
        "ci_matrix_report": evidence / "ci_matrix_report.json",
        "distribution_report": evidence / "distribution_report.json",
        "feature_engineering_report": evidence / "feature_engineering_report.json",
        "robustness_report": evidence / "robustness_report.json",
        "seed_sensitivity_report": evidence / "seed_sensitivity_report.json",
        "permutation_null_metrics": evidence / "permutation_null_pr_auc.txt",
        "model_artifact": models / "demo_model.joblib",
    }
    cmd = [
        python_bin,
        str(SCRIPTS_ROOT / "train_select_evaluate.py"),
        "--train",
        str(data / "train.csv"),
        "--valid",
        str(data / "valid.csv"),
        "--test",
        str(data / "test.csv"),
        "--target-col",
        "y",
        "--patient-id-col",
        "patient_id",
        "--ignore-cols",
        "patient_id,event_time",
        "--performance-policy",
        str(cfg / "performance_policy.json"),
        "--missingness-policy",
        str(cfg / "missingness_policy.json"),
        "--feature-group-spec",
        str(cfg / "feature_group_spec.json"),
        "--external-cohort-spec",
        str(cfg / "external_cohort_spec.json"),
        "--model-selection-report-out",
        str(out["model_selection_report"]),
        "--evaluation-report-out",
        str(out["evaluation_report"]),
        "--prediction-trace-out",
        str(out["prediction_trace"]),
        "--external-validation-report-out",
        str(out["external_validation_report"]),
        "--ci-matrix-report-out",
        str(out["ci_matrix_report"]),
        "--distribution-report-out",
        str(out["distribution_report"]),
        "--feature-engineering-report-out",
        str(out["feature_engineering_report"]),
        "--robustness-report-out",
        str(out["robustness_report"]),
        "--robustness-time-slices",
        "3",
        "--robustness-group-count",
        "2",
        "--seed-sensitivity-out",
        str(out["seed_sensitivity_report"]),
        "--permutation-null-out",
        str(out["permutation_null_metrics"]),
        "--model-out",
        str(out["model_artifact"]),
        "--n-jobs",
        "1",
        "--selection-data",
        "valid",
        "--calibration-method",
        "power",
    ]
    return cmd, out


def update_request_actual_metric(project_root: Path) -> float:
    request_path = project_root / "configs" / "request.json"
    eval_path = project_root / "evidence" / "evaluation_report.json"
    request = load_json(request_path)
    evaluation = load_json(eval_path)
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("pr_auc"), (int, float)):
        raise ValueError("evaluation_report.metrics.pr_auc missing or invalid.")
    value = float(metrics["pr_auc"])
    request["actual_primary_metric"] = value
    write_json(request_path, request)
    return value


def ensure_keypair(openssl_bin: str, private_key: Path, public_key: Path) -> None:
    if private_key.exists() and public_key.exists():
        return
    ensure_parent(private_key)
    ensure_parent(public_key)
    gen = [openssl_bin, "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:3072", "-out", str(private_key)]
    pub = [openssl_bin, "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)]
    for cmd in (gen, pub):
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"key_generation_failed: {shlex.join(cmd)}\n{proc.stderr.strip()}")


def collect_failure_codes(
    project_root: Path,
    *,
    min_mtime_epoch: Optional[float] = None,
    exclude_paths: Optional[Sequence[Path]] = None,
) -> List[str]:
    evidence = project_root / "evidence"
    out: List[str] = []
    if not evidence.exists():
        return out
    excluded = {p.resolve() for p in (exclude_paths or [])}
    saw_failed_report_without_code = False
    for p in sorted(evidence.rglob("*report*.json")):
        if not p.is_file():
            continue
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved in excluded:
            continue
        if min_mtime_epoch is not None:
            try:
                if p.stat().st_mtime < float(min_mtime_epoch):
                    continue
            except OSError:
                continue
        try:
            payload = load_json(p)
        except Exception:
            continue
        report_has_code = False
        failures = payload.get("failures")
        if isinstance(failures, list):
            for row in failures:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code", "")).strip()
                if code:
                    report_has_code = True
                    if code not in out:
                        out.append(code)
        status = str(payload.get("status", "")).strip().lower()
        if status == "fail" and not report_has_code:
            saw_failed_report_without_code = True
    if saw_failed_report_without_code and not out:
        out.append("onboarding_unknown_failure")
    return out


def collect_step_failure_codes(steps: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for row in steps:
        if not isinstance(row, dict):
            continue
        code = str(row.get("failure_code", "")).strip()
        if code and code not in out:
            out.append(code)
    return out


def absolutize_repo_python_command(raw: str) -> str:
    # Keep command semantics unchanged while making copy-ready commands runnable from any cwd.
    return re.sub(
        r"\bpython3\s+scripts/([A-Za-z0-9_.-]+)",
        lambda m: "python3 " + str((SCRIPTS_ROOT / m.group(1)).resolve()),
        str(raw),
    )


def build_next_actions(failure_codes: Sequence[str], status: str, lang: str, mode: str) -> List[str]:
    lang_mode = str(lang).strip().lower()
    actions: List[str] = []
    mlgg_cli = f"python3 {(SCRIPTS_ROOT / 'mlgg.py').resolve()}"
    if mode == "preview":
        if lang_mode == "zh":
            return [
                "当前是 preview 模式：仅生成计划，未执行任何训练/门控步骤。",
                f"执行完整 onboarding: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"仅打印步骤计划: {mlgg_cli} onboarding --project-root <project> --mode preview",
            ]
        if lang_mode == "en":
            return [
                "Current run is preview-only: commands were generated but not executed.",
                f"Run full onboarding: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"Print step plan only: {mlgg_cli} onboarding --project-root <project> --mode preview",
            ]
        return [
            "Current run is preview-only / 当前为 preview，仅生成命令计划未执行。",
            f"Run full onboarding / 执行完整 onboarding: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
            f"Preview only / 仅预览: {mlgg_cli} onboarding --project-root <project> --mode preview",
        ]
    if "onboarding_step_cancelled" in set(failure_codes):
        if lang_mode == "zh":
            return [
                "onboarding 被用户取消，请从当前命令重新执行。",
                f"跳过逐步确认：{mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"收集完整诊断：{mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        if lang_mode == "en":
            return [
                "Onboarding was cancelled by user confirmation; rerun from the same command to continue.",
                f"Run non-interactive confirm mode: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"Collect full diagnostics without early stop: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        return [
            "Onboarding was cancelled by user confirmation / onboarding 被用户取消，请重新执行。",
            f"Run non-interactive confirm mode / 跳过确认: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
            f"Collect full diagnostics / 收集完整诊断: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
        ]
    if "onboarding_interactive_input_unavailable" in set(failure_codes):
        if lang_mode == "zh":
            return [
                "guided 模式缺少交互输入（stdin/TTY 不可用）。",
                f"无交互执行：{mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"或自动执行并收集完整诊断：{mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        if lang_mode == "en":
            return [
                "Guided mode has no interactive stdin (TTY unavailable).",
                f"Run non-interactive confirm mode: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
                f"Or run auto diagnosis mode: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        return [
            "Guided mode has no stdin / guided 模式缺少交互输入。",
            f"Run with --yes / 使用 --yes: {mlgg_cli} onboarding --project-root <project> --mode guided --yes",
            f"Or run auto diagnosis / 或自动诊断: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
        ]
    if status == "pass" and not failure_codes:
        if lang_mode == "zh":
            return [
                "当前无阻断失败，建议持续使用 compare-manifest 保持可复现复跑。",
                f"复跑命令: {mlgg_cli} workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
                f"发布级基准（推荐）: {mlgg_cli} authority-release --summary-file <project>/evidence/authority_release_summary.json",
                f"高级研究路径（heart，高压模式）: {mlgg_cli} authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060 --summary-file <project>/evidence/authority_research_heart_summary.json",
            ]
        if lang_mode == "en":
            return [
                "No blocking failures. Keep using compare-manifest for reproducible reruns.",
                f"Re-run: {mlgg_cli} workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
                f"Recommended release-grade benchmark: {mlgg_cli} authority-release --summary-file <project>/evidence/authority_release_summary.json",
                f"Advanced research route (heart/high-pressure): {mlgg_cli} authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060 --summary-file <project>/evidence/authority_research_heart_summary.json",
            ]
        return [
            "No blocking failures / 当前无阻断失败，建议持续使用 compare-manifest。",
            f"Re-run / 复跑: {mlgg_cli} workflow --request <project>/configs/request.json --strict --compare-manifest <project>/evidence/manifest_baseline.bootstrap.json",
            f"Recommended release benchmark / 发布级基准（推荐）: {mlgg_cli} authority-release --summary-file <project>/evidence/authority_release_summary.json",
            f"Advanced heart research route / 高级 heart 研究路径: {mlgg_cli} authority-research-heart --stress-seed-min 20250003 --stress-seed-max 20250060 --summary-file <project>/evidence/authority_research_heart_summary.json",
        ]
    if status != "pass" and not failure_codes:
        if lang_mode == "zh":
            return [
                "检测到阻断失败，但 gate 未返回 failure code。",
                "请在 onboarding_report.json 中定位首个 status=fail 且 exit_code!=0 的步骤。",
                f"完整诊断重跑: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        if lang_mode == "en":
            return [
                "Blocking failure detected but no gate failure code was emitted.",
                "Inspect onboarding_report.json and find the first step with status=fail and non-zero exit_code.",
                f"Re-run full diagnosis: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
            ]
        return [
            "Blocking failure detected but no gate code / 检测到阻断失败但无 gate 失败码。",
            "Inspect first failed step in onboarding_report.json / 在报告中定位首个失败步骤。",
            f"Re-run full diagnosis / 完整诊断重跑: {mlgg_cli} onboarding --project-root <project> --mode auto --no-stop-on-fail",
        ]
    for code in failure_codes[:5]:
        spec = TROUBLESHOOTING_TOP20.get(code)
        if not spec:
            if lang_mode == "zh":
                actions.append(f"{code}: 查看 <project>/evidence 下对应 gate 报告并修复。")
            elif lang_mode == "en":
                actions.append(f"{code}: inspect corresponding gate report under <project>/evidence and remediate.")
            else:
                actions.append(f"{code}: inspect gate report / 查看 <project>/evidence 下对应 gate 报告并修复。")
            continue
        diagnose = absolutize_repo_python_command(spec["diagnose"])
        verify = absolutize_repo_python_command(spec["verify"])
        if lang_mode == "zh":
            actions.append(
                f"{code}: 诊断 `{diagnose}`; 修复请参考 `references/Troubleshooting-Top20.md`; 复验 `{verify}`"
            )
        elif lang_mode == "en":
            actions.append(
                f"{code}: diagnose `{diagnose}`; apply fix in `references/Troubleshooting-Top20.md`; verify `{verify}`"
            )
        else:
            actions.append(
                f"{code}: diagnose/诊断 `{diagnose}`; fix/修复参见 `references/Troubleshooting-Top20.md`; verify/复验 `{verify}`"
            )
    if not actions:
        if lang_mode == "zh":
            actions.append("查看 <project>/evidence 下 strict pipeline 报告并重跑 workflow。")
        elif lang_mode == "en":
            actions.append("Inspect strict pipeline reports under <project>/evidence and rerun workflow.")
        else:
            actions.append("Inspect strict pipeline reports / 查看 strict pipeline 报告并重跑 workflow。")
    return actions


def build_copy_ready_commands(project_root: Path) -> Dict[str, str]:
    mlgg_entry = str((SCRIPTS_ROOT / "mlgg.py").resolve())
    request_path = project_root / "configs" / "request.json"
    evidence_dir = project_root / "evidence"
    compare_manifest = evidence_dir / "manifest_baseline.bootstrap.json"
    return {
        "workflow_bootstrap": shlex.join(
            [
                "python3",
                mlgg_entry,
                "workflow",
                "--request",
                str(request_path),
                "--strict",
                "--allow-missing-compare",
            ]
        ),
        "workflow_compare": shlex.join(
            [
                "python3",
                mlgg_entry,
                "workflow",
                "--request",
                str(request_path),
                "--strict",
                "--compare-manifest",
                str(compare_manifest),
            ]
        ),
        "authority_release": shlex.join(
            [
                "python3",
                mlgg_entry,
                "authority-release",
                "--summary-file",
                str(evidence_dir / "authority_release_summary.json"),
            ]
        ),
        "authority_research_heart": shlex.join(
            [
                "python3",
                mlgg_entry,
                "authority-research-heart",
                "--stress-seed-min",
                "20250003",
                "--stress-seed-max",
                "20250060",
                "--summary-file",
                str(evidence_dir / "authority_research_heart_summary.json"),
            ]
        ),
        "adversarial": shlex.join(["python3", mlgg_entry, "adversarial"]),
    }


def derive_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    run_started_epoch = datetime.now(tz=timezone.utc).timestamp()
    project_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = project_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id.strip() or datetime.now(tz=timezone.utc).strftime("onboarding-%Y%m%dT%H%M%SZ")
    python_bin = str(args.python).strip() or sys.executable
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else (evidence_dir / "onboarding_report.json").resolve()
    )
    mode = str(args.mode)
    auto_yes = bool(args.yes)

    if not bool(args.strict):
        print("[FAIL] onboarding requires strict mode and does not allow downgrade.", file=sys.stderr)
        return 2

    step_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {
        "project_root": str(project_root),
        "request_file": str(project_root / "configs" / "request.json"),
        "onboarding_report": str(report_path),
        "recommended_authority_command": (
            "python3 scripts/mlgg.py authority-release --summary-file "
            "<project>/evidence/authority_release_summary.json"
        ),
        "advanced_authority_command": (
            "python3 scripts/mlgg.py authority-research-heart --stress-seed-min 20250003 "
            "--stress-seed-max 20250060 --summary-file <project>/evidence/authority_research_heart_summary.json"
        ),
    }
    stop_on_fail = bool(args.stop_on_fail)

    def finish(status: str) -> int:
        return finalize(
            report_path=report_path,
            project_root=project_root,
            run_id=run_id,
            mode=mode,
            lang=args.lang,
            steps=step_rows,
            artifacts=artifacts,
            status=status,
            stop_on_fail=stop_on_fail,
            run_started_epoch=run_started_epoch,
        )

    def should_continue(ok: bool) -> bool:
        if ok:
            return True
        return not stop_on_fail

    # Step 1 doctor
    doctor_report = evidence_dir / "env_doctor_report.json"
    ok = run_command_step(
        name="step1_doctor",
        description="Environment check for core/optional dependencies.",
        command=[python_bin, str(SCRIPTS_ROOT / "env_doctor.py"), "--report", str(doctor_report)],
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=[doctor_report],
    )
    artifacts["env_doctor_report"] = str(doctor_report)
    if not should_continue(ok):
        return finish("fail")

    # Step 2 init
    init_report = evidence_dir / "init_report.json"
    ok = run_command_step(
        name="step2_init",
        description="Initialize project directories and baseline config templates.",
        command=[
            python_bin,
            str(SCRIPTS_ROOT / "init_project.py"),
            "--project-root",
            str(project_root),
            "--study-id",
            "demo-medical-leakage-guard",
            "--run-id",
            run_id,
            "--target-name",
            "disease_risk",
            "--label-col",
            "y",
            "--patient-id-col",
            "patient_id",
            "--index-time-col",
            "event_time",
            "--force",
            "--report",
            str(init_report),
        ],
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=[project_root / "configs" / "request.json", init_report],
    )
    artifacts["init_report"] = str(init_report)
    if not should_continue(ok):
        return finish("fail")

    # Step 3 demo data
    demo_report = evidence_dir / "demo_dataset_report.json"
    ok = run_command_step(
        name="step3_generate_demo_data",
        description="Generate offline synthetic medical splits (train/valid/test + dual external cohorts).",
        command=[
            python_bin,
            str(SCRIPTS_ROOT / "generate_demo_medical_dataset.py"),
            "--project-root",
            str(project_root),
            "--seed",
            str(int(args.seed)),
            "--report",
            str(demo_report),
        ],
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=[
            project_root / "data" / "train.csv",
            project_root / "data" / "valid.csv",
            project_root / "data" / "test.csv",
            project_root / "data" / "external_2025_q4.csv",
            project_root / "data" / "external_site_b.csv",
            demo_report,
        ],
    )
    artifacts["demo_dataset_report"] = str(demo_report)
    if not should_continue(ok):
        return finish("fail")

    # Step 4 align configs
    ok = run_internal_step(
        name="step4_align_configs",
        description="Align request/spec files with demo schema and publication-grade artifact paths.",
        internal_command="internal: align_demo_configs(project_root, run_id)",
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        fn=lambda: align_demo_configs(project_root=project_root, run_id=run_id),
        expected_artifacts=[
            project_root / "configs" / "request.json",
            project_root / "configs" / "feature_group_spec.json",
            project_root / "configs" / "feature_lineage.json",
            project_root / "configs" / "external_cohort_spec.json",
        ],
    )
    if not should_continue(ok):
        return finish("fail")

    # Step 5 train
    train_cmd, train_outputs = build_train_command(project_root=project_root, python_bin=python_bin)
    train_log = evidence_dir / "train_command.log"
    train_cfg = project_root / "configs" / "train_runtime_config.json"
    ensure_parent(train_cfg)
    write_json(
        train_cfg,
        {
            "run_id": run_id,
            "generated_at_utc": utc_now(),
            "train_command": shlex.join(train_cmd),
            "outputs": {k: str(v) for k, v in train_outputs.items()},
            "seed": int(args.seed),
        },
    )
    ok = run_command_step(
        name="step5_train",
        description="Train/select/evaluate and emit full publication-grade evidence artifacts.",
        command=train_cmd,
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=list(train_outputs.values()) + [train_log, train_cfg],
        output_log=train_log,
    )
    artifacts["train_log"] = str(train_log)
    artifacts["train_runtime_config"] = str(train_cfg)
    artifacts.update({k: str(v) for k, v in train_outputs.items()})
    if not should_continue(ok):
        return finish("fail")

    if mode != "preview":
        try:
            pr_auc = update_request_actual_metric(project_root=project_root)
            artifacts["request_actual_primary_metric"] = pr_auc
        except Exception as exc:
            step_rows.append(
                {
                    "name": "step5b_update_request_metric",
                    "description": "Update request.actual_primary_metric from evaluation report.",
                    "command": "internal: update_request_actual_metric",
                    "start_utc": utc_now(),
                    "end_utc": utc_now(),
                    "exit_code": 2,
                    "stdout_tail": "",
                    "stderr_tail": str(exc),
                    "expected_artifacts": [str(project_root / "configs" / "request.json")],
                    "status": "fail",
                }
            )
            if not should_continue(False):
                return finish("fail")

    # Step 6 attestation
    def step6_fn() -> None:
        openssl_bin = shutil.which("openssl")
        if not openssl_bin:
            raise RuntimeError(
                "onboarding_openssl_missing: openssl not found in PATH. Install and retry "
                "(macOS: `brew install openssl`, Ubuntu: `sudo apt-get install openssl`)."
            )

        keys_dir = project_root / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        key_names = [
            ("attestation_priv.pem", "attestation_pub.pem"),
            ("timestamp_priv.pem", "timestamp_pub.pem"),
            ("transparency_priv.pem", "transparency_pub.pem"),
            ("execution_priv.pem", "execution_pub.pem"),
            ("execution_log_priv.pem", "execution_log_pub.pem"),
            ("witness_a_priv.pem", "witness_a_pub.pem"),
            ("witness_b_priv.pem", "witness_b_pub.pem"),
        ]
        for priv_name, pub_name in key_names:
            ensure_keypair(openssl_bin=openssl_bin, private_key=keys_dir / priv_name, public_key=keys_dir / pub_name)

        req = load_json(project_root / "configs" / "request.json")
        study_id = str(req.get("study_id", "demo-medical-leakage-guard"))
        cmd = [
            python_bin,
            str(SCRIPTS_ROOT / "generate_execution_attestation.py"),
            "--study-id",
            study_id,
            "--run-id",
            run_id,
            "--payload-out",
            str(project_root / "evidence" / "attestation_payload.json"),
            "--signature-out",
            str(project_root / "evidence" / "attestation.sig"),
            "--spec-out",
            str(project_root / "configs" / "execution_attestation.json"),
            "--private-key-file",
            str(keys_dir / "attestation_priv.pem"),
            "--public-key-file",
            str(keys_dir / "attestation_pub.pem"),
            "--timestamp-private-key-file",
            str(keys_dir / "timestamp_priv.pem"),
            "--timestamp-public-key-file",
            str(keys_dir / "timestamp_pub.pem"),
            "--transparency-private-key-file",
            str(keys_dir / "transparency_priv.pem"),
            "--transparency-public-key-file",
            str(keys_dir / "transparency_pub.pem"),
            "--execution-private-key-file",
            str(keys_dir / "execution_priv.pem"),
            "--execution-public-key-file",
            str(keys_dir / "execution_pub.pem"),
            "--execution-log-private-key-file",
            str(keys_dir / "execution_log_priv.pem"),
            "--execution-log-public-key-file",
            str(keys_dir / "execution_log_pub.pem"),
            "--require-independent-timestamp-authority",
            "--require-independent-execution-authority",
            "--require-independent-log-authority",
            "--require-witness-quorum",
            "--min-witness-count",
            "2",
            "--require-independent-witness-keys",
            "--require-witness-independence-from-signing",
            "--witness",
            f"witness-a|{keys_dir / 'witness_a_pub.pem'}|{keys_dir / 'witness_a_priv.pem'}",
            "--witness",
            f"witness-b|{keys_dir / 'witness_b_pub.pem'}|{keys_dir / 'witness_b_priv.pem'}",
            "--command",
            shlex.join(train_cmd),
            "--git-commit",
            derive_git_commit(),
            "--artifact",
            f"training_log={project_root / 'evidence' / 'train_command.log'}",
            "--artifact",
            f"training_config={project_root / 'configs' / 'train_runtime_config.json'}",
            "--artifact",
            f"model_artifact={train_outputs['model_artifact']}",
            "--artifact",
            f"model_selection_report={train_outputs['model_selection_report']}",
            "--artifact",
            f"robustness_report={train_outputs['robustness_report']}",
            "--artifact",
            f"seed_sensitivity_report={train_outputs['seed_sensitivity_report']}",
            "--artifact",
            f"evaluation_report={train_outputs['evaluation_report']}",
            "--artifact",
            f"prediction_trace={train_outputs['prediction_trace']}",
            "--artifact",
            f"external_validation_report={train_outputs['external_validation_report']}",
        ]
        proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "attestation_generation_failed:\n"
                + shlex.join(cmd)
                + "\n"
                + (proc.stderr[-3000:] if proc.stderr else proc.stdout[-3000:])
            )

    ok = run_internal_step(
        name="step6_attestation",
        description="Generate keys, payload signatures, witness records, and execution_attestation spec.",
        internal_command="internal: generate_execution_attestation_bundle",
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        fn=step6_fn,
        expected_artifacts=[
            project_root / "configs" / "execution_attestation.json",
            project_root / "evidence" / "attestation_payload.json",
            project_root / "evidence" / "attestation.sig",
            project_root / "evidence" / "attestation_timestamp_record.json",
            project_root / "evidence" / "attestation_transparency_record.json",
            project_root / "evidence" / "attestation_execution_receipt_record.json",
            project_root / "evidence" / "attestation_execution_log_record.json",
            project_root / "evidence" / "attestation_witness_record_1.json",
            project_root / "evidence" / "attestation_witness_record_2.json",
        ],
    )
    if not should_continue(ok):
        return finish("fail")

    # Step 7 workflow bootstrap
    workflow_bootstrap_report = evidence_dir / "workflow_report_bootstrap.json"
    ok = run_command_step(
        name="step7_workflow_bootstrap",
        description="Run strict workflow with bootstrap baseline generation.",
        command=[
            python_bin,
            str(SCRIPTS_ROOT / "run_productized_workflow.py"),
            "--request",
            str(project_root / "configs" / "request.json"),
            "--evidence-dir",
            str(project_root / "evidence"),
            "--strict",
            "--allow-missing-compare",
            "--report",
            str(workflow_bootstrap_report),
        ],
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=[
            project_root / "evidence" / "strict_pipeline_report.json",
            project_root / "evidence" / "manifest_baseline.bootstrap.json",
            project_root / "evidence" / "user_summary.md",
            workflow_bootstrap_report,
        ],
    )
    artifacts["workflow_report_bootstrap"] = str(workflow_bootstrap_report)
    if not should_continue(ok):
        return finish("fail")

    # Step 8 workflow compare rerun
    compare_manifest = project_root / "evidence" / "manifest_baseline.bootstrap.json"
    workflow_compare_report = evidence_dir / "workflow_report_compare.json"
    ok = run_command_step(
        name="step8_workflow_compare",
        description="Rerun strict workflow against locked baseline manifest.",
        command=[
            python_bin,
            str(SCRIPTS_ROOT / "run_productized_workflow.py"),
            "--request",
            str(project_root / "configs" / "request.json"),
            "--evidence-dir",
            str(project_root / "evidence"),
            "--strict",
            "--compare-manifest",
            str(compare_manifest),
            "--report",
            str(workflow_compare_report),
        ],
        cwd=project_root,
        mode=mode,
        auto_yes=auto_yes,
        steps=step_rows,
        expected_artifacts=[
            compare_manifest,
            project_root / "evidence" / "strict_pipeline_report.json",
            project_root / "evidence" / "user_summary.md",
            workflow_compare_report,
        ],
    )
    artifacts["workflow_report_compare"] = str(workflow_compare_report)
    if not should_continue(ok):
        return finish("fail")

    has_failed_step = any(str(row.get("status", "")).strip().lower() == "fail" for row in step_rows if isinstance(row, dict))
    return finish("fail" if has_failed_step else "pass")


def finalize(
    report_path: Path,
    project_root: Path,
    run_id: str,
    mode: str,
    lang: str,
    steps: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    status: str,
    stop_on_fail: bool,
    run_started_epoch: float,
) -> int:
    failure_codes: List[str] = []
    if status != "pass":
        for code in collect_failure_codes(
            project_root=project_root,
            min_mtime_epoch=run_started_epoch,
            exclude_paths=[report_path],
        ):
            if code not in failure_codes:
                failure_codes.append(code)
        for code in collect_step_failure_codes(steps):
            if code not in failure_codes:
                failure_codes.append(code)
        if not failure_codes:
            failure_codes.append("onboarding_unknown_failure")

    if status == "pass":
        termination_reason = "completed_successfully"
    elif "onboarding_step_cancelled" in failure_codes:
        termination_reason = "cancelled_by_user"
    elif stop_on_fail:
        termination_reason = "stopped_on_failure"
    else:
        termination_reason = "completed_with_failures"

    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "lang": lang,
        "strict_mode": True,
        "stop_on_fail": bool(stop_on_fail),
        "termination_reason": termination_reason,
        "generated_at_utc": utc_now(),
        "project_root": str(project_root),
        "steps": steps,
        "artifacts": artifacts,
        "failure_codes": failure_codes,
        "preview_only": mode == "preview",
        "display_status": "preview" if mode == "preview" else status,
        "next_actions": build_next_actions(failure_codes, status=status, lang=lang, mode=mode),
        "copy_ready_commands": build_copy_ready_commands(project_root),
    }
    write_json(report_path, report)

    display_status = "preview" if mode == "preview" else status
    print(f"\nOnboardingStatus: {display_status}")
    if mode == "preview":
        print("OnboardingNote: preview mode generated command plan only (no execution).")
    print(f"OnboardingReport: {report_path}")
    if failure_codes:
        print("TopFailureCodes: " + ", ".join(failure_codes[:8]))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
