#!/usr/bin/env python3
"""
User-facing wrapper: env doctor -> schema preflight -> strict pipeline -> user summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_utils import load_json_from_path as load_json, write_json, resolve_path


CONTRACT_VERSION = "productized_workflow_report.v2"
MTIME_EPSILON_SECONDS = 0.5
BLOCKING_STEP_NAMES = {
    "env_doctor",
    "schema_preflight",
    "run_dag_pipeline",
    "run_dag_pipeline_with_bootstrap_baseline",
    "render_user_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run productized ml-leakage-guard workflow in one command.")
    parser.add_argument("--request", required=True, help="Path to request JSON.")
    parser.add_argument("--evidence-dir", default="evidence", help="Evidence output directory.")
    parser.add_argument("--compare-manifest", help="Optional manifest baseline for strict pipeline.")
    parser.add_argument("--allow-missing-compare", action="store_true", help="Allow bootstrap run without baseline compare.")
    parser.add_argument("--strict", action="store_true", help="Run strict pipeline (required for publication-grade).")
    parser.add_argument("--continue-on-fail", action="store_true", help="Pass through to strict pipeline diagnostic mode.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child scripts.")
    parser.add_argument("--report", help="Optional summary JSON report path for this wrapper.")
    return parser.parse_args()


def run_step(name: str, cmd: List[str]) -> Dict[str, Any]:
    print(f"\n== Step: {name} ==")
    print(f"$ {shlex.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return {
        "name": name,
        "command": shlex.join(cmd),
        "exit_code": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "status": "pass" if int(proc.returncode) == 0 else "fail",
        "blocking": bool(name in BLOCKING_STEP_NAMES),
        "recovered_by_step": None,
    }


def infer_project_base(request_path: Path) -> Path:
    """
    Infer project root from request path.
    If request is under <project>/configs/request.json, use <project>;
    otherwise use request parent directory.
    """
    parent = request_path.parent
    if parent.name.lower() == "configs" and parent.parent != parent:
        return parent.parent
    return parent


def main() -> int:
    args = parse_args()
    if not bool(args.strict):
        print(
            "[FAIL] run_productized_workflow.py requires --strict for publication-grade workflow.",
            file=sys.stderr,
        )
        return 2

    request_path = Path(args.request).expanduser().resolve()
    if not request_path.exists():
        print(f"[FAIL] request file not found: {request_path}", file=sys.stderr)
        return 2

    request_payload = load_json(request_path)
    base_dir = request_path.parent
    project_base = infer_project_base(request_path)
    split_paths = request_payload.get("split_paths")
    if not isinstance(split_paths, dict):
        print("[FAIL] request.split_paths missing.", file=sys.stderr)
        return 2

    train_path = resolve_path(base_dir, str(split_paths.get("train", "")))
    valid_path = resolve_path(base_dir, str(split_paths.get("valid", "")))
    test_path = resolve_path(base_dir, str(split_paths.get("test", "")))
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        print("[FAIL] split CSV files not found from request.split_paths.", file=sys.stderr)
        return 2

    evidence_dir = resolve_path(project_base, args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent

    steps: List[Dict[str, Any]] = []
    bootstrap_recovery_applied = False
    bootstrap_recovery_source: Optional[str] = None

    def append_step(name: str, cmd: List[str]) -> Dict[str, Any]:
        step = run_step(name, cmd)
        steps.append(step)
        return step

    def is_recent(path: Path, min_mtime_epoch: float) -> bool:
        if not path.exists():
            return False
        try:
            return float(path.stat().st_mtime) >= (float(min_mtime_epoch) - MTIME_EPSILON_SECONDS)
        except OSError:
            return False

    env_report = evidence_dir / "env_doctor_report.json"
    append_step(
        "env_doctor",
        [args.python, str(scripts_dir / "env_doctor.py"), "--report", str(env_report)],
    )

    schema_report = evidence_dir / "schema_preflight_report.json"
    schema_mapping = evidence_dir / "schema_mapping.json"
    append_step(
        "schema_preflight",
        [
            args.python,
            str(scripts_dir / "schema_preflight.py"),
            "--train",
            str(train_path),
            "--valid",
            str(valid_path),
            "--test",
            str(test_path),
            "--target-col",
            str(request_payload.get("label_col", "y")),
            "--patient-id-col",
            str(request_payload.get("patient_id_col", "patient_id")),
            "--time-col",
            str(request_payload.get("index_time_col", "event_time")),
            "--strict",
            "--mapping-out",
            str(schema_mapping),
            "--report",
            str(schema_report),
        ],
    )

    strict_run_started_epoch = time.time()
    dag_cmd = [
        args.python,
        str(scripts_dir / "run_dag_pipeline.py"),
        "--request",
        str(request_path),
        "--evidence-dir",
        str(evidence_dir),
        "--strict",
    ]
    if args.compare_manifest:
        dag_cmd.extend(["--compare-manifest", str(resolve_path(project_base, args.compare_manifest))])
    if args.allow_missing_compare:
        dag_cmd.append("--allow-missing-compare")
    if args.continue_on_fail:
        dag_cmd.append("--continue-on-fail")
    strict_step = append_step("run_dag_pipeline", dag_cmd)

    def _publication_missing_manifest(evidence: Path, min_mtime_epoch: float) -> tuple[bool, Optional[str]]:
        pub_path = evidence / "publication_gate_report.json"
        if not is_recent(pub_path, min_mtime_epoch):
            return False, None
        try:
            payload = load_json(pub_path)
        except Exception:
            return False, None
        failures = payload.get("failures")
        if not isinstance(failures, list):
            return False, None
        for row in failures:
            if isinstance(row, dict) and str(row.get("code")) == "manifest_comparison_missing":
                return True, f"{pub_path.name}:manifest_comparison_missing"
        return False, None

    strict_exit = int(strict_step["exit_code"])
    bootstrap_baseline_path: Optional[Path] = None

    missing_manifest, missing_source = _publication_missing_manifest(evidence_dir, strict_run_started_epoch)
    if (
        strict_exit != 0
        and bool(args.allow_missing_compare)
        and not args.compare_manifest
        and missing_manifest
    ):
        manifest_path = evidence_dir / "manifest.json"
        if is_recent(manifest_path, strict_run_started_epoch):
            bootstrap_baseline_path = evidence_dir / "manifest_baseline.bootstrap.json"
            shutil.copy2(manifest_path, bootstrap_baseline_path)
            retry_cmd = [
                args.python,
                str(scripts_dir / "run_dag_pipeline.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence_dir),
                "--strict",
                "--compare-manifest",
                str(bootstrap_baseline_path),
            ]
            if args.continue_on_fail:
                retry_cmd.append("--continue-on-fail")
            retry_step = append_step("run_dag_pipeline_with_bootstrap_baseline", retry_cmd)
            strict_exit = int(retry_step["exit_code"])
            if strict_exit == 0:
                strict_step["status"] = "recovered"
                strict_step["blocking"] = False
                strict_step["recovered_by_step"] = "run_dag_pipeline_with_bootstrap_baseline"
                bootstrap_recovery_applied = True
                bootstrap_recovery_source = missing_source

    summary_md = evidence_dir / "user_summary.md"
    summary_json = evidence_dir / "user_summary.json"
    append_step(
        "render_user_summary",
        [
            args.python,
            str(scripts_dir / "render_user_summary.py"),
            "--evidence-dir",
            str(evidence_dir),
            "--request",
            str(request_path),
            "--out-markdown",
            str(summary_md),
            "--out-json",
            str(summary_json),
        ],
    )

    blocking_failure_count = sum(
        1
        for step in steps
        if bool(step.get("blocking")) and str(step.get("status", "")).strip().lower() == "fail"
    )
    recovered_failure_count = sum(1 for step in steps if str(step.get("status", "")).strip().lower() == "recovered")

    if blocking_failure_count > 0:
        overall_status = "fail"
        status_reason = "blocking_step_failed"
    elif recovered_failure_count > 0:
        overall_status = "pass"
        status_reason = "bootstrap_recovered"
    else:
        overall_status = "pass"
        status_reason = "all_blocking_steps_passed"

    wrapper_report = {
        "contract_version": CONTRACT_VERSION,
        "status": overall_status,
        "status_reason": status_reason,
        "blocking_failure_count": int(blocking_failure_count),
        "recovered_failure_count": int(recovered_failure_count),
        "bootstrap_recovery_applied": bool(bootstrap_recovery_applied),
        "bootstrap_recovery_source": bootstrap_recovery_source,
        "request": str(request_path),
        "project_base": str(project_base),
        "evidence_dir": str(evidence_dir),
        "steps": steps,
        "artifacts": {
            "env_doctor_report": str(env_report),
            "schema_preflight_report": str(schema_report),
            "schema_mapping": str(schema_mapping),
            "user_summary_markdown": str(summary_md),
            "user_summary_json": str(summary_json),
            "bootstrap_manifest_baseline": str(bootstrap_baseline_path) if bootstrap_baseline_path else None,
        },
    }
    out_report = (
        Path(args.report).expanduser().resolve()
        if args.report
        else (evidence_dir / "productized_workflow_report.json").resolve()
    )
    write_json(out_report, wrapper_report)

    print(f"\nOverallStatus: {overall_status}")
    print(f"WrapperReport: {out_report}")
    print(f"UserSummaryMarkdown: {summary_md}")
    print(f"UserSummaryJSON: {summary_json}")
    return 0 if overall_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
