#!/usr/bin/env python3
"""
Run structured benchmark suites to evaluate release-grade stability.

This wrapper executes authority and adversarial runners with isolated output
files, then emits a single machine-readable matrix summary.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
AUTHORITY_SCRIPT = EXPERIMENT_ROOT / "run_authority_e2e.py"
ADVERSARIAL_SCRIPT = EXPERIMENT_ROOT / "run_adversarial_gate_checks.py"


@dataclass
class SuiteSpec:
    suite_id: str
    name: str
    kind: str
    blocking: bool
    command: List[str]
    summary_file: Path


def utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object root: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release benchmark matrix for ml-leakage-guard.")
    parser.add_argument(
        "--profile",
        choices=["quick", "release", "extended"],
        default="release",
        help=(
            "Benchmark profile: quick=release authority + adversarial, "
            "release=quick + large-case authority, extended=release + heart research stress."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENT_ROOT / "release_benchmark_matrix_summary.json"),
        help="Path to write matrix summary JSON.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(EXPERIMENT_ROOT / "_benchmark_matrix_runs"),
        help="Directory for per-suite summary artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run nested scripts.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag. If omitted, UTC timestamp token is used.",
    )
    parser.add_argument(
        "--stress-seed-min",
        type=int,
        default=20250003,
        help="Heart research stress minimum seed (extended profile only).",
    )
    parser.add_argument(
        "--stress-seed-max",
        type=int,
        default=20250060,
        help="Heart research stress maximum seed (extended profile only).",
    )
    parser.add_argument(
        "--heart-blocking",
        action="store_true",
        help="Treat heart research suite as blocking in extended profile.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write dry-run report without execution.",
    )
    return parser.parse_args()


def build_suites(args: argparse.Namespace, run_tag: str, artifacts_dir: Path, python_bin: str) -> List[SuiteSpec]:
    suites: List[SuiteSpec] = []

    def authority_suite(
        suite_id: str,
        name: str,
        extras: List[str],
        blocking: bool = True,
    ) -> SuiteSpec:
        summary_file = artifacts_dir / f"{run_tag}.{suite_id}.authority_summary.json"
        cmd = [python_bin, str(AUTHORITY_SCRIPT), "--summary-file", str(summary_file), "--run-tag", f"{run_tag}-{suite_id}", *extras]
        return SuiteSpec(
            suite_id=suite_id,
            name=name,
            kind="authority",
            blocking=blocking,
            command=cmd,
            summary_file=summary_file,
        )

    def adversarial_suite(suite_id: str, name: str, blocking: bool = True) -> SuiteSpec:
        output_file = artifacts_dir / f"{run_tag}.{suite_id}.adversarial_summary.json"
        cmd = [python_bin, str(ADVERSARIAL_SCRIPT), "--output", str(output_file)]
        return SuiteSpec(
            suite_id=suite_id,
            name=name,
            kind="adversarial",
            blocking=blocking,
            command=cmd,
            summary_file=output_file,
        )

    suites.append(
        authority_suite(
            suite_id="authority_release_core",
            name="Authority release route (WDBC + CKD stress)",
            extras=["--include-stress-cases", "--stress-case-id", "uci-chronic-kidney-disease"],
            blocking=True,
        )
    )

    if args.profile in {"release", "extended"}:
        suites.append(
            authority_suite(
                suite_id="authority_release_extended",
                name="Authority extended route (+ Diabetes130 large cohort)",
                extras=[
                    "--include-stress-cases",
                    "--stress-case-id",
                    "uci-chronic-kidney-disease",
                    "--include-large-cases",
                ],
                blocking=True,
            )
        )

    if args.profile == "extended":
        suites.append(
            authority_suite(
                suite_id="authority_research_heart",
                name="Authority heart research stress route",
                extras=[
                    "--include-stress-cases",
                    "--stress-case-id",
                    "uci-heart-disease",
                    "--stress-seed-search",
                    "--stress-seed-min",
                    str(int(args.stress_seed_min)),
                    "--stress-seed-max",
                    str(int(args.stress_seed_max)),
                ],
                blocking=bool(args.heart_blocking),
            )
        )

    suites.append(
        adversarial_suite(
            suite_id="adversarial_fail_closed",
            name="Adversarial fail-closed scenarios",
            blocking=True,
        )
    )
    return suites


def parse_suite_outcome(kind: str, summary_file: Path, exit_code: int) -> Dict[str, Any]:
    if not summary_file.exists():
        return {
            "status": "fail",
            "failure_reason": "summary_missing",
            "exit_code": int(exit_code),
            "overall_status": None,
        }
    try:
        payload = load_json(summary_file)
    except Exception as exc:
        return {
            "status": "fail",
            "failure_reason": "summary_parse_error",
            "exit_code": int(exit_code),
            "overall_status": None,
            "error": str(exc),
        }

    overall = str(payload.get("overall_status", "")).strip().lower()
    status = "pass" if exit_code == 0 and overall == "pass" else "fail"
    detail: Dict[str, Any] = {
        "status": status,
        "failure_reason": None if status == "pass" else "suite_failed",
        "exit_code": int(exit_code),
        "overall_status": overall or None,
    }
    if kind == "adversarial":
        passed_count = int(payload.get("passed_count", 0) or 0)
        scenario_count = int(payload.get("scenario_count", 0) or 0)
        detail["passed_count"] = passed_count
        detail["scenario_count"] = scenario_count
        if status == "pass" and passed_count != scenario_count:
            detail["status"] = "fail"
            detail["failure_reason"] = "adversarial_partial_pass"
    elif kind == "authority":
        results = payload.get("results")
        if isinstance(results, list):
            detail["case_count"] = len(results)
            detail["failed_cases"] = [
                str(row.get("case_id"))
                for row in results
                if isinstance(row, dict) and str(row.get("status")) != "pass"
            ]
    return detail


def main() -> int:
    args = parse_args()
    run_tag = str(args.run_tag).strip() or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = Path(str(args.output)).expanduser().resolve()
    artifacts_dir = Path(str(args.artifacts_dir)).expanduser().resolve()
    python_bin = str(args.python).strip() or sys.executable
    suites = build_suites(args, run_tag, artifacts_dir, python_bin)

    suite_rows: List[Dict[str, Any]] = []
    blocking_failures: List[str] = []
    nonblocking_failures: List[str] = []

    for suite in suites:
        print(f"$ {shlex.join(suite.command)}")
        if args.dry_run:
            suite_rows.append(
                {
                    "suite_id": suite.suite_id,
                    "name": suite.name,
                    "kind": suite.kind,
                    "blocking": bool(suite.blocking),
                    "status": "dry_run",
                    "exit_code": None,
                    "overall_status": None,
                    "summary_file": str(suite.summary_file),
                    "command": suite.command,
                }
            )
            continue

        suite.summary_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            suite.command,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        outcome = parse_suite_outcome(suite.kind, suite.summary_file, int(proc.returncode))
        row = {
            "suite_id": suite.suite_id,
            "name": suite.name,
            "kind": suite.kind,
            "blocking": bool(suite.blocking),
            "status": outcome.get("status"),
            "exit_code": outcome.get("exit_code"),
            "overall_status": outcome.get("overall_status"),
            "failure_reason": outcome.get("failure_reason"),
            "summary_file": str(suite.summary_file),
            "command": suite.command,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
        for key in ("case_count", "failed_cases", "passed_count", "scenario_count", "error"):
            if key in outcome:
                row[key] = outcome.get(key)
        suite_rows.append(row)
        if str(row["status"]) != "pass":
            if suite.blocking:
                blocking_failures.append(suite.suite_id)
            else:
                nonblocking_failures.append(suite.suite_id)

    if args.dry_run:
        overall_status = "dry_run"
        exit_code = 0
        status_reason = "commands_printed_only"
    elif blocking_failures:
        overall_status = "fail"
        exit_code = 2
        status_reason = "blocking_suite_failed"
    else:
        overall_status = "pass"
        exit_code = 0
        status_reason = "all_blocking_suites_passed"

    report: Dict[str, Any] = {
        "contract_version": "release_benchmark_matrix.v1",
        "generated_at_utc": utc_now_text(),
        "repo_root": str(REPO_ROOT),
        "run_tag": run_tag,
        "profile": str(args.profile),
        "overall_status": overall_status,
        "status_reason": status_reason,
        "blocking_failure_count": len(blocking_failures),
        "blocking_failures": blocking_failures,
        "nonblocking_failure_count": len(nonblocking_failures),
        "nonblocking_failures": nonblocking_failures,
        "suites": suite_rows,
    }
    write_json_atomic(output_path, report)
    print(f"Release benchmark matrix summary: {output_path}")
    print(f"overall_status={overall_status} blocking_failures={len(blocking_failures)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
