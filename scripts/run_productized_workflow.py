#!/usr/bin/env python3
"""
User-facing wrapper: env doctor -> schema preflight -> strict pipeline -> user summary.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be object.")
    return payload


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


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
    }


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


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

    evidence_dir = resolve_path(Path.cwd(), args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = Path(__file__).resolve().parent

    steps: List[Dict[str, Any]] = []

    env_report = evidence_dir / "env_doctor_report.json"
    steps.append(
        run_step(
            "env_doctor",
            [args.python, str(scripts_dir / "env_doctor.py"), "--report", str(env_report)],
        )
    )

    schema_report = evidence_dir / "schema_preflight_report.json"
    schema_mapping = evidence_dir / "schema_mapping.json"
    steps.append(
        run_step(
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
                "--mapping-out",
                str(schema_mapping),
                "--report",
                str(schema_report),
            ],
        )
    )

    strict_cmd = [
        args.python,
        str(scripts_dir / "run_strict_pipeline.py"),
        "--request",
        str(request_path),
        "--evidence-dir",
        str(evidence_dir),
        "--strict",
    ]
    if args.compare_manifest:
        strict_cmd.extend(["--compare-manifest", str(Path(args.compare_manifest).expanduser().resolve())])
    if args.allow_missing_compare:
        strict_cmd.append("--allow-missing-compare")
    if args.continue_on_fail:
        strict_cmd.append("--continue-on-fail")
    steps.append(run_step("run_strict_pipeline", strict_cmd))

    def _publication_missing_manifest(evidence: Path) -> bool:
        pub_path = evidence / "publication_gate_report.json"
        payload = load_json(pub_path) if pub_path.exists() else {}
        failures = payload.get("failures")
        if not isinstance(failures, list):
            return False
        for row in failures:
            if isinstance(row, dict) and str(row.get("code")) == "manifest_comparison_missing":
                return True
        return False

    strict_step = next((x for x in steps if x["name"] == "run_strict_pipeline"), None)
    strict_exit = int(strict_step["exit_code"]) if isinstance(strict_step, dict) else 2
    bootstrap_baseline_path: Optional[Path] = None

    if (
        strict_exit != 0
        and bool(args.allow_missing_compare)
        and not args.compare_manifest
        and _publication_missing_manifest(evidence_dir)
    ):
        manifest_path = evidence_dir / "manifest.json"
        if manifest_path.exists():
            bootstrap_baseline_path = evidence_dir / "manifest_baseline.bootstrap.json"
            shutil.copy2(manifest_path, bootstrap_baseline_path)
            strict_retry_cmd = [
                args.python,
                str(scripts_dir / "run_strict_pipeline.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence_dir),
                "--strict",
                "--compare-manifest",
                str(bootstrap_baseline_path),
            ]
            if args.continue_on_fail:
                strict_retry_cmd.append("--continue-on-fail")
            steps.append(run_step("run_strict_pipeline_with_bootstrap_baseline", strict_retry_cmd))
            strict_step = next(
                (x for x in steps if x["name"] == "run_strict_pipeline_with_bootstrap_baseline"),
                strict_step,
            )
            strict_exit = int(strict_step["exit_code"]) if isinstance(strict_step, dict) else strict_exit

    summary_md = evidence_dir / "user_summary.md"
    summary_json = evidence_dir / "user_summary.json"
    steps.append(
        run_step(
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
    )

    overall_status = "pass" if strict_exit == 0 else "fail"

    wrapper_report = {
        "status": overall_status,
        "request": str(request_path),
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
    return 0 if strict_exit == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
