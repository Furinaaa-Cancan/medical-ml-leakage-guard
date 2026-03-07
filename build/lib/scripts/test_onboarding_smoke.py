#!/usr/bin/env python3
"""
Lightweight smoke checks for onboarding productization layer.

Run:
    python3 scripts/test_onboarding_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: List[str] = []


def assert_true(cond: bool, test_name: str, detail: str = "") -> None:
    if cond:
        print(f"  [{PASS}] {test_name}")
    else:
        print(f"  [{FAIL}] {test_name}" + (f": {detail}" if detail else ""))
        _failures.append(test_name)


def run_cmd(args: List[str], env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run([PYTHON] + args, text=True, capture_output=True, env=run_env)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return payload


def test_onboarding_preview_report_contract() -> None:
    print("\n=== onboarding preview: report contract ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project = td / "demo"
        report = td / "onboarding_report.json"
        proc = run_cmd(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "onboarding",
                "--project-root",
                str(project),
                "--mode",
                "preview",
                "--report",
                str(report),
            ]
        )
        assert_true(proc.returncode == 0, "onboarding preview exits 0")
        assert_true(report.exists(), "onboarding preview report exists")
        payload = load_json(report)
        assert_true(payload.get("contract_version") == "onboarding_report.v2", "report contract_version is v2")
        assert_true(payload.get("stop_on_fail") is True, "default stop_on_fail is true")
        assert_true(payload.get("termination_reason") == "completed_successfully", "preview termination reason is completed_successfully")
        assert_true(payload.get("preview_only") is True, "preview report marks preview_only=true")
        assert_true(payload.get("display_status") == "preview", "preview report display_status is preview")
        next_actions = payload.get("next_actions", [])
        assert_true(
            any("preview-only" in str(item).lower() for item in next_actions),
            "preview next_actions explain non-executed preview mode",
        )
        assert_true(
            any("onboarding --project-root" in str(item) and "--mode guided --yes" in str(item) for item in next_actions),
            "preview next_actions includes direct full-run onboarding command",
        )
        assert_true(
            any("--mode preview" in str(item) for item in next_actions),
            "preview next_actions includes preview rerun command",
        )
        copy_ready = payload.get("copy_ready_commands", {})
        assert_true(isinstance(copy_ready, dict), "copy_ready_commands is a dict")
        assert_true("workflow_bootstrap" in copy_ready, "copy_ready_commands includes workflow_bootstrap")
        assert_true("workflow_compare" in copy_ready, "copy_ready_commands includes workflow_compare")
        assert_true("authority_release" in copy_ready, "copy_ready_commands includes authority_release")
        assert_true("authority_research_heart" in copy_ready, "copy_ready_commands includes authority_research_heart")
        workflow_compare = str(copy_ready.get("workflow_compare", ""))
        assert_true(
            str((SCRIPTS_DIR / "mlgg.py").resolve()) in workflow_compare,
            "copy_ready workflow command uses absolute mlgg.py path",
        )
        steps = payload.get("steps")
        assert_true(isinstance(steps, list) and len(steps) == 8, "report contains 8 onboarding steps")


def test_onboarding_preview_has_required_step_fields() -> None:
    print("\n=== onboarding preview: required step fields ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project = td / "demo"
        report = td / "onboarding_report.json"
        proc = run_cmd(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "onboarding",
                "--project-root",
                str(project),
                "--mode",
                "preview",
                "--report",
                str(report),
            ]
        )
        assert_true(proc.returncode == 0, "onboarding preview exits 0 (step fields test)")
        payload = load_json(report)
        steps = payload.get("steps", [])
        required = {"name", "command", "exit_code", "start_utc", "end_utc", "stdout_tail", "stderr_tail"}
        ok = True
        for row in steps:
            if not isinstance(row, dict):
                ok = False
                break
            if not required.issubset(set(row.keys())):
                ok = False
                break
        assert_true(ok, "every step contains required report fields")


def main() -> int:
    print("Running onboarding smoke tests...")
    test_onboarding_preview_report_contract()
    test_onboarding_preview_has_required_step_fields()
    print(f"\n{'='*50}")
    if _failures:
        print(f"\033[31mFAILED {len(_failures)} test(s):\033[0m")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("\033[32mAll onboarding smoke tests passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
