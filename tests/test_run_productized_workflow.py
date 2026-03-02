"""Tests for scripts/run_productized_workflow.py helper functions and CLI.

Covers run_step, infer_project_base, CLI --help,
missing --strict, missing request file, missing split_paths.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "run_productized_workflow.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import run_productized_workflow as rpw


# ── helper functions ─────────────────────────────────────────────────────────

class TestRunStep:
    def test_pass(self):
        step = rpw.run_step("echo_test", [sys.executable, "-c", "print('ok')"])
        assert step["exit_code"] == 0
        assert step["status"] == "pass"
        assert "ok" in step["stdout_tail"]

    def test_fail(self):
        step = rpw.run_step("fail_test", [sys.executable, "-c", "raise SystemExit(1)"])
        assert step["exit_code"] == 1
        assert step["status"] == "fail"

    def test_blocking_flag(self):
        step = rpw.run_step("env_doctor", [sys.executable, "-c", "pass"])
        assert step["blocking"] is True

    def test_non_blocking(self):
        step = rpw.run_step("custom_step", [sys.executable, "-c", "pass"])
        assert step["blocking"] is False


class TestInferProjectBase:
    def test_configs_parent(self, tmp_path):
        configs = tmp_path / "myproject" / "configs"
        configs.mkdir(parents=True)
        req = configs / "request.json"
        req.write_text("{}")
        result = rpw.infer_project_base(req)
        assert result == tmp_path / "myproject"

    def test_flat(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text("{}")
        result = rpw.infer_project_base(req)
        assert result == tmp_path


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "workflow" in proc.stdout.lower()


class TestCLIMissingStrict:
    def test_no_strict(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text('{"split_paths": {}}')
        cmd = [sys.executable, str(GATE_SCRIPT), "--request", str(req)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "strict" in proc.stderr.lower()


class TestCLIMissingRequest:
    def test_missing(self, tmp_path):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(tmp_path / "nope.json"),
            "--strict",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "not found" in proc.stderr.lower()


class TestCLIMissingSplitPaths:
    def test_no_split_paths(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text('{"study_id": "test"}')
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(req),
            "--strict",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "split_paths" in proc.stderr.lower()


class TestCLIMissingSplitFiles:
    def test_split_files_not_found(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text(json.dumps({
            "split_paths": {
                "train": "train.csv",
                "valid": "valid.csv",
                "test": "test.csv",
            }
        }))
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(req),
            "--strict",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "not found" in proc.stderr.lower()


class TestSchemaPreflightStrictForwarding:
    def test_schema_preflight_receives_strict_flag(self, tmp_path):
        project = tmp_path / "project"
        data_dir = project / "data"
        configs = project / "configs"
        evidence = project / "evidence"
        data_dir.mkdir(parents=True)
        configs.mkdir(parents=True)
        evidence.mkdir(parents=True)

        csv_content = "patient_id,event_time,y,x1\nP1,2025-01-01,0,1.0\nP2,2025-01-02,1,2.0\n"
        for split in ("train", "valid", "test"):
            (data_dir / f"{split}.csv").write_text(csv_content, encoding="utf-8")

        request = {
            "split_paths": {
                "train": "../data/train.csv",
                "valid": "../data/valid.csv",
                "test": "../data/test.csv",
            },
            "label_col": "y",
            "patient_id_col": "patient_id",
            "index_time_col": "event_time",
        }
        req_path = configs / "request.json"
        req_path.write_text(json.dumps(request), encoding="utf-8")

        fake_log = tmp_path / "fake_python.log"
        fake_python = tmp_path / "fake_python.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"echo \"$@\" >> \"{fake_log}\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        report_path = evidence / "productized_workflow_report.json"
        cmd = [
            sys.executable,
            str(GATE_SCRIPT),
            "--request",
            str(req_path),
            "--evidence-dir",
            str(evidence),
            "--strict",
            "--allow-missing-compare",
            "--python",
            str(fake_python),
            "--report",
            str(report_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
        log_text = fake_log.read_text(encoding="utf-8")
        schema_lines = [line for line in log_text.splitlines() if "schema_preflight.py" in line]
        assert schema_lines, "schema_preflight invocation not found in fake python log"
        assert any("--strict" in line for line in schema_lines), "schema_preflight invocation must include --strict"
