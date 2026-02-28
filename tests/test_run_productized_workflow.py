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
