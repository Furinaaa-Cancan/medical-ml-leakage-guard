"""Tests for scripts/run_strict_pipeline.py helper functions and CLI.

Covers run_step, ensure_number, finalize, CLI --help,
missing --strict, missing --compare-manifest, missing request file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "run_strict_pipeline.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import run_strict_pipeline as rsp


# ── helper functions ─────────────────────────────────────────────────────────

class TestRunStep:
    def test_echo(self):
        code, stdout, stderr = rsp.run_step("test_echo", [sys.executable, "-c", "print('hello')"])
        assert code == 0
        assert "hello" in stdout

    def test_failure(self):
        code, stdout, stderr = rsp.run_step("test_fail", [sys.executable, "-c", "raise SystemExit(1)"])
        assert code == 1


class TestEnsureNumber:
    def test_int(self):
        assert rsp.ensure_number(42, "x") == 42.0

    def test_float(self):
        assert rsp.ensure_number(3.14, "x") == 3.14

    def test_invalid(self):
        with pytest.raises(ValueError, match="Missing or invalid"):
            rsp.ensure_number("abc", "x")

    def test_none(self):
        with pytest.raises(ValueError, match="Missing or invalid"):
            rsp.ensure_number(None, "x")


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "strict" in proc.stdout.lower()


class TestCLIMissingStrict:
    def test_no_strict(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text('{"study_id": "test"}')
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(req),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "strict" in proc.stderr.lower()


class TestCLIMissingCompare:
    def test_no_compare_no_allow(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text('{"study_id": "test"}')
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(req),
            "--strict",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "compare-manifest" in proc.stderr.lower() or "bootstrap" in proc.stderr.lower()


class TestCLIMissingRequest:
    def test_missing_file(self, tmp_path):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(tmp_path / "nonexistent.json"),
            "--strict",
            "--allow-missing-compare",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "not found" in proc.stderr.lower()


class TestCLIReportGeneration:
    def test_report_written_on_failure(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text('{"study_id": "test"}')
        report = tmp_path / "pipeline_report.json"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--request", str(req),
            "--strict",
            "--allow-missing-compare",
            "--report", str(report),
            "--evidence-dir", str(evidence),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["status"] == "fail"
        assert data["strict_mode"] is True
