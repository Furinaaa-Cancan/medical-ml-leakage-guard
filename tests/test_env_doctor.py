"""Tests for scripts/env_doctor.py.

Covers helper functions (parse_required_optional), and CLI integration
for core checks, optional backend detection, strict mode, and report output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "env_doctor.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import env_doctor as ed


# ── helper functions ─────────────────────────────────────────────────────────

class TestParseRequiredOptional:
    def test_empty(self):
        assert ed.parse_required_optional("") == []

    def test_single(self):
        assert ed.parse_required_optional("xgboost") == ["xgboost"]

    def test_multiple(self):
        result = ed.parse_required_optional("xgboost,lightgbm,optuna")
        assert result == ["xgboost", "lightgbm", "optuna"]

    def test_dedup(self):
        result = ed.parse_required_optional("xgboost,xgboost")
        assert result == ["xgboost"]

    def test_whitespace(self):
        result = ed.parse_required_optional(" xgboost , lightgbm ")
        assert result == ["xgboost", "lightgbm"]

    def test_case_insensitive(self):
        result = ed.parse_required_optional("XGBoost")
        assert result == ["xgboost"]


# ── CLI integration ──────────────────────────────────────────────────────────

def _run_gate(tmp_path, require_optional="", strict=False):
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--report", str(report_path),
    ]
    if require_optional:
        cmd.extend(["--require-optional-models", require_optional])
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_basic_pass(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert "summary" in report
        s = report["summary"]
        assert "python_version" in s
        assert "core_packages" in s
        assert "optional_packages" in s

    def test_core_packages_installed(self, tmp_path):
        report = _run_gate(tmp_path)
        core = report["summary"]["core_packages"]
        for pkg in ("numpy", "pandas", "scikit-learn", "joblib"):
            assert core[pkg]["installed"] is True


class TestOptionalBackends:
    def test_missing_required_optional(self, tmp_path):
        # Use a known optional key; if it happens to be installed, skip
        report = _run_gate(tmp_path, require_optional="tabpfn")
        opt = report["summary"]["optional_packages"]
        if not opt.get("tabpfn", {}).get("installed", False):
            codes = [f["code"] for f in report["failures"]]
            assert "optional_backend_missing" in codes
        else:
            assert report["status"] == "pass"

    def test_optional_warning(self, tmp_path):
        report = _run_gate(tmp_path)
        opt = report["summary"]["optional_packages"]
        for key, info in opt.items():
            if not info["installed"]:
                warn_codes = [w["code"] for w in report["warnings"]]
                assert "optional_backend_not_installed" in warn_codes
                break


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True

    def test_strict_promotes_warnings(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        if report["warning_count"] > 0:
            codes = [f["code"] for f in report["failures"]]
            assert "strict_warning_promoted_to_failure" in codes


class TestReportStructure:
    def test_fields(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "cpu_count" in report["summary"]
