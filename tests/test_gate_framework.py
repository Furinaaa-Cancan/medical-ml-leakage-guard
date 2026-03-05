"""Unit tests for scripts/_gate_framework.py core functions."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    format_issue_line,
    get_remediation,
    load_gate_report,
    print_gate_summary,
    register_remediation,
    register_remediations,
    validate_input_files,
    wrap_legacy_report,
    REPORT_ENVELOPE_VERSION,
)


# ────────────────────────────────────────────────────────
# Severity
# ────────────────────────────────────────────────────────

class TestSeverity:
    def test_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_rank_ordering(self):
        assert Severity.CRITICAL.rank < Severity.ERROR.rank
        assert Severity.ERROR.rank < Severity.WARNING.rank
        assert Severity.WARNING.rank < Severity.INFO.rank

    def test_lt_operator(self):
        assert Severity.CRITICAL < Severity.ERROR
        assert Severity.ERROR < Severity.WARNING
        assert not (Severity.INFO < Severity.WARNING)

    def test_lt_returns_not_implemented_for_non_severity(self):
        result = Severity.CRITICAL.__lt__("not_severity")
        assert result is NotImplemented


# ────────────────────────────────────────────────────────
# GateIssue
# ────────────────────────────────────────────────────────

class TestGateIssue:
    def test_basic_construction(self):
        issue = GateIssue(code="test_code", severity=Severity.ERROR, message="test msg")
        assert issue.code == "test_code"
        assert issue.severity == Severity.ERROR
        assert issue.message == "test msg"
        assert issue.details == {}
        assert issue.remediation is None
        assert issue.source_file is None

    def test_full_construction(self):
        issue = GateIssue(
            code="c", severity=Severity.WARNING, message="m",
            details={"k": 1}, remediation="fix it", source_file="foo.py",
        )
        assert issue.details == {"k": 1}
        assert issue.remediation == "fix it"
        assert issue.source_file == "foo.py"

    def test_to_dict_minimal(self):
        issue = GateIssue(code="c", severity=Severity.ERROR, message="m")
        d = issue.to_dict()
        assert d == {"code": "c", "severity": "error", "message": "m", "details": {}}
        assert "remediation" not in d
        assert "source_file" not in d

    def test_to_dict_with_remediation(self):
        issue = GateIssue(code="c", severity=Severity.ERROR, message="m", remediation="fix")
        d = issue.to_dict()
        assert d["remediation"] == "fix"

    def test_to_dict_with_source_file(self):
        issue = GateIssue(code="c", severity=Severity.ERROR, message="m", source_file="a.py")
        d = issue.to_dict()
        assert d["source_file"] == "a.py"

    def test_from_legacy_basic(self):
        legacy = {"code": "leak", "message": "found leak", "details": {"col": "id"}}
        issue = GateIssue.from_legacy(legacy, Severity.ERROR)
        assert issue.code == "leak"
        assert issue.severity == Severity.ERROR
        assert issue.message == "found leak"
        assert issue.details == {"col": "id"}

    def test_from_legacy_missing_fields(self):
        issue = GateIssue.from_legacy({}, Severity.WARNING)
        assert issue.code == "unknown"
        assert issue.message == ""
        assert issue.details == {}

    def test_from_legacy_non_dict_details(self):
        legacy = {"code": "c", "message": "m", "details": "not_a_dict"}
        issue = GateIssue.from_legacy(legacy, Severity.ERROR)
        assert issue.details == {}


# ────────────────────────────────────────────────────────
# Remediation registry
# ────────────────────────────────────────────────────────

class TestRemediationRegistry:
    def test_register_and_get(self):
        register_remediation("_test_unique_code_1", "hint_1")
        assert get_remediation("_test_unique_code_1") == "hint_1"

    def test_get_missing_returns_none(self):
        assert get_remediation("_nonexistent_code_xyz") is None

    def test_register_remediations_bulk(self):
        register_remediations({
            "_test_bulk_a": "hint_a",
            "_test_bulk_b": "hint_b",
        })
        assert get_remediation("_test_bulk_a") == "hint_a"
        assert get_remediation("_test_bulk_b") == "hint_b"

    def test_overwrite(self):
        register_remediation("_test_overwrite", "old")
        register_remediation("_test_overwrite", "new")
        assert get_remediation("_test_overwrite") == "new"


# ────────────────────────────────────────────────────────
# build_report_envelope
# ────────────────────────────────────────────────────────

class TestBuildReportEnvelope:
    def _make_issues(self, n_fail=1, n_warn=0):
        failures = [
            GateIssue(code=f"f{i}", severity=Severity.ERROR, message=f"fail {i}")
            for i in range(n_fail)
        ]
        warnings = [
            GateIssue(code=f"w{i}", severity=Severity.WARNING, message=f"warn {i}")
            for i in range(n_warn)
        ]
        return failures, warnings

    def test_basic_envelope_structure(self):
        fi, wi = self._make_issues(1, 1)
        env = build_report_envelope(
            gate_name="test_gate", status="fail", strict_mode=True,
            failures=fi, warnings=wi,
        )
        assert env["envelope_version"] == REPORT_ENVELOPE_VERSION
        assert env["gate_name"] == "test_gate"
        assert env["status"] == "fail"
        assert env["strict_mode"] is True
        assert env["failure_count"] == 1
        assert env["warning_count"] == 1
        assert "execution_timestamp_utc" in env
        assert "execution_time_seconds" in env
        assert len(env["failures"]) == 1
        assert len(env["warnings"]) == 1

    def test_envelope_with_summary(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
            summary={"key": "val"},
        )
        assert env["summary"] == {"key": "val"}

    def test_envelope_without_summary(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
        )
        assert "summary" not in env

    def test_envelope_with_input_files(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
            input_files={"train": "/data/train.csv"},
        )
        assert env["input_files"] == {"train": "/data/train.csv"}

    def test_envelope_empty_input_files_not_included(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
            input_files={},
        )
        assert "input_files" not in env

    def test_envelope_extra_merges_to_top_level(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
            extra={"normalized_request": {"study_id": "s1"}},
        )
        assert env["normalized_request"] == {"study_id": "s1"}

    def test_envelope_gate_version(self):
        fi, wi = self._make_issues(0, 0)
        env = build_report_envelope(
            gate_name="g", status="pass", strict_mode=False,
            failures=fi, warnings=wi,
            gate_version="2.0.0",
        )
        assert env["gate_version"] == "2.0.0"

    def test_failures_sorted_by_severity(self):
        fi = [
            GateIssue(code="warn_level", severity=Severity.ERROR, message="e"),
            GateIssue(code="crit_level", severity=Severity.CRITICAL, message="c"),
        ]
        env = build_report_envelope(
            gate_name="g", status="fail", strict_mode=True,
            failures=fi, warnings=[],
        )
        assert env["failures"][0]["severity"] == "critical"
        assert env["failures"][1]["severity"] == "error"


# ────────────────────────────────────────────────────────
# wrap_legacy_report
# ────────────────────────────────────────────────────────

class TestWrapLegacyReport:
    def test_already_envelope_passthrough(self):
        report = {"envelope_version": "2.0.0", "status": "pass"}
        result = wrap_legacy_report("g", report)
        assert result is report

    def test_legacy_conversion(self):
        legacy = {
            "status": "fail",
            "strict_mode": True,
            "failures": [{"code": "leak", "message": "found", "details": {}}],
            "warnings": [{"code": "w1", "message": "warn", "details": {}}],
        }
        result = wrap_legacy_report("test_gate", legacy)
        assert result["envelope_version"] == REPORT_ENVELOPE_VERSION
        assert result["gate_name"] == "test_gate"
        assert result["status"] == "fail"
        assert result["strict_mode"] is True
        assert result["failure_count"] == 1
        assert result["warning_count"] == 1
        assert result["failures"][0]["code"] == "leak"
        assert result["failures"][0]["severity"] == "error"
        assert result["warnings"][0]["code"] == "w1"
        assert result["warnings"][0]["severity"] == "warning"

    def test_legacy_preserves_summary(self):
        legacy = {"status": "pass", "summary": {"key": "val"}}
        result = wrap_legacy_report("g", legacy)
        assert result["summary"] == {"key": "val"}

    def test_legacy_preserves_normalized_request(self):
        legacy = {"status": "pass", "normalized_request": {"study_id": "s1"}}
        result = wrap_legacy_report("g", legacy)
        assert result["normalized_request"] == {"study_id": "s1"}

    def test_legacy_empty_failures(self):
        legacy = {"status": "pass"}
        result = wrap_legacy_report("g", legacy)
        assert result["failures"] == []
        assert result["warnings"] == []
        assert result["failure_count"] == 0

    def test_legacy_non_dict_failure_items_skipped(self):
        legacy = {"status": "fail", "failures": ["not_a_dict", {"code": "c", "message": "m"}]}
        result = wrap_legacy_report("g", legacy)
        assert len(result["failures"]) == 1

    def test_legacy_remediation_injected(self):
        register_remediation("_test_legacy_rem", "fix this")
        legacy = {"status": "fail", "failures": [{"code": "_test_legacy_rem", "message": "m"}]}
        result = wrap_legacy_report("g", legacy)
        assert result["failures"][0]["remediation"] == "fix this"


# ────────────────────────────────────────────────────────
# validate_input_files
# ────────────────────────────────────────────────────────

class TestValidateInputFiles:
    def test_existing_file_no_issues(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        import argparse
        args = argparse.Namespace(data=str(f))
        issues = validate_input_files(args, ["--data"])
        assert len(issues) == 0

    def test_missing_file_produces_critical(self, tmp_path):
        import argparse
        args = argparse.Namespace(data=str(tmp_path / "nonexistent.csv"))
        issues = validate_input_files(args, ["--data"])
        assert len(issues) == 1
        assert issues[0].code == "file_not_found"
        assert issues[0].severity == Severity.CRITICAL

    def test_directory_produces_not_file(self, tmp_path):
        import argparse
        args = argparse.Namespace(data=str(tmp_path))
        issues = validate_input_files(args, ["--data"])
        assert len(issues) == 1
        assert issues[0].code == "path_not_file"

    def test_none_arg_skipped(self):
        import argparse
        args = argparse.Namespace(data=None)
        issues = validate_input_files(args, ["--data"])
        assert len(issues) == 0


# ────────────────────────────────────────────────────────
# format_issue_line
# ────────────────────────────────────────────────────────

class TestFormatIssueLine:
    def test_basic_format(self):
        issue = GateIssue(code="c", severity=Severity.ERROR, message="m")
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            line = format_issue_line(issue)
        assert "[FAIL]" in line
        assert "c: m" in line

    def test_with_remediation(self):
        issue = GateIssue(code="c", severity=Severity.WARNING, message="m", remediation="do X")
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            line = format_issue_line(issue)
        assert "Fix: do X" in line


# ────────────────────────────────────────────────────────
# load_gate_report
# ────────────────────────────────────────────────────────

class TestLoadGateReport:
    def test_loads_and_wraps_legacy(self, tmp_path):
        report = {"status": "pass", "failures": [], "warnings": []}
        p = tmp_path / "report.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        result = load_gate_report(p, "test_gate")
        assert result["envelope_version"] == REPORT_ENVELOPE_VERSION
        assert result["gate_name"] == "test_gate"

    def test_loads_envelope_passthrough(self, tmp_path):
        report = {"envelope_version": "2.0.0", "gate_name": "g", "status": "pass"}
        p = tmp_path / "report.json"
        p.write_text(json.dumps(report), encoding="utf-8")
        result = load_gate_report(p, "g")
        assert result["envelope_version"] == "2.0.0"
