"""Unit tests for scripts/_gate_framework.py core functions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _gate_framework import (
    GateBase,
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


# ────────────────────────────────────────────────────────
# GateBase
# ────────────────────────────────────────────────────────

class _StubGate(GateBase):
    """Minimal concrete GateBase for testing."""

    gate_name = "stub_gate"
    gate_version = "0.1.0"
    input_file_args = ("--spec",)

    def __init__(self, check_fn=None, summary_fn=None):
        super().__init__()
        self._check_fn = check_fn
        self._summary_fn = summary_fn

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--spec", help="Spec file.")
        parser.add_argument("--extra", help="Extra arg.")

    def run_checks(self, args: argparse.Namespace) -> None:
        if self._check_fn:
            self._check_fn(self, args)

    def build_summary(self, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
        if self._summary_fn:
            return self._summary_fn(self, args)
        return None


class TestGateBase:
    """Tests for the GateBase abstract class lifecycle."""

    def test_add_failure(self):
        gate = _StubGate()
        gate.add_failure("code_a", "msg_a", {"k": 1})
        assert len(gate._failures) == 1
        issue = gate._failures[0]
        assert issue.code == "code_a"
        assert issue.message == "msg_a"
        assert issue.details == {"k": 1}
        assert issue.severity == Severity.ERROR

    def test_add_warning(self):
        gate = _StubGate()
        gate.add_warning("code_w", "warn msg")
        assert len(gate._warnings) == 1
        issue = gate._warnings[0]
        assert issue.code == "code_w"
        assert issue.severity == Severity.WARNING

    def test_add_failure_with_custom_severity(self):
        gate = _StubGate()
        gate.add_failure("crit_code", "critical msg", severity=Severity.CRITICAL)
        assert gate._failures[0].severity == Severity.CRITICAL

    def test_add_failure_with_remediation(self):
        gate = _StubGate()
        gate.add_failure("x", "m", remediation="do Y")
        assert gate._failures[0].remediation == "do Y"

    def test_add_failure_legacy(self):
        gate = _StubGate()
        bucket: List[Dict[str, Any]] = []
        gate.add_failure_legacy(bucket, "lc", "legacy msg", {"d": 2})
        assert len(bucket) == 1
        assert bucket[0]["code"] == "lc"
        assert len(gate._failures) == 1
        assert gate._failures[0].code == "lc"

    def test_add_warning_legacy(self):
        gate = _StubGate()
        bucket: List[Dict[str, Any]] = []
        gate.add_warning_legacy(bucket, "wl", "warn legacy", {"d": 3})
        assert len(bucket) == 1
        assert len(gate._warnings) == 1
        assert gate._warnings[0].code == "wl"

    def test_create_parser(self):
        gate = _StubGate()
        parser = gate.create_parser()
        args = parser.parse_args(["--report", "/tmp/r.json", "--spec", "/tmp/s.json"])
        assert args.report == "/tmp/r.json"
        assert args.spec == "/tmp/s.json"

    def test_get_description(self):
        gate = _StubGate()
        assert "stub_gate" in gate.get_description()

    def test_get_epilog_default_none(self):
        gate = _StubGate()
        assert gate.get_epilog() is None

    def test_execute_pass_no_failures(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        gate = _StubGate()
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path)])
        assert rc == 0

        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert report["gate_name"] == "stub_gate"
        assert report["gate_version"] == "0.1.0"
        assert report["envelope_version"] == REPORT_ENVELOPE_VERSION
        assert report["failure_count"] == 0
        assert report["warning_count"] == 0

    def test_execute_fail_with_failures(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        def add_issue(gate, args):
            gate.add_failure("test_fail", "something bad")

        gate = _StubGate(check_fn=add_issue)
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path)])
        assert rc == 2

        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        assert report["failure_count"] == 1
        assert report["failures"][0]["code"] == "test_fail"

    def test_execute_strict_mode_fail_on_warnings(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        def add_warn(gate, args):
            gate.add_warning("test_warn", "something meh")

        gate = _StubGate(check_fn=add_warn)
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path), "--strict"])
        assert rc == 2

        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        assert report["strict_mode"] is True
        assert report["warning_count"] == 1

    def test_execute_non_strict_warnings_pass(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        def add_warn(gate, args):
            gate.add_warning("test_warn", "not critical")

        gate = _StubGate(check_fn=add_warn)
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path)])
        assert rc == 0

        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert report["warning_count"] == 1

    def test_execute_missing_spec_file(self, tmp_path):
        report_path = tmp_path / "report.json"

        gate = _StubGate()
        rc = gate.execute(["--spec", str(tmp_path / "missing.json"), "--report", str(report_path)])
        assert rc == 2

        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        assert any("missing" in f["code"] or "not_found" in f["code"] for f in report["failures"])

    def test_execute_dry_run_pass(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")

        gate = _StubGate()
        rc = gate.execute(["--spec", str(spec), "--dry-run"])
        assert rc == 0

    def test_execute_dry_run_fail_missing_input(self, tmp_path):
        gate = _StubGate()
        rc = gate.execute(["--spec", str(tmp_path / "nope.json"), "--dry-run"])
        assert rc == 2

    def test_build_summary_included_in_report(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        def summary_fn(gate, args):
            return {"metric": 0.95, "checked": True}

        gate = _StubGate(summary_fn=summary_fn)
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path)])
        assert rc == 0

        report = json.loads(report_path.read_text())
        assert report["summary"]["metric"] == 0.95
        assert report["summary"]["checked"] is True

    def test_input_files_in_report(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        gate = _StubGate()
        rc = gate.execute(["--spec", str(spec), "--report", str(report_path)])
        assert rc == 0

        report = json.loads(report_path.read_text())
        assert "input_files" in report
        assert "spec" in report["input_files"]

    def test_execute_no_report_file(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")

        gate = _StubGate()
        rc = gate.execute(["--spec", str(spec)])
        assert rc == 0

    def test_critical_file_issue_skips_run_checks(self, tmp_path):
        report_path = tmp_path / "report.json"
        check_called = []

        def track_checks(gate, args):
            check_called.append(True)

        gate = _StubGate(check_fn=track_checks)
        rc = gate.execute(["--spec", str(tmp_path / "missing.json"), "--report", str(report_path)])
        assert rc == 2
        assert len(check_called) == 0

    def test_execution_timestamp_present(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        report_path = tmp_path / "report.json"

        gate = _StubGate()
        gate.execute(["--spec", str(spec), "--report", str(report_path)])

        report = json.loads(report_path.read_text())
        assert "execution_timestamp_utc" in report
        assert "execution_time_seconds" in report
        assert report["execution_time_seconds"] >= 0
