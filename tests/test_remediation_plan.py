"""Tests for scripts/remediation_plan.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from remediation_plan import (
    _lookup_remediation,
    build_plan,
    collect_issues,
    main,
    to_markdown,
    to_text,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_report(path: Path, status: str = "pass",
                  failures: list | None = None,
                  warnings: list | None = None) -> None:
    data = {
        "status": status,
        "failures": failures or [],
        "warnings": warnings or [],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_evidence(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Create a minimal evidence directory with all gates passing."""
    ev = tmp_path / "evidence"
    ev.mkdir()
    from remediation_plan import GATE_ORDER
    for _, filename in GATE_ORDER:
        _write_report(ev / filename)
    if overrides:
        for filename, data in overrides.items():
            (ev / filename).write_text(json.dumps(data), encoding="utf-8")
    return ev


# ── _lookup_remediation ─────────────────────────────────────────────────────

class TestLookupRemediation:
    def test_known_code(self):
        pri, cat, remedy = _lookup_remediation("row_overlap")
        assert pri == 1
        assert cat == "data"
        assert "Re-split" in remedy

    def test_prefix_match(self):
        pri, cat, remedy = _lookup_remediation("temporal_overlap_train_test")
        assert pri == 1
        assert cat == "data"

    def test_unknown_code(self):
        pri, cat, remedy = _lookup_remediation("zzzz_unknown_code")
        assert pri == 3
        assert cat == "other"
        assert "Review" in remedy

    def test_attestation_code(self):
        pri, cat, remedy = _lookup_remediation("execution_attestation_policy_disabled")
        assert cat == "attestation"

    def test_publication_code(self):
        pri, cat, remedy = _lookup_remediation("component_not_passed")
        assert cat == "publication"


# ── collect_issues ──────────────────────────────────────────────────────────

class TestCollectIssues:
    def test_all_pass_no_issues(self, tmp_path):
        ev = _make_evidence(tmp_path)
        issues = collect_issues(ev)
        assert issues == []

    def test_missing_report(self, tmp_path):
        ev = _make_evidence(tmp_path)
        (ev / "leakage_report.json").unlink()
        issues = collect_issues(ev)
        codes = [i["code"] for i in issues]
        assert "gate_report_missing" in codes
        missing = [i for i in issues if i["gate"] == "leakage"]
        assert len(missing) == 1
        assert missing[0]["severity"] == "error"

    def test_failing_gate(self, tmp_path):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [
                    {"code": "row_overlap", "message": "Duplicate rows found."},
                    {"code": "id_overlap", "message": "IDs overlap."},
                ],
                "warnings": [],
            }
        })
        issues = collect_issues(ev)
        codes = [i["code"] for i in issues]
        assert "row_overlap" in codes
        assert "id_overlap" in codes

    def test_warnings_collected(self, tmp_path):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [{"code": "row_overlap", "message": "dup"}],
                "warnings": [{"code": "suspicious_feature_names", "message": "warn"}],
            }
        })
        issues = collect_issues(ev)
        warning_issues = [i for i in issues if i["severity"] == "warning"]
        assert len(warning_issues) >= 1
        assert warning_issues[0]["priority"] >= 4  # warnings demoted

    def test_empty_evidence_dir(self, tmp_path):
        ev = tmp_path / "empty"
        ev.mkdir()
        issues = collect_issues(ev)
        assert len(issues) > 0  # all reports missing
        assert all(i["code"] == "gate_report_missing" for i in issues)

    def test_invalid_json_treated_as_missing(self, tmp_path):
        ev = _make_evidence(tmp_path)
        (ev / "leakage_report.json").write_text("{bad", encoding="utf-8")
        issues = collect_issues(ev)
        codes = [i["code"] for i in issues]
        assert "gate_report_missing" in codes


# ── build_plan ──────────────────────────────────────────────────────────────

class TestBuildPlan:
    def test_empty_issues(self):
        plan = build_plan([])
        assert plan["total_issues"] == 0
        assert plan["steps"] == []
        assert plan["errors"] == 0
        assert plan["warnings"] == 0

    def test_deduplicates_codes(self):
        issues = [
            {"gate": "leakage", "gate_order": 3, "code": "row_overlap",
             "severity": "error", "message": "dup", "priority": 1,
             "category": "data", "remediation": "fix"},
            {"gate": "split_protocol", "gate_order": 2, "code": "row_overlap",
             "severity": "error", "message": "dup2", "priority": 1,
             "category": "data", "remediation": "fix"},
        ]
        plan = build_plan(issues)
        assert plan["unique_codes"] == 1
        assert plan["total_issues"] == 2
        assert len(plan["steps"]) == 1
        assert plan["steps"][0]["occurrences"] == 2

    def test_sorts_by_priority(self):
        issues = [
            {"gate": "publication", "gate_order": 28, "code": "component_not_passed",
             "severity": "error", "message": "", "priority": 5,
             "category": "publication", "remediation": "fix pub"},
            {"gate": "leakage", "gate_order": 3, "code": "row_overlap",
             "severity": "error", "message": "", "priority": 1,
             "category": "data", "remediation": "fix data"},
        ]
        plan = build_plan(issues)
        assert plan["steps"][0]["code"] == "row_overlap"
        assert plan["steps"][1]["code"] == "component_not_passed"

    def test_gates_affected_count(self):
        issues = [
            {"gate": "a", "gate_order": 0, "code": "x", "severity": "error",
             "message": "", "priority": 1, "category": "c", "remediation": "r"},
            {"gate": "b", "gate_order": 1, "code": "y", "severity": "warning",
             "message": "", "priority": 4, "category": "c", "remediation": "r"},
        ]
        plan = build_plan(issues)
        assert plan["gates_affected"] == 2
        assert plan["errors"] == 1
        assert plan["warnings"] == 1


# ── to_markdown ─────────────────────────────────────────────────────────────

class TestToMarkdown:
    def test_empty_plan(self):
        plan = build_plan([])
        md = to_markdown(plan)
        assert "No issues found" in md

    def test_has_steps(self, tmp_path):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [{"code": "row_overlap", "message": "dup"}],
                "warnings": [],
            }
        })
        issues = collect_issues(ev)
        plan = build_plan(issues)
        md = to_markdown(plan)
        assert "# Remediation Plan" in md
        assert "row_overlap" in md
        assert "CRITICAL" in md


# ── to_text ─────────────────────────────────────────────────────────────────

class TestToText:
    def test_empty_plan(self):
        plan = build_plan([])
        txt = to_text(plan)
        assert "No issues found" in txt

    def test_has_steps(self, tmp_path):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [{"code": "row_overlap", "message": "dup"}],
                "warnings": [],
            }
        })
        issues = collect_issues(ev)
        plan = build_plan(issues)
        txt = to_text(plan)
        assert "Remediation Plan" in txt
        assert "row_overlap" in txt
        assert "CRITICAL" in txt


# ── main() CLI tests ────────────────────────────────────────────────────────

class TestMain:
    def test_all_pass(self, tmp_path, monkeypatch):
        ev = _make_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(ev),
        ])
        rc = main()
        assert rc == 0

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [{"code": "row_overlap", "message": "dup"}],
                "warnings": [],
            }
        })
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(ev), "--json",
        ])
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["total_issues"] > 0
        assert any(s["code"] == "row_overlap" for s in data["steps"])

    def test_markdown_output(self, tmp_path, monkeypatch, capsys):
        ev = _make_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(ev), "--markdown",
        ])
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "# Remediation Plan" in out

    def test_output_to_file(self, tmp_path, monkeypatch):
        ev = _make_evidence(tmp_path)
        out_file = tmp_path / "plan.txt"
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(ev), "--output", str(out_file),
        ])
        rc = main()
        assert rc == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "Remediation Plan" in content

    def test_missing_evidence_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(tmp_path / "nonexistent"),
        ])
        rc = main()
        assert rc == 1

    def test_multiple_failures_across_gates(self, tmp_path, monkeypatch, capsys):
        ev = _make_evidence(tmp_path, overrides={
            "leakage_report.json": {
                "status": "fail",
                "failures": [{"code": "row_overlap", "message": "dup"}],
                "warnings": [],
            },
            "split_protocol_report.json": {
                "status": "fail",
                "failures": [{"code": "split_seed_not_locked", "message": "no seed"}],
                "warnings": [],
            },
        })
        monkeypatch.setattr("sys.argv", [
            "rp", "--evidence-dir", str(ev), "--json",
        ])
        rc = main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["gates_affected"] >= 2
        codes = [s["code"] for s in data["steps"]]
        assert "row_overlap" in codes
        assert "split_seed_not_locked" in codes
