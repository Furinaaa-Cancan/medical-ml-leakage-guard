"""Unit tests for scripts/report_health_check.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from report_health_check import (
    EXPECTED_REPORTS,
    _gate_name,
    _load_report,
    check_health,
    format_text,
    main as health_main,
)


# ── helpers ──────────────────────────────────────────────────


def _write(directory: Path, filename: str, data: dict) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


# ── _gate_name ───────────────────────────────────────────────


class TestGateName:
    def test_report_suffix(self):
        assert _gate_name("leakage_report.json") == "leakage"

    def test_json_only(self):
        assert _gate_name("manifest.json") == "manifest"


# ── _load_report ─────────────────────────────────────────────


class TestLoadReport:
    def test_missing(self, tmp_path):
        assert _load_report(tmp_path / "nope.json") is None

    def test_valid(self, tmp_path):
        _write(tmp_path, "r.json", {"status": "pass"})
        assert _load_report(tmp_path / "r.json") == {"status": "pass"}

    def test_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert _load_report(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        (tmp_path / "arr.json").write_text("[1]", encoding="utf-8")
        assert _load_report(tmp_path / "arr.json") is None


# ── check_health ─────────────────────────────────────────────


class TestCheckHealth:
    def test_empty_directory(self, tmp_path):
        result = check_health(tmp_path)
        assert result["total_gates"] == len(EXPECTED_REPORTS)
        assert result["present"] == 0
        assert result["missing"] == len(EXPECTED_REPORTS)
        assert result["completeness_pct"] == 0.0

    def test_all_pass(self, tmp_path):
        for filename in EXPECTED_REPORTS:
            _write(tmp_path, filename, {"status": "pass", "failures": [], "warnings": []})
        result = check_health(tmp_path)
        assert result["present"] == len(EXPECTED_REPORTS)
        assert result["passed"] == len(EXPECTED_REPORTS)
        assert result["failed"] == 0
        assert result["missing"] == 0
        assert result["completeness_pct"] == 100.0
        assert result["pass_rate_pct"] == 100.0
        assert any("ready for publication" in r for r in result["recommendations"])

    def test_mixed_status(self, tmp_path):
        _write(tmp_path, "leakage_report.json", {
            "status": "fail",
            "failures": [{"code": "patient_overlap", "message": "overlap"}],
            "warnings": [],
        })
        _write(tmp_path, "split_protocol_report.json", {
            "status": "pass", "failures": [], "warnings": [],
        })
        result = check_health(tmp_path)
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert result["missing"] == len(EXPECTED_REPORTS) - 2
        assert any("failing" in r for r in result["recommendations"])
        assert any("missing" in r for r in result["recommendations"])

    def test_top_failure_codes(self, tmp_path):
        _write(tmp_path, "leakage_report.json", {
            "status": "fail",
            "failures": [
                {"code": "patient_overlap", "message": "a"},
                {"code": "patient_overlap", "message": "b"},
                {"code": "row_overlap", "message": "c"},
            ],
            "warnings": [],
        })
        result = check_health(tmp_path)
        codes = {e["code"]: e["count"] for e in result["top_failure_codes"]}
        assert codes["patient_overlap"] == 2
        assert codes["row_overlap"] == 1

    def test_execution_time_captured(self, tmp_path):
        _write(tmp_path, "leakage_report.json", {
            "status": "pass", "failures": [], "warnings": [],
            "execution_time_seconds": 1.5,
        })
        result = check_health(tmp_path)
        leakage = [g for g in result["gates"] if g["gate"] == "leakage"][0]
        assert leakage["execution_time"] == 1.5


# ── format_text ──────────────────────────────────────────────


class TestFormatText:
    def test_contains_header(self, tmp_path):
        result = check_health(tmp_path)
        text = format_text(result)
        assert "Evidence Health Check" in text

    def test_shows_completeness(self, tmp_path):
        result = check_health(tmp_path)
        text = format_text(result)
        assert "Completeness:" in text
        assert "0.0%" in text

    def test_shows_gate_statuses(self, tmp_path):
        _write(tmp_path, "leakage_report.json", {
            "status": "fail",
            "failures": [{"code": "x", "message": "y"}],
            "warnings": [],
        })
        result = check_health(tmp_path)
        text = format_text(result)
        assert "FAIL" in text
        assert "leakage" in text

    def test_shows_recommendations(self, tmp_path):
        result = check_health(tmp_path)
        text = format_text(result)
        assert "Recommendations:" in text


# ── main() CLI ───────────────────────────────────────────────


class TestHealthMain:
    def test_missing_dir_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path / "nope"),
        ])
        assert health_main() == 1

    def test_basic_run(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path),
        ])
        rc = health_main()
        assert rc == 0
        assert "Evidence Health Check" in capsys.readouterr().out

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path), "--json",
        ])
        rc = health_main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "health_check.v1"

    def test_file_output(self, tmp_path, monkeypatch):
        out = tmp_path / "health.json"
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path),
            "--json", "--output", str(out),
        ])
        rc = health_main()
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_gates"] == len(EXPECTED_REPORTS)

    def test_text_file_output(self, tmp_path, monkeypatch):
        out = tmp_path / "health.txt"
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path),
            "--output", str(out),
        ])
        rc = health_main()
        assert rc == 0
        assert "Evidence Health Check" in out.read_text()

    def test_full_evidence_dir(self, tmp_path, monkeypatch, capsys):
        for filename in EXPECTED_REPORTS:
            _write(tmp_path, filename, {"status": "pass", "failures": [], "warnings": []})
        monkeypatch.setattr("sys.argv", [
            "health", "--evidence-dir", str(tmp_path), "--json",
        ])
        rc = health_main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["completeness_pct"] == 100.0
        assert data["pass_rate_pct"] == 100.0
