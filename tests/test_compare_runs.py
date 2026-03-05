#!/usr/bin/env python3
"""Unit tests for scripts/compare_runs.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_runs import (
    _extract_codes,
    _extract_metric,
    _extract_status,
    _load_report,
    compare,
    to_markdown,
)


# ── _load_report ──────────────────────────────────────────────


class TestLoadReport:
    def test_missing_file(self, tmp_path: Path) -> None:
        assert _load_report(tmp_path / "no_such_file.json") is None

    def test_valid_json_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "r.json"
        p.write_text('{"status": "pass"}', encoding="utf-8")
        result = _load_report(p)
        assert result == {"status": "pass"}

    def test_valid_json_non_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "r.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert _load_report(p) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "r.json"
        p.write_text("{broken", encoding="utf-8")
        assert _load_report(p) is None


# ── _extract_status ───────────────────────────────────────────


class TestExtractStatus:
    def test_none_report(self) -> None:
        assert _extract_status(None) == "missing"

    def test_pass(self) -> None:
        assert _extract_status({"status": "pass"}) == "pass"

    def test_fail(self) -> None:
        assert _extract_status({"status": "fail"}) == "fail"

    def test_no_status_key(self) -> None:
        assert _extract_status({"other": 1}) == "unknown"


# ── _extract_codes ────────────────────────────────────────────


class TestExtractCodes:
    def test_none_report(self) -> None:
        assert _extract_codes(None) == set()

    def test_no_issues_key(self) -> None:
        assert _extract_codes({"status": "pass"}) == set()

    def test_issues_with_codes(self) -> None:
        report = {
            "issues": [
                {"code": "alpha", "message": "a"},
                {"code": "beta", "message": "b"},
            ]
        }
        assert _extract_codes(report) == {"alpha", "beta"}

    def test_issues_non_dict_entries_ignored(self) -> None:
        report = {"issues": ["string_entry", {"code": "gamma"}]}
        assert _extract_codes(report) == {"gamma"}

    def test_issues_missing_code_key(self) -> None:
        report = {"issues": [{"message": "no code"}]}
        assert _extract_codes(report) == set()

    def test_issues_not_list(self) -> None:
        report = {"issues": "not_a_list"}
        assert _extract_codes(report) == set()


# ── _extract_metric ───────────────────────────────────────────


class TestExtractMetric:
    def test_none_report(self) -> None:
        assert _extract_metric(None, "pr_auc") is None

    def test_existing_int(self) -> None:
        assert _extract_metric({"pr_auc": 1}, "pr_auc") == 1.0

    def test_existing_float(self) -> None:
        assert _extract_metric({"pr_auc": 0.85}, "pr_auc") == 0.85

    def test_missing_key(self) -> None:
        assert _extract_metric({"other": 0.5}, "pr_auc") is None

    def test_string_value(self) -> None:
        assert _extract_metric({"pr_auc": "0.85"}, "pr_auc") is None


# ── compare ───────────────────────────────────────────────────


def _write_report(directory: Path, filename: str, data: Dict[str, Any]) -> None:
    p = directory / filename
    p.write_text(json.dumps(data), encoding="utf-8")


class TestCompare:
    def test_empty_directories(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        result = compare(base, cand)
        assert result["schema_version"] == "v1.0"
        assert result["summary"]["gates_changed"] == 0
        assert result["summary"]["new_failures"] == 0
        assert result["summary"]["resolved_failures"] == 0
        assert result["summary"]["metric_changes"] == 0

    def test_status_change_detected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        _write_report(base, "leakage_report.json", {"status": "pass"})
        _write_report(cand, "leakage_report.json", {"status": "fail"})
        result = compare(base, cand)
        assert result["summary"]["gates_changed"] >= 1
        changed = [d for d in result["gate_status_changes"] if d["gate"] == "leakage"]
        assert len(changed) == 1
        assert changed[0]["baseline_status"] == "pass"
        assert changed[0]["candidate_status"] == "fail"

    def test_new_failure_code(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        _write_report(base, "leakage_report.json", {"status": "pass", "issues": []})
        _write_report(
            cand,
            "leakage_report.json",
            {"status": "fail", "issues": [{"code": "patient_overlap", "message": "overlap"}]},
        )
        result = compare(base, cand)
        assert result["summary"]["new_failures"] >= 1
        assert any(d["code"] == "patient_overlap" for d in result["new_failure_codes"])

    def test_resolved_failure_code(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        _write_report(
            base,
            "leakage_report.json",
            {"status": "fail", "issues": [{"code": "row_overlap", "message": "dup"}]},
        )
        _write_report(cand, "leakage_report.json", {"status": "pass", "issues": []})
        result = compare(base, cand)
        assert result["summary"]["resolved_failures"] >= 1
        assert any(d["code"] == "row_overlap" for d in result["resolved_failure_codes"])

    def test_metric_delta_detected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        _write_report(base, "evaluation_report.json", {"pr_auc": 0.80})
        _write_report(cand, "evaluation_report.json", {"pr_auc": 0.85})
        result = compare(base, cand)
        assert result["summary"]["metric_changes"] >= 1
        pr_delta = [d for d in result["metric_deltas"] if d["metric"] == "pr_auc"]
        assert len(pr_delta) >= 1
        assert pr_delta[0]["delta"] == pytest.approx(0.05, abs=1e-6)

    def test_identical_runs_no_changes(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        for d in (base, cand):
            _write_report(d, "leakage_report.json", {"status": "pass", "issues": []})
            _write_report(d, "evaluation_report.json", {"pr_auc": 0.90})
        result = compare(base, cand)
        assert result["summary"]["gates_changed"] == 0
        assert result["summary"]["new_failures"] == 0
        assert result["summary"]["resolved_failures"] == 0
        assert result["summary"]["metric_changes"] == 0


# ── to_markdown ───────────────────────────────────────────────


class TestToMarkdown:
    def test_basic_structure(self) -> None:
        result = {
            "baseline_dir": "/a",
            "candidate_dir": "/b",
            "pipeline_status": {"baseline": "pass", "candidate": "fail"},
            "gate_status_changes": [],
            "new_failure_codes": [],
            "resolved_failure_codes": [],
            "metric_deltas": [],
            "summary": {
                "gates_changed": 0,
                "new_failures": 0,
                "resolved_failures": 0,
                "metric_changes": 0,
            },
        }
        md = to_markdown(result)
        assert "# Run Comparison Report" in md
        assert "`/a`" in md
        assert "`/b`" in md
        assert "pass → fail" in md

    def test_gate_changes_table(self) -> None:
        result = {
            "baseline_dir": "/a",
            "candidate_dir": "/b",
            "pipeline_status": {"baseline": "pass", "candidate": "pass"},
            "gate_status_changes": [
                {"gate": "leakage", "file": "leakage_report.json", "baseline_status": "pass", "candidate_status": "fail"}
            ],
            "new_failure_codes": [],
            "resolved_failure_codes": [],
            "metric_deltas": [],
            "summary": {"gates_changed": 1, "new_failures": 0, "resolved_failures": 0, "metric_changes": 0},
        }
        md = to_markdown(result)
        assert "## Gate Status Changes" in md
        assert "| leakage | pass | fail |" in md

    def test_metric_deltas_table(self) -> None:
        result = {
            "baseline_dir": "/a",
            "candidate_dir": "/b",
            "pipeline_status": {"baseline": "pass", "candidate": "pass"},
            "gate_status_changes": [],
            "new_failure_codes": [],
            "resolved_failure_codes": [],
            "metric_deltas": [
                {"metric": "pr_auc", "source": "evaluation_report.json", "baseline": 0.8, "candidate": 0.85, "delta": 0.05}
            ],
            "summary": {"gates_changed": 0, "new_failures": 0, "resolved_failures": 0, "metric_changes": 1},
        }
        md = to_markdown(result)
        assert "## Metric Deltas" in md
        assert "pr_auc" in md
        assert "+0.0500" in md

    def test_new_failure_codes_section(self) -> None:
        result = {
            "baseline_dir": "/a",
            "candidate_dir": "/b",
            "pipeline_status": {"baseline": "pass", "candidate": "pass"},
            "gate_status_changes": [],
            "new_failure_codes": [{"gate": "leakage", "code": "row_overlap"}],
            "resolved_failure_codes": [],
            "metric_deltas": [],
            "summary": {"gates_changed": 0, "new_failures": 1, "resolved_failures": 0, "metric_changes": 0},
        }
        md = to_markdown(result)
        assert "## New Failure Codes" in md
        assert "`row_overlap`" in md

    def test_resolved_failure_codes_section(self) -> None:
        result = {
            "baseline_dir": "/a",
            "candidate_dir": "/b",
            "pipeline_status": {"baseline": "pass", "candidate": "pass"},
            "gate_status_changes": [],
            "new_failure_codes": [],
            "resolved_failure_codes": [{"gate": "split_protocol", "code": "split_not_frozen"}],
            "metric_deltas": [],
            "summary": {"gates_changed": 0, "new_failures": 0, "resolved_failures": 1, "metric_changes": 0},
        }
        md = to_markdown(result)
        assert "## Resolved Failure Codes" in md
        assert "`split_not_frozen`" in md
