"""Tests for scripts/render_user_summary.py.

Covers helper functions (write_text, summarize_gate, get_top_failure_codes,
extract_metrics, extract_gap_rows, extract_external_summary, to_markdown,
derive_next_actions), and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "render_user_summary.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import render_user_summary as rus


# ── helper functions ─────────────────────────────────────────────────────────

class TestWriteText:
    def test_basic(self, tmp_path):
        p = tmp_path / "out.md"
        rus.write_text(p, "hello")
        assert p.read_text() == "hello"

    def test_nested(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "out.md"
        rus.write_text(p, "nested")
        assert p.read_text() == "nested"


class TestSummarizeGate:
    def test_pass(self):
        result = rus.summarize_gate("leakage", {"status": "pass", "failure_count": 0, "warning_count": 1})
        assert result["name"] == "leakage"
        assert result["status"] == "pass"
        assert result["failure_count"] == 0
        assert result["warning_count"] == 1

    def test_missing(self):
        result = rus.summarize_gate("leakage", None)
        assert result["status"] == "missing"

    def test_non_dict(self):
        result = rus.summarize_gate("leakage", "not_a_dict")
        assert result["status"] == "missing"


class TestGetTopFailureCodes:
    def test_basic(self):
        payload = {"failures": [{"code": "a"}, {"code": "b"}, {"code": "a"}]}
        codes = rus.get_top_failure_codes(payload)
        assert codes == ["a", "b"]

    def test_none(self):
        assert rus.get_top_failure_codes(None) == []

    def test_limit(self):
        payload = {"failures": [{"code": f"c{i}"} for i in range(10)]}
        codes = rus.get_top_failure_codes(payload, limit=3)
        assert len(codes) == 3

    def test_no_failures_key(self):
        assert rus.get_top_failure_codes({"status": "pass"}) == []


class TestExtractMetrics:
    def test_basic(self):
        ev = {"metrics": {"pr_auc": 0.85, "roc_auc": 0.90, "brier": 0.12}}
        result = rus.extract_metrics(ev)
        assert result["pr_auc"] == 0.85
        assert result["roc_auc"] == 0.90

    def test_none(self):
        assert rus.extract_metrics(None) == {}

    def test_no_metrics_key(self):
        assert rus.extract_metrics({"status": "pass"}) == {}


class TestExtractGapRows:
    def test_basic(self):
        gap = {
            "summary": {
                "gaps": [
                    {"left_split": "train", "right_split": "test", "metric": "pr_auc",
                     "directional_gap": 0.05, "warn_threshold": 0.08, "fail_threshold": 0.12}
                ]
            }
        }
        rows = rus.extract_gap_rows(gap)
        assert len(rows) == 1
        assert rows[0]["pair"] == "train->test"
        assert rows[0]["metric"] == "pr_auc"

    def test_none(self):
        assert rus.extract_gap_rows(None) == []

    def test_no_summary(self):
        assert rus.extract_gap_rows({"status": "pass"}) == []


class TestExtractExternalSummary:
    def test_basic(self):
        ext = {
            "cohorts": [
                {"cohort_id": "h1", "cohort_type": "cross_institution",
                 "row_count": 50, "positive_count": 10,
                 "metrics": {"pr_auc": 0.80, "f2_beta": 0.75, "brier": 0.15}}
            ]
        }
        result = rus.extract_external_summary(ext)
        assert result["cohort_count"] == 1
        assert result["cohorts"][0]["cohort_id"] == "h1"
        assert result["cohorts"][0]["pr_auc"] == 0.80

    def test_none(self):
        result = rus.extract_external_summary(None)
        assert result["cohort_count"] == 0


class TestToMarkdown:
    def test_returns_string(self):
        summary = {
            "generated_at_utc": "2025-01-01T00:00:00Z",
            "study_id": "S1",
            "run_id": "R1",
            "overall_status": "pass",
            "publication_status": "pass",
            "self_critique_status": "pass",
            "self_critique_score": 97.5,
            "selected_model_id": "logistic",
            "primary_metric": "pr_auc",
            "test_metrics": {"pr_auc": 0.85},
            "gap_rows": [],
            "external_validation": {"cohort_count": 0, "cohorts": []},
            "gate_status": [{"name": "publication_gate", "status": "pass", "failure_count": 0, "warning_count": 0}],
            "next_actions": ["Archive artifacts."],
        }
        md = rus.to_markdown(summary)
        assert "# ML Leakage Guard User Summary" in md
        assert "pr_auc" in md
        assert "pass" in md

    def test_missing_metrics(self):
        summary = {
            "generated_at_utc": "2025-01-01T00:00:00Z",
            "overall_status": "fail",
            "publication_status": "missing",
            "self_critique_status": "missing",
            "self_critique_score": None,
            "test_metrics": {},
            "gap_rows": [],
            "external_validation": {"cohort_count": 0},
            "gate_status": [],
            "next_actions": [],
        }
        md = rus.to_markdown(summary)
        assert "No evaluation metrics found" in md


class TestDeriveNextActions:
    def test_all_pass(self):
        summary = {
            "overall_status": "pass",
            "gate_status": [{"name": "pub", "status": "pass"}],
            "top_failure_codes": {},
        }
        actions = rus.derive_next_actions(summary)
        assert any("Archive" in a for a in actions)

    def test_failing_gate(self):
        summary = {
            "overall_status": "fail",
            "gate_status": [{"name": "leakage", "status": "fail"}],
            "top_failure_codes": {"leakage": ["code_a"]},
        }
        actions = rus.derive_next_actions(summary)
        assert any("leakage" in a for a in actions)
        assert any("code_a" in a for a in actions)


# ── CLI integration ──────────────────────────────────────────────────────────

def _setup_evidence(tmp_path, overall_pass=True):
    ev_dir = tmp_path / "evidence"
    ev_dir.mkdir()

    strict_report = {"status": "pass" if overall_pass else "fail", "failure_count": 0, "warning_count": 0}
    pub_report = {"status": "pass", "failure_count": 0, "warning_count": 0}
    sc_report = {"status": "pass", "failure_count": 0, "warning_count": 0, "quality_score": 97.0}
    eval_report = {"model_id": "logistic", "primary_metric": "pr_auc", "metrics": {"pr_auc": 0.85, "roc_auc": 0.90}}
    ext_report = {"cohorts": []}
    gap_report = {"summary": {"gaps": []}}
    clinical = {"status": "pass", "failure_count": 0, "warning_count": 0}

    for name, data in [
        ("strict_pipeline_report.json", strict_report),
        ("publication_gate_report.json", pub_report),
        ("self_critique_report.json", sc_report),
        ("evaluation_report.json", eval_report),
        ("external_validation_report.json", ext_report),
        ("generalization_gap_report.json", gap_report),
        ("clinical_metrics_report.json", clinical),
    ]:
        (ev_dir / name).write_text(json.dumps(data))

    return ev_dir


class TestCLIPass:
    def test_generates_outputs(self, tmp_path):
        ev_dir = _setup_evidence(tmp_path)
        md_path = tmp_path / "summary.md"
        json_path = tmp_path / "summary.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evidence-dir", str(ev_dir),
            "--out-markdown", str(md_path),
            "--out-json", str(json_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert md_path.exists()
        assert json_path.exists()
        summary = json.loads(json_path.read_text())
        assert summary["overall_status"] == "pass"

    def test_fail_status(self, tmp_path):
        ev_dir = _setup_evidence(tmp_path, overall_pass=False)
        json_path = tmp_path / "summary.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evidence-dir", str(ev_dir),
            "--out-json", str(json_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        summary = json.loads(json_path.read_text())
        assert summary["overall_status"] == "fail"

    def test_with_request(self, tmp_path):
        ev_dir = _setup_evidence(tmp_path)
        req_path = tmp_path / "request.json"
        req_path.write_text(json.dumps({"study_id": "ST1", "run_id": "RN1"}))
        json_path = tmp_path / "summary.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--evidence-dir", str(ev_dir),
            "--request", str(req_path),
            "--out-json", str(json_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        summary = json.loads(json_path.read_text())
        assert summary["study_id"] == "ST1"
        assert summary["run_id"] == "RN1"
