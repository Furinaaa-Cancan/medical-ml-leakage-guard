#!/usr/bin/env python3
"""Unit tests for scripts/explain_gate.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explain_gate import explain_code, explain_report


# ── explain_code ──────────────────────────────────────────────


class TestExplainCode:
    # Exact prefix matches
    def test_patient_overlap_en(self) -> None:
        result = explain_code("patient_overlap", "en")
        assert result["code"] == "patient_overlap"
        assert "Patient IDs" in result["explanation"]
        assert result["fix"]

    def test_patient_overlap_zh(self) -> None:
        result = explain_code("patient_overlap", "zh")
        assert result["code"] == "patient_overlap"
        assert "患者" in result["explanation"]
        assert result["fix"]

    def test_row_overlap(self) -> None:
        result = explain_code("row_overlap", "en")
        assert "Duplicate rows" in result["explanation"]

    def test_temporal_overlap(self) -> None:
        result = explain_code("temporal_overlap", "en")
        assert "Temporal leakage" in result["explanation"]

    def test_temporal_boundary(self) -> None:
        result = explain_code("temporal_boundary_violated", "en")
        assert "boundary" in result["explanation"].lower()

    def test_test_data_usage(self) -> None:
        result = explain_code("test_data_usage_detected", "en")
        assert "Test data" in result["explanation"]

    def test_leakage_generic(self) -> None:
        result = explain_code("leakage_unknown_code", "en")
        assert "leakage" in result["explanation"].lower()

    # Prefix-based matching
    def test_missing_required(self) -> None:
        result = explain_code("missing_required_artifact", "en")
        assert "missing" in result["explanation"].lower()

    def test_missing_split(self) -> None:
        result = explain_code("missing_split_file", "en")
        assert "Split file" in result["explanation"]

    def test_missing_evaluation(self) -> None:
        result = explain_code("missing_evaluation_report", "en")
        assert "Evaluation" in result["explanation"]

    def test_missing_prediction(self) -> None:
        result = explain_code("missing_prediction_trace", "en")
        assert "Prediction" in result["explanation"]

    def test_missing_generic(self) -> None:
        result = explain_code("missing_something_else", "en")
        assert "artifact" in result["explanation"].lower() or "missing" in result["explanation"].lower()

    def test_split_seed_not_locked(self) -> None:
        result = explain_code("split_seed_not_locked", "en")
        assert "seed" in result["explanation"].lower()

    def test_split_generic(self) -> None:
        result = explain_code("split_unknown", "en")
        assert "Split" in result["explanation"] or "split" in result["explanation"].lower()

    def test_covariate_shift(self) -> None:
        result = explain_code("covariate_shift_detected", "en")
        assert "covariate" in result["explanation"].lower() or "shift" in result["explanation"].lower()

    def test_calibration(self) -> None:
        result = explain_code("calibration_ece_too_high", "en")
        assert "Calibration" in result["explanation"] or "calibration" in result["explanation"].lower()

    def test_robustness(self) -> None:
        result = explain_code("robustness_temporal", "en")
        assert "Robustness" in result["explanation"] or "robustness" in result["explanation"].lower()

    def test_seed_stability(self) -> None:
        result = explain_code("seed_stability_exceeded", "en")
        assert "seed" in result["explanation"].lower() or "Seed" in result["explanation"]

    def test_ci_prefix(self) -> None:
        result = explain_code("ci_width_exceeds_threshold", "en")
        assert "Confidence" in result["explanation"] or "confidence" in result["explanation"].lower()

    def test_publication_prefix(self) -> None:
        result = explain_code("publication_gate_not_met", "en")
        assert "Publication" in result["explanation"] or "publication" in result["explanation"].lower()

    def test_gate_timeout(self) -> None:
        result = explain_code("gate_timeout", "en")
        assert "timeout" in result["explanation"].lower()

    # Unknown code fallback
    def test_unknown_code_en(self) -> None:
        result = explain_code("zzz_totally_unknown_code", "en")
        assert result["code"] == "zzz_totally_unknown_code"
        assert "Unknown" in result["explanation"]

    def test_unknown_code_zh(self) -> None:
        result = explain_code("zzz_totally_unknown_code", "zh")
        assert "未知" in result["explanation"]

    # Default language is en
    def test_default_lang(self) -> None:
        result = explain_code("patient_overlap")
        assert "Patient IDs" in result["explanation"]


# ── explain_report ────────────────────────────────────────────


class TestExplainReport:
    def test_empty_report(self) -> None:
        report: Dict[str, Any] = {"status": "pass", "gate": "leakage", "issues": []}
        result = explain_report(report, "en")
        assert result["gate"] == "leakage"
        assert result["status"] == "pass"
        assert result["issue_count"] == 0
        assert result["explanations"] == []

    def test_report_with_issues(self) -> None:
        report: Dict[str, Any] = {
            "status": "fail",
            "gate": "leakage",
            "issues": [
                {"code": "patient_overlap", "message": "Found 5 shared patient IDs."},
                {"code": "row_overlap", "message": "3 duplicate rows."},
            ],
        }
        result = explain_report(report, "en")
        assert result["gate"] == "leakage"
        assert result["status"] == "fail"
        assert result["issue_count"] == 2
        assert result["explanations"][0]["code"] == "patient_overlap"
        assert result["explanations"][0]["original_message"] == "Found 5 shared patient IDs."
        assert result["explanations"][1]["code"] == "row_overlap"

    def test_report_with_gate_name_key(self) -> None:
        report: Dict[str, Any] = {
            "status": "pass",
            "gate_name": "split_protocol",
            "issues": [],
        }
        result = explain_report(report, "en")
        assert result["gate"] == "split_protocol"

    def test_report_zh(self) -> None:
        report: Dict[str, Any] = {
            "status": "fail",
            "gate": "leakage",
            "issues": [{"code": "patient_overlap", "message": "overlap"}],
        }
        result = explain_report(report, "zh")
        assert "患者" in result["explanations"][0]["explanation"]

    def test_non_dict_issues_ignored(self) -> None:
        report: Dict[str, Any] = {
            "status": "fail",
            "gate": "test",
            "issues": ["string_issue", {"code": "leakage", "message": "m"}],
        }
        result = explain_report(report, "en")
        assert result["issue_count"] == 1
        assert result["explanations"][0]["code"] == "leakage"

    def test_missing_status(self) -> None:
        report: Dict[str, Any] = {"gate": "test", "issues": []}
        result = explain_report(report, "en")
        assert result["status"] == "unknown"

    def test_missing_gate(self) -> None:
        report: Dict[str, Any] = {"status": "pass", "issues": []}
        result = explain_report(report, "en")
        assert result["gate"] == "unknown"

    def test_unknown_codes_in_report(self) -> None:
        report: Dict[str, Any] = {
            "status": "fail",
            "gate": "custom",
            "issues": [{"code": "zzz_novel_error", "message": "novel"}],
        }
        result = explain_report(report, "en")
        assert result["issue_count"] == 1
        assert "Unknown" in result["explanations"][0]["explanation"]
