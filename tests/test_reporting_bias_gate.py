"""Tests for scripts/reporting_bias_gate.py.

Covers helper functions (load_json, require_section, require_true_fields),
TRIPOD+AI/PROBAST+AI/STARD-AI checklist validation, overall_risk_of_bias,
claim_level, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "reporting_bias_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import reporting_bias_gate as rbg


# ── helper functions ─────────────────────────────────────────────────────────

class TestLoadJson:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "spec.json"
        p.write_text(json.dumps({"key": "value"}))
        failures: List[Dict[str, Any]] = []
        result = rbg.load_json(p, failures)
        assert result == {"key": "value"}
        assert len(failures) == 0

    def test_missing_file(self, tmp_path: Path):
        failures: List[Dict[str, Any]] = []
        result = rbg.load_json(tmp_path / "nope.json", failures)
        assert result is None
        assert failures[0]["code"] == "missing_checklist_spec"

    def test_not_a_file(self, tmp_path: Path):
        failures: List[Dict[str, Any]] = []
        result = rbg.load_json(tmp_path, failures)
        assert result is None
        assert failures[0]["code"] == "invalid_checklist_spec_path"

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid")
        failures: List[Dict[str, Any]] = []
        result = rbg.load_json(p, failures)
        assert result is None
        assert failures[0]["code"] == "invalid_checklist_spec_json"

    def test_non_object_root(self, tmp_path: Path):
        p = tmp_path / "arr.json"
        p.write_text("[1,2,3]")
        failures: List[Dict[str, Any]] = []
        result = rbg.load_json(p, failures)
        assert result is None
        assert failures[0]["code"] == "invalid_checklist_spec_root"


class TestRequireSection:
    def test_valid_section(self):
        failures: List[Dict[str, Any]] = []
        result = rbg.require_section({"sec": {"a": True}}, "sec", failures)
        assert result == {"a": True}
        assert len(failures) == 0

    def test_missing_section(self):
        failures: List[Dict[str, Any]] = []
        result = rbg.require_section({}, "sec", failures)
        assert result is None
        assert failures[0]["code"] == "invalid_checklist_section"

    def test_non_dict_section(self):
        failures: List[Dict[str, Any]] = []
        result = rbg.require_section({"sec": "bad"}, "sec", failures)
        assert result is None
        assert failures[0]["code"] == "invalid_checklist_section"


class TestRequireTrueFields:
    def test_all_true(self):
        failures: List[Dict[str, Any]] = []
        count = rbg.require_true_fields({"a": True, "b": True}, "sec", ["a", "b"], failures)
        assert count == 2
        assert len(failures) == 0

    def test_one_false(self):
        failures: List[Dict[str, Any]] = []
        count = rbg.require_true_fields({"a": True, "b": False}, "sec", ["a", "b"], failures)
        assert count == 1
        assert len(failures) == 1
        assert failures[0]["code"] == "checklist_item_not_satisfied"

    def test_missing_key(self):
        failures: List[Dict[str, Any]] = []
        count = rbg.require_true_fields({"a": True}, "sec", ["a", "b"], failures)
        assert count == 1
        assert len(failures) == 1

    def test_none_section(self):
        failures: List[Dict[str, Any]] = []
        count = rbg.require_true_fields(None, "sec", ["a"], failures)
        assert count == 0
        assert len(failures) == 0

    def test_string_value(self):
        failures: List[Dict[str, Any]] = []
        count = rbg.require_true_fields({"a": "true"}, "sec", ["a"], failures)
        assert count == 0
        assert len(failures) == 1


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_full_checklist() -> dict:
    """Build a fully-passing checklist spec."""
    tripod = {k: True for k in rbg.TRIPOD_REQUIRED_TRUE}
    probast = {k: True for k in rbg.PROBAST_REQUIRED_TRUE}
    stard = {"applicable": False, "not_applicable_justification": "Non-diagnostic prediction model."}
    for k in rbg.STARD_REQUIRED_TRUE:
        stard[k] = False
    return {
        "tripod_ai": tripod,
        "probast_ai": probast,
        "stard_ai": stard,
        "overall_risk_of_bias": "low",
        "claim_level": "publication-grade",
    }


def _run_gate(spec_path: Path, report_path: Path, strict: bool = False) -> dict:
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--checklist-spec", str(spec_path),
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_full_checklist_pass(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_stard_applicable_true_all_met(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["stard_ai"]["applicable"] = True
        for k in rbg.STARD_REQUIRED_TRUE:
            spec["stard_ai"][k] = True
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "pass"

    def test_report_structure(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        s = report["summary"]
        assert "tripod_required_count" in s
        assert "tripod_true_count" in s
        assert "probast_required_count" in s
        assert "probast_true_count" in s
        assert "stard_applicable" in s
        assert "overall_risk_of_bias" in s
        assert "claim_level" in s


class TestTripodFailures:
    def test_single_tripod_item_false(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["tripod_ai"]["title_identifies_prediction_model"] = False
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "checklist_item_not_satisfied" in codes
        items = [f["details"]["item"] for f in report["failures"] if f["code"] == "checklist_item_not_satisfied"]
        assert "title_identifies_prediction_model" in items

    def test_missing_tripod_section(self, tmp_path: Path):
        spec = _make_full_checklist()
        del spec["tripod_ai"]
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_checklist_section" in codes


class TestProbastFailures:
    def test_probast_item_false(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["probast_ai"]["no_data_leakage_signals"] = False
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "fail"
        items = [f["details"]["item"] for f in report["failures"] if f["code"] == "checklist_item_not_satisfied"]
        assert "no_data_leakage_signals" in items

    def test_missing_probast_section(self, tmp_path: Path):
        spec = _make_full_checklist()
        del spec["probast_ai"]
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_checklist_section" in codes


class TestStardAI:
    def test_stard_applicable_missing_items(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["stard_ai"]["applicable"] = True
        # Leave STARD items as False → should fail
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "fail"
        items = [f["details"]["item"] for f in report["failures"] if f["code"] == "checklist_item_not_satisfied" and f["details"]["section"] == "stard_ai"]
        assert len(items) > 0

    def test_stard_not_applicable_needs_justification(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["stard_ai"]["applicable"] = False
        del spec["stard_ai"]["not_applicable_justification"]
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "missing_stard_not_applicable_justification" in codes

    def test_stard_not_applicable_empty_justification(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["stard_ai"]["applicable"] = False
        spec["stard_ai"]["not_applicable_justification"] = "   "
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "missing_stard_not_applicable_justification" in codes

    def test_stard_applicable_not_boolean(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["stard_ai"]["applicable"] = "yes"
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_stard_applicable" in codes


class TestOverallBias:
    def test_overall_bias_not_low(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["overall_risk_of_bias"] = "high"
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "overall_risk_not_low" in codes

    def test_overall_bias_invalid(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["overall_risk_of_bias"] = "medium"
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_overall_risk_of_bias" in codes

    def test_overall_bias_missing(self, tmp_path: Path):
        spec = _make_full_checklist()
        del spec["overall_risk_of_bias"]
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_overall_risk_of_bias" in codes


class TestClaimLevel:
    def test_claim_not_publication(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["claim_level"] = "preliminary"
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "claim_level_not_publication_grade" in codes

    def test_claim_invalid(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec["claim_level"] = "unknown"
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_claim_level" in codes

    def test_claim_missing(self, tmp_path: Path):
        spec = _make_full_checklist()
        del spec["claim_level"]
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json")
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_claim_level" in codes


class TestFileErrors:
    def test_missing_file(self, tmp_path: Path):
        report = _run_gate(tmp_path / "nonexistent.json", tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "missing_checklist_spec" in codes

    def test_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid")
        report = _run_gate(bad, tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_checklist_spec_json" in codes

    def test_non_object_root(self, tmp_path: Path):
        arr = tmp_path / "arr.json"
        arr.write_text("[1,2]")
        report = _run_gate(arr, tmp_path / "report.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_checklist_spec_root" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path: Path):
        spec = _make_full_checklist()
        spec_path = tmp_path / "checklist.json"
        spec_path.write_text(json.dumps(spec))
        report = _run_gate(spec_path, tmp_path / "report.json", strict=True)
        assert report["strict_mode"] is True
