"""Tests for scripts/feature_engineering_audit_gate.py.

Covers helper functions (extract_groups, build_feature_to_group,
collect_forbidden_features), scope validation, stability evidence,
reproducibility checks, forbidden feature detection, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "feature_engineering_audit_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import feature_engineering_audit_gate as feag


# ── helper functions ─────────────────────────────────────────────────────────

class TestExtractGroups:
    def test_valid(self):
        spec = {"groups": {"vitals": ["hr", "bp"], "labs": ["creatinine"]}}
        result = feag.extract_groups(spec)
        assert result == {"vitals": ["hr", "bp"], "labs": ["creatinine"]}

    def test_empty_groups(self):
        assert feag.extract_groups({"groups": {}}) == {}

    def test_missing_groups(self):
        assert feag.extract_groups({}) == {}

    def test_non_dict_groups(self):
        assert feag.extract_groups({"groups": "bad"}) == {}

    def test_empty_feature_list(self):
        result = feag.extract_groups({"groups": {"vitals": []}})
        assert result == {}

    def test_non_list_features(self):
        result = feag.extract_groups({"groups": {"vitals": "hr"}})
        assert result == {}

    def test_strips_names(self):
        result = feag.extract_groups({"groups": {"  vitals  ": ["  hr  "]}})
        assert "vitals" in result
        assert result["vitals"] == ["hr"]

    def test_skips_empty_group_name(self):
        result = feag.extract_groups({"groups": {"": ["hr"], "vitals": ["bp"]}})
        assert "" not in result
        assert "vitals" in result


class TestBuildFeatureToGroup:
    def test_mapping(self):
        groups = {"vitals": ["hr", "bp"], "labs": ["creatinine"]}
        result = feag.build_feature_to_group(groups)
        assert result["hr"] == "vitals"
        assert result["bp"] == "vitals"
        assert result["creatinine"] == "labs"

    def test_first_group_wins(self):
        groups = {"a": ["shared"], "b": ["shared"]}
        result = feag.build_feature_to_group(groups)
        assert result["shared"] == "a"

    def test_empty(self):
        assert feag.build_feature_to_group({}) == {}


class TestCollectForbiddenFeatures:
    def test_explicit_forbidden(self):
        lineage = {"features": {"leak_feat": {"forbidden_for_modeling": True}}}
        result = feag.collect_forbidden_features(lineage)
        assert "leak_feat" in result

    def test_ancestor_target(self):
        lineage = {"features": {"derived_feat": {"ancestors": ["target_column"]}}}
        result = feag.collect_forbidden_features(lineage)
        assert "derived_feat" in result

    def test_ancestor_label(self):
        lineage = {"features": {"derived": {"ancestors": ["label_encoded"]}}}
        result = feag.collect_forbidden_features(lineage)
        assert "derived" in result

    def test_ancestor_outcome(self):
        lineage = {"features": {"derived": {"ancestors": ["outcome_status"]}}}
        result = feag.collect_forbidden_features(lineage)
        assert "derived" in result

    def test_ancestor_diagnosis(self):
        lineage = {"features": {"derived": {"ancestors": ["diagnosis_code"]}}}
        result = feag.collect_forbidden_features(lineage)
        assert "derived" in result

    def test_clean_feature(self):
        lineage = {"features": {"safe_feat": {"ancestors": ["age", "weight"]}}}
        result = feag.collect_forbidden_features(lineage)
        assert "safe_feat" not in result

    def test_missing_features(self):
        assert feag.collect_forbidden_features({}) == set()

    def test_non_dict_features(self):
        assert feag.collect_forbidden_features({"features": "bad"}) == set()


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_group_spec(groups=None):
    if groups is None:
        groups = {"vitals": ["hr", "bp"], "labs": ["creatinine"]}
    return {"groups": groups}


def _make_engineering_report(overrides=None):
    report = {
        "selection_scope": "train_only",
        "data_scopes_used": ["train_only"],
        "selected_features": ["hr", "bp", "creatinine"],
        "stability": {
            "feature_selection_frequency": {"hr": 0.9, "bp": 0.85, "creatinine": 0.8},
            "group_selection_frequency": {"vitals": 0.95, "labs": 0.8},
        },
        "reproducibility": {
            "random_seed": 42,
            "cv": 5,
            "selection_thresholds": {"min_frequency": 0.5},
            "retained_feature_list": ["hr", "bp", "creatinine"],
            "selection_scope": "train_only",
        },
    }
    if overrides:
        report.update(overrides)
    return report


def _make_lineage_spec(overrides=None):
    spec = {
        "features": {
            "hr": {"ancestors": ["heart_rate_raw"]},
            "bp": {"ancestors": ["blood_pressure_raw"]},
            "creatinine": {"ancestors": ["lab_creatinine"]},
        }
    }
    if overrides:
        spec.update(overrides)
    return spec


def _make_tuning_spec(overrides=None):
    spec = {"model_selection_data": "cv_inner"}
    if overrides:
        spec.update(overrides)
    return spec


def _run_gate(tmp_path, group_spec=None, eng_report=None, lineage=None,
              tuning=None, strict=False):
    if group_spec is None:
        group_spec = _make_group_spec()
    if eng_report is None:
        eng_report = _make_engineering_report()
    if lineage is None:
        lineage = _make_lineage_spec()
    if tuning is None:
        tuning = _make_tuning_spec()

    gs_path = tmp_path / "group_spec.json"
    gs_path.write_text(json.dumps(group_spec))
    er_path = tmp_path / "eng_report.json"
    er_path.write_text(json.dumps(eng_report))
    ls_path = tmp_path / "lineage.json"
    ls_path.write_text(json.dumps(lineage))
    ts_path = tmp_path / "tuning.json"
    ts_path.write_text(json.dumps(tuning))
    report_path = tmp_path / "report.json"

    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--feature-group-spec", str(gs_path),
        "--feature-engineering-report", str(er_path),
        "--lineage-spec", str(ls_path),
        "--tuning-spec", str(ts_path),
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_pass(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        s = report["summary"]
        assert "selection_scope" in s
        assert "selected_feature_count" in s
        assert "group_count" in s


class TestScopeValidation:
    def test_invalid_scope(self, tmp_path):
        eng = _make_engineering_report({"selection_scope": "all_data"})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_scope_violation" in codes

    def test_cv_inner_scope_valid(self, tmp_path):
        eng = _make_engineering_report({
            "selection_scope": "cv_inner_train_only",
            "data_scopes_used": ["cv_inner_train_only"],
            "reproducibility": {
                "random_seed": 42, "cv": 5,
                "selection_thresholds": {"min_frequency": 0.5},
                "retained_feature_list": ["hr", "bp", "creatinine"],
                "selection_scope": "cv_inner_train_only",
            },
        })
        report = _run_gate(tmp_path, eng_report=eng)
        scope_failures = [f for f in report["failures"] if f["code"] == "feature_engineering_scope_violation"]
        assert len(scope_failures) == 0

    def test_forbidden_scope_test(self, tmp_path):
        eng = _make_engineering_report({"data_scopes_used": ["train_only", "test"]})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_selection_data_leakage" in codes

    def test_forbidden_scope_valid(self, tmp_path):
        eng = _make_engineering_report({"data_scopes_used": ["train_only", "valid"]})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_selection_data_leakage" in codes

    def test_empty_scopes_used(self, tmp_path):
        eng = _make_engineering_report({"data_scopes_used": []})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_scope_violation" in codes

    def test_missing_train_scope_marker(self, tmp_path):
        eng = _make_engineering_report({"data_scopes_used": ["other_scope"]})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_scope_violation" in codes


class TestForbiddenFeatures:
    def test_forbidden_feature_selected(self, tmp_path):
        lineage = _make_lineage_spec()
        lineage["features"]["hr"]["forbidden_for_modeling"] = True
        report = _run_gate(tmp_path, lineage=lineage)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_selection_data_leakage" in codes

    def test_ancestor_target_leakage(self, tmp_path):
        lineage = _make_lineage_spec()
        lineage["features"]["bp"]["ancestors"] = ["target_encoded"]
        report = _run_gate(tmp_path, lineage=lineage)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_selection_data_leakage" in codes


class TestGroupSpec:
    def test_empty_groups(self, tmp_path):
        report = _run_gate(tmp_path, group_spec={"groups": {}})
        codes = [f["code"] for f in report["failures"]]
        assert "feature_group_spec_missing_or_invalid" in codes

    def test_missing_file(self, tmp_path):
        er_path = tmp_path / "eng.json"
        er_path.write_text(json.dumps(_make_engineering_report()))
        ls_path = tmp_path / "lin.json"
        ls_path.write_text(json.dumps(_make_lineage_spec()))
        ts_path = tmp_path / "tun.json"
        ts_path.write_text(json.dumps(_make_tuning_spec()))
        rp = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--feature-group-spec", str(tmp_path / "nope.json"),
            "--feature-engineering-report", str(er_path),
            "--lineage-spec", str(ls_path),
            "--tuning-spec", str(ts_path),
            "--report", str(rp),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(rp.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "feature_group_spec_missing_or_invalid" in codes

    def test_selected_feature_not_in_group(self, tmp_path):
        eng = _make_engineering_report({
            "selected_features": ["hr", "bp", "creatinine", "unknown_feat"],
            "stability": {
                "feature_selection_frequency": {"hr": 0.9, "bp": 0.85, "creatinine": 0.8, "unknown_feat": 0.7},
                "group_selection_frequency": {"vitals": 0.95, "labs": 0.8},
            },
            "reproducibility": {
                "random_seed": 42, "cv": 5,
                "selection_thresholds": {"min_frequency": 0.5},
                "retained_feature_list": ["hr", "bp", "creatinine", "unknown_feat"],
                "selection_scope": "train_only",
            },
        })
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_group_spec_missing_or_invalid" in codes


class TestStabilityEvidence:
    def test_missing_stability(self, tmp_path):
        eng = _make_engineering_report()
        del eng["stability"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_stability_evidence_missing" in codes

    def test_missing_feature_freq(self, tmp_path):
        eng = _make_engineering_report()
        del eng["stability"]["feature_selection_frequency"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_stability_evidence_missing" in codes

    def test_missing_group_freq(self, tmp_path):
        eng = _make_engineering_report()
        del eng["stability"]["group_selection_frequency"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_stability_evidence_missing" in codes

    def test_feature_missing_from_freq(self, tmp_path):
        eng = _make_engineering_report()
        del eng["stability"]["feature_selection_frequency"]["hr"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_stability_evidence_missing" in codes

    def test_group_missing_from_freq(self, tmp_path):
        eng = _make_engineering_report()
        del eng["stability"]["group_selection_frequency"]["vitals"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_stability_evidence_missing" in codes


class TestReproducibility:
    def test_missing_reproducibility(self, tmp_path):
        eng = _make_engineering_report()
        del eng["reproducibility"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_reproducibility_missing" in codes

    def test_missing_random_seed(self, tmp_path):
        eng = _make_engineering_report()
        del eng["reproducibility"]["random_seed"]
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_reproducibility_missing" in codes

    def test_retained_list_mismatch(self, tmp_path):
        eng = _make_engineering_report()
        eng["reproducibility"]["retained_feature_list"] = ["hr", "bp"]  # missing creatinine
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_reproducibility_missing" in codes

    def test_empty_selected_features(self, tmp_path):
        eng = _make_engineering_report({"selected_features": []})
        report = _run_gate(tmp_path, eng_report=eng)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_reproducibility_missing" in codes


class TestTuningSpec:
    def test_invalid_model_selection_data(self, tmp_path):
        tuning = _make_tuning_spec({"model_selection_data": "all_data"})
        report = _run_gate(tmp_path, tuning=tuning)
        codes = [f["code"] for f in report["failures"]]
        assert "feature_engineering_reproducibility_missing" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True
