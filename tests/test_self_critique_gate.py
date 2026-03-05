"""Tests for scripts/self_critique_gate.py.

Covers helper functions (score_component, summarize_recommendations,
warning_is_blocking), component validation, manifest checks, quality
scoring, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "self_critique_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import self_critique_gate as scg


# ── helper functions ─────────────────────────────────────────────────────────

class TestScoreComponent:
    def test_pass_no_warnings(self):
        report = {"status": "pass", "failure_count": 0, "warning_count": 0}
        assert scg.score_component(report, 10.0, 1.0) == 10.0

    def test_pass_with_warnings(self):
        report = {"status": "pass", "failure_count": 0, "warning_count": 3}
        assert scg.score_component(report, 10.0, 1.0) == 7.0

    def test_fail(self):
        report = {"status": "fail", "failure_count": 1, "warning_count": 0}
        assert scg.score_component(report, 10.0, 1.0) == 0.0

    def test_pass_many_warnings_clamped(self):
        report = {"status": "pass", "failure_count": 0, "warning_count": 20}
        assert scg.score_component(report, 10.0, 1.0) == 0.0


class TestSummarizeRecommendations:
    def test_no_issues(self):
        recs = scg.summarize_recommendations([])
        assert len(recs) == 1
        assert "No blocking" in recs[0]

    def test_component_not_passed(self):
        issues = [{"code": "component_not_passed", "details": {"component": "leakage_report"}}]
        recs = scg.summarize_recommendations(issues)
        assert any("Resolve" in r for r in recs)

    def test_attestation_component(self):
        issues = [{"code": "component_has_failures", "details": {"component": "execution_attestation_report"}}]
        recs = scg.summarize_recommendations(issues)
        assert any("attestation" in r.lower() for r in recs)


class TestWarningIsBlocking:
    def test_normal_strict(self):
        args = type("A", (), {"strict": True, "allow_missing_comparison": False})()
        issue = {"code": "some_warning"}
        assert scg.warning_is_blocking(issue, args) is True

    def test_manifest_not_comparable_allowed(self):
        args = type("A", (), {"strict": True, "allow_missing_comparison": True})()
        issue = {"code": "manifest_not_comparable"}
        assert scg.warning_is_blocking(issue, args) is False


# ── CLI integration ──────────────────────────────────────────────────────────

ARTIFACT_NAMES = [
    "request-report", "manifest", "execution-attestation-report",
    "reporting-bias-report", "leakage-report", "split-protocol-report",
    "covariate-shift-report", "definition-report", "lineage-report",
    "imbalance-report", "missingness-report", "tuning-report",
    "model-selection-audit-report", "feature-engineering-audit-report",
    "clinical-metrics-report", "prediction-replay-report",
    "distribution-generalization-report", "generalization-gap-report",
    "robustness-report", "seed-stability-report",
    "external-validation-report", "calibration-dca-report",
    "ci-matrix-report", "metric-report", "evaluation-quality-report",
    "permutation-report", "publication-report",
]


def _make_passing_report(strict_mode=True):
    return {"status": "pass", "failure_count": 0, "warning_count": 0, "strict_mode": strict_mode}


def _make_manifest(status="pass", with_comparison=True, matched=True):
    m = {"status": status, "files": [{"path": "a"}, {"path": "b"}]}
    if with_comparison:
        m["comparison"] = {"matched": matched}
    return m


def _write_artifacts(tmp_path, overrides=None):
    """Write all artifact JSONs and return dict of name -> path."""
    if overrides is None:
        overrides = {}
    paths = {}
    for name in ARTIFACT_NAMES:
        key = name.replace("-", "_")
        if key == "manifest":
            data = overrides.get(key, _make_manifest())
        elif key == "request_report":
            data = overrides.get(key, {
                "status": "pass", "failure_count": 0, "warning_count": 0,
                "strict_mode": True,
                "normalized_request": {"claim_tier_target": "publication-grade"},
            })
        else:
            data = overrides.get(key, _make_passing_report())
        fpath = tmp_path / f"{key}.json"
        fpath.write_text(json.dumps(data))
        paths[name] = str(fpath)
    return paths


def _run_gate(tmp_path, overrides=None, strict=False, allow_missing=False, min_score=None):
    paths = _write_artifacts(tmp_path, overrides)
    report_path = tmp_path / "report.json"
    cmd = [sys.executable, str(GATE_SCRIPT)]
    for name in ARTIFACT_NAMES:
        cmd.extend([f"--{name}", paths[name]])
    cmd.extend(["--report", str(report_path)])
    if strict:
        cmd.append("--strict")
    if allow_missing:
        cmd.append("--allow-missing-comparison")
    if min_score is not None:
        cmd.extend(["--min-score", str(min_score)])
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_all_passing(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0
        assert report["summary"]["quality_score"] > 0

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "quality_score" in report["summary"]
        assert "claim_tier_decision" in report["summary"]
        assert "recommendations" in report["summary"]
        assert "artifacts" in report["summary"]


class TestComponentFailure:
    def test_one_component_fail(self, tmp_path):
        overrides = {"leakage_report": {"status": "fail", "failure_count": 1, "warning_count": 0, "strict_mode": True}}
        report = _run_gate(tmp_path, overrides=overrides)
        codes = [f["code"] for f in report["failures"]]
        assert "component_not_passed" in codes

    def test_component_has_failures(self, tmp_path):
        overrides = {"leakage_report": {"status": "pass", "failure_count": 2, "warning_count": 0, "strict_mode": True}}
        report = _run_gate(tmp_path, overrides=overrides)
        codes = [f["code"] for f in report["failures"]]
        assert "component_has_failures" in codes


class TestManifest:
    def test_manifest_no_comparison(self, tmp_path):
        overrides = {"manifest": _make_manifest(with_comparison=False)}
        report = _run_gate(tmp_path, overrides=overrides)
        all_codes = [f["code"] for f in report["failures"]] + [w["code"] for w in report["warnings"]]
        assert "manifest_not_comparable" in all_codes

    def test_manifest_comparison_mismatch(self, tmp_path):
        overrides = {"manifest": _make_manifest(matched=False)}
        report = _run_gate(tmp_path, overrides=overrides)
        codes = [f["code"] for f in report["failures"]]
        assert "manifest_comparison_mismatch" in codes


class TestQualityScore:
    def test_score_below_threshold(self, tmp_path):
        overrides = {
            "leakage_report": {"status": "fail", "failure_count": 5, "warning_count": 0, "strict_mode": True},
            "definition_report": {"status": "fail", "failure_count": 3, "warning_count": 0, "strict_mode": True},
            "lineage_report": {"status": "fail", "failure_count": 2, "warning_count": 0, "strict_mode": True},
        }
        report = _run_gate(tmp_path, overrides=overrides, min_score=95.0)
        all_codes = [f["code"] for f in report["failures"]] + [w["code"] for w in report["warnings"]]
        assert "insufficient_quality_score" in all_codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True

    def test_not_strict_component(self, tmp_path):
        overrides = {"leakage_report": {"status": "pass", "failure_count": 0, "warning_count": 0, "strict_mode": False}}
        report = _run_gate(tmp_path, overrides=overrides)
        all_codes = [f["code"] for f in report["failures"]] + [w["code"] for w in report["warnings"]]
        assert "component_not_strict" in all_codes


class TestMissingArtifact:
    def test_missing_artifact(self, tmp_path):
        paths = _write_artifacts(tmp_path)
        # Remove one artifact file
        Path(paths["leakage-report"]).unlink()
        report_path = tmp_path / "report.json"
        cmd = [sys.executable, str(GATE_SCRIPT)]
        for name in ARTIFACT_NAMES:
            cmd.extend([f"--{name}", paths[name]])
        cmd.extend(["--report", str(report_path)])
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_or_invalid_artifact" in codes
