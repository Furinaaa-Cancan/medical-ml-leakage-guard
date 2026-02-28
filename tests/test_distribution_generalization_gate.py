"""Tests for scripts/distribution_generalization_gate.py.

Covers helper functions (parse_ignore_cols, normalize_binary, parse_thresholds,
is_numeric_feature, js_divergence_from_probs, feature_jsd, safe_prevalence,
group_drift_summary, build_external_paths), and CLI integration.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "distribution_generalization_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import distribution_generalization_gate as dgg


# ── helper functions ─────────────────────────────────────────────────────────

class TestParseIgnoreCols:
    def test_basic(self):
        result = dgg.parse_ignore_cols("patient_id,event_time", "y")
        assert "y" in result
        assert "patient_id" in result
        assert "event_time" in result

    def test_target_always_included(self):
        result = dgg.parse_ignore_cols("", "target")
        assert "target" in result

    def test_empty(self):
        result = dgg.parse_ignore_cols("", "y")
        assert result == ["y"]


class TestNormalizeBinary:
    def test_valid(self):
        s = pd.Series([0, 1, 0, 1])
        result = dgg.normalize_binary(s)
        assert result is not None
        assert list(result) == [0, 1, 0, 1]

    def test_non_binary(self):
        s = pd.Series([0, 1, 2])
        assert dgg.normalize_binary(s) is None

    def test_nan(self):
        s = pd.Series([0, 1, None])
        assert dgg.normalize_binary(s) is None


class TestParseThresholds:
    def test_defaults(self):
        result = dgg.parse_thresholds(None)
        assert result["split_classifier_auc_fail"] == 0.75
        assert result["top_feature_jsd_fail"] == 0.30

    def test_custom(self):
        policy = {"distribution_thresholds_v2": {"split_classifier_auc_fail": 0.80}}
        result = dgg.parse_thresholds(policy)
        assert result["split_classifier_auc_fail"] == 0.80


class TestIsNumericFeature:
    def test_numeric(self):
        s = pd.Series(list(range(20)))
        assert bool(dgg.is_numeric_feature(s)) is True

    def test_categorical(self):
        s = pd.Series(["a", "b", "c"] * 10)
        assert bool(dgg.is_numeric_feature(s)) is False


class TestJsDivergenceFromProbs:
    def test_identical(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        result = dgg.js_divergence_from_probs(a, b)
        assert result < 0.001

    def test_different(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = dgg.js_divergence_from_probs(a, b)
        assert result > 0.5


class TestFeatureJsd:
    def test_similar_numeric(self):
        rng = np.random.RandomState(42)
        train = pd.Series(rng.normal(0, 1, 100))
        other = pd.Series(rng.normal(0, 1, 100))
        result = dgg.feature_jsd(train, other)
        assert result is not None
        assert result < 0.3

    def test_different_categorical(self):
        train = pd.Series(["a"] * 50 + ["b"] * 50)
        other = pd.Series(["a"] * 10 + ["b"] * 90)
        result = dgg.feature_jsd(train, other)
        assert result is not None
        assert result > 0.0

    def test_empty(self):
        train = pd.Series(dtype=float)
        other = pd.Series(dtype=float)
        assert dgg.feature_jsd(train, other) is None


class TestSafePrevalence:
    def test_normal(self):
        df = pd.DataFrame({"y": [0, 0, 1, 1]})
        assert dgg.safe_prevalence(df, "y") == 0.5

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert dgg.safe_prevalence(df, "y") is None


class TestGroupDriftSummary:
    def test_basic(self):
        features = [
            {"feature": "hr", "jsd": 0.1, "missing_ratio_delta": 0.05},
            {"feature": "bp", "jsd": 0.2, "missing_ratio_delta": 0.02},
        ]
        groups = {"vitals": ["hr", "bp"]}
        result = dgg.group_drift_summary(features, groups)
        assert "vitals" in result
        assert result["vitals"]["n_features_covered"] == 2
        assert abs(result["vitals"]["mean_jsd"] - 0.15) < 0.001


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_split_csv(tmp_path, name, n=60, rng=None):
    """Generate a synthetic split CSV with overlapping distributions."""
    if rng is None:
        rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "patient_id": [f"p_{name}_{i}" for i in range(n)],
        "event_time": pd.date_range("2020-01-01", periods=n, freq="D"),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
        "hr": rng.normal(70, 10, n),
        "bp": rng.normal(120, 15, n),
        "creatinine": rng.normal(1.0, 0.3, n),
    })
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def _make_external_csv(tmp_path, name="ext_cohort", n=60, rng=None):
    if rng is None:
        rng = np.random.RandomState(99)
    df = pd.DataFrame({
        "patient_id": [f"p_{name}_{i}" for i in range(n)],
        "event_time": pd.date_range("2021-01-01", periods=n, freq="D"),
        "y": rng.choice([0, 1], n, p=[0.5, 0.5]),
        "hr": rng.normal(72, 10, n),
        "bp": rng.normal(122, 15, n),
        "creatinine": rng.normal(1.1, 0.3, n),
    })
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def _make_external_validation_report(tmp_path, ext_csv_path):
    report = {
        "cohorts": [
            {
                "cohort_id": "hospital_b",
                "cohort_type": "cross_institution",
                "data_fingerprint": {"path": str(ext_csv_path), "sha256": "abc123"},
                "metrics": {"roc_auc": 0.80, "pr_auc": 0.75},
                "transport_gap": {
                    "pr_auc_drop_from_internal_test": 0.05,
                    "f2_beta_drop_from_internal_test": 0.03,
                    "brier_increase_from_internal_test": 0.02,
                },
            }
        ]
    }
    path = tmp_path / "external_validation_report.json"
    path.write_text(json.dumps(report))
    return path


def _make_feature_group_spec(tmp_path):
    spec = {"groups": {"vitals": ["hr", "bp"], "labs": ["creatinine"]}}
    path = tmp_path / "feature_group_spec.json"
    path.write_text(json.dumps(spec))
    return path


def _run_gate(tmp_path, strict=False, extra_args=None, skip_setup=False,
              train_path=None, valid_path=None, test_path=None,
              ext_report_path=None, group_spec_path=None):
    if not skip_setup:
        rng = np.random.RandomState(42)
        if train_path is None:
            train_path = _make_split_csv(tmp_path, "train", rng=rng)
        if valid_path is None:
            valid_path = _make_split_csv(tmp_path, "valid", rng=np.random.RandomState(43))
        if test_path is None:
            test_path = _make_split_csv(tmp_path, "test", rng=np.random.RandomState(44))
        if ext_report_path is None:
            ext_csv = _make_external_csv(tmp_path, rng=np.random.RandomState(99))
            ext_report_path = _make_external_validation_report(tmp_path, ext_csv)
        if group_spec_path is None:
            group_spec_path = _make_feature_group_spec(tmp_path)

    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--train", str(train_path),
        "--valid", str(valid_path),
        "--test", str(test_path),
        "--external-validation-report", str(ext_report_path),
        "--feature-group-spec", str(group_spec_path),
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(SCRIPTS_DIR))
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
        assert "thresholds" in s
        assert "distribution_matrix" in s


class TestMissingFiles:
    def test_missing_external_report(self, tmp_path):
        rng = np.random.RandomState(42)
        train = _make_split_csv(tmp_path, "train", rng=rng)
        valid = _make_split_csv(tmp_path, "valid", rng=np.random.RandomState(43))
        test = _make_split_csv(tmp_path, "test", rng=np.random.RandomState(44))
        group_spec = _make_feature_group_spec(tmp_path)
        report = _run_gate(
            tmp_path, train_path=train, valid_path=valid, test_path=test,
            ext_report_path=tmp_path / "nope.json", group_spec_path=group_spec,
        )
        codes = [f["code"] for f in report["failures"]]
        assert "distribution_report_schema_invalid" in codes

    def test_empty_groups(self, tmp_path):
        rng = np.random.RandomState(42)
        train = _make_split_csv(tmp_path, "train", rng=rng)
        valid = _make_split_csv(tmp_path, "valid", rng=np.random.RandomState(43))
        test = _make_split_csv(tmp_path, "test", rng=np.random.RandomState(44))
        ext_csv = _make_external_csv(tmp_path)
        ext_report = _make_external_validation_report(tmp_path, ext_csv)
        empty_spec = tmp_path / "empty_spec.json"
        empty_spec.write_text(json.dumps({"groups": {}}))
        report = _run_gate(
            tmp_path, train_path=train, valid_path=valid, test_path=test,
            ext_report_path=ext_report, group_spec_path=empty_spec,
        )
        codes = [f["code"] for f in report["failures"]]
        assert "distribution_report_schema_invalid" in codes


class TestExternalCohortValidation:
    def test_invalid_cohort_type(self, tmp_path):
        rng = np.random.RandomState(42)
        train = _make_split_csv(tmp_path, "train", rng=rng)
        valid = _make_split_csv(tmp_path, "valid", rng=np.random.RandomState(43))
        test = _make_split_csv(tmp_path, "test", rng=np.random.RandomState(44))
        ext_csv = _make_external_csv(tmp_path)
        ext_report = tmp_path / "ext_report.json"
        ext_report.write_text(json.dumps({
            "cohorts": [{
                "cohort_id": "bad",
                "cohort_type": "invalid_type",
                "data_fingerprint": {"path": str(ext_csv)},
            }]
        }))
        group_spec = _make_feature_group_spec(tmp_path)
        report = _run_gate(
            tmp_path, train_path=train, valid_path=valid, test_path=test,
            ext_report_path=ext_report, group_spec_path=group_spec,
        )
        codes = [f["code"] for f in report["failures"]]
        assert "distribution_report_schema_invalid" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True
