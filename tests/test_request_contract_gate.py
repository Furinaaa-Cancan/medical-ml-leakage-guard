"""Tests for scripts/request_contract_gate.py.

Covers helper functions, shape validators, cross-artifact alignment,
performance policy validation, and CLI integration.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "request_contract_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import request_contract_gate as rcg


# ── helpers ──────────────────────────────────────────────────────────────────

class TestIsFiniteNumber:
    def test_int(self):
        assert rcg.is_finite_number(42) is True

    def test_float(self):
        assert rcg.is_finite_number(3.14) is True

    def test_zero(self):
        assert rcg.is_finite_number(0) is True

    def test_negative(self):
        assert rcg.is_finite_number(-1.5) is True

    def test_nan(self):
        assert rcg.is_finite_number(float("nan")) is False

    def test_inf(self):
        assert rcg.is_finite_number(float("inf")) is False

    def test_neg_inf(self):
        assert rcg.is_finite_number(float("-inf")) is False

    def test_bool_false(self):
        assert rcg.is_finite_number(False) is False

    def test_bool_true(self):
        assert rcg.is_finite_number(True) is False

    def test_string(self):
        assert rcg.is_finite_number("3.14") is False

    def test_none(self):
        assert rcg.is_finite_number(None) is False


class TestSha256File:
    def test_known_content(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"hello world\n")
        expected = hashlib.sha256(b"hello world\n").hexdigest()
        assert rcg.sha256_file(f) == expected

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert rcg.sha256_file(f) == expected


class TestMustBeNonEmptyStr:
    def test_valid(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({"key": "value"}, "key", failures)
        assert result == "value"
        assert len(failures) == 0

    def test_whitespace_stripped(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({"key": "  hello  "}, "key", failures)
        assert result == "hello"

    def test_empty_string(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({"key": ""}, "key", failures)
        assert result is None
        assert len(failures) == 1
        assert failures[0]["code"] == "invalid_field"

    def test_whitespace_only(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({"key": "   "}, "key", failures)
        assert result is None
        assert len(failures) == 1

    def test_missing_key(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({}, "key", failures)
        assert result is None
        assert len(failures) == 1

    def test_non_string(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.must_be_non_empty_str({"key": 123}, "key", failures)
        assert result is None
        assert len(failures) == 1


class TestIsValidDotPath:
    def test_simple(self):
        assert rcg.is_valid_dot_path("metrics") is True

    def test_dotted(self):
        assert rcg.is_valid_dot_path("split_metrics.test.pr_auc") is True

    def test_with_numbers(self):
        assert rcg.is_valid_dot_path("block1.value2") is True

    def test_empty(self):
        assert rcg.is_valid_dot_path("") is False

    def test_starts_with_dot(self):
        assert rcg.is_valid_dot_path(".metrics") is False

    def test_ends_with_dot(self):
        assert rcg.is_valid_dot_path("metrics.") is False

    def test_double_dot(self):
        assert rcg.is_valid_dot_path("a..b") is False

    def test_special_chars(self):
        assert rcg.is_valid_dot_path("a-b") is False

    def test_space(self):
        assert rcg.is_valid_dot_path("a b") is False


class TestCanonicalMetricToken:
    def test_basic(self):
        assert rcg.canonical_metric_token("pr_auc") == "prauc"

    def test_uppercase(self):
        assert rcg.canonical_metric_token("PR_AUC") == "prauc"

    def test_mixed_separators(self):
        assert rcg.canonical_metric_token("pr-auc") == "prauc"

    def test_spaces(self):
        assert rcg.canonical_metric_token("pr auc") == "prauc"


class TestToInt:
    def test_int(self):
        assert rcg.to_int(5) == 5

    def test_float_whole(self):
        assert rcg.to_int(5.0) == 5

    def test_float_fractional(self):
        assert rcg.to_int(5.5) is None

    def test_bool(self):
        assert rcg.to_int(True) is None

    def test_string(self):
        assert rcg.to_int("5") is None

    def test_none(self):
        assert rcg.to_int(None) is None

    def test_nan(self):
        assert rcg.to_int(float("nan")) is None

    def test_inf(self):
        assert rcg.to_int(float("inf")) is None


class TestGetGapPairBlock:
    def test_underscore(self):
        data = {"train_valid": {"pr_auc": {"warn": 0.05}}}
        assert rcg.get_gap_pair_block(data, "train", "valid") == {"pr_auc": {"warn": 0.05}}

    def test_dash(self):
        data = {"train-valid": {"pr_auc": {"warn": 0.05}}}
        assert rcg.get_gap_pair_block(data, "train", "valid") == {"pr_auc": {"warn": 0.05}}

    def test_concat(self):
        data = {"trainvalid": {"pr_auc": {"warn": 0.05}}}
        assert rcg.get_gap_pair_block(data, "train", "valid") == {"pr_auc": {"warn": 0.05}}

    def test_not_found(self):
        assert rcg.get_gap_pair_block({}, "train", "valid") is None


class TestRequireNumeric:
    def test_valid(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.require_numeric({"val": 0.95}, "val", failures)
        assert result == 0.95
        assert len(failures) == 0

    def test_missing(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.require_numeric({}, "val", failures)
        assert result is None
        assert len(failures) == 1
        assert failures[0]["code"] == "invalid_numeric_field"

    def test_string(self):
        failures: List[Dict[str, Any]] = []
        result = rcg.require_numeric({"val": "abc"}, "val", failures)
        assert result is None
        assert len(failures) == 1


# ── validate_thresholds ─────────────────────────────────────────────────────

class TestValidateThresholds:
    def test_valid_thresholds(self):
        request = {"thresholds": {"alpha": 0.05, "min_delta": 0.03, "ci_max_width": 0.2, "ci_min_resamples": 500}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        parsed = rcg.validate_thresholds(request, failures, warnings, strict=False)
        assert len(failures) == 0
        assert parsed["alpha"] == 0.05
        assert parsed["ci_min_resamples"] == 500.0

    def test_alpha_out_of_range(self):
        request = {"thresholds": {"alpha": 0.0}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_alpha_range" in codes

    def test_alpha_above_one(self):
        request = {"thresholds": {"alpha": 1.5}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_alpha_range" in codes

    def test_negative_min_delta(self):
        request = {"thresholds": {"min_delta": -0.01}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_min_delta_range" in codes

    def test_ci_max_width_zero(self):
        request = {"thresholds": {"ci_max_width": 0.0}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_ci_max_width_range" in codes

    def test_ci_min_resamples_bool(self):
        request = {"thresholds": {"ci_min_resamples": True}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_value" in codes

    def test_ci_min_resamples_below_one(self):
        request = {"thresholds": {"ci_min_resamples": 0}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_ci_min_resamples_range" in codes

    def test_thresholds_not_dict(self):
        request = {"thresholds": "bad"}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        parsed = rcg.validate_thresholds(request, failures, warnings, strict=False)
        assert parsed == {}
        assert len(failures) == 1
        assert failures[0]["code"] == "invalid_thresholds"

    def test_thresholds_none(self):
        request = {"thresholds": None}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        parsed = rcg.validate_thresholds(request, failures, warnings, strict=False)
        assert parsed == {}

    def test_strict_missing_alpha_warning(self):
        request = {"thresholds": {}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=True)
        codes = [w["code"] for w in warnings]
        assert "missing_threshold_alpha" in codes

    def test_strict_missing_min_delta_warning(self):
        request = {"thresholds": {"alpha": 0.05}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=True)
        codes = [w["code"] for w in warnings]
        assert "missing_threshold_min_delta" in codes

    def test_invalid_threshold_value_type(self):
        request = {"thresholds": {"alpha": "not_a_number"}}
        failures: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        rcg.validate_thresholds(request, failures, warnings, strict=False)
        codes = [f["code"] for f in failures]
        assert "invalid_threshold_value" in codes


# ── shape validators ─────────────────────────────────────────────────────────

class TestValidateEvaluationReportShape:
    def _make_valid_eval(self, tmp_path: Path) -> Path:
        p = tmp_path / "eval.json"
        data = {
            "split_metrics": {
                "train": {"metrics": {"pr_auc": 0.9}, "confusion_matrix": {"tp": 10}},
                "valid": {"metrics": {"pr_auc": 0.85}, "confusion_matrix": {"tp": 8}},
                "test": {"metrics": {"pr_auc": 0.82}, "confusion_matrix": {"tp": 7}},
            },
            "threshold_selection": {"selection_split": "valid", "selected_threshold": 0.5},
            "feature_engineering": {"provenance": {"selected_features": ["a"]}},
            "distribution_summary": {"status": "ok"},
            "ci_matrix_ref": "ci_matrix_report.json",
            "transport_ci_ref": "transport_ci.json",
            "metadata": {
                "imputation": {
                    "policy_strategy": "median",
                    "executed_strategy": "median",
                    "fit_scope": "train_only",
                    "scale_guard": {"method": "standard"},
                },
                "prediction_trace_sha256": "a" * 64,
                "external_validation_report_sha256": "b" * 64,
                "external_cohort_count": 2,
            },
        }
        p.write_text(json.dumps(data))
        return p

    def test_valid_report(self, tmp_path: Path):
        p = self._make_valid_eval(tmp_path)
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_split_metrics(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text(json.dumps({"threshold_selection": {}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "evaluation_report_missing_split_metrics" in codes

    def test_missing_threshold_selection(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text(json.dumps({
            "split_metrics": {
                "train": {"metrics": {}, "confusion_matrix": {}},
                "valid": {"metrics": {}, "confusion_matrix": {}},
                "test": {"metrics": {}, "confusion_matrix": {}},
            }
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "evaluation_report_missing_threshold_selection" in codes

    def test_invalid_selection_split(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text(json.dumps({
            "split_metrics": {
                "train": {"metrics": {}, "confusion_matrix": {}},
                "valid": {"metrics": {}, "confusion_matrix": {}},
                "test": {"metrics": {}, "confusion_matrix": {}},
            },
            "threshold_selection": {"selection_split": "train"},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "threshold_selection_split_invalid" in codes

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        p.write_text("not json")
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "invalid_evaluation_report" in codes

    def test_missing_metadata_trace_sha(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        data = {
            "split_metrics": {
                "train": {"metrics": {}, "confusion_matrix": {}},
                "valid": {"metrics": {}, "confusion_matrix": {}},
                "test": {"metrics": {}, "confusion_matrix": {}},
            },
            "threshold_selection": {"selection_split": "valid"},
            "feature_engineering": {"provenance": {}},
            "distribution_summary": {},
            "ci_matrix_ref": "x",
            "transport_ci_ref": "y",
            "metadata": {
                "imputation": {
                    "policy_strategy": "median",
                    "executed_strategy": "median",
                    "fit_scope": "train_only",
                    "scale_guard": {},
                },
                "prediction_trace_sha256": "short",
                "external_validation_report_sha256": "b" * 64,
                "external_cohort_count": 2,
            },
        }
        p.write_text(json.dumps(data))
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "evaluation_report_missing_prediction_trace_hash" in codes

    def test_external_cohort_count_below_2(self, tmp_path: Path):
        p = tmp_path / "eval.json"
        data = {
            "split_metrics": {
                "train": {"metrics": {}, "confusion_matrix": {}},
                "valid": {"metrics": {}, "confusion_matrix": {}},
                "test": {"metrics": {}, "confusion_matrix": {}},
            },
            "threshold_selection": {"selection_split": "valid"},
            "feature_engineering": {"provenance": {}},
            "distribution_summary": {},
            "ci_matrix_ref": "x",
            "transport_ci_ref": "y",
            "metadata": {
                "imputation": {
                    "policy_strategy": "median",
                    "executed_strategy": "median",
                    "fit_scope": "train_only",
                    "scale_guard": {},
                },
                "prediction_trace_sha256": "a" * 64,
                "external_validation_report_sha256": "b" * 64,
                "external_cohort_count": 1,
            },
        }
        p.write_text(json.dumps(data))
        failures: List[Dict[str, Any]] = []
        rcg.validate_evaluation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "evaluation_report_external_cohort_count_invalid" in codes


class TestValidateModelSelectionReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "ms.json"
        candidates = [
            {"selection_metrics": {"pr_auc": {"n_folds": 5, "fold_scores": [0.8, 0.82, 0.81, 0.83, 0.79]}}}
            for _ in range(3)
        ]
        p.write_text(json.dumps({
            "selection_policy": {"primary_metric": "pr_auc"},
            "candidates": candidates,
            "data_fingerprints": {"train": {}, "valid": {}, "test": {}},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_model_selection_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_too_few_candidates(self, tmp_path: Path):
        p = tmp_path / "ms.json"
        p.write_text(json.dumps({
            "selection_policy": {},
            "candidates": [{"selection_metrics": {"pr_auc": {"n_folds": 5, "fold_scores": [0.8]}}}],
            "data_fingerprints": {},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_model_selection_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "model_selection_invalid_candidates" in codes

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "ms.json"
        p.write_text("{bad")
        failures: List[Dict[str, Any]] = []
        rcg.validate_model_selection_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "invalid_model_selection_report" in codes


class TestValidateSeedSensitivityReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "seed.json"
        p.write_text(json.dumps({
            "primary_metric": "pr_auc",
            "per_seed_results": [{"seed": 1}],
            "summary": {"std": 0.01},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_seed_sensitivity_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_primary_metric(self, tmp_path: Path):
        p = tmp_path / "seed.json"
        p.write_text(json.dumps({"per_seed_results": [{}], "summary": {}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_seed_sensitivity_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "invalid_seed_sensitivity_report" in codes


class TestValidateRobustnessReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "robust.json"
        p.write_text(json.dumps({
            "overall_test_metrics": {"pr_auc": 0.85},
            "time_slices": {"slices": [{"metric": 0.8}]},
            "patient_hash_groups": {"groups": [{"metric": 0.82}]},
            "summary": {"status": "pass"},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_robustness_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_non_finite_pr_auc(self, tmp_path: Path):
        p = tmp_path / "robust.json"
        p.write_text(json.dumps({
            "overall_test_metrics": {"pr_auc": "bad"},
            "time_slices": {"slices": [{}]},
            "patient_hash_groups": {"groups": [{}]},
            "summary": {},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_robustness_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "invalid_robustness_report" in codes


class TestValidateExecutionAttestationShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "attest.json"
        p.write_text(json.dumps({
            "required_artifact_names": [
                "training_log", "training_config", "model_artifact",
                "model_selection_report", "robustness_report",
                "seed_sensitivity_report", "evaluation_report",
                "prediction_trace", "external_validation_report",
            ]
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_execution_attestation_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_artifacts(self, tmp_path: Path):
        p = tmp_path / "attest.json"
        p.write_text(json.dumps({"required_artifact_names": ["training_log"]}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_execution_attestation_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "missing_execution_attestation_required_artifact" in codes


class TestValidateFeatureGroupSpecShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "fg.json"
        p.write_text(json.dumps({"groups": {"lab": ["age", "bp"], "vitals": ["hr"]}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_feature_group_spec_shape(str(p), failures)
        assert len(failures) == 0

    def test_duplicate_feature(self, tmp_path: Path):
        p = tmp_path / "fg.json"
        p.write_text(json.dumps({"groups": {"a": ["age"], "b": ["age"]}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_feature_group_spec_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "feature_group_spec_missing_or_invalid" in codes

    def test_empty_groups(self, tmp_path: Path):
        p = tmp_path / "fg.json"
        p.write_text(json.dumps({"groups": {}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_feature_group_spec_shape(str(p), failures)
        assert len(failures) == 1


class TestValidateExternalCohortSpecShape:
    def test_valid(self, tmp_path: Path):
        # Create cohort data files
        c1 = tmp_path / "cohort1.csv"
        c2 = tmp_path / "cohort2.csv"
        c1.write_text("a,b\n1,2\n")
        c2.write_text("a,b\n3,4\n")
        p = tmp_path / "cohort_spec.json"
        p.write_text(json.dumps({
            "cohorts": [
                {"cohort_id": "c1", "cohort_type": "cross_period", "path": str(c1)},
                {"cohort_id": "c2", "cohort_type": "cross_institution", "path": str(c2)},
            ]
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_external_cohort_spec_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_cohort_type(self, tmp_path: Path):
        c1 = tmp_path / "cohort1.csv"
        c1.write_text("a\n1\n")
        p = tmp_path / "cohort_spec.json"
        p.write_text(json.dumps({
            "cohorts": [
                {"cohort_id": "c1", "cohort_type": "cross_period", "path": str(c1)},
            ]
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_external_cohort_spec_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "external_cohort_spec_missing_supported_type" in codes


class TestValidateDistributionReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "dist.json"
        p.write_text(json.dumps({"schema_version": "2.0", "distribution_matrix": [{"feature": "a"}]}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_distribution_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_matrix(self, tmp_path: Path):
        p = tmp_path / "dist.json"
        p.write_text(json.dumps({"schema_version": "2.0"}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_distribution_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "distribution_report_schema_invalid" in codes


class TestValidateCiMatrixReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "ci.json"
        p.write_text(json.dumps({"split_metrics_ci": {"test": {}}, "transport_drop_ci": {"valid_test": {}}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_ci_matrix_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_transport(self, tmp_path: Path):
        p = tmp_path / "ci.json"
        p.write_text(json.dumps({"split_metrics_ci": {}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_ci_matrix_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "transport_ci_invalid" in codes


class TestValidateFeatureEngineeringReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "fe.json"
        p.write_text(json.dumps({
            "feature_groups": {"lab": ["a"]},
            "stability": {"cv_frequency": 0.8},
            "reproducibility": {"hash": "abc"},
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_feature_engineering_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_stability(self, tmp_path: Path):
        p = tmp_path / "fe.json"
        p.write_text(json.dumps({"feature_groups": {}}))
        failures: List[Dict[str, Any]] = []
        rcg.validate_feature_engineering_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "feature_stability_evidence_missing" in codes


class TestValidateExternalValidationReportShape:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "ext.json"
        p.write_text(json.dumps({
            "cohorts": [
                {"cohort_id": "c1", "cohort_type": "cross_period", "metrics": {"pr_auc": 0.8}, "confusion_matrix": {}},
                {"cohort_id": "c2", "cohort_type": "cross_institution", "metrics": {"pr_auc": 0.75}, "confusion_matrix": {}},
            ]
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_external_validation_report_shape(str(p), failures)
        assert len(failures) == 0

    def test_missing_both_types(self, tmp_path: Path):
        p = tmp_path / "ext.json"
        p.write_text(json.dumps({
            "cohorts": [
                {"cohort_id": "c1", "cohort_type": "cross_period", "metrics": {}, "confusion_matrix": {}},
            ]
        }))
        failures: List[Dict[str, Any]] = []
        rcg.validate_external_validation_report_shape(str(p), failures)
        codes = [f["code"] for f in failures]
        assert "external_validation_report_invalid_cohort" in codes


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_minimal_request(tmp_path: Path) -> Path:
    """Build a minimal leakage-audited request that should pass."""
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    test = tmp_path / "test.csv"
    for f in (train, valid, test):
        f.write_text("patient_id,y\nP001,0\nP002,1\n")

    request = {
        "study_id": "study-001",
        "run_id": "run-001",
        "target_name": "readmission",
        "prediction_unit": "admission",
        "index_time_col": "event_time",
        "label_col": "y",
        "patient_id_col": "patient_id",
        "primary_metric": "pr_auc",
        "phenotype_definition_spec": "pheno.json",
        "claim_tier_target": "leakage-audited",
        "split_paths": {
            "train": str(train),
            "valid": str(valid),
            "test": str(test),
        },
        "thresholds": {"alpha": 0.05, "min_delta": 0.03},
    }
    # Create phenotype spec
    pheno = tmp_path / "pheno.json"
    pheno.write_text(json.dumps({"definition": "test"}))

    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request))
    return req_path


def _run_gate(request_path: Path, report_path: Path, strict: bool = False) -> dict:
    cmd = [sys.executable, str(GATE_SCRIPT), "--request", str(request_path), "--report", str(report_path)]
    if strict:
        cmd.append("--strict")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


class TestCLIIntegration:
    def test_minimal_pass(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_missing_request_file(self, tmp_path: Path):
        report_path = tmp_path / "report.json"
        req = tmp_path / "nonexistent.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "missing_request_file" in codes

    def test_invalid_json(self, tmp_path: Path):
        req = tmp_path / "bad.json"
        req.write_text("{not valid json")
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_request_json" in codes

    def test_non_object_root(self, tmp_path: Path):
        req = tmp_path / "arr.json"
        req.write_text("[1, 2, 3]")
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_request_json" in codes

    def test_missing_required_fields(self, tmp_path: Path):
        req = tmp_path / "empty.json"
        req.write_text("{}")
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_field" in codes

    def test_invalid_claim_tier(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["claim_tier_target"] = "unknown-tier"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_claim_tier_target" in codes

    def test_duplicate_split_paths(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["split_paths"]["valid"] = data["split_paths"]["train"]
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "duplicate_split_path" in codes

    def test_missing_split_file(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["split_paths"]["train"] = str(tmp_path / "does_not_exist.csv")
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "split_path_not_found" in codes

    def test_strict_mode_warnings_become_fail(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        del data["thresholds"]
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path, strict=True)
        assert report["strict_mode"] is True
        if report["warning_count"] > 0:
            assert report["status"] == "fail"

    def test_report_structure(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "request_path" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "normalized_request" in report

    def test_split_paths_not_dict(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["split_paths"] = "bad"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_split_paths" in codes

    def test_evaluation_metric_path_valid(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["evaluation_metric_path"] = "split_metrics.test.metrics.pr_auc"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "pass"
        assert report["normalized_request"]["evaluation_metric_path"] == "split_metrics.test.metrics.pr_auc"

    def test_evaluation_metric_path_mismatch(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["evaluation_metric_path"] = "split_metrics.test.metrics.roc_auc"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        codes = [f["code"] for f in report["failures"]]
        assert "metric_path_metric_mismatch" in codes

    def test_evaluation_metric_path_invalid_dot_path(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["evaluation_metric_path"] = "bad..path"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_field" in codes

    def test_actual_primary_metric_valid(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["actual_primary_metric"] = 0.85
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "pass"
        assert report["normalized_request"]["actual_primary_metric"] == 0.85

    def test_actual_primary_metric_non_numeric(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["actual_primary_metric"] = "not_a_number"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_numeric_field" in codes

    def test_context_non_dict(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["context"] = "not_a_dict"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_context" in codes


class TestPublicationGradeRequest:
    """Test publication-grade specific requirements."""

    def test_publication_requires_pr_auc(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["claim_tier_target"] = "publication-grade"
        data["primary_metric"] = "roc_auc"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        codes = [f["code"] for f in report["failures"]]
        assert "unsupported_primary_metric" in codes

    def test_publication_missing_lineage_fields(self, tmp_path: Path):
        req = _make_minimal_request(tmp_path)
        data = json.loads(req.read_text())
        data["claim_tier_target"] = "publication-grade"
        req.write_text(json.dumps(data))
        report_path = tmp_path / "report.json"
        report = _run_gate(req, report_path)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "missing_required_path" in codes or "missing_publication_grade_v3_field" in codes
