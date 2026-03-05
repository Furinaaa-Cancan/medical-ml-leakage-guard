"""Comprehensive unit tests for scripts/tuning_leakage_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tuning_leakage_gate import (
    contains_test_token,
    require_bool,
    require_int,
    require_str,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _good_spec():
    return {
        "search_method": "grid_search",
        "model_selection_data": "valid",
        "early_stopping_data": "valid",
        "preprocessing_fit_scope": "train_only",
        "feature_selection_scope": "train_only",
        "resampling_scope": "train_only",
        "final_model_refit_scope": "train_only",
        "objective_metric": "roc_auc",
        "hyperparameter_trials": 10,
        "test_used_for_model_selection": False,
        "test_used_for_early_stopping": False,
        "test_used_for_threshold_selection": False,
        "test_used_for_calibration": False,
        "outer_evaluation_split_locked": True,
        "random_seed_controlled": True,
        "cv": {
            "enabled": True,
            "type": "group_k_fold",
            "n_splits": 5,
            "nested": False,
            "group_col": "patient_id",
        },
    }


# ────────────────────────────────────────────────────────
# contains_test_token
# ────────────────────────────────────────────────────────

class TestContainsTestToken:
    def test_valid(self):
        assert contains_test_token("valid") is False

    def test_test(self):
        assert contains_test_token("test") is True

    def test_train_plus_test(self):
        assert contains_test_token("train_plus_test") is True

    def test_no_test(self):
        assert contains_test_token("no_test") is False

    def test_without_test(self):
        assert contains_test_token("without_test") is False

    def test_exclude_test(self):
        assert contains_test_token("exclude_test") is False

    def test_none(self):
        assert contains_test_token(None) is False

    def test_empty(self):
        assert contains_test_token("") is False

    def test_latest(self):
        assert contains_test_token("latest") is False

    def test_attest(self):
        assert contains_test_token("attest") is False

    def test_test_split(self):
        assert contains_test_token("test_split") is True

    def test_notest(self):
        assert contains_test_token("notest") is False


# ────────────────────────────────────────────────────────
# require_str / require_bool / require_int
# ────────────────────────────────────────────────────────

class TestRequireStr:
    def test_normal(self):
        f = []
        assert require_str({"k": "val"}, "k", f) == "val"
        assert f == []

    def test_missing(self):
        f = []
        assert require_str({}, "k", f) is None
        assert len(f) == 1

    def test_empty(self):
        f = []
        assert require_str({"k": ""}, "k", f) is None
        assert len(f) == 1


class TestRequireBool:
    def test_true(self):
        f = []
        assert require_bool({"k": True}, "k", f) is True

    def test_false(self):
        f = []
        assert require_bool({"k": False}, "k", f) is False

    def test_not_bool(self):
        f = []
        assert require_bool({"k": "yes"}, "k", f) is None
        assert len(f) == 1


class TestRequireInt:
    def test_int(self):
        f = []
        assert require_int({"k": 5}, "k", f) == 5

    def test_float_integer(self):
        f = []
        assert require_int({"k": 5.0}, "k", f) == 5

    def test_float_non_integer(self):
        f = []
        assert require_int({"k": 5.5}, "k", f) is None
        assert len(f) == 1

    def test_bool_rejected(self):
        f = []
        assert require_int({"k": True}, "k", f) is None
        assert len(f) == 1

    def test_string(self):
        f = []
        assert require_int({"k": "5"}, "k", f) is None
        assert len(f) == 1


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, spec_path, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "tuning_leakage_gate.py"),
            "--tuning-spec", str(spec_path),
            "--report", str(tmp_path / "report.json"),
            "--has-valid-split",
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_pass(self, tmp_path: Path):
        spec_path = _write_json(tmp_path / "spec.json", _good_spec())
        result = self._run(tmp_path, spec_path, extra_args=["--id-col", "patient_id"])
        assert result.returncode == 0, f"stdout: {result.stdout}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_missing_spec(self, tmp_path: Path):
        result = self._run(tmp_path, tmp_path / "nonexistent.json")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_tuning_spec" in codes

    def test_invalid_spec_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{bad", encoding="utf-8")
        result = self._run(tmp_path, p)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_tuning_spec" in codes

    def test_unsupported_search_method(self, tmp_path: Path):
        spec = _good_spec()
        spec["search_method"] = "brute_force"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "unsupported_search_method" in codes

    def test_invalid_preprocessing_scope(self, tmp_path: Path):
        spec = _good_spec()
        spec["preprocessing_fit_scope"] = "all_data"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_scope" in codes

    def test_test_used_for_model_selection_true(self, tmp_path: Path):
        spec = _good_spec()
        spec["test_used_for_model_selection"] = True
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "explicit_test_usage" in codes

    def test_test_used_for_early_stopping_true(self, tmp_path: Path):
        spec = _good_spec()
        spec["test_used_for_early_stopping"] = True
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "explicit_test_usage" in codes

    def test_outer_evaluation_not_locked(self, tmp_path: Path):
        spec = _good_spec()
        spec["outer_evaluation_split_locked"] = False
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "outer_evaluation_not_locked" in codes

    def test_seed_not_controlled(self, tmp_path: Path):
        spec = _good_spec()
        spec["random_seed_controlled"] = False
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "seed_not_controlled" in codes

    def test_invalid_hyperparameter_trials(self, tmp_path: Path):
        spec = _good_spec()
        spec["hyperparameter_trials"] = 0
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_hyperparameter_trials" in codes

    def test_test_data_token_in_model_selection(self, tmp_path: Path):
        spec = _good_spec()
        spec["model_selection_data"] = "train_plus_test"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "test_data_usage_detected" in codes

    def test_invalid_early_stopping_data(self, tmp_path: Path):
        spec = _good_spec()
        spec["early_stopping_data"] = "all_data"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_early_stopping_data" in codes

    def test_invalid_final_refit_scope(self, tmp_path: Path):
        spec = _good_spec()
        spec["final_model_refit_scope"] = "all_data"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_final_model_refit_scope" in codes

    def test_missing_cv_config(self, tmp_path: Path):
        spec = _good_spec()
        del spec["cv"]
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_cv_config" in codes

    def test_unsupported_cv_type(self, tmp_path: Path):
        spec = _good_spec()
        spec["cv"]["type"] = "leave_one_out"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "unsupported_cv_type" in codes

    def test_insufficient_cv_splits(self, tmp_path: Path):
        spec = _good_spec()
        spec["cv"]["n_splits"] = 2
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_cv_splits" in codes

    def test_cv_group_col_mismatch(self, tmp_path: Path):
        spec = _good_spec()
        spec["cv"]["group_col"] = "subject_id"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path, extra_args=["--id-col", "patient_id"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "cv_group_col_mismatch" in codes

    def test_nested_cv_required(self, tmp_path: Path):
        spec = _good_spec()
        spec["model_selection_data"] = "nested_cv"
        spec["cv"]["nested"] = False
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "nested_cv_required" in codes

    def test_cv_inner_without_cv_enabled(self, tmp_path: Path):
        spec = _good_spec()
        spec["model_selection_data"] = "cv_inner"
        spec["cv"]["enabled"] = False
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "cv_inner_without_cv" in codes

    def test_valid_model_selection_without_valid_split(self, tmp_path: Path):
        spec = _good_spec()
        spec_path = _write_json(tmp_path / "spec.json", spec)
        # Do NOT pass --has-valid-split
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "tuning_leakage_gate.py"),
            "--tuning-spec", str(spec_path),
            "--report", str(tmp_path / "report.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "valid_model_selection_without_valid_split" in codes

    def test_report_structure(self, tmp_path: Path):
        spec_path = _write_json(tmp_path / "spec.json", _good_spec())
        self._run(tmp_path, spec_path, extra_args=["--id-col", "patient_id"])
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "tuning_spec" in report.get("input_files", {})
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "has_valid_split" in report["summary"]

    def test_objective_metric_with_test_token(self, tmp_path: Path):
        spec = _good_spec()
        spec["objective_metric"] = "test_auc"
        spec_path = _write_json(tmp_path / "spec.json", spec)
        result = self._run(tmp_path, spec_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_objective_metric" in codes

    def test_nested_cv_overconfigured_warning(self, tmp_path: Path):
        spec = _good_spec()
        spec["model_selection_data"] = "cv_inner"
        spec["cv"]["nested"] = True
        spec_path = _write_json(tmp_path / "spec.json", spec)
        self._run(tmp_path, spec_path)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "nested_cv_overconfigured" in warn_codes
