"""Comprehensive unit tests for scripts/imbalance_policy_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from imbalance_policy_gate import (
    normalize_scope_list,
    parse_label,
    read_label_stats,
    require_bool,
    require_number,
    require_str,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_csv(path: Path, target_col: str, labels: list) -> Path:
    lines = [f"patient_id,{target_col}"]
    for i, lab in enumerate(labels):
        lines.append(f"P{i:04d},{lab}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _write_evaluation_report(path: Path, selected_strategy: str) -> Path:
    payload = {
        "status": "pass",
        "metadata": {
            "imbalance": {
                "selected_strategy": selected_strategy,
            }
        },
    }
    return _write_json(path, payload)


def _good_policy():
    return {
        "strategy": "class_weight",
        "fit_scope": "train_only",
        "threshold_selection_split": "valid",
        "calibration_split": "valid",
        "forbid_test_usage": True,
        "imbalance_alert_ratio": 5.0,
        "minimum_minority_cases": 10,
    }


def _make_setup(tmp_path, policy=None, train_labels=None, test_labels=None, valid_labels=None):
    if policy is None:
        policy = _good_policy()
    if train_labels is None:
        train_labels = [0] * 20 + [1] * 20
    if test_labels is None:
        test_labels = [0] * 10 + [1] * 10
    spec_path = _write_json(tmp_path / "policy.json", policy)
    train_path = _write_csv(tmp_path / "train.csv", "y", train_labels)
    test_path = _write_csv(tmp_path / "test.csv", "y", test_labels)
    paths = {"spec": spec_path, "train": train_path, "test": test_path}
    if valid_labels is not None:
        paths["valid"] = _write_csv(tmp_path / "valid.csv", "y", valid_labels)
    return paths


# ────────────────────────────────────────────────────────
# parse_label
# ────────────────────────────────────────────────────────

class TestParseLabel:
    def test_zero_string(self):
        assert parse_label("0") == 0

    def test_one_string(self):
        assert parse_label("1") == 1

    def test_zero_float_string(self):
        assert parse_label("0.0") == 0

    def test_one_float_string(self):
        assert parse_label("1.0") == 1

    def test_empty(self):
        assert parse_label("") is None

    def test_whitespace(self):
        assert parse_label("   ") is None

    def test_invalid(self):
        assert parse_label("bad") is None

    def test_two(self):
        assert parse_label("2") is None

    def test_negative(self):
        assert parse_label("-1") is None

    def test_inf(self):
        assert parse_label("inf") is None

    def test_nan(self):
        assert parse_label("nan") is None

    def test_whitespace_around(self):
        assert parse_label("  1  ") == 1


# ────────────────────────────────────────────────────────
# read_label_stats
# ────────────────────────────────────────────────────────

class TestReadLabelStats:
    def test_normal(self, tmp_path: Path):
        p = _write_csv(tmp_path / "d.csv", "y", [0, 0, 1, 1, 1])
        stats = read_label_stats(str(p), "train", "y")
        assert stats["row_count"] == 5
        assert stats["positive_count"] == 3
        assert stats["negative_count"] == 2
        assert stats["invalid_label_rows"] == 0
        assert stats["minority_count"] == 2
        assert stats["majority_count"] == 3

    def test_all_positive(self, tmp_path: Path):
        p = _write_csv(tmp_path / "d.csv", "y", [1, 1, 1])
        stats = read_label_stats(str(p), "train", "y")
        assert stats["positive_count"] == 3
        assert stats["negative_count"] == 0
        assert stats["minority_count"] == 0

    def test_invalid_labels(self, tmp_path: Path):
        p = _write_csv(tmp_path / "d.csv", "y", [0, 1, "bad", ""])
        stats = read_label_stats(str(p), "train", "y")
        assert stats["invalid_label_rows"] == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_label_stats("/nonexistent.csv", "train", "y")

    def test_missing_target_col(self, tmp_path: Path):
        p = tmp_path / "d.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing target_col"):
            read_label_stats(str(p), "train", "y")

    def test_prevalence(self, tmp_path: Path):
        p = _write_csv(tmp_path / "d.csv", "y", [0, 0, 0, 1])
        stats = read_label_stats(str(p), "train", "y")
        assert abs(stats["prevalence"] - 0.25) < 0.01


# ────────────────────────────────────────────────────────
# require_str / require_bool / require_number
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

    def test_empty_string(self):
        f = []
        assert require_str({"k": ""}, "k", f) is None
        assert len(f) == 1


class TestRequireBool:
    def test_true(self):
        f = []
        assert require_bool({"k": True}, "k", f) is True
        assert f == []

    def test_false(self):
        f = []
        assert require_bool({"k": False}, "k", f) is False

    def test_not_bool(self):
        f = []
        assert require_bool({"k": "yes"}, "k", f) is None
        assert len(f) == 1


class TestRequireNumber:
    def test_normal(self):
        f = []
        assert require_number({"k": 5.0}, "k", f) == 5.0

    def test_missing_uses_default(self):
        f = []
        assert require_number({}, "k", f, default=10.0) == 10.0
        assert f == []

    def test_bool_rejected(self):
        f = []
        assert require_number({"k": True}, "k", f, default=1.0) == 1.0
        assert len(f) == 1

    def test_inf_rejected(self):
        f = []
        assert require_number({"k": float("inf")}, "k", f, default=0.0) == 0.0
        assert len(f) == 1

    def test_string_number(self):
        f = []
        assert require_number({"k": "3.14"}, "k", f) == 3.14
        assert f == []


# ────────────────────────────────────────────────────────
# normalize_scope_list
# ────────────────────────────────────────────────────────

class TestNormalizeScopeList:
    def test_normal(self):
        f = []
        assert normalize_scope_list(["Train"], "field", f) == ["train"]

    def test_none(self):
        f = []
        assert normalize_scope_list(None, "field", f) == []

    def test_not_list(self):
        f = []
        assert normalize_scope_list("train", "field", f) is None
        assert len(f) == 1

    def test_empty_string_item(self):
        f = []
        assert normalize_scope_list(["train", ""], "field", f) is None
        assert len(f) == 1

    def test_dedup_and_sort(self):
        f = []
        assert normalize_scope_list(["Train", "train"], "field", f) == ["train"]


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, setup, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "imbalance_policy_gate.py"),
            "--policy-spec", str(setup["spec"]),
            "--train", str(setup["train"]),
            "--test", str(setup["test"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if "valid" in setup:
            cmd.extend(["--valid", str(setup["valid"])])
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_pass(self, tmp_path: Path):
        setup = _make_setup(tmp_path, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_smote_train_only_pass(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote_train_only"
        policy["resampling_applied_to"] = ["train"]
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0

    def test_smote_alias_pass(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote"
        policy["resampling_applied_to"] = ["train"]
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["strategy"] == "smote"

    def test_smote_all_splits_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote_train_only"
        policy["resampling_applied_to"] = ["train", "valid", "test"]
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "resampling_scope_leakage" in codes

    def test_smote_missing_scope_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote_train_only"
        # No resampling_applied_to
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_resampling_scope" in codes

    def test_invalid_strategy(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "unknown_strategy"
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "unsupported_imbalance_strategy" in codes

    def test_invalid_fit_scope(self, tmp_path: Path):
        policy = _good_policy()
        policy["fit_scope"] = "all_splits"
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_fit_scope" in codes

    def test_forbid_test_usage_false_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["forbid_test_usage"] = False
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "test_usage_not_forbidden" in codes

    def test_missing_policy_spec(self, tmp_path: Path):
        train_path = _write_csv(tmp_path / "train.csv", "y", [0, 1])
        test_path = _write_csv(tmp_path / "test.csv", "y", [0, 1])
        setup = {"spec": tmp_path / "nonexistent.json", "train": train_path, "test": test_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_policy_spec" in codes

    def test_invalid_policy_json(self, tmp_path: Path):
        spec_path = tmp_path / "bad.json"
        spec_path.write_text("{bad json", encoding="utf-8")
        train_path = _write_csv(tmp_path / "train.csv", "y", [0, 1])
        test_path = _write_csv(tmp_path / "test.csv", "y", [0, 1])
        setup = {"spec": spec_path, "train": train_path, "test": test_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_policy_spec" in codes

    def test_single_class_split_fails(self, tmp_path: Path):
        setup = _make_setup(
            tmp_path,
            train_labels=[0] * 20 + [1] * 20,
            test_labels=[0] * 20,  # single class
            valid_labels=[0] * 10 + [1] * 10,
        )
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "single_class_split" in codes

    def test_insufficient_minority(self, tmp_path: Path):
        policy = _good_policy()
        policy["minimum_minority_cases"] = 50
        setup = _make_setup(
            tmp_path, policy=policy,
            train_labels=[0] * 30 + [1] * 10,  # minority=10 < 50
            valid_labels=[0] * 10 + [1] * 10,
        )
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_minority_samples" in codes

    def test_imbalance_unmitigated(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "none"
        policy["imbalance_alert_ratio"] = 3.0
        setup = _make_setup(
            tmp_path, policy=policy,
            train_labels=[0] * 40 + [1] * 5,  # ratio=8 > 3
            valid_labels=[0] * 10 + [1] * 10,
        )
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "imbalance_unmitigated" in codes

    def test_invalid_labels_in_split(self, tmp_path: Path):
        setup = _make_setup(
            tmp_path,
            train_labels=[0, 1, "bad", ""],
            valid_labels=[0] * 10 + [1] * 10,
        )
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_labels" in codes

    def test_report_structure(self, tmp_path: Path):
        setup = _make_setup(tmp_path, valid_labels=[0] * 10 + [1] * 10)
        self._run(tmp_path, setup)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "strategy" in report["summary"]
        assert "policy_spec" in report.get("summary", {}).get("policy_fields_present", []) or "policy_spec" in report.get("input_files", {})
        assert "failure_count" in report
        assert "warning_count" in report
        assert "summary" in report
        assert "splits" in report["summary"]
        for split_name in ("train", "test"):
            s = report["summary"]["splits"][split_name]
            assert "row_count" in s
            assert "positive_count" in s
            assert "negative_count" in s
            assert "prevalence" in s

    def test_threshold_on_test_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["threshold_selection_split"] = "test"
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_postprocessing_split" in codes

    def test_calibration_on_train_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["calibration_split"] = "train"
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_postprocessing_split" in codes

    def test_evaluation_reconciliation_match_pass(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote"
        policy["resampling_applied_to"] = ["train"]
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        eval_path = _write_evaluation_report(tmp_path / "evaluation_report.json", "smote")
        result = self._run(tmp_path, setup, ["--evaluation-report", str(eval_path)])
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        recon = report.get("summary", {}).get("execution_reconciliation", {})
        assert recon.get("match") is True

    def test_evaluation_reconciliation_mismatch_fails(self, tmp_path: Path):
        policy = _good_policy()
        policy["strategy"] = "smote"
        policy["resampling_applied_to"] = ["train"]
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        eval_path = _write_evaluation_report(tmp_path / "evaluation_report.json", "class_weight")
        result = self._run(tmp_path, setup, ["--evaluation-report", str(eval_path)])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "imbalance_execution_policy_mismatch" in codes

    def test_evaluation_reconciliation_missing_metadata_fails(self, tmp_path: Path):
        policy = _good_policy()
        setup = _make_setup(tmp_path, policy=policy, valid_labels=[0] * 10 + [1] * 10)
        _write_json(tmp_path / "evaluation_report.json", {"status": "pass", "metadata": {}})
        result = self._run(tmp_path, setup, ["--evaluation-report", str(tmp_path / "evaluation_report.json")])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "evaluation_report_imbalance_metadata_missing" in codes
