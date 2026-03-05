"""Adversarial gate tests: inject protocol violations and verify gates detect them.

Each test creates a targeted injection scenario and verifies the corresponding
gate fails with the expected failure code. No full onboarded project required.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _make_split_csvs(tmp_path: Path):
    """Create minimal train/valid/test CSVs."""
    header = "patient_id,event_time,y,age,bp,wbc"
    train = [f"P{i},2024-01-{i+1:02d},{i%2},{20+i},{100+i},{5+i}" for i in range(20)]
    valid = [f"P{i},2024-03-{(i-20)+1:02d},{i%2},{20+i},{100+i},{5+i}" for i in range(20, 30)]
    test = [f"P{i},2024-05-{(i-30)+1:02d},{i%2},{20+i},{100+i},{5+i}" for i in range(30, 40)]
    data = tmp_path / "data"
    _make_csv(data / "train.csv", header, train)
    _make_csv(data / "valid.csv", header, valid)
    _make_csv(data / "test.csv", header, test)
    return data


class TestAdversarialDefinitionVariable:
    """Inject a training feature into defining_variables → expect detection."""

    def test_definition_variable_leakage_detected(self, tmp_path: Path):
        data = _make_split_csvs(tmp_path)
        spec = {
            "targets": {
                "disease_risk": {
                    "defining_variables": ["age"],  # "age" is a training feature!
                    "forbidden_patterns": [],
                }
            }
        }
        spec_path = _write_json(tmp_path / "phenotype.json", spec)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "definition_variable_guard.py"),
            "--target", "disease_risk",
            "--definition-spec", str(spec_path),
            "--train", str(data / "train.csv"),
            "--valid", str(data / "valid.csv"),
            "--test", str(data / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--strict",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2, f"Expected fail, got pass. stdout: {result.stdout}"
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_variable_leakage" in codes


class TestAdversarialTuningLeakage:
    """Set test_used_for_model_selection=True → expect detection."""

    def test_tuning_test_usage_detected(self, tmp_path: Path):
        spec = {
            "search_method": "grid",
            "preprocessing_fit_scope": "train_only",
            "feature_selection_scope": "train_only",
            "resampling_scope": "train_only",
            "early_stopping_data": "valid",
            "final_model_refit_scope": "train_only",
            "test_used_for_model_selection": True,  # VIOLATION
            "test_used_for_threshold_tuning": False,
            "outer_evaluation_locked": True,
            "random_seed_controlled": True,
            "random_seed": 42,
            "hyperparameter_trial_count": 10,
            "cv": {"enabled": True, "type": "stratified_k_fold", "n_splits": 5, "nested": False},
        }
        spec_path = _write_json(tmp_path / "tuning.json", spec)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "tuning_leakage_gate.py"),
            "--tuning-spec", str(spec_path),
            "--id-col", "patient_id",
            "--has-valid-split",
            "--strict",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "explicit_test_usage" in codes


class TestAdversarialMissingnessLeakage:
    """Set use_target_in_imputation=True → expect detection."""

    def test_target_in_imputation_detected(self, tmp_path: Path):
        data = _make_split_csvs(tmp_path)
        spec = {
            "strategy": "simple",
            "imputer_fit_scope": "train_only",
            "add_missing_indicators": False,
            "complete_case_analysis": False,
            "forbid_test_usage": True,
            "test_used_for_fit": False,
            "valid_used_for_fit": False,
            "use_target_in_imputation": True,  # VIOLATION
            "max_feature_missing_ratio": 0.5,
            "min_non_missing_per_feature": 2,
            "indicator_required_above_ratio": 0.9,
            "missingness_drift_tolerance": 0.3,
        }
        spec_path = _write_json(tmp_path / "miss.json", spec)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "missingness_policy_gate.py"),
            "--policy-spec", str(spec_path),
            "--train", str(data / "train.csv"),
            "--valid", str(data / "valid.csv"),
            "--test", str(data / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--strict",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "target_used_in_imputation" in codes


class TestAdversarialImbalanceLeakage:
    """Set threshold_selection_split=test → expect detection."""

    def test_postprocessing_on_test_detected(self, tmp_path: Path):
        data = _make_split_csvs(tmp_path)
        spec = {
            "strategy": "none",
            "resampling_scope": "train_only",
            "test_used_for_postprocessing": False,
            "valid_used_for_postprocessing": False,
            "threshold_selection_split": "test",  # VIOLATION
            "class_weight_strategy": "none",
        }
        spec_path = _write_json(tmp_path / "imb.json", spec)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "imbalance_policy_gate.py"),
            "--policy-spec", str(spec_path),
            "--train", str(data / "train.csv"),
            "--valid", str(data / "valid.csv"),
            "--test", str(data / "test.csv"),
            "--target-col", "y",
            "--strict",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "test_split_used_for_postprocessing" in codes


class TestAdversarialLeakageOverlap:
    """Inject patient overlap between train and test → expect detection."""

    def test_patient_overlap_detected(self, tmp_path: Path):
        header = "patient_id,event_time,y,age"
        # Deliberately put P1 in both train and test
        train = ["P1,2024-01-01,0,30", "P2,2024-01-02,1,40", "P3,2024-01-03,0,50"]
        valid = ["P4,2024-02-01,1,35", "P5,2024-02-02,0,45"]
        test = ["P1,2024-03-01,0,30", "P6,2024-03-02,1,55"]  # P1 overlap!
        data = tmp_path / "data"
        _make_csv(data / "train.csv", header, train)
        _make_csv(data / "valid.csv", header, valid)
        _make_csv(data / "test.csv", header, test)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(data / "train.csv"),
            "--valid", str(data / "valid.csv"),
            "--test", str(data / "test.csv"),
            "--target-col", "y",
            "--id-cols", "patient_id",
            "--time-col", "event_time",
            "--strict",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert any("overlap" in c.lower() or "entity" in c.lower() for c in codes), (
            f"Expected overlap/entity failure code, got: {codes}"
        )


class TestAdversarialEvaluationQuality:
    """Primary metric outside CI → expect detection."""

    def test_metric_outside_ci_detected(self, tmp_path: Path):
        eval_data = {
            "metrics": {"roc_auc": 0.99},  # Way outside CI
            "metrics_ci": {
                "roc_auc": {
                    "ci_95": [0.70, 0.80],
                    "method": "bootstrap",
                    "n_resamples": 1000,
                }
            },
            "baselines": {"random": {"metrics": {"roc_auc": 0.50}}},
        }
        eval_path = _write_json(tmp_path / "eval.json", eval_data)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "evaluation_quality_gate.py"),
            "--evaluation-report", str(eval_path),
            "--metric-name", "roc_auc",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "primary_metric_outside_ci" in codes


class TestAdversarialPermutationInsignificant:
    """Model metric barely above null → not significant."""

    def test_not_significant_detected(self, tmp_path: Path):
        null_values = [0.50 + i * 0.001 for i in range(200)]
        null_path = _write_json(tmp_path / "null.json", null_values)
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "permutation_significance_gate.py"),
            "--metric-name", "roc_auc",
            "--actual", "0.505",  # Barely above null mean
            "--null-metrics-file", str(null_path),
            "--alpha", "0.01",
            "--report", str(report_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "permutation_not_significant" in codes
