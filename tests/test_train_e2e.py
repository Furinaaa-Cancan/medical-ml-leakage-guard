"""E2E integration tests for scripts/train_select_evaluate.py.

Uses pre-split heart_disease.csv data to run a minimal training pipeline
and validate all output artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
HEART_CSV = EXAMPLES_DIR / "heart_disease.csv"


def _split_data(tmp_path: Path) -> Path:
    """Pre-split heart data into train/valid/test under tmp_path/data."""
    out_dir = tmp_path / "data"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "split_data.py"),
        "--input", str(HEART_CSV),
        "--output-dir", str(out_dir),
        "--patient-id-col", "patient_id",
        "--target-col", "y",
        "--time-col", "event_time",
        "--strategy", "grouped_temporal",
        "--seed", "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Split failed: {result.stderr[-2000:]}"
    return out_dir


def _run_train(tmp_path: Path, data_dir: Path, extra_args: list = None,
               timeout: int = 300) -> subprocess.CompletedProcess:
    evidence = tmp_path / "evidence"
    models = tmp_path / "models"
    evidence.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "train_select_evaluate.py"),
        "--train", str(data_dir / "train.csv"),
        "--valid", str(data_dir / "valid.csv"),
        "--test", str(data_dir / "test.csv"),
        "--target-col", "y",
        "--patient-id-col", "patient_id",
        "--ignore-cols", "patient_id,event_time",
        "--model-selection-report-out", str(evidence / "model_selection_report.json"),
        "--evaluation-report-out", str(evidence / "evaluation_report.json"),
        "--prediction-trace-out", str(evidence / "prediction_trace.csv.gz"),
        "--model-out", str(models / "model.joblib"),
        "--n-jobs", "1",
        "--selection-data", "valid",
        "--bootstrap-resamples", "50",
        "--ci-bootstrap-resamples", "50",
        "--permutation-resamples", "30",
        "--permutation-null-out", str(evidence / "permutation_null.txt"),
        "--ci-matrix-report-out", str(evidence / "ci_matrix_report.json"),
        "--seed-sensitivity-out", str(evidence / "seed_sensitivity_report.json"),
        "--seed-sensitivity-seeds", "42,43",
        "--robustness-report-out", str(evidence / "robustness_report.json"),
        "--robustness-time-slices", "2",
        "--robustness-group-count", "2",
        "--distribution-report-out", str(evidence / "distribution_report.json"),
        "--feature-engineering-report-out", str(evidence / "feature_engineering_report.json"),
        "--calibration-method", "power",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.mark.slow
class TestTrainE2E:
    @pytest.fixture(autouse=True, scope="class")
    def train_result(self, tmp_path_factory):
        """Run training once for all tests in this class."""
        tmp_path = tmp_path_factory.mktemp("train_e2e")
        data_dir = _split_data(tmp_path)
        result = _run_train(tmp_path, data_dir)
        # Store paths for use in tests
        self.__class__._tmp_path = tmp_path
        self.__class__._result = result
        self.__class__._evidence = tmp_path / "evidence"
        self.__class__._models = tmp_path / "models"

    def test_exit_code_zero(self):
        assert self._result.returncode == 0, (
            f"Train failed with exit code {self._result.returncode}\n"
            f"stdout: {self._result.stdout[-3000:]}\n"
            f"stderr: {self._result.stderr[-3000:]}"
        )

    def test_model_selection_report_exists(self):
        path = self._evidence / "model_selection_report.json"
        assert path.exists()
        report = json.loads(path.read_text())
        assert "candidates" in report or "selected_model" in report

    def test_evaluation_report_exists(self):
        path = self._evidence / "evaluation_report.json"
        assert path.exists()
        report = json.loads(path.read_text())
        assert "metrics" in report

    def test_evaluation_report_has_split_test(self):
        path = self._evidence / "evaluation_report.json"
        report = json.loads(path.read_text())
        # The evaluation should be on the test split
        split_val = report.get("split") or report.get("evaluation_split")
        if split_val:
            assert str(split_val).lower() == "test"

    def test_evaluation_report_has_metrics(self):
        path = self._evidence / "evaluation_report.json"
        report = json.loads(path.read_text())
        metrics = report.get("metrics", {})
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_prediction_trace_exists(self):
        assert (self._evidence / "prediction_trace.csv.gz").exists()

    def test_model_artifact_exists(self):
        assert (self._models / "model.joblib").exists()

    def test_permutation_null_exists(self):
        path = self._evidence / "permutation_null.txt"
        assert path.exists()
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1

    def test_ci_matrix_report_exists(self):
        path = self._evidence / "ci_matrix_report.json"
        assert path.exists()
        report = json.loads(path.read_text())
        assert isinstance(report, dict)

    def test_robustness_report_exists(self):
        path = self._evidence / "robustness_report.json"
        assert path.exists()

    def test_distribution_report_exists(self):
        path = self._evidence / "distribution_report.json"
        assert path.exists()

    def test_seed_sensitivity_report_exists(self):
        path = self._evidence / "seed_sensitivity_report.json"
        assert path.exists()

    def test_feature_engineering_report_exists(self):
        path = self._evidence / "feature_engineering_report.json"
        assert path.exists()


class TestTrainMinimal:
    """A quick non-slow test with minimal config to verify CLI starts."""

    def test_missing_train_file(self, tmp_path: Path):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_select_evaluate.py"),
            "--train", str(tmp_path / "nonexistent.csv"),
            "--valid", str(tmp_path / "nonexistent.csv"),
            "--test", str(tmp_path / "nonexistent.csv"),
            "--model-selection-report-out", str(tmp_path / "msr.json"),
            "--evaluation-report-out", str(tmp_path / "er.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
