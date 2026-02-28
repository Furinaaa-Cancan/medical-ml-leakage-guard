"""Tests for examples/download_real_data.py and pre-generated example CSVs.

Validates that the existing example datasets conform to pipeline expectations.
Network download tests are marked @pytest.mark.network.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
DOWNLOAD_SCRIPT = EXAMPLES_DIR / "download_real_data.py"


class TestHeartDiseaseCSV:
    """Validate examples/heart_disease.csv."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = EXAMPLES_DIR / "heart_disease.csv"
        if not self.path.exists():
            pytest.skip("heart_disease.csv not present")
        self.df = pd.read_csv(self.path)

    def test_exists(self):
        assert self.path.exists()

    def test_required_columns(self):
        assert "patient_id" in self.df.columns
        assert "y" in self.df.columns
        assert "event_time" in self.df.columns

    def test_row_count(self):
        assert 250 <= len(self.df) <= 350

    def test_target_binary(self):
        assert set(self.df["y"].unique()).issubset({0, 1})

    def test_patient_id_unique(self):
        assert self.df["patient_id"].is_unique

    def test_patient_id_no_nan(self):
        assert self.df["patient_id"].notna().all()

    def test_event_time_parseable(self):
        times = pd.to_datetime(self.df["event_time"], errors="coerce")
        assert times.notna().all()

    def test_feature_count(self):
        features = [c for c in self.df.columns if c not in ("patient_id", "event_time", "y")]
        assert len(features) >= 10


class TestBreastCancerCSV:
    """Validate examples/breast_cancer.csv."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = EXAMPLES_DIR / "breast_cancer.csv"
        if not self.path.exists():
            pytest.skip("breast_cancer.csv not present")
        self.df = pd.read_csv(self.path)

    def test_exists(self):
        assert self.path.exists()

    def test_required_columns(self):
        assert "patient_id" in self.df.columns
        assert "y" in self.df.columns
        assert "event_time" in self.df.columns

    def test_row_count(self):
        assert 500 <= len(self.df) <= 600

    def test_target_binary(self):
        assert set(self.df["y"].unique()).issubset({0, 1})

    def test_patient_id_unique(self):
        assert self.df["patient_id"].is_unique

    def test_feature_count(self):
        features = [c for c in self.df.columns if c not in ("patient_id", "event_time", "y")]
        assert len(features) >= 25


class TestChronicKidneyCSV:
    """Validate examples/chronic_kidney_disease.csv."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.path = EXAMPLES_DIR / "chronic_kidney_disease.csv"
        if not self.path.exists():
            pytest.skip("chronic_kidney_disease.csv not present")
        self.df = pd.read_csv(self.path)

    def test_exists(self):
        assert self.path.exists()

    def test_required_columns(self):
        assert "patient_id" in self.df.columns
        assert "y" in self.df.columns
        assert "event_time" in self.df.columns

    def test_row_count(self):
        assert 300 <= len(self.df) <= 450

    def test_target_binary(self):
        assert set(self.df["y"].unique()).issubset({0, 1})

    def test_patient_id_unique(self):
        assert self.df["patient_id"].is_unique

    def test_feature_count(self):
        features = [c for c in self.df.columns if c not in ("patient_id", "event_time", "y")]
        assert len(features) >= 10


class TestDownloadScriptCLI:
    """Test CLI argument handling (no network)."""

    def test_invalid_dataset_name(self):
        result = subprocess.run(
            [sys.executable, str(DOWNLOAD_SCRIPT), "nonexistent"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0

    def test_heart_output_to_tmp(self, tmp_path: Path):
        """Generate heart data to custom output path (uses local raw if available)."""
        out = tmp_path / "heart_out.csv"
        result = subprocess.run(
            [sys.executable, str(DOWNLOAD_SCRIPT), "heart", "--output", str(out)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            assert out.exists()
            df = pd.read_csv(out)
            assert "patient_id" in df.columns
            assert "y" in df.columns
            assert set(df["y"].unique()).issubset({0, 1})
        else:
            # May fail if no local raw and no network — acceptable
            assert "ERROR" in result.stdout or "error" in result.stderr.lower()
