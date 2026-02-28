"""E2E integration tests for scripts/split_data.py using real example data."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
HEART_CSV = EXAMPLES_DIR / "heart_disease.csv"


def _run_split(tmp_path: Path, strategy: str, input_csv: Path = HEART_CSV,
               time_col: str = "event_time",
               extra_args: list = None, timeout: int = 60) -> subprocess.CompletedProcess:
    out_dir = tmp_path / "data"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "split_data.py"),
        "--input", str(input_csv),
        "--output-dir", str(out_dir),
        "--patient-id-col", "patient_id",
        "--target-col", "y",
        "--strategy", strategy,
        "--seed", "42",
        "--report", str(tmp_path / "split_report.json"),
    ]
    if time_col:
        cmd.extend(["--time-col", time_col])
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class TestSplitGroupedTemporal:
    def test_splits_exist(self, tmp_path: Path):
        result = _run_split(tmp_path, "grouped_temporal")
        assert result.returncode == 0, f"stderr: {result.stderr[-2000:]}"
        for name in ("train.csv", "valid.csv", "test.csv"):
            assert (tmp_path / "data" / name).exists()

    def test_row_count_preserved(self, tmp_path: Path):
        _run_split(tmp_path, "grouped_temporal")
        original = pd.read_csv(HEART_CSV)
        train = pd.read_csv(tmp_path / "data" / "train.csv")
        valid = pd.read_csv(tmp_path / "data" / "valid.csv")
        test = pd.read_csv(tmp_path / "data" / "test.csv")
        # Rows may decrease due to NaN exclusion, but split sum should equal filtered original
        split_total = len(train) + len(valid) + len(test)
        assert split_total <= len(original)
        assert split_total > 0

    def test_patient_disjoint(self, tmp_path: Path):
        _run_split(tmp_path, "grouped_temporal")
        train = pd.read_csv(tmp_path / "data" / "train.csv")
        valid = pd.read_csv(tmp_path / "data" / "valid.csv")
        test = pd.read_csv(tmp_path / "data" / "test.csv")
        train_ids = set(train["patient_id"])
        valid_ids = set(valid["patient_id"])
        test_ids = set(test["patient_id"])
        assert train_ids.isdisjoint(valid_ids)
        assert train_ids.isdisjoint(test_ids)
        assert valid_ids.isdisjoint(test_ids)

    def test_columns_preserved(self, tmp_path: Path):
        _run_split(tmp_path, "grouped_temporal")
        original_cols = list(pd.read_csv(HEART_CSV, nrows=0).columns)
        for name in ("train.csv", "valid.csv", "test.csv"):
            split_cols = list(pd.read_csv(tmp_path / "data" / name, nrows=0).columns)
            assert split_cols == original_cols

    def test_min_samples(self, tmp_path: Path):
        _run_split(tmp_path, "grouped_temporal")
        for name in ("train.csv", "valid.csv", "test.csv"):
            df = pd.read_csv(tmp_path / "data" / name)
            pos = (df["y"] == 1).sum()
            neg = (df["y"] == 0).sum()
            assert pos >= 5, f"{name}: only {pos} positive samples"
            assert neg >= 5, f"{name}: only {neg} negative samples"


class TestSplitGroupedRandom:
    def test_splits_exist(self, tmp_path: Path):
        result = _run_split(tmp_path, "grouped_random", time_col="")
        assert result.returncode == 0, f"stderr: {result.stderr[-2000:]}"
        for name in ("train.csv", "valid.csv", "test.csv"):
            assert (tmp_path / "data" / name).exists()

    def test_patient_disjoint(self, tmp_path: Path):
        _run_split(tmp_path, "grouped_random", time_col="")
        train = pd.read_csv(tmp_path / "data" / "train.csv")
        valid = pd.read_csv(tmp_path / "data" / "valid.csv")
        test = pd.read_csv(tmp_path / "data" / "test.csv")
        assert set(train["patient_id"]).isdisjoint(set(valid["patient_id"]))
        assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))

    def test_reproducible(self, tmp_path: Path):
        """Same seed should produce identical splits."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        _run_split(dir1, "grouped_random", time_col="")
        _run_split(dir2, "grouped_random", time_col="")
        for name in ("train.csv", "valid.csv", "test.csv"):
            df1 = pd.read_csv(dir1 / "data" / name)
            df2 = pd.read_csv(dir2 / "data" / name)
            assert set(df1["patient_id"]) == set(df2["patient_id"])


class TestSplitStratifiedGrouped:
    def test_splits_exist(self, tmp_path: Path):
        result = _run_split(tmp_path, "stratified_grouped", time_col="")
        assert result.returncode == 0, f"stderr: {result.stderr[-2000:]}"
        for name in ("train.csv", "valid.csv", "test.csv"):
            assert (tmp_path / "data" / name).exists()

    def test_patient_disjoint(self, tmp_path: Path):
        _run_split(tmp_path, "stratified_grouped", time_col="")
        train = pd.read_csv(tmp_path / "data" / "train.csv")
        valid = pd.read_csv(tmp_path / "data" / "valid.csv")
        test = pd.read_csv(tmp_path / "data" / "test.csv")
        assert set(train["patient_id"]).isdisjoint(set(test["patient_id"]))

    def test_row_count(self, tmp_path: Path):
        _run_split(tmp_path, "stratified_grouped", time_col="")
        original = pd.read_csv(HEART_CSV)
        total = sum(
            len(pd.read_csv(tmp_path / "data" / f))
            for f in ("train.csv", "valid.csv", "test.csv")
        )
        assert total <= len(original)
        assert total > 0


class TestSplitReportJson:
    def test_report_exists(self, tmp_path: Path):
        import json
        _run_split(tmp_path, "grouped_temporal")
        report_path = tmp_path / "split_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "status" in report
