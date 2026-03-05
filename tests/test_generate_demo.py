"""Unit tests for scripts/generate_demo_medical_dataset.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_demo_medical_dataset import (
    ensure_parent,
    make_cohort,
    random_dates,
    save_csv,
    sigmoid,
    split_summary,
)


# ────────────────────────────────────────────────────────
# sigmoid
# ────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_large_positive(self):
        assert sigmoid(np.array([100.0]))[0] > 0.999

    def test_large_negative(self):
        assert sigmoid(np.array([-100.0]))[0] < 0.001

    def test_no_overflow(self):
        result = sigmoid(np.array([-1000.0, 1000.0]))
        assert np.all(np.isfinite(result))


# ────────────────────────────────────────────────────────
# random_dates
# ────────────────────────────────────────────────────────

class TestRandomDates:
    def test_length(self):
        rng = np.random.default_rng(42)
        dates = random_dates(rng, "2024-01-01", "2024-12-31", 50)
        assert len(dates) == 50

    def test_within_range(self):
        rng = np.random.default_rng(42)
        dates = random_dates(rng, "2024-06-01", "2024-06-30", 100)
        start = np.datetime64("2024-06-01")
        end = np.datetime64("2024-06-30")
        assert np.all(dates >= start)
        assert np.all(dates <= end)

    def test_same_day(self):
        rng = np.random.default_rng(42)
        dates = random_dates(rng, "2024-01-01", "2024-01-01", 5)
        assert len(dates) == 5


# ────────────────────────────────────────────────────────
# make_cohort
# ────────────────────────────────────────────────────────

class TestMakeCohort:
    def test_shape_and_columns(self):
        rng = np.random.default_rng(1)
        df = make_cohort(rng, 100, 1000, "2024-01-01", "2024-06-30", {})
        assert df.shape[0] == 100
        expected_cols = {
            "patient_id", "event_time", "y", "age", "sex_male",
            "bmi", "systolic_bp", "heart_rate", "wbc", "creatinine",
            "lactate", "crp", "comorbidity_index", "smoke_status",
        }
        assert set(df.columns) == expected_cols

    def test_binary_target(self):
        rng = np.random.default_rng(2)
        df = make_cohort(rng, 200, 5000, "2024-01-01", "2024-03-31", {})
        assert set(df["y"].unique()).issubset({0, 1})

    def test_patient_ids_unique(self):
        rng = np.random.default_rng(3)
        df = make_cohort(rng, 50, 10000, "2024-01-01", "2024-06-30", {})
        assert df["patient_id"].nunique() == 50

    def test_deterministic(self):
        df1 = make_cohort(np.random.default_rng(99), 30, 0, "2024-01-01", "2024-03-31", {})
        df2 = make_cohort(np.random.default_rng(99), 30, 0, "2024-01-01", "2024-03-31", {})
        pd.testing.assert_frame_equal(df1, df2)

    def test_shifts_alter_distribution(self):
        rng1 = np.random.default_rng(10)
        df_no_shift = make_cohort(rng1, 500, 0, "2024-01-01", "2024-06-30", {})
        rng2 = np.random.default_rng(10)
        df_shifted = make_cohort(rng2, 500, 0, "2024-01-01", "2024-06-30", {"age": 5.0})
        assert df_shifted["age"].mean() > df_no_shift["age"].mean()


# ────────────────────────────────────────────────────────
# split_summary
# ────────────────────────────────────────────────────────

class TestSplitSummary:
    def test_keys(self):
        rng = np.random.default_rng(7)
        df = make_cohort(rng, 50, 0, "2024-01-01", "2024-03-31", {})
        s = split_summary(df)
        assert set(s.keys()) == {"rows", "positive_count", "negative_count", "positive_rate", "min_date", "max_date"}

    def test_counts_add_up(self):
        rng = np.random.default_rng(8)
        df = make_cohort(rng, 100, 0, "2024-01-01", "2024-06-30", {})
        s = split_summary(df)
        assert s["positive_count"] + s["negative_count"] == s["rows"]
        assert 0.0 <= s["positive_rate"] <= 1.0


# ────────────────────────────────────────────────────────
# save_csv / ensure_parent
# ────────────────────────────────────────────────────────

class TestSaveCsv:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = tmp_path / "sub" / "data.csv"
        save_csv(df, out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert loaded.shape == (2, 2)

    def test_ensure_parent_nested(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "file.txt"
        ensure_parent(nested)
        assert nested.parent.exists()


# ────────────────────────────────────────────────────────
# main (integration)
# ────────────────────────────────────────────────────────

class TestMain:
    def test_generates_all_files(self, tmp_path, monkeypatch):
        from generate_demo_medical_dataset import main
        monkeypatch.setattr(
            "sys.argv",
            ["gen", "--project-root", str(tmp_path), "--seed", "42"],
        )
        rc = main()
        assert rc == 0
        data_dir = tmp_path / "data"
        assert (data_dir / "train.csv").exists()
        assert (data_dir / "valid.csv").exists()
        assert (data_dir / "test.csv").exists()
        assert (data_dir / "external_2025_q4.csv").exists()
        assert (data_dir / "external_site_b.csv").exists()

        train = pd.read_csv(data_dir / "train.csv")
        assert train.shape[0] == 840

    def test_report_generated(self, tmp_path, monkeypatch):
        from generate_demo_medical_dataset import main
        report_path = tmp_path / "report.json"
        monkeypatch.setattr(
            "sys.argv",
            ["gen", "--project-root", str(tmp_path), "--report", str(report_path)],
        )
        rc = main()
        assert rc == 0
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert "summary" in report
        assert "train" in report["summary"]
