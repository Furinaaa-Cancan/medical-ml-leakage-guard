"""Tests for scripts/generate_demo_medical_dataset.py.

Covers sigmoid, random_dates, make_cohort, split_summary, save_csv,
CLI --help, full generation, reproducibility, and output validation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "generate_demo_medical_dataset.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import generate_demo_medical_dataset as gd


# ── helper functions ─────────────────────────────────────────────────────────

class TestSigmoid:
    def test_zero(self):
        assert abs(gd.sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6

    def test_large_positive(self):
        assert gd.sigmoid(np.array([100.0]))[0] > 0.999

    def test_large_negative(self):
        assert gd.sigmoid(np.array([-100.0]))[0] < 0.001

    def test_clip(self):
        result = gd.sigmoid(np.array([1000.0, -1000.0]))
        assert np.all(np.isfinite(result))


class TestRandomDates:
    def test_range(self):
        rng = np.random.default_rng(42)
        dates = gd.random_dates(rng, "2024-01-01", "2024-12-31", 100)
        assert len(dates) == 100
        assert all(d >= np.datetime64("2024-01-01") for d in dates)
        assert all(d <= np.datetime64("2024-12-31") for d in dates)

    def test_single_day(self):
        rng = np.random.default_rng(42)
        dates = gd.random_dates(rng, "2024-06-15", "2024-06-15", 5)
        assert len(dates) == 5
        assert all(d == np.datetime64("2024-06-15") for d in dates)


class TestMakeCohort:
    def test_basic(self):
        rng = np.random.default_rng(42)
        df = gd.make_cohort(rng, n_rows=50, patient_id_start=1000,
                            date_start="2024-01-01", date_end="2024-06-30", shifts={})
        assert df.shape[0] == 50
        assert "patient_id" in df.columns
        assert "event_time" in df.columns
        assert "y" in df.columns
        assert set(df["y"].unique()).issubset({0, 1})

    def test_columns(self):
        rng = np.random.default_rng(42)
        df = gd.make_cohort(rng, n_rows=20, patient_id_start=0,
                            date_start="2024-01-01", date_end="2024-03-31", shifts={})
        expected = {"patient_id", "event_time", "y", "age", "sex_male", "bmi",
                    "systolic_bp", "heart_rate", "wbc", "creatinine", "lactate",
                    "crp", "comorbidity_index", "smoke_status"}
        assert set(df.columns) == expected

    def test_shifts(self):
        rng1 = np.random.default_rng(99)
        df1 = gd.make_cohort(rng1, n_rows=500, patient_id_start=0,
                             date_start="2024-01-01", date_end="2024-06-30", shifts={})
        rng2 = np.random.default_rng(99)
        df2 = gd.make_cohort(rng2, n_rows=500, patient_id_start=0,
                             date_start="2024-01-01", date_end="2024-06-30",
                             shifts={"age": 10.0})
        assert df2["age"].mean() > df1["age"].mean()


class TestSplitSummary:
    def test_basic(self):
        df = pd.DataFrame({
            "y": [0, 0, 1, 1, 1],
            "event_time": ["2024-01-01"] * 5,
        })
        s = gd.split_summary(df)
        assert s["rows"] == 5
        assert s["positive_count"] == 3
        assert s["negative_count"] == 2
        assert abs(s["positive_rate"] - 0.6) < 1e-6


class TestSaveCsv:
    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = tmp_path / "sub" / "out.csv"
        gd.save_csv(df, p)
        assert p.exists()
        loaded = pd.read_csv(p)
        assert loaded.shape == (2, 2)


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "project-root" in proc.stdout


class TestCLIGenerate:
    def test_full_generation(self, tmp_path):
        project = tmp_path / "demo"
        report = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--seed", "42",
            "--report", str(report),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        for name in ("train.csv", "valid.csv", "test.csv", "external_2025_q4.csv", "external_site_b.csv"):
            assert (project / "data" / name).exists()
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["status"] == "pass"
        assert data["seed"] == 42

    def test_reproducibility(self, tmp_path):
        p1 = tmp_path / "run1"
        p2 = tmp_path / "run2"
        for p in (p1, p2):
            cmd = [
                sys.executable, str(GATE_SCRIPT),
                "--project-root", str(p),
                "--seed", "12345",
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        df1 = pd.read_csv(p1 / "data" / "train.csv")
        df2 = pd.read_csv(p2 / "data" / "train.csv")
        pd.testing.assert_frame_equal(df1, df2)

    def test_row_counts(self, tmp_path):
        project = tmp_path / "counts"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--seed", "99",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        train = pd.read_csv(project / "data" / "train.csv")
        valid = pd.read_csv(project / "data" / "valid.csv")
        test = pd.read_csv(project / "data" / "test.csv")
        assert train.shape[0] == 840
        assert valid.shape[0] == 420
        assert test.shape[0] == 320

    def test_target_binary(self, tmp_path):
        project = tmp_path / "binary"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--seed", "77",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        for name in ("train.csv", "valid.csv", "test.csv"):
            df = pd.read_csv(project / "data" / name)
            assert set(df["y"].unique()).issubset({0, 1})
