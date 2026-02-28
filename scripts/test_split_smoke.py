#!/usr/bin/env python3
"""
Smoke tests for split_data.py and schema_preflight.py single-file mode.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
SPLIT_SCRIPT = str(SCRIPTS_DIR / "split_data.py")
PREFLIGHT_SCRIPT = str(SCRIPTS_DIR / "schema_preflight.py")
PYTHON = sys.executable


def make_demo_csv(path: Path, n_rows: int = 300, seed: int = 42, with_time: bool = True, pos_rate: float = 0.3) -> pd.DataFrame:
    """Generate a small demo CSV for testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n_rows)],
        "age": rng.normal(60, 12, n_rows).round(1),
        "bmi": rng.normal(27, 4, n_rows).round(1),
        "wbc": rng.normal(7, 2, n_rows).round(2),
        "y": rng.binomial(1, pos_rate, n_rows),
    })
    if with_time:
        base = np.datetime64("2024-01-01")
        offsets = np.sort(rng.integers(0, 365, n_rows))
        df["event_time"] = (base + offsets.astype("timedelta64[D]")).astype(str)
    df.to_csv(path, index=False)
    return df


def run(cmd, expect_ok=True):
    """Run a command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok and proc.returncode != 0:
        print(f"FAIL: {' '.join(cmd[:4])}")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[-500:]}")
    return proc.returncode, proc.stdout, proc.stderr


def test_basic_temporal_split():
    """Test grouped_temporal split with normal data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)

        output_dir = Path(tmpdir) / "data"
        report_path = Path(tmpdir) / "report.json"

        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--time-col", "event_time",
            "--strategy", "grouped_temporal",
            "--report", str(report_path),
        ])
        assert rc == 0, f"temporal split failed: {stderr}"
        assert (output_dir / "train.csv").exists()
        assert (output_dir / "valid.csv").exists()
        assert (output_dir / "test.csv").exists()
        assert report_path.exists()

        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert report["strategy"] == "grouped_temporal"
        assert report.get("contract_version") == "split_report.v1", "report missing or wrong contract_version"
        assert "input_sha256" in report, "report missing SHA256 fingerprint"
        assert len(report["input_sha256"]) == 64, "SHA256 should be 64 hex chars"
        assert "input_rows_excluded" in report, "report missing excluded rows info"
        assert "input_rows_raw" in report, "report missing raw row count"
        assert "input_rows_clean" in report, "report missing clean row count"
        assert report["input_rows_excluded"]["total"] == 0, "no rows should be excluded in clean data"

        # Verify patient-level disjoint
        train_df = pd.read_csv(output_dir / "train.csv")
        valid_df = pd.read_csv(output_dir / "valid.csv")
        test_df = pd.read_csv(output_dir / "test.csv")

        train_ids = set(train_df["patient_id"])
        valid_ids = set(valid_df["patient_id"])
        test_ids = set(test_df["patient_id"])
        assert not (train_ids & valid_ids), "train/valid patient overlap"
        assert not (train_ids & test_ids), "train/test patient overlap"
        assert not (valid_ids & test_ids), "valid/test patient overlap"

        # Verify temporal order
        train_max = pd.to_datetime(train_df["event_time"]).max()
        valid_min = pd.to_datetime(valid_df["event_time"]).min()
        test_min = pd.to_datetime(test_df["event_time"]).min()
        assert train_max < valid_min, f"train max {train_max} >= valid min {valid_min}"

        # Verify total rows (row count preservation)
        assert len(train_df) + len(valid_df) + len(test_df) == 300

        # Verify split_protocol.json was generated
        protocol_path = Path(tmpdir) / "configs" / "split_protocol.json"
        assert protocol_path.exists(), "split_protocol.json not generated"
        protocol = json.loads(protocol_path.read_text())
        assert protocol["requires_group_disjoint"] is True
        assert protocol["frozen_before_modeling"] is True

    print("  PASS: test_basic_temporal_split")


def test_grouped_random_split():
    """Test grouped_random split (no time column)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300, with_time=False)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
        ])
        assert rc == 0, f"random split failed: {stderr}"

        train_df = pd.read_csv(output_dir / "train.csv")
        valid_df = pd.read_csv(output_dir / "valid.csv")
        test_df = pd.read_csv(output_dir / "test.csv")

        train_ids = set(train_df["patient_id"])
        valid_ids = set(valid_df["patient_id"])
        test_ids = set(test_df["patient_id"])
        assert not (train_ids & valid_ids), "patient overlap"
        assert not (train_ids & test_ids), "patient overlap"
        assert len(train_df) + len(valid_df) + len(test_df) == 300

    print("  PASS: test_grouped_random_split")


def test_stratified_grouped_split():
    """Test stratified_grouped split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "stratified_grouped",
        ])
        assert rc == 0, f"stratified split failed: {stderr}"

        train_df = pd.read_csv(output_dir / "train.csv")
        valid_df = pd.read_csv(output_dir / "valid.csv")
        test_df = pd.read_csv(output_dir / "test.csv")
        assert len(train_df) + len(valid_df) + len(test_df) == 300

        # Check positive rate is similar across splits
        train_rate = train_df["y"].mean()
        test_rate = test_df["y"].mean()
        assert abs(train_rate - test_rate) < 0.15, f"prevalence imbalance: train={train_rate:.3f} test={test_rate:.3f}"

    print("  PASS: test_stratified_grouped_split")


def test_too_few_patients():
    """Test that splitting fails with too few patients."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "tiny.csv"
        df = pd.DataFrame({
            "patient_id": ["P001", "P002", "P003"],
            "y": [0, 1, 0],
            "age": [50, 60, 70],
            "event_time": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--time-col", "event_time",
        ], expect_ok=False)
        assert rc != 0, "should fail with too few patients"

    print("  PASS: test_too_few_patients")


def test_missing_target_col():
    """Test that splitting fails when target column is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "no_target.csv"
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(20)],
            "age": list(range(20)),
        })
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
        ], expect_ok=False)
        assert rc != 0, "should fail with missing target col"

    print("  PASS: test_missing_target_col")


def test_non_binary_target():
    """Test that splitting fails with non-binary target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "multi_class.csv"
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(30)],
            "y": list(range(30)),  # Non-binary
            "age": list(range(30)),
        })
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
        ], expect_ok=False)
        assert rc != 0, "should fail with non-binary target"

    print("  PASS: test_non_binary_target")


def test_temporal_no_time_col():
    """Test that grouped_temporal fails without --time-col."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_temporal",
            # No --time-col
        ], expect_ok=False)
        assert rc != 0, "should fail without time col for temporal strategy"

    print("  PASS: test_temporal_no_time_col")


def test_bad_ratios():
    """Test that invalid ratios are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
            "--train-ratio", "0.8",
            "--valid-ratio", "0.5",
            "--test-ratio", "0.2",
        ], expect_ok=False)
        assert rc != 0, "should fail with ratios > 1.0"

    print("  PASS: test_bad_ratios")


def test_reproducible_seed():
    """Test that same seed produces same split."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)

        for run_id in ("a", "b"):
            output_dir = Path(tmpdir) / f"data_{run_id}"
            run([
                PYTHON, SPLIT_SCRIPT,
                "--input", str(csv_path),
                "--output-dir", str(output_dir),
                "--patient-id-col", "patient_id",
                "--target-col", "y",
                "--strategy", "grouped_random",
                "--seed", "12345",
            ])

        train_a = pd.read_csv(Path(tmpdir) / "data_a" / "train.csv")
        train_b = pd.read_csv(Path(tmpdir) / "data_b" / "train.csv")
        assert set(train_a["patient_id"]) == set(train_b["patient_id"]), "same seed should produce same split"

    print("  PASS: test_reproducible_seed")


def test_nan_patient_id_excluded():
    """Test that rows with NaN patient_id are excluded and split still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "with_nan_pid.csv"
        rng = np.random.default_rng(99)
        n = 300
        pids = [f"P{i:04d}" for i in range(n)]
        pids[0] = np.nan  # inject one NaN patient_id
        pids[1] = np.nan
        df = pd.DataFrame({
            "patient_id": pids,
            "y": rng.binomial(1, 0.3, n),
            "age": rng.normal(60, 12, n).round(1),
        })
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        report_path = Path(tmpdir) / "report.json"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
            "--report", str(report_path),
        ])
        assert rc == 0, f"should succeed after excluding NaN pids: {stderr}"
        report = json.loads(report_path.read_text())
        assert report["input_rows_excluded"]["nan_patient_id"] == 2

        train_df = pd.read_csv(output_dir / "train.csv")
        valid_df = pd.read_csv(output_dir / "valid.csv")
        test_df = pd.read_csv(output_dir / "test.csv")
        total = len(train_df) + len(valid_df) + len(test_df)
        assert total == n - 2, f"expected {n-2} rows, got {total}"

    print("  PASS: test_nan_patient_id_excluded")


def test_nan_target_excluded():
    """Test that rows with NaN target are excluded and split still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "with_nan_target.csv"
        rng = np.random.default_rng(88)
        n = 300
        targets = rng.binomial(1, 0.3, n).astype(float)
        targets[0] = np.nan
        targets[5] = np.nan
        targets[10] = np.nan
        df = pd.DataFrame({
            "patient_id": [f"P{i:04d}" for i in range(n)],
            "y": targets,
            "age": rng.normal(60, 12, n).round(1),
        })
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        report_path = Path(tmpdir) / "report.json"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
            "--report", str(report_path),
        ])
        assert rc == 0, f"should succeed after excluding NaN targets: {stderr}"
        report = json.loads(report_path.read_text())
        assert report["input_rows_excluded"]["nan_target_after_pid_clean"] == 3

    print("  PASS: test_nan_target_excluded")


def test_min_patients_per_split():
    """Test that splits with too few patients fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "few_patients.csv"
        # 10 patients, each with many rows -> test split gets ~2 patients < MIN_PATIENTS_PER_SPLIT=5
        rows = []
        for pid in range(10):
            for _ in range(10):
                rows.append({"patient_id": f"P{pid:04d}", "y": pid % 2, "age": 50 + pid})
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        output_dir = Path(tmpdir) / "data"
        rc, stdout, stderr = run([
            PYTHON, SPLIT_SCRIPT,
            "--input", str(csv_path),
            "--output-dir", str(output_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", "grouped_random",
        ], expect_ok=False)
        assert rc != 0, f"should fail with too few patients per split: {stderr}"

    print("  PASS: test_min_patients_per_split")


def test_preflight_single_file():
    """Test schema_preflight.py --input-csv mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "full.csv"
        make_demo_csv(csv_path, n_rows=300)
        report_path = Path(tmpdir) / "preflight_report.json"

        rc, stdout, stderr = run([
            PYTHON, PREFLIGHT_SCRIPT,
            "--input-csv", str(csv_path),
            "--target-col", "y",
            "--patient-id-col", "patient_id",
            "--time-col", "event_time",
            "--report", str(report_path),
        ])
        assert rc == 0, f"preflight single-file failed: {stderr}"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert report["summary"]["mode"] == "single_file"

    print("  PASS: test_preflight_single_file")


def test_preflight_missing_column():
    """Test schema_preflight.py detects missing columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "no_target.csv"
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(20)],
            "age": list(range(20)),
            "event_time": ["2024-01-01"] * 20,
        })
        df.to_csv(csv_path, index=False)
        report_path = Path(tmpdir) / "preflight_report.json"

        rc, stdout, stderr = run([
            PYTHON, PREFLIGHT_SCRIPT,
            "--input-csv", str(csv_path),
            "--target-col", "y",
            "--patient-id-col", "patient_id",
            "--report", str(report_path),
        ], expect_ok=False)
        assert rc != 0, "should fail with missing target column"
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"

    print("  PASS: test_preflight_missing_column")


def test_split_help():
    """Test that split_data.py --help works."""
    rc, stdout, stderr = run([PYTHON, SPLIT_SCRIPT, "--help"])
    assert rc == 0
    assert "medical safety" in stdout.lower() or "patient" in stdout.lower()
    print("  PASS: test_split_help")


def test_mlgg_split_help():
    """Test that mlgg.py split -- --help works."""
    mlgg_script = str(SCRIPTS_DIR / "mlgg.py")
    rc, stdout, stderr = run([PYTHON, mlgg_script, "split", "--", "--help"])
    assert rc == 0
    assert "split" in stdout.lower() or "patient" in stdout.lower()
    print("  PASS: test_mlgg_split_help")


def main() -> int:
    tests = [
        test_basic_temporal_split,
        test_grouped_random_split,
        test_stratified_grouped_split,
        test_too_few_patients,
        test_missing_target_col,
        test_non_binary_target,
        test_temporal_no_time_col,
        test_bad_ratios,
        test_reproducible_seed,
        test_nan_patient_id_excluded,
        test_nan_target_excluded,
        test_min_patients_per_split,
        test_preflight_single_file,
        test_preflight_missing_column,
        test_split_help,
        test_mlgg_split_help,
    ]

    passed = 0
    failed = 0
    errors = []

    print(f"\n=== Split Smoke Tests ({len(tests)} tests) ===\n")

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((test_fn.__name__, str(exc)))
            print(f"  FAIL: {test_fn.__name__}: {exc}")

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    if errors:
        for name, err in errors:
            print(f"  FAIL: {name}: {err}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
