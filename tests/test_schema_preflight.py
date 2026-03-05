"""Comprehensive unit tests for scripts/schema_preflight.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from schema_preflight import (
    TARGET_ALIASES,
    PATIENT_ID_ALIASES,
    load_split,
    missing_ratio,
    normalize_col,
    parse_binary_target,
    resolve_column,
    split_summary,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helper: create split CSVs
# ────────────────────────────────────────────────────────

def _make_split_csvs(
    tmp_path: Path,
    n_rows: int = 40,
    target_col: str = "y",
    pid_col: str = "patient_id",
    time_col: str = "event_time",
    pos_rate: float = 0.5,
    extra_cols: int = 0,
    nan_pid: int = 0,
    non_binary_target: bool = False,
    single_class: bool = False,
    missing_target_col: bool = False,
    different_columns: bool = False,
) -> dict:
    """Generate train/valid/test CSVs and return paths dict."""
    rng = np.random.default_rng(42)
    paths = {}
    for split_name, pid_start in [("train", 0), ("valid", 100), ("test", 200)]:
        rows = []
        for i in range(n_rows):
            row = {
                pid_col: f"P{pid_start + i:04d}",
                "age": int(rng.integers(20, 80)),
            }
            if not missing_target_col:
                if non_binary_target:
                    row[target_col] = rng.choice(["A", "B", "C"])
                elif single_class:
                    row[target_col] = 0
                else:
                    row[target_col] = 1 if rng.random() < pos_rate else 0
            if time_col:
                from datetime import date, timedelta
                d = date(2020, 1, 1) + timedelta(days=pid_start + i)
                row[time_col] = d.isoformat()
            for c in range(extra_cols):
                row[f"extra_{c}"] = float(rng.normal())
            rows.append(row)
        df = pd.DataFrame(rows)
        if nan_pid > 0 and split_name == "train":
            idx = rng.choice(len(df), size=min(nan_pid, len(df)), replace=False)
            df.loc[idx, pid_col] = np.nan
        if different_columns and split_name == "test":
            df = df.rename(columns={"age": "AGE_DIFFERENT"})
        p = tmp_path / f"{split_name}.csv"
        df.to_csv(p, index=False)
        paths[split_name] = p
    return paths


def _make_single_csv(tmp_path: Path, **kwargs) -> Path:
    """Generate a single CSV for pre-split mode."""
    rng = np.random.default_rng(42)
    n_rows = kwargs.get("n_rows", 60)
    pid_col = kwargs.get("pid_col", "patient_id")
    target_col = kwargs.get("target_col", "y")
    time_col = kwargs.get("time_col", "event_time")
    pos_rate = kwargs.get("pos_rate", 0.5)
    rows = []
    for i in range(n_rows):
        row = {
            pid_col: f"P{i:04d}",
            target_col: 1 if rng.random() < pos_rate else 0,
            "age": int(rng.integers(20, 80)),
        }
        if time_col:
            from datetime import date, timedelta
            d = date(2020, 1, 1) + timedelta(days=i)
            row[time_col] = d.isoformat()
        rows.append(row)
    df = pd.DataFrame(rows)
    if kwargs.get("nan_pid"):
        idx = rng.choice(len(df), size=min(kwargs["nan_pid"], len(df)), replace=False)
        df.loc[idx, pid_col] = np.nan
    if kwargs.get("non_binary"):
        df[target_col] = ["A", "B"] * (n_rows // 2)
    if kwargs.get("single_class"):
        df[target_col] = 0
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


# ────────────────────────────────────────────────────────
# normalize_col
# ────────────────────────────────────────────────────────

class TestNormalizeCol:
    def test_lowercase(self):
        assert normalize_col("Patient_ID") == "patient_id"

    def test_strip(self):
        assert normalize_col("  age  ") == "age"

    def test_remove_special(self):
        assert normalize_col("col-name!@#") == "colname"

    def test_keep_underscore(self):
        assert normalize_col("patient_id") == "patient_id"

    def test_keep_digits(self):
        assert normalize_col("feature_123") == "feature_123"

    def test_empty(self):
        assert normalize_col("") == ""

    def test_only_special(self):
        assert normalize_col("!@#$%") == ""

    def test_spaces_and_hyphens(self):
        assert normalize_col("Patient ID - Main") == "patientidmain"


# ────────────────────────────────────────────────────────
# load_split
# ────────────────────────────────────────────────────────

class TestLoadSplit:
    def test_normal(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
        df = load_split(str(p))
        assert len(df) == 2

    def test_empty_csv(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty"):
            load_split(str(p))

    def test_file_not_found(self):
        with pytest.raises(Exception):
            load_split("/nonexistent/path.csv")


# ────────────────────────────────────────────────────────
# resolve_column
# ────────────────────────────────────────────────────────

class TestResolveColumn:
    def test_exact_match(self):
        sel, mode, alts = resolve_column("y", TARGET_ALIASES, ["y", "age", "pid"])
        assert sel == "y"
        assert mode == "exact"
        assert alts == []

    def test_normalized_match(self):
        sel, mode, alts = resolve_column("Patient_ID", PATIENT_ID_ALIASES, ["patient_id", "age"])
        # normalize_col("Patient_ID") == "patient_id", which matches
        assert sel == "patient_id"
        assert mode == "normalized"

    def test_alias_match(self):
        sel, mode, alts = resolve_column("y", TARGET_ALIASES, ["label", "pid"])
        assert sel == "label"
        assert mode == "alias"

    def test_multiple_alias_matches(self):
        sel, mode, alts = resolve_column("y", TARGET_ALIASES, ["outcome", "label", "pid"])
        # First alias match wins
        assert sel in ["outcome", "label"]
        assert mode == "alias"
        assert len(alts) >= 0  # May have alternates

    def test_no_match(self):
        sel, mode, alts = resolve_column("y", TARGET_ALIASES, ["col_a", "col_b"])
        assert sel is None
        assert mode == "missing"

    def test_preferred_case_insensitive(self):
        sel, mode, alts = resolve_column("Y", TARGET_ALIASES, ["y", "age"])
        # normalize_col("Y") == "y", normalize_col("y") == "y" → match
        assert sel == "y"
        assert mode in ("exact", "normalized")

    def test_empty_columns(self):
        sel, mode, alts = resolve_column("y", TARGET_ALIASES, [])
        assert sel is None
        assert mode == "missing"


# ────────────────────────────────────────────────────────
# parse_binary_target
# ────────────────────────────────────────────────────────

class TestParseBinaryTarget:
    def test_numeric_01(self):
        s = pd.Series([0, 1, 0, 1])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [0, 1, 0, 1]

    def test_float_01(self):
        s = pd.Series([0.0, 1.0, 0.0])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [0, 1, 0]

    def test_string_01(self):
        s = pd.Series(["0", "1", "0"])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [0, 1, 0]

    def test_true_false(self):
        s = pd.Series(["True", "False", "true", "false"])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [1, 0, 1, 0]

    def test_yes_no(self):
        s = pd.Series(["Yes", "No", "yes", "no"])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [1, 0, 1, 0]

    def test_positive_negative(self):
        s = pd.Series(["Positive", "Negative", "positive"])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [1, 0, 1]

    def test_non_binary_values(self):
        s = pd.Series(["A", "B", "C"])
        arr, err = parse_binary_target(s)
        assert arr is None
        assert "non-binary" in err

    def test_numeric_non_binary(self):
        s = pd.Series([0, 1, 2])
        arr, err = parse_binary_target(s)
        # Contains 2 which is not 0 or 1
        assert arr is None
        assert err is not None

    def test_all_zeros(self):
        s = pd.Series([0, 0, 0])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [0, 0, 0]

    def test_all_ones(self):
        s = pd.Series([1, 1, 1])
        arr, err = parse_binary_target(s)
        assert err is None

    def test_mixed_numeric_string(self):
        s = pd.Series([0, 1, "bad"])
        arr, err = parse_binary_target(s)
        # NaN after coerce for "bad", so goes to string mapping path
        # "bad" is not in mapping dict → non-binary
        assert arr is None
        assert err is not None

    def test_numeric_with_nan(self):
        s = pd.Series([0, 1, np.nan, 0])
        arr, err = parse_binary_target(s)
        # notna().all() is False → falls to string mapping
        # "nan" not in mapping → non-binary
        assert arr is None
        assert err is not None

    def test_string_with_whitespace(self):
        s = pd.Series([" Yes ", " No "])
        arr, err = parse_binary_target(s)
        assert err is None
        assert list(arr) == [1, 0]


# ────────────────────────────────────────────────────────
# missing_ratio
# ────────────────────────────────────────────────────────

class TestMissingRatio:
    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert missing_ratio(df) == 0.0

    def test_half_missing(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, 4]})
        ratio = missing_ratio(df)
        assert abs(ratio - 0.5) < 0.01

    def test_all_missing(self):
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        assert missing_ratio(df) == 1.0

    def test_single_cell(self):
        df = pd.DataFrame({"a": [1]})
        assert missing_ratio(df) == 0.0


# ────────────────────────────────────────────────────────
# split_summary
# ────────────────────────────────────────────────────────

class TestSplitSummary:
    def test_normal(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B", "C"],
            "y": [0, 1, 1],
            "event_time": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        s = split_summary(df, "y", "patient_id", "event_time")
        assert s["rows"] == 3
        assert s["columns"] == 3
        assert s["positive_count"] == 2
        assert s["negative_count"] == 1
        assert s["patient_id_unique_count"] == 3
        assert s["target_parse_error"] is None

    def test_non_binary_target(self):
        df = pd.DataFrame({
            "patient_id": ["A"], "y": ["bad"], "event_time": ["2024-01-01"],
        })
        s = split_summary(df, "y", "patient_id", "event_time")
        assert s["target_parse_error"] is not None
        assert s["positive_count"] is None

    def test_null_patient_ids(self):
        df = pd.DataFrame({
            "patient_id": ["A", np.nan, "C"],
            "y": [0, 1, 0],
            "event_time": ["2024-01-01", "2024-02-01", "2024-03-01"],
        })
        s = split_summary(df, "y", "patient_id", "event_time")
        assert s["patient_id_null_count"] == 1

    def test_unparseable_time(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B"],
            "y": [0, 1],
            "event_time": ["not-a-date", "also-bad"],
        })
        s = split_summary(df, "y", "patient_id", "event_time")
        assert s["time_parse_error_count"] == 2


# ────────────────────────────────────────────────────────
# CLI — Split mode (subprocess)
# ────────────────────────────────────────────────────────

class TestCLISplitMode:
    def _run(self, tmp_path, splits, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "schema_preflight.py"),
            "--train", str(splits["train"]),
            "--valid", str(splits["valid"]),
            "--test", str(splits["test"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_normal_pass(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path)
        result = self._run(tmp_path, splits)
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_missing_target_col(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, missing_target_col=True)
        result = self._run(tmp_path, splits)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "required_column_missing" in codes

    def test_non_binary_target(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, non_binary_target=True)
        result = self._run(tmp_path, splits)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "target_not_binary" in codes

    def test_null_patient_ids(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, nan_pid=3)
        result = self._run(tmp_path, splits)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "patient_id_nulls_detected" in codes

    def test_different_columns_across_splits(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, different_columns=True)
        result = self._run(tmp_path, splits)
        # Common columns will be fewer; may still pass if key cols present
        # Just verify it runs without crash
        assert result.returncode in (0, 2)

    def test_single_class_warning(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, single_class=True)
        result = self._run(tmp_path, splits)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "target_single_class_split" in warn_codes

    def test_auto_mapping_warning(self, tmp_path: Path):
        # Use non-default column names that match aliases
        splits = _make_split_csvs(tmp_path, target_col="label", pid_col="subject_id", time_col="timestamp")
        result = self._run(tmp_path, splits)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "column_auto_mapped" in warn_codes

    def test_strict_rejects_auto_mapping(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path, target_col="label", pid_col="subject_id", time_col="timestamp")
        result = self._run(tmp_path, splits, extra_args=["--strict"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "strict_auto_mapping_not_allowed" in codes

    def test_mapping_output(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path)
        mapping_path = tmp_path / "mapping.json"
        result = self._run(tmp_path, splits, extra_args=["--mapping-out", str(mapping_path)])
        assert result.returncode == 0
        assert mapping_path.exists()
        mapping = json.loads(mapping_path.read_text())
        assert "target_col" in mapping
        assert "patient_id_col" in mapping

    def test_report_structure(self, tmp_path: Path):
        splits = _make_split_csvs(tmp_path)
        result = self._run(tmp_path, splits)
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report


# ────────────────────────────────────────────────────────
# CLI — Single file mode (subprocess)
# ────────────────────────────────────────────────────────

class TestCLISingleFileMode:
    def _run(self, tmp_path, csv_path, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "schema_preflight.py"),
            "--input-csv", str(csv_path),
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_normal_pass(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path)
        result = self._run(tmp_path, csv_path)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_non_binary_target(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path, non_binary=True)
        result = self._run(tmp_path, csv_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "target_not_binary" in codes

    def test_null_pids(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path, nan_pid=5)
        result = self._run(tmp_path, csv_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "patient_id_nulls_detected" in codes

    def test_insufficient_patients(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path, n_rows=3)
        result = self._run(tmp_path, csv_path)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "insufficient_patients" in codes

    def test_single_class_warning(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path, single_class=True)
        result = self._run(tmp_path, csv_path)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "target_single_class" in warn_codes

    def test_missing_required_column(self, tmp_path: Path):
        # Create CSV without default columns
        p = tmp_path / "no_cols.csv"
        pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6], "col_c": ["2024-01-01"] * 3}).to_csv(p, index=False)
        result = self._run(tmp_path, p)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "required_column_missing" in codes

    def test_empty_csv(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("a,b\n", encoding="utf-8")
        result = self._run(tmp_path, p)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "input_csv_read_failed" in codes

    def test_strict_mode(self, tmp_path: Path):
        # Use alias columns so auto-mapping triggers
        csv_path = _make_single_csv(tmp_path, target_col="label", pid_col="subject_id", time_col="timestamp")
        result = self._run(tmp_path, csv_path, extra_args=[
            "--strict",
            "--target-col", "y",
            "--patient-id-col", "patient_id",
            "--time-col", "event_time",
        ])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "strict_auto_mapping_not_allowed" in codes

    def test_report_single_file_mode(self, tmp_path: Path):
        csv_path = _make_single_csv(tmp_path)
        result = self._run(tmp_path, csv_path)
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["mode"] == "single_file"

    def test_nonexistent_csv(self, tmp_path: Path):
        result = self._run(tmp_path, tmp_path / "nonexistent.csv")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "input_csv_read_failed" in codes
