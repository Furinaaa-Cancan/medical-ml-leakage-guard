"""Comprehensive unit tests for scripts/split_data.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from split_data import (
    _temp_col_name,
    file_sha256,
    generate_split_protocol,
    get_patient_label,
    load_csv,
    split_grouped_random,
    split_grouped_temporal,
    split_stratified_grouped,
    split_summary,
    validate_binary_target,
    validate_columns,
    validate_ratios,
    validate_splits,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Mock data generators
# ────────────────────────────────────────────────────────

def _make_csv(
    tmp_path: Path,
    n_patients: int = 30,
    rows_per_patient: int = 3,
    pos_rate: float = 0.5,
    with_time: bool = True,
    filename: str = "data.csv",
    nan_pid_rows: int = 0,
    nan_target_rows: int = 0,
    extra_cols: int = 0,
) -> Path:
    """Generate a test CSV with controllable properties."""
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(n_patients):
        label = 1 if rng.random() < pos_rate else 0
        for visit in range(rows_per_patient):
            row = {
                "patient_id": f"P{pid:04d}",
                "y": label,
                "age": int(rng.integers(20, 80)),
                "feature_1": round(float(rng.normal(0, 1)), 4),
            }
            if with_time:
                # Spread times across a wide range so temporal splits work
                from datetime import date, timedelta
                base_date = date(2020, 1, 1)
                day_offset = pid * 10 + visit
                d = base_date + timedelta(days=day_offset)
                row["event_time"] = d.isoformat()
            for c in range(extra_cols):
                row[f"extra_{c}"] = float(rng.normal())
            rows.append(row)

    df = pd.DataFrame(rows)

    # Inject NaN patient_ids
    if nan_pid_rows > 0:
        idx = rng.choice(len(df), size=min(nan_pid_rows, len(df)), replace=False)
        df.loc[idx, "patient_id"] = np.nan

    # Inject NaN targets
    if nan_target_rows > 0:
        valid_idx = df["patient_id"].notna()
        candidates = df.index[valid_idx].tolist()
        chosen = rng.choice(candidates, size=min(nan_target_rows, len(candidates)), replace=False)
        df.loc[chosen, "y"] = np.nan

    p = tmp_path / filename
    df.to_csv(p, index=False)
    return p


# ────────────────────────────────────────────────────────
# file_sha256
# ────────────────────────────────────────────────────────

class TestFileSha256:
    def test_deterministic(self, tmp_path: Path):
        p = tmp_path / "test.txt"
        p.write_text("hello", encoding="utf-8")
        h1 = file_sha256(p)
        h2 = file_sha256(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex length

    def test_different_content(self, tmp_path: Path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("hello", encoding="utf-8")
        p2.write_text("world", encoding="utf-8")
        assert file_sha256(p1) != file_sha256(p2)

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        h = file_sha256(p)
        assert len(h) == 64

    def test_known_hash(self, tmp_path: Path):
        import hashlib
        p = tmp_path / "known.txt"
        content = b"test content for hash"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert file_sha256(p) == expected


# ────────────────────────────────────────────────────────
# load_csv
# ────────────────────────────────────────────────────────

class TestLoadCsv:
    def test_normal(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path)
        df = load_csv(str(csv_path))
        assert len(df) > 0
        assert "patient_id" in df.columns

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_not_a_file(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not a file"):
            load_csv(str(tmp_path))

    def test_empty_csv(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("col_a,col_b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_csv(str(p))


# ────────────────────────────────────────────────────────
# validate_columns
# ────────────────────────────────────────────────────────

class TestValidateColumns:
    def test_all_present(self):
        df = pd.DataFrame({"pid": [1], "y": [0], "t": ["2024-01-01"]})
        validate_columns(df, "pid", "y", "t")  # Should not raise

    def test_missing_patient_id(self):
        df = pd.DataFrame({"y": [0], "t": ["2024-01-01"]})
        with pytest.raises(ValueError, match="patient_id_col"):
            validate_columns(df, "pid", "y", "t")

    def test_missing_target(self):
        df = pd.DataFrame({"pid": [1], "t": ["2024-01-01"]})
        with pytest.raises(ValueError, match="target_col"):
            validate_columns(df, "pid", "y", "t")

    def test_missing_time(self):
        df = pd.DataFrame({"pid": [1], "y": [0]})
        with pytest.raises(ValueError, match="time_col"):
            validate_columns(df, "pid", "y", "t")

    def test_empty_time_col_skips_check(self):
        df = pd.DataFrame({"pid": [1], "y": [0]})
        validate_columns(df, "pid", "y", "")  # Should not raise

    def test_multiple_missing(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="Missing columns"):
            validate_columns(df, "pid", "y", "t")


# ────────────────────────────────────────────────────────
# validate_binary_target
# ────────────────────────────────────────────────────────

class TestValidateBinaryTarget:
    def test_normal_binary(self):
        df = pd.DataFrame({"y": [0, 1, 0, 1]})
        na_count = validate_binary_target(df, "y")
        assert na_count == 0

    def test_nan_target(self):
        df = pd.DataFrame({"y": [0, 1, np.nan, 1]})
        na_count = validate_binary_target(df, "y")
        assert na_count == 1

    def test_non_numeric_value(self):
        df = pd.DataFrame({"y": [0, 1, "bad"]})
        with pytest.raises(ValueError, match="non-numeric"):
            validate_binary_target(df, "y")

    def test_non_binary_value(self):
        df = pd.DataFrame({"y": [0, 1, 2]})
        with pytest.raises(ValueError, match="non-binary"):
            validate_binary_target(df, "y")

    def test_negative_value(self):
        df = pd.DataFrame({"y": [0, 1, -1]})
        with pytest.raises(ValueError, match="non-binary"):
            validate_binary_target(df, "y")

    def test_all_positive(self):
        df = pd.DataFrame({"y": [1, 1, 1]})
        # Should not raise, just warn (single class)
        validate_binary_target(df, "y")

    def test_all_nan(self):
        df = pd.DataFrame({"y": [np.nan, np.nan]})
        with pytest.raises(ValueError, match="no valid values"):
            validate_binary_target(df, "y")

    def test_float_01(self):
        df = pd.DataFrame({"y": [0.0, 1.0, 0.0]})
        na_count = validate_binary_target(df, "y")
        assert na_count == 0


# ────────────────────────────────────────────────────────
# validate_ratios
# ────────────────────────────────────────────────────────

class TestValidateRatios:
    def test_valid(self):
        validate_ratios(0.6, 0.2, 0.2)  # Should not raise

    def test_sum_too_large(self):
        with pytest.raises(ValueError, match="sum to"):
            validate_ratios(0.6, 0.3, 0.3)

    def test_sum_too_small(self):
        with pytest.raises(ValueError, match="sum to"):
            validate_ratios(0.3, 0.1, 0.1)

    def test_ratio_too_small(self):
        with pytest.raises(ValueError, match="must be >= 0.05"):
            validate_ratios(0.9, 0.06, 0.04)

    def test_exact_boundary(self):
        validate_ratios(0.6, 0.2, 0.2)  # Exactly 1.0


# ────────────────────────────────────────────────────────
# get_patient_label
# ────────────────────────────────────────────────────────

class TestGetPatientLabel:
    def test_majority_label(self):
        df = pd.DataFrame({
            "pid": ["A", "A", "A", "B", "B", "B"],
            "y": [1, 1, 0, 0, 0, 1],
        })
        result = get_patient_label(df, "pid", "y")
        # A: mean=2/3=0.667 → label=1, B: mean=1/3=0.333 → label=0
        assert result.loc["A", "label"] == 1
        assert result.loc["B", "label"] == 0

    def test_all_same_label(self):
        df = pd.DataFrame({"pid": ["A", "A"], "y": [1, 1]})
        result = get_patient_label(df, "pid", "y")
        assert result.loc["A", "label"] == 1

    def test_exactly_half(self):
        df = pd.DataFrame({"pid": ["A", "A"], "y": [0, 1]})
        result = get_patient_label(df, "pid", "y")
        # mean=0.5, >= 0.5 → label=1
        assert result.loc["A", "label"] == 1


# ────────────────────────────────────────────────────────
# _temp_col_name
# ────────────────────────────────────────────────────────

class TestTempColName:
    def test_no_collision(self):
        df = pd.DataFrame({"a": [1]})
        assert _temp_col_name(df) == "__split_tmp__"

    def test_collision(self):
        df = pd.DataFrame({"__split_tmp__": [1], "a": [2]})
        assert _temp_col_name(df) == "__split_tmp__1"

    def test_multiple_collisions(self):
        df = pd.DataFrame({
            "__split_tmp__": [1],
            "__split_tmp__1": [2],
            "__split_tmp__2": [3],
        })
        assert _temp_col_name(df) == "__split_tmp__3"

    def test_custom_base(self):
        df = pd.DataFrame({"__custom__": [1]})
        assert _temp_col_name(df, "__custom__") == "__custom__1"

    def test_custom_base_no_collision(self):
        df = pd.DataFrame({"a": [1]})
        assert _temp_col_name(df, "__mybase__") == "__mybase__"


# ────────────────────────────────────────────────────────
# split_grouped_temporal
# ────────────────────────────────────────────────────────

class TestSplitGroupedTemporal:
    def _make_temporal_df(self, n_patients=30, rows_per=3):
        rng = np.random.default_rng(42)
        rows = []
        for pid in range(n_patients):
            for visit in range(rows_per):
                day = pid * 10 + visit + 1
                rows.append({
                    "patient_id": f"P{pid:04d}",
                    "y": int(rng.random() < 0.5),
                    "event_time": f"2024-{1 + day // 29:02d}-{1 + day % 28:02d}",
                })
        return pd.DataFrame(rows)

    def test_normal_split(self):
        df = self._make_temporal_df()
        train, valid, test = split_grouped_temporal(df, "patient_id", "event_time", "y", 0.6, 0.2)
        total = len(train) + len(valid) + len(test)
        assert total == len(df)

    def test_patient_disjoint(self):
        df = self._make_temporal_df()
        train, valid, test = split_grouped_temporal(df, "patient_id", "event_time", "y", 0.6, 0.2)
        train_pids = set(train["patient_id"])
        valid_pids = set(valid["patient_id"])
        test_pids = set(test["patient_id"])
        assert train_pids.isdisjoint(valid_pids)
        assert train_pids.isdisjoint(test_pids)
        assert valid_pids.isdisjoint(test_pids)

    def test_no_time_col_raises(self):
        df = self._make_temporal_df()
        with pytest.raises(ValueError, match="requires --time-col"):
            split_grouped_temporal(df, "patient_id", "", "y", 0.6, 0.2)

    def test_all_nat_raises(self):
        df = pd.DataFrame({
            "patient_id": [f"P{i}" for i in range(10) for _ in range(3)],
            "y": [0] * 30,
            "event_time": ["not-a-date"] * 30,
        })
        with pytest.raises(ValueError, match="Cannot parse"):
            split_grouped_temporal(df, "patient_id", "event_time", "y", 0.6, 0.2)

    def test_row_count_preserved(self):
        df = self._make_temporal_df(n_patients=50, rows_per=5)
        train, valid, test = split_grouped_temporal(df, "patient_id", "event_time", "y", 0.6, 0.2)
        assert len(train) + len(valid) + len(test) == len(df)

    def test_no_tmp_col_in_output(self):
        df = self._make_temporal_df()
        train, valid, test = split_grouped_temporal(df, "patient_id", "event_time", "y", 0.6, 0.2)
        for split_df in [train, valid, test]:
            for col in split_df.columns:
                assert "__split_tmp__" not in col


# ────────────────────────────────────────────────────────
# split_grouped_random
# ────────────────────────────────────────────────────────

class TestSplitGroupedRandom:
    def _make_df(self, n_patients=30, rows_per=3):
        rng = np.random.default_rng(42)
        rows = []
        for pid in range(n_patients):
            for _ in range(rows_per):
                rows.append({
                    "patient_id": f"P{pid:04d}",
                    "y": int(rng.random() < 0.5),
                })
        return pd.DataFrame(rows)

    def test_normal_split(self):
        df = self._make_df()
        train, valid, test = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=42)
        assert len(train) + len(valid) + len(test) == len(df)

    def test_patient_disjoint(self):
        df = self._make_df()
        train, valid, test = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=42)
        train_pids = set(train["patient_id"])
        valid_pids = set(valid["patient_id"])
        test_pids = set(test["patient_id"])
        assert train_pids.isdisjoint(valid_pids)
        assert train_pids.isdisjoint(test_pids)
        assert valid_pids.isdisjoint(test_pids)

    def test_reproducible(self):
        df = self._make_df()
        t1, v1, te1 = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=42)
        t2, v2, te2 = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=42)
        assert set(t1["patient_id"]) == set(t2["patient_id"])
        assert set(v1["patient_id"]) == set(v2["patient_id"])
        assert set(te1["patient_id"]) == set(te2["patient_id"])

    def test_different_seed_different_split(self):
        df = self._make_df()
        t1, _, _ = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=1)
        t2, _, _ = split_grouped_random(df, "patient_id", "y", 0.6, 0.2, seed=2)
        # With different seeds, splits should (very likely) differ
        assert set(t1["patient_id"]) != set(t2["patient_id"])


# ────────────────────────────────────────────────────────
# split_stratified_grouped
# ────────────────────────────────────────────────────────

class TestSplitStratifiedGrouped:
    def _make_df(self, n_patients=30, rows_per=3, pos_rate=0.5):
        rng = np.random.default_rng(42)
        rows = []
        for pid in range(n_patients):
            label = 1 if rng.random() < pos_rate else 0
            for _ in range(rows_per):
                rows.append({"patient_id": f"P{pid:04d}", "y": label})
        return pd.DataFrame(rows)

    def test_normal_split(self):
        df = self._make_df()
        train, valid, test = split_stratified_grouped(df, "patient_id", "y", 0.6, 0.2, seed=42)
        assert len(train) + len(valid) + len(test) == len(df)

    def test_patient_disjoint(self):
        df = self._make_df()
        train, valid, test = split_stratified_grouped(df, "patient_id", "y", 0.6, 0.2, seed=42)
        train_pids = set(train["patient_id"])
        valid_pids = set(valid["patient_id"])
        test_pids = set(test["patient_id"])
        assert train_pids.isdisjoint(valid_pids)
        assert train_pids.isdisjoint(test_pids)
        assert valid_pids.isdisjoint(test_pids)

    def test_preserves_prevalence(self):
        df = self._make_df(n_patients=100, pos_rate=0.3)
        train, valid, test = split_stratified_grouped(df, "patient_id", "y", 0.6, 0.2, seed=42)
        train_prev = train["y"].mean()
        test_prev = test["y"].mean()
        # Stratified should keep prevalence close (within 0.15)
        assert abs(train_prev - test_prev) < 0.15

    def test_few_patients_one_class(self):
        """If one class has <3 patients, they go to train only."""
        np.random.default_rng(42)
        rows = []
        # 2 positive patients (will be < 3, should go to train)
        for pid in range(2):
            for _ in range(3):
                rows.append({"patient_id": f"POS{pid}", "y": 1})
        # 20 negative patients
        for pid in range(20):
            for _ in range(3):
                rows.append({"patient_id": f"NEG{pid}", "y": 0})
        df = pd.DataFrame(rows)
        train, valid, test = split_stratified_grouped(df, "patient_id", "y", 0.6, 0.2, seed=42)
        # Both POS patients should be in train
        train_pids = set(train["patient_id"])
        assert "POS0" in train_pids
        assert "POS1" in train_pids


# ────────────────────────────────────────────────────────
# validate_splits
# ────────────────────────────────────────────────────────

class TestValidateSplits:
    def _make_splits(self, n_train=80, n_valid=30, n_test=30, pos_rate=0.5):
        rng = np.random.default_rng(42)
        def _make_df(n, pid_start, time_start):
            rows = []
            for i in range(n):
                rows.append({
                    "patient_id": f"P{pid_start + i:04d}",
                    "y": 1 if rng.random() < pos_rate else 0,
                    "event_time": f"2024-{1 + (time_start + i) // 28:02d}-{1 + (time_start + i) % 28:02d}",
                })
            return pd.DataFrame(rows)
        return {
            "train": _make_df(n_train, 0, 0),
            "valid": _make_df(n_valid, n_train, n_train),
            "test": _make_df(n_test, n_train + n_valid, n_train + n_valid),
        }

    def test_no_issues(self):
        splits = self._make_splits()
        issues = validate_splits(splits, "patient_id", "y", "event_time", 20)
        hard = [i for i in issues if i.get("level") != "warn"]
        assert len(hard) == 0

    def test_insufficient_rows(self):
        splits = self._make_splits(n_test=5)
        issues = validate_splits(splits, "patient_id", "y", "event_time", 20)
        codes = [i["code"] for i in issues]
        assert "insufficient_rows" in codes

    def test_patient_overlap(self):
        splits = self._make_splits()
        # Inject overlap
        overlap_row = splits["train"].iloc[0:1].copy()
        splits["test"] = pd.concat([splits["test"], overlap_row], ignore_index=True)
        issues = validate_splits(splits, "patient_id", "y", "event_time", 20)
        codes = [i["code"] for i in issues]
        assert "patient_overlap" in codes

    def test_prevalence_shift(self):
        np.random.default_rng(42)
        splits = {
            "train": pd.DataFrame({
                "patient_id": [f"T{i}" for i in range(30)],
                "y": [1] * 25 + [0] * 5,  # 83% positive
            }),
            "test": pd.DataFrame({
                "patient_id": [f"E{i}" for i in range(30)],
                "y": [0] * 25 + [1] * 5,  # 17% positive
            }),
        }
        issues = validate_splits(splits, "patient_id", "y", "", 5)
        codes = [i["code"] for i in issues]
        assert "prevalence_shift" in codes

    def test_insufficient_positive(self):
        splits = {
            "train": pd.DataFrame({
                "patient_id": [f"T{i}" for i in range(30)],
                "y": [1] * 15 + [0] * 15,
            }),
            "test": pd.DataFrame({
                "patient_id": [f"E{i}" for i in range(30)],
                "y": [0] * 25 + [1] * 5,  # only 5 positive < MIN_POSITIVE=10
            }),
        }
        issues = validate_splits(splits, "patient_id", "y", "", 5)
        codes = [i["code"] for i in issues]
        assert "insufficient_positive" in codes

    def test_insufficient_negative(self):
        splits = {
            "train": pd.DataFrame({
                "patient_id": [f"T{i}" for i in range(30)],
                "y": [0] * 15 + [1] * 15,
            }),
            "test": pd.DataFrame({
                "patient_id": [f"E{i}" for i in range(30)],
                "y": [1] * 25 + [0] * 5,  # only 5 negative < MIN_NEGATIVE=10
            }),
        }
        issues = validate_splits(splits, "patient_id", "y", "", 5)
        codes = [i["code"] for i in issues]
        assert "insufficient_negative" in codes

    def test_insufficient_patients(self):
        splits = {
            "train": pd.DataFrame({
                "patient_id": ["A", "A", "B", "B"],
                "y": [0, 1, 0, 1],
            }),
        }
        issues = validate_splits(splits, "patient_id", "y", "", 1)
        codes = [i["code"] for i in issues]
        assert "insufficient_patients_in_split" in codes

    def test_temporal_overlap(self):
        splits = {
            "train": pd.DataFrame({
                "patient_id": [f"T{i}" for i in range(20)],
                "y": [0, 1] * 10,
                "event_time": [f"2024-06-{i+1:02d}" for i in range(20)],
            }),
            "valid": pd.DataFrame({
                "patient_id": [f"V{i}" for i in range(10)],
                "y": [0, 1] * 5,
                "event_time": [f"2024-05-{i+1:02d}" for i in range(10)],  # BEFORE train
            }),
            "test": pd.DataFrame({
                "patient_id": [f"E{i}" for i in range(10)],
                "y": [0, 1] * 5,
                "event_time": [f"2024-07-{i+1:02d}" for i in range(10)],
            }),
        }
        issues = validate_splits(splits, "patient_id", "y", "event_time", 5)
        codes = [i["code"] for i in issues]
        assert "temporal_overlap" in codes


# ────────────────────────────────────────────────────────
# split_summary
# ────────────────────────────────────────────────────────

class TestSplitSummary:
    def test_with_time(self):
        df = pd.DataFrame({
            "patient_id": ["A", "B", "C"],
            "y": [0, 1, 0],
            "event_time": ["2024-01-01", "2024-06-15", "2024-12-31"],
        })
        s = split_summary(df, "y", "event_time", "patient_id")
        assert s["rows"] == 3
        assert s["patients"] == 3
        assert s["positive_count"] == 1
        assert s["negative_count"] == 2
        assert "time_min" in s
        assert "time_max" in s

    def test_without_time(self):
        df = pd.DataFrame({"patient_id": ["A", "B"], "y": [0, 1]})
        s = split_summary(df, "y", "", "patient_id")
        assert s["rows"] == 2
        assert "time_min" not in s


# ────────────────────────────────────────────────────────
# generate_split_protocol
# ────────────────────────────────────────────────────────

class TestGenerateSplitProtocol:
    def test_temporal(self):
        p = generate_split_protocol("grouped_temporal", "pid", "event_time", 42)
        assert p["split_strategy"] == "grouped_temporal"
        assert p["requires_temporal_order"] is True
        assert p["index_time_col"] == "event_time"
        assert p["requires_group_disjoint"] is True
        assert p["allow_patient_overlap"] is False

    def test_random(self):
        p = generate_split_protocol("grouped_random", "pid", "", 42)
        assert p["requires_temporal_order"] is False
        assert p["index_time_col"] == ""


# ────────────────────────────────────────────────────────
# CLI (subprocess) integration tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run_split(self, tmp_path: Path, csv_path: Path, strategy: str = "grouped_random",
                   time_col: str = "", extra_args: Optional[list] = None) -> subprocess.CompletedProcess:
        out_dir = tmp_path / "output" / "data"
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "split_data.py"),
            "--input", str(csv_path),
            "--output-dir", str(out_dir),
            "--patient-id-col", "patient_id",
            "--target-col", "y",
            "--strategy", strategy,
            "--seed", "42",
        ]
        if time_col:
            cmd += ["--time-col", time_col]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    def test_grouped_random_success(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        out_dir = tmp_path / "output" / "data"
        assert (out_dir / "train.csv").exists()
        assert (out_dir / "valid.csv").exists()
        assert (out_dir / "test.csv").exists()

    def test_grouped_temporal_success(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=True)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_temporal", time_col="event_time")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_stratified_grouped_success(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="stratified_grouped")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_row_count_preservation(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        out_dir = tmp_path / "output" / "data"
        original = pd.read_csv(csv_path)
        train = pd.read_csv(out_dir / "train.csv")
        valid = pd.read_csv(out_dir / "valid.csv")
        test = pd.read_csv(out_dir / "test.csv")
        assert len(train) + len(valid) + len(test) == len(original)

    def test_patient_disjoint_cli(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        out_dir = tmp_path / "output" / "data"
        train = pd.read_csv(out_dir / "train.csv")
        valid = pd.read_csv(out_dir / "valid.csv")
        test = pd.read_csv(out_dir / "test.csv")
        train_pids = set(train["patient_id"])
        valid_pids = set(valid["patient_id"])
        test_pids = set(test["patient_id"])
        assert train_pids.isdisjoint(valid_pids)
        assert train_pids.isdisjoint(test_pids)
        assert valid_pids.isdisjoint(test_pids)

    def test_column_preservation(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=True, extra_cols=5)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        out_dir = tmp_path / "output" / "data"
        original = pd.read_csv(csv_path)
        train = pd.read_csv(out_dir / "train.csv")
        assert list(train.columns) == list(original.columns)

    def test_sha256_in_report(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        report_path = tmp_path / "output" / "evidence" / "split_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "input_sha256" in report
        assert len(report["input_sha256"]) == 64

    def test_split_protocol_generated(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        protocol_path = tmp_path / "output" / "configs" / "split_protocol.json"
        assert protocol_path.exists()
        protocol = json.loads(protocol_path.read_text())
        assert protocol["split_strategy"] == "grouped_random"
        assert protocol["requires_group_disjoint"] is True

    def test_nan_pid_excluded(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False, nan_pid_rows=3)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        assert "NaN patient IDs" in result.stderr

    def test_nan_target_excluded(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False, nan_target_rows=3)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        assert "NaN target" in result.stderr

    def test_too_few_patients(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=3, rows_per_patient=10, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 2
        assert "at least 6" in result.stderr

    def test_temporal_without_time_col_fails(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=30, with_time=True)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_temporal", time_col="")
        assert result.returncode == 2
        assert "time-col" in result.stderr.lower() or "time_col" in result.stderr.lower()

    def test_report_contract_version(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        result = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert result.returncode == 0
        report_path = tmp_path / "output" / "evidence" / "split_report.json"
        report = json.loads(report_path.read_text())
        assert report["contract_version"] == "split_report.v1"
        assert report["status"] == "pass"

    def test_idempotent(self, tmp_path: Path):
        csv_path = _make_csv(tmp_path, n_patients=60, rows_per_patient=4, with_time=False)
        r1 = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert r1.returncode == 0
        out_dir = tmp_path / "output" / "data"
        train1 = pd.read_csv(out_dir / "train.csv")
        # Run again (same seed, same data)
        r2 = self._run_split(tmp_path, csv_path, strategy="grouped_random")
        assert r2.returncode == 0
        train2 = pd.read_csv(out_dir / "train.csv")
        assert set(train1["patient_id"]) == set(train2["patient_id"])
