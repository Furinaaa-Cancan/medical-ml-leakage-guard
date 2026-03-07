"""Comprehensive unit tests for scripts/leakage_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from leakage_gate import (
    bounds_for_time,
    epoch_to_iso,
    parse_comma_set,
    parse_csv,
    row_signature,
    try_parse_time,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_csv(path: Path, headers: list, rows: list) -> Path:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_clean_splits(tmp_path: Path, n=20, with_time=True):
    """Create non-overlapping train/valid/test CSVs."""
    paths = {}
    for split, start in [("train", 0), ("valid", 100), ("test", 200)]:
        headers = ["patient_id", "age", "y"]
        if with_time:
            headers.append("event_time")
        rows = []
        for i in range(n):
            row = [f"P{start+i:04d}", str(20 + i), str(i % 2)]
            if with_time:
                base_day = start + i
                row.append(f"2024-{1 + base_day // 28:02d}-{1 + base_day % 28:02d}")
            rows.append(row)
        p = tmp_path / f"{split}.csv"
        _write_csv(p, headers, rows)
        paths[split] = p
    return paths


# ────────────────────────────────────────────────────────
# parse_csv
# ────────────────────────────────────────────────────────

class TestParseCsv:
    def test_normal(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        _write_csv(p, ["a", "b"], [["1", "2"], ["3", "4"]])
        result = parse_csv(str(p), "train")
        assert result["headers"] == ["a", "b"]
        assert len(result["rows"]) == 2
        assert result["rows"][0]["a"] == "1"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_csv("/nonexistent/path.csv", "train")

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="missing CSV header"):
            parse_csv(str(p), "train")

    def test_header_only(self, tmp_path: Path):
        p = tmp_path / "header.csv"
        _write_csv(p, ["a", "b"], [])
        result = parse_csv(str(p), "train")
        assert result["headers"] == ["a", "b"]
        assert len(result["rows"]) == 0

    def test_whitespace_stripping(self, tmp_path: Path):
        p = tmp_path / "ws.csv"
        p.write_text(" a , b \n 1 , 2 \n", encoding="utf-8")
        result = parse_csv(str(p), "train")
        assert result["headers"] == ["a", "b"]
        assert result["rows"][0]["a"] == "1"
        assert result["rows"][0]["b"] == "2"


# ────────────────────────────────────────────────────────
# parse_comma_set
# ────────────────────────────────────────────────────────

class TestParseCommaSet:
    def test_normal(self):
        assert parse_comma_set("a,b,c") == {"a", "b", "c"}

    def test_whitespace(self):
        assert parse_comma_set(" a , b ") == {"a", "b"}

    def test_empty(self):
        assert parse_comma_set("") == set()

    def test_single(self):
        assert parse_comma_set("x") == {"x"}

    def test_trailing_comma(self):
        assert parse_comma_set("a,b,") == {"a", "b"}


# ────────────────────────────────────────────────────────
# row_signature
# ────────────────────────────────────────────────────────

class TestRowSignature:
    def test_deterministic(self):
        row = {"a": "1", "b": "2"}
        s1 = row_signature(row, set())
        s2 = row_signature(row, set())
        assert s1 == s2
        assert len(s1) == 64  # SHA256 hex

    def test_different_rows(self):
        r1 = {"a": "1", "b": "2"}
        r2 = {"a": "1", "b": "3"}
        assert row_signature(r1, set()) != row_signature(r2, set())

    def test_ignore_cols(self):
        r1 = {"a": "1", "b": "2"}
        r2 = {"a": "1", "b": "99"}
        # Ignoring "b" should make signatures equal
        assert row_signature(r1, {"b"}) == row_signature(r2, {"b"})

    def test_order_independent(self):
        r1 = {"b": "2", "a": "1"}
        r2 = {"a": "1", "b": "2"}
        assert row_signature(r1, set()) == row_signature(r2, set())

    def test_empty_row(self):
        sig = row_signature({}, set())
        assert len(sig) == 64


# ────────────────────────────────────────────────────────
# try_parse_time
# ────────────────────────────────────────────────────────

class TestTryParseTime:
    def test_iso_date(self):
        ts = try_parse_time("2024-01-15")
        assert ts is not None

    def test_iso_datetime(self):
        ts = try_parse_time("2024-01-15 10:30:00")
        assert ts is not None

    def test_numeric_timestamp(self):
        ts = try_parse_time("1700000000")
        assert ts == 1700000000.0

    def test_slash_date(self):
        ts = try_parse_time("01/15/2024")
        assert ts is not None

    def test_iso_with_z(self):
        ts = try_parse_time("2024-01-15T10:30:00Z")
        assert ts is not None

    def test_empty_string(self):
        assert try_parse_time("") is None

    def test_whitespace(self):
        assert try_parse_time("   ") is None

    def test_unparseable(self):
        assert try_parse_time("not-a-date") is None

    def test_ordering(self):
        t1 = try_parse_time("2024-01-01")
        t2 = try_parse_time("2024-06-01")
        assert t1 < t2


# ────────────────────────────────────────────────────────
# bounds_for_time
# ────────────────────────────────────────────────────────

class TestBoundsForTime:
    def test_normal(self):
        rows = [
            {"t": "2024-01-01"},
            {"t": "2024-06-15"},
            {"t": "2024-12-31"},
        ]
        b = bounds_for_time(rows, "t")
        assert b["count"] == 3
        assert b["min"] is not None
        assert b["max"] is not None
        assert b["min"] < b["max"]
        assert b["missing"] == 0
        assert b["invalid"] == 0

    def test_missing_values(self):
        rows = [{"t": "2024-01-01"}, {"t": ""}, {"t": "2024-06-01"}]
        b = bounds_for_time(rows, "t")
        assert b["count"] == 2
        assert b["missing"] == 1

    def test_invalid_values(self):
        rows = [{"t": "2024-01-01"}, {"t": "bad"}, {"t": "2024-06-01"}]
        b = bounds_for_time(rows, "t")
        assert b["count"] == 2
        assert b["invalid"] == 1

    def test_all_invalid(self):
        rows = [{"t": "bad1"}, {"t": "bad2"}]
        b = bounds_for_time(rows, "t")
        assert b["count"] == 0
        assert b["min"] is None
        assert b["max"] is None

    def test_empty_rows(self):
        b = bounds_for_time([], "t")
        assert b["count"] == 0

    def test_missing_col(self):
        rows = [{"other": "val"}]
        b = bounds_for_time(rows, "t")
        assert b["missing"] == 1


# ────────────────────────────────────────────────────────
# epoch_to_iso
# ────────────────────────────────────────────────────────

class TestEpochToIso:
    def test_normal(self):
        result = epoch_to_iso(0.0)
        assert "1970" in result
        assert result.endswith("Z")

    def test_none(self):
        assert epoch_to_iso(None) is None


# ────────────────────────────────────────────────────────
# CLI tests (subprocess)
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, splits, extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(splits["train"]),
            "--test", str(splits["test"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if "valid" in splits:
            cmd.extend(["--valid", str(splits["valid"])])
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_clean_pass(self, tmp_path: Path):
        splits = _make_clean_splits(tmp_path, with_time=False)
        result = self._run(tmp_path, splits)
        assert result.returncode == 0, f"stdout: {result.stdout}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_row_overlap_detected(self, tmp_path: Path):
        """Inject identical row in train and test."""
        splits = _make_clean_splits(tmp_path, with_time=False)
        # Append a train row to test
        train_text = splits["train"].read_text()
        first_data_line = train_text.strip().split("\n")[1]
        test_text = splits["test"].read_text().strip() + "\n" + first_data_line + "\n"
        splits["test"].write_text(test_text, encoding="utf-8")
        result = self._run(tmp_path, splits)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "row_overlap" in codes

    def test_entity_id_overlap(self, tmp_path: Path):
        """Same patient_id in train and test."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["patient_id", "age", "y"],
                   [["P001", "30", "0"], ["P002", "40", "1"]])
        _write_csv(test_p, ["patient_id", "age", "y"],
                   [["P002", "50", "0"], ["P003", "60", "1"]])  # P002 overlaps
        splits = {"train": train_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--id-cols", "patient_id"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "id_overlap" in codes

    def test_no_entity_overlap(self, tmp_path: Path):
        splits = _make_clean_splits(tmp_path, with_time=False)
        result = self._run(tmp_path, splits, extra_args=["--id-cols", "patient_id"])
        assert result.returncode == 0

    def test_temporal_overlap(self, tmp_path: Path):
        """Test time range in test is before train."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "t"],
                   [["A", "0", "2024-06-01"], ["B", "1", "2024-06-15"]])
        _write_csv(test_p, ["pid", "y", "t"],
                   [["C", "0", "2024-01-01"], ["D", "1", "2024-01-15"]])  # Before train
        splits = {"train": train_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--time-col", "t"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_overlap" in codes

    def test_temporal_ordering_pass(self, tmp_path: Path):
        """Correct temporal ordering: train < valid < test."""
        train_p = tmp_path / "train.csv"
        valid_p = tmp_path / "valid.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "t"],
                   [["A", "0", "2024-01-01"], ["B", "1", "2024-01-15"]])
        _write_csv(valid_p, ["pid", "y", "t"],
                   [["C", "0", "2024-03-01"], ["D", "1", "2024-03-15"]])
        _write_csv(test_p, ["pid", "y", "t"],
                   [["E", "0", "2024-06-01"], ["F", "1", "2024-06-15"]])
        splits = {"train": train_p, "valid": valid_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--time-col", "t"])
        assert result.returncode == 0

    def test_temporal_boundary_exact(self, tmp_path: Path):
        """Train max == test min: SHOULD trigger overlap (uses >=, fail-closed)."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "t"],
                   [["A", "0", "2024-01-01"], ["B", "1", "2024-06-01"]])
        _write_csv(test_p, ["pid", "y", "t"],
                   [["C", "0", "2024-06-01"], ["D", "1", "2024-12-01"]])
        splits = {"train": train_p, "test": test_p}
        self._run(tmp_path, splits, extra_args=["--time-col", "t"])
        # left_max >= right_min is the check (fail-closed), so equal times SHOULD fail
        report = json.loads((tmp_path / "report.json").read_text())
        temporal_failures = [f for f in report["failures"] if f["code"] == "temporal_overlap"]
        assert len(temporal_failures) == 1

    def test_suspicious_feature_names(self, tmp_path: Path):
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        # Use names matching the regex: (?:^|_)(target|label|outcome)(?:_|$)
        # "label_encoded" matches ^label_ ; "has_outcome_flag" matches _outcome_
        _write_csv(train_p, ["pid", "y", "label_encoded", "has_outcome_flag"],
                   [["A", "0", "1", "2"]])
        _write_csv(test_p, ["pid", "y", "label_encoded", "has_outcome_flag"],
                   [["B", "1", "3", "4"]])
        splits = {"train": train_p, "test": test_p}
        self._run(tmp_path, splits, extra_args=["--target-col", "y"])
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "suspicious_feature_names" in warn_codes

    def test_target_col_not_suspicious(self, tmp_path: Path):
        """Target col itself should NOT be flagged as suspicious."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "target", "age"],
                   [["A", "0", "30"]])
        _write_csv(test_p, ["pid", "target", "age"],
                   [["B", "1", "40"]])
        splits = {"train": train_p, "test": test_p}
        self._run(tmp_path, splits, extra_args=["--target-col", "target"])
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "suspicious_feature_names" not in warn_codes

    def test_column_mismatch_warning(self, tmp_path: Path):
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "extra_col"],
                   [["A", "0", "1"]])
        _write_csv(test_p, ["pid", "y"],
                   [["B", "1"]])
        splits = {"train": train_p, "test": test_p}
        self._run(tmp_path, splits)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "column_mismatch" in warn_codes

    def test_missing_target_column(self, tmp_path: Path):
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "age"],
                   [["A", "30"]])
        _write_csv(test_p, ["pid", "age"],
                   [["B", "40"]])
        splits = {"train": train_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--target-col", "y"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_target_column" in codes

    def test_strict_mode_warns_become_failures(self, tmp_path: Path):
        """With --strict, warnings should cause failure."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "extra"],
                   [["A", "0", "1"]])
        _write_csv(test_p, ["pid", "y"],
                   [["B", "1"]])
        splits = {"train": train_p, "test": test_p}
        # Without strict: should pass (column_mismatch is warning)
        r1 = self._run(tmp_path, splits)
        assert r1.returncode == 0
        # With strict: should fail
        r2 = self._run(tmp_path, splits, extra_args=["--strict"])
        assert r2.returncode == 2

    def test_missing_time_column(self, tmp_path: Path):
        splits = _make_clean_splits(tmp_path, with_time=False)
        result = self._run(tmp_path, splits, extra_args=["--time-col", "nonexistent"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_time_column" in codes

    def test_ignore_cols_in_row_overlap(self, tmp_path: Path):
        """Rows differ only in ignored column should still overlap."""
        train_p = tmp_path / "train.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "age", "random_id"],
                   [["A", "30", "111"]])
        _write_csv(test_p, ["pid", "age", "random_id"],
                   [["A", "30", "222"]])
        splits = {"train": train_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--ignore-cols", "random_id"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "row_overlap" in codes

    def test_file_not_found(self, tmp_path: Path):
        train_p = tmp_path / "train.csv"
        _write_csv(train_p, ["a", "b"], [["1", "2"]])
        splits = {"train": train_p, "test": tmp_path / "nonexistent.csv"}
        result = self._run(tmp_path, splits)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "io_error" in codes

    def test_report_structure(self, tmp_path: Path):
        splits = _make_clean_splits(tmp_path, with_time=False)
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
        assert "rows_per_split" in report["summary"]

    def test_multiple_temporal_violations(self, tmp_path: Path):
        """Both train>valid and valid>test violations."""
        train_p = tmp_path / "train.csv"
        valid_p = tmp_path / "valid.csv"
        test_p = tmp_path / "test.csv"
        _write_csv(train_p, ["pid", "y", "t"],
                   [["A", "0", "2024-12-01"]])  # Latest
        _write_csv(valid_p, ["pid", "y", "t"],
                   [["B", "1", "2024-06-01"]])  # Middle
        _write_csv(test_p, ["pid", "y", "t"],
                   [["C", "0", "2024-01-01"]])  # Earliest
        splits = {"train": train_p, "valid": valid_p, "test": test_p}
        result = self._run(tmp_path, splits, extra_args=["--time-col", "t"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        temporal_codes = [f for f in report["failures"] if f["code"] == "temporal_overlap"]
        assert len(temporal_codes) >= 2  # train>valid and train>test at minimum


# ── direct main() tests (for coverage) ──────────────────────────────────────

from leakage_gate import main as leak_main


class TestLeakageGateMain:
    def test_clean_pass(self, tmp_path, monkeypatch):
        splits = _make_clean_splits(tmp_path, with_time=True)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(splits["train"]),
            "--valid", str(splits["valid"]),
            "--test", str(splits["test"]),
            "--id-cols", "patient_id",
            "--time-col", "event_time",
            "--target-col", "y",
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 0
        data = json.loads(rpt.read_text())
        assert data["status"] == "pass"

    def test_row_overlap_fails(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        _write_csv(train, ["pid", "y"], [["A", "1"], ["B", "0"]])
        _write_csv(test, ["pid", "y"], [["A", "1"], ["C", "0"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(test),
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 2
        data = json.loads(rpt.read_text())
        codes = [f["code"] for f in data["failures"]]
        assert "row_overlap" in codes

    def test_id_overlap_fails(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        _write_csv(train, ["pid", "age", "y"], [["A", "30", "1"], ["B", "40", "0"]])
        _write_csv(test, ["pid", "age", "y"], [["A", "30", "0"], ["C", "50", "1"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(test),
            "--id-cols", "pid",
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 2
        data = json.loads(rpt.read_text())
        codes = [f["code"] for f in data["failures"]]
        assert "id_overlap" in codes

    def test_temporal_overlap_fails(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        _write_csv(train, ["pid", "t", "y"], [["A", "2025-01-01", "1"], ["B", "2025-06-01", "0"]])
        _write_csv(test, ["pid", "t", "y"], [["C", "2024-01-01", "1"], ["D", "2024-06-01", "0"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(test),
            "--time-col", "t",
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 2
        data = json.loads(rpt.read_text())
        codes = [f["code"] for f in data["failures"]]
        assert "temporal_overlap" in codes

    def test_suspicious_features_warning(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        _write_csv(train, ["pid", "outcome_flag", "y"], [["A", "10", "1"], ["B", "20", "0"]])
        _write_csv(test, ["pid", "outcome_flag", "y"], [["C", "30", "1"], ["D", "40", "0"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(test),
            "--target-col", "y",
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 0  # warnings only
        data = json.loads(rpt.read_text())
        codes = [w["code"] for w in data["warnings"]]
        assert "suspicious_feature_names" in codes

    def test_strict_mode_warnings_fail(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        test = tmp_path / "test.csv"
        _write_csv(train, ["pid", "outcome_flag", "y"], [["A", "10", "1"], ["B", "20", "0"]])
        _write_csv(test, ["pid", "outcome_flag", "y"], [["C", "30", "1"], ["D", "40", "0"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(test),
            "--target-col", "y", "--strict",
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 2  # strict promotes warnings

    def test_file_not_found_returns_2(self, tmp_path, monkeypatch):
        train = tmp_path / "train.csv"
        _write_csv(train, ["a", "b"], [["1", "2"]])
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(train), "--test", str(tmp_path / "nope.csv"),
            "--report", str(rpt),
        ])
        rc = leak_main()
        assert rc == 2

    def test_no_report_flag(self, tmp_path, monkeypatch, capsys):
        splits = _make_clean_splits(tmp_path, with_time=False)
        monkeypatch.setattr("sys.argv", [
            "lg", "--train", str(splits["train"]),
            "--valid", str(splits["valid"]),
            "--test", str(splits["test"]),
        ])
        rc = leak_main()
        assert rc == 0
