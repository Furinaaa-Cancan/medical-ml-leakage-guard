"""Tests for scripts/split_protocol_gate.py.

Covers helper functions (parse_bool, parse_non_empty_str, parse_label,
try_parse_time, epoch_to_iso, read_split), protocol field validation,
entity overlap, temporal ordering, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "split_protocol_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import split_protocol_gate as spg


# ── helper functions ─────────────────────────────────────────────────────────

class TestParseBool:
    def test_true(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({"k": True}, "k", failures) is True
        assert len(failures) == 0

    def test_false(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({"k": False}, "k", failures) is False
        assert len(failures) == 0

    def test_string(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({"k": "true"}, "k", failures) is None
        assert len(failures) == 1
        assert failures[0]["code"] == "invalid_protocol_field"

    def test_int(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({"k": 1}, "k", failures) is None
        assert len(failures) == 1

    def test_missing(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({}, "k", failures) is None
        assert len(failures) == 1

    def test_none(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_bool({"k": None}, "k", failures) is None
        assert len(failures) == 1


class TestParseNonEmptyStr:
    def test_valid(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({"k": "hello"}, "k", failures) == "hello"
        assert len(failures) == 0

    def test_strips_whitespace(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({"k": "  hello  "}, "k", failures) == "hello"

    def test_empty(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({"k": ""}, "k", failures) is None
        assert len(failures) == 1

    def test_whitespace_only(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({"k": "   "}, "k", failures) is None
        assert len(failures) == 1

    def test_missing(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({}, "k", failures) is None
        assert len(failures) == 1

    def test_non_string(self):
        failures: List[Dict[str, Any]] = []
        assert spg.parse_non_empty_str({"k": 123}, "k", failures) is None
        assert len(failures) == 1


class TestParseLabel:
    def test_zero(self):
        assert spg.parse_label("0") == 0

    def test_one(self):
        assert spg.parse_label("1") == 1

    def test_zero_float(self):
        assert spg.parse_label("0.0") == 0

    def test_one_float(self):
        assert spg.parse_label("1.0") == 1

    def test_empty(self):
        assert spg.parse_label("") is None

    def test_whitespace(self):
        assert spg.parse_label("  ") is None

    def test_two(self):
        assert spg.parse_label("2") is None

    def test_negative(self):
        assert spg.parse_label("-1") is None

    def test_string(self):
        assert spg.parse_label("abc") is None

    def test_nan(self):
        assert spg.parse_label("nan") is None

    def test_inf(self):
        assert spg.parse_label("inf") is None

    def test_padded(self):
        assert spg.parse_label("  1  ") == 1

    def test_half(self):
        assert spg.parse_label("0.5") is None


class TestTryParseTime:
    def test_iso_date(self):
        result = spg.try_parse_time("2023-01-15")
        assert result is not None

    def test_iso_datetime(self):
        result = spg.try_parse_time("2023-01-15 10:30:00")
        assert result is not None

    def test_slash_date(self):
        result = spg.try_parse_time("2023/01/15")
        assert result is not None

    def test_us_format(self):
        result = spg.try_parse_time("01/15/2023")
        assert result is not None

    def test_numeric_timestamp(self):
        result = spg.try_parse_time("1700000000.0")
        assert result is not None
        assert abs(result - 1700000000.0) < 0.001

    def test_iso_z_suffix(self):
        result = spg.try_parse_time("2023-01-15T10:30:00Z")
        assert result is not None

    def test_empty(self):
        assert spg.try_parse_time("") is None

    def test_whitespace(self):
        assert spg.try_parse_time("   ") is None

    def test_garbage(self):
        assert spg.try_parse_time("not-a-date") is None

    def test_ordering(self):
        t1 = spg.try_parse_time("2023-01-01")
        t2 = spg.try_parse_time("2023-12-31")
        assert t1 is not None and t2 is not None
        assert t1 < t2


class TestEpochToIso:
    def test_none(self):
        assert spg.epoch_to_iso(None) is None

    def test_known_epoch(self):
        result = spg.epoch_to_iso(0.0)
        assert result is not None
        assert "1970" in result

    def test_returns_z_suffix(self):
        result = spg.epoch_to_iso(1700000000.0)
        assert result is not None
        assert result.endswith("Z")


class TestReadSplit:
    def _make_csv(self, tmp_path: Path, name: str, rows: list) -> Path:
        p = tmp_path / name
        lines = ["patient_id,event_time,y"] + rows
        p.write_text("\n".join(lines) + "\n")
        return p

    def test_valid(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,2023-01-01,1",
            "P002,2023-06-15,0",
            "P003,2023-12-31,1",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert result["row_count"] == 3
        assert result["id_count"] == 3
        assert result["positive_count"] == 2
        assert result["negative_count"] == 1
        assert result["missing_id_rows"] == 0
        assert result["invalid_label_rows"] == 0
        assert result["time_min"] is not None
        assert result["time_max"] is not None
        assert result["time_min"] < result["time_max"]

    def test_missing_ids(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,2023-01-01,1",
            ",2023-06-15,0",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert result["missing_id_rows"] == 1

    def test_invalid_labels(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,2023-01-01,1",
            "P002,2023-06-15,abc",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert result["invalid_label_rows"] == 1

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            spg.read_split(str(tmp_path / "nope.csv"), "train", "patient_id", "event_time", "y")

    def test_missing_column(self, tmp_path: Path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="missing id_col"):
            spg.read_split(str(p), "train", "patient_id", "event_time", "y")

    def test_prevalence(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,2023-01-01,1",
            "P002,2023-06-15,0",
            "P003,2023-09-01,0",
            "P004,2023-12-01,0",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert abs(result["prevalence"] - 0.25) < 0.001

    def test_unparseable_time(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,bad_time,1",
            "P002,2023-06-15,0",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert result["invalid_time_rows"] == 1
        assert result["time_parsed_count"] == 1

    def test_missing_time(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "split.csv", [
            "P001,,1",
            "P002,2023-06-15,0",
        ])
        result = spg.read_split(str(p), "train", "patient_id", "event_time", "y")
        assert result["missing_time_rows"] == 1


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_protocol(tmp_path: Path, overrides: dict = None) -> Path:
    spec = {
        "split_strategy": "grouped_temporal",
        "split_reference": "protocol-v1",
        "id_col": "patient_id",
        "index_time_col": "event_time",
        "frozen_before_modeling": True,
        "requires_group_disjoint": True,
        "requires_temporal_order": True,
        "allow_patient_overlap": False,
        "allow_time_overlap": False,
        "split_seed_locked": True,
    }
    if overrides:
        spec.update(overrides)
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(spec))
    return p


def _make_splits(tmp_path: Path, train_rows=None, valid_rows=None, test_rows=None):
    if train_rows is None:
        train_rows = ["P001,2023-01-01,1", "P002,2023-02-01,0", "P003,2023-03-01,1"]
    if valid_rows is None:
        valid_rows = ["P004,2023-07-01,0", "P005,2023-08-01,1"]
    if test_rows is None:
        test_rows = ["P006,2024-01-01,1", "P007,2024-02-01,0"]

    for name, rows in [("train.csv", train_rows), ("valid.csv", valid_rows), ("test.csv", test_rows)]:
        p = tmp_path / name
        lines = ["patient_id,event_time,y"] + rows
        p.write_text("\n".join(lines) + "\n")


def _run_gate(tmp_path: Path, protocol_path: Path, extra_args: list = None, strict: bool = False) -> dict:
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--protocol-spec", str(protocol_path),
        "--train", str(tmp_path / "train.csv"),
        "--valid", str(tmp_path / "valid.csv"),
        "--test", str(tmp_path / "test.csv"),
        "--id-col", "patient_id",
        "--time-col", "event_time",
        "--target-col", "y",
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_protocol_pass(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        assert "status" in report
        assert "strict_mode" in report
        assert "split_strategy" in report["summary"]
        assert "split_reference" in report["summary"]
        assert "protocol_spec" in report.get("input_files", {})
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "splits" in report["summary"]

    def test_split_summary_fields(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        for split_name in ("train", "valid", "test"):
            s = report["summary"]["splits"][split_name]
            assert "row_count" in s
            assert "id_count" in s
            assert "positive_count" in s
            assert "negative_count" in s
            assert "prevalence" in s
            assert "time_min" in s
            assert "time_max" in s


class TestMissingProtocol:
    def test_missing_protocol_spec(self, tmp_path: Path):
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, tmp_path / "nonexistent.json")
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "missing_protocol_spec" in codes

    def test_invalid_json_protocol(self, tmp_path: Path):
        _make_splits(tmp_path)
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid")
        report = _run_gate(tmp_path, bad)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_protocol_spec" in codes

    def test_non_object_protocol(self, tmp_path: Path):
        _make_splits(tmp_path)
        bad = tmp_path / "arr.json"
        bad.write_text("[1,2,3]")
        report = _run_gate(tmp_path, bad)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_protocol_spec" in codes


class TestProtocolFieldValidation:
    def test_frozen_false(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"frozen_before_modeling": False})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "split_not_frozen" in codes

    def test_seed_not_locked(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"split_seed_locked": False})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "split_seed_not_locked" in codes

    def test_group_disjoint_false(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"requires_group_disjoint": False})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "group_disjoint_not_required" in codes

    def test_temporal_order_false(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"requires_temporal_order": False})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_order_not_required" in codes

    def test_allow_patient_overlap_true(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"allow_patient_overlap": True})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "patient_overlap_allowed" in codes

    def test_allow_time_overlap_true(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"allow_time_overlap": True})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "time_overlap_allowed" in codes

    def test_id_col_mismatch(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"id_col": "other_id"})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "protocol_id_col_mismatch" in codes

    def test_time_col_mismatch(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"index_time_col": "other_time"})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "protocol_time_col_mismatch" in codes

    def test_missing_strategy_field(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path, {"split_strategy": ""})
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_protocol_field" in codes


class TestEntityOverlap:
    def test_train_test_overlap(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-01-01,1", "P002,2023-02-01,0"],
            valid_rows=["P003,2023-07-01,1", "P004,2023-08-01,0"],
            test_rows=["P001,2024-01-01,1", "P005,2024-02-01,0"],  # P001 overlaps
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "entity_overlap" in codes
        overlap_issue = [f for f in report["failures"] if f["code"] == "entity_overlap"][0]
        assert overlap_issue["details"]["overlap_count"] == 1

    def test_train_valid_overlap(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-01-01,1", "P002,2023-02-01,0"],
            valid_rows=["P002,2023-07-01,1", "P003,2023-08-01,0"],  # P002 overlaps
            test_rows=["P004,2024-01-01,1", "P005,2024-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "entity_overlap" in codes

    def test_no_overlap(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "entity_overlap" not in codes


class TestTemporalOrdering:
    def test_temporal_violation_train_test(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2024-06-01,1", "P002,2024-07-01,0"],  # train AFTER test
            valid_rows=["P003,2023-07-01,1", "P004,2023-08-01,0"],
            test_rows=["P005,2023-01-01,1", "P006,2023-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_boundary_violation" in codes

    def test_temporal_violation_train_valid(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-09-01,1", "P002,2023-10-01,0"],  # train AFTER valid
            valid_rows=["P003,2023-01-01,1", "P004,2023-02-01,0"],
            test_rows=["P005,2024-01-01,1", "P006,2024-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_boundary_violation" in codes

    def test_temporal_boundary_equal(self, tmp_path: Path):
        """Boundary case: train max == valid min should fail (>= check)."""
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-06-15,1", "P002,2023-06-15,0"],
            valid_rows=["P003,2023-06-15,1", "P004,2023-07-01,0"],  # same date
            test_rows=["P005,2024-01-01,1", "P006,2024-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_boundary_violation" in codes

    def test_correct_temporal_order(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)  # defaults have correct ordering
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "temporal_boundary_violation" not in codes


class TestSplitDataQuality:
    def test_invalid_labels(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-01-01,1", "P002,2023-02-01,abc"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_labels" in codes

    def test_single_class(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-01-01,1", "P002,2023-02-01,1"],  # all positive
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "single_class_split" in codes

    def test_missing_entity_ids(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,2023-01-01,1", ",2023-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "missing_entity_ids" in codes

    def test_unparseable_times(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path,
            train_rows=["P001,bad_date,1", "P002,2023-02-01,0"],
        )
        report = _run_gate(tmp_path, protocol)
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_time_values" in codes

    def test_split_file_not_found(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        # Only create train, no valid/test
        (tmp_path / "train.csv").write_text("patient_id,event_time,y\nP001,2023-01-01,1\n")
        (tmp_path / "valid.csv").write_text("patient_id,event_time,y\nP002,2023-07-01,0\n")
        # test.csv missing
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--protocol-spec", str(protocol),
            "--train", str(tmp_path / "train.csv"),
            "--valid", str(tmp_path / "valid.csv"),
            "--test", str(tmp_path / "nonexistent.csv"),
            "--id-col", "patient_id",
            "--time-col", "event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        if report_path.exists():
            report = json.loads(report_path.read_text())
            assert report["status"] == "fail"
            codes = [f["code"] for f in report["failures"]]
            assert "split_io_error" in codes


class TestNoValidSplit:
    """Test without --valid argument."""

    def test_two_split_pass(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        (tmp_path / "train.csv").write_text(
            "patient_id,event_time,y\nP001,2023-01-01,1\nP002,2023-02-01,0\nP003,2023-03-01,1\n"
        )
        (tmp_path / "test.csv").write_text(
            "patient_id,event_time,y\nP004,2024-01-01,1\nP005,2024-02-01,0\n"
        )
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--protocol-spec", str(protocol),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--id-col", "patient_id",
            "--time-col", "event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"
        assert "valid" not in report["summary"]["splits"]


class TestStrictMode:
    def test_strict_flag_in_report(self, tmp_path: Path):
        protocol = _make_protocol(tmp_path)
        _make_splits(tmp_path)
        report = _run_gate(tmp_path, protocol, strict=True)
        assert report["strict_mode"] is True
