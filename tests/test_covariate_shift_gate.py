"""Tests for scripts/covariate_shift_gate.py.

Covers helper functions (is_missing, parse_ignore_cols, parse_binary_label,
try_parse_float, js_divergence, assign_numeric_bin, categorical_bucket,
safe_ratio, top_n_feature_summary), feature type discovery, split profiling,
pair evaluation, threshold validation, prevalence checks, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "covariate_shift_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import covariate_shift_gate as csg


# ── helper functions ─────────────────────────────────────────────────────────

class TestIsMissing:
    def test_empty(self):
        assert csg.is_missing("") is True

    def test_na(self):
        assert csg.is_missing("NA") is True

    def test_nan(self):
        assert csg.is_missing("nan") is True

    def test_null(self):
        assert csg.is_missing("null") is True

    def test_none_str(self):
        assert csg.is_missing("None") is True

    def test_na_slash(self):
        assert csg.is_missing("N/A") is True

    def test_question_mark(self):
        assert csg.is_missing("?") is True

    def test_value(self):
        assert csg.is_missing("3.14") is False

    def test_zero(self):
        assert csg.is_missing("0") is False

    def test_whitespace_na(self):
        assert csg.is_missing("  na  ") is True


class TestParseIgnoreCols:
    def test_empty(self):
        result = csg.parse_ignore_cols("", "y")
        assert result == {"y"}

    def test_single(self):
        result = csg.parse_ignore_cols("patient_id", "y")
        assert result == {"y", "patient_id"}

    def test_multiple(self):
        result = csg.parse_ignore_cols("patient_id,event_time", "y")
        assert result == {"y", "patient_id", "event_time"}

    def test_whitespace(self):
        result = csg.parse_ignore_cols("  a , b  ", "y")
        assert result == {"y", "a", "b"}

    def test_target_always_included(self):
        result = csg.parse_ignore_cols("", "target")
        assert "target" in result


class TestParseBinaryLabel:
    def test_zero(self):
        assert csg.parse_binary_label("0") == 0

    def test_one(self):
        assert csg.parse_binary_label("1") == 1

    def test_zero_float(self):
        assert csg.parse_binary_label("0.0") == 0

    def test_one_float(self):
        assert csg.parse_binary_label("1.0") == 1

    def test_empty(self):
        assert csg.parse_binary_label("") is None

    def test_two(self):
        assert csg.parse_binary_label("2") is None

    def test_negative(self):
        assert csg.parse_binary_label("-1") is None

    def test_text(self):
        assert csg.parse_binary_label("abc") is None

    def test_nan(self):
        assert csg.parse_binary_label("nan") is None

    def test_inf(self):
        assert csg.parse_binary_label("inf") is None


class TestTryParseFloat:
    def test_int(self):
        assert csg.try_parse_float("42") == 42.0

    def test_float(self):
        assert csg.try_parse_float("3.14") == 3.14

    def test_negative(self):
        assert csg.try_parse_float("-1.5") == -1.5

    def test_empty(self):
        assert csg.try_parse_float("") is None

    def test_text(self):
        assert csg.try_parse_float("abc") is None

    def test_nan(self):
        assert csg.try_parse_float("nan") is None

    def test_inf(self):
        assert csg.try_parse_float("inf") is None

    def test_whitespace(self):
        assert csg.try_parse_float("  3.14  ") == 3.14


class TestJsDivergence:
    def test_identical(self):
        result = csg.js_divergence([10, 20, 30], [10, 20, 30])
        assert result is not None
        assert result < 0.01

    def test_different(self):
        result = csg.js_divergence([100, 0, 0], [0, 0, 100])
        assert result is not None
        assert result > 0.5

    def test_empty_list(self):
        assert csg.js_divergence([], []) is None

    def test_mismatched_length(self):
        assert csg.js_divergence([1, 2], [1, 2, 3]) is None

    def test_zero_total(self):
        assert csg.js_divergence([0, 0], [1, 1]) is None

    def test_both_zero(self):
        assert csg.js_divergence([0, 0], [0, 0]) is None

    def test_symmetric(self):
        a = [10, 20, 30]
        b = [30, 20, 10]
        jsd_ab = csg.js_divergence(a, b)
        jsd_ba = csg.js_divergence(b, a)
        assert jsd_ab is not None and jsd_ba is not None
        assert abs(jsd_ab - jsd_ba) < 1e-10

    def test_bounded_0_1(self):
        result = csg.js_divergence([100, 0], [0, 100])
        assert result is not None
        assert 0.0 <= result <= 1.0


class TestAssignNumericBin:
    def test_low_end(self):
        assert csg.assign_numeric_bin(0.0, 0.0, 10.0, 10) == 0

    def test_high_end(self):
        assert csg.assign_numeric_bin(10.0, 0.0, 10.0, 10) == 9

    def test_middle(self):
        result = csg.assign_numeric_bin(5.0, 0.0, 10.0, 10)
        assert 0 <= result <= 9

    def test_below_low(self):
        assert csg.assign_numeric_bin(-1.0, 0.0, 10.0, 10) == 0

    def test_above_high(self):
        assert csg.assign_numeric_bin(15.0, 0.0, 10.0, 10) == 9

    def test_equal_range(self):
        assert csg.assign_numeric_bin(5.0, 5.0, 5.0, 10) == 0


class TestCategoricalBucket:
    def test_deterministic(self):
        b1 = csg.categorical_bucket("hello", 64)
        b2 = csg.categorical_bucket("hello", 64)
        assert b1 == b2

    def test_case_insensitive(self):
        b1 = csg.categorical_bucket("Hello", 64)
        b2 = csg.categorical_bucket("hello", 64)
        assert b1 == b2

    def test_in_range(self):
        for val in ["a", "b", "cat", "dog", "xyz"]:
            b = csg.categorical_bucket(val, 32)
            assert 0 <= b < 32


class TestSafeRatio:
    def test_normal(self):
        assert csg.safe_ratio(5, 10) == 0.5

    def test_zero_denominator(self):
        assert csg.safe_ratio(5, 0) is None

    def test_negative_denominator(self):
        assert csg.safe_ratio(5, -1) is None


class TestTopNFeatureSummary:
    def test_sorts_desc(self):
        rows = [
            {"feature": "a", "max_jsd": 0.1},
            {"feature": "b", "max_jsd": 0.3},
            {"feature": "c", "max_jsd": 0.2},
        ]
        result = csg.top_n_feature_summary(rows, "max_jsd", n=2)
        assert len(result) == 2
        assert result[0]["feature"] == "b"
        assert result[1]["feature"] == "c"

    def test_skips_none(self):
        rows = [
            {"feature": "a", "max_jsd": None},
            {"feature": "b", "max_jsd": 0.3},
        ]
        result = csg.top_n_feature_summary(rows, "max_jsd", n=10)
        assert len(result) == 1

    def test_empty(self):
        assert csg.top_n_feature_summary([], "max_jsd") == []


# ── discover_feature_types ───────────────────────────────────────────────────

class TestDiscoverFeatureTypes:
    def _make_csv(self, tmp_path: Path, name: str, header: str, rows: list) -> Path:
        p = tmp_path / name
        p.write_text(header + "\n" + "\n".join(rows) + "\n")
        return p

    def test_numeric_detection(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "train.csv", "y,age,bp",
            ["1,30,120", "0,45,130", "1,50,140"])
        features, types, disc = csg.discover_feature_types(str(p), "y", {"y"}, 0.98, 0)
        assert "age" in features
        assert types["age"]["type"] == "numeric"
        assert types["bp"]["type"] == "numeric"

    def test_categorical_detection(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "train.csv", "y,gender",
            ["1,M", "0,F", "1,M"])
        features, types, disc = csg.discover_feature_types(str(p), "y", {"y"}, 0.98, 0)
        assert types["gender"]["type"] == "categorical"

    def test_ignore_cols(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "train.csv", "y,pid,age",
            ["1,P1,30", "0,P2,45"])
        features, types, disc = csg.discover_feature_types(str(p), "y", {"y", "pid"}, 0.98, 0)
        assert "pid" not in features
        assert "age" in features

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            csg.discover_feature_types(str(tmp_path / "nope.csv"), "y", {"y"}, 0.98, 0)

    def test_max_rows_truncation(self, tmp_path: Path):
        rows = [f"{'10'[i%2]},{i},{i*10}" for i in range(100)]
        p = self._make_csv(tmp_path, "train.csv", "y,a,b", rows)
        features, types, disc = csg.discover_feature_types(str(p), "y", {"y"}, 0.98, 5)
        assert disc["sample_truncated"] is True
        assert disc["row_count_profiled"] == 5

    def test_no_features_after_ignore(self, tmp_path: Path):
        p = self._make_csv(tmp_path, "train.csv", "y,a", ["1,x", "0,y"])
        with pytest.raises(ValueError, match="No usable feature"):
            csg.discover_feature_types(str(p), "y", {"y", "a"}, 0.98, 0)


# ── CLI integration ──────────────────────────────────────────────────────────

def _make_split_csvs(tmp_path: Path, train_rows=None, test_rows=None, valid_rows=None):
    """Create train/test/(valid) CSVs with identical columns."""
    if train_rows is None:
        # Values drawn from same range so distributions overlap
        train_rows = [f"P{i:03d},2023-01-{(i%28)+1:02d},{'10'[i%2]},{30+(i%20)},{120+(i%20)}" for i in range(1, 31)]
    if test_rows is None:
        test_rows = [f"P{i:03d},2024-01-{(i%28)+1:02d},{'10'[i%2]},{30+(i%20)},{120+(i%20)}" for i in range(31, 51)]

    header = "patient_id,event_time,y,age,bp"
    (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(train_rows) + "\n")
    (tmp_path / "test.csv").write_text(header + "\n" + "\n".join(test_rows) + "\n")
    if valid_rows is not None:
        (tmp_path / "valid.csv").write_text(header + "\n" + "\n".join(valid_rows) + "\n")


def _run_gate(tmp_path: Path, extra_args: list = None, strict: bool = False, with_valid: bool = False) -> dict:
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--train", str(tmp_path / "train.csv"),
        "--test", str(tmp_path / "test.csv"),
        "--target-col", "y",
        "--ignore-cols", "patient_id,event_time",
        "--report", str(report_path),
    ]
    if with_valid:
        cmd.extend(["--valid", str(tmp_path / "valid.csv")])
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_similar_distributions_pass(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "top_shift_features" in report
        assert "top_missingness_shift_features" in report

    def test_summary_aggregates(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path)
        agg = report["summary"]["aggregates"]
        assert "feature_count" in agg
        assert "top_feature_jsd" in agg
        assert "mean_top10_jsd" in agg
        assert "high_shift_feature_count" in agg
        assert "high_shift_feature_fraction" in agg

    def test_with_valid_split(self, tmp_path: Path):
        valid_rows = [f"P{i:03d},2023-07-{(i%28)+1:02d},{'10'[i%2]},{30+(i%20)},{120+(i%20)}" for i in range(51, 61)]
        _make_split_csvs(tmp_path, valid_rows=valid_rows)
        report = _run_gate(tmp_path, with_valid=True)
        assert report["status"] == "pass"
        pairs = report["summary"]["pairs"]
        pair_names = [p["pair"] for p in pairs]
        assert "train_vs_valid" in pair_names
        assert "train_vs_test" in pair_names


class TestHighShift:
    def test_very_different_distributions_fail(self, tmp_path: Path):
        """Train has all low values, test has all high values → high JSD."""
        train_rows = [f"P{i:03d},2023-01-01,{'10'[i%2]},{i},{i}" for i in range(1, 51)]
        test_rows = [f"P{i:03d},2024-01-01,{'10'[i%2]},{i+1000},{i+1000}" for i in range(51, 71)]
        header = "patient_id,event_time,y,feat_a,feat_b"
        (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(train_rows) + "\n")
        (tmp_path / "test.csv").write_text(header + "\n" + "\n".join(test_rows) + "\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "top_feature_shift_too_high" in codes or "too_many_high_shift_features" in codes


class TestPrevalenceShift:
    def test_prevalence_hard_failure(self, tmp_path: Path):
        """Train prevalence ≈ 0.8, test prevalence ≈ 0.0 → large delta."""
        train_rows = [
            "P001,2023-01-01,1,30,120,M",
            "P002,2023-02-01,1,45,130,F",
            "P003,2023-03-01,1,50,140,M",
            "P004,2023-04-01,1,35,125,F",
            "P005,2023-05-01,0,60,150,M",
        ]
        test_rows = [
            "P006,2024-01-01,0,32,122,M",
            "P007,2024-02-01,0,47,132,F",
            "P008,2024-03-01,0,52,142,M",
            "P009,2024-04-01,0,55,145,F",
            "P010,2024-05-01,1,40,135,M",
        ]
        _make_split_csvs(tmp_path, train_rows=train_rows, test_rows=test_rows)
        report = _run_gate(tmp_path, extra_args=["--max-prevalence-delta", "0.20", "--warn-prevalence-delta", "0.10"])
        codes = [f["code"] for f in report["failures"]]
        assert "prevalence_shift_too_high" in codes

    def test_prevalence_warning(self, tmp_path: Path):
        """Train prevalence ≈ 0.6, test prevalence ≈ 0.33 → moderate delta."""
        train_rows = [
            "P001,2023-01-01,1,30,120,M",
            "P002,2023-02-01,1,45,130,F",
            "P003,2023-03-01,1,50,140,M",
            "P004,2023-04-01,0,35,125,F",
            "P005,2023-05-01,0,60,150,M",
        ]
        test_rows = [
            "P006,2024-01-01,1,32,122,M",
            "P007,2024-02-01,0,47,132,F",
            "P008,2024-03-01,0,52,142,M",
        ]
        _make_split_csvs(tmp_path, train_rows=train_rows, test_rows=test_rows)
        report = _run_gate(tmp_path, extra_args=["--max-prevalence-delta", "0.50", "--warn-prevalence-delta", "0.10"])
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "prevalence_shift_warning" in warn_codes


class TestMissingnessShift:
    def test_missing_ratio_delta_warning(self, tmp_path: Path):
        """Train has no missing, test has ~50% missing → warning."""
        header = "patient_id,event_time,y,feat_a"
        train_rows = [f"P{i:03d},2023-01-01,{'10'[i%2]},{i}" for i in range(1, 21)]
        # Half missing, half present so JSD can still be computed
        test_rows = [f"P{i:03d},2024-01-01,{'10'[i%2]},{'NA' if i%2==0 else str(i)}" for i in range(21, 41)]
        (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(train_rows) + "\n")
        (tmp_path / "test.csv").write_text(header + "\n" + "\n".join(test_rows) + "\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--max-missing-ratio-delta", "0.10",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "missingness_shift_exceeds_threshold" in warn_codes


class TestStrictMode:
    def test_strict_warnings_become_fail(self, tmp_path: Path):
        """With --strict, warnings cause overall fail."""
        header = "patient_id,event_time,y,feat_a"
        train_rows = [f"P{i:03d},2023-01-01,{'10'[i%2]},{i}" for i in range(1, 21)]
        test_rows = [f"P{i:03d},2024-01-01,{'10'[i%2]},NA" for i in range(21, 41)]
        (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(train_rows) + "\n")
        (tmp_path / "test.csv").write_text(header + "\n" + "\n".join(test_rows) + "\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--max-missing-ratio-delta", "0.10",
            "--strict",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["strict_mode"] is True
        if report["warning_count"] > 0:
            assert report["status"] == "fail"


class TestEdgeCases:
    def test_missing_train_file(self, tmp_path: Path):
        (tmp_path / "test.csv").write_text("y,a\n1,10\n0,20\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "nonexistent.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "feature_type_discovery_failed" in codes

    def test_empty_split(self, tmp_path: Path):
        header = "patient_id,event_time,y,feat_a"
        (tmp_path / "train.csv").write_text(header + "\nP001,2023-01-01,1,10\nP002,2023-02-01,0,20\n")
        (tmp_path / "test.csv").write_text(header + "\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "empty_split" in codes

    def test_invalid_labels_in_split(self, tmp_path: Path):
        header = "patient_id,event_time,y,feat_a"
        (tmp_path / "train.csv").write_text(header + "\nP001,2023-01-01,1,10\nP002,2023-02-01,0,20\n")
        (tmp_path / "test.csv").write_text(header + "\nP003,2024-01-01,abc,10\nP004,2024-02-01,0,20\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_labels" in codes

    def test_categorical_features(self, tmp_path: Path):
        header = "patient_id,event_time,y,color"
        train_rows = ["P001,2023-01-01,1,red", "P002,2023-02-01,0,blue", "P003,2023-03-01,1,green"]
        test_rows = ["P004,2024-01-01,1,red", "P005,2024-02-01,0,blue"]
        (tmp_path / "train.csv").write_text(header + "\n" + "\n".join(train_rows) + "\n")
        (tmp_path / "test.csv").write_text(header + "\n" + "\n".join(test_rows) + "\n")
        report_path = tmp_path / "report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--train", str(tmp_path / "train.csv"),
            "--test", str(tmp_path / "test.csv"),
            "--target-col", "y",
            "--ignore-cols", "patient_id,event_time",
            "--report", str(report_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        report = json.loads(report_path.read_text())
        assert report["status"] == "pass"

    def test_max_rows_per_split(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path, extra_args=["--max-rows-per-split", "2"])
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "split_profile_truncated" in warn_codes


class TestThresholdValidation:
    def test_negative_threshold(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path, extra_args=["--high-shift-jsd", "-0.1"])
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_threshold_range" in codes

    def test_threshold_above_one(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path, extra_args=["--max-top-feature-jsd", "1.5"])
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_threshold_range" in codes

    def test_prevalence_threshold_order(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path, extra_args=[
            "--warn-prevalence-delta", "0.50",
            "--max-prevalence-delta", "0.10",
        ])
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_prevalence_threshold_order" in codes

    def test_numeric_bins_too_low(self, tmp_path: Path):
        _make_split_csvs(tmp_path)
        report = _run_gate(tmp_path, extra_args=["--numeric-bins", "1"])
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_numeric_bins" in codes
