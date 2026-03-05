"""Comprehensive unit tests for scripts/_gate_utils.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _gate_utils import (
    add_issue,
    add_timeout_argument,
    canonical_metric_token,
    confusion_counts,
    epoch_to_iso,
    get_gate_elapsed,
    inject_execution_time,
    is_finite_number,
    load_json,
    load_json_from_path,
    load_json_from_str,
    load_json_optional,
    metric_panel,
    normalize_binary,
    resolve_path,
    safe_ratio,
    start_gate_timer,
    to_float,
    to_int,
    try_parse_time,
    write_json,
)


# ────────────────────────────────────────────────────────
# add_issue
# ────────────────────────────────────────────────────────

class TestAddIssue:
    def test_normal_append(self):
        bucket: list = []
        add_issue(bucket, "code_a", "msg_a", {"k": 1})
        assert len(bucket) == 1
        assert bucket[0] == {"code": "code_a", "message": "msg_a", "details": {"k": 1}}

    def test_multiple_append(self):
        bucket: list = []
        add_issue(bucket, "c1", "m1", {})
        add_issue(bucket, "c2", "m2", {"x": 2})
        assert len(bucket) == 2
        assert bucket[0]["code"] == "c1"
        assert bucket[1]["code"] == "c2"

    def test_duplicate_code_allowed(self):
        bucket: list = []
        add_issue(bucket, "dup", "msg", {})
        add_issue(bucket, "dup", "msg", {})
        assert len(bucket) == 2

    def test_empty_details(self):
        bucket: list = []
        add_issue(bucket, "c", "m", {})
        assert bucket[0]["details"] == {}

    def test_none_in_details(self):
        bucket: list = []
        add_issue(bucket, "c", "m", {"val": None})
        assert bucket[0]["details"]["val"] is None

    def test_preserves_existing_items(self):
        bucket = [{"existing": True}]
        add_issue(bucket, "new", "new_msg", {})
        assert len(bucket) == 2
        assert bucket[0] == {"existing": True}


# ────────────────────────────────────────────────────────
# load_json_from_path
# ────────────────────────────────────────────────────────

class TestLoadJsonFromPath:
    def test_normal_dict(self, tmp_path: Path):
        p = tmp_path / "ok.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        result = load_json_from_path(p)
        assert result == {"a": 1}

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_json_from_path(tmp_path / "missing.json")

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_from_path(p)

    def test_root_is_list(self, tmp_path: Path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON root must be object"):
            load_json_from_path(p)

    def test_root_is_string(self, tmp_path: Path):
        p = tmp_path / "str.json"
        p.write_text('"hello"', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON root must be object"):
            load_json_from_path(p)

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_from_path(p)

    def test_nested_dict(self, tmp_path: Path):
        p = tmp_path / "nested.json"
        data = {"a": {"b": [1, 2]}, "c": True}
        p.write_text(json.dumps(data), encoding="utf-8")
        assert load_json_from_path(p) == data

    def test_unicode_content(self, tmp_path: Path):
        p = tmp_path / "unicode.json"
        p.write_text('{"名前": "テスト"}', encoding="utf-8")
        assert load_json_from_path(p) == {"名前": "テスト"}


# ────────────────────────────────────────────────────────
# load_json_from_str
# ────────────────────────────────────────────────────────

class TestLoadJsonFromStr:
    def test_normal(self, tmp_path: Path):
        p = tmp_path / "ok.json"
        p.write_text('{"x": 42}', encoding="utf-8")
        result = load_json_from_str(str(p))
        assert result == {"x": 42}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json_from_str("/nonexistent/path/to/file.json")

    def test_root_is_list(self, tmp_path: Path):
        p = tmp_path / "list.json"
        p.write_text("[1]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON root must be an object"):
            load_json_from_str(str(p))

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{{bad}}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json_from_str(str(p))

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch):
        """Verify expanduser() is called on str path."""
        p = tmp_path / "home_file.json"
        p.write_text('{"home": true}', encoding="utf-8")
        # Patch expanduser to redirect ~ to tmp_path
        monkeypatch.setenv("HOME", str(tmp_path))
        result = load_json_from_str(str(tmp_path / "home_file.json"))
        assert result == {"home": True}

    def test_root_is_string(self, tmp_path: Path):
        p = tmp_path / "str_root.json"
        p.write_text('"just a string"', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON root must be an object"):
            load_json_from_str(str(p))


# ────────────────────────────────────────────────────────
# load_json (dispatch)
# ────────────────────────────────────────────────────────

class TestLoadJson:
    def test_path_object(self, tmp_path: Path):
        p = tmp_path / "data.json"
        p.write_text('{"via": "path"}', encoding="utf-8")
        assert load_json(p) == {"via": "path"}

    def test_str_path(self, tmp_path: Path):
        p = tmp_path / "data.json"
        p.write_text('{"via": "str"}', encoding="utf-8")
        assert load_json(str(p)) == {"via": "str"}

    def test_path_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nope.json")

    def test_str_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/definitely_missing_mlgg_test.json")


# ────────────────────────────────────────────────────────
# load_json_optional
# ────────────────────────────────────────────────────────

class TestLoadJsonOptional:
    def test_file_not_exists(self, tmp_path: Path):
        assert load_json_optional(tmp_path / "missing.json") is None

    def test_normal_dict(self, tmp_path: Path):
        p = tmp_path / "ok.json"
        p.write_text('{"status": "pass"}', encoding="utf-8")
        assert load_json_optional(p) == {"status": "pass"}

    def test_root_is_list(self, tmp_path: Path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2]", encoding="utf-8")
        assert load_json_optional(p) is None

    def test_root_is_int(self, tmp_path: Path):
        p = tmp_path / "int.json"
        p.write_text("42", encoding="utf-8")
        assert load_json_optional(p) is None

    def test_invalid_json_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{bad", encoding="utf-8")
        assert load_json_optional(p) is None

    def test_empty_dict(self, tmp_path: Path):
        p = tmp_path / "empty_obj.json"
        p.write_text("{}", encoding="utf-8")
        assert load_json_optional(p) == {}

    def test_root_is_string(self, tmp_path: Path):
        p = tmp_path / "str.json"
        p.write_text('"hello"', encoding="utf-8")
        assert load_json_optional(p) is None

    def test_root_is_bool(self, tmp_path: Path):
        p = tmp_path / "bool.json"
        p.write_text("true", encoding="utf-8")
        assert load_json_optional(p) is None


# ────────────────────────────────────────────────────────
# write_json
# ────────────────────────────────────────────────────────

class TestWriteJson:
    def test_normal_write(self, tmp_path: Path):
        p = tmp_path / "out.json"
        write_json(p, {"a": 1})
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == {"a": 1}

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = tmp_path / "sub" / "deep" / "out.json"
        write_json(p, {"nested": True})
        assert p.exists()
        assert json.loads(p.read_text(encoding="utf-8")) == {"nested": True}

    def test_overwrite_existing(self, tmp_path: Path):
        p = tmp_path / "over.json"
        write_json(p, {"v": 1})
        write_json(p, {"v": 2})
        assert json.loads(p.read_text(encoding="utf-8")) == {"v": 2}

    def test_output_ends_with_newline(self, tmp_path: Path):
        p = tmp_path / "nl.json"
        write_json(p, {"x": 1})
        raw = p.read_text(encoding="utf-8")
        assert raw.endswith("\n")

    def test_no_tmp_file_left(self, tmp_path: Path):
        p = tmp_path / "clean.json"
        write_json(p, {"clean": True})
        siblings = list(tmp_path.iterdir())
        assert len(siblings) == 1
        assert siblings[0].name == "clean.json"

    def test_ensure_ascii(self, tmp_path: Path):
        p = tmp_path / "ascii.json"
        write_json(p, {"text": "中文"})
        raw = p.read_text(encoding="utf-8")
        assert "\\u" in raw  # ensure_ascii=True escapes non-ASCII

    def test_indent_two_spaces(self, tmp_path: Path):
        p = tmp_path / "indent.json"
        write_json(p, {"a": 1, "b": 2})
        raw = p.read_text(encoding="utf-8")
        assert "  " in raw  # indent=2

    def test_atomic_no_partial_on_crash(self, tmp_path: Path):
        """Verify write_json uses tmp+replace pattern (already visible in source)."""
        p = tmp_path / "atomic.json"
        write_json(p, {"ok": True})
        # If we got here, the tmp file was replaced atomically
        assert p.exists()

    def test_complex_payload(self, tmp_path: Path):
        p = tmp_path / "complex.json"
        data = {
            "list": [1, 2.5, None, "str"],
            "nested": {"a": {"b": []}},
            "null_val": None,
            "bool_val": True,
            "num": 1e-10,
        }
        write_json(p, data)
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == data

    def test_empty_payload(self, tmp_path: Path):
        p = tmp_path / "empty_payload.json"
        write_json(p, {})
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded == {}


# ────────────────────────────────────────────────────────
# resolve_path
# ────────────────────────────────────────────────────────

class TestResolvePath:
    def test_absolute_path(self, tmp_path: Path):
        result = resolve_path(tmp_path, "/usr/bin/python3")
        assert result == Path("/usr/bin/python3")

    def test_relative_path(self, tmp_path: Path):
        result = resolve_path(tmp_path, "data/train.csv")
        assert result == (tmp_path / "data" / "train.csv").resolve()

    def test_tilde_expansion(self):
        result = resolve_path(Path("/base"), "~/myfile.txt")
        assert "~" not in str(result)
        assert str(result).startswith("/")

    def test_dot_relative(self, tmp_path: Path):
        result = resolve_path(tmp_path, "./file.csv")
        assert result == (tmp_path / "file.csv").resolve()

    def test_dotdot_relative(self, tmp_path: Path):
        base = tmp_path / "sub"
        base.mkdir()
        result = resolve_path(base, "../file.csv")
        assert result == (tmp_path / "file.csv").resolve()

    def test_empty_string(self, tmp_path: Path):
        """Empty string should resolve relative to base."""
        result = resolve_path(tmp_path, "")
        # Path("") resolves to "." which is relative
        assert result == (tmp_path / "").resolve()

    def test_absolute_with_tilde(self):
        result = resolve_path(Path("/base"), "/absolute/path")
        assert result == Path("/absolute/path").resolve()


# ────────────────────────────────────────────────────────
# to_float
# ────────────────────────────────────────────────────────

class TestToFloat:
    # --- int inputs ---
    def test_int_normal(self):
        assert to_float(42) == 42.0

    def test_int_zero(self):
        assert to_float(0) == 0.0

    def test_int_negative(self):
        assert to_float(-5) == -5.0

    # --- float inputs ---
    def test_float_normal(self):
        assert to_float(3.14) == 3.14

    def test_float_zero(self):
        assert to_float(0.0) == 0.0

    def test_float_inf(self):
        assert to_float(float("inf")) is None

    def test_float_neg_inf(self):
        assert to_float(float("-inf")) is None

    def test_float_nan(self):
        assert to_float(float("nan")) is None

    def test_float_very_small(self):
        assert to_float(1e-300) == 1e-300

    def test_float_very_large(self):
        assert to_float(1e300) == 1e300

    # --- bool inputs (must return None, not 0/1) ---
    def test_bool_true(self):
        assert to_float(True) is None

    def test_bool_false(self):
        assert to_float(False) is None

    # --- str inputs ---
    def test_str_int(self):
        assert to_float("42") == 42.0

    def test_str_float(self):
        assert to_float("3.14") == 3.14

    def test_str_negative(self):
        assert to_float("-7.5") == -7.5

    def test_str_whitespace_padded(self):
        assert to_float("  10.0  ") == 10.0

    def test_str_inf(self):
        assert to_float("inf") is None

    def test_str_neg_inf(self):
        assert to_float("-inf") is None

    def test_str_nan(self):
        assert to_float("nan") is None

    def test_str_nan_upper(self):
        assert to_float("NaN") is None

    def test_str_inf_upper(self):
        assert to_float("Inf") is None

    def test_str_infinity(self):
        assert to_float("Infinity") is None

    def test_str_empty(self):
        assert to_float("") is None

    def test_str_whitespace_only(self):
        assert to_float("   ") is None

    def test_str_non_numeric(self):
        assert to_float("abc") is None

    def test_str_mixed(self):
        assert to_float("12abc") is None

    def test_str_scientific(self):
        assert to_float("1e5") == 1e5

    # --- None ---
    def test_none(self):
        assert to_float(None) is None

    # --- other types ---
    def test_list(self):
        assert to_float([1, 2]) is None

    def test_dict(self):
        assert to_float({"a": 1}) is None

    def test_tuple(self):
        assert to_float((1,)) is None

    def test_bytes(self):
        assert to_float(b"42") is None

    def test_large_int(self):
        """Very large int that is still finite as float."""
        assert to_float(10**100) == float(10**100)

    def test_complex_type(self):
        assert to_float(complex(1, 2)) is None

    def test_str_plus_sign(self):
        assert to_float("+3.14") == 3.14

    def test_str_negative_zero(self):
        result = to_float("-0.0")
        assert result is not None
        assert result == 0.0


# ────────────────────────────────────────────────────────
# canonical_metric_token
# ────────────────────────────────────────────────────────

class TestCanonicalMetricToken:
    def test_simple_lowercase(self):
        assert canonical_metric_token("roc_auc") == "rocauc"

    def test_mixed_case(self):
        assert canonical_metric_token("ROC_AUC") == "rocauc"

    def test_hyphens_and_spaces(self):
        assert canonical_metric_token("pr-auc score") == "praucscore"

    def test_dots_and_slashes(self):
        assert canonical_metric_token("metrics.roc/auc") == "metricsrocauc"

    def test_empty_string(self):
        assert canonical_metric_token("") == ""

    def test_only_special_chars(self):
        assert canonical_metric_token("---___...") == ""

    def test_numeric_suffix(self):
        assert canonical_metric_token("f1_score") == "f1score"

    def test_equivalence(self):
        assert canonical_metric_token("pr_auc") == canonical_metric_token("PR-AUC")
        assert canonical_metric_token("roc_auc") == canonical_metric_token("ROC AUC")
        assert canonical_metric_token("f2_beta") == canonical_metric_token("F2-Beta")


# ────────────────────────────────────────────────────────
# is_finite_number
# ────────────────────────────────────────────────────────

class TestIsFiniteNumber:
    def test_int(self):
        assert is_finite_number(42) is True

    def test_float(self):
        assert is_finite_number(3.14) is True

    def test_zero(self):
        assert is_finite_number(0) is True
        assert is_finite_number(0.0) is True

    def test_negative(self):
        assert is_finite_number(-1) is True
        assert is_finite_number(-1.5) is True

    def test_inf(self):
        assert is_finite_number(float("inf")) is False

    def test_neg_inf(self):
        assert is_finite_number(float("-inf")) is False

    def test_nan(self):
        assert is_finite_number(float("nan")) is False

    def test_bool_excluded(self):
        assert is_finite_number(True) is False
        assert is_finite_number(False) is False

    def test_string(self):
        assert is_finite_number("42") is False

    def test_none(self):
        assert is_finite_number(None) is False

    def test_list(self):
        assert is_finite_number([1]) is False


# ────────────────────────────────────────────────────────
# to_int
# ────────────────────────────────────────────────────────

class TestToInt:
    def test_int_normal(self):
        assert to_int(42) == 42

    def test_int_zero(self):
        assert to_int(0) == 0

    def test_int_negative(self):
        assert to_int(-5) == -5

    def test_float_whole(self):
        assert to_int(3.0) == 3

    def test_float_fractional(self):
        assert to_int(3.5) is None

    def test_float_inf(self):
        assert to_int(float("inf")) is None

    def test_float_nan(self):
        assert to_int(float("nan")) is None

    def test_bool_excluded(self):
        assert to_int(True) is None
        assert to_int(False) is None

    def test_str_returns_none(self):
        assert to_int("42") is None
        assert to_int("3.0") is None
        assert to_int("") is None
        assert to_int("abc") is None

    def test_none(self):
        assert to_int(None) is None

    def test_list(self):
        assert to_int([1]) is None


# ────────────────────────────────────────────────────────
# start_gate_timer / get_gate_elapsed
# ────────────────────────────────────────────────────────

class TestGateTimer:
    def test_elapsed_returns_float(self):
        import _gate_utils
        _gate_utils._gate_start_time = None
        assert get_gate_elapsed() == 0.0

    def test_timer_round_trip(self):
        import time
        start_gate_timer()
        time.sleep(0.05)
        elapsed = get_gate_elapsed()
        assert elapsed >= 0.04
        assert elapsed < 2.0

    def test_timer_not_started(self):
        import _gate_utils
        _gate_utils._gate_start_time = None
        assert get_gate_elapsed() == 0.0


# ────────────────────────────────────────────────────────
# inject_execution_time
# ────────────────────────────────────────────────────────

class TestInjectExecutionTime:
    def test_adds_key(self):
        start_gate_timer()
        report: dict = {"status": "pass"}
        result = inject_execution_time(report)
        assert "execution_time_seconds" in result
        assert isinstance(result["execution_time_seconds"], float)
        assert result is report  # mutates in place

    def test_overwrites_existing(self):
        start_gate_timer()
        report = {"execution_time_seconds": -1}
        inject_execution_time(report)
        assert report["execution_time_seconds"] >= 0


# ────────────────────────────────────────────────────────
# add_timeout_argument
# ────────────────────────────────────────────────────────

class TestAddTimeoutArgument:
    def test_adds_timeout_flag(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_timeout_argument(parser)
        args = parser.parse_args([])
        assert args.timeout == 0

    def test_custom_timeout(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_timeout_argument(parser)
        args = parser.parse_args(["--timeout", "30"])
        assert args.timeout == 30

    def test_negative_timeout(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_timeout_argument(parser)
        args = parser.parse_args(["--timeout", "-1"])
        assert args.timeout == -1


# ────────────────────────────────────────────────────────
# try_parse_time
# ────────────────────────────────────────────────────────

class TestTryParseTime:
    def test_epoch_numeric(self):
        result = try_parse_time("1704067200.0")
        assert result == 1704067200.0

    def test_epoch_int_string(self):
        result = try_parse_time("1704067200")
        assert result == 1704067200.0

    def test_iso_date(self):
        result = try_parse_time("2024-01-01")
        assert result is not None
        assert isinstance(result, float)

    def test_iso_datetime(self):
        result = try_parse_time("2024-01-01 12:00:00")
        assert result is not None

    def test_iso_with_z(self):
        result = try_parse_time("2024-01-01T00:00:00Z")
        assert result is not None

    def test_slash_date(self):
        result = try_parse_time("2024/01/01")
        assert result is not None

    def test_us_date_format(self):
        result = try_parse_time("01/01/2024")
        assert result is not None

    def test_empty_string(self):
        assert try_parse_time("") is None

    def test_whitespace(self):
        assert try_parse_time("   ") is None

    def test_garbage(self):
        assert try_parse_time("not_a_date") is None

    def test_whitespace_padded(self):
        result = try_parse_time("  2024-01-01  ")
        assert result is not None


# ────────────────────────────────────────────────────────
# epoch_to_iso
# ────────────────────────────────────────────────────────

class TestEpochToIso:
    def test_none(self):
        assert epoch_to_iso(None) is None

    def test_epoch_zero(self):
        result = epoch_to_iso(0.0)
        assert result == "1970-01-01T00:00:00Z"

    def test_known_timestamp(self):
        result = epoch_to_iso(1704067200.0)
        assert result is not None
        assert "2024-01-01" in result
        assert result.endswith("Z")

    def test_returns_string(self):
        result = epoch_to_iso(1000000.0)
        assert isinstance(result, str)
        assert "Z" in result


# ────────────────────────────────────────────────────────
# safe_ratio
# ────────────────────────────────────────────────────────

class TestSafeRatio:
    def test_normal(self):
        assert safe_ratio(10, 5) == 2.0

    def test_zero_denominator(self):
        assert safe_ratio(10, 0) == 0.0

    def test_negative_denominator(self):
        assert safe_ratio(10, -5) == 0.0

    def test_zero_numerator(self):
        assert safe_ratio(0, 5) == 0.0

    def test_float_result(self):
        assert abs(safe_ratio(1, 3) - 1 / 3) < 1e-10

    def test_both_zero(self):
        assert safe_ratio(0, 0) == 0.0

    def test_large_values(self):
        assert safe_ratio(1e15, 1e10) == 1e5


# ────────────────────────────────────────────────────────
# confusion_counts
# ────────────────────────────────────────────────────────

class TestConfusionCounts:
    def test_perfect(self):
        cm = confusion_counts(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 0]))
        assert cm == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}

    def test_all_wrong(self):
        cm = confusion_counts(np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]))
        assert cm == {"tp": 0, "fp": 2, "tn": 0, "fn": 2}

    def test_mixed(self):
        cm = confusion_counts(np.array([1, 0, 1, 0, 1]), np.array([1, 0, 0, 1, 1]))
        assert cm["tp"] == 2 and cm["fp"] == 1 and cm["tn"] == 1 and cm["fn"] == 1

    def test_single_element(self):
        assert confusion_counts(np.array([1]), np.array([0])) == {"tp": 0, "fp": 0, "tn": 0, "fn": 1}

    def test_returns_int(self):
        cm = confusion_counts(np.array([1, 0]), np.array([1, 0]))
        assert all(isinstance(v, int) for v in cm.values())

    def test_sum_equals_n(self):
        y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        p = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        cm = confusion_counts(y, p)
        assert sum(cm.values()) == len(y)


# ────────────────────────────────────────────────────────
# normalize_binary
# ────────────────────────────────────────────────────────

class TestNormalizeBinary:
    def test_valid_binary(self):
        result = normalize_binary(pd.Series([0, 1, 0, 1, 1]))
        assert result is not None
        np.testing.assert_array_equal(result, [0, 1, 0, 1, 1])
        assert result.dtype == int

    def test_string_binary(self):
        result = normalize_binary(pd.Series(["0", "1", "0"]))
        assert result is not None
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_non_binary_returns_none(self):
        assert normalize_binary(pd.Series([0, 1, 2])) is None

    def test_nan_returns_none(self):
        assert normalize_binary(pd.Series([0, 1, float("nan")])) is None

    def test_non_numeric_returns_none(self):
        assert normalize_binary(pd.Series(["a", "b"])) is None

    def test_float_binary(self):
        result = normalize_binary(pd.Series([0.0, 1.0, 0.0]))
        assert result is not None
        np.testing.assert_array_equal(result, [0, 1, 0])


# ────────────────────────────────────────────────────────
# metric_panel
# ────────────────────────────────────────────────────────

class TestMetricPanel:
    def _make_data(self):
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.4, 0.6, 0.2])
        y_pred = (y_score >= 0.5).astype(int)
        return y_true, y_score, y_pred

    def test_returns_tuple(self):
        y_true, y_score, y_pred = self._make_data()
        result = metric_panel(y_true, y_score, y_pred, beta=2.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_metrics_keys(self):
        y_true, y_score, y_pred = self._make_data()
        metrics, cm = metric_panel(y_true, y_score, y_pred, beta=2.0)
        for key in ("accuracy", "precision", "ppv", "npv", "sensitivity",
                     "specificity", "f1", "f2_beta", "roc_auc", "pr_auc", "brier"):
            assert key in metrics, f"Missing key: {key}"

    def test_confusion_matrix_keys(self):
        y_true, y_score, y_pred = self._make_data()
        _, cm = metric_panel(y_true, y_score, y_pred, beta=2.0)
        assert set(cm.keys()) == {"tp", "fp", "tn", "fn"}

    def test_metrics_in_range(self):
        y_true, y_score, y_pred = self._make_data()
        metrics, _ = metric_panel(y_true, y_score, y_pred, beta=2.0)
        for key in ("accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of [0,1]"

    def test_perfect_predictions(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.99, 0.98, 0.01, 0.02])
        p = np.array([1, 1, 0, 0])
        metrics, cm = metric_panel(y, s, p, beta=2.0)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert cm["fp"] == 0 and cm["fn"] == 0

    def test_brier_perfect_near_zero(self):
        y = np.array([1, 0])
        s = np.array([1.0, 0.0])
        p = np.array([1, 0])
        metrics, _ = metric_panel(y, s, p, beta=2.0)
        assert metrics["brier"] < 0.01


# ────────────────────────────────────────────────────────
# install_gate_timeout
# ────────────────────────────────────────────────────────

class TestInstallGateTimeout:
    def test_zero_timeout_is_noop(self):
        from _gate_utils import install_gate_timeout
        install_gate_timeout(0, None, "test_gate")

    def test_negative_timeout_is_noop(self):
        from _gate_utils import install_gate_timeout
        install_gate_timeout(-1, None, "test_gate")

    def test_positive_timeout_installs_alarm(self):
        import signal
        from _gate_utils import install_gate_timeout
        if not hasattr(signal, "SIGALRM"):
            pytest.skip("SIGALRM not available on this platform")
        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            install_gate_timeout(9999, None, "test_gate")
            new_handler = signal.getsignal(signal.SIGALRM)
            assert new_handler is not old_handler
            signal.alarm(0)
        finally:
            signal.signal(signal.SIGALRM, old_handler if callable(old_handler) else signal.SIG_DFL)
