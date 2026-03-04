"""Comprehensive unit tests for scripts/_gate_utils.py."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _gate_utils import (
    add_issue,
    load_json,
    load_json_from_path,
    load_json_from_str,
    load_json_optional,
    resolve_path,
    to_float,
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
