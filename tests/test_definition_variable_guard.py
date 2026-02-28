"""Comprehensive unit tests for scripts/definition_variable_guard.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from definition_variable_guard import (
    compile_patterns,
    list_from,
    norm,
    parse_comma_set,
    read_csv_header,
    resolve_target_block,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_csv(path: Path, headers: list, rows: list = None) -> Path:
    lines = [",".join(headers)]
    if rows:
        for row in rows:
            lines.append(",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_spec(path: Path, spec: dict) -> Path:
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def _make_test_setup(
    tmp_path: Path,
    spec: dict,
    train_headers: list = None,
    test_headers: list = None,
    valid_headers: list = None,
) -> dict:
    """Create spec file + CSVs for CLI tests."""
    if train_headers is None:
        train_headers = ["patient_id", "y", "age", "bp_systolic", "creatinine"]
    spec_path = _write_spec(tmp_path / "spec.json", spec)
    train_path = _write_csv(tmp_path / "train.csv", train_headers, [["P1", "0", "30", "120", "1.0"]])
    paths = {"spec": spec_path, "train": train_path}
    if test_headers is not None:
        paths["test"] = _write_csv(tmp_path / "test.csv", test_headers, [["P2", "1", "40", "130", "1.2"]])
    if valid_headers is not None:
        paths["valid"] = _write_csv(tmp_path / "valid.csv", valid_headers, [["P3", "0", "50", "110", "0.8"]])
    return paths


# ────────────────────────────────────────────────────────
# read_csv_header
# ────────────────────────────────────────────────────────

class TestReadCsvHeader:
    def test_normal(self, tmp_path: Path):
        p = _write_csv(tmp_path / "d.csv", ["a", "b", "c"])
        assert read_csv_header(str(p)) == ["a", "b", "c"]

    def test_whitespace_stripping(self, tmp_path: Path):
        p = tmp_path / "ws.csv"
        p.write_text(" a , b , c \n1,2,3\n", encoding="utf-8")
        assert read_csv_header(str(p)) == ["a", "b", "c"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_csv_header("/nonexistent/path.csv")

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing header"):
            read_csv_header(str(p))

    def test_header_only(self, tmp_path: Path):
        p = _write_csv(tmp_path / "h.csv", ["x", "y"])
        assert read_csv_header(str(p)) == ["x", "y"]


# ────────────────────────────────────────────────────────
# parse_comma_set
# ────────────────────────────────────────────────────────

class TestParseCommaSet:
    def test_normal(self):
        assert parse_comma_set("a,b,c") == {"a", "b", "c"}

    def test_empty(self):
        assert parse_comma_set("") == set()

    def test_whitespace(self):
        assert parse_comma_set(" a , b ") == {"a", "b"}

    def test_trailing_comma(self):
        assert parse_comma_set("x,") == {"x"}


# ────────────────────────────────────────────────────────
# norm
# ────────────────────────────────────────────────────────

class TestNorm:
    def test_lowercase(self):
        assert norm("Patient_ID") == "patientid"

    def test_remove_special(self):
        assert norm("bp-systolic!") == "bpsystolic"

    def test_keep_digits(self):
        assert norm("feature_123") == "feature123"

    def test_empty(self):
        assert norm("") == ""

    def test_only_special(self):
        assert norm("!@#$%") == ""

    def test_spaces(self):
        assert norm("  Age  ") == "age"


# ────────────────────────────────────────────────────────
# resolve_target_block
# ────────────────────────────────────────────────────────

class TestResolveTargetBlock:
    def test_exact_match(self):
        spec = {"targets": {"sepsis": {"defining_variables": ["lactate"]}}}
        result = resolve_target_block(spec, "sepsis")
        assert result == {"defining_variables": ["lactate"]}

    def test_case_insensitive(self):
        spec = {"targets": {"Sepsis": {"defining_variables": ["lactate"]}}}
        result = resolve_target_block(spec, "sepsis")
        assert result is not None
        assert result["defining_variables"] == ["lactate"]

    def test_not_found(self):
        spec = {"targets": {"sepsis": {"defining_variables": []}}}
        result = resolve_target_block(spec, "diabetes")
        assert result is None

    def test_no_targets_key(self):
        spec = {"other": "data"}
        assert resolve_target_block(spec, "sepsis") is None

    def test_targets_not_dict(self):
        spec = {"targets": "not_a_dict"}
        assert resolve_target_block(spec, "sepsis") is None

    def test_target_value_not_dict(self):
        spec = {"targets": {"sepsis": "not_a_dict"}}
        assert resolve_target_block(spec, "sepsis") is None

    def test_empty_targets(self):
        spec = {"targets": {}}
        assert resolve_target_block(spec, "sepsis") is None


# ────────────────────────────────────────────────────────
# list_from
# ────────────────────────────────────────────────────────

class TestListFrom:
    def test_normal(self):
        assert list_from({"vars": ["a", "b"]}, "vars") == ["a", "b"]

    def test_missing_key(self):
        assert list_from({}, "vars") == []

    def test_none_value(self):
        assert list_from({"vars": None}, "vars") == []

    def test_not_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            list_from({"vars": "not_a_list"}, "vars")

    def test_mixed_types(self):
        # Non-string items are silently skipped
        assert list_from({"vars": ["a", 123, None, "b"]}, "vars") == ["a", "b"]

    def test_whitespace_stripping(self):
        assert list_from({"vars": ["  a  ", "  b  "]}, "vars") == ["a", "b"]

    def test_empty_strings_skipped(self):
        assert list_from({"vars": ["", "  ", "a"]}, "vars") == ["a"]


# ────────────────────────────────────────────────────────
# compile_patterns
# ────────────────────────────────────────────────────────

class TestCompilePatterns:
    def test_valid_patterns(self):
        compiled, errors = compile_patterns(["^creat", "lactate.*"])
        assert len(compiled) == 2
        assert errors == []

    def test_invalid_pattern(self):
        compiled, errors = compile_patterns(["[invalid", "^valid$"])
        assert len(compiled) == 1  # Only valid one compiled
        assert len(errors) == 1
        assert "Invalid regex" in errors[0]

    def test_empty(self):
        compiled, errors = compile_patterns([])
        assert compiled == []
        assert errors == []

    def test_case_insensitive(self):
        compiled, errors = compile_patterns(["^CREAT"])
        assert compiled[0].search("creatinine") is not None


# ────────────────────────────────────────────────────────
# CLI tests (subprocess)
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, setup, target="sepsis", extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "definition_variable_guard.py"),
            "--target", target,
            "--definition-spec", str(setup["spec"]),
            "--train", str(setup["train"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if "test" in setup:
            cmd.extend(["--test", str(setup["test"])])
        if "valid" in setup:
            cmd.extend(["--valid", str(setup["valid"])])
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def test_no_leakage(self, tmp_path: Path):
        """No forbidden variables in features → pass."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["lactate", "sofa_score"],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_exact_leakage_detected(self, tmp_path: Path):
        """Defining variable 'creatinine' appears as feature → fail."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["creatinine"],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_variable_leakage" in codes

    def test_global_forbidden_variable(self, tmp_path: Path):
        """Global forbidden variable 'creatinine' → fail."""
        spec = {
            "targets": {"sepsis": {"defining_variables": []}},
            "global_forbidden_variables": ["creatinine"],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_variable_leakage" in codes

    def test_pattern_leakage_detected(self, tmp_path: Path):
        """Pattern '^creat' matches 'creatinine' → fail."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": [],
                    "forbidden_variables": [],
                    "forbidden_patterns": ["^creat"],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_proxy_leakage" in codes

    def test_global_pattern_leakage(self, tmp_path: Path):
        """Global pattern 'bp.*' matches 'bp_systolic' → fail."""
        spec = {
            "targets": {"sepsis": {"defining_variables": []}},
            "global_forbidden_variables": [],
            "global_forbidden_patterns": ["^bp"],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_proxy_leakage" in codes

    def test_case_insensitive_match(self, tmp_path: Path):
        """Defining var 'Creatinine' should match feature 'creatinine' (norm strips case)."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["Creatinine"],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2

    def test_ignore_cols(self, tmp_path: Path):
        """Ignored columns should not trigger leakage."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["creatinine"],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup,
                           extra_args=["--ignore-cols", "creatinine"])
        assert result.returncode == 0

    def test_target_col_auto_ignored(self, tmp_path: Path):
        """Target column 'y' should be automatically ignored."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["y"],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0  # y is target_col, auto-ignored

    def test_missing_spec_file(self, tmp_path: Path):
        """Non-existent spec file → fail."""
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {"spec": tmp_path / "nonexistent.json", "train": train_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_definition_spec" in codes

    def test_invalid_spec_json(self, tmp_path: Path):
        """Malformed JSON spec → fail."""
        spec_path = tmp_path / "bad.json"
        spec_path.write_text("{invalid json", encoding="utf-8")
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {"spec": spec_path, "train": train_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_definition_spec" in codes

    def test_spec_not_dict(self, tmp_path: Path):
        """Spec is a JSON list instead of object → fail."""
        spec_path = tmp_path / "list.json"
        spec_path.write_text("[1,2,3]", encoding="utf-8")
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {"spec": spec_path, "train": train_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_definition_spec" in codes

    def test_target_not_found_fails(self, tmp_path: Path):
        """Target missing in spec without --allow-missing-target → fail."""
        spec = {"targets": {"sepsis": {"defining_variables": []}}}
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup, target="diabetes")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "target_not_found" in codes

    def test_allow_missing_target(self, tmp_path: Path):
        """--allow-missing-target → pass even if target not in spec."""
        spec = {
            "targets": {"sepsis": {"defining_variables": []}},
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup, target="diabetes",
                           extra_args=["--allow-missing-target"])
        assert result.returncode == 0

    def test_invalid_regex_pattern(self, tmp_path: Path):
        """Invalid regex in forbidden_patterns → fail."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": [],
                    "forbidden_variables": [],
                    "forbidden_patterns": ["[invalid_regex"],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_forbidden_pattern" in codes

    def test_strict_empty_rules_fails(self, tmp_path: Path):
        """Strict mode with no forbidden rules → fail."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": [],
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup, extra_args=["--strict"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "empty_forbidden_rules" in codes

    def test_column_mismatch_warning(self, tmp_path: Path):
        """Different columns across splits → warning."""
        spec = {
            "targets": {"sepsis": {"defining_variables": []}},
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(
            tmp_path, spec,
            train_headers=["patient_id", "y", "age", "bp_systolic", "creatinine"],
            test_headers=["patient_id", "y", "age", "extra_col"],
        )
        result = self._run(tmp_path, setup)
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "column_mismatch" in warn_codes

    def test_train_csv_not_found(self, tmp_path: Path):
        """Non-existent train CSV → fail."""
        spec_path = _write_spec(tmp_path / "spec.json", {"targets": {}})
        setup = {"spec": spec_path, "train": tmp_path / "missing_train.csv"}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "input_error" in codes

    def test_report_structure(self, tmp_path: Path):
        """Verify report has all required fields."""
        spec = {
            "targets": {"sepsis": {"defining_variables": []}},
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "target" in report
        assert "definition_spec" in report
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "splits" in report["summary"]
        assert "forbidden_exact_count" in report["summary"]
        assert "forbidden_pattern_count" in report["summary"]
        assert "checked_feature_count" in report["summary"]
        assert "checked_features" in report["summary"]

    def test_field_type_not_list_fails(self, tmp_path: Path):
        """defining_variables as string instead of list → fail."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": "not_a_list",
                    "forbidden_variables": [],
                    "forbidden_patterns": [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_definition_spec" in codes

    def test_strict_warnings_become_failures(self, tmp_path: Path):
        """With --strict, column_mismatch warning should cause failure."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["lactate"],  # not in features
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(
            tmp_path, spec,
            train_headers=["patient_id", "y", "age"],
            test_headers=["patient_id", "y", "age", "extra"],
        )
        # Without strict: warning only, pass
        r1 = self._run(tmp_path, setup)
        assert r1.returncode == 0
        # With strict: warning → fail
        r2 = self._run(tmp_path, setup, extra_args=["--strict"])
        assert r2.returncode == 2

    def test_both_exact_and_pattern_hits(self, tmp_path: Path):
        """Both exact and pattern matches in same run."""
        spec = {
            "targets": {
                "sepsis": {
                    "defining_variables": ["age"],
                    "forbidden_variables": [],
                    "forbidden_patterns": ["^bp"],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }
        setup = _make_test_setup(tmp_path, spec)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "definition_variable_leakage" in codes
        assert "definition_proxy_leakage" in codes
