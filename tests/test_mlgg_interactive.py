"""Tests for scripts/mlgg_interactive.py helper functions and CLI.

Covers normalize_path, infer_project_base_from_request, read_csv_columns,
validate_binary_target, validate_profile_name, merged_seed, require_string,
parse_command_overrides, build_command, profile save/load, CLI --help,
--print-only, --accept-defaults.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "mlgg_interactive.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import mlgg_interactive as mi


# ── pure helpers ─────────────────────────────────────────────────────────────

class TestNormalizePath:
    def test_relative(self):
        result = mi.normalize_path("foo/bar")
        assert Path(result).is_absolute()

    def test_tilde(self, tmp_path):
        result = mi.normalize_path("~/somefile")
        assert "~" not in result


class TestInferProjectBaseFromRequest:
    def test_configs(self, tmp_path):
        configs = tmp_path / "proj" / "configs"
        configs.mkdir(parents=True)
        req = configs / "request.json"
        req.write_text("{}")
        assert mi.infer_project_base_from_request(str(req)) == tmp_path / "proj"

    def test_flat(self, tmp_path):
        req = tmp_path / "request.json"
        req.write_text("{}")
        assert mi.infer_project_base_from_request(str(req)) == tmp_path


class TestInferProjectBaseFromSplitPath:
    def test_data(self, tmp_path):
        data = tmp_path / "proj" / "data"
        data.mkdir(parents=True)
        split = data / "train.csv"
        split.write_text("a,b\n1,2")
        assert mi.infer_project_base_from_split_path(str(split)) == tmp_path / "proj"

    def test_flat(self, tmp_path):
        split = tmp_path / "train.csv"
        split.write_text("a,b\n1,2")
        assert mi.infer_project_base_from_split_path(str(split)) == tmp_path


class TestReadCsvColumns:
    def test_basic(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("col_a,col_b,col_c\n1,2,3\n")
        cols = mi.read_csv_columns(str(p))
        assert cols == ["col_a", "col_b", "col_c"]

    def test_empty(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        cols = mi.read_csv_columns(str(p))
        assert cols == []


class TestValidateBinaryTarget:
    def test_valid(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("y,x\n0,1\n1,2\n0,3\n")
        ok, msg = mi.validate_binary_target(str(p), "y")
        assert ok is True

    def test_non_binary(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("y,x\n0,1\n2,2\n1,3\n")
        ok, msg = mi.validate_binary_target(str(p), "y")
        assert ok is False
        assert "non-binary" in msg

    def test_missing_col(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n")
        ok, msg = mi.validate_binary_target(str(p), "y")
        assert ok is False
        assert "not found" in msg


class TestValidateProfileName:
    def test_valid(self):
        assert mi.validate_profile_name("my-profile_1") == "my-profile_1"

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            mi.validate_profile_name("")

    def test_invalid_chars(self):
        with pytest.raises(ValueError, match="must match"):
            mi.validate_profile_name("foo bar!")


class TestMergedSeed:
    def test_cli_priority(self):
        val, src = mi.merged_seed("k", "default", {"k": "profile"}, {"k": "cli"})
        assert val == "cli"
        assert src == "cli"

    def test_profile_priority(self):
        val, src = mi.merged_seed("k", "default", {"k": "profile"}, {})
        assert val == "profile"
        assert src == "profile"

    def test_default(self):
        val, src = mi.merged_seed("k", "default", {}, {})
        assert val == "default"
        assert src == "default"


class TestRequireString:
    def test_ok(self):
        assert mi.require_string("hello", "field") == "hello"

    def test_empty(self):
        with pytest.raises(ValueError, match="required"):
            mi.require_string("", "field")

    def test_whitespace(self):
        with pytest.raises(ValueError, match="required"):
            mi.require_string("   ", "field")


class TestParseCommandOverrides:
    def test_init(self):
        result = mi.parse_command_overrides("init", ["--project-root", "/tmp/proj"])
        assert result["project_root"] == "/tmp/proj"

    def test_workflow(self):
        result = mi.parse_command_overrides("workflow", ["--request", "/tmp/req.json"])
        assert result["request"] == "/tmp/req.json"

    def test_unknown_args(self):
        with pytest.raises(ValueError, match="Unknown"):
            mi.parse_command_overrides("init", ["--nonexistent", "val"])

    def test_train(self):
        result = mi.parse_command_overrides("train", ["--target-col", "y", "--n-jobs", "4"])
        assert result["target_col"] == "y"
        assert result["n_jobs"] == 4


class TestBuildCommand:
    def test_init(self):
        values = {
            "project_root": "/tmp/proj",
            "study_id": "s1",
            "target_name": "disease",
            "label_col": "y",
            "patient_id_col": "pid",
            "index_time_col": "time",
            "force": True,
        }
        cmd = mi.build_command("init", sys.executable, values)
        assert "--project-root" in cmd
        assert "--force" in cmd
        assert "/tmp/proj" in cmd

    def test_workflow_strict(self):
        values = {
            "request": "/tmp/req.json",
            "evidence_dir": "/tmp/evidence",
            "compare_manifest": "",
            "allow_missing_compare": True,
            "continue_on_fail": False,
        }
        cmd = mi.build_command("workflow", sys.executable, values)
        assert "--strict" in cmd
        assert "--allow-missing-compare" in cmd

    def test_train(self):
        values = {
            "train": "/tmp/train.csv",
            "valid": "/tmp/valid.csv",
            "test": "/tmp/test.csv",
            "target_col": "y",
            "patient_id_col": "pid",
            "ignore_cols": "pid,time",
            "model_pool": "logistic_l1,logistic_l2",
            "include_optional_models": False,
            "ensemble_top_k": 0,
            "n_jobs": 1,
            "calibration_method": "none",
            "feature_group_spec": "",
            "external_cohort_spec": "",
            "model_selection_report_out": "/tmp/ms.json",
            "evaluation_report_out": "/tmp/eval.json",
            "prediction_trace_out": "/tmp/trace.csv.gz",
            "external_validation_report_out": "",
            "ci_matrix_report_out": "/tmp/ci.json",
            "distribution_report_out": "/tmp/dist.json",
            "feature_engineering_report_out": "",
            "robustness_report_out": "/tmp/rob.json",
            "seed_sensitivity_out": "/tmp/seed.json",
        }
        cmd = mi.build_command("train", sys.executable, values)
        assert "--train" in cmd
        assert "--model-pool" in cmd


class TestProfileSaveLoad:
    def test_roundtrip(self, tmp_path):
        profile_path = tmp_path / "test_profile.json"
        values = {
            "project_root": "/tmp/proj",
            "study_id": "s1",
            "target_name": "disease",
            "label_col": "y",
            "patient_id_col": "pid",
            "index_time_col": "time",
            "force": True,
        }
        mi.save_profile(profile_path, "init", values, sys.executable, "/tmp")
        assert profile_path.exists()
        loaded = mi.load_profile(profile_path, "init")
        assert loaded["project_root"] == "/tmp/proj"
        assert loaded["force"] is True

    def test_command_mismatch(self, tmp_path):
        profile_path = tmp_path / "prof.json"
        mi.save_profile(profile_path, "init", {"project_root": "/tmp"}, sys.executable, "/tmp")
        with pytest.raises(ValueError, match="mismatch"):
            mi.load_profile(profile_path, "workflow")


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "command" in proc.stdout.lower()


class TestCLIPrintOnly:
    def test_init_print_only(self, tmp_path):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--command", "init",
            "--print-only",
            "--accept-defaults",
            "--", "--project-root", str(tmp_path / "proj"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "print-only" in proc.stdout.lower() or "not executed" in proc.stdout.lower()

    def test_authority_print_only(self, tmp_path):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--command", "authority",
            "--print-only",
            "--accept-defaults",
            "--", "--summary-file", str(tmp_path / "summary.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0


class TestCLISaveLoadProfile:
    def test_save_and_load(self, tmp_path):
        profile_dir = tmp_path / "profiles"
        cmd_save = [
            sys.executable, str(GATE_SCRIPT),
            "--command", "init",
            "--print-only",
            "--accept-defaults",
            "--save-profile",
            "--profile-name", "test1",
            "--profile-dir", str(profile_dir),
            "--", "--project-root", str(tmp_path / "proj"),
        ]
        proc = subprocess.run(cmd_save, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert (profile_dir / "test1.json").exists()

        cmd_load = [
            sys.executable, str(GATE_SCRIPT),
            "--command", "init",
            "--print-only",
            "--accept-defaults",
            "--load-profile",
            "--profile-name", "test1",
            "--profile-dir", str(profile_dir),
        ]
        proc2 = subprocess.run(cmd_load, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc2.returncode == 0
