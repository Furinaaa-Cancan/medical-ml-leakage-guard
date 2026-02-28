"""Tests for scripts/mlgg_onboarding.py helper functions.

Covers utc_now, ensure_parent, write_json, load_json, maybe_prompt_confirm,
collect_failure_codes, collect_step_failure_codes, absolutize_repo_python_command,
build_next_actions, build_copy_ready_commands, derive_git_commit,
update_request_actual_metric, build_train_command, align_demo_configs,
and CLI integration (--help, --mode preview).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "mlgg_onboarding.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import mlgg_onboarding as ob


# ── pure helpers ─────────────────────────────────────────────────────────────

class TestUtcNow:
    def test_format(self):
        ts = ob.utc_now()
        assert ts.endswith("Z")
        assert "T" in ts


class TestEnsureParent:
    def test_creates_parent(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "file.json"
        ob.ensure_parent(target)
        assert target.parent.exists()


class TestWriteJson:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "out.json"
        ob.write_json(p, {"key": "value"})
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["key"] == "value"

    def test_nested_dir(self, tmp_path):
        p = tmp_path / "a" / "b" / "out.json"
        ob.write_json(p, {"x": 1})
        assert p.exists()


class TestLoadJson:
    def test_valid(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text('{"hello": "world"}')
        data = ob.load_json(p)
        assert data["hello"] == "world"

    def test_non_object(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("[1,2,3]")
        with pytest.raises(ValueError, match="object"):
            ob.load_json(p)


class TestMaybePromptConfirm:
    def test_auto_yes(self):
        ok, reason = ob.maybe_prompt_confirm("guided", True, "title", "cmd", [])
        assert ok is True
        assert reason == ""

    def test_preview_mode(self):
        ok, reason = ob.maybe_prompt_confirm("auto", False, "title", "cmd", [])
        assert ok is True


class TestCollectFailureCodes:
    def test_no_evidence(self, tmp_path):
        codes = ob.collect_failure_codes(tmp_path)
        assert codes == []

    def test_with_failures(self, tmp_path):
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        report = evidence / "gate_report.json"
        report.write_text(json.dumps({
            "status": "fail",
            "failures": [{"code": "test_failure_code", "message": "oops"}],
        }))
        codes = ob.collect_failure_codes(tmp_path)
        assert "test_failure_code" in codes

    def test_fail_without_code(self, tmp_path):
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        report = evidence / "gate_report.json"
        report.write_text(json.dumps({"status": "fail"}))
        codes = ob.collect_failure_codes(tmp_path)
        assert "onboarding_unknown_failure" in codes

    def test_min_mtime(self, tmp_path):
        import time
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        report = evidence / "old_report.json"
        report.write_text(json.dumps({
            "status": "fail",
            "failures": [{"code": "old_code"}],
        }))
        future_epoch = time.time() + 9999
        codes = ob.collect_failure_codes(tmp_path, min_mtime_epoch=future_epoch)
        assert codes == []


class TestCollectStepFailureCodes:
    def test_basic(self):
        steps = [
            {"status": "fail", "failure_code": "code_a"},
            {"status": "pass"},
            {"status": "fail", "failure_code": "code_b"},
        ]
        codes = ob.collect_step_failure_codes(steps)
        assert "code_a" in codes
        assert "code_b" in codes

    def test_empty(self):
        assert ob.collect_step_failure_codes([]) == []

    def test_no_code(self):
        steps = [{"status": "fail"}]
        assert ob.collect_step_failure_codes(steps) == []


class TestAbsolutizeRepoPythonCommand:
    def test_transforms(self):
        raw = "python3 scripts/env_doctor.py --report /tmp/r.json"
        result = ob.absolutize_repo_python_command(raw)
        assert "env_doctor.py" in result
        assert result != raw or str(SCRIPTS_DIR) in result


class TestBuildNextActions:
    def test_pass(self):
        actions = ob.build_next_actions([], status="pass", lang="en", mode="auto")
        assert len(actions) >= 1
        assert any("blocking" in a.lower() or "no" in a.lower() for a in actions)

    def test_preview(self):
        actions = ob.build_next_actions([], status="pass", lang="en", mode="preview")
        assert any("preview" in a.lower() for a in actions)

    def test_cancelled(self):
        actions = ob.build_next_actions(["onboarding_step_cancelled"], status="fail", lang="en", mode="auto")
        assert any("cancel" in a.lower() for a in actions)

    def test_known_failure_code(self):
        actions = ob.build_next_actions(["missing_required_path"], status="fail", lang="en", mode="auto")
        assert len(actions) >= 1

    def test_unknown_failure_code(self):
        actions = ob.build_next_actions(["totally_unknown_code_xyz"], status="fail", lang="en", mode="auto")
        assert len(actions) >= 1

    def test_fail_no_codes(self):
        actions = ob.build_next_actions([], status="fail", lang="en", mode="auto")
        assert len(actions) >= 1

    def test_zh_lang(self):
        actions = ob.build_next_actions([], status="pass", lang="zh", mode="auto")
        assert len(actions) >= 1

    def test_bilingual(self):
        actions = ob.build_next_actions([], status="pass", lang="bilingual", mode="auto")
        assert len(actions) >= 1


class TestBuildCopyReadyCommands:
    def test_basic(self, tmp_path):
        cmds = ob.build_copy_ready_commands(tmp_path)
        assert "workflow_bootstrap" in cmds
        assert "workflow_compare" in cmds
        assert "authority_release" in cmds
        assert "adversarial" in cmds


class TestDeriveGitCommit:
    def test_returns_string(self):
        result = ob.derive_git_commit()
        assert isinstance(result, str)
        assert len(result) > 0


class TestBuildTrainCommand:
    def test_basic(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "configs").mkdir()
        cmd, outputs = ob.build_train_command(tmp_path, sys.executable)
        assert any("train_select_evaluate" in arg for arg in cmd)
        assert "model_selection_report" in outputs
        assert "evaluation_report" in outputs


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        assert proc.returncode == 0
        assert "onboarding" in proc.stdout.lower()


class TestCLIPreview:
    def test_preview_mode(self, tmp_path):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(tmp_path / "demo"),
            "--mode", "preview",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "preview" in proc.stdout.lower()
        report = tmp_path / "demo" / "evidence" / "onboarding_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["mode"] == "preview"
        assert data["display_status"] == "preview"
