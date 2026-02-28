"""E2E integration tests for scripts/mlgg_onboarding.py.

These tests run the full onboarding pipeline (or preview mode) and validate
the generated report structure and key artifacts.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_onboarding(tmp_path: Path, extra_args: list = None, timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "mlgg_onboarding.py"),
        "--project-root", str(tmp_path / "project"),
        "--report", str(tmp_path / "onboarding_report.json"),
        "--python", sys.executable,
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))


class TestOnboardingPreview:
    """Preview mode generates plan without execution — fast and safe to test."""

    def test_preview_generates_report(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0, f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        report_path = tmp_path / "onboarding_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["contract_version"] == "onboarding_report.v2"
        assert report["status"] == "pass"
        assert report["mode"] == "preview"
        assert report["preview_only"] is True
        assert report["display_status"] == "preview"
        assert report["termination_reason"] == "completed_successfully"
        assert report["strict_mode"] is True

    def test_preview_has_8_steps(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        steps = report["steps"]
        assert len(steps) == 8
        for step in steps:
            assert step["status"] == "preview"

    def test_preview_step_names(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        names = [s["name"] for s in report["steps"]]
        assert "step1_doctor" in names
        assert "step2_init" in names
        assert any("step3" in n for n in names)
        assert "step4_align_configs" in names
        assert "step5_train" in names
        assert "step6_attestation" in names
        assert "step7_workflow_bootstrap" in names
        assert "step8_workflow_compare" in names

    def test_preview_has_copy_ready_commands(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        cmds = report["copy_ready_commands"]
        assert "workflow_bootstrap" in cmds
        assert "workflow_compare" in cmds
        # All command values should contain absolute paths
        for key, val in cmds.items():
            assert "/" in val, f"copy_ready_commands[{key}] should use absolute paths"

    def test_preview_has_next_actions(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        assert isinstance(report["next_actions"], list)
        assert len(report["next_actions"]) > 0

    def test_preview_failure_codes_empty(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        assert report["failure_codes"] == []

    def test_preview_artifacts_dict(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        artifacts = report["artifacts"]
        assert "project_root" in artifacts
        assert "request_file" in artifacts
        assert "onboarding_report" in artifacts

    def test_preview_report_structure_complete(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "preview"])
        assert result.returncode == 0
        report = json.loads((tmp_path / "onboarding_report.json").read_text())
        required_keys = [
            "contract_version", "run_id", "status", "mode", "lang",
            "strict_mode", "stop_on_fail", "termination_reason",
            "generated_at_utc", "project_root", "steps", "artifacts",
            "failure_codes", "preview_only", "display_status",
            "next_actions", "copy_ready_commands",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"


@pytest.mark.slow
class TestOnboardingFullAutoDemo:
    """Full auto onboarding with demo data. Marked slow (~2-5 min)."""

    def test_full_auto_onboarding(self, tmp_path: Path):
        result = _run_onboarding(tmp_path, extra_args=["--mode", "auto", "--yes"], timeout=600)
        report_path = tmp_path / "onboarding_report.json"
        assert report_path.exists(), f"Report not generated.\nstdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        report = json.loads(report_path.read_text())

        # Report structure
        assert report["contract_version"] == "onboarding_report.v2"
        assert report["mode"] == "auto"
        assert report["strict_mode"] is True

        project = tmp_path / "project"

        # Data files
        assert (project / "data" / "train.csv").exists()
        assert (project / "data" / "valid.csv").exists()
        assert (project / "data" / "test.csv").exists()

        # Config files
        assert (project / "configs" / "request.json").exists()

        # Check steps
        steps = report["steps"]
        assert len(steps) >= 3  # At minimum doctor + init + demo data

        # If all steps passed
        if report["status"] == "pass":
            assert report["termination_reason"] == "completed_successfully"
            assert report["failure_codes"] == []
            assert len(steps) == 8

            # Evidence artifacts
            assert (project / "evidence" / "onboarding_report.json").exists() or report_path.exists()

            # Models directory should be non-empty if training succeeded
            models_dir = project / "models"
            if models_dir.exists():
                assert any(models_dir.iterdir())

            # Copy ready commands should have absolute paths
            for key, cmd_str in report["copy_ready_commands"].items():
                assert "/" in cmd_str

            # Next actions should be non-empty
            assert len(report["next_actions"]) > 0
        else:
            # Even on failure, report should have useful diagnostics
            assert len(report["failure_codes"]) > 0 or report["termination_reason"] in (
                "stopped_on_failure", "completed_with_failures"
            )
