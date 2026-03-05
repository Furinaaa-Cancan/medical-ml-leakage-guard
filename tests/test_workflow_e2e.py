"""E2E integration tests for scripts/run_productized_workflow.py.

Tests the workflow wrapper including env_doctor, schema_preflight,
strict pipeline, and user summary rendering.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_workflow(request_path: Path, evidence_dir: Path, report_path: Path,
                  extra_args: list = None, timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_productized_workflow.py"),
        "--request", str(request_path),
        "--evidence-dir", str(evidence_dir),
        "--strict",
        "--report", str(report_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(REPO_ROOT))


class TestWorkflowRequiresStrict:
    def test_no_strict_fails(self, tmp_path: Path):
        req = tmp_path / "request.json"
        req.write_text("{}", encoding="utf-8")
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "run_productized_workflow.py"),
            "--request", str(req),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        assert "strict" in result.stderr.lower()


class TestWorkflowMissingRequest:
    def test_missing_request_file(self, tmp_path: Path):
        result = _run_workflow(
            request_path=tmp_path / "nonexistent.json",
            evidence_dir=tmp_path / "evidence",
            report_path=tmp_path / "report.json",
        )
        assert result.returncode == 2


class TestWorkflowMissingSplitPaths:
    def test_no_split_paths(self, tmp_path: Path):
        req = tmp_path / "configs" / "request.json"
        req.parent.mkdir(parents=True)
        req.write_text(json.dumps({"study_id": "test"}), encoding="utf-8")
        result = _run_workflow(
            request_path=req,
            evidence_dir=tmp_path / "evidence",
            report_path=tmp_path / "report.json",
        )
        assert result.returncode == 2


class TestWorkflowReportStructure:
    """Test report structure using a minimal setup that will at least produce a report."""

    def _setup_minimal_project(self, tmp_path: Path) -> Path:
        """Create minimal project structure with split files."""
        project = tmp_path / "project"
        configs = project / "configs"
        data = project / "data"
        evidence = project / "evidence"
        configs.mkdir(parents=True)
        data.mkdir(parents=True)
        evidence.mkdir(parents=True)

        # Create minimal CSV files
        header = "patient_id,event_time,y,age,bp\n"
        train_rows = "".join(
            f"P{i},2024-01-{i+1:02d},{i%2},{20+i},{100+i}\n" for i in range(40)
        )
        valid_rows = "".join(
            f"P{i},2024-03-{(i-40)+1:02d},{i%2},{20+i},{100+i}\n" for i in range(40, 60)
        )
        test_rows = "".join(
            f"P{i},2024-05-{(i-60)+1:02d},{i%2},{20+i},{100+i}\n" for i in range(60, 80)
        )
        (data / "train.csv").write_text(header + train_rows, encoding="utf-8")
        (data / "valid.csv").write_text(header + valid_rows, encoding="utf-8")
        (data / "test.csv").write_text(header + test_rows, encoding="utf-8")

        request = {
            "study_id": "test-workflow",
            "run_id": "test-run",
            "label_col": "y",
            "patient_id_col": "patient_id",
            "index_time_col": "event_time",
            "split_paths": {
                "train": "../data/train.csv",
                "valid": "../data/valid.csv",
                "test": "../data/test.csv",
            },
        }
        (configs / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
        return project

    def test_report_generated(self, tmp_path: Path):
        project = self._setup_minimal_project(tmp_path)
        report_path = tmp_path / "workflow_report.json"
        result = _run_workflow(
            request_path=project / "configs" / "request.json",
            evidence_dir=project / "evidence",
            report_path=report_path,
            extra_args=["--allow-missing-compare", "--continue-on-fail"],
            timeout=120,
        )
        # Report should always be generated regardless of pass/fail
        assert report_path.exists(), (
            f"Report not generated. rc={result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
        report = json.loads(report_path.read_text())
        assert report["contract_version"] == "productized_workflow_report.v2"
        assert "status" in report
        assert "status_reason" in report
        assert "blocking_failure_count" in report
        assert "recovered_failure_count" in report
        assert "bootstrap_recovery_applied" in report
        assert "steps" in report
        assert "artifacts" in report
        assert isinstance(report["steps"], list)
        assert len(report["steps"]) >= 2  # at least doctor + preflight

    def test_report_has_step_structure(self, tmp_path: Path):
        project = self._setup_minimal_project(tmp_path)
        report_path = tmp_path / "workflow_report.json"
        _run_workflow(
            request_path=project / "configs" / "request.json",
            evidence_dir=project / "evidence",
            report_path=report_path,
            extra_args=["--allow-missing-compare", "--continue-on-fail"],
            timeout=120,
        )
        report = json.loads(report_path.read_text())
        for step in report["steps"]:
            assert "name" in step
            assert "command" in step
            assert "exit_code" in step
            assert "status" in step

    def test_env_doctor_runs(self, tmp_path: Path):
        project = self._setup_minimal_project(tmp_path)
        report_path = tmp_path / "workflow_report.json"
        _run_workflow(
            request_path=project / "configs" / "request.json",
            evidence_dir=project / "evidence",
            report_path=report_path,
            extra_args=["--allow-missing-compare", "--continue-on-fail"],
            timeout=120,
        )
        report = json.loads(report_path.read_text())
        step_names = [s["name"] for s in report["steps"]]
        assert "env_doctor" in step_names

    def test_schema_preflight_runs(self, tmp_path: Path):
        project = self._setup_minimal_project(tmp_path)
        report_path = tmp_path / "workflow_report.json"
        _run_workflow(
            request_path=project / "configs" / "request.json",
            evidence_dir=project / "evidence",
            report_path=report_path,
            extra_args=["--allow-missing-compare", "--continue-on-fail"],
            timeout=120,
        )
        report = json.loads(report_path.read_text())
        step_names = [s["name"] for s in report["steps"]]
        assert "schema_preflight" in step_names

    def test_artifacts_section(self, tmp_path: Path):
        project = self._setup_minimal_project(tmp_path)
        report_path = tmp_path / "workflow_report.json"
        _run_workflow(
            request_path=project / "configs" / "request.json",
            evidence_dir=project / "evidence",
            report_path=report_path,
            extra_args=["--allow-missing-compare", "--continue-on-fail"],
            timeout=120,
        )
        report = json.loads(report_path.read_text())
        artifacts = report["artifacts"]
        assert "env_doctor_report" in artifacts
        assert "schema_preflight_report" in artifacts
        assert "user_summary_markdown" in artifacts
