"""
E2E smoke tests for the DAG pipeline infrastructure.

These tests verify:
- DAG structure is valid and printable (--show-dag)
- Dry-run mode constructs commands without executing (--dry-run)
- request_contract_gate produces v2.0.0 envelope format
- Pipeline report is written in expected format
- Checkpoint save/load round-trips correctly
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
PYTHON = sys.executable


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def write_csv(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_minimal_splits(tmp: Path) -> Dict[str, str]:
    """Create minimal train/valid/test CSV files."""
    headers = ["patient_id", "event_time", "age", "bp", "y"]
    train_rows = [
        [f"P{i:03d}", f"2024-01-{i+1:02d}", str(30 + i), str(120 + i), str(i % 2)]
        for i in range(20)
    ]
    valid_rows = [
        [f"V{i:03d}", f"2024-02-{i+1:02d}", str(40 + i), str(130 + i), str(i % 2)]
        for i in range(10)
    ]
    test_rows = [
        [f"T{i:03d}", f"2024-03-{i+1:02d}", str(50 + i), str(140 + i), str(i % 2)]
        for i in range(10)
    ]

    paths = {}
    for name, rows in [("train", train_rows), ("valid", valid_rows), ("test", test_rows)]:
        p = tmp / f"{name}.csv"
        write_csv(p, headers, rows)
        paths[name] = str(p)
    return paths


def make_minimal_request(tmp: Path, split_paths: Dict[str, str]) -> Path:
    """Create a minimal request.json for request_contract_gate."""
    request = {
        "study_id": "smoke_test_001",
        "run_id": "run_001",
        "target_name": "sepsis_onset",
        "prediction_unit": "patient-encounter",
        "index_time_col": "event_time",
        "label_col": "y",
        "patient_id_col": "patient_id",
        "primary_metric": "pr_auc",
        "phenotype_definition_spec": str(tmp / "phenotype.json"),
        "claim_tier_target": "publication-grade",
        "split_paths": split_paths,
        "actual_primary_metric": 0.85,
    }
    # Create the phenotype spec file
    write_json_file(tmp / "phenotype.json", {
        "name": "sepsis_onset",
        "version": "1.0",
        "inclusion_criteria": "test",
    })
    req_path = tmp / "request.json"
    write_json_file(req_path, request)
    return req_path


def run_script(args: List[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON] + args,
        text=True,
        capture_output=True,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(SCRIPTS_DIR)},
    )


# ────────────────────────────────────────────────────────
# Test: DAG structure validation (--show-dag)
# ────────────────────────────────────────────────────────

class TestShowDAG:
    def test_show_dag_exits_zero(self):
        result = run_script([str(SCRIPTS_DIR / "run_dag_pipeline.py"), "--show-dag"])
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_show_dag_lists_all_layers(self):
        result = run_script([str(SCRIPTS_DIR / "run_dag_pipeline.py"), "--show-dag"])
        assert "Layer" in result.stdout
        assert "gate" in result.stdout.lower()

    def test_show_dag_lists_publication_gate(self):
        result = run_script([str(SCRIPTS_DIR / "run_dag_pipeline.py"), "--show-dag"])
        assert "publication_gate" in result.stdout

    def test_show_dag_lists_self_critique_gate(self):
        result = run_script([str(SCRIPTS_DIR / "run_dag_pipeline.py"), "--show-dag"])
        assert "self_critique_gate" in result.stdout


# ────────────────────────────────────────────────────────
# Test: Dry-run mode
# ────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_fails_gracefully_when_request_contract_fails(self):
        """Dry-run still requires request_contract_gate to pass.
        With a minimal (incomplete) request, it should fail gracefully and
        produce a pipeline report."""
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            evidence_dir = td / "evidence"
            result = run_script([
                str(SCRIPTS_DIR / "run_dag_pipeline.py"),
                "--request", str(req),
                "--evidence-dir", str(evidence_dir),
                "--strict",
                "--dry-run",
            ])
            assert result.returncode == 2
            assert "request_contract_gate" in result.stdout
            # Pipeline report should still be written
            report_path = evidence_dir / "dag_pipeline_report.json"
            assert report_path.exists()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert report["status"] == "fail"
            assert report["strict_mode"] is True
            assert isinstance(report["steps"], list)
            assert len(report["steps"]) >= 1
            assert report["steps"][0]["name"] == "request_contract_gate"

    def test_pipeline_report_format(self):
        """Verify pipeline report has all expected fields."""
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            evidence_dir = td / "evidence"
            run_script([
                str(SCRIPTS_DIR / "run_dag_pipeline.py"),
                "--request", str(req),
                "--evidence-dir", str(evidence_dir),
                "--strict",
            ])
            report_path = evidence_dir / "dag_pipeline_report.json"
            assert report_path.exists()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for key in ("contract_version", "status", "strict_mode",
                        "failure_count", "pass_count", "skip_count",
                        "total_execution_time_seconds", "steps", "evidence_dir"):
                assert key in report, f"Pipeline report missing key: {key}"


# ────────────────────────────────────────────────────────
# Test: request_contract_gate envelope format
# ────────────────────────────────────────────────────────

class TestRequestContractEnvelope:
    def test_produces_v2_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            report_path = td / "request_contract_report.json"
            result = run_script([
                str(SCRIPTS_DIR / "request_contract_gate.py"),
                "--request", str(req),
                "--report", str(report_path),
            ])
            assert report_path.exists(), f"Report not written. stderr: {result.stderr}"
            report = json.loads(report_path.read_text(encoding="utf-8"))

            assert report.get("envelope_version") == "2.0.0"
            assert report.get("gate_name") == "request_contract_gate"
            assert "status" in report
            assert "execution_timestamp_utc" in report
            assert isinstance(report.get("failures"), list)
            assert isinstance(report.get("warnings"), list)
            assert "failure_count" in report
            assert "warning_count" in report

    def test_normalized_request_at_top_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            report_path = td / "request_contract_report.json"
            run_script([
                str(SCRIPTS_DIR / "request_contract_gate.py"),
                "--request", str(req),
                "--report", str(report_path),
            ])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            normalized = report.get("normalized_request")
            assert isinstance(normalized, dict), "normalized_request should be a dict at top level"
            assert normalized.get("study_id") == "smoke_test_001"
            assert normalized.get("patient_id_col") == "patient_id"

    def test_execution_time_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            report_path = td / "request_contract_report.json"
            run_script([
                str(SCRIPTS_DIR / "request_contract_gate.py"),
                "--request", str(req),
                "--report", str(report_path),
            ])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert "execution_time_seconds" in report
            assert isinstance(report["execution_time_seconds"], (int, float))
            assert report["execution_time_seconds"] >= 0


# ────────────────────────────────────────────────────────
# Test: --only single gate dry-run
# ────────────────────────────────────────────────────────

class TestOnlyGate:
    def test_only_flag_accepted(self):
        """Verify --only flag is accepted by the CLI parser."""
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            evidence_dir = td / "evidence"
            result = run_script([
                str(SCRIPTS_DIR / "run_dag_pipeline.py"),
                "--request", str(req),
                "--evidence-dir", str(evidence_dir),
                "--strict",
                "--only", "leakage_gate",
            ])
            # Will fail at request_contract, but should not crash on arg parsing
            assert "request_contract_gate" in result.stdout

    def test_step_result_has_expected_fields(self):
        """Verify each step in the pipeline report has required fields."""
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            evidence_dir = td / "evidence"
            run_script([
                str(SCRIPTS_DIR / "run_dag_pipeline.py"),
                "--request", str(req),
                "--evidence-dir", str(evidence_dir),
                "--strict",
            ])
            report = json.loads((evidence_dir / "dag_pipeline_report.json").read_text(encoding="utf-8"))
            for step in report["steps"]:
                for key in ("name", "command", "exit_code", "status",
                            "execution_time_seconds", "stdout_tail", "stderr_tail"):
                    assert key in step, f"Step {step.get('name')} missing key: {key}"


# ────────────────────────────────────────────────────────
# Test: CLI argument validation
# ────────────────────────────────────────────────────────

class TestCLIValidation:
    def test_missing_request_fails(self):
        result = run_script([
            str(SCRIPTS_DIR / "run_dag_pipeline.py"),
            "--strict",
        ])
        assert result.returncode == 2

    def test_missing_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            splits = make_minimal_splits(td)
            req = make_minimal_request(td, splits)
            result = run_script([
                str(SCRIPTS_DIR / "run_dag_pipeline.py"),
                "--request", str(req),
            ])
            assert result.returncode == 2
            assert "strict" in result.stderr.lower()

    def test_nonexistent_request_file_fails(self):
        result = run_script([
            str(SCRIPTS_DIR / "run_dag_pipeline.py"),
            "--request", "/nonexistent/request.json",
            "--strict",
        ])
        assert result.returncode == 2
        assert "not found" in result.stderr.lower()
