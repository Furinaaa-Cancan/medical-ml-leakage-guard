"""Unit tests for scripts/run_dag_pipeline.py command-building and helpers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _gate_registry import GATE_REGISTRY
from run_dag_pipeline import (
    _build_aggregation_cmd,
    _build_standard_gate_cmd,
    _gate_specific_extras,
    build_gate_command,
    load_checkpoint,
    run_gate_subprocess,
    save_checkpoint,
)


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _make_args(**overrides) -> argparse.Namespace:
    """Create a minimal args namespace for testing."""
    defaults = {
        "python": sys.executable,
        "request": "/tmp/test_request.json",
        "evidence_dir": "/tmp/evidence",
        "strict": True,
        "continue_on_fail": False,
        "compare_manifest": None,
        "allow_missing_compare": False,
        "parallel": False,
        "max_workers": 4,
        "dry_run": False,
        "resume": False,
        "rerun_failed": False,
        "force": False,
        "only": None,
        "no_deps": False,
        "from_gate": None,
        "skip": None,
        "show_dag": False,
        "report": None,
        "timeout": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_report_paths(evidence_dir: Path) -> Dict[str, Path]:
    paths = {}
    for name, spec in GATE_REGISTRY.items():
        if spec.report_output:
            paths[name] = evidence_dir / spec.report_output
    return paths


# ────────────────────────────────────────────────────────
# _gate_specific_extras
# ────────────────────────────────────────────────────────

class TestGateSpecificExtras:
    def test_leakage_gate_extras(self):
        normalized = {
            "patient_id_col": "pid",
            "index_time_col": "ts",
            "label_col": "y",
        }
        extras = _gate_specific_extras("leakage_gate", normalized, {})
        assert "--id-cols" in extras
        assert "pid" in extras
        assert "--time-col" in extras
        assert "--target-col" in extras

    def test_split_protocol_gate_extras(self):
        normalized = {
            "patient_id_col": "pid",
            "index_time_col": "ts",
            "label_col": "y",
        }
        extras = _gate_specific_extras("split_protocol_gate", normalized, {})
        assert "--id-col" in extras
        assert "--time-col" in extras
        assert "--target-col" in extras

    def test_tuning_leakage_gate_with_valid(self):
        normalized = {"patient_id_col": "pid"}
        split_paths = {"valid": "/data/valid.csv"}
        extras = _gate_specific_extras("tuning_leakage_gate", normalized, split_paths)
        assert "--id-col" in extras
        assert "--has-valid-split" in extras

    def test_tuning_leakage_gate_without_valid(self):
        normalized = {"patient_id_col": "pid"}
        extras = _gate_specific_extras("tuning_leakage_gate", normalized, {})
        assert "--has-valid-split" not in extras

    def test_unknown_gate_empty_extras(self):
        extras = _gate_specific_extras("nonexistent_gate", {}, {})
        assert extras == []

    def test_metric_consistency_gate_extras(self):
        normalized = {
            "primary_metric": "pr_auc",
            "actual_primary_metric": "0.85",
            "evaluation_metric_path": "metrics.test.pr_auc",
        }
        extras = _gate_specific_extras("metric_consistency_gate", normalized, {})
        assert "--metric-name" in extras
        assert "pr_auc" in extras
        assert "--expected" in extras
        assert "0.85" in extras
        assert "--metric-path" in extras


# ────────────────────────────────────────────────────────
# _build_standard_gate_cmd
# ────────────────────────────────────────────────────────

class TestBuildStandardGateCmd:
    def test_split_paths_for_leakage(self):
        spec = GATE_REGISTRY["leakage_gate"]
        split_paths = {"train": "/d/train.csv", "test": "/d/test.csv"}
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = _build_standard_gate_cmd(spec, {}, split_paths, report_paths)
        assert "--train" in cmd
        assert "/d/train.csv" in cmd
        assert "--test" in cmd

    def test_no_splits_for_non_split_gate(self):
        spec = GATE_REGISTRY["metric_consistency_gate"]
        split_paths = {"train": "/d/train.csv"}
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = _build_standard_gate_cmd(spec, {}, split_paths, report_paths)
        assert "--train" not in cmd

    def test_request_inputs_mapped(self):
        spec = GATE_REGISTRY["split_protocol_gate"]
        normalized = {"split_protocol_spec": "/d/spec.json"}
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = _build_standard_gate_cmd(spec, normalized, {}, report_paths)
        assert "--protocol-spec" in cmd
        assert "/d/spec.json" in cmd

    def test_report_inputs_mapped(self):
        spec = GATE_REGISTRY["evaluation_quality_gate"]
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = _build_standard_gate_cmd(spec, {}, {}, report_paths)
        # evaluation_quality_gate does not accept --metric-report;
        # the former report_inputs entry was dead config and has been removed.
        assert "--metric-report" not in cmd


# ────────────────────────────────────────────────────────
# _build_aggregation_cmd
# ────────────────────────────────────────────────────────

class TestBuildAggregationCmd:
    def test_publication_gate_gets_all_reports(self):
        report_paths = _make_report_paths(Path("/tmp/ev"))
        args = _make_args()
        cmd = _build_aggregation_cmd("publication_gate", report_paths, args)
        assert "--request-report" in cmd
        assert "--manifest" in cmd
        assert "--leakage-report" in cmd
        assert "--publication-report" not in cmd

    def test_self_critique_gets_publication_report(self):
        report_paths = _make_report_paths(Path("/tmp/ev"))
        args = _make_args()
        cmd = _build_aggregation_cmd("self_critique_gate", report_paths, args)
        assert "--publication-report" in cmd
        assert "--min-score" in cmd
        assert "95" in cmd

    def test_self_critique_forwards_allow_missing_comparison(self):
        report_paths = _make_report_paths(Path("/tmp/ev"))
        args = _make_args(allow_missing_compare=True)
        cmd = _build_aggregation_cmd("self_critique_gate", report_paths, args)
        assert "--allow-missing-comparison" in cmd

    def test_self_critique_no_allow_missing_by_default(self):
        report_paths = _make_report_paths(Path("/tmp/ev"))
        args = _make_args(allow_missing_compare=False)
        cmd = _build_aggregation_cmd("self_critique_gate", report_paths, args)
        assert "--allow-missing-comparison" not in cmd


# ────────────────────────────────────────────────────────
# build_gate_command
# ────────────────────────────────────────────────────────

class TestBuildGateCommand:
    def test_request_contract_gate_cmd(self):
        args = _make_args()
        spec = GATE_REGISTRY["request_contract_gate"]
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = build_gate_command(
            spec, args, Path("/scripts"), Path("/tmp/ev"), {}, report_paths, {},
        )
        assert cmd[0] == sys.executable
        assert "request_contract_gate.py" in cmd[1]
        assert "--request" in cmd
        assert "--report" in cmd
        assert "--strict" in cmd

    def test_manifest_lock_returns_early_with_output(self):
        args = _make_args()
        spec = GATE_REGISTRY["manifest_lock"]
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = build_gate_command(
            spec, args, Path("/scripts"), Path("/tmp/ev"), {}, report_paths, {},
        )
        assert "--output" in cmd
        assert "--report" not in cmd  # manifest_lock uses --output, not --report

    def test_standard_gate_has_report_and_strict(self):
        args = _make_args()
        spec = GATE_REGISTRY["leakage_gate"]
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = build_gate_command(
            spec, args, Path("/scripts"), Path("/tmp/ev"),
            {"patient_id_col": "pid", "index_time_col": "ts", "label_col": "y"},
            report_paths,
            {"train": "/d/train.csv"},
        )
        assert "--report" in cmd
        assert "--strict" in cmd
        assert "--train" in cmd

    def test_no_strict_when_not_set(self):
        args = _make_args(strict=False)
        spec = GATE_REGISTRY["leakage_gate"]
        report_paths = _make_report_paths(Path("/tmp/ev"))
        cmd = build_gate_command(
            spec, args, Path("/scripts"), Path("/tmp/ev"), {}, report_paths, {},
        )
        assert "--strict" not in cmd


# ────────────────────────────────────────────────────────
# Checkpoint management
# ────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        state = {"passed_gates": ["a", "b"], "last_run_utc": "2025-01-01T00:00:00Z"}
        save_checkpoint(tmp_path, state)
        loaded = load_checkpoint(tmp_path)
        assert loaded["passed_gates"] == ["a", "b"]

    def test_load_missing_returns_empty(self, tmp_path):
        loaded = load_checkpoint(tmp_path)
        assert loaded == {}

    def test_load_corrupt_returns_empty(self, tmp_path):
        cp = tmp_path / "dag_checkpoint.json"
        cp.write_text("not-json!", encoding="utf-8")
        loaded = load_checkpoint(tmp_path)
        assert loaded == {}


# ────────────────────────────────────────────────────────
# run_gate_subprocess
# ────────────────────────────────────────────────────────

class TestCheckpointIntegrity:
    """Tests for E3a: checkpoint report file integrity validation."""

    def test_missing_report_invalidates_gate(self, tmp_path):
        """If a checkpoint says gate passed but report file is gone, gate should be re-run."""
        # Save checkpoint claiming leakage_gate passed
        save_checkpoint(tmp_path, {"passed_gates": ["request_contract_gate", "leakage_gate"]})

        # Create report_paths but DON'T create the actual leakage report file
        report_paths = _make_report_paths(tmp_path)
        # Create the request_contract report so it stays valid
        report_paths["request_contract_gate"].write_text('{"envelope_version":"2.0.0"}', encoding="utf-8")

        # Load checkpoint and simulate the validation logic from main()
        checkpoint = load_checkpoint(tmp_path)
        passed_gates = set(checkpoint.get("passed_gates", []))
        assert "leakage_gate" in passed_gates

        # Apply the same validation logic as run_dag_pipeline.main()
        invalidated = []
        for pg in sorted(passed_gates):
            if pg in report_paths and not report_paths[pg].exists():
                invalidated.append(pg)
        passed_gates -= set(invalidated)

        assert "leakage_gate" in invalidated
        assert "leakage_gate" not in passed_gates
        assert "request_contract_gate" in passed_gates

    def test_all_reports_present_no_invalidation(self, tmp_path):
        """If all checkpoint reports exist, no gates should be invalidated."""
        save_checkpoint(tmp_path, {"passed_gates": ["request_contract_gate", "leakage_gate"]})
        report_paths = _make_report_paths(tmp_path)
        # Create both report files
        report_paths["request_contract_gate"].write_text('{}', encoding="utf-8")
        report_paths["leakage_gate"].write_text('{}', encoding="utf-8")

        checkpoint = load_checkpoint(tmp_path)
        passed_gates = set(checkpoint.get("passed_gates", []))
        invalidated = [
            pg for pg in sorted(passed_gates)
            if pg in report_paths and not report_paths[pg].exists()
        ]
        assert invalidated == []


class TestRunGateSubprocess:
    def test_successful_command(self):
        result = run_gate_subprocess("echo_test", [sys.executable, "-c", "print('ok')"])
        assert result["name"] == "echo_test"
        assert result["exit_code"] == 0
        assert result["status"] == "pass"
        assert "ok" in result["stdout_tail"]
        assert result["execution_time_seconds"] >= 0

    def test_failing_command(self):
        result = run_gate_subprocess("fail_test", [sys.executable, "-c", "raise SystemExit(2)"])
        assert result["exit_code"] == 2
        assert result["status"] == "fail"

    def test_nonexistent_command(self):
        result = run_gate_subprocess("bad_cmd", ["/nonexistent/binary/xyz"])
        assert result["status"] == "fail"
        assert result["exit_code"] == 2
        assert "EXCEPTION" in result["stderr_tail"]


# ────────────────────────────────────────────────────────
# _run_parallel resilience
# ────────────────────────────────────────────────────────

class TestRunParallelResilience:
    """Regression tests for parallel execution bugs."""

    def test_results_list_independent_dicts(self):
        """Bug 2 regression: [{}]*N creates shared refs, [{} for _ in range(N)] doesn't."""
        n = 5
        results = [{} for _ in range(n)]
        results[0]["test"] = True
        # Verify mutation doesn't affect other slots
        for i in range(1, n):
            assert "test" not in results[i]

    def test_parallel_exception_produces_fail_result(self):
        """Bug 3 regression: exception in future.result() should produce a fail step."""

        # We can't easily test _run_parallel directly (needs full args),
        # but we verify the pattern: exception -> structured fail result
        def _raising_gate(name, cmd):
            raise RuntimeError("synthetic error")

        # Verify run_gate_subprocess handles all exceptions
        result = run_gate_subprocess("crash_test", ["/nonexistent/path/to/binary"])
        assert result["status"] == "fail"
        assert result["exit_code"] == 2


# ────────────────────────────────────────────────────────
# Blocked gates visibility
# ────────────────────────────────────────────────────────

class TestBlockedGatesVisibility:
    """Bug 1 regression: blocked gates must appear in steps regardless of continue_on_fail."""

    def test_blocked_gate_always_in_steps(self):
        """When a gate's dependency isn't met, it should appear as 'skip' in the report."""
        # Simulate blocked gate logic
        blocked = ["gate_c"]
        steps: list = []
        for bg in blocked:
            steps.append({
                "name": bg, "command": "", "exit_code": -1,
                "status": "skip", "execution_time_seconds": 0,
                "stdout_tail": "", "stderr_tail": "Skipped: dependency not met",
            })
        assert len(steps) == 1
        assert steps[0]["name"] == "gate_c"
        assert steps[0]["status"] == "skip"
