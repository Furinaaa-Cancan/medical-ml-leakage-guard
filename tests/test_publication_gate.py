"""Comprehensive unit tests for scripts/publication_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from publication_gate import (
    parse_int_like,
    validate_component_status,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# All 26 component artifact names expected by publication_gate.py
COMPONENT_NAMES = [
    "request_report",
    "manifest",
    "execution_attestation_report",
    "reporting_bias_report",
    "leakage_report",
    "split_protocol_report",
    "covariate_shift_report",
    "definition_report",
    "lineage_report",
    "imbalance_report",
    "missingness_report",
    "tuning_report",
    "model_selection_audit_report",
    "feature_engineering_audit_report",
    "clinical_metrics_report",
    "prediction_replay_report",
    "distribution_generalization_report",
    "generalization_gap_report",
    "robustness_report",
    "seed_stability_report",
    "external_validation_report",
    "calibration_dca_report",
    "ci_matrix_report",
    "metric_report",
    "evaluation_quality_report",
    "permutation_report",
]

# CLI arg names corresponding to each component
COMPONENT_ARGS = [
    "--request-report",
    "--manifest",
    "--execution-attestation-report",
    "--reporting-bias-report",
    "--leakage-report",
    "--split-protocol-report",
    "--covariate-shift-report",
    "--definition-report",
    "--lineage-report",
    "--imbalance-report",
    "--missingness-report",
    "--tuning-report",
    "--model-selection-audit-report",
    "--feature-engineering-audit-report",
    "--clinical-metrics-report",
    "--prediction-replay-report",
    "--distribution-generalization-report",
    "--generalization-gap-report",
    "--robustness-report",
    "--seed-stability-report",
    "--external-validation-report",
    "--calibration-dca-report",
    "--ci-matrix-report",
    "--metric-report",
    "--evaluation-quality-report",
    "--permutation-report",
]


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _good_component():
    return {"status": "pass", "strict_mode": True, "failure_count": 0}


def _good_manifest():
    return {
        "status": "pass",
        "strict_mode": True,
        "failure_count": 0,
        "files": [{"path": "train.csv", "sha256": "abc123"}],
        "errors": [],
        "comparison": {"matched": True},
    }


def _good_execution_attestation():
    return {
        "status": "pass",
        "strict_mode": True,
        "failure_count": 0,
        "summary": {
            "key_assurance": {
                "policy": {
                    "require_revocation_list": True,
                    "require_timestamp_trust": True,
                    "require_transparency_log": True,
                    "require_transparency_log_signature": True,
                    "require_execution_receipt": True,
                    "require_execution_log_attestation": True,
                    "require_independent_timestamp_authority": True,
                    "require_independent_execution_authority": True,
                    "require_independent_log_authority": True,
                    "require_distinct_authority_roles": True,
                    "require_witness_quorum": True,
                    "require_independent_witness_keys": True,
                    "require_witness_independence_from_signing": True,
                    "min_witness_count": 3,
                }
            },
            "timestamp_trust": {"present": True},
            "transparency_log": {"present": True},
            "execution_receipt": {"present": True},
            "execution_log_attestation": {"present": True},
            "witness_quorum": {
                "present": True,
                "required": True,
                "validated_witnesses": 3,
                "min_witness_count": 3,
            },
            "authority_role_distinctness": {
                "enforced": True,
                "status": "pass",
            },
        },
    }


def _good_metric_report():
    return {
        "status": "pass",
        "strict_mode": True,
        "failure_count": 0,
        "actual_metric": 0.85,
    }


def _make_all_artifacts(tmp_path: Path) -> dict:
    """Create all 26 artifact files with passing status."""
    paths = {}
    for name in COMPONENT_NAMES:
        if name == "manifest":
            data = _good_manifest()
        elif name == "execution_attestation_report":
            data = _good_execution_attestation()
        elif name == "metric_report":
            data = _good_metric_report()
        else:
            data = _good_component()
        p = tmp_path / f"{name}.json"
        _write_json(p, data)
        paths[name] = p
    return paths


def _build_cmd(tmp_path, paths, extra_args=None):
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "publication_gate.py"),
        "--report", str(tmp_path / "report.json"),
    ]
    for arg_name, comp_name in zip(COMPONENT_ARGS, COMPONENT_NAMES):
        cmd.extend([arg_name, str(paths[comp_name])])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


# ────────────────────────────────────────────────────────
# parse_int_like
# ────────────────────────────────────────────────────────

class TestParseIntLike:
    def test_int(self):
        assert parse_int_like(5) == 5

    def test_float_integer(self):
        assert parse_int_like(5.0) == 5

    def test_bool_rejected(self):
        assert parse_int_like(True) is None

    def test_string(self):
        assert parse_int_like("5") is None

    def test_float_non_integer(self):
        assert parse_int_like(5.5) is None


# ────────────────────────────────────────────────────────
# validate_component_status
# ────────────────────────────────────────────────────────

class TestValidateComponentStatus:
    def test_pass(self):
        f, w = [], []
        validate_component_status("test", {"status": "pass", "strict_mode": True, "failure_count": 0}, f, w, True)
        assert f == []

    def test_none_report(self):
        f, w = [], []
        validate_component_status("test", None, f, w, False)
        assert any(i["code"] == "missing_component_report" for i in f)

    def test_failed_status(self):
        f, w = [], []
        validate_component_status("test", {"status": "fail", "strict_mode": True, "failure_count": 1}, f, w, False)
        codes = [i["code"] for i in f]
        assert "component_not_passed" in codes

    def test_not_strict_warning(self):
        f, w = [], []
        validate_component_status("test", {"status": "pass", "strict_mode": False, "failure_count": 0}, f, w, False)
        assert any(i["code"] == "component_not_strict" for i in w)

    def test_strict_required(self):
        f, w = [], []
        validate_component_status("test", {"status": "pass", "strict_mode": False, "failure_count": 0}, f, w, True)
        assert any(i["code"] == "component_not_strict" for i in f)


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def test_all_pass(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_one_component_fails(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        # Overwrite leakage_report with failing status
        _write_json(paths["leakage_report"], {"status": "fail", "strict_mode": True, "failure_count": 1})
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "fail"
        codes = [f["code"] for f in report["failures"]]
        assert "component_not_passed" in codes

    def test_missing_component_file(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        # Point one component to nonexistent file
        paths["leakage_report"] = tmp_path / "nonexistent.json"
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_or_missing_json" in codes

    def test_invalid_json_component(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        paths["leakage_report"].write_text("{bad json", encoding="utf-8")
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_or_missing_json" in codes

    def test_manifest_not_passed(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["status"] = "fail"
        _write_json(paths["manifest"], m)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "manifest_not_passed" in codes

    def test_manifest_comparison_mismatch(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["comparison"] = {"matched": False, "hash_mismatches": ["train.csv"]}
        _write_json(paths["manifest"], m)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "manifest_comparison_mismatch" in codes

    def test_manifest_missing_files(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["files"] = []
        _write_json(paths["manifest"], m)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "manifest_missing_files" in codes

    def test_execution_attestation_policy_disabled(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["key_assurance"]["policy"]["require_revocation_list"] = False
        _write_json(paths["execution_attestation_report"], ea)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "execution_attestation_policy_disabled" in codes

    def test_witness_quorum_not_met(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["witness_quorum"]["validated_witnesses"] = 1
        _write_json(paths["execution_attestation_report"], ea)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "execution_attestation_witness_quorum_not_met" in codes

    def test_metric_report_missing_actual(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        mr = _good_metric_report()
        mr["actual_metric"] = "not_a_number"
        _write_json(paths["metric_report"], mr)
        cmd = _build_cmd(tmp_path, paths)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "metric_report_missing_actual" in codes

    def test_report_structure(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        cmd = _build_cmd(tmp_path, paths)
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "quality_score" in report["summary"]
        assert "failure_count" in report
        assert "warning_count" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        assert "artifacts" in report["summary"]

    def test_quality_score_pass(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        cmd = _build_cmd(tmp_path, paths)
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["summary"]["quality_score"] == 100.0

    def test_strict_mode_non_strict_components_fail(self, tmp_path: Path):
        paths = _make_all_artifacts(tmp_path)
        # Set one component to non-strict
        comp = _good_component()
        comp["strict_mode"] = False
        _write_json(paths["leakage_report"], comp)
        cmd = _build_cmd(tmp_path, paths, extra_args=["--strict"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "component_not_strict" in codes


# ── direct main() tests (for coverage) ──────────────────────────────────────

from publication_gate import main as pub_main


def _build_argv(tmp_path, paths, strict=False):
    """Build sys.argv for direct main() call."""
    argv = ["pub"]
    for arg_name, comp_name in zip(COMPONENT_ARGS, COMPONENT_NAMES):
        argv.extend([arg_name, str(paths[comp_name])])
    argv.extend(["--report", str(tmp_path / "rpt.json")])
    if strict:
        argv.append("--strict")
    return argv


class TestPublicationGateMain:
    def test_all_pass(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 0
        data = json.loads((tmp_path / "rpt.json").read_text())
        assert data["status"] == "pass"
        assert data["summary"]["quality_score"] == 100.0

    def test_component_fail(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        _write_json(paths["leakage_report"], {"status": "fail", "strict_mode": True, "failure_count": 1})
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_manifest_comparison_mismatch(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["comparison"] = {"matched": False}
        _write_json(paths["manifest"], m)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2
        data = json.loads((tmp_path / "rpt.json").read_text())
        codes = [f["code"] for f in data["failures"]]
        assert "manifest_comparison_mismatch" in codes

    def test_attestation_policy_disabled(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["key_assurance"]["policy"]["require_revocation_list"] = False
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_metric_missing_actual(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        mr = _good_metric_report()
        mr["actual_metric"] = None
        _write_json(paths["metric_report"], mr)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_strict_mode(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths, strict=True))
        rc = pub_main()
        assert rc == 0
        data = json.loads((tmp_path / "rpt.json").read_text())
        assert data["strict_mode"] is True

    def test_no_report_flag(self, tmp_path, monkeypatch, capsys):
        paths = _make_all_artifacts(tmp_path)
        argv = ["pub"]
        for arg_name, comp_name in zip(COMPONENT_ARGS, COMPONENT_NAMES):
            argv.extend([arg_name, str(paths[comp_name])])
        monkeypatch.setattr("sys.argv", argv)
        rc = pub_main()
        assert rc == 0

    def test_attestation_missing_summary(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        _write_json(paths["execution_attestation_report"], {"status": "pass", "failure_count": 0})
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_missing_policy(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["key_assurance"] = {}
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_witness_quorum_missing(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        del ea["summary"]["witness_quorum"]
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_role_distinctness_missing(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        del ea["summary"]["authority_role_distinctness"]
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_manifest_missing_files(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["files"] = []
        _write_json(paths["manifest"], m)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_none_report(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        _write_json(paths["execution_attestation_report"], [1, 2, 3])
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_min_witness_too_low(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["key_assurance"]["policy"]["min_witness_count"] = 1
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_witness_not_present(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["witness_quorum"]["present"] = False
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_witness_not_required(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["witness_quorum"]["required"] = False
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_validated_witnesses_invalid(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["witness_quorum"]["validated_witnesses"] = "bad"
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_witness_min_count_invalid(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["witness_quorum"]["min_witness_count"] = "bad"
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_attestation_role_status_fail(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        ea = _good_execution_attestation()
        ea["summary"]["authority_role_distinctness"]["status"] = "fail"
        _write_json(paths["execution_attestation_report"], ea)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_manifest_has_errors(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        m["errors"] = [{"msg": "checksum mismatch"}]
        _write_json(paths["manifest"], m)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2

    def test_manifest_no_comparison(self, tmp_path, monkeypatch):
        paths = _make_all_artifacts(tmp_path)
        m = _good_manifest()
        del m["comparison"]
        _write_json(paths["manifest"], m)
        monkeypatch.setattr("sys.argv", _build_argv(tmp_path, paths))
        rc = pub_main()
        assert rc == 2
