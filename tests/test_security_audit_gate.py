"""Tests for security_audit_gate.py — the 29th pipeline gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from security_audit_gate import main as gate_main


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _run_gate(evidence_dir: str, report_path: str, strict: bool = False, model_dir: str = None) -> Dict[str, Any]:
    """Run the gate via main() and return the report dict."""
    argv = ["security_audit_gate", "--evidence-dir", evidence_dir, "--report", report_path]
    if strict:
        argv.append("--strict")
    if model_dir:
        argv.extend(["--model-dir", model_dir])
    with mock.patch("sys.argv", argv):
        rc = gate_main()
    report = json.loads(Path(report_path).read_text())
    return report, rc


class TestSecurityAuditGateBasic:
    def test_pass_with_signed_model_and_manifest(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        models = tmp_path / "models"
        models.mkdir()

        # Create a model and sign it
        model = models / "model.pkl"
        model.write_bytes(b"test-model-data")

        from _security import sign_model_artifact, ArtifactManifest
        sign_model_artifact(model)

        # Create evidence files and manifest
        _write_json(evidence / "evaluation_report.json", {"metrics": {"roc_auc": 0.85}})
        manifest = ArtifactManifest()
        manifest.add_file(evidence / "evaluation_report.json")
        manifest.save(evidence / ".manifest.json")

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        assert rc == 0
        assert report["status"] == "pass"
        assert report["gate_name"] == "security_audit_gate"
        assert report["failure_count"] == 0

    def test_fail_unsigned_model(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        models = tmp_path / "models"
        models.mkdir()
        (models / "model.pkl").write_bytes(b"unsigned-model")

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        assert rc == 2
        assert report["status"] == "fail"
        unsigned = [f for f in report["failures"] if f["code"] == "unsigned_model"]
        assert len(unsigned) == 1

    def test_warn_no_manifest(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "eval.json", {"ok": True})

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        # No model dir → no model failures, but manifest warning
        no_manifest = [w for w in report["warnings"] if w["code"] == "manifest_missing"]
        assert len(no_manifest) == 1

    def test_strict_promotes_warnings(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "eval.json", {"ok": True})

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"), strict=True)
        assert rc == 2
        assert report["status"] == "fail"
        assert report["strict_mode"] is True
        # Warnings promoted to failures
        assert report["warning_count"] == 0
        assert report["failure_count"] > 0


class TestSecurityAuditGateChecks:
    def test_detects_sensitive_data(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "bad_report.json", {"api_key": "sk-secret-123"})

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        sensitive = [f for f in report["failures"] if f["code"] == "sensitive_data_in_evidence"]
        assert len(sensitive) >= 1

    def test_detects_tampered_manifest(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()

        _write_json(evidence / "eval.json", {"original": True})

        from _security import ArtifactManifest
        manifest = ArtifactManifest()
        manifest.add_file(evidence / "eval.json")
        manifest.save(evidence / ".manifest.json")

        # Tamper with the file
        _write_json(evidence / "eval.json", {"tampered": True})

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        integrity = [f for f in report["failures"] if f["code"] == "manifest_integrity_failure"]
        assert len(integrity) >= 1

    def test_dependency_integrity_passes(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        dep = report["summary"]["dependency_integrity"]
        assert dep["verified"] is True

    def test_report_has_all_summary_sections(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        summary = report["summary"]
        assert "model_signatures" in summary
        assert "evidence_manifest" in summary
        assert "dependency_integrity" in summary
        assert "file_permissions" in summary
        assert "sensitive_data_scan" in summary
        assert "artifact_sizes" in summary

    def test_no_model_dir_no_model_checks(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        # model_dir doesn't exist → no model signature checks
        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"),
                               model_dir=str(tmp_path / "nonexistent_models"))
        sig = report["summary"]["model_signatures"]
        assert sig["models_checked"] == 0

    def test_envelope_version_present(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        assert "envelope_version" in report
        assert "execution_timestamp_utc" in report
        assert "execution_time_seconds" in report

    def test_missing_evidence_dir_returns_2(self, tmp_path: Path) -> None:
        argv = ["security_audit_gate", "--evidence-dir", str(tmp_path / "missing"),
                "--report", str(tmp_path / "report.json")]
        with mock.patch("sys.argv", argv):
            rc = gate_main()
        assert rc == 2


class TestSecurityAuditGateModelSignatures:
    def test_tampered_model_critical_failure(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        models = tmp_path / "models"
        models.mkdir()

        model = models / "model.pkl"
        model.write_bytes(b"original-model-data")

        from _security import sign_model_artifact
        sign_model_artifact(model)

        # Tamper with model after signing
        model.write_bytes(b"tampered-model-data")

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        assert rc == 2
        critical = [f for f in report["failures"]
                    if f["code"] == "model_signature_invalid" and f["severity"] == "critical"]
        assert len(critical) == 1

    def test_multiple_models_all_checked(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        models = tmp_path / "models"
        models.mkdir()

        from _security import sign_model_artifact
        for name in ["model_a.pkl", "model_b.pkl"]:
            m = models / name
            m.write_bytes(f"model-{name}".encode())
            sign_model_artifact(m)

        report, rc = _run_gate(str(evidence), str(tmp_path / "report.json"))
        sig = report["summary"]["model_signatures"]
        assert sig["models_checked"] == 2
        assert sig["models_verified"] == 2
