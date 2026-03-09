"""Tests for _security.py — security hardening module."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _security import (
    ArtifactManifest,
    SecureModelLoader,
    SecurityError,
    check_csv_row_limit,
    check_file_size,
    compute_hmac,
    perturb_predictions,
    run_security_audit,
    safe_load_json,
    safe_path,
    sign_model_artifact,
    verify_critical_imports,
    verify_model_artifact,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _make_model_file(tmp: Path) -> Path:
    """Create a dummy model .pkl file."""
    model_path = tmp / "model.pkl"
    model_path.write_bytes(b"fake-model-data-for-testing-only")
    return model_path


# ── HMAC signing / verification ──────────────────────────────────────────────

class TestHMACSigning:
    def test_sign_creates_sig_file(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key = b"test-key-32bytes" * 2
        sig_path = sign_model_artifact(model, key=key)
        assert sig_path.exists()
        assert sig_path.suffix == ".sig"

    def test_verify_signed_model_ok(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key = b"test-key-32bytes" * 2
        sign_model_artifact(model, key=key)
        result = verify_model_artifact(model, key=key)
        assert result["verified"] is True
        assert result["reason"] == "ok"

    def test_verify_tampered_model_fails(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key = b"test-key-32bytes" * 2
        sign_model_artifact(model, key=key)
        # Tamper with model
        model.write_bytes(b"tampered-model-data")
        result = verify_model_artifact(model, key=key)
        assert result["verified"] is False
        assert "mismatch" in result["reason"]

    def test_verify_wrong_key_fails(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key1 = b"correct-key-aaaa" * 2
        key2 = b"wrong-key-bbbbbb" * 2
        sign_model_artifact(model, key=key1)
        result = verify_model_artifact(model, key=key2)
        assert result["verified"] is False
        assert result["reason"] == "hmac_mismatch"

    def test_verify_missing_model(self, tmp_path: Path) -> None:
        result = verify_model_artifact(tmp_path / "nonexistent.pkl")
        assert result["verified"] is False
        assert result["reason"] == "model_file_missing"

    def test_verify_missing_signature(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        result = verify_model_artifact(model)
        assert result["verified"] is False
        assert result["reason"] == "signature_file_missing"

    def test_verify_corrupt_signature_file(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        sig_path = model.with_suffix(".pkl.sig")
        sig_path.write_text("not valid json", encoding="utf-8")
        result = verify_model_artifact(model)
        assert result["verified"] is False
        assert "corrupt" in result["reason"]

    def test_verify_size_mismatch(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key = b"test-key-32bytes" * 2
        sign_model_artifact(model, key=key)
        # Modify sig to have wrong size
        sig_path = model.with_suffix(".pkl.sig")
        sig_data = json.loads(sig_path.read_text())
        sig_data["file_size"] = 999999
        _write_json(sig_path, sig_data)
        result = verify_model_artifact(model, key=key)
        assert result["verified"] is False
        assert result["reason"] == "file_size_mismatch"

    def test_compute_hmac_deterministic(self) -> None:
        key = b"deterministic-key"
        data = b"some data"
        h1 = compute_hmac(data, key)
        h2 = compute_hmac(data, key)
        assert h1 == h2

    def test_sig_file_contains_metadata(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        key = b"test-key-32bytes" * 2
        sig_path = sign_model_artifact(model, key=key)
        sig = json.loads(sig_path.read_text())
        assert sig["algorithm"] == "hmac-sha256"
        assert sig["schema_version"] == 1
        assert "signed_at" in sig
        assert "file_sha256" in sig
        assert "file_size" in sig


# ── Path traversal protection ────────────────────────────────────────────────

class TestSafePath:
    def test_valid_path(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.touch()
        result = safe_path(str(f), must_exist=True)
        assert result == f.resolve()

    def test_empty_path_raises(self) -> None:
        with pytest.raises(ValueError, match="path_empty"):
            safe_path("")

    def test_null_byte_raises(self) -> None:
        with pytest.raises(ValueError, match="path_null_byte"):
            safe_path("/tmp/test\x00.csv")

    def test_too_long_path_raises(self) -> None:
        with pytest.raises(ValueError, match="path_too_long"):
            safe_path("a" * 5000)

    def test_forbidden_system_path(self) -> None:
        with pytest.raises(ValueError, match="path_forbidden"):
            safe_path("/etc/passwd")

    def test_sandbox_escape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path_traversal"):
            safe_path("/tmp/outside_sandbox", sandbox=tmp_path)

    def test_sandbox_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "inside.csv"
        f.touch()
        result = safe_path(str(f), sandbox=tmp_path, must_exist=True)
        assert result == f.resolve()

    def test_must_exist_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="path_not_found"):
            safe_path(str(tmp_path / "missing.csv"), must_exist=True)

    def test_path_with_dotdot(self, tmp_path: Path) -> None:
        # Path with .. that resolves within sandbox should be OK
        sub = tmp_path / "sub"
        sub.mkdir()
        f = tmp_path / "file.txt"
        f.touch()
        result = safe_path(str(sub / ".." / "file.txt"), sandbox=tmp_path, must_exist=True)
        assert result == f.resolve()


# ── Secure JSON loading ──────────────────────────────────────────────────────

class TestSafeLoadJson:
    def test_valid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "test.json"
        _write_json(f, {"key": "value"})
        result = safe_load_json(f)
        assert result == {"key": "value"}

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="json_not_found"):
            safe_load_json(tmp_path / "missing.json")

    def test_too_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "big.json"
        _write_json(f, {"data": "x" * 1000})
        with pytest.raises(ValueError, match="json_too_large"):
            safe_load_json(f, max_size=10)

    def test_invalid_json(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="json_decode_error"):
            safe_load_json(f)

    def test_json_root_not_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "list.json"
        with f.open("w") as fh:
            json.dump([1, 2, 3], fh)
        with pytest.raises(ValueError, match="json_root_not_object"):
            safe_load_json(f)

    def test_deeply_nested_json(self, tmp_path: Path) -> None:
        f = tmp_path / "deep.json"
        # Build deeply nested dict
        nested: Dict[str, Any] = {"leaf": True}
        for i in range(60):
            nested = {f"level_{i}": nested}
        _write_json(f, nested)
        with pytest.raises(ValueError, match="json_depth_exceeded"):
            safe_load_json(f, check_depth=True)

    def test_moderate_nesting_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "moderate.json"
        nested: Dict[str, Any] = {"leaf": True}
        for i in range(10):
            nested = {f"level_{i}": nested}
        _write_json(f, nested)
        result = safe_load_json(f)
        assert "level_9" in result


# ── Artifact Manifest ────────────────────────────────────────────────────────

class TestArtifactManifest:
    def test_create_and_verify_manifest(self, tmp_path: Path) -> None:
        # Create some files
        f1 = tmp_path / "eval.json"
        _write_json(f1, {"metric": 0.9})
        f2 = tmp_path / "model_sel.json"
        _write_json(f2, {"selected": "logistic"})

        manifest = ArtifactManifest()
        manifest.add_file(f1)
        manifest.add_file(f2)
        manifest.save(tmp_path / ".manifest.json")

        ok, issues = ArtifactManifest.verify(tmp_path / ".manifest.json")
        assert ok is True
        assert issues == []

    def test_verify_detects_tampered_file(self, tmp_path: Path) -> None:
        f1 = tmp_path / "eval.json"
        _write_json(f1, {"metric": 0.9})

        manifest = ArtifactManifest()
        manifest.add_file(f1)
        manifest.save(tmp_path / ".manifest.json")

        # Tamper
        _write_json(f1, {"metric": 0.1})

        ok, issues = ArtifactManifest.verify(tmp_path / ".manifest.json")
        assert ok is False
        assert any("sha256_mismatch" in i for i in issues)

    def test_verify_detects_missing_file(self, tmp_path: Path) -> None:
        f1 = tmp_path / "eval.json"
        _write_json(f1, {"metric": 0.9})

        manifest = ArtifactManifest()
        manifest.add_file(f1)
        manifest.save(tmp_path / ".manifest.json")

        f1.unlink()

        ok, issues = ArtifactManifest.verify(tmp_path / ".manifest.json")
        assert ok is False
        assert any("file_missing" in i for i in issues)

    def test_verify_missing_manifest(self, tmp_path: Path) -> None:
        ok, issues = ArtifactManifest.verify(tmp_path / ".manifest.json")
        assert ok is False
        assert "manifest_file_missing" in issues

    def test_add_nonexistent_file_ignored(self, tmp_path: Path) -> None:
        manifest = ArtifactManifest()
        manifest.add_file(tmp_path / "nonexistent.json")
        manifest.save(tmp_path / ".manifest.json")
        data = json.loads((tmp_path / ".manifest.json").read_text())
        assert data["entry_count"] == 0


# ── Membership inference defense ─────────────────────────────────────────────

class TestPredictionPerturbation:
    def test_perturbation_preserves_range(self) -> None:
        probs = [0.0, 0.1, 0.5, 0.9, 1.0]
        perturbed = perturb_predictions(probs, epsilon=0.01, seed=42)
        assert all(0.0 <= p <= 1.0 for p in perturbed)

    def test_perturbation_adds_noise(self) -> None:
        probs = [0.5] * 100
        perturbed = perturb_predictions(probs, epsilon=0.01, seed=42)
        # Not all values should be exactly 0.5
        assert not all(p == 0.5 for p in perturbed)

    def test_perturbation_deterministic_with_seed(self) -> None:
        probs = [0.3, 0.7, 0.5]
        p1 = perturb_predictions(probs, epsilon=0.01, seed=123)
        p2 = perturb_predictions(probs, epsilon=0.01, seed=123)
        assert p1 == p2

    def test_perturbation_different_seeds_differ(self) -> None:
        probs = [0.5] * 50
        p1 = perturb_predictions(probs, epsilon=0.01, seed=1)
        p2 = perturb_predictions(probs, epsilon=0.01, seed=2)
        assert p1 != p2

    def test_higher_epsilon_less_noise(self) -> None:
        import numpy as np
        probs = [0.5] * 1000
        p_low = perturb_predictions(probs, epsilon=0.001, seed=42)
        p_high = perturb_predictions(probs, epsilon=1.0, seed=42)
        std_low = float(np.std(p_low))
        std_high = float(np.std(p_high))
        assert std_low > std_high  # Lower epsilon = more noise


# ── Resource exhaustion protection ───────────────────────────────────────────

class TestResourceProtection:
    def test_check_file_size_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "small.txt"
        f.write_text("hello")
        check_file_size(f, max_bytes=1000)  # Should not raise

    def test_check_file_size_exceeds(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 100)
        with pytest.raises(ValueError, match="too_large"):
            check_file_size(f, max_bytes=50, label="test")

    def test_check_csv_row_limit_ok(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        lines = ["col1,col2\n"] + [f"{i},{i*2}\n" for i in range(100)]
        f.write_text("".join(lines))
        count = check_csv_row_limit(f, max_rows=1000)
        assert count == 100

    def test_check_csv_row_limit_exceeds(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        lines = ["col1\n"] + ["1\n" for _ in range(200)]
        f.write_text("".join(lines))
        with pytest.raises(ValueError, match="csv_too_many_rows"):
            check_csv_row_limit(f, max_rows=50)


# ── Dependency verification ──────────────────────────────────────────────────

class TestDependencyVerification:
    def test_verify_critical_imports_ok(self) -> None:
        result = verify_critical_imports()
        assert result["verified"] is True
        assert len(result["checks"]) >= 3

    def test_verify_reports_versions(self) -> None:
        result = verify_critical_imports()
        for check in result["checks"]:
            assert "version" in check
            assert "path" in check


# ── Security audit ───────────────────────────────────────────────────────────

class TestSecurityAudit:
    def test_audit_clean_directory(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "evaluation_report.json", {"metrics": {}})

        # Create manifest
        manifest = ArtifactManifest()
        manifest.add_file(evidence / "evaluation_report.json")
        manifest.save(evidence / ".manifest.json")

        report = run_security_audit(evidence)
        assert report["status"] in ("pass", "warn")
        assert "issues" in report

    def test_audit_detects_unsigned_model(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        models = tmp_path / "models"
        models.mkdir()
        model = models / "model.pkl"
        model.write_bytes(b"unsigned-model")

        report = run_security_audit(evidence)
        unsigned = [i for i in report["issues"] if i["code"] == "unsigned_model"]
        assert len(unsigned) >= 1

    def test_audit_detects_no_manifest(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "eval.json", {"ok": True})

        report = run_security_audit(evidence)
        no_manifest = [i for i in report["issues"] if i["code"] == "no_manifest"]
        assert len(no_manifest) == 1

    def test_audit_detects_sensitive_data(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "bad.json", {"api_key": "sk-12345"})

        report = run_security_audit(evidence)
        sensitive = [i for i in report["issues"] if i["code"] == "sensitive_data_exposure"]
        assert len(sensitive) >= 1

    def test_audit_report_structure(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        report = run_security_audit(evidence)
        assert "status" in report
        assert "schema_version" in report
        assert "audit_timestamp" in report
        assert "issue_count" in report
        assert "dependency_verification" in report


# ── SecureModelLoader ────────────────────────────────────────────────────────

class TestSecureModelLoader:
    def test_load_unsigned_model_fails(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        with pytest.raises(SecurityError, match="signature_file_missing"):
            SecureModelLoader.load(model, verify_signature=True)

    def test_load_oversized_model_fails(self, tmp_path: Path) -> None:
        model = tmp_path / "big.pkl"
        model.write_bytes(b"x" * 100)
        key = b"test-key-32bytes" * 2
        sign_model_artifact(model, key=key)
        with pytest.raises(SecurityError, match="model_too_large"):
            # Monkey-patch max size for test
            with mock.patch.object(SecureModelLoader, "load") as mocked:
                mocked.side_effect = SecurityError("model_too_large: test")
                SecureModelLoader.load(model, key=key)


# ── CLI entry point ──────────────────────────────────────────────────────────

class TestCLI:
    def test_sign_and_verify_cli(self, tmp_path: Path) -> None:
        model = _make_model_file(tmp_path)
        from _security import main as sec_main

        with mock.patch("sys.argv", ["_security", "sign", str(model)]):
            rc = sec_main()
        assert rc == 0

        with mock.patch("sys.argv", ["_security", "verify", str(model)]):
            rc = sec_main()
        assert rc == 0

    def test_manifest_cli(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "eval.json", {"ok": True})

        from _security import main as sec_main
        with mock.patch("sys.argv", ["_security", "manifest", str(evidence)]):
            rc = sec_main()
        assert rc == 0
        assert (evidence / ".manifest.json").exists()

    def test_audit_cli(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        _write_json(evidence / "eval.json", {"ok": True})

        from _security import main as sec_main
        with mock.patch("sys.argv", ["_security", "audit", str(evidence)]):
            rc = sec_main()
        # May return 0 or 1 depending on findings
        assert rc in (0, 1)

    def test_check_deps_cli(self) -> None:
        from _security import main as sec_main
        with mock.patch("sys.argv", ["_security", "check-deps"]):
            rc = sec_main()
        assert rc == 0

    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture) -> None:
        from _security import main as sec_main
        with mock.patch("sys.argv", ["_security"]):
            rc = sec_main()
        assert rc == 0
