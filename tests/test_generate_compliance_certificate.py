"""Tests for scripts/generate_compliance_certificate.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_compliance_certificate as gcc


# ── crypto helpers ────────────────────────────────────────────────────────────

class TestCryptoHelpers:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_bytes(b"hello")
        digest = gcc.sha256_file(f)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_sha256_file_deterministic(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_bytes(b"hello world")
        assert gcc.sha256_file(f) == gcc.sha256_file(f)

    def test_sha256_str(self):
        digest = gcc.sha256_str("hello")
        assert len(digest) == 64

    def test_sha256_str_deterministic(self):
        assert gcc.sha256_str("test") == gcc.sha256_str("test")

    def test_hmac_sign_and_verify(self):
        key = b"test-key-32bytes-for-testing----"
        body = "canonical body"
        sig = gcc.hmac_sign(body, key)
        assert gcc.hmac_verify(body, key, sig) is True

    def test_hmac_verify_wrong_key(self):
        key1 = b"correct-key-32bytes-for-testing-"
        key2 = b"wrong---key-32bytes-for-testing-"
        sig = gcc.hmac_sign("body", key1)
        assert gcc.hmac_verify("body", key2, sig) is False

    def test_hmac_verify_tampered_body(self):
        key = b"test-key-32bytes-for-testing----"
        sig = gcc.hmac_sign("original", key)
        assert gcc.hmac_verify("tampered", key, sig) is False


# ── load_gate_report ──────────────────────────────────────────────────────────

class TestLoadGateReport:
    def test_load_valid(self, tmp_path):
        f = tmp_path / "report.json"
        f.write_text(json.dumps({"status": "pass"}))
        report = gcc.load_gate_report(f)
        assert report == {"status": "pass"}

    def test_load_missing(self, tmp_path):
        assert gcc.load_gate_report(tmp_path / "nonexistent.json") is None

    def test_load_malformed(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json {{{")
        assert gcc.load_gate_report(f) is None

    def test_load_non_dict(self, tmp_path):
        f = tmp_path / "array.json"
        f.write_text(json.dumps([1, 2, 3]))
        assert gcc.load_gate_report(f) is None


# ── get_gate_status ───────────────────────────────────────────────────────────

class TestGetGateStatus:
    def test_pass(self):
        assert gcc.get_gate_status({"status": "pass"}) == "pass"

    def test_fail(self):
        assert gcc.get_gate_status({"status": "fail"}) == "fail"

    def test_none(self):
        assert gcc.get_gate_status(None) == "missing"

    def test_unknown(self):
        assert gcc.get_gate_status({}) == "unknown"


# ── determine_conformance_level ───────────────────────────────────────────────

class TestDetermineConformanceLevel:
    def _all_pass(self):
        return {g: "pass" for g in gcc.GATE_NAME_TO_REPORT}

    def test_below_l1_missing_required(self):
        outcomes = {"request_contract_gate": "pass"}
        level, reasons = gcc.determine_conformance_level(outcomes, True, 95.0, {})
        assert level == "BELOW_L1"
        assert len(reasons) > 0

    def test_l1_achieved(self):
        outcomes = {g: "pass" for g in gcc.L1_REQUIRED_GATES}
        # L2 gates missing → L1
        level, reasons = gcc.determine_conformance_level(outcomes, True, 95.0, {"tripod_true_count": 20, "tripod_required_count": 27})
        assert level == "L1-Leakage-Audited"

    def test_l2_achieved(self):
        outcomes = {g: "pass" for g in gcc.L2_REQUIRED_GATES}
        reporting = {"tripod_true_count": 20, "tripod_required_count": 27, "overall_risk_of_bias": "low"}
        level, reasons = gcc.determine_conformance_level(outcomes, True, 95.0, reporting)
        assert level in ("L2-Statistically-Valid", "L3-Publication-Grade")

    def test_l3_achieved_all_pass(self):
        outcomes = self._all_pass()
        reporting = {
            "tripod_true_count": 25,
            "tripod_required_count": 27,
            "overall_risk_of_bias": "low",
        }
        level, reasons = gcc.determine_conformance_level(outcomes, True, 95.0, reporting)
        assert level == "L3-Publication-Grade"
        assert reasons == []

    def test_l3_fails_without_strict_mode(self):
        outcomes = self._all_pass()
        reporting = {
            "tripod_true_count": 25,
            "tripod_required_count": 27,
            "overall_risk_of_bias": "low",
        }
        level, reasons = gcc.determine_conformance_level(outcomes, False, 95.0, reporting)
        assert level == "L2-Statistically-Valid"
        assert any("strict" in r.lower() for r in reasons)

    def test_l3_fails_low_score(self):
        outcomes = self._all_pass()
        reporting = {
            "tripod_true_count": 25,
            "tripod_required_count": 27,
            "overall_risk_of_bias": "low",
        }
        level, reasons = gcc.determine_conformance_level(outcomes, True, 85.0, reporting)
        assert level == "L2-Statistically-Valid"
        assert any("score" in r.lower() for r in reasons)

    def test_l3_fails_high_rob(self):
        outcomes = self._all_pass()
        reporting = {
            "tripod_true_count": 25,
            "tripod_required_count": 27,
            "overall_risk_of_bias": "high",
        }
        level, reasons = gcc.determine_conformance_level(outcomes, True, 95.0, reporting)
        assert level == "L2-Statistically-Valid"
        assert any("probast" in r.lower() or "rob" in r.lower() for r in reasons)


# ── generate_certificate ──────────────────────────────────────────────────────

class TestGenerateCertificate:
    def _setup_evidence(self, evidence_dir: Path, status: str = "pass") -> None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        report = {"status": status, "failure_count": 0, "strict_mode": True}
        for fname in gcc.GATE_REPORT_FILENAMES:
            (evidence_dir / fname).write_text(json.dumps(report))

    def test_generates_certificate(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir)
        out = tmp_path / "cert.json"
        rc = gcc.generate_certificate(evidence_dir, None, out)
        assert rc == 0
        assert out.exists()
        cert = json.loads(out.read_text())
        assert "certificate_id" in cert
        assert "conformance_level" in cert
        assert "gates_summary" in cert
        assert "integrity" in cert

    def test_certificate_has_all_required_fields(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir)
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, None, out)
        cert = json.loads(out.read_text())
        for key in ("certificate_id", "mlgg_standard_version", "conformance_level",
                    "study", "issuance", "gates_summary", "evidence_manifest", "integrity"):
            assert key in cert, f"Missing key: {key}"

    def test_integrity_block_present(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir)
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, None, out)
        cert = json.loads(out.read_text())
        integrity = cert["integrity"]
        assert "signature" in integrity
        assert "body_sha256" in integrity
        assert integrity["signature_algorithm"] == "HMAC-SHA256"

    def test_gate_outcomes_included(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir)
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, None, out)
        cert = json.loads(out.read_text())
        gate_outcomes = cert["gates_summary"]["gate_outcomes"]
        assert "leakage_gate" in gate_outcomes
        assert "publication_gate" in gate_outcomes

    def test_with_request_json(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir)
        request = tmp_path / "request.json"
        request.write_text(json.dumps({"study_id": "STUDY-001", "target_name": "mortality"}))
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, request, out)
        cert = json.loads(out.read_text())
        assert cert["study"]["study_id"] == "STUDY-001"
        assert cert["study"]["target_name"] == "mortality"

    def test_all_pass_gates_counted(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        self._setup_evidence(evidence_dir, status="pass")
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, None, out)
        cert = json.loads(out.read_text())
        gs = cert["gates_summary"]
        assert gs["passed"] == len(gcc.GATE_NAME_TO_REPORT)
        assert gs["failed"] == 0

    def test_missing_evidence_dir(self, tmp_path):
        out = tmp_path / "cert.json"
        # Should still run (gate reports will be missing)
        rc = gcc.generate_certificate(tmp_path / "nonexistent", None, out)
        assert rc == 0  # generates cert with missing gates


# ── verify_certificate ────────────────────────────────────────────────────────

class TestVerifyCertificate:
    def _generate(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        for fname in gcc.GATE_REPORT_FILENAMES:
            (evidence_dir / fname).write_text(json.dumps({"status": "pass", "failure_count": 0, "strict_mode": True}))
        out = tmp_path / "cert.json"
        gcc.generate_certificate(evidence_dir, None, out)
        return out

    def test_verify_valid_cert(self, tmp_path, capsys):
        out = self._generate(tmp_path)
        rc = gcc.verify_certificate(out)
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_verify_missing_file(self, tmp_path):
        rc = gcc.verify_certificate(tmp_path / "nonexistent.json")
        assert rc == 2

    def test_verify_tampered_body(self, tmp_path):
        out = self._generate(tmp_path)
        cert = json.loads(out.read_text())
        # Tamper with the body
        cert["conformance_level"] = "TAMPERED"
        out.write_text(json.dumps(cert))
        rc = gcc.verify_certificate(out)
        assert rc == 2

    def test_verify_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{")
        rc = gcc.verify_certificate(bad)
        assert rc == 2
