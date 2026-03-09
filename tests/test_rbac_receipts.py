"""Tests for RBAC access control and signed execution receipts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _security import (
    AccessControl,
    Role,
    SecurityError,
    get_current_user,
    sign_execution_receipt,
    verify_execution_receipt,
)


# ---------------------------------------------------------------------------
# RBAC tests
# ---------------------------------------------------------------------------

class TestRBAC:
    def test_default_role_is_viewer(self) -> None:
        ac = AccessControl()
        assert ac.get_role("unknown_user") == Role.VIEWER

    def test_assign_and_get_role(self, tmp_path: Path) -> None:
        config = tmp_path / ".mlgg_rbac.json"
        ac = AccessControl(config)
        ac.assign_role("alice", Role.ADMIN)
        assert ac.get_role("alice") == Role.ADMIN
        assert config.exists()

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        config = tmp_path / ".mlgg_rbac.json"
        ac1 = AccessControl(config)
        ac1.assign_role("bob", Role.OPERATOR)

        ac2 = AccessControl(config)
        assert ac2.get_role("bob") == Role.OPERATOR

    def test_invalid_role_rejected(self) -> None:
        ac = AccessControl()
        with pytest.raises(ValueError, match="Unknown role"):
            ac.assign_role("eve", "superuser")

    def test_admin_has_all_permissions(self) -> None:
        ac = AccessControl()
        ac.assign_role("admin_user", Role.ADMIN)
        assert ac.check_permission("admin_user", "pipeline.run")
        assert ac.check_permission("admin_user", "user.manage")
        assert ac.check_permission("admin_user", "evidence.decrypt")

    def test_viewer_limited_permissions(self) -> None:
        ac = AccessControl()
        assert ac.check_permission("viewer_user", "evidence.read")
        assert ac.check_permission("viewer_user", "audit.read")
        assert not ac.check_permission("viewer_user", "pipeline.run")
        assert not ac.check_permission("viewer_user", "model.sign")

    def test_operator_permissions(self) -> None:
        ac = AccessControl()
        ac.assign_role("op", Role.OPERATOR)
        assert ac.check_permission("op", "pipeline.run")
        assert ac.check_permission("op", "model.sign")
        assert not ac.check_permission("op", "user.manage")
        assert not ac.check_permission("op", "evidence.decrypt")

    def test_auditor_permissions(self) -> None:
        ac = AccessControl()
        ac.assign_role("auditor", Role.AUDITOR)
        assert ac.check_permission("auditor", "evidence.read")
        assert ac.check_permission("auditor", "audit.verify")
        assert not ac.check_permission("auditor", "pipeline.run")
        assert not ac.check_permission("auditor", "model.sign")

    def test_require_permission_passes(self) -> None:
        ac = AccessControl()
        ac.assign_role("admin", Role.ADMIN)
        ac.require_permission("admin", "pipeline.run")  # Should not raise

    def test_require_permission_fails(self) -> None:
        ac = AccessControl()
        with pytest.raises(SecurityError, match="Access denied"):
            ac.require_permission("viewer_user", "pipeline.run")

    def test_list_permissions(self) -> None:
        ac = AccessControl()
        ac.assign_role("op", Role.OPERATOR)
        perms = ac.list_permissions("op")
        assert "pipeline.run" in perms
        assert isinstance(perms, list)
        assert perms == sorted(perms)  # Sorted

    def test_config_file_permissions(self, tmp_path: Path) -> None:
        config = tmp_path / ".mlgg_rbac.json"
        ac = AccessControl(config)
        ac.assign_role("test", Role.ADMIN)
        mode = config.stat().st_mode & 0o777
        assert mode == 0o600

    def test_get_current_user(self) -> None:
        user = get_current_user()
        assert isinstance(user, str)
        assert len(user) > 0


# ---------------------------------------------------------------------------
# Execution receipt tests
# ---------------------------------------------------------------------------

class TestExecutionReceipts:
    def test_sign_and_verify(self, tmp_path: Path) -> None:
        gate_results = {"leakage_gate": "pass", "publication_gate": "pass"}
        receipt_path = sign_execution_receipt(tmp_path, gate_results, "pass")
        assert receipt_path.exists()

        result = verify_execution_receipt(receipt_path)
        assert result["valid"] is True
        assert result["final_status"] == "pass"
        assert result["gate_count"] == 2
        assert result["passed"] == 2
        assert result["failed"] == 0

    def test_tampered_receipt_fails(self, tmp_path: Path) -> None:
        gate_results = {"gate_a": "pass"}
        receipt_path = sign_execution_receipt(tmp_path, gate_results, "pass")

        # Tamper with the receipt
        data = json.loads(receipt_path.read_text())
        data["final_status"] = "fail"
        receipt_path.write_text(json.dumps(data, indent=2))

        result = verify_execution_receipt(receipt_path)
        assert result["valid"] is False
        assert result["reason"] == "signature_mismatch"

    def test_missing_receipt(self, tmp_path: Path) -> None:
        result = verify_execution_receipt(tmp_path / "nonexistent.json")
        assert result["valid"] is False
        assert result["reason"] == "receipt_not_found"

    def test_receipt_contains_executor(self, tmp_path: Path) -> None:
        receipt_path = sign_execution_receipt(tmp_path, {}, "pass")
        data = json.loads(receipt_path.read_text())
        assert "executor" in data
        assert "hostname" in data
        assert "timestamp_utc" in data
        assert "hmac_signature" in data

    def test_receipt_with_failures(self, tmp_path: Path) -> None:
        gate_results = {"gate_a": "pass", "gate_b": "fail", "gate_c": "pass"}
        receipt_path = sign_execution_receipt(tmp_path, gate_results, "fail")
        result = verify_execution_receipt(receipt_path)
        assert result["valid"] is True
        assert result["failed"] == 1
        assert result["passed"] == 2
