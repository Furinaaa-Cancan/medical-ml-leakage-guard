"""Tests for tamper-evident audit logging in _gate_utils.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _gate_utils import append_audit_entry, verify_audit_chain


class TestAuditLogging:
    def test_append_creates_log_file(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "test_gate", "pass")
        log = tmp_path / ".gate_audit.jsonl"
        assert log.exists()
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["gate_name"] == "test_gate"
        assert entry["status"] == "pass"
        assert "chain_hash" in entry
        assert "timestamp_utc" in entry
        assert "pid" in entry

    def test_chain_hash_links_entries(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "gate_a", "pass")
        append_audit_entry(tmp_path, "gate_b", "fail", failure_count=2)
        append_audit_entry(tmp_path, "gate_c", "pass", warning_count=1)

        lines = (tmp_path / ".gate_audit.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

        hashes = [json.loads(l)["chain_hash"] for l in lines]
        assert len(set(hashes)) == 3  # All unique

    def test_verify_valid_chain(self, tmp_path: Path) -> None:
        for name in ["gate_1", "gate_2", "gate_3"]:
            append_audit_entry(tmp_path, name, "pass")

        result = verify_audit_chain(tmp_path)
        assert result["valid"] is True
        assert result["entries"] == 3

    def test_verify_empty_dir(self, tmp_path: Path) -> None:
        result = verify_audit_chain(tmp_path)
        assert result["valid"] is True
        assert result["entries"] == 0

    def test_verify_detects_tampered_entry(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "gate_1", "pass")
        append_audit_entry(tmp_path, "gate_2", "pass")

        # Tamper with first entry
        log = tmp_path / ".gate_audit.jsonl"
        lines = log.read_text().strip().splitlines()
        entry = json.loads(lines[0])
        entry["status"] = "fail"  # tamper
        lines[0] = json.dumps(entry, ensure_ascii=True, sort_keys=True)
        log.write_text("\n".join(lines) + "\n")

        result = verify_audit_chain(tmp_path)
        assert result["valid"] is False
        assert result["broken_at"] == 0
        assert result["reason"] == "chain_hash_mismatch"

    def test_verify_detects_deleted_entry(self, tmp_path: Path) -> None:
        for i in range(3):
            append_audit_entry(tmp_path, f"gate_{i}", "pass")

        # Delete middle entry
        log = tmp_path / ".gate_audit.jsonl"
        lines = log.read_text().strip().splitlines()
        log.write_text(lines[0] + "\n" + lines[2] + "\n")

        result = verify_audit_chain(tmp_path)
        assert result["valid"] is False

    def test_verify_detects_corrupted_json(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "gate_1", "pass")
        log = tmp_path / ".gate_audit.jsonl"
        log.write_text("not-valid-json\n")

        result = verify_audit_chain(tmp_path)
        assert result["valid"] is False
        assert result["reason"] == "json_parse_error"

    def test_extra_metadata_included(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "gate_x", "pass", extra={"key": "value"})
        lines = (tmp_path / ".gate_audit.jsonl").read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["extra"] == {"key": "value"}

    def test_execution_time_recorded(self, tmp_path: Path) -> None:
        append_audit_entry(tmp_path, "gate_y", "pass", execution_time=1.234)
        lines = (tmp_path / ".gate_audit.jsonl").read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["execution_time_seconds"] == 1.234


class TestResolvePathHardened:
    def test_null_byte_rejected(self) -> None:
        from _gate_utils import resolve_path
        with pytest.raises(ValueError, match="Null byte"):
            resolve_path(Path("/tmp"), "/tmp/foo\x00bar")

    def test_forbidden_path_rejected(self) -> None:
        from _gate_utils import resolve_path
        with pytest.raises(ValueError, match="forbidden"):
            resolve_path(Path("/"), "/etc/passwd")

    def test_sandbox_escape_rejected(self, tmp_path: Path) -> None:
        from _gate_utils import resolve_path
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        with pytest.raises(ValueError, match="escapes sandbox"):
            resolve_path(sandbox, "/tmp/outside", sandbox=sandbox)

    def test_sandbox_valid_path(self, tmp_path: Path) -> None:
        from _gate_utils import resolve_path
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        inner = sandbox / "data.csv"
        inner.touch()
        result = resolve_path(sandbox, "data.csv", sandbox=sandbox)
        assert str(result).startswith(str(sandbox))

    def test_normal_path_resolves(self, tmp_path: Path) -> None:
        from _gate_utils import resolve_path
        result = resolve_path(tmp_path, "subdir/file.txt")
        assert str(result) == str(tmp_path / "subdir" / "file.txt")
