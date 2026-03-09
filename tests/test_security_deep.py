"""Tests for deep security hardening features in _security.py."""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _security import (
    SecurityError,
    RestrictedUnpickler,
    safe_pickle_load,
    encrypt_evidence,
    decrypt_evidence,
    encrypt_file,
    decrypt_file,
    secure_delete,
    secure_cleanup_dir,
)


# ---------------------------------------------------------------------------
# RestrictedUnpickler tests
# ---------------------------------------------------------------------------

class TestRestrictedUnpickler:
    def test_allows_builtin_types(self) -> None:
        """Basic Python types should deserialize fine."""
        data = {"key": "value", "num": 42, "lst": [1, 2, 3]}
        buf = io.BytesIO()
        pickle.dump(data, buf)
        buf.seek(0)
        result = safe_pickle_load(buf)
        assert result == data

    def _make_exploit_pickle(self, module: str, func: str, arg: str) -> bytes:
        """Build a pickle that calls module.func(arg) via REDUCE opcode."""
        # pickle protocol 2: GLOBAL + arg + REDUCE
        return (
            b"\x80\x02"  # PROTO 2
            + pickle.GLOBAL
            + f"{module}\n{func}\n".encode()
            + pickle.SHORT_BINUNICODE
            + len(arg).to_bytes(1, "little")
            + arg.encode()
            + pickle.TUPLE1
            + pickle.REDUCE
            + pickle.STOP
        )

    def test_blocks_os_system(self) -> None:
        """os.system should be blocked."""
        payload = self._make_exploit_pickle("os", "system", "echo pwned")
        buf = io.BytesIO(payload)
        with pytest.raises(SecurityError, match="Blocked dangerous callable"):
            safe_pickle_load(buf)

    def test_blocks_subprocess(self) -> None:
        """subprocess.Popen should be blocked."""
        payload = self._make_exploit_pickle("subprocess", "Popen", "ls")
        buf = io.BytesIO(payload)
        with pytest.raises(SecurityError):
            safe_pickle_load(buf)

    def test_blocks_unknown_module(self) -> None:
        """Modules not in whitelist should be rejected."""
        payload = self._make_exploit_pickle("shutil", "rmtree", "/tmp")
        buf = io.BytesIO(payload)
        with pytest.raises(SecurityError, match="Disallowed module"):
            safe_pickle_load(buf)

    def test_allows_numpy_array(self) -> None:
        """numpy arrays should deserialize."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        arr = np.array([1, 2, 3])
        buf = io.BytesIO()
        pickle.dump(arr, buf)
        buf.seek(0)
        result = safe_pickle_load(buf)
        assert list(result) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Evidence encryption tests
# ---------------------------------------------------------------------------

class TestEvidenceEncryption:
    def test_encrypt_decrypt_roundtrip(self) -> None:
        key = os.urandom(32)
        plaintext = b"sensitive medical data 12345"
        encrypted = encrypt_evidence(plaintext, key)
        assert encrypted != plaintext
        assert encrypted.startswith(b"MLGG-ENC-v1\x00")
        decrypted = decrypt_evidence(encrypted, key)
        assert decrypted == plaintext

    def test_different_nonces(self) -> None:
        key = os.urandom(32)
        data = b"same data"
        enc1 = encrypt_evidence(data, key)
        enc2 = encrypt_evidence(data, key)
        assert enc1 != enc2  # Different nonces → different ciphertexts

    def test_wrong_key_fails(self) -> None:
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        encrypted = encrypt_evidence(b"secret", key1)
        with pytest.raises((SecurityError, Exception)):
            decrypt_evidence(encrypted, key2)

    def test_tampered_ciphertext_fails(self) -> None:
        key = os.urandom(32)
        encrypted = encrypt_evidence(b"data", key)
        # Tamper with last byte
        tampered = encrypted[:-1] + bytes([(encrypted[-1] + 1) % 256])
        with pytest.raises((SecurityError, Exception)):
            decrypt_evidence(tampered, key)

    def test_invalid_header_fails(self) -> None:
        key = os.urandom(32)
        with pytest.raises(SecurityError, match="Invalid encryption header"):
            decrypt_evidence(b"WRONG-HEADER" + os.urandom(40), key)

    def test_encrypt_decrypt_file(self, tmp_path: Path) -> None:
        key = os.urandom(32)
        original = tmp_path / "report.json"
        original.write_text('{"status": "pass"}', encoding="utf-8")

        enc_path = encrypt_file(original, key)
        assert enc_path.suffix == ".enc"
        assert enc_path.exists()

        decrypted = decrypt_file(enc_path, key)
        assert decrypted == b'{"status": "pass"}'

    def test_empty_data(self) -> None:
        key = os.urandom(32)
        encrypted = encrypt_evidence(b"", key)
        decrypted = decrypt_evidence(encrypted, key)
        assert decrypted == b""

    def test_large_data(self) -> None:
        key = os.urandom(32)
        data = os.urandom(1024 * 1024)  # 1 MB
        encrypted = encrypt_evidence(data, key)
        decrypted = decrypt_evidence(encrypted, key)
        assert decrypted == data


# ---------------------------------------------------------------------------
# Secure delete tests
# ---------------------------------------------------------------------------

class TestSecureDelete:
    def test_secure_delete_removes_file(self, tmp_path: Path) -> None:
        f = tmp_path / "secret.txt"
        f.write_text("sensitive data")
        assert f.exists()
        secure_delete(f)
        assert not f.exists()

    def test_secure_delete_nonexistent(self, tmp_path: Path) -> None:
        f = tmp_path / "nonexistent.txt"
        secure_delete(f)  # Should not raise

    def test_secure_delete_directory_ignored(self, tmp_path: Path) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        secure_delete(d)  # Should not delete directories
        assert d.exists()

    def test_secure_cleanup_dir(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file_{i}.txt").write_text(f"data {i}")
        count = secure_cleanup_dir(tmp_path, "*.txt")
        assert count == 5
        assert len(list(tmp_path.glob("*.txt"))) == 0

    def test_secure_cleanup_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "keep.json").write_text("{}")
        (tmp_path / "delete.tmp").write_text("temp")
        (tmp_path / "delete2.tmp").write_text("temp2")
        count = secure_cleanup_dir(tmp_path, "*.tmp")
        assert count == 2
        assert (tmp_path / "keep.json").exists()


# ---------------------------------------------------------------------------
# CLI sanitization tests
# ---------------------------------------------------------------------------

class TestCLISanitization:
    def test_sanitize_long_arg(self) -> None:
        import argparse
        from _gate_framework import sanitize_cli_args
        args = argparse.Namespace(path="x" * 5000, strict=True)
        issues = sanitize_cli_args(args)
        assert any("exceeds" in i for i in issues)

    def test_sanitize_null_byte(self) -> None:
        import argparse
        from _gate_framework import sanitize_cli_args
        args = argparse.Namespace(path="/tmp/foo\x00bar")
        issues = sanitize_cli_args(args)
        assert any("null byte" in i for i in issues)

    def test_sanitize_normal_args(self) -> None:
        import argparse
        from _gate_framework import sanitize_cli_args
        args = argparse.Namespace(
            report="/tmp/report.json", strict=True, timeout=0
        )
        issues = sanitize_cli_args(args)
        assert issues == []
