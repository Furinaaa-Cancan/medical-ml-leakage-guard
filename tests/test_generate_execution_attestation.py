"""Tests for scripts/generate_execution_attestation.py.

Covers helper functions (parse_iso_utc, iso_now_utc, iso_after_days,
sha256_file, sha256_text, resolve_for_output, parse_artifact,
parse_witness, default_executor, ensure_file, count_lines,
log_boundary_hashes, ensure_revocation_file, write_json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import generate_execution_attestation as gea


class TestParseIsoUtc:
    def test_basic(self):
        result = gea.parse_iso_utc("2025-01-15T10:00:00Z")
        assert result is not None

    def test_empty(self):
        assert gea.parse_iso_utc("") is None

    def test_invalid(self):
        assert gea.parse_iso_utc("not-a-date") is None

    def test_naive(self):
        result = gea.parse_iso_utc("2025-01-15T10:00:00")
        assert result is not None


class TestIsoNowUtc:
    def test_returns_string(self):
        result = gea.iso_now_utc()
        assert isinstance(result, str)
        assert "T" in result


class TestIsoAfterDays:
    def test_basic(self):
        result = gea.iso_after_days("2025-01-15T10:00:00Z", 30)
        assert isinstance(result, str)
        parsed = gea.parse_iso_utc(result)
        assert parsed is not None
        base = gea.parse_iso_utc("2025-01-15T10:00:00Z")
        assert (parsed - base).days == 30


class TestSha256File:
    def test_basic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = gea.sha256_file(f)
        assert len(h) == 64


class TestSha256Text:
    def test_deterministic(self):
        assert gea.sha256_text("hello") == gea.sha256_text("hello")
        assert len(gea.sha256_text("hello")) == 64


class TestResolveForOutput:
    def test_relative(self, tmp_path):
        base = tmp_path / "project"
        base.mkdir()
        target = base / "sub" / "file.json"
        result = gea.resolve_for_output(base, target)
        assert "sub" in result
        assert "file.json" in result


class TestParseArtifact:
    def test_valid(self):
        name, path = gea.parse_artifact("evaluation_report=/tmp/eval.json")
        assert name == "evaluation_report"
        assert path == Path("/tmp/eval.json").resolve()

    def test_no_equals(self):
        with pytest.raises(ValueError, match="Expected NAME=PATH"):
            gea.parse_artifact("bad_format")

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            gea.parse_artifact("=/tmp/file.json")

    def test_empty_path(self):
        with pytest.raises(ValueError, match="non-empty"):
            gea.parse_artifact("name=")


class TestParseWitness:
    def test_valid(self):
        auth, pub, priv = gea.parse_witness("witness1|/tmp/pub.pem|/tmp/priv.pem")
        assert auth == "witness1"

    def test_wrong_parts(self):
        with pytest.raises(ValueError, match="Expected"):
            gea.parse_witness("only_one_part")

    def test_empty_part(self):
        with pytest.raises(ValueError, match="non-empty"):
            gea.parse_witness("||/tmp/priv.pem")


class TestDefaultExecutor:
    def test_returns_string(self):
        result = gea.default_executor()
        assert isinstance(result, str)
        assert "@" in result


class TestEnsureFile:
    def test_existing(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("data")
        gea.ensure_file(f, "test")  # should not raise

    def test_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gea.ensure_file(tmp_path / "nope.txt", "test")

    def test_directory(self, tmp_path):
        with pytest.raises(ValueError, match="must be a file"):
            gea.ensure_file(tmp_path, "test")


class TestCountLines:
    def test_basic(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("line1\nline2\nline3\n")
        assert gea.count_lines(f) == 3


class TestLogBoundaryHashes:
    def test_basic(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("first\nmiddle\nlast\n")
        total, first_h, last_h = gea.log_boundary_hashes(f)
        assert total == 3
        assert first_h == gea.sha256_text("first")
        assert last_h == gea.sha256_text("last")

    def test_single_line(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("only\n")
        total, first_h, last_h = gea.log_boundary_hashes(f)
        assert total == 1
        assert first_h == last_h


class TestEnsureRevocationFile:
    def test_create_new(self, tmp_path):
        path = tmp_path / "revocations.json"
        result = gea.ensure_revocation_file(path)
        assert path.exists()
        assert "revoked_key_ids" in result
        assert "revoked_public_key_fingerprints_sha256" in result

    def test_existing(self, tmp_path):
        path = tmp_path / "revocations.json"
        path.write_text(json.dumps({"revoked_key_ids": ["k1"], "revoked_public_key_fingerprints_sha256": []}))
        result = gea.ensure_revocation_file(path)
        assert "k1" in result["revoked_key_ids"]


class TestWriteJson:
    def test_basic(self, tmp_path):
        path = tmp_path / "out.json"
        gea.write_json(path, {"key": "value"})
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["key"] == "value"

    def test_nested_dir(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        gea.write_json(path, {"a": 1})
        assert path.exists()
