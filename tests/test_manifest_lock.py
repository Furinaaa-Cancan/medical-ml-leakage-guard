"""Comprehensive unit tests for scripts/manifest_lock.py."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import manifest_lock as ml
from manifest_lock import (
    compare_manifest,
    csv_summary,
    digest_map,
    file_sha256,
    parse_meta,
    utc_now,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# utc_now
# ────────────────────────────────────────────────────────

class TestUtcNow:
    def test_format(self):
        ts = utc_now()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_contains_year(self):
        ts = utc_now()
        assert "20" in ts  # Year 20xx


# ────────────────────────────────────────────────────────
# file_sha256
# ────────────────────────────────────────────────────────

class TestFileSha256:
    def test_deterministic(self, tmp_path: Path):
        p = tmp_path / "f.txt"
        p.write_text("hello", encoding="utf-8")
        assert file_sha256(p) == file_sha256(p)
        assert len(file_sha256(p)) == 64

    def test_known_hash(self, tmp_path: Path):
        p = tmp_path / "known.txt"
        content = b"test content"
        p.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert file_sha256(p) == expected

    def test_different_content(self, tmp_path: Path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("aaa", encoding="utf-8")
        p2.write_text("bbb", encoding="utf-8")
        assert file_sha256(p1) != file_sha256(p2)

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")
        h = file_sha256(p)
        assert h == hashlib.sha256(b"").hexdigest()


# ────────────────────────────────────────────────────────
# csv_summary
# ────────────────────────────────────────────────────────

class TestCsvSummary:
    def test_normal(self, tmp_path: Path):
        p = tmp_path / "data.csv"
        p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        s = csv_summary(p)
        assert s["rows"] == 2
        assert s["columns"] == 2
        assert s["header"] == ["a", "b"]
        assert s["header_sha256"] is not None
        assert len(s["header_sha256"]) == 64

    def test_header_only(self, tmp_path: Path):
        p = tmp_path / "header.csv"
        p.write_text("x,y,z\n", encoding="utf-8")
        s = csv_summary(p)
        assert s["rows"] == 0
        assert s["columns"] == 3
        assert s["header"] == ["x", "y", "z"]

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        s = csv_summary(p)
        assert s["rows"] == 0
        assert s["header"] is None

    def test_whitespace_headers(self, tmp_path: Path):
        p = tmp_path / "ws.csv"
        p.write_text(" a , b \n1,2\n", encoding="utf-8")
        s = csv_summary(p)
        assert s["header"] == ["a", "b"]

    def test_header_sha256_deterministic(self, tmp_path: Path):
        p1 = tmp_path / "d1.csv"
        p2 = tmp_path / "d2.csv"
        p1.write_text("a,b\n1,2\n", encoding="utf-8")
        p2.write_text("a,b\n3,4\n", encoding="utf-8")
        s1 = csv_summary(p1)
        s2 = csv_summary(p2)
        # Same headers → same header_sha256
        assert s1["header_sha256"] == s2["header_sha256"]


# ────────────────────────────────────────────────────────
# parse_meta
# ────────────────────────────────────────────────────────

class TestParseMeta:
    def test_normal(self):
        result = parse_meta(["key1=val1", "key2=val2"])
        assert result == {"key1": "val1", "key2": "val2"}

    def test_empty(self):
        assert parse_meta([]) == {}

    def test_value_with_equals(self):
        result = parse_meta(["key=a=b=c"])
        assert result == {"key": "a=b=c"}

    def test_whitespace(self):
        result = parse_meta([" key = val "])
        assert result == {"key": "val"}

    def test_no_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            parse_meta(["invalid"])

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            parse_meta(["=value"])


# ────────────────────────────────────────────────────────
# digest_map
# ────────────────────────────────────────────────────────

class TestDigestMap:
    def test_normal(self):
        manifest = {
            "files": [
                {"path": "/a.csv", "sha256": "aaa"},
                {"path": "/b.csv", "sha256": "bbb"},
            ]
        }
        result = digest_map(manifest)
        assert result == {"/a.csv": "aaa", "/b.csv": "bbb"}

    def test_empty_files(self):
        assert digest_map({"files": []}) == {}

    def test_no_files_key(self):
        assert digest_map({}) == {}

    def test_files_not_list(self):
        assert digest_map({"files": "invalid"}) == {}

    def test_entry_not_dict(self):
        assert digest_map({"files": ["not_a_dict"]}) == {}

    def test_missing_path(self):
        assert digest_map({"files": [{"sha256": "abc"}]}) == {}

    def test_missing_sha256(self):
        assert digest_map({"files": [{"path": "/a.csv"}]}) == {}

    def test_empty_strings_skipped(self):
        assert digest_map({"files": [{"path": "", "sha256": "abc"}]}) == {}


# ────────────────────────────────────────────────────────
# compare_manifest
# ────────────────────────────────────────────────────────

class TestCompareManifest:
    def test_matching(self):
        m = {"files": [{"path": "/a", "sha256": "aaa"}]}
        result = compare_manifest(m, m)
        assert result["matched"] is True
        assert result["missing_in_current"] == []
        assert result["missing_in_baseline"] == []
        assert result["hash_mismatches"] == []

    def test_hash_mismatch(self):
        current = {"files": [{"path": "/a", "sha256": "new"}]}
        baseline = {"files": [{"path": "/a", "sha256": "old"}]}
        result = compare_manifest(current, baseline)
        assert result["matched"] is False
        assert len(result["hash_mismatches"]) == 1
        assert result["hash_mismatches"][0]["path"] == "/a"

    def test_missing_in_current(self):
        current = {"files": []}
        baseline = {"files": [{"path": "/a", "sha256": "aaa"}]}
        result = compare_manifest(current, baseline)
        assert result["matched"] is False
        assert "/a" in result["missing_in_current"]

    def test_missing_in_baseline(self):
        current = {"files": [{"path": "/a", "sha256": "aaa"}]}
        baseline = {"files": []}
        result = compare_manifest(current, baseline)
        assert result["matched"] is False
        assert "/a" in result["missing_in_baseline"]

    def test_mixed_issues(self):
        current = {"files": [
            {"path": "/a", "sha256": "new_a"},
            {"path": "/c", "sha256": "ccc"},
        ]}
        baseline = {"files": [
            {"path": "/a", "sha256": "old_a"},
            {"path": "/b", "sha256": "bbb"},
        ]}
        result = compare_manifest(current, baseline)
        assert result["matched"] is False
        assert "/b" in result["missing_in_current"]
        assert "/c" in result["missing_in_baseline"]
        assert len(result["hash_mismatches"]) == 1

    def test_empty_manifests(self):
        result = compare_manifest({"files": []}, {"files": []})
        assert result["matched"] is True


# ────────────────────────────────────────────────────────
# CLI tests (subprocess)
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, args, cwd=None):
        cmd = [sys.executable, str(SCRIPTS_DIR / "manifest_lock.py")] + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                              cwd=cwd or str(SCRIPTS_DIR.parent))

    def test_single_file(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f), "--output", str(out)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        manifest = json.loads(out.read_text())
        assert manifest["status"] == "pass"
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["sha256"] == file_sha256(f)

    def test_multiple_files(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f1), str(f2), "--output", str(out)])
        assert result.returncode == 0
        manifest = json.loads(out.read_text())
        assert len(manifest["files"]) == 2

    def test_csv_gets_summary(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f), "--output", str(out)])
        assert result.returncode == 0
        manifest = json.loads(out.read_text())
        assert "csv_summary" in manifest["files"][0]
        assert manifest["files"][0]["csv_summary"]["rows"] == 2

    def test_nonexistent_input(self, tmp_path: Path):
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(tmp_path / "missing.txt"), "--output", str(out)])
        assert result.returncode == 2
        manifest = json.loads(out.read_text())
        assert manifest["status"] == "fail"
        assert manifest["files"][0]["exists"] is False

    def test_directory_input(self, tmp_path: Path):
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(tmp_path), "--output", str(out)])
        assert result.returncode == 2
        manifest = json.loads(out.read_text())
        assert manifest["status"] == "fail"

    def test_unchanged_baseline_match(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"
        # Generate baseline
        self._run(["--inputs", str(f), "--output", str(baseline)])
        # Compare against baseline (same file, unchanged)
        result = self._run(["--inputs", str(f), "--output", str(current),
                            "--compare-with", str(baseline)])
        assert result.returncode == 0
        manifest = json.loads(current.read_text())
        assert manifest["comparison"]["matched"] is True

    def test_changed_baseline_mismatch(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("original", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"
        # Generate baseline
        self._run(["--inputs", str(f), "--output", str(baseline)])
        # Modify file
        f.write_text("modified", encoding="utf-8")
        # Compare → mismatch
        result = self._run(["--inputs", str(f), "--output", str(current),
                            "--compare-with", str(baseline)])
        assert result.returncode == 2
        manifest = json.loads(current.read_text())
        assert manifest["comparison"]["matched"] is False
        assert len(manifest["comparison"]["hash_mismatches"]) == 1

    def test_missing_baseline(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f), "--output", str(out),
                            "--compare-with", str(tmp_path / "nonexistent.json")])
        assert result.returncode == 2
        manifest = json.loads(out.read_text())
        assert manifest["status"] == "fail"

    def test_meta_key_value(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f), "--output", str(out),
                            "--meta", "study=test_study", "--meta", "version=1.0"])
        assert result.returncode == 0
        manifest = json.loads(out.read_text())
        assert manifest["meta"]["study"] == "test_study"
        assert manifest["meta"]["version"] == "1.0"

    def test_invalid_meta(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = self._run(["--inputs", str(f), "--output", str(out),
                            "--meta", "invalid_no_equals"])
        assert result.returncode == 2

    def test_fingerprint_unchanged(self, tmp_path: Path):
        """Same file content → same SHA256 in manifest."""
        f = tmp_path / "data.txt"
        f.write_text("deterministic content", encoding="utf-8")
        out1 = tmp_path / "m1.json"
        out2 = tmp_path / "m2.json"
        self._run(["--inputs", str(f), "--output", str(out1)])
        self._run(["--inputs", str(f), "--output", str(out2)])
        m1 = json.loads(out1.read_text())
        m2 = json.loads(out2.read_text())
        assert m1["files"][0]["sha256"] == m2["files"][0]["sha256"]

    def test_report_structure(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        self._run(["--inputs", str(f), "--output", str(out)])
        manifest = json.loads(out.read_text())
        assert "status" in manifest
        assert "created_at_utc" in manifest
        assert "hash_algorithm" in manifest
        assert "files" in manifest
        assert "meta" in manifest
        assert "errors" in manifest
        entry = manifest["files"][0]
        assert "path" in entry
        assert "sha256" in entry
        assert "size_bytes" in entry
        assert "mtime_utc" in entry


# ────────────────────────────────────────────────────────
# Direct main() tests
# ────────────────────────────────────────────────────────

class TestMainSingleFile:
    def test_pass(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(f), "--output", str(out)])
        rc = ml.main()
        assert rc == 0
        m = json.loads(out.read_text())
        assert m["status"] == "pass"
        assert len(m["files"]) == 1


class TestMainMultipleFiles:
    def test_two_files(self, tmp_path, monkeypatch):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(f1), str(f2), "--output", str(out)])
        rc = ml.main()
        assert rc == 0
        m = json.loads(out.read_text())
        assert len(m["files"]) == 2


class TestMainCsvSummary:
    def test_csv_gets_summary(self, tmp_path, monkeypatch):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(f), "--output", str(out)])
        rc = ml.main()
        assert rc == 0
        m = json.loads(out.read_text())
        assert "csv_summary" in m["files"][0]
        assert m["files"][0]["csv_summary"]["rows"] == 2


class TestMainMissingInput:
    def test_nonexistent(self, tmp_path, monkeypatch):
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(tmp_path / "missing.txt"), "--output", str(out)])
        rc = ml.main()
        assert rc == 2
        m = json.loads(out.read_text())
        assert m["status"] == "fail"


class TestMainDirInput:
    def test_directory_input(self, tmp_path, monkeypatch):
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(tmp_path), "--output", str(out)])
        rc = ml.main()
        assert rc == 2
        m = json.loads(out.read_text())
        assert m["status"] == "fail"


class TestMainMeta:
    def test_meta_key_value(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(out),
            "--meta", "study=test_study", "--meta", "version=1.0",
        ])
        rc = ml.main()
        assert rc == 0
        m = json.loads(out.read_text())
        assert m["meta"]["study"] == "test_study"


class TestMainInvalidMeta:
    def test_invalid_meta(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(out),
            "--meta", "invalid_no_equals",
        ])
        rc = ml.main()
        assert rc == 2


class TestMainCompareMatch:
    def test_baseline_match(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        # Generate baseline first
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(f), "--output", str(baseline)])
        ml.main()
        # Compare against baseline
        current = tmp_path / "current.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(current),
            "--compare-with", str(baseline),
        ])
        rc = ml.main()
        assert rc == 0
        m = json.loads(current.read_text())
        assert m["comparison"]["matched"] is True


class TestMainCompareMismatch:
    def test_baseline_mismatch(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("original", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        monkeypatch.setattr("sys.argv", ["ml", "--inputs", str(f), "--output", str(baseline)])
        ml.main()
        f.write_text("modified", encoding="utf-8")
        current = tmp_path / "current.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(current),
            "--compare-with", str(baseline),
        ])
        rc = ml.main()
        assert rc == 2
        m = json.loads(current.read_text())
        assert m["comparison"]["matched"] is False


class TestMainMissingBaseline:
    def test_missing_baseline(self, tmp_path, monkeypatch):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(out),
            "--compare-with", str(tmp_path / "nonexistent.json"),
        ])
        rc = ml.main()
        assert rc == 2
        m = json.loads(out.read_text())
        assert m["status"] == "fail"


class TestMainBaselineNotDict:
    def test_baseline_not_dict(self, tmp_path, monkeypatch):
        """Baseline JSON root is a list, not a dict → fail."""
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text("[1,2,3]", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(out),
            "--compare-with", str(baseline),
        ])
        rc = ml.main()
        assert rc == 2
        m = json.loads(out.read_text())
        assert m["status"] == "fail"
        assert any("Failed to read baseline" in e for e in m["errors"])


class TestMainBaselineCorrupt:
    def test_corrupt_baseline(self, tmp_path, monkeypatch):
        """Baseline file is not valid JSON → fail."""
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text("{corrupt", encoding="utf-8")
        out = tmp_path / "manifest.json"
        monkeypatch.setattr("sys.argv", [
            "ml", "--inputs", str(f), "--output", str(out),
            "--compare-with", str(baseline),
        ])
        rc = ml.main()
        assert rc == 2
        m = json.loads(out.read_text())
        assert m["status"] == "fail"
        assert any("Failed to read baseline" in e for e in m["errors"])
