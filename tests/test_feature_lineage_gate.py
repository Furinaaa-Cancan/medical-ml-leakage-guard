"""Comprehensive unit tests for scripts/feature_lineage_gate.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from feature_lineage_gate import (
    build_lineage_key_index,
    collect_transitive_candidates,
    normalize_lineage_payload,
    resolve_lineage_key,
)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _write_csv(path: Path, headers: list) -> Path:
    path.write_text(",".join(headers) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _make_setup(tmp_path, def_spec, lineage_spec, headers=None):
    if headers is None:
        headers = ["patient_id", "y", "age", "bp_systolic", "creatinine"]
    return {
        "def_spec": _write_json(tmp_path / "def.json", def_spec),
        "lineage": _write_json(tmp_path / "lineage.json", lineage_spec),
        "train": _write_csv(tmp_path / "train.csv", headers),
    }


# ────────────────────────────────────────────────────────
# normalize_lineage_payload
# ────────────────────────────────────────────────────────

class TestNormalizeLineagePayload:
    def test_list_format(self):
        raw = {"feat_a": ["raw_x", "raw_y"]}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": ["raw_x", "raw_y"]}

    def test_dict_with_ancestors(self):
        raw = {"feat_a": {"ancestors": ["raw_x"]}}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": ["raw_x"]}

    def test_string_ancestor(self):
        raw = {"feat_a": "raw_x"}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": ["raw_x"]}

    def test_features_wrapper(self):
        raw = {"features": {"feat_a": ["raw_x"]}}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": ["raw_x"]}

    def test_empty_string_skipped(self):
        raw = {"feat_a": ["", "  ", "raw_x"]}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": ["raw_x"]}

    def test_empty_key_skipped(self):
        raw = {"": ["raw_x"], "feat_a": ["raw_y"]}
        result = normalize_lineage_payload(raw)
        assert "feat_a" in result
        assert "" not in result

    def test_dict_no_ancestors_key(self):
        raw = {"feat_a": {"other": "value"}}
        result = normalize_lineage_payload(raw)
        assert result == {"feat_a": []}

    def test_empty_payload(self):
        assert normalize_lineage_payload({}) == {}


# ────────────────────────────────────────────────────────
# build_lineage_key_index
# ────────────────────────────────────────────────────────

class TestBuildLineageKeyIndex:
    def test_no_collisions(self):
        lineage = {"feat_a": [], "feat_b": []}
        index, collisions = build_lineage_key_index(lineage)
        assert index["feata"] == "feat_a"
        assert index["featb"] == "feat_b"
        assert collisions == {}

    def test_collision(self):
        lineage = {"feat_a": [], "Feat_A": []}
        index, collisions = build_lineage_key_index(lineage)
        assert len(collisions) == 1
        nk = list(collisions.keys())[0]
        assert len(collisions[nk]) == 2


# ────────────────────────────────────────────────────────
# resolve_lineage_key
# ────────────────────────────────────────────────────────

class TestResolveLineageKey:
    def test_exact_match(self):
        lineage = {"feat_a": []}
        index = {"feata": "feat_a"}
        assert resolve_lineage_key("feat_a", lineage, index) == "feat_a"

    def test_normalized_match(self):
        lineage = {"feat_a": []}
        index = {"feata": "feat_a"}
        assert resolve_lineage_key("Feat_A", lineage, index) == "feat_a"

    def test_no_match(self):
        lineage = {"feat_a": []}
        index = {"feata": "feat_a"}
        assert resolve_lineage_key("unknown", lineage, index) is None


# ────────────────────────────────────────────────────────
# collect_transitive_candidates
# ────────────────────────────────────────────────────────

class TestCollectTransitiveCandidates:
    def test_no_ancestors(self):
        lineage = {"feat_a": []}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert "feat_a" in candidates
        assert cycles == []
        assert overflow is False

    def test_single_ancestor(self):
        lineage = {"feat_a": ["raw_x"], "raw_x": []}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert "raw_x" in candidates
        assert "feat_a" in candidates

    def test_transitive_chain(self):
        lineage = {"feat_a": ["feat_b"], "feat_b": ["feat_c"], "feat_c": []}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert {"feat_a", "feat_b", "feat_c"} <= candidates

    def test_cycle_detection(self):
        lineage = {"feat_a": ["feat_b"], "feat_b": ["feat_a"]}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert len(cycles) > 0

    def test_unresolved_ancestor(self):
        lineage = {"feat_a": ["unknown_raw"]}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert "unknown_raw" in candidates

    def test_feature_not_in_lineage(self):
        lineage = {"other": []}
        index, _ = build_lineage_key_index(lineage)
        candidates, cycles, overflow = collect_transitive_candidates("feat_a", lineage, index)
        assert candidates == {"feat_a"}


# ────────────────────────────────────────────────────────
# CLI tests
# ────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, tmp_path, setup, target="sepsis", extra_args=None):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "feature_lineage_gate.py"),
            "--target", target,
            "--definition-spec", str(setup["def_spec"]),
            "--lineage-spec", str(setup["lineage"]),
            "--train", str(setup["train"]),
            "--report", str(tmp_path / "report.json"),
        ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    def _def_spec(self, defining=None, forbidden=None, patterns=None):
        return {
            "targets": {
                "sepsis": {
                    "defining_variables": defining or [],
                    "forbidden_variables": forbidden or [],
                    "forbidden_patterns": patterns or [],
                }
            },
            "global_forbidden_variables": [],
            "global_forbidden_patterns": [],
        }

    def test_no_leakage(self, tmp_path: Path):
        lineage = {"age": [], "bp_systolic": [], "creatinine": []}
        setup = _make_setup(tmp_path, self._def_spec(defining=["lactate"]), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass"

    def test_direct_ancestor_leakage(self, tmp_path: Path):
        """Feature 'creatinine' derives from forbidden 'lactate'."""
        lineage = {"age": [], "bp_systolic": [], "creatinine": ["lactate"]}
        setup = _make_setup(tmp_path, self._def_spec(defining=["lactate"]), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "lineage_definition_leakage" in codes

    def test_transitive_ancestor_leakage(self, tmp_path: Path):
        """creatinine → intermediate → lactate (forbidden)."""
        lineage = {
            "age": [],
            "bp_systolic": [],
            "creatinine": ["intermediate"],
            "intermediate": ["lactate"],
        }
        setup = _make_setup(tmp_path, self._def_spec(defining=["lactate"]), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "lineage_definition_leakage" in codes

    def test_pattern_in_lineage(self, tmp_path: Path):
        """Pattern '^lact' matches ancestor 'lactate_level'."""
        lineage = {"age": [], "bp_systolic": [], "creatinine": ["lactate_level"]}
        setup = _make_setup(tmp_path, self._def_spec(patterns=["^lact"]), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "lineage_proxy_leakage" in codes

    def test_missing_lineage_warning(self, tmp_path: Path):
        """Features without lineage → warning (non-strict)."""
        lineage = {"age": []}  # bp_systolic and creatinine have no lineage
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 0  # warning only
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        assert "missing_lineage_entries" in warn_codes

    def test_missing_lineage_strict_fails(self, tmp_path: Path):
        """Strict mode + missing lineage → fail."""
        lineage = {"age": []}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup, extra_args=["--strict"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "missing_lineage_entries" in codes

    def test_allow_missing_lineage_strict(self, tmp_path: Path):
        """--allow-missing-lineage + --strict → warning instead of fail."""
        lineage = {"age": []}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup,
                           extra_args=["--strict", "--allow-missing-lineage"])
        report = json.loads((tmp_path / "report.json").read_text())
        warn_codes = [w["code"] for w in report["warnings"]]
        # missing_lineage should be warning, not failure
        fail_codes = [f["code"] for f in report["failures"]]
        assert "missing_lineage_entries" in warn_codes
        assert "missing_lineage_entries" not in fail_codes

    def test_cycle_detected(self, tmp_path: Path):
        """Cyclic lineage → fail."""
        lineage = {
            "age": [],
            "bp_systolic": [],
            "creatinine": ["feat_x"],
            "feat_x": ["creatinine"],
        }
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "lineage_cycle_detected" in codes

    def test_missing_definition_spec(self, tmp_path: Path):
        lineage_path = _write_json(tmp_path / "lineage.json", {})
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {
            "def_spec": tmp_path / "nonexistent.json",
            "lineage": lineage_path,
            "train": train_path,
        }
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_definition_spec" in codes

    def test_missing_lineage_spec(self, tmp_path: Path):
        def_spec_path = _write_json(tmp_path / "def.json", self._def_spec())
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {
            "def_spec": def_spec_path,
            "lineage": tmp_path / "nonexistent.json",
            "train": train_path,
        }
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_lineage_spec" in codes

    def test_invalid_lineage_json(self, tmp_path: Path):
        def_spec_path = _write_json(tmp_path / "def.json", self._def_spec())
        lineage_path = tmp_path / "lineage.json"
        lineage_path.write_text("{bad json", encoding="utf-8")
        train_path = _write_csv(tmp_path / "train.csv", ["patient_id", "y", "age"])
        setup = {"def_spec": def_spec_path, "lineage": lineage_path, "train": train_path}
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "invalid_lineage_spec" in codes

    def test_lineage_key_collision(self, tmp_path: Path):
        """Two lineage keys normalize to same → fail."""
        lineage = {"feat_a": [], "Feat_A": [], "age": [], "bp_systolic": [], "creatinine": []}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup)
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "lineage_key_normalization_collision" in codes

    def test_target_not_found(self, tmp_path: Path):
        lineage = {"age": []}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup, target="diabetes")
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "target_not_found" in codes

    def test_empty_lineage_strict_fails(self, tmp_path: Path):
        lineage = {}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        result = self._run(tmp_path, setup, extra_args=["--strict"])
        assert result.returncode == 2
        report = json.loads((tmp_path / "report.json").read_text())
        codes = [f["code"] for f in report["failures"]]
        assert "empty_lineage_map" in codes

    def test_report_structure(self, tmp_path: Path):
        lineage = {"age": [], "bp_systolic": [], "creatinine": []}
        setup = _make_setup(tmp_path, self._def_spec(), lineage)
        self._run(tmp_path, setup)
        report = json.loads((tmp_path / "report.json").read_text())
        assert "status" in report
        assert "strict_mode" in report
        assert "target" in report.get("summary", {})
        assert "lineage_spec" in report.get("input_files", {})
        assert "summary" in report
        s = report["summary"]
        assert "lineage_feature_count" in s
        assert "checked_feature_count" in s
        assert "lineage_coverage_ratio" in s

    def test_ignore_cols_excludes_features(self, tmp_path: Path):
        """Ignored column with forbidden ancestor should not trigger failure."""
        lineage = {"age": [], "bp_systolic": [], "creatinine": ["lactate"]}
        setup = _make_setup(tmp_path, self._def_spec(defining=["lactate"]), lineage)
        result = self._run(tmp_path, setup, extra_args=["--ignore-cols", "creatinine"])
        assert result.returncode == 0
