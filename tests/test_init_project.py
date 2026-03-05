"""Tests for scripts/init_project.py helper functions and CLI.

Covers copy_template_json, make_phenotype_template, build_request_payload,
ensure_dirs, CLI --help, basic init, force overwrite, idempotent re-run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "init_project.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import init_project as ip


# ── helper functions ─────────────────────────────────────────────────────────

class TestMakePhenotypeTemplate:
    def test_structure(self):
        tpl = ip.make_phenotype_template("disease_risk")
        assert "targets" in tpl
        assert "disease_risk" in tpl["targets"]
        assert "defining_variables" in tpl["targets"]["disease_risk"]
        assert "global_forbidden_patterns" in tpl

    def test_custom_target(self):
        tpl = ip.make_phenotype_template("my_target")
        assert "my_target" in tpl["targets"]


class TestBuildRequestPayload:
    def test_fields(self):
        payload = ip.build_request_payload(
            template={},
            study_id="s1",
            run_id="r1",
            target_name="disease_risk",
            prediction_unit="patient-episode",
            index_time_col="event_time",
            label_col="y",
            patient_id_col="patient_id",
            claim_tier="publication-grade",
        )
        assert payload["study_id"] == "s1"
        assert payload["run_id"] == "r1"
        assert payload["claim_tier_target"] == "publication-grade"
        assert payload["split_paths"]["train"] == "../data/train.csv"
        assert payload["actual_primary_metric"] == 0.0


class TestEnsureDirs:
    def test_creates_dirs(self, tmp_path):
        created = ip.ensure_dirs(tmp_path)
        for d in ("configs", "data", "evidence", "models", "keys"):
            assert (tmp_path / d).exists()

    def test_idempotent(self, tmp_path):
        ip.ensure_dirs(tmp_path)
        ip.ensure_dirs(tmp_path)
        for d in ("configs", "data", "evidence", "models", "keys"):
            assert (tmp_path / d).exists()


class TestCopyTemplateJson:
    def test_write(self, tmp_path):
        dst = tmp_path / "out.json"
        result = ip.copy_template_json("performance-policy.example.json", dst, force=False)
        assert result == "written"
        assert dst.exists()
        data = json.loads(dst.read_text())
        assert isinstance(data, dict)

    def test_preserved(self, tmp_path):
        dst = tmp_path / "out.json"
        dst.write_text('{"existing": true}')
        result = ip.copy_template_json("performance-policy.example.json", dst, force=False)
        assert result == "preserved"
        data = json.loads(dst.read_text())
        assert data.get("existing") is True

    def test_force_overwrite(self, tmp_path):
        dst = tmp_path / "out.json"
        dst.write_text('{"existing": true}')
        result = ip.copy_template_json("performance-policy.example.json", dst, force=True)
        assert result == "written"
        data = json.loads(dst.read_text())
        assert "existing" not in data


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "project-root" in proc.stdout


class TestCLIInit:
    def test_basic_init(self, tmp_path):
        project = tmp_path / "testproj"
        report = tmp_path / "init_report.json"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--force",
            "--report", str(report),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert (project / "configs" / "request.json").exists()
        assert (project / "configs" / "phenotype_definitions.json").exists()
        assert (project / "configs" / "performance_policy.json").exists()
        assert report.exists()
        data = json.loads(report.read_text())
        assert data["status"] == "pass"

    def test_idempotent_no_force(self, tmp_path):
        project = tmp_path / "testproj2"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--force",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        req1 = json.loads((project / "configs" / "request.json").read_text())

        cmd2 = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
        ]
        subprocess.run(cmd2, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        req2 = json.loads((project / "configs" / "request.json").read_text())
        assert req1["study_id"] == req2["study_id"]

    def test_custom_study_id(self, tmp_path):
        project = tmp_path / "testproj3"
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "--project-root", str(project),
            "--study-id", "my-custom-study",
            "--force",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        req = json.loads((project / "configs" / "request.json").read_text())
        assert req["study_id"] == "my-custom-study"
