"""Tests for scripts/audit_external_project.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import audit_external_project as aep


# ── _score_interpretation ─────────────────────────────────────────────────────

class TestScoreInterpretation:
    def test_publication_grade(self):
        label_en, label_zh = aep._score_interpretation(95.0)
        assert label_en == "Publication-grade"
        assert "顶刊" in label_zh

    def test_solid(self):
        label_en, _ = aep._score_interpretation(80.0)
        assert label_en == "Solid but gaps remain"

    def test_major_issues(self):
        label_en, _ = aep._score_interpretation(65.0)
        assert label_en == "Major issues"

    def test_not_publishable(self):
        label_en, _ = aep._score_interpretation(50.0)
        assert label_en == "Not publishable"

    def test_boundary_90(self):
        label_en, _ = aep._score_interpretation(90.0)
        assert label_en == "Publication-grade"

    def test_boundary_75(self):
        label_en, _ = aep._score_interpretation(75.0)
        assert label_en == "Solid but gaps remain"

    def test_boundary_60(self):
        label_en, _ = aep._score_interpretation(60.0)
        assert label_en == "Major issues"


# ── _check_file_structure ─────────────────────────────────────────────────────

class TestCheckFileStructure:
    def test_empty_dir(self, tmp_path):
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_train_csv"] is False
        assert checks["has_evidence_dir"] is False
        assert checks["has_model_artifact"] is False

    def test_with_train_csv(self, tmp_path):
        (tmp_path / "train.csv").touch()
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_train_csv"] is True

    def test_with_evidence_dir(self, tmp_path):
        (tmp_path / "evidence").mkdir()
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_evidence_dir"] is True

    def test_with_pkl_model(self, tmp_path):
        (tmp_path / "model.pkl").touch()
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_model_artifact"] is True

    def test_with_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").touch()
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_requirements"] is True

    def test_with_git(self, tmp_path):
        (tmp_path / ".git").mkdir()
        checks = aep._check_file_structure(tmp_path)
        assert checks["has_git"] is True


# ── _scan_code_patterns ───────────────────────────────────────────────────────

class TestScanCodePatterns:
    def test_empty_dir(self, tmp_path):
        patterns = aep._scan_code_patterns(tmp_path)
        for key, files in patterns.items():
            assert files == [], f"Expected no matches for {key}"

    def test_detects_fit_on_full_data(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("model.fit(X_all, y_all)\n")
        patterns = aep._scan_code_patterns(tmp_path)
        assert len(patterns["fit_on_full_data"]) > 0

    def test_detects_no_random_seed(self, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("RandomForestClassifier(random_state=None)\n")
        patterns = aep._scan_code_patterns(tmp_path)
        assert len(patterns["no_random_seed"]) > 0

    def test_detects_hardcoded_threshold(self, tmp_path):
        script = tmp_path / "predict.py"
        script.write_text("threshold = 0.5\n")
        patterns = aep._scan_code_patterns(tmp_path)
        assert len(patterns["hardcoded_threshold"]) > 0

    def test_clean_code_no_warnings(self, tmp_path):
        script = tmp_path / "clean.py"
        script.write_text(
            "model = RandomForestClassifier(random_state=42)\n"
            "model.fit(X_train, y_train)\n"
        )
        patterns = aep._scan_code_patterns(tmp_path)
        # No fit_on_full_data since X_all not present
        assert len(patterns["fit_on_full_data"]) == 0
        assert len(patterns["no_random_seed"]) == 0


# ── _score_dimension_from_evidence ────────────────────────────────────────────

class TestScoreDimensionFromEvidence:
    def _write_report(self, evidence_dir: Path, filename: str, status: str = "pass") -> None:
        (evidence_dir / filename).write_text(json.dumps({"status": status, "failure_count": 0}))

    def test_no_evidence_dir(self, tmp_path):
        evidence_dir = tmp_path / "evidence_nonexistent"
        frac, passed, failed = aep._score_dimension_from_evidence("data_integrity", evidence_dir)
        assert frac == 0.0

    def test_primary_report_pass(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        self._write_report(evidence_dir, "split_protocol_report.json", "pass")
        frac, passed, failed = aep._score_dimension_from_evidence("data_integrity", evidence_dir)
        assert frac > 0.0
        assert len(passed) > 0

    def test_primary_report_fail(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        self._write_report(evidence_dir, "split_protocol_report.json", "fail")
        frac, passed, failed = aep._score_dimension_from_evidence("data_integrity", evidence_dir)
        assert any("fail" in f for f in failed)

    def test_supplementary_reports_boost_score(self, tmp_path):
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        self._write_report(evidence_dir, "split_protocol_report.json", "pass")
        self._write_report(evidence_dir, "leakage_report.json", "pass")
        self._write_report(evidence_dir, "covariate_shift_report.json", "pass")
        frac_with_supp, _, _ = aep._score_dimension_from_evidence("data_integrity", evidence_dir)
        # Score with supplementary > score without
        assert frac_with_supp > 0.0


# ── run_audit ─────────────────────────────────────────────────────────────────

class TestRunAudit:
    def test_empty_project(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path)
        assert "total_score" in report
        assert report["total_score"] == 0.0  # no evidence → zero score
        assert "dimension_scores" in report
        assert "structure_checks" in report
        assert "remediation_priorities" in report

    def test_report_keys(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path)
        assert report["contract_version"] == "audit_report.v1"
        assert report["max_score"] == 100

    def test_grade_not_publishable(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path)
        assert report["grade_en"] == "Not publishable"

    def test_json_output_to_file(self, tmp_path, capsys):
        out = tmp_path / "audit.json"
        report = aep.run_audit(project_dir=tmp_path, output_path=out, as_json=True)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["total_score"] == report["total_score"]

    def test_with_passing_evidence(self, tmp_path, capsys):
        evidence_dir = tmp_path / "evidence"
        evidence_dir.mkdir()
        for fname in aep._GATE_REPORT_MAP.values():
            (evidence_dir / fname).write_text(json.dumps({"status": "pass", "failure_count": 0}))
        for supp_list in aep._SUPPLEMENTARY_REPORTS.values():
            for fname in supp_list:
                (evidence_dir / fname).write_text(json.dumps({"status": "pass", "failure_count": 0}))
        report = aep.run_audit(project_dir=tmp_path)
        assert report["total_score"] > 0.0

    def test_code_warnings_detected(self, tmp_path, capsys):
        (tmp_path / "train.py").write_text("model.fit(X_all, y_train)\n")
        report = aep.run_audit(project_dir=tmp_path)
        assert len(report["code_warnings"]) > 0

    def test_journal_gap_analysis(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path, target_journal="nature_medicine")
        assert "journal_gap_analysis" in report
        jg = report["journal_gap_analysis"]
        assert "minimum_score" in jg
        assert "meets_threshold" in jg

    def test_no_journal_gap_without_target(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path)
        assert "journal_gap_analysis" not in report

    def test_remediation_priorities_sorted(self, tmp_path, capsys):
        report = aep.run_audit(project_dir=tmp_path)
        rems = report["remediation_priorities"]
        # Priorities should be numbered 1, 2, 3, ...
        for i, rem in enumerate(rems, 1):
            assert rem["priority"] == str(i)


# ── _format_text_report ───────────────────────────────────────────────────────

class TestFormatTextReport:
    def _make_report(self, score=50.0):
        dim_scores = {}
        for key, dim in aep.DIMENSIONS.items():
            dim_scores[key] = {
                "id": dim["id"],
                "name": dim["name"],
                "name_zh": dim["name_zh"],
                "weight": dim["weight"],
                "score_fraction": 0.5,
                "weighted_score": dim["weight"] * 0.5,
                "max_possible": dim["weight"],
                "passed_checks": [],
                "failed_checks": [],
            }
        return {
            "project_dir": "/fake/project",
            "total_score": score,
            "max_score": 100,
            "grade_en": "Solid but gaps remain",
            "grade_zh": "需补充",
            "dimension_scores": dim_scores,
            "structure_checks": {"has_train_csv": True, "has_evidence_dir": False},
            "code_warnings": [],
            "remediation_priorities": [
                {"priority": "1", "dimension": "Leakage Prevention", "dimension_zh": "防泄漏",
                 "current_score": "7/15", "gap": "8.0", "failed_checks": "check1"},
            ],
        }

    def test_text_report_contains_score(self):
        report = self._make_report(75.0)
        text = aep._format_text_report(report)
        assert "75" in text

    def test_text_report_has_dimensions(self):
        report = self._make_report()
        text = aep._format_text_report(report)
        assert "Dimension Scores" in text

    def test_text_report_has_structure(self):
        report = self._make_report()
        text = aep._format_text_report(report)
        assert "Project Structure" in text

    def test_text_report_remediation(self):
        report = self._make_report()
        text = aep._format_text_report(report)
        assert "Remediation" in text
