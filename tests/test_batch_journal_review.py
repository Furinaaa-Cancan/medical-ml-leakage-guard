"""Tests for scripts/batch_journal_review.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import batch_journal_review as bjr


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_audit_report(score: float = 75.0, grade: str = "Solid but gaps remain") -> dict:
    return {
        "total_score": score,
        "grade": {"label_en": grade},
        "dimensions": {
            "data_integrity": {"score": 10.0, "weight": 12.0},
            "leakage_prevention": {"score": 12.0, "weight": 15.0},
        },
        "remediation_priorities": [
            {"action": "Add bootstrap CI", "severity": "HIGH", "dimension": "statistical_validity", "score_impact": 5.0},
            {"action": "Add external cohort", "severity": "MEDIUM", "dimension": "generalization_evidence", "score_impact": 3.0},
        ],
        "journal_gap_analysis": {
            "mandatory_met": 5,
            "mandatory_total": 8,
            "unmet_mandatory": ["Provide calibration plot", "Add DCA"],
        },
    }


def _make_success_result(project_id: str, score: float = 75.0) -> bjr.AuditResult:
    return bjr.AuditResult(
        project_id=project_id,
        project_label=f"Project {project_id}",
        project_path=f"/fake/{project_id}",
        success=True,
        audit_report=_make_audit_report(score),
        elapsed_seconds=1.5,
    )


def _make_error_result(project_id: str) -> bjr.AuditResult:
    return bjr.AuditResult(
        project_id=project_id,
        project_label=f"Project {project_id}",
        project_path=f"/fake/{project_id}",
        success=False,
        error="Directory not found",
        elapsed_seconds=0.1,
    )


# ── load_manifest ─────────────────────────────────────────────────────────────

class TestLoadManifest:
    def test_basic_manifest(self, tmp_path):
        manifest = {
            "contract_version": "batch_manifest.v1",
            "projects": [
                {"id": "proj1", "path": "/data/proj1", "label": "Project One"},
                {"id": "proj2", "path": "/data/proj2"},
            ],
        }
        mf = tmp_path / "manifest.json"
        mf.write_text(json.dumps(manifest))
        entries, journals = bjr.load_manifest(mf)
        assert len(entries) == 2
        assert entries[0].id == "proj1"
        assert entries[0].label == "Project One"
        assert entries[1].id == "proj2"
        assert journals is None

    def test_manifest_with_target_journals(self, tmp_path):
        manifest = {
            "contract_version": "batch_manifest.v1",
            "projects": [{"id": "p1", "path": "/x"}],
            "target_journals": ["nature_medicine"],
        }
        mf = tmp_path / "manifest.json"
        mf.write_text(json.dumps(manifest))
        entries, journals = bjr.load_manifest(mf)
        assert journals == ["nature_medicine"]

    def test_empty_projects(self, tmp_path):
        manifest = {"projects": []}
        mf = tmp_path / "manifest.json"
        mf.write_text(json.dumps(manifest))
        entries, _ = bjr.load_manifest(mf)
        assert entries == []


# ── build_comparison_matrix ───────────────────────────────────────────────────

class TestBuildComparisonMatrix:
    def test_success_entries(self):
        results = [_make_success_result("p1", 80.0), _make_success_result("p2", 65.0)]
        matrix = bjr.build_comparison_matrix(results)
        assert len(matrix) == 2
        assert matrix[0]["total_score"] == 80.0
        assert matrix[0]["status"] == "audited"

    def test_error_entry(self):
        results = [_make_error_result("p1")]
        matrix = bjr.build_comparison_matrix(results)
        assert matrix[0]["status"] == "error"
        assert "error" in matrix[0]

    def test_mixed_results(self):
        results = [_make_success_result("p1"), _make_error_result("p2")]
        matrix = bjr.build_comparison_matrix(results)
        assert len(matrix) == 2
        statuses = {m["project"]: m["status"] for m in matrix}
        assert statuses["p1"] == "audited"
        assert statuses["p2"] == "error"


# ── build_cross_cutting_analysis ──────────────────────────────────────────────

class TestBuildCrossCuttingAnalysis:
    def test_no_successful_results(self):
        results = [_make_error_result("p1")]
        analysis = bjr.build_cross_cutting_analysis(results)
        assert analysis["most_failed_dimensions"] == []
        assert analysis["most_common_gaps"] == []

    def test_aggregates_dimensions(self):
        results = [_make_success_result("p1", 70.0), _make_success_result("p2", 60.0)]
        analysis = bjr.build_cross_cutting_analysis(results)
        assert "most_failed_dimensions" in analysis
        assert isinstance(analysis["most_failed_dimensions"], list)

    def test_common_gaps_deduplication(self):
        r1 = _make_success_result("p1")
        r2 = _make_success_result("p2")
        # Both have "Add bootstrap CI" gap
        results = [r1, r2]
        analysis = bjr.build_cross_cutting_analysis(results)
        gaps = {g["requirement"]: g["missing_in_n_projects"] for g in analysis["most_common_gaps"]}
        assert gaps.get("Add bootstrap CI", 0) == 2


# ── build_aggregated_remediation ──────────────────────────────────────────────

class TestBuildAggregatedRemediation:
    def test_deduplicates_actions(self):
        results = [_make_success_result("p1"), _make_success_result("p2")]
        rems = bjr.build_aggregated_remediation(results)
        actions = [r["action"] for r in rems]
        assert len(actions) == len(set(actions))  # no duplicates

    def test_priority_assigned(self):
        results = [_make_success_result("p1"), _make_success_result("p2")]
        rems = bjr.build_aggregated_remediation(results)
        priorities = [r["priority"] for r in rems]
        assert priorities == list(range(1, len(priorities) + 1))

    def test_highest_severity_kept(self):
        r1 = _make_success_result("p1")
        r2 = bjr.AuditResult(
            project_id="p2", project_label="p2", project_path="/x",
            success=True,
            audit_report={
                "total_score": 50.0,
                "grade": {"label_en": "Major issues"},
                "dimensions": {},
                "remediation_priorities": [
                    {"action": "Add bootstrap CI", "severity": "CRITICAL", "dimension": "stat", "score_impact": 10.0},
                ],
            },
        )
        rems = bjr.build_aggregated_remediation([r1, r2])
        ci_rem = next(r for r in rems if r["action"] == "Add bootstrap CI")
        assert ci_rem["severity"] == "CRITICAL"  # CRITICAL > HIGH

    def test_empty_results(self):
        rems = bjr.build_aggregated_remediation([_make_error_result("p1")])
        assert rems == []


# ── build_summary ─────────────────────────────────────────────────────────────

class TestBuildSummary:
    def test_basic_stats(self):
        results = [_make_success_result("p1", 92.0), _make_success_result("p2", 65.0)]
        summary = bjr.build_summary(results)
        assert summary["audited"] == 2
        assert summary["errors"] == 0
        assert summary["mean_score"] == 78.5

    def test_error_counted(self):
        results = [_make_success_result("p1", 80.0), _make_error_result("p2")]
        summary = bjr.build_summary(results)
        assert summary["audited"] == 1
        assert summary["errors"] == 1

    def test_all_errors(self):
        results = [_make_error_result("p1")]
        summary = bjr.build_summary(results)
        assert summary["audited"] == 0
        assert summary["mean_score"] == 0.0

    def test_publication_ready_count(self):
        results = [
            bjr.AuditResult("p1", "p1", "/x", True, _make_audit_report(95.0, "Publication-grade")),
            bjr.AuditResult("p2", "p2", "/x", True, _make_audit_report(70.0, "Major issues")),
        ]
        summary = bjr.build_summary(results)
        assert summary["publication_ready"] == 1
        assert summary["major_issues"] == 1


# ── format functions ──────────────────────────────────────────────────────────

class TestFormatFunctions:
    def _make_batch(self):
        results = [_make_success_result("p1", 80.0), _make_error_result("p2")]
        batch = bjr.BatchResult(
            target_journal="nature_medicine",
            total_projects=2,
            results=results,
        )
        batch.comparison_matrix = bjr.build_comparison_matrix(results)
        batch.cross_cutting_analysis = bjr.build_cross_cutting_analysis(results)
        batch.aggregated_remediation = bjr.build_aggregated_remediation(results)
        batch.summary = bjr.build_summary(results)
        return batch

    def test_json_format_valid(self):
        batch = self._make_batch()
        output = bjr.format_json_report(batch)
        parsed = json.loads(output)
        assert parsed["contract_version"] == bjr.BATCH_REPORT_CONTRACT_VERSION
        assert parsed["total_projects"] == 2

    def test_markdown_format(self):
        batch = self._make_batch()
        output = bjr.format_markdown_report(batch)
        assert "# Batch Journal Review Report" in output
        assert "## Summary" in output
        assert "## Comparison Matrix" in output

    def test_text_format(self):
        batch = self._make_batch()
        output = bjr.format_text_report(batch)
        assert "BATCH JOURNAL REVIEW REPORT" in output
        assert "PROJECT SCORES" in output

    def test_csv_format(self):
        batch = self._make_batch()
        output = bjr.format_summary_csv(batch)
        lines = output.strip().split("\n")
        # Header + 2 data rows
        assert len(lines) == 3
        assert "project_id" in lines[0]


# ── run_batch_audit (sequential) ─────────────────────────────────────────────

class TestRunBatchAuditSequential:
    def test_missing_project_dir(self, tmp_path):
        entries = [bjr.ProjectEntry(id="p1", path=str(tmp_path / "nonexistent"))]
        batch = bjr.run_batch_audit(entries, max_workers=1)
        assert batch.total_projects == 1
        assert batch.results[0].success is False
        assert "not found" in batch.results[0].error.lower()

    def test_preserves_manifest_order(self, tmp_path):
        # Create 3 project dirs (all empty so audit fails, but still returns results)
        entries = []
        for i in range(3):
            d = tmp_path / f"proj{i}"
            d.mkdir()
            entries.append(bjr.ProjectEntry(id=f"p{i}", path=str(d)))
        batch = bjr.run_batch_audit(entries, max_workers=1)
        ids = [r.project_id for r in batch.results]
        assert ids == ["p0", "p1", "p2"]
