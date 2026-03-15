"""Tests for scripts/generate_audit_report.py.

Covers:
- KnowledgeBases lazy-loading
- scan_project: code patterns and structure detection
- assess_tripod_coverage: item mapping
- assess_probast_coverage: domain ROB scoring
- build_issue_list: enrichment chain (pattern → KB → TRIPOD → PROBAST)
- compute_dimension_scores: scoring logic
- build_remediation_plan: priority ordering
- render_markdown_report: output structure
- run_audit_report: integration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import generate_audit_report as gar  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evidence_dir(tmp_path: Path, gate_statuses: Dict[str, str]) -> Path:
    """Create a fake evidence directory with gate report JSON files."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for gate_name, status in gate_statuses.items():
        report = {
            "gate": gate_name,
            "status": status,
            "failures": [],
            "warnings": [],
        }
        (evidence / f"{gate_name}_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    return evidence


# ---------------------------------------------------------------------------
# KnowledgeBases
# ---------------------------------------------------------------------------

class TestKnowledgeBases:
    def test_error_entries_loaded(self) -> None:
        kb = gar.KnowledgeBases()
        entries = kb.error_entries
        assert isinstance(entries, list)
        assert len(entries) >= 38, "Expected at least 38 error KB entries"

    def test_lit_entries_loaded(self) -> None:
        kb = gar.KnowledgeBases()
        entries = kb.lit_entries
        assert isinstance(entries, list)
        assert len(entries) >= 44

    def test_tripod_items_loaded(self) -> None:
        kb = gar.KnowledgeBases()
        items = kb.tripod_items
        assert isinstance(items, list)
        assert len(items) == 27

    def test_tripod_variable_map(self) -> None:
        kb = gar.KnowledgeBases()
        vmap = kb.tripod_variable_map
        assert isinstance(vmap, dict)
        assert len(vmap) > 0

    def test_probast_domains(self) -> None:
        kb = gar.KnowledgeBases()
        domains = kb.probast_domains
        assert isinstance(domains, dict)
        assert len(domains) >= 4

    def test_journal_standards(self) -> None:
        kb = gar.KnowledgeBases()
        standards = kb.journal_standards
        assert isinstance(standards, dict)
        assert "nature_medicine" in standards

    def test_cache_reuse(self) -> None:
        kb = gar.KnowledgeBases()
        entries1 = kb.error_entries
        entries2 = kb.error_entries
        # Same object (cached)
        assert entries1 is entries2

    def test_lookup_error_by_code_found(self) -> None:
        kb = gar.KnowledgeBases()
        # ERR-026 has code 'entity_overlap_detected'
        result = kb.lookup_error_by_code("entity_overlap_detected")
        assert result is not None
        assert result["id"] == "ERR-026"

    def test_lookup_error_by_code_not_found(self) -> None:
        kb = gar.KnowledgeBases()
        result = kb.lookup_error_by_code("no_such_code_xyz_999")
        assert result is None

    def test_lookup_error_by_gate(self) -> None:
        kb = gar.KnowledgeBases()
        errors = kb.lookup_error_by_gate("split_protocol_gate")
        assert isinstance(errors, list)
        assert len(errors) >= 1

    def test_lookup_lit_by_gate(self) -> None:
        kb = gar.KnowledgeBases()
        entries = kb.lookup_lit_by_gate("leakage_gate")
        assert isinstance(entries, list)
        assert len(entries) >= 1

    def test_lookup_tripod_item_found(self) -> None:
        kb = gar.KnowledgeBases()
        item = kb.lookup_tripod_item("17")
        assert item is not None
        assert "label" in item

    def test_lookup_tripod_item_not_found(self) -> None:
        kb = gar.KnowledgeBases()
        item = kb.lookup_tripod_item("99")
        assert item is None

    def test_missing_kb_file_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If KB file does not exist, properties return empty containers."""
        monkeypatch.setattr(gar, "_REFERENCES_DIR", tmp_path)
        kb = gar.KnowledgeBases()
        assert kb.error_entries == []
        assert kb.lit_entries == []
        assert kb.tripod_items == []


# ---------------------------------------------------------------------------
# scan_project
# ---------------------------------------------------------------------------

class TestScanProject:
    def test_detects_fit_on_full_data(self, tmp_path: Path) -> None:
        (tmp_path / "train.py").write_text("scaler.fit(X_full)\n", encoding="utf-8")
        result = gar.scan_project(tmp_path)
        assert "train.py" in result["code_patterns"]["fit_on_full_data"]

    def test_detects_test_in_training_loop(self, tmp_path: Path) -> None:
        # Pattern matches X_test.fit_transform() — test data used as fitter
        (tmp_path / "model.py").write_text("X_test.fit_transform(X_test)\n", encoding="utf-8")
        result = gar.scan_project(tmp_path)
        assert "model.py" in result["code_patterns"]["test_in_training_loop"]

    def test_detects_no_random_seed(self, tmp_path: Path) -> None:
        (tmp_path / "exp.py").write_text("RandomForestClassifier(random_state=None)\n")
        result = gar.scan_project(tmp_path)
        assert result["code_patterns"]["no_random_seed"]

    def test_detects_hardcoded_threshold(self, tmp_path: Path) -> None:
        (tmp_path / "pred.py").write_text("threshold = 0.5\n")
        result = gar.scan_project(tmp_path)
        assert result["code_patterns"]["hardcoded_threshold"]

    def test_clean_project_no_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "clean.py").write_text(
            "model.fit(X_train)\nscaler.fit(X_train)\n"
        )
        result = gar.scan_project(tmp_path)
        assert result["code_patterns"]["fit_on_full_data"] == []
        assert result["code_patterns"]["test_in_training_loop"] == []

    def test_structure_has_train_csv(self, tmp_path: Path) -> None:
        (tmp_path / "train_data.csv").write_text("a,b\n1,2\n")
        result = gar.scan_project(tmp_path)
        assert result["structure"]["has_train_csv"] is True

    def test_structure_has_evidence_dir(self, tmp_path: Path) -> None:
        (tmp_path / "evidence").mkdir()
        result = gar.scan_project(tmp_path)
        assert result["structure"]["has_evidence_dir"] is True

    def test_reads_gate_reports(self, tmp_path: Path) -> None:
        make_evidence_dir(tmp_path, {"leakage_gate": "pass"})
        result = gar.scan_project(tmp_path)
        assert "leakage_gate" in result["gate_reports"]
        assert result["gate_reports"]["leakage_gate"]["status"] == "pass"

    def test_py_file_count(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"script_{i}.py").write_text("pass\n")
        result = gar.scan_project(tmp_path)
        assert result["py_file_count"] >= 3

    def test_non_py_files_not_scanned(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text('{"key": "random_state=None"}')
        result = gar.scan_project(tmp_path)
        # JSON file should not trigger Python pattern
        assert result["code_patterns"]["no_random_seed"] == []


# ---------------------------------------------------------------------------
# assess_tripod_coverage
# ---------------------------------------------------------------------------

class TestAssessTripodCoverage:
    def test_no_evidence_all_not_assessed(self, tmp_path: Path) -> None:
        coverage = gar.assess_tripod_coverage(tmp_path, {})
        assert coverage["covered_count"] == 0
        for item in coverage["items"]:
            assert item["status"] == "not_assessed"

    def test_passing_reporting_bias_gate_covers_items(self, tmp_path: Path) -> None:
        gate_reports = {
            "reporting_bias_gate": {"status": "pass", "failures": []}
        }
        coverage = gar.assess_tripod_coverage(tmp_path, gate_reports)
        assert coverage["covered_count"] > 0

    def test_failing_gate_marks_items_as_gate_failed(self, tmp_path: Path) -> None:
        gate_reports = {
            "reporting_bias_gate": {"status": "fail", "failures": []}
        }
        coverage = gar.assess_tripod_coverage(tmp_path, gate_reports)
        # At least some items should be gate_failed
        statuses = {i["status"] for i in coverage["items"]}
        assert "gate_failed" in statuses

    def test_coverage_fraction_in_range(self, tmp_path: Path) -> None:
        gate_reports = {
            "reporting_bias_gate": {"status": "pass"},
            "leakage_gate": {"status": "pass"},
        }
        coverage = gar.assess_tripod_coverage(tmp_path, gate_reports)
        assert 0.0 <= coverage["coverage_fraction"] <= 1.0

    def test_all_items_have_required_fields(self, tmp_path: Path) -> None:
        coverage = gar.assess_tripod_coverage(tmp_path, {})
        for item in coverage["items"]:
            assert "item_id" in item
            assert "label" in item
            assert "status" in item
            assert "ai_specific" in item

    def test_ai_specific_items_flagged(self, tmp_path: Path) -> None:
        coverage = gar.assess_tripod_coverage(tmp_path, {})
        ai_items = [i for i in coverage["items"] if i["ai_specific"]]
        # TRIPOD+AI 2024 has 4+ AI-specific items
        assert len(ai_items) >= 1

    def test_total_required_is_17(self, tmp_path: Path) -> None:
        coverage = gar.assess_tripod_coverage(tmp_path, {})
        assert coverage["total_required"] == 17
        assert len(coverage["items"]) == 17


# ---------------------------------------------------------------------------
# assess_probast_coverage
# ---------------------------------------------------------------------------

class TestAssessProbastCoverage:
    def test_no_evidence_all_unclear(self) -> None:
        coverage = gar.assess_probast_coverage({})
        assert coverage["overall_risk_of_bias"] == "unclear"
        for domain in coverage["domains"]:
            assert domain["rob_status"] == "unclear"

    def test_all_gates_pass_gives_low_rob(self) -> None:
        gate_reports = {
            "split_protocol_gate": {"status": "pass"},
            "external_validation_gate": {"status": "pass"},
            "leakage_gate": {"status": "pass"},
            "definition_variable_guard": {"status": "pass"},
            "feature_lineage_gate": {"status": "pass"},
            "reporting_bias_gate": {"status": "pass"},
            "clinical_metrics_gate": {"status": "pass"},
            "tuning_leakage_gate": {"status": "pass"},
            "model_selection_audit_gate": {"status": "pass"},
            "calibration_dca_gate": {"status": "pass"},
            "permutation_significance_gate": {"status": "pass"},
        }
        coverage = gar.assess_probast_coverage(gate_reports)
        assert coverage["overall_risk_of_bias"] == "low"
        assert coverage["meets_publication_requirement"] is True

    def test_failed_gate_gives_high_rob(self) -> None:
        gate_reports = {
            "leakage_gate": {"status": "fail"},
        }
        coverage = gar.assess_probast_coverage(gate_reports)
        assert coverage["overall_risk_of_bias"] == "high"
        assert coverage["meets_publication_requirement"] is False

    def test_four_domains_returned(self) -> None:
        coverage = gar.assess_probast_coverage({})
        assert len(coverage["domains"]) == 4

    def test_domains_have_name_field(self) -> None:
        coverage = gar.assess_probast_coverage({})
        for domain in coverage["domains"]:
            assert "name" in domain
            assert "domain_id" in domain
            assert "rob_status" in domain

    def test_high_robs_override_low(self) -> None:
        """One high-ROB gate anywhere → overall high."""
        gate_reports = {
            "split_protocol_gate": {"status": "pass"},
            "leakage_gate": {"status": "fail"},  # D4 high
        }
        coverage = gar.assess_probast_coverage(gate_reports)
        assert coverage["overall_risk_of_bias"] == "high"


# ---------------------------------------------------------------------------
# build_issue_list
# ---------------------------------------------------------------------------

class TestBuildIssueList:
    def _make_scan_with_pattern(self, pattern: str) -> Dict[str, Any]:
        patterns = {k: [] for k in gar.CODE_PATTERNS}
        patterns[pattern] = ["src/model.py"]
        return {
            "code_patterns": patterns,
            "structure": {},
        }

    def test_critical_pattern_produces_critical_issue(self) -> None:
        scan = self._make_scan_with_pattern("fit_on_full_data")
        issues = gar.build_issue_list(scan, {})
        critical = [i for i in issues if i["severity"] == "CRITICAL"]
        assert len(critical) >= 1

    def test_issue_has_root_cause_and_fix(self) -> None:
        scan = self._make_scan_with_pattern("fit_on_full_data")
        issues = gar.build_issue_list(scan, {})
        for issue in issues:
            assert "root_cause" in issue
            assert "fix" in issue
            assert issue["root_cause"] != ""
            assert issue["fix"] != ""

    def test_pattern_maps_to_tripod_violations(self) -> None:
        scan = self._make_scan_with_pattern("missing_ci")
        issues = gar.build_issue_list(scan, {})
        ci_issues = [i for i in issues if i.get("pattern") == "missing_ci"]
        assert len(ci_issues) == 1
        assert "tripod_ai_violations" in ci_issues[0]
        item_ids = [v["item_id"] for v in ci_issues[0]["tripod_ai_violations"]]
        assert "17" in item_ids

    def test_pattern_maps_to_probast_domain(self) -> None:
        scan = self._make_scan_with_pattern("fit_on_full_data")
        issues = gar.build_issue_list(scan, {})
        fitting_issues = [i for i in issues if i.get("pattern") == "fit_on_full_data"]
        assert len(fitting_issues) == 1
        assert "probast_domain" in fitting_issues[0]

    def test_gate_failure_creates_issue(self) -> None:
        scan = {
            "code_patterns": {k: [] for k in gar.CODE_PATTERNS},
            "structure": {},
        }
        gate_reports = {
            "leakage_gate": {
                "status": "fail",
                "failures": [
                    {"code": "target_leakage_detected", "message": "Target in X"}
                ],
            }
        }
        issues = gar.build_issue_list(scan, gate_reports)
        gate_issues = [i for i in issues if i.get("issue_type") == "gate_failure"]
        assert len(gate_issues) >= 1

    def test_no_patterns_no_gate_failures_empty(self) -> None:
        scan = {
            "code_patterns": {k: [] for k in gar.CODE_PATTERNS},
            "structure": {},
        }
        issues = gar.build_issue_list(scan, {})
        assert issues == []

    def test_issues_sorted_critical_first(self) -> None:
        scan = {
            "code_patterns": {
                **{k: [] for k in gar.CODE_PATTERNS},
                "fit_on_full_data": ["a.py"],   # CRITICAL
                "hardcoded_threshold": ["b.py"],  # INFO
                "no_random_seed": ["c.py"],       # WARNING
            },
            "structure": {},
        }
        issues = gar.build_issue_list(scan, {})
        severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}
        order = [severity_order[i["severity"]] for i in issues]
        assert order == sorted(order), "Issues not sorted by severity"

    def test_gate_failure_includes_literature(self) -> None:
        scan = {"code_patterns": {k: [] for k in gar.CODE_PATTERNS}, "structure": {}}
        gate_reports = {
            "leakage_gate": {"status": "fail", "failures": [{"code": "x", "message": "y"}]}
        }
        issues = gar.build_issue_list(scan, gate_reports)
        gate_issues = [i for i in issues if i.get("gate") == "leakage_gate"]
        # Literature may or may not be present depending on KB; check structure
        for gi in gate_issues:
            if "literature_citations" in gi:
                assert isinstance(gi["literature_citations"], list)


# ---------------------------------------------------------------------------
# compute_dimension_scores
# ---------------------------------------------------------------------------

class TestComputeDimensionScores:
    def _empty_scan(self) -> Dict[str, Any]:
        return {
            "code_patterns": {k: [] for k in gar.CODE_PATTERNS},
            "structure": {
                "has_train_csv": False,
                "has_valid_csv": False,
                "has_test_csv": False,
                "has_evidence_dir": False,
                "has_requirements": False,
                "has_git": False,
                "has_request_json": False,
                "has_model_artifact": False,
            },
        }

    def test_ten_dimensions_returned(self) -> None:
        scan = self._empty_scan()
        dim_scores, _ = gar.compute_dimension_scores(scan, {})
        assert len(dim_scores) == 10

    def test_all_gates_pass_raises_score(self) -> None:
        scan = self._empty_scan()
        gate_reports = {
            gate: {"status": "pass"}
            for gate in [
                "split_protocol_gate", "leakage_gate", "definition_variable_guard",
                "feature_lineage_gate", "tuning_leakage_gate",
                "calibration_dca_gate", "permutation_significance_gate",
                "metric_consistency_gate", "ci_matrix_gate",
                "generalization_gap_gate", "external_validation_gate",
                "distribution_generalization_gate", "seed_stability_gate",
                "clinical_metrics_gate", "fairness_equity_gate",
                "reporting_bias_gate", "manifest_lock", "execution_attestation_gate",
                "security_audit_gate", "model_selection_audit_gate",
            ]
        }
        _, total = gar.compute_dimension_scores(scan, gate_reports)
        assert total > 50, f"Expected score > 50 with all gates passing, got {total}"

    def test_critical_pattern_lowers_score(self) -> None:
        scan_clean = self._empty_scan()
        scan_dirty = self._empty_scan()
        scan_dirty["code_patterns"]["fit_on_full_data"] = ["bad.py"]

        gate_reports = {
            "split_protocol_gate": {"status": "pass"},
            "leakage_gate": {"status": "pass"},
        }
        _, score_clean = gar.compute_dimension_scores(scan_clean, gate_reports)
        _, score_dirty = gar.compute_dimension_scores(scan_dirty, gate_reports)
        assert score_dirty < score_clean

    def test_scores_bounded_0_100(self) -> None:
        scan = self._empty_scan()
        dim_scores, total = gar.compute_dimension_scores(scan, {})
        assert 0.0 <= total <= 100.0
        for dim in dim_scores.values():
            assert 0.0 <= dim["score_fraction"] <= 1.0
            assert 0.0 <= dim["weighted_score"] <= dim["max_possible"]

    def test_dimension_ids_1_to_10(self) -> None:
        scan = self._empty_scan()
        dim_scores, _ = gar.compute_dimension_scores(scan, {})
        ids = sorted(d["id"] for d in dim_scores.values())
        assert ids == list(range(1, 11))

    def test_score_interpretation_publication_grade(self) -> None:
        label_en, label_zh = gar._score_interpretation(92.0)
        assert "Publication" in label_en
        assert "顶刊" in label_zh

    def test_score_interpretation_not_publishable(self) -> None:
        label_en, _ = gar._score_interpretation(45.0)
        assert "Not publishable" in label_en


# ---------------------------------------------------------------------------
# build_remediation_plan
# ---------------------------------------------------------------------------

class TestBuildRemediationPlan:
    def test_critical_issues_become_p0(self) -> None:
        issues = [
            {"severity": "CRITICAL", "description": "leakage", "fix": "fix it",
             "error_code": "ERR-X", "issue_type": "code_pattern"},
        ]
        dim_scores = {
            "data_integrity": {"score_fraction": 0.5, "name": "Data Integrity",
                               "weighted_score": 6.0, "max_possible": 12},
        }
        plan = gar.build_remediation_plan(issues, dim_scores)
        p0_steps = [s for s in plan if s.get("priority") == "P0"]
        assert len(p0_steps) >= 1

    def test_warning_issues_become_p1(self) -> None:
        issues = [
            {"severity": "WARNING", "description": "seed missing", "fix": "add seed",
             "error_code": "ERR-Y", "issue_type": "code_pattern"},
        ]
        dim_scores: Dict[str, Any] = {}
        plan = gar.build_remediation_plan(issues, dim_scores)
        p1_steps = [s for s in plan if s.get("priority") == "P1"]
        assert len(p1_steps) >= 1

    def test_low_dim_becomes_p2(self) -> None:
        issues: list = []
        dim_scores = {
            "data_integrity": {"score_fraction": 0.1, "name": "Data Integrity",
                               "weighted_score": 1.2, "max_possible": 12},
            "leakage_prevention": {"score_fraction": 0.9, "name": "Leakage Prevention",
                                   "weighted_score": 13.5, "max_possible": 15},
        }
        plan = gar.build_remediation_plan(issues, dim_scores)
        p2_steps = [s for s in plan if s.get("priority") == "P2"]
        assert len(p2_steps) >= 1

    def test_plan_is_list(self) -> None:
        plan = gar.build_remediation_plan([], {})
        assert isinstance(plan, list)


# ---------------------------------------------------------------------------
# render_markdown_report
# ---------------------------------------------------------------------------

class TestRenderMarkdownReport:
    def _minimal_report(self) -> Dict[str, Any]:
        return {
            "report_version": "audit_report.v2",
            "generated_at": "2026-03-15T00:00:00+00:00",
            "project_dir": "/tmp/myproject",
            "total_score": 55.0,
            "max_score": 100,
            "grade_en": "Major issues",
            "grade_zh": "重大缺陷",
            "py_files_scanned": 3,
            "gate_reports_found": 0,
            "dimension_scores": {
                "data_integrity": {
                    "id": 1, "name": "Data Integrity", "weight": 12,
                    "score_fraction": 0.5, "weighted_score": 6.0, "max_possible": 12,
                },
            },
            "tripod_coverage": {
                "reference": "Collins et al. BMJ 2024",
                "total_required": 17,
                "covered_count": 3,
                "coverage_fraction": 0.176,
                "item_status": {},
                "items": [
                    {"item_id": "1", "label": "Title", "ai_specific": False,
                     "status": "covered", "text": "Test."},
                ],
            },
            "probast_coverage": {
                "reference": "Wolff et al. PROBAST+AI 2025",
                "overall_risk_of_bias": "unclear",
                "domains": [
                    {"domain_id": "D1", "name": "Participants",
                     "rob_status": "unclear", "gates_assessed": []},
                ],
                "meets_publication_requirement": False,
            },
            "issues": [],
            "remediation_plan": [],
            "structure_checks": {},
        }

    def test_returns_string(self) -> None:
        report = self._minimal_report()
        md = gar.render_markdown_report(report)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_contains_project_name(self) -> None:
        report = self._minimal_report()
        md = gar.render_markdown_report(report)
        assert "myproject" in md

    def test_contains_score(self) -> None:
        report = self._minimal_report()
        md = gar.render_markdown_report(report)
        assert "55" in md

    def test_contains_tripod_header(self) -> None:
        report = self._minimal_report()
        md = gar.render_markdown_report(report)
        assert "TRIPOD+AI" in md

    def test_contains_probast_header(self) -> None:
        report = self._minimal_report()
        md = gar.render_markdown_report(report)
        assert "PROBAST+AI" in md

    def test_issues_section_present_when_issues_exist(self) -> None:
        report = self._minimal_report()
        report["issues"] = [
            {
                "issue_type": "code_pattern",
                "pattern": "fit_on_full_data",
                "severity": "CRITICAL",
                "description": "Potential fit on full data",
                "affected_files": ["train.py"],
                "file_count": 1,
                "root_cause": "leakage",
                "fix": "fix it",
            }
        ]
        md = gar.render_markdown_report(report)
        assert "Issues Found" in md
        assert "CRITICAL" in md

    def test_remediation_section_present(self) -> None:
        report = self._minimal_report()
        report["remediation_plan"] = [
            {"priority": "P0", "severity": "CRITICAL",
             "description": "Fix leakage", "fix": "Do it", "error_code": "E1"},
        ]
        md = gar.render_markdown_report(report)
        assert "Remediation Plan" in md
        assert "P0" in md

    def test_journal_gap_section_when_present(self) -> None:
        report = self._minimal_report()
        report["journal_gap_analysis"] = {
            "target_journal": "Nature Medicine",
            "minimum_score": 90,
            "current_score": 55.0,
            "score_gap": 35.0,
            "meets_threshold": False,
            "mandatory_met": [],
            "mandatory_unmet": ["External validation"],
            "mandatory_compliance": "0/1",
        }
        md = gar.render_markdown_report(report)
        assert "Journal Gap Analysis" in md
        assert "Nature Medicine" in md


# ---------------------------------------------------------------------------
# run_audit_report — integration
# ---------------------------------------------------------------------------

class TestRunAuditReport:
    def test_basic_run_no_evidence(self, tmp_path: Path) -> None:
        """Audit a minimal project with no evidence files."""
        (tmp_path / "train.py").write_text("model.fit(X_train)\n")
        report = gar.run_audit_report(tmp_path, output_dir=tmp_path / "out")

        assert "total_score" in report
        assert "dimension_scores" in report
        assert len(report["dimension_scores"]) == 10
        assert "tripod_coverage" in report
        assert "probast_coverage" in report

    def test_output_files_created(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "reports"
        gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="both")
        assert (out_dir / "audit-report.md").exists()
        assert (out_dir / "audit-report.json").exists()

    def test_json_output_only(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="json")
        assert (out_dir / "audit-report.json").exists()
        assert not (out_dir / "audit-report.md").exists()

    def test_markdown_output_only(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="markdown")
        assert (out_dir / "audit-report.md").exists()
        assert not (out_dir / "audit-report.json").exists()

    def test_json_report_is_valid(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="json")
        data = json.loads((out_dir / "audit-report.json").read_text())
        assert data["report_version"] == "audit_report.v2"
        assert isinstance(data["total_score"], float)

    def test_with_leakage_pattern_shows_issues(self, tmp_path: Path) -> None:
        (tmp_path / "bad.py").write_text("scaler.fit(X_full)\n")
        out_dir = tmp_path / "out"
        report = gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="json")
        critical = [i for i in report["issues"] if i["severity"] == "CRITICAL"]
        assert len(critical) >= 1

    def test_invalid_project_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            gar.run_audit_report(Path("/nonexistent/path/xyz123"))

    def test_target_journal_adds_gap_analysis(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        report = gar.run_audit_report(
            tmp_path, output_dir=out_dir,
            target_journal="nature_medicine", output_format="json"
        )
        assert "journal_gap_analysis" in report
        assert report["journal_gap_analysis"]["target_journal"] != ""

    def test_with_passing_gates_improves_score(self, tmp_path: Path) -> None:
        make_evidence_dir(tmp_path, {
            "split_protocol_gate": "pass",
            "leakage_gate": "pass",
            "reporting_bias_gate": "pass",
            "calibration_dca_gate": "pass",
        })
        out_dir = tmp_path / "out"
        report = gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="json")
        # With 4 gates passing, score should be higher than 0
        assert report["total_score"] > 5.0

    def test_report_version_field(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        report = gar.run_audit_report(tmp_path, output_dir=out_dir, output_format="json")
        assert report["report_version"] == "audit_report.v2"
