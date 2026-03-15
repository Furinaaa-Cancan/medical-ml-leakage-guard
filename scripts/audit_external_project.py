#!/usr/bin/env python3
"""
Quantitative audit tool for evaluating medical ML projects.

Scores any medical ML prediction project across 10 dimensions (100-point scale)
by analyzing project artifacts, code patterns, and evidence files. Produces a
structured audit report with remediation priorities.

Usage:
    python3 scripts/audit_external_project.py --project-dir /path/to/project
    python3 scripts/audit_external_project.py --project-dir /path/to/project --target-journal nature_medicine
    python3 scripts/audit_external_project.py --project-dir /path/to/project --json --output audit_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 10-Dimension scoring weights (total = 100)
# ---------------------------------------------------------------------------
DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "data_integrity": {
        "id": 1,
        "name": "Data Integrity",
        "name_zh": "数据完整性",
        "weight": 12,
        "checks": [
            "split_files_exist",
            "no_row_overlap",
            "patient_level_disjoint",
            "temporal_ordering",
            "documented_exclusion",
            "prevalence_per_split",
        ],
    },
    "leakage_prevention": {
        "id": 2,
        "name": "Leakage Prevention",
        "name_zh": "防泄漏",
        "weight": 15,
        "checks": [
            "no_target_leakage",
            "no_definition_variable_leak",
            "no_lineage_leakage",
            "no_post_index_features",
            "feature_availability_audit",
            "temporal_boundary_check",
        ],
    },
    "pipeline_isolation": {
        "id": 3,
        "name": "Pipeline Isolation",
        "name_zh": "流水线隔离",
        "weight": 12,
        "checks": [
            "preprocessor_train_only",
            "imputer_train_only",
            "resampling_train_only",
            "scaler_in_pipeline",
            "no_target_in_imputation",
            "mice_scale_guard",
        ],
    },
    "model_selection_rigor": {
        "id": 4,
        "name": "Model Selection Rigor",
        "name_zh": "模型选择严谨性",
        "weight": 10,
        "checks": [
            "candidate_pool_ge_3",
            "includes_baseline",
            "no_test_peeking",
            "one_se_rule",
            "threshold_on_valid_only",
            "model_pool_documented",
        ],
    },
    "statistical_validity": {
        "id": 5,
        "name": "Statistical Validity",
        "name_zh": "统计有效性",
        "weight": 12,
        "checks": [
            "bootstrap_ci",
            "permutation_test",
            "calibration_metrics",
            "dca_analysis",
            "metric_consistency",
            "no_non_finite_values",
        ],
    },
    "generalization_evidence": {
        "id": 6,
        "name": "Generalization Evidence",
        "name_zh": "泛化证据",
        "weight": 10,
        "checks": [
            "train_test_gap_acceptable",
            "external_cohort_present",
            "transport_drop_acceptable",
            "seed_stability",
            "subgroup_robustness",
            "covariate_shift_check",
        ],
    },
    "clinical_completeness": {
        "id": 7,
        "name": "Clinical Completeness",
        "name_zh": "临床完整性",
        "weight": 8,
        "checks": [
            "full_metric_panel",
            "confusion_matrix_consistent",
            "threshold_feasibility",
            "clinical_utility_demonstrated",
        ],
    },
    "reporting_standards": {
        "id": 8,
        "name": "Reporting Standards",
        "name_zh": "报告标准",
        "weight": 8,
        "checks": [
            "tripod_adherence",
            "probast_assessment",
            "exclusion_criteria_documented",
            "limitations_documented",
        ],
    },
    "reproducibility": {
        "id": 9,
        "name": "Reproducibility",
        "name_zh": "可重复性",
        "weight": 8,
        "checks": [
            "seed_logged",
            "version_tracked",
            "config_saved",
            "execution_attestation",
            "manifest_lock",
            "end_to_end_rerun",
        ],
    },
    "security_provenance": {
        "id": 10,
        "name": "Security & Provenance",
        "name_zh": "安全与溯源",
        "weight": 5,
        "checks": [
            "model_signed",
            "artifact_integrity",
            "sensitive_data_scan",
            "dependency_verification",
        ],
    },
}


def _score_interpretation(score: float) -> Tuple[str, str]:
    """Return (label_en, label_zh) for a total score."""
    if score >= 90:
        return ("Publication-grade", "顶刊级")
    if score >= 75:
        return ("Solid but gaps remain", "需补充")
    if score >= 60:
        return ("Major issues", "重大缺陷")
    return ("Not publishable", "不可发表")


# ---------------------------------------------------------------------------
# Evidence-based scoring: scan evidence directory for gate reports
# ---------------------------------------------------------------------------

_GATE_REPORT_MAP: Dict[str, str] = {
    "data_integrity": "split_protocol_report.json",
    "leakage_prevention": "leakage_report.json",
    "pipeline_isolation": "tuning_leakage_report.json",
    "model_selection_rigor": "model_selection_audit_report.json",
    "statistical_validity": "ci_matrix_gate_report.json",
    "generalization_evidence": "generalization_gap_report.json",
    "clinical_completeness": "clinical_metrics_report.json",
    "reporting_standards": "reporting_bias_report.json",
    "reproducibility": "manifest.json",
    "security_provenance": "security_audit_gate_report.json",
}

_SUPPLEMENTARY_REPORTS: Dict[str, List[str]] = {
    "data_integrity": ["leakage_report.json", "covariate_shift_report.json"],
    "leakage_prevention": [
        "definition_guard_report.json",
        "lineage_report.json",
        "covariate_shift_report.json",
    ],
    "pipeline_isolation": [
        "imbalance_policy_report.json",
        "missingness_policy_report.json",
    ],
    "model_selection_rigor": ["feature_engineering_audit_report.json"],
    "statistical_validity": [
        "permutation_report.json",
        "calibration_dca_report.json",
        "metric_consistency_report.json",
        "evaluation_quality_report.json",
    ],
    "generalization_evidence": [
        "robustness_gate_report.json",
        "seed_stability_report.json",
        "external_validation_gate_report.json",
        "distribution_generalization_report.json",
    ],
    "clinical_completeness": ["prediction_replay_report.json"],
    "reporting_standards": [],
    "reproducibility": [
        "execution_attestation_report.json",
        "request_contract_report.json",
    ],
    "security_provenance": [],
}


def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON without raising."""
    try:
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _score_dimension_from_evidence(
    dim_key: str,
    evidence_dir: Path,
) -> Tuple[float, List[str], List[str]]:
    """Score a single dimension by scanning gate report files.

    Returns:
        (score_fraction 0.0-1.0, passed_checks, failed_checks)
    """
    dim = DIMENSIONS[dim_key]
    total_checks = len(dim["checks"])
    passed: List[str] = []
    failed: List[str] = []

    # Primary report
    primary_file = _GATE_REPORT_MAP.get(dim_key)
    primary_report: Optional[Dict[str, Any]] = None
    if primary_file:
        primary_report = _load_json_safe(evidence_dir / primary_file)

    if primary_report is not None:
        status = primary_report.get("status", "").lower()
        if status == "pass":
            # Primary report passes → credit half the checks automatically
            half = total_checks // 2
            for check in dim["checks"][:half]:
                passed.append(check)
        else:
            failed.append(f"primary_report_{primary_file}_status_{status}")

    # Supplementary reports
    supp_files = _SUPPLEMENTARY_REPORTS.get(dim_key, [])
    for sf in supp_files:
        sr = _load_json_safe(evidence_dir / sf)
        if sr is not None:
            if sr.get("status", "").lower() == "pass":
                passed.append(f"supp_{sf}")
            else:
                failed.append(f"supp_{sf}_fail")
        else:
            failed.append(f"supp_{sf}_missing")

    # Remaining checks that aren't covered by reports → mark as unchecked
    covered = len(passed) + len(failed)
    remaining = [c for c in dim["checks"] if c not in passed]
    for c in remaining:
        if covered < total_checks:
            failed.append(c)
            covered += 1

    score = len(passed) / max(total_checks, 1)
    return score, passed, failed


def _scan_code_patterns(project_dir: Path) -> Dict[str, List[str]]:
    """Scan Python files for common leakage and quality anti-patterns.

    Returns dict of {pattern_name: [file_paths_with_issue]}.
    """
    patterns: Dict[str, re.Pattern[str]] = {
        "fit_on_full_data": re.compile(
            r"\.fit\s*\([^)]*(?:X_all|X_full|df\b|data\b)", re.IGNORECASE
        ),
        "test_in_training_loop": re.compile(
            r"(?:X_test|y_test|test_data)\s*(?:\.fit|fit_transform)", re.IGNORECASE
        ),
        "smote_on_full": re.compile(
            r"SMOTE|ADASYN|BorderlineSMOTE", re.IGNORECASE
        ),
        "no_random_seed": re.compile(
            r"random_state\s*=\s*None", re.IGNORECASE
        ),
        "hardcoded_threshold": re.compile(
            r"threshold\s*=\s*0\.5\b"
        ),
        "missing_ci": re.compile(
            r"(?:accuracy|auc|f1)(?:_score)?\s*=", re.IGNORECASE
        ),
    }
    results: Dict[str, List[str]] = {k: [] for k in patterns}

    py_files = list(project_dir.rglob("*.py"))
    for pf in py_files[:200]:  # Limit to avoid huge projects
        try:
            content = pf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for name, pat in patterns.items():
            if pat.search(content):
                results[name].append(str(pf.relative_to(project_dir)))

    return results


def _check_file_structure(project_dir: Path) -> Dict[str, bool]:
    """Check for expected project artifacts."""
    checks: Dict[str, bool] = {}
    # Data splits
    checks["has_train_csv"] = any(project_dir.rglob("*train*.csv"))
    checks["has_valid_csv"] = any(project_dir.rglob("*valid*.csv"))
    checks["has_test_csv"] = any(project_dir.rglob("*test*.csv"))
    # Config
    checks["has_request_json"] = any(project_dir.rglob("request*.json"))
    # Evidence
    checks["has_evidence_dir"] = (project_dir / "evidence").is_dir()
    # Model
    checks["has_model_artifact"] = any(
        project_dir.rglob("*.pkl")
    ) or any(project_dir.rglob("*.joblib"))
    # Requirements
    checks["has_requirements"] = (
        (project_dir / "requirements.txt").is_file()
        or (project_dir / "pyproject.toml").is_file()
    )
    # Git
    checks["has_git"] = (project_dir / ".git").is_dir()
    return checks


def _load_journal_standards(
    target_journal: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Load journal-specific requirements."""
    if not target_journal:
        return None
    standards_path = (
        Path(__file__).parent.parent / "references" / "journal-rigor-standards.json"
    )
    data = _load_json_safe(standards_path)
    if data is None:
        return None
    journals = data.get("journals", {})
    return journals.get(target_journal)


def run_audit(
    project_dir: Path,
    target_journal: Optional[str] = None,
    output_path: Optional[Path] = None,
    as_json: bool = False,
) -> Dict[str, Any]:
    """Run full 10-dimension audit on a project.

    Args:
        project_dir: Path to the project to audit.
        target_journal: Optional journal key for gap analysis.
        output_path: Optional output file path.
        as_json: If True, output JSON format.

    Returns:
        Structured audit report dict.
    """
    evidence_dir = project_dir / "evidence"

    # Structure checks
    structure = _check_file_structure(project_dir)

    # Code pattern scan
    code_patterns = _scan_code_patterns(project_dir)

    # Score each dimension
    dimension_scores: Dict[str, Dict[str, Any]] = {}
    total_score: float = 0.0

    for dim_key, dim_info in DIMENSIONS.items():
        if evidence_dir.is_dir():
            frac, passed, failed = _score_dimension_from_evidence(
                dim_key, evidence_dir
            )
        else:
            frac = 0.0
            passed = []
            failed = list(dim_info["checks"])

        weighted = frac * dim_info["weight"]
        total_score += weighted

        dimension_scores[dim_key] = {
            "id": dim_info["id"],
            "name": dim_info["name"],
            "name_zh": dim_info["name_zh"],
            "weight": dim_info["weight"],
            "score_fraction": round(frac, 3),
            "weighted_score": round(weighted, 2),
            "max_possible": dim_info["weight"],
            "passed_checks": passed,
            "failed_checks": failed,
        }

    total_score = round(total_score, 2)
    label_en, label_zh = _score_interpretation(total_score)

    # Code pattern warnings
    code_warnings: List[Dict[str, Any]] = []
    pattern_severity: Dict[str, str] = {
        "fit_on_full_data": "CRITICAL",
        "test_in_training_loop": "CRITICAL",
        "smote_on_full": "WARNING",
        "no_random_seed": "WARNING",
        "hardcoded_threshold": "INFO",
        "missing_ci": "INFO",
    }
    pattern_descriptions: Dict[str, str] = {
        "fit_on_full_data": "Potential fit on full/combined data (data leakage risk)",
        "test_in_training_loop": "Test data may be used in training loop",
        "smote_on_full": "SMOTE/oversampling detected — verify it's train-only",
        "no_random_seed": "random_state=None found — reproducibility risk",
        "hardcoded_threshold": "Hardcoded threshold=0.5 — should be optimized on validation",
        "missing_ci": "Metric computed without confidence interval context",
    }
    for pat_name, files in code_patterns.items():
        if files:
            code_warnings.append({
                "pattern": pat_name,
                "severity": pattern_severity.get(pat_name, "INFO"),
                "description": pattern_descriptions.get(pat_name, pat_name),
                "affected_files": files[:10],
                "count": len(files),
            })

    # Journal gap analysis
    journal_gap: Optional[Dict[str, Any]] = None
    journal_info = _load_journal_standards(target_journal)
    if journal_info is not None:
        min_score = journal_info.get("ml_prediction_requirements", {}).get(
            "minimum_score", 80
        )
        mandatory = journal_info.get("ml_prediction_requirements", {}).get(
            "mandatory", []
        )
        unmet: List[str] = []
        met: List[str] = []
        for req in mandatory:
            gate = req.get("gate")
            if gate and evidence_dir.is_dir():
                # Check if corresponding gate report exists and passes
                possible_files = [
                    f"{gate}_report.json",
                    f"{gate.replace('_gate', '')}_report.json",
                ]
                found_pass = False
                for pf in possible_files:
                    r = _load_json_safe(evidence_dir / pf)
                    if r and r.get("status", "").lower() == "pass":
                        found_pass = True
                        break
                if found_pass:
                    met.append(req["requirement"])
                else:
                    unmet.append(req["requirement"])
            else:
                unmet.append(req["requirement"])

        journal_gap = {
            "target_journal": journal_info.get("full_name", target_journal),
            "minimum_score": min_score,
            "current_score": total_score,
            "score_gap": round(max(0, min_score - total_score), 2),
            "meets_threshold": total_score >= min_score,
            "mandatory_met": met,
            "mandatory_unmet": unmet,
            "mandatory_compliance": (
                f"{len(met)}/{len(met)+len(unmet)}"
                if (met or unmet)
                else "N/A"
            ),
        }

    # Remediation priorities
    remediation: List[Dict[str, str]] = []
    sorted_dims = sorted(
        dimension_scores.items(),
        key=lambda x: x[1]["score_fraction"],
    )
    for dim_key, dim_data in sorted_dims:
        if dim_data["score_fraction"] < 1.0:
            remediation.append({
                "priority": str(len(remediation) + 1),
                "dimension": dim_data["name"],
                "dimension_zh": dim_data["name_zh"],
                "current_score": f"{dim_data['weighted_score']}/{dim_data['max_possible']}",
                "gap": str(
                    round(dim_data["max_possible"] - dim_data["weighted_score"], 2)
                ),
                "failed_checks": ", ".join(dim_data["failed_checks"][:5]),
            })

    report: Dict[str, Any] = {
        "contract_version": "audit_report.v1",
        "project_dir": str(project_dir),
        "total_score": total_score,
        "max_score": 100,
        "grade_en": label_en,
        "grade_zh": label_zh,
        "dimension_scores": dimension_scores,
        "structure_checks": structure,
        "code_warnings": code_warnings,
        "remediation_priorities": remediation,
    }
    if journal_gap is not None:
        report["journal_gap_analysis"] = journal_gap

    # Output
    if as_json:
        output_text = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        output_text = _format_text_report(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"Audit report written to {output_path}")
    else:
        print(output_text)

    return report


def _format_text_report(report: Dict[str, Any]) -> str:
    """Format audit report as human-readable text."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  ML Leakage Guard — External Project Audit Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"  Project:  {report['project_dir']}"
    )
    lines.append(
        f"  Score:    {report['total_score']} / {report['max_score']}  "
        f"({report['grade_en']} / {report['grade_zh']})"
    )
    lines.append("")

    # Dimension breakdown
    lines.append("-" * 70)
    lines.append("  Dimension Scores")
    lines.append("-" * 70)
    header = f"  {'Dimension':<28} {'Score':>8} {'Max':>6} {'%':>6}"
    lines.append(header)
    lines.append("  " + "-" * 52)
    for dim_data in sorted(
        report["dimension_scores"].values(), key=lambda x: x["id"]
    ):
        pct = (
            f"{dim_data['score_fraction']*100:.0f}%"
            if dim_data["score_fraction"] > 0
            else "0%"
        )
        line = (
            f"  {dim_data['name']:<28} "
            f"{dim_data['weighted_score']:>8.1f} "
            f"{dim_data['max_possible']:>6} "
            f"{pct:>6}"
        )
        lines.append(line)

    # Structure checks
    lines.append("")
    lines.append("-" * 70)
    lines.append("  Project Structure")
    lines.append("-" * 70)
    for check, ok in report["structure_checks"].items():
        symbol = "OK" if ok else "MISSING"
        lines.append(f"  [{symbol:>7}] {check}")

    # Code warnings
    if report["code_warnings"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("  Code Pattern Warnings")
        lines.append("-" * 70)
        for warn in report["code_warnings"]:
            lines.append(
                f"  [{warn['severity']:>8}] {warn['description']}"
            )
            for af in warn["affected_files"][:3]:
                lines.append(f"             -> {af}")

    # Journal gap
    if "journal_gap_analysis" in report:
        jg = report["journal_gap_analysis"]
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  Journal Gap Analysis: {jg['target_journal']}")
        lines.append("-" * 70)
        lines.append(f"  Minimum score:  {jg['minimum_score']}")
        lines.append(f"  Current score:  {jg['current_score']}")
        lines.append(f"  Gap:            {jg['score_gap']}")
        lines.append(
            f"  Meets threshold: {'YES' if jg['meets_threshold'] else 'NO'}"
        )
        lines.append(
            f"  Mandatory compliance: {jg['mandatory_compliance']}"
        )
        if jg["mandatory_unmet"]:
            lines.append("  Unmet requirements:")
            for req in jg["mandatory_unmet"]:
                lines.append(f"    - {req}")

    # Remediation
    if report["remediation_priorities"]:
        lines.append("")
        lines.append("-" * 70)
        lines.append("  Remediation Priorities (lowest score first)")
        lines.append("-" * 70)
        for rem in report["remediation_priorities"][:7]:
            lines.append(
                f"  #{rem['priority']} {rem['dimension']} "
                f"({rem['dimension_zh']}) — "
                f"Score: {rem['current_score']}, Gap: {rem['gap']}"
            )
            if rem["failed_checks"]:
                lines.append(f"       Failed: {rem['failed_checks']}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantitative audit tool for medical ML projects."
    )
    parser.add_argument(
        "--project-dir",
        required=True,
        type=Path,
        help="Path to the project directory to audit.",
    )
    parser.add_argument(
        "--target-journal",
        type=str,
        default=None,
        choices=[
            "nature_medicine",
            "lancet_digital_health",
            "jama",
            "bmj",
            "npj_digital_medicine",
        ],
        help="Target journal for gap analysis.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to file instead of stdout.",
    )
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        print(f"Error: {args.project_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    report = run_audit(
        project_dir=args.project_dir,
        target_journal=args.target_journal,
        output_path=args.output,
        as_json=args.json,
    )

    total = report["total_score"]
    if total < 60:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
