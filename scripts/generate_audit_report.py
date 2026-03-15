#!/usr/bin/env python3
"""
generate_audit_report.py — Unified ML project audit report generator.

Given any medical ML project directory, this tool:
1. Runs the 10-dimension audit (via audit_external_project.py)
2. Scans code for anti-patterns and maps each to error-knowledge-base.json
3. Checks TRIPOD+AI 2024 item coverage (27 items)
4. Assesses PROBAST+AI 2025 risk-of-bias domains
5. Enriches every finding with root cause, fix, and literature citations
6. Outputs a publication-ready audit report in Markdown + JSON

Usage:
    python3 scripts/generate_audit_report.py --project-dir /path/to/project
    python3 scripts/generate_audit_report.py --project-dir /path/to/project \\
        --output-dir reports/ --target-journal nature_medicine
    python3 scripts/generate_audit_report.py --project-dir /path/to/project \\
        --format json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
_REFERENCES_DIR = _SCRIPTS_DIR.parent / "references"

# ---------------------------------------------------------------------------
# Knowledge base loaders
# ---------------------------------------------------------------------------

def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file; return None on error."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


class KnowledgeBases:
    """Lazy-loaded knowledge base container."""

    def __init__(self) -> None:
        self._error_kb: Optional[Dict[str, Any]] = None
        self._lit_kb: Optional[Dict[str, Any]] = None
        self._tripod: Optional[Dict[str, Any]] = None
        self._probast: Optional[Dict[str, Any]] = None
        self._journal_standards: Optional[Dict[str, Any]] = None

    @property
    def error_entries(self) -> List[Dict[str, Any]]:
        if self._error_kb is None:
            self._error_kb = _load_json_safe(
                _REFERENCES_DIR / "error-knowledge-base.json"
            ) or {}
        return self._error_kb.get("entries", [])  # type: ignore[union-attr]

    @property
    def lit_entries(self) -> List[Dict[str, Any]]:
        if self._lit_kb is None:
            self._lit_kb = _load_json_safe(
                _REFERENCES_DIR / "literature-knowledge-base.json"
            ) or {}
        return self._lit_kb.get("entries", [])  # type: ignore[union-attr]

    @property
    def tripod_items(self) -> List[Dict[str, Any]]:
        if self._tripod is None:
            self._tripod = _load_json_safe(
                _REFERENCES_DIR / "tripod-ai-official-checklist.json"
            ) or {}
        return self._tripod.get("items", [])  # type: ignore[union-attr]

    @property
    def tripod_variable_map(self) -> Dict[str, str]:
        if self._tripod is None:
            self._tripod = _load_json_safe(
                _REFERENCES_DIR / "tripod-ai-official-checklist.json"
            ) or {}
        return self._tripod.get("variable_name_to_item_id", {})  # type: ignore[union-attr]

    @property
    def probast_domains(self) -> Dict[str, Any]:
        if self._probast is None:
            self._probast = _load_json_safe(
                _REFERENCES_DIR / "probast-ai-signalling-questions.json"
            ) or {}
        return self._probast.get("domains", {})  # type: ignore[union-attr]

    @property
    def probast_gate_mapping(self) -> Dict[str, Any]:
        if self._probast is None:
            self._probast = _load_json_safe(
                _REFERENCES_DIR / "probast-ai-signalling-questions.json"
            ) or {}
        return self._probast.get("mlgg_gate_mapping", {})  # type: ignore[union-attr]

    @property
    def journal_standards(self) -> Dict[str, Any]:
        if self._journal_standards is None:
            data = _load_json_safe(
                _REFERENCES_DIR / "journal-rigor-standards.json"
            ) or {}
            self._journal_standards = data.get("journals", {})  # type: ignore[union-attr]
        return self._journal_standards  # type: ignore[return-value]

    def lookup_error_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        for entry in self.error_entries:
            if entry.get("code") == code:
                return entry
        return None

    def lookup_error_by_gate(self, gate: str) -> List[Dict[str, Any]]:
        return [e for e in self.error_entries if e.get("gate") == gate]

    def lookup_lit_by_gate(self, gate: str) -> List[Dict[str, Any]]:
        return [
            e for e in self.lit_entries
            if gate in e.get("gates_implementing", [])
        ]

    def lookup_lit_by_dimension(self, dim: str) -> List[Dict[str, Any]]:
        return [
            e for e in self.lit_entries
            if dim in e.get("dimensions_affected", [])
        ]

    def lookup_tripod_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        for item in self.tripod_items:
            if item.get("item_id") == item_id:
                return item
        return None


KB = KnowledgeBases()

# ---------------------------------------------------------------------------
# Code pattern → error code mapping
# ---------------------------------------------------------------------------

PATTERN_TO_ERROR_CODE: Dict[str, str] = {
    "fit_on_full_data": "preprocessor_fit_on_full_data",
    "test_in_training_loop": "test_data_in_training_loop",
    "smote_on_full": "resampling_on_full_data",
    "no_random_seed": "missing_random_seed",
    "hardcoded_threshold": "hardcoded_threshold_05",
    "missing_ci": "metrics_without_confidence_intervals",
}

PATTERN_TO_TRIPOD_ITEMS: Dict[str, List[str]] = {
    "fit_on_full_data": ["10", "11"],        # Model building + internal validation
    "test_in_training_loop": ["10", "11"],
    "smote_on_full": ["8", "10"],            # Missing data + model building
    "no_random_seed": ["9", "24"],           # Statistical analysis + supplementary
    "hardcoded_threshold": ["17", "22"],     # Performance + interpretation
    "missing_ci": ["17"],                    # Performance measures with CI
}

PATTERN_TO_PROBAST_DOMAIN: Dict[str, str] = {
    "fit_on_full_data": "D4_analysis",
    "test_in_training_loop": "D4_analysis",
    "smote_on_full": "D4_analysis",
    "no_random_seed": "D4_analysis",
    "hardcoded_threshold": "D4_analysis",
    "missing_ci": "D4_analysis",
}

# Code pattern definitions (expanded from audit_external_project.py)
CODE_PATTERNS: Dict[str, re.Pattern[str]] = {
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
        r"(?:accuracy|auc|roc_auc|f1|precision|recall)(?:_score)?\s*=",
        re.IGNORECASE
    ),
    "shell_true": re.compile(
        r"subprocess\.[^\n]*shell\s*=\s*True", re.IGNORECASE
    ),
    "pickle_load_unsafe": re.compile(
        r"pickle\.load\s*\(", re.IGNORECASE
    ),
    "eval_use": re.compile(
        r"\beval\s*\(", re.IGNORECASE
    ),
    "no_train_test_split": re.compile(
        r"train_test_split\s*\([^,)]+\)", re.IGNORECASE
    ),
    "global_scaler_leak": re.compile(
        r"StandardScaler|MinMaxScaler|RobustScaler", re.IGNORECASE
    ),
    "leakage_via_future": re.compile(
        r"(?:discharge_date|death_date|outcome_date)\s*[^=]", re.IGNORECASE
    ),
}

PATTERN_SEVERITY: Dict[str, str] = {
    "fit_on_full_data": "CRITICAL",
    "test_in_training_loop": "CRITICAL",
    "smote_on_full": "WARNING",
    "no_random_seed": "WARNING",
    "hardcoded_threshold": "INFO",
    "missing_ci": "INFO",
    "shell_true": "WARNING",
    "pickle_load_unsafe": "WARNING",
    "eval_use": "WARNING",
    "no_train_test_split": "INFO",
    "global_scaler_leak": "WARNING",
    "leakage_via_future": "CRITICAL",
}

PATTERN_DESCRIPTION: Dict[str, str] = {
    "fit_on_full_data": "Potential fit on full/combined data — preprocessor trained on test set (data leakage)",
    "test_in_training_loop": "Test data referenced in training loop — direct data leakage",
    "smote_on_full": "SMOTE/oversampling detected — must verify it's applied only to training fold",
    "no_random_seed": "random_state=None — results are non-reproducible across runs",
    "hardcoded_threshold": "Hardcoded decision threshold 0.5 — threshold should be optimized on validation set",
    "missing_ci": "Metric computed without confidence interval — violates TRIPOD+AI Item 17",
    "shell_true": "subprocess with shell=True — shell injection vulnerability",
    "pickle_load_unsafe": "pickle.load() without source verification — arbitrary code execution risk",
    "eval_use": "eval() usage — code injection risk",
    "no_train_test_split": "train_test_split used without stratify= — may produce imbalanced splits",
    "global_scaler_leak": "Scaler instantiation detected — verify it's fitted only on training data",
    "leakage_via_future": "Possible future-dated feature — verify it's not outcome-proximate",
}

# ---------------------------------------------------------------------------
# TRIPOD+AI required items for quick assessment
# ---------------------------------------------------------------------------

TRIPOD_REQUIRED_ITEMS = [
    "1", "4", "5", "6a", "6b", "7", "8", "10", "11", "12",
    "15b", "16", "17", "18", "20", "21", "27",
]

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_interpretation(score: float) -> Tuple[str, str]:
    if score >= 90:
        return "Publication-grade", "顶刊级"
    if score >= 75:
        return "Solid but gaps remain", "需补充"
    if score >= 60:
        return "Major issues", "重大缺陷"
    return "Not publishable", "不可发表"


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def scan_project(project_dir: Path) -> Dict[str, Any]:
    """Scan project directory for code patterns and file structure."""
    results: Dict[str, List[str]] = {k: [] for k in CODE_PATTERNS}

    py_files = list(project_dir.rglob("*.py"))
    for pf in py_files[:300]:
        try:
            content = pf.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for name, pat in CODE_PATTERNS.items():
            if pat.search(content):
                results[name].append(str(pf.relative_to(project_dir)))

    structure: Dict[str, bool] = {
        "has_train_csv": any(project_dir.rglob("*train*.csv")),
        "has_valid_csv": any(project_dir.rglob("*val*.csv"))
            or any(project_dir.rglob("*valid*.csv")),
        "has_test_csv": any(project_dir.rglob("*test*.csv")),
        "has_evidence_dir": (project_dir / "evidence").is_dir(),
        "has_requirements": (
            (project_dir / "requirements.txt").is_file()
            or (project_dir / "pyproject.toml").is_file()
        ),
        "has_git": (project_dir / ".git").is_dir(),
        "has_request_json": any(project_dir.rglob("request*.json")),
        "has_model_artifact": (
            any(project_dir.rglob("*.pkl"))
            or any(project_dir.rglob("*.joblib"))
        ),
    }

    # Read evidence gate reports if present
    gate_reports: Dict[str, Dict[str, Any]] = {}
    evidence_dir = project_dir / "evidence"
    if evidence_dir.is_dir():
        for report_file in evidence_dir.glob("*_report.json"):
            data = _load_json_safe(report_file)
            if data:
                gate_name = report_file.stem.replace("_report", "")
                gate_reports[gate_name] = data

    return {
        "code_patterns": results,
        "structure": structure,
        "gate_reports": gate_reports,
        "py_file_count": len(py_files),
    }


def assess_tripod_coverage(
    project_dir: Path,
    gate_reports: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assess TRIPOD+AI 2024 item coverage from gate reports and project structure."""
    # Map gate pass/fail to TRIPOD items
    gate_to_tripod: Dict[str, List[str]] = {
        "reporting_bias_gate": ["1","4","5","6a","6b","7","8","10","11","12","14","15b","16","17","18","19","20","21","22","23","24","25","26","27"],
        "split_protocol_gate": ["4","14"],
        "leakage_gate": ["10","11"],
        "clinical_metrics_gate": ["15b","17"],
        "calibration_dca_gate": ["17"],
        "permutation_significance_gate": ["17","18"],
        "fairness_equity_gate": ["12","20"],
        "generalization_gap_gate": ["11","19","23"],
        "external_validation_gate": ["23"],
        "seed_stability_gate": ["9","24"],
        "manifest_lock": ["24","26"],
    }

    covered_items: set = set()
    uncovered_items: set = set(TRIPOD_REQUIRED_ITEMS)
    item_status: Dict[str, str] = {}

    for gate_name, report in gate_reports.items():
        status = report.get("status", "").lower()
        tripod_items = gate_to_tripod.get(gate_name, [])
        for item_id in tripod_items:
            if status == "pass":
                covered_items.add(item_id)
                if item_id in uncovered_items:
                    uncovered_items.discard(item_id)
                item_status[item_id] = "covered"
            elif item_id not in item_status:
                item_status[item_id] = "gate_failed"

    for item_id in TRIPOD_REQUIRED_ITEMS:
        if item_id not in item_status:
            item_status[item_id] = "not_assessed"

    # Enrich with official item text
    enriched: List[Dict[str, Any]] = []
    for item_id in TRIPOD_REQUIRED_ITEMS:
        tripod_item = KB.lookup_tripod_item(item_id)
        status = item_status.get(item_id, "not_assessed")
        enriched.append({
            "item_id": item_id,
            "label": tripod_item.get("label", f"Item {item_id}") if tripod_item else f"Item {item_id}",
            "text": tripod_item.get("text", "") if tripod_item else "",
            "ai_specific": tripod_item.get("ai_specific", False) if tripod_item else False,
            "status": status,
            "gate_check": tripod_item.get("gate_check", "") if tripod_item else "",
        })

    covered_count = sum(1 for s in item_status.values() if s == "covered")
    return {
        "reference": "Collins et al. BMJ 2024;385:e078378",
        "total_required": len(TRIPOD_REQUIRED_ITEMS),
        "covered_count": covered_count,
        "coverage_fraction": round(covered_count / len(TRIPOD_REQUIRED_ITEMS), 3),
        "item_status": item_status,
        "items": enriched,
    }


def assess_probast_coverage(
    gate_reports: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assess PROBAST+AI domain coverage from gate reports."""
    domain_gates: Dict[str, List[str]] = {
        "D1_participants": ["split_protocol_gate", "external_validation_gate"],
        "D2_predictors": ["leakage_gate", "definition_variable_guard", "feature_lineage_gate"],
        "D3_outcome": ["reporting_bias_gate", "clinical_metrics_gate"],
        "D4_analysis": [
            "leakage_gate", "tuning_leakage_gate", "split_protocol_gate",
            "model_selection_audit_gate", "calibration_dca_gate",
            "permutation_significance_gate", "reporting_bias_gate",
        ],
    }

    domain_results: Dict[str, str] = {}
    for domain_id, domain_gate_list in domain_gates.items():
        passed_any = False
        failed_any = False
        for gate in domain_gate_list:
            report = gate_reports.get(gate, {})
            if report.get("status", "").lower() == "pass":
                passed_any = True
            elif report.get("status", "").lower() == "fail":
                failed_any = True

        if failed_any:
            domain_results[domain_id] = "high"
        elif passed_any:
            domain_results[domain_id] = "low"
        else:
            domain_results[domain_id] = "unclear"

    overall_rob = "low"
    for status in domain_results.values():
        if status == "high":
            overall_rob = "high"
            break
        if status == "unclear":
            overall_rob = "unclear"

    # Enrich with domain descriptions
    domain_info: List[Dict[str, Any]] = []
    probast_domains = KB.probast_domains
    for domain_id, rob_status in domain_results.items():
        domain_data = probast_domains.get(domain_id, {})
        domain_info.append({
            "domain_id": domain_id,
            "name": domain_data.get("name", domain_id),
            "rob_status": rob_status,
            "gates_assessed": domain_gates.get(domain_id, []),
        })

    return {
        "reference": "Wolff et al. PROBAST+AI 2025",
        "overall_risk_of_bias": overall_rob,
        "domains": domain_info,
        "meets_publication_requirement": overall_rob == "low",
    }


def build_issue_list(
    scan_results: Dict[str, Any],
    gate_reports: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build enriched issue list combining code patterns, gate failures, and KB lookups."""
    issues: List[Dict[str, Any]] = []

    # 1. Code pattern issues
    for pattern_name, affected_files in scan_results["code_patterns"].items():
        if not affected_files:
            continue

        severity = PATTERN_SEVERITY.get(pattern_name, "INFO")
        description = PATTERN_DESCRIPTION.get(pattern_name, pattern_name)
        error_code = PATTERN_TO_ERROR_CODE.get(pattern_name, pattern_name)

        # Look up in error KB
        kb_entry = KB.lookup_error_by_code(error_code)

        # Map to TRIPOD items
        tripod_item_ids = PATTERN_TO_TRIPOD_ITEMS.get(pattern_name, [])
        tripod_items = []
        for item_id in tripod_item_ids:
            ti = KB.lookup_tripod_item(item_id)
            if ti:
                tripod_items.append({
                    "item_id": item_id,
                    "label": ti.get("label", ""),
                    "reference": "TRIPOD+AI 2024",
                })

        # Map to PROBAST domain
        probast_domain = PATTERN_TO_PROBAST_DOMAIN.get(pattern_name)

        issue: Dict[str, Any] = {
            "issue_type": "code_pattern",
            "pattern": pattern_name,
            "severity": severity,
            "description": description,
            "affected_files": affected_files[:10],
            "file_count": len(affected_files),
        }

        if kb_entry:
            issue["error_code"] = kb_entry.get("code", error_code)
            issue["root_cause"] = kb_entry.get("root_cause", "")
            issue["fix"] = kb_entry.get("fix", "")
            issue["prevention"] = kb_entry.get("prevention", "")
        else:
            issue["error_code"] = error_code
            issue["root_cause"] = description
            issue["fix"] = "Review and fix the flagged pattern in each affected file."
            issue["prevention"] = "Use linting and pre-commit hooks to catch this pattern."

        if tripod_items:
            issue["tripod_ai_violations"] = tripod_items
            issue["tripod_ai_reference"] = "Collins et al. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378"

        if probast_domain:
            issue["probast_domain"] = probast_domain

        issues.append(issue)

    # 2. Gate failure issues
    for gate_name, report in gate_reports.items():
        if report.get("status", "").lower() != "fail":
            continue

        gate_failures = report.get("failures", [])
        kb_errors = KB.lookup_error_by_gate(gate_name)
        lit_refs = KB.lookup_lit_by_gate(gate_name)

        lit_citations = [
            {
                "id": e["id"],
                "title": e["title"],
                "journal": e["journal"],
                "year": e["year"],
                "doi": e.get("doi", ""),
            }
            for e in lit_refs[:3]
        ]

        for failure in gate_failures[:5]:  # Limit to 5 failures per gate
            f_code = failure.get("code", "") if isinstance(failure, dict) else ""
            f_message = failure.get("message", str(failure)) if isinstance(failure, dict) else str(failure)

            # Find matching KB entry
            kb_match = next(
                (e for e in kb_errors if e.get("code") == f_code),
                kb_errors[0] if kb_errors else None,
            )

            issue: Dict[str, Any] = {
                "issue_type": "gate_failure",
                "gate": gate_name,
                "severity": "ERROR",
                "description": f_message,
                "error_code": f_code or gate_name,
            }

            if kb_match:
                issue["root_cause"] = kb_match.get("root_cause", "")
                issue["fix"] = kb_match.get("fix", "")
                issue["prevention"] = kb_match.get("prevention", "")
                issue["kb_reference"] = kb_match.get("id", "")
            else:
                issue["root_cause"] = f"Gate {gate_name} failed: {f_message}"
                issue["fix"] = f"Review gate requirements for {gate_name} and address the listed failure."
                issue["prevention"] = "Run gates regularly during development, not just at publication."

            if lit_citations:
                issue["literature_citations"] = lit_citations
                issue["literature_note"] = (
                    "See cited works for methodological standards underlying this check."
                )

            issues.append(issue)

    # Sort: CRITICAL first, then ERROR, WARNING, INFO
    severity_order = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}
    issues.sort(key=lambda x: severity_order.get(x.get("severity", "INFO"), 4))

    return issues


def compute_dimension_scores(
    scan_results: Dict[str, Any],
    gate_reports: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], float]:
    """Compute 10-dimension scores from available evidence."""
    dimensions = {
        "data_integrity": {
            "id": 1, "name": "Data Integrity", "weight": 12,
            "gate_signals": ["split_protocol_gate", "leakage_gate"],
            "pattern_penalties": {"fit_on_full_data": 0.5, "test_in_training_loop": 0.5},
            "structure_bonuses": {
                "has_train_csv": 0.2, "has_valid_csv": 0.2, "has_test_csv": 0.2,
            },
        },
        "leakage_prevention": {
            "id": 2, "name": "Leakage Prevention", "weight": 15,
            "gate_signals": ["leakage_gate", "definition_variable_guard", "feature_lineage_gate", "tuning_leakage_gate"],
            "pattern_penalties": {
                "fit_on_full_data": 0.4, "test_in_training_loop": 0.4, "leakage_via_future": 0.5,
            },
        },
        "pipeline_isolation": {
            "id": 3, "name": "Pipeline Isolation", "weight": 12,
            "gate_signals": ["split_protocol_gate"],
            "pattern_penalties": {"smote_on_full": 0.3, "global_scaler_leak": 0.2},
        },
        "model_selection_rigor": {
            "id": 4, "name": "Model Selection Rigor", "weight": 10,
            "gate_signals": ["model_selection_audit_gate"],
            "pattern_penalties": {"hardcoded_threshold": 0.2},
        },
        "statistical_validity": {
            "id": 5, "name": "Statistical Validity", "weight": 12,
            "gate_signals": [
                "calibration_dca_gate", "permutation_significance_gate",
                "metric_consistency_gate", "ci_matrix_gate",
            ],
            "pattern_penalties": {"missing_ci": 0.3, "no_random_seed": 0.2},
        },
        "generalization_evidence": {
            "id": 6, "name": "Generalization Evidence", "weight": 10,
            "gate_signals": [
                "generalization_gap_gate", "external_validation_gate",
                "distribution_generalization_gate", "seed_stability_gate",
            ],
        },
        "clinical_completeness": {
            "id": 7, "name": "Clinical Completeness", "weight": 8,
            "gate_signals": ["clinical_metrics_gate", "fairness_equity_gate"],
        },
        "reporting_standards": {
            "id": 8, "name": "Reporting Standards", "weight": 8,
            "gate_signals": ["reporting_bias_gate"],
        },
        "reproducibility": {
            "id": 9, "name": "Reproducibility", "weight": 8,
            "gate_signals": ["manifest_lock", "execution_attestation_gate", "seed_stability_gate"],
            "pattern_penalties": {"no_random_seed": 0.4},
            "structure_bonuses": {"has_requirements": 0.3, "has_git": 0.3},
        },
        "security_provenance": {
            "id": 10, "name": "Security & Provenance", "weight": 5,
            "gate_signals": ["security_audit_gate", "manifest_lock"],
            "pattern_penalties": {"shell_true": 0.3, "pickle_load_unsafe": 0.2, "eval_use": 0.3},
        },
    }

    structure = scan_results["structure"]
    code_patterns = scan_results["code_patterns"]

    dimension_scores: Dict[str, Any] = {}
    total_score: float = 0.0

    for dim_key, dim_info in dimensions.items():
        frac: float = 0.5  # Start at 50% (neutral baseline)

        # Gate signals (each passing gate adds value)
        gate_sigs = dim_info.get("gate_signals", [])
        if gate_sigs:
            pass_count = sum(
                1 for g in gate_sigs
                if gate_reports.get(g, {}).get("status", "").lower() == "pass"
            )
            fail_count = sum(
                1 for g in gate_sigs
                if gate_reports.get(g, {}).get("status", "").lower() == "fail"
            )
            assessed = pass_count + fail_count
            if assessed > 0:
                gate_frac = pass_count / len(gate_sigs)
                frac = 0.4 + 0.6 * gate_frac  # Gates dominate score
            elif not gate_reports:
                frac = 0.0  # No evidence = zero

        # Pattern penalties
        penalty = 0.0
        for pattern, pen_weight in dim_info.get("pattern_penalties", {}).items():
            if code_patterns.get(pattern):
                penalty += pen_weight
        frac = max(0.0, frac - penalty)

        # Structure bonuses
        bonus = 0.0
        for check, bon_weight in dim_info.get("structure_bonuses", {}).items():
            if structure.get(check):
                bonus += bon_weight
        frac = min(1.0, frac + bonus * 0.2)  # Bonuses are minor

        frac = round(max(0.0, min(1.0, frac)), 3)
        weighted = round(frac * dim_info["weight"], 2)
        total_score += weighted

        dimension_scores[dim_key] = {
            "id": dim_info["id"],
            "name": dim_info["name"],
            "weight": dim_info["weight"],
            "score_fraction": frac,
            "weighted_score": weighted,
            "max_possible": dim_info["weight"],
        }

    return dimension_scores, round(total_score, 2)


def build_remediation_plan(
    issues: List[Dict[str, Any]],
    dimension_scores: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build prioritized remediation plan."""
    plan: List[Dict[str, Any]] = []

    # Group critical/error issues first
    critical_issues = [i for i in issues if i.get("severity") in ("CRITICAL", "ERROR")]
    warning_issues = [i for i in issues if i.get("severity") == "WARNING"]

    # Lowest scoring dimensions
    sorted_dims = sorted(
        dimension_scores.items(),
        key=lambda x: x[1]["score_fraction"],
    )

    # P0: Critical issues
    for issue in critical_issues[:5]:
        step: Dict[str, Any] = {
            "priority": "P0",
            "severity": issue["severity"],
            "description": issue.get("description", ""),
            "error_code": issue.get("error_code", ""),
            "fix": issue.get("fix", ""),
        }
        if issue.get("tripod_ai_violations"):
            violations = issue["tripod_ai_violations"]
            step["tripod_ai_items"] = [
                f"Item {v['item_id']}: {v['label']}" for v in violations
            ]
        if issue.get("literature_citations"):
            step["supporting_literature"] = [
                f"{c['id']}: {c['title']} ({c['journal']}, {c['year']})"
                for c in issue["literature_citations"][:2]
            ]
        plan.append(step)

    # P1: Warning issues
    for issue in warning_issues[:5]:
        plan.append({
            "priority": "P1",
            "severity": issue["severity"],
            "description": issue.get("description", ""),
            "error_code": issue.get("error_code", ""),
            "fix": issue.get("fix", ""),
        })

    # P2: Low-scoring dimensions
    for dim_key, dim_data in sorted_dims[:3]:
        if dim_data["score_fraction"] < 0.7:
            # Get literature support
            lit_refs = KB.lookup_lit_by_dimension(dim_key)
            step: Dict[str, Any] = {
                "priority": "P2",
                "severity": "INFO",
                "description": f"Improve {dim_data['name']} dimension score "
                               f"({dim_data['weighted_score']}/{dim_data['max_possible']})",
                "dimension": dim_data["name"],
                "fix": f"Address all failing checks in the {dim_data['name']} dimension.",
            }
            if lit_refs:
                step["supporting_literature"] = [
                    f"{e['id']}: {e['title']} ({e['journal']}, {e['year']})"
                    for e in lit_refs[:2]
                ]
            plan.append(step)

    return plan


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown_report(report: Dict[str, Any]) -> str:
    """Render the full audit report as publication-quality Markdown."""
    lines: List[str] = []

    project_name = Path(report["project_dir"]).name
    score = report["total_score"]
    grade_en = report["grade_en"]
    grade_zh = report["grade_zh"]
    timestamp = report["generated_at"]

    lines += [
        "# ML Leakage Guard — External Project Audit Report",
        "",
        f"**Project**: `{project_name}`  ",
        f"**Path**: `{report['project_dir']}`  ",
        f"**Generated**: {timestamp}  ",
        f"**MLGG Version**: 1.0 (31-gate pipeline)  ",
        f"**Standard References**: TRIPOD+AI 2024, PROBAST+AI 2025, STARD-AI 2021",
        "",
        "---",
        "",
    ]

    # Overall score
    score_bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
    lines += [
        "## Overall Score",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Total Score** | **{score} / 100** |",
        f"| Grade | {grade_en} / {grade_zh} |",
        f"| Score Bar | `{score_bar}` {score}% |",
        "",
    ]

    # Grade explanation
    if score >= 90:
        lines.append("> ✓ **Publication-grade**: Meets requirements for top-tier journals (Nature Medicine, JAMA, BMJ).")
    elif score >= 75:
        lines.append("> ⚠ **Solid but gaps remain**: Suitable for conference submission; address gaps before top-journal submission.")
    elif score >= 60:
        lines.append("> ✗ **Major issues**: Significant methodological concerns must be resolved before any submission.")
    else:
        lines.append("> ✗ **Not publishable**: Fundamental flaws detected. Comprehensive rework required.")
    lines.append("")

    # 10-Dimension scores
    lines += [
        "## 10-Dimension Scores",
        "",
        "| # | Dimension | Score | Max | Grade |",
        "|---|-----------|-------|-----|-------|",
    ]
    for dim_key, dim in sorted(report["dimension_scores"].items(), key=lambda x: x[1]["id"]):
        frac = dim["score_fraction"]
        grade_icon = "✓" if frac >= 0.8 else ("⚠" if frac >= 0.5 else "✗")
        lines.append(
            f"| {dim['id']} | {dim['name']} | {dim['weighted_score']} | {dim['max_possible']} | {grade_icon} |"
        )
    lines.append("")

    # TRIPOD+AI coverage
    tripod = report.get("tripod_coverage", {})
    if tripod:
        covered = tripod.get("covered_count", 0)
        total = tripod.get("total_required", 17)
        frac = tripod.get("coverage_fraction", 0)
        lines += [
            "## TRIPOD+AI 2024 Coverage",
            "",
            f"**Reference**: {tripod.get('reference', 'Collins et al. BMJ 2024;385:e078378')}  ",
            f"**Coverage**: {covered}/{total} required items ({int(frac*100)}%)  ",
            "",
            "| Item ID | Label | AI-Specific | Status |",
            "|---------|-------|-------------|--------|",
        ]
        for item in tripod.get("items", []):
            status_icon = {
                "covered": "✓ Covered",
                "gate_failed": "✗ Gate Failed",
                "not_assessed": "? Not Assessed",
            }.get(item["status"], item["status"])
            ai_flag = "★" if item.get("ai_specific") else ""
            lines.append(
                f"| {item['item_id']} | {item['label']} | {ai_flag} | {status_icon} |"
            )
        lines.append("")

    # PROBAST+AI assessment
    probast = report.get("probast_coverage", {})
    if probast:
        overall_rob = probast.get("overall_risk_of_bias", "unclear")
        rob_icon = {"low": "✓", "unclear": "?", "high": "✗"}.get(overall_rob, "?")
        lines += [
            "## PROBAST+AI 2025 Risk-of-Bias Assessment",
            "",
            f"**Reference**: {probast.get('reference', 'Wolff et al. PROBAST+AI 2025')}  ",
            f"**Overall Risk of Bias**: {rob_icon} **{overall_rob.upper()}**  ",
            "",
            "| Domain | ROB Status | Gates Assessed |",
            "|--------|-----------|----------------|",
        ]
        for domain in probast.get("domains", []):
            rob = domain["rob_status"]
            icon = {"low": "✓", "unclear": "?", "high": "✗"}.get(rob, "?")
            gates_str = ", ".join(domain["gates_assessed"][:3])
            lines.append(
                f"| {domain['name']} | {icon} {rob} | `{gates_str}` |"
            )
        lines.append("")

    # Issues found
    issues = report.get("issues", [])
    if issues:
        critical_errors = [i for i in issues if i.get("severity") in ("CRITICAL", "ERROR")]
        warnings = [i for i in issues if i.get("severity") == "WARNING"]
        infos = [i for i in issues if i.get("severity") == "INFO"]

        lines += [
            "## Issues Found",
            "",
            f"**Total**: {len(issues)} issues "
            f"({len(critical_errors)} critical/error, {len(warnings)} warning, {len(infos)} info)",
            "",
        ]

        if critical_errors:
            lines += ["### Critical / Error Issues", ""]
            for issue in critical_errors[:10]:
                lines += _render_issue_block(issue)

        if warnings:
            lines += ["### Warnings", ""]
            for issue in warnings[:8]:
                lines += _render_issue_block(issue)

        if infos:
            lines += ["### Informational", ""]
            for issue in infos[:5]:
                lines += _render_issue_block(issue)

    # Remediation plan
    plan = report.get("remediation_plan", [])
    if plan:
        lines += [
            "## Remediation Plan",
            "",
            "Prioritized action list to reach publication-grade status:",
            "",
        ]
        for i, step in enumerate(plan, 1):
            priority = step.get("priority", "P?")
            severity = step.get("severity", "")
            desc = step.get("description", "")
            fix = step.get("fix", "")
            lines += [
                f"### [{priority}] {desc}",
                "",
                f"**Severity**: {severity}  ",
            ]
            if step.get("error_code"):
                lines.append(f"**Error Code**: `{step['error_code']}`  ")
            lines += ["", f"**Fix**: {fix}", ""]
            if step.get("tripod_ai_items"):
                lines.append(f"**TRIPOD+AI Violations**: {', '.join(step['tripod_ai_items'])}  ")
                lines.append("")
            if step.get("supporting_literature"):
                lines.append("**Supporting Literature**:")
                for lit in step["supporting_literature"]:
                    lines.append(f"- {lit}")
                lines.append("")

    # Journal gap analysis
    journal_gap = report.get("journal_gap_analysis")
    if journal_gap:
        lines += [
            "## Journal Gap Analysis",
            "",
            f"**Target Journal**: {journal_gap.get('target_journal', 'N/A')}  ",
            f"**Minimum Score Required**: {journal_gap.get('minimum_score', 'N/A')}  ",
            f"**Current Score**: {journal_gap.get('current_score', 'N/A')}  ",
            f"**Meets Threshold**: {'✓ Yes' if journal_gap.get('meets_threshold') else '✗ No'}  ",
            f"**Mandatory Compliance**: {journal_gap.get('mandatory_compliance', 'N/A')}  ",
            "",
        ]
        unmet = journal_gap.get("mandatory_unmet", [])
        if unmet:
            lines += ["**Unmet Requirements**:", ""]
            for req in unmet:
                lines.append(f"- ✗ {req}")
            lines.append("")

    # Footer
    lines += [
        "---",
        "",
        "## Report Metadata",
        "",
        "| Key | Value |",
        "|-----|-------|",
        f"| Generated by | ML Leakage Guard (MLGG) v1.0 |",
        f"| Report version | {report.get('report_version', 'audit_report.v2')} |",
        f"| Error KB entries | {len(KB.error_entries)} |",
        f"| Literature KB entries | {len(KB.lit_entries)} |",
        f"| TRIPOD+AI items | 27 (Collins et al. BMJ 2024;385:e078378) |",
        f"| PROBAST+AI domains | 4 + AI supplementary (Wolff et al. 2025) |",
        "",
    ]

    return "\n".join(lines)


def _render_issue_block(issue: Dict[str, Any]) -> List[str]:
    """Render a single issue as a Markdown block."""
    lines: List[str] = []
    issue_type = issue.get("issue_type", "")
    severity = issue.get("severity", "INFO")
    description = issue.get("description", "")
    error_code = issue.get("error_code", "")

    if issue_type == "code_pattern":
        source = f"Code pattern: `{issue.get('pattern', '')}`"
        files = issue.get("affected_files", [])
        file_str = f" ({issue.get('file_count', len(files))} files)"
    else:
        source = f"Gate: `{issue.get('gate', '')}`"
        file_str = ""

    lines += [
        f"#### [{severity}] {description}{file_str}",
        "",
        f"**Source**: {source}  ",
    ]

    if error_code:
        lines.append(f"**Error Code**: `{error_code}`  ")

    root_cause = issue.get("root_cause", "")
    if root_cause and root_cause != description:
        lines.append(f"**Root Cause**: {root_cause}  ")

    fix = issue.get("fix", "")
    if fix:
        lines += ["", f"**Fix**: {fix}  "]

    prevention = issue.get("prevention", "")
    if prevention:
        lines.append(f"**Prevention**: {prevention}  ")

    tripod_violations = issue.get("tripod_ai_violations", [])
    if tripod_violations:
        items_str = ", ".join(
            f"Item {v['item_id']} ({v['label']})" for v in tripod_violations
        )
        lines.append(f"**TRIPOD+AI Violations**: {items_str}  ")
        ref = issue.get("tripod_ai_reference", "")
        if ref:
            lines.append(f"**Reference**: {ref}  ")

    probast_domain = issue.get("probast_domain", "")
    if probast_domain:
        lines.append(f"**PROBAST+AI Domain**: {probast_domain}  ")

    lit_citations = issue.get("literature_citations", [])
    if lit_citations:
        lines += ["", "**Literature**:"]
        for cite in lit_citations[:2]:
            lines.append(
                f"- {cite['id']}: {cite['title']} (*{cite['journal']}*, {cite['year']})"
            )

    if issue_type == "code_pattern" and issue.get("affected_files"):
        files = issue["affected_files"][:5]
        lines += ["", "**Affected Files**:"]
        for f in files:
            lines.append(f"- `{f}`")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_audit_report(
    project_dir: Path,
    output_dir: Optional[Path] = None,
    target_journal: Optional[str] = None,
    output_format: str = "both",
) -> Dict[str, Any]:
    """Generate comprehensive audit report for a project.

    Args:
        project_dir: Path to the project to audit.
        output_dir: Where to write the report (defaults to project_dir/audit-reports/).
        target_journal: Optional journal key for gap analysis.
        output_format: "markdown", "json", or "both" (default).

    Returns:
        Full report dict.
    """
    project_dir = project_dir.expanduser().resolve()
    if not project_dir.is_dir():
        raise ValueError(f"Project directory not found: {project_dir}")

    if output_dir is None:
        output_dir = project_dir / "audit-reports"

    # Scan project
    scan_results = scan_project(project_dir)
    gate_reports = scan_results["gate_reports"]

    # Score dimensions
    dimension_scores, total_score = compute_dimension_scores(scan_results, gate_reports)
    grade_en, grade_zh = _score_interpretation(total_score)

    # Assess reporting standards
    tripod_coverage = assess_tripod_coverage(project_dir, gate_reports)
    probast_coverage = assess_probast_coverage(gate_reports)

    # Build issues
    issues = build_issue_list(scan_results, gate_reports)

    # Build remediation plan
    remediation_plan = build_remediation_plan(issues, dimension_scores)

    # Journal gap analysis (if requested)
    journal_gap: Optional[Dict[str, Any]] = None
    if target_journal:
        journal_info = KB.journal_standards.get(target_journal)
        if journal_info:
            ml_reqs = journal_info.get("ml_prediction_requirements", {})
            min_score = ml_reqs.get("minimum_score", 80)
            mandatory = ml_reqs.get("mandatory", [])
            unmet: List[str] = []
            met: List[str] = []
            for req in mandatory:
                gate = req.get("gate")
                if gate and gate_reports.get(gate, {}).get("status", "").lower() == "pass":
                    met.append(req["requirement"])
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
                "mandatory_compliance": f"{len(met)}/{len(met)+len(unmet)}",
            }

    report: Dict[str, Any] = {
        "report_version": "audit_report.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir),
        "total_score": total_score,
        "max_score": 100,
        "grade_en": grade_en,
        "grade_zh": grade_zh,
        "py_files_scanned": scan_results["py_file_count"],
        "gate_reports_found": len(gate_reports),
        "dimension_scores": dimension_scores,
        "tripod_coverage": tripod_coverage,
        "probast_coverage": probast_coverage,
        "issues": issues,
        "remediation_plan": remediation_plan,
        "structure_checks": scan_results["structure"],
    }
    if journal_gap is not None:
        report["journal_gap_analysis"] = journal_gap

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format in ("json", "both"):
        json_path = output_dir / "audit-report.json"
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON report → {json_path}")

    if output_format in ("markdown", "both"):
        md_path = output_dir / "audit-report.md"
        md_content = render_markdown_report(report)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown report → {md_path}")

    # Print summary to stdout
    _print_summary(report)

    return report


def _print_summary(report: Dict[str, Any]) -> None:
    """Print a concise console summary."""
    score = report["total_score"]
    grade_en = report["grade_en"]
    project = Path(report["project_dir"]).name
    issues = report.get("issues", [])
    critical = sum(1 for i in issues if i.get("severity") in ("CRITICAL", "ERROR"))
    tripod = report.get("tripod_coverage", {})
    probast = report.get("probast_coverage", {})

    bar_width = 40
    filled = int(score / 100 * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    print()
    print("=" * 65)
    print(f"  MLGG Audit Report — {project}")
    print("=" * 65)
    print(f"  Score:   {score:5.1f} / 100  [{bar}]")
    print(f"  Grade:   {grade_en}")
    print(f"  Issues:  {len(issues)} total  ({critical} critical/error)")
    if tripod:
        covered = tripod.get("covered_count", 0)
        total_req = tripod.get("total_required", 17)
        print(f"  TRIPOD+AI 2024:  {covered}/{total_req} required items covered")
    if probast:
        rob = probast.get("overall_risk_of_bias", "unclear").upper()
        print(f"  PROBAST+AI 2025: Overall ROB = {rob}")
    print("=" * 65)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive audit report for a medical ML project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full audit with markdown + JSON output
  python3 scripts/generate_audit_report.py --project-dir /path/to/project

  # Target a specific journal
  python3 scripts/generate_audit_report.py --project-dir /path/to/project \\
      --target-journal nature_medicine

  # JSON only
  python3 scripts/generate_audit_report.py --project-dir /path/to/project \\
      --format json

  # Custom output directory
  python3 scripts/generate_audit_report.py --project-dir /path/to/project \\
      --output-dir /tmp/reports/
""",
    )
    parser.add_argument(
        "--project-dir", required=True,
        help="Path to the project to audit.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for reports. Defaults to <project-dir>/audit-reports/",
    )
    parser.add_argument(
        "--target-journal",
        choices=["nature_medicine", "lancet_digital_health", "jama", "bmj", "npj_digital_medicine"],
        help="Target journal for gap analysis.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format (default: both).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    try:
        run_audit_report(
            project_dir=project_dir,
            output_dir=output_dir,
            target_journal=args.target_journal,
            output_format=args.format,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
