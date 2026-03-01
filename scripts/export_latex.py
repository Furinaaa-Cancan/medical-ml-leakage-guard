#!/usr/bin/env python3
"""
LaTeX Table Export Tool for ML Leakage Guard.

Generates publication-ready LaTeX tabular code from evidence JSON reports,
following common medical journal conventions:
  - Table 1: Baseline Characteristics
  - Table 2: Model Performance Metrics
  - Table 3: External Validation (if available)

Usage:
    python3 scripts/export_latex.py \
        --evaluation-report evidence/evaluation_report.json \
        --output evidence/tables.tex

    python3 scripts/export_latex.py \
        --evaluation-report evidence/evaluation_report.json \
        --external-report evidence/external_validation_gate_report.json \
        --output evidence/tables.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Export evidence reports as LaTeX tables.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json.")
    parser.add_argument("--model-selection-report", help="Path to model_selection_report.json.")
    parser.add_argument("--external-report", help="Path to external_validation_gate_report.json.")
    parser.add_argument("--ci-matrix-report", help="Path to ci_matrix_gate_report.json.")
    parser.add_argument("--output", default="tables.tex", help="Output .tex file path.")
    parser.add_argument("--decimal-places", type=int, default=3, help="Decimal places for metrics (default: 3).")
    return parser.parse_args()


def _fmt(value: Any, dp: int = 3) -> str:
    """Format a numeric value to fixed decimal places."""
    if value is None:
        return "---"
    try:
        v = float(value)
        return f"{v:.{dp}f}"
    except (ValueError, TypeError):
        return str(value)


def _fmt_ci(low: Any, high: Any, dp: int = 3) -> str:
    """Format a confidence interval as [low, high]."""
    if low is None or high is None:
        return ""
    return f"[{_fmt(low, dp)}, {_fmt(high, dp)}]"


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a JSON file if path is provided and exists."""
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _escape(text: str) -> str:
    """Escape LaTeX special characters."""
    for ch in ["_", "&", "%", "#", "$"]:
        text = text.replace(ch, f"\\{ch}")
    return text


def table_performance(eval_report: Dict[str, Any], dp: int) -> str:
    """Generate Table 2: Model Performance Metrics."""
    lines: List[str] = []
    lines.append("% Table 2: Model Performance Metrics")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model Performance Metrics}")
    lines.append("\\label{tab:performance}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Split & AUC-ROC & AUC-PR & Brier Score & Accuracy \\\\")
    lines.append("\\midrule")

    splits = eval_report.get("splits", eval_report.get("per_split_metrics", {}))
    if isinstance(splits, dict):
        for split_name, metrics in splits.items():
            if not isinstance(metrics, dict):
                continue
            roc = _fmt(metrics.get("roc_auc", metrics.get("auroc")), dp)
            pr = _fmt(metrics.get("pr_auc", metrics.get("auprc", metrics.get("average_precision"))), dp)
            brier = _fmt(metrics.get("brier_score", metrics.get("brier")), dp)
            acc = _fmt(metrics.get("accuracy"), dp)
            lines.append(f"{_escape(split_name)} & {roc} & {pr} & {brier} & {acc} \\\\")
    elif isinstance(splits, list):
        for entry in splits:
            if not isinstance(entry, dict):
                continue
            split_name = str(entry.get("split", entry.get("name", "?")))
            roc = _fmt(entry.get("roc_auc", entry.get("auroc")), dp)
            pr = _fmt(entry.get("pr_auc", entry.get("auprc")), dp)
            brier = _fmt(entry.get("brier_score", entry.get("brier")), dp)
            acc = _fmt(entry.get("accuracy"), dp)
            lines.append(f"{_escape(split_name)} & {roc} & {pr} & {brier} & {acc} \\\\")

    # Top-level metrics fallback
    if not splits:
        for key in ["test", "valid", "train"]:
            roc = eval_report.get(f"{key}_roc_auc", eval_report.get(f"{key}_auroc"))
            if roc is not None:
                pr = eval_report.get(f"{key}_pr_auc", eval_report.get(f"{key}_auprc"))
                brier = eval_report.get(f"{key}_brier_score")
                acc = eval_report.get(f"{key}_accuracy")
                lines.append(f"{_escape(key)} & {_fmt(roc, dp)} & {_fmt(pr, dp)} & {_fmt(brier, dp)} & {_fmt(acc, dp)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_model_selection(ms_report: Dict[str, Any], dp: int) -> str:
    """Generate Table: Model Selection Summary."""
    candidates = ms_report.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""

    lines: List[str] = []
    lines.append("% Table: Model Selection Summary")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Model Selection Summary}")
    lines.append("\\label{tab:model_selection}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Model & CV Score & Complexity & Selected \\\\")
    lines.append("\\midrule")

    for cand in candidates[:10]:
        if not isinstance(cand, dict):
            continue
        name = _escape(str(cand.get("family", cand.get("model_id", "?"))))
        score = _fmt(cand.get("cv_score", cand.get("mean_cv_score")), dp)
        complexity = str(cand.get("complexity_rank", cand.get("complexity", "---")))
        selected = "\\checkmark" if cand.get("selected", False) else ""
        lines.append(f"{name} & {score} & {complexity} & {selected} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_external(ext_report: Dict[str, Any], dp: int) -> str:
    """Generate Table 3: External Validation."""
    lines: List[str] = []
    lines.append("% Table 3: External Validation")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{External Validation Results}")
    lines.append("\\label{tab:external}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Cohort & AUC-ROC & AUC-PR & Brier Score \\\\")
    lines.append("\\midrule")

    cohorts = ext_report.get("cohorts", ext_report.get("external_cohorts", []))
    if isinstance(cohorts, list):
        for cohort in cohorts:
            if not isinstance(cohort, dict):
                continue
            name = _escape(str(cohort.get("name", cohort.get("cohort_name", "?"))))
            roc = _fmt(cohort.get("roc_auc", cohort.get("auroc")), dp)
            pr = _fmt(cohort.get("pr_auc", cohort.get("auprc")), dp)
            brier = _fmt(cohort.get("brier_score", cohort.get("brier")), dp)
            lines.append(f"{name} & {roc} & {pr} & {brier} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def table_ci_matrix(ci_report: Dict[str, Any], dp: int) -> str:
    """Generate Table: Confidence Intervals."""
    matrix = ci_report.get("ci_matrix", ci_report.get("confidence_intervals", []))
    if not isinstance(matrix, list) or not matrix:
        return ""

    lines: List[str] = []
    lines.append("% Table: Confidence Intervals")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Confidence Intervals for Key Metrics}")
    lines.append("\\label{tab:ci}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append("Metric & Split & Point Estimate & 95\\% CI \\\\")
    lines.append("\\midrule")

    for entry in matrix[:20]:
        if not isinstance(entry, dict):
            continue
        metric = _escape(str(entry.get("metric", "?")))
        split = _escape(str(entry.get("split", "?")))
        point = _fmt(entry.get("point_estimate", entry.get("value")), dp)
        ci = _fmt_ci(entry.get("ci_lower", entry.get("lower")),
                     entry.get("ci_upper", entry.get("upper")), dp)
        lines.append(f"{metric} & {split} & {point} & {ci} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    args = parse_args()
    dp = int(args.decimal_places)

    eval_report = _load(args.evaluation_report)
    if eval_report is None:
        print(f"Evaluation report not found: {args.evaluation_report}", file=sys.stderr)
        return 1

    sections: List[str] = []
    sections.append("% Auto-generated by ML Leakage Guard export_latex.py")
    sections.append("% Requires: \\usepackage{booktabs}")
    sections.append("")

    sections.append(table_performance(eval_report, dp))
    sections.append("")

    ms_report = _load(args.model_selection_report)
    if ms_report:
        ms_table = table_model_selection(ms_report, dp)
        if ms_table:
            sections.append(ms_table)
            sections.append("")

    ext_report = _load(args.external_report)
    if ext_report:
        sections.append(table_external(ext_report, dp))
        sections.append("")

    ci_report = _load(args.ci_matrix_report)
    if ci_report:
        ci_table = table_ci_matrix(ci_report, dp)
        if ci_table:
            sections.append(ci_table)
            sections.append("")

    output = "\n".join(sections)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"LaTeX tables written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
