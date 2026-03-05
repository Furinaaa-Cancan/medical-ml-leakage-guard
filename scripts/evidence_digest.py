#!/usr/bin/env python3
"""
Evidence Digest — generate a compact, shareable summary from an evidence directory.

Produces a one-page Markdown or JSON digest suitable for paper submissions,
reviewer sharing, or quick pipeline status checks. Extracts key metrics,
gate statuses, model info, and data split statistics.

Usage:
    python3 scripts/evidence_digest.py --evidence-dir evidence/
    python3 scripts/evidence_digest.py --evidence-dir evidence/ --json
    python3 scripts/evidence_digest.py --evidence-dir evidence/ --output digest.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _get(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _fmt_pct(val: Any) -> str:
    """Format a float as percentage string."""
    if isinstance(val, (int, float)):
        return f"{val * 100:.1f}%"
    return "—"


def _fmt_float(val: Any, decimals: int = 4) -> str:
    """Format a float with fixed decimals."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return "—"


def extract_digest(evidence_dir: Path) -> Dict[str, Any]:
    """Extract digest data from an evidence directory."""

    # Pipeline status
    pipeline = _load(evidence_dir / "dag_pipeline_report.json")
    if pipeline is None:
        pipeline = _load(evidence_dir / "strict_pipeline_report.json")

    pipeline_status = _get(pipeline, "status", default="unknown")

    # Key metrics from evaluation report
    eval_rpt = _load(evidence_dir / "evaluation_report.json")
    metrics = {}
    metric_keys = [
        "roc_auc", "pr_auc", "sensitivity", "specificity",
        "ppv", "npv", "brier_score", "f_beta",
    ]
    if eval_rpt:
        for k in metric_keys:
            val = eval_rpt.get(k)
            if val is None:
                val = _get(eval_rpt, "summary", k)
            if isinstance(val, (int, float)):
                metrics[k] = round(float(val), 6)

    # Model selection info
    model_sel = _load(evidence_dir / "model_selection_audit_report.json")
    selected_model = _get(model_sel, "summary", "selected_model_name", default=None)
    candidate_count = _get(model_sel, "summary", "candidate_count", default=None)

    # Split statistics
    split_rpt = _load(evidence_dir / "split_protocol_report.json")
    splits_summary: Dict[str, Any] = {}
    raw_splits = _get(split_rpt, "summary", "splits", default={})
    if isinstance(raw_splits, dict):
        for split_name, stats in raw_splits.items():
            if isinstance(stats, dict):
                splits_summary[split_name] = {
                    "rows": stats.get("row_count"),
                    "patients": stats.get("id_count"),
                    "prevalence": stats.get("prevalence"),
                }

    # Gate status counts
    gate_files = [
        "request_contract_report.json", "manifest.json",
        "split_protocol_report.json", "leakage_report.json",
        "definition_guard_report.json", "lineage_report.json",
        "covariate_shift_report.json", "imbalance_policy_report.json",
        "missingness_policy_report.json", "tuning_leakage_report.json",
        "model_selection_audit_report.json", "feature_engineering_audit_report.json",
        "clinical_metrics_report.json", "prediction_replay_report.json",
        "distribution_generalization_report.json", "generalization_gap_report.json",
        "robustness_gate_report.json", "seed_stability_report.json",
        "external_validation_gate_report.json", "calibration_dca_report.json",
        "ci_matrix_gate_report.json", "metric_consistency_report.json",
        "evaluation_quality_report.json", "permutation_report.json",
        "reporting_bias_report.json", "execution_attestation_report.json",
        "self_critique_report.json", "publication_gate_report.json",
    ]
    passed = 0
    failed = 0
    missing = 0
    for gf in gate_files:
        rpt = _load(evidence_dir / gf)
        if rpt is None:
            missing += 1
        elif rpt.get("status") == "pass":
            passed += 1
        else:
            failed += 1

    # Calibration
    cal_rpt = _load(evidence_dir / "calibration_dca_report.json")
    ece = _get(cal_rpt, "summary", "ece")

    # Publication gate
    pub_rpt = _load(evidence_dir / "publication_gate_report.json")
    pub_status = _get(pub_rpt, "status", default="missing")

    return {
        "schema_version": "evidence_digest.v1",
        "evidence_dir": str(evidence_dir),
        "pipeline_status": pipeline_status,
        "publication_status": pub_status,
        "gates": {
            "total": len(gate_files),
            "passed": passed,
            "failed": failed,
            "missing": missing,
        },
        "metrics": metrics,
        "calibration_ece": ece,
        "model": {
            "selected": selected_model,
            "candidates_evaluated": candidate_count,
        },
        "splits": splits_summary,
    }


def to_markdown(digest: Dict[str, Any]) -> str:
    """Render digest as compact Markdown."""
    lines: List[str] = []
    lines.append("# Evidence Digest")
    lines.append("")
    lines.append(f"**Evidence**: `{digest['evidence_dir']}`")
    lines.append(f"**Pipeline**: {digest['pipeline_status']}  ")
    lines.append(f"**Publication Gate**: {digest['publication_status']}")
    lines.append("")

    # Gates
    g = digest["gates"]
    lines.append("## Gate Summary")
    lines.append(f"- **Passed**: {g['passed']}/{g['total']}")
    lines.append(f"- **Failed**: {g['failed']}")
    lines.append(f"- **Missing**: {g['missing']}")
    lines.append("")

    # Model
    m = digest["model"]
    if m["selected"]:
        lines.append("## Model")
        lines.append(f"- **Selected**: {m['selected']}")
        if m["candidates_evaluated"]:
            lines.append(f"- **Candidates Evaluated**: {m['candidates_evaluated']}")
        lines.append("")

    # Metrics
    if digest["metrics"]:
        lines.append("## Key Metrics")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in digest["metrics"].items():
            lines.append(f"| {k} | {_fmt_float(v)} |")
        if digest.get("calibration_ece") is not None:
            lines.append(f"| ECE | {_fmt_float(digest['calibration_ece'])} |")
        lines.append("")

    # Splits
    if digest["splits"]:
        lines.append("## Data Splits")
        lines.append("| Split | Rows | Patients | Prevalence |")
        lines.append("|-------|------|----------|------------|")
        for name, info in digest["splits"].items():
            rows = info.get("rows", "—")
            patients = info.get("patients", "—")
            prev = _fmt_pct(info.get("prevalence"))
            lines.append(f"| {name} | {rows} | {patients} | {prev} |")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a compact evidence digest for sharing."
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to evidence directory.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON instead of Markdown.",
    )
    parser.add_argument(
        "--output", help="Write to file instead of stdout.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence = Path(args.evidence_dir).expanduser().resolve()

    if not evidence.is_dir():
        print(f"Evidence directory not found: {evidence}", file=sys.stderr)
        return 1

    digest = extract_digest(evidence)

    if args.json_output:
        output = json.dumps(digest, indent=2, ensure_ascii=False)
    else:
        output = to_markdown(digest)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Digest written to: {out_path}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
