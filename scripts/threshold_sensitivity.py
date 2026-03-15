#!/usr/bin/env python3
"""
Threshold Sensitivity Analyzer for ML Leakage Guard.

Scans gate reports in an evidence directory and analyzes how close each
metric sits relative to its pass/fail threshold.  Identifies fragile gates
(within a configurable margin) and simulates stricter / looser policies.

Usage:
    python3 scripts/threshold_sensitivity.py --evidence-dir evidence/
    python3 scripts/threshold_sensitivity.py --evidence-dir evidence/ --margin 0.05
    python3 scripts/threshold_sensitivity.py --evidence-dir evidence/ --json
    python3 scripts/threshold_sensitivity.py --evidence-dir evidence/ --markdown --output sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Gate metric extraction rules ────────────────────────────────────────────
# Each entry: (report_filename, list of (metric_label, json_path_parts,
#              threshold_source, direction))
# direction: "lower_is_better" means metric must stay BELOW threshold,
#            "higher_is_better" means metric must stay ABOVE threshold.

_GATE_METRIC_SPECS: List[Tuple[str, List[Tuple[str, List[str], float, str]]]] = [
    ("robustness_gate_report.json", [
        ("time_slices.pr_auc_worst_drop",
         ["summary", "computed", "time_slices", "pr_auc_worst_drop_from_overall"],
         0.14, "lower_is_better"),
        ("time_slices.pr_auc_range",
         ["summary", "computed", "time_slices", "pr_auc_range"],
         0.20, "lower_is_better"),
        ("patient_hash_groups.pr_auc_worst_drop",
         ["summary", "computed", "patient_hash_groups", "pr_auc_worst_drop_from_overall"],
         0.14, "lower_is_better"),
        ("patient_hash_groups.pr_auc_range",
         ["summary", "computed", "patient_hash_groups", "pr_auc_range"],
         0.20, "lower_is_better"),
    ]),
    ("generalization_gap_report.json", [
        ("train_test_auc_gap",
         ["summary", "train_test_auc_gap"],
         0.05, "lower_is_better"),
    ]),
    ("calibration_dca_report.json", [
        ("hosmer_lemeshow_p",
         ["summary", "hosmer_lemeshow_p_value"],
         0.05, "higher_is_better"),
        ("brier_score",
         ["summary", "brier_score"],
         0.25, "lower_is_better"),
    ]),
    ("seed_stability_report.json", [
        ("metric_cv",
         ["summary", "pr_auc_cv"],
         0.05, "lower_is_better"),
    ]),
    ("external_validation_gate_report.json", [
        ("cohort_pr_auc_drop",
         ["summary", "replayed_cohorts", 0, "transport_gap", "pr_auc_drop_from_internal_test"],
         0.08, "lower_is_better"),
    ]),
]


def _deep_get(data: Any, keys: List[Any]) -> Optional[Any]:
    """Walk a nested dict/list by key path, returning None on miss."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list):
            if isinstance(key, int) and 0 <= key < len(current):
                current = current[key]
            else:
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _is_finite(value: Any) -> bool:
    """Return True if value is a finite number (not bool)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def load_report(evidence_dir: Path, filename: str) -> Optional[Dict[str, Any]]:
    """Load a JSON gate report, returning None if missing or invalid."""
    path = evidence_dir / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def compute_margin(
    value: float,
    threshold: float,
    direction: str,
) -> Dict[str, Any]:
    """Compute how far a metric is from its threshold.

    Returns a dict with margin (positive = safe, negative = failing),
    headroom_pct (percentage of threshold), and status.
    """
    if direction == "lower_is_better":
        margin = threshold - value
    else:  # higher_is_better
        margin = value - threshold

    if threshold != 0.0:
        headroom_pct = (margin / abs(threshold)) * 100.0
    else:
        headroom_pct = float("inf") if margin > 0 else float("-inf") if margin < 0 else 0.0

    if margin < 0:
        status = "FAIL"
    elif margin == 0:
        status = "BORDERLINE"
    else:
        status = "PASS"

    return {
        "margin": round(margin, 6),
        "headroom_pct": round(headroom_pct, 2) if math.isfinite(headroom_pct) else None,
        "status": status,
    }


def extract_metrics(
    evidence_dir: Path,
) -> List[Dict[str, Any]]:
    """Extract all threshold-relevant metrics from gate reports."""
    results: List[Dict[str, Any]] = []
    for filename, specs in _GATE_METRIC_SPECS:
        report = load_report(evidence_dir, filename)
        if report is None:
            continue
        gate_name = filename.replace("_report.json", "").replace(".json", "")
        for label, path_keys, default_threshold, direction in specs:
            value = _deep_get(report, path_keys)
            if not _is_finite(value):
                continue
            value_f = float(value)
            margin_info = compute_margin(value_f, default_threshold, direction)
            results.append({
                "gate": gate_name,
                "metric": label,
                "value": round(value_f, 6),
                "threshold": default_threshold,
                "direction": direction,
                **margin_info,
            })
    return results


def classify_fragile(
    metrics: List[Dict[str, Any]],
    margin_pct: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split metrics into failing, fragile (within margin%), and safe."""
    failing: List[Dict[str, Any]] = []
    fragile: List[Dict[str, Any]] = []
    safe: List[Dict[str, Any]] = []
    for m in metrics:
        if m["status"] == "FAIL":
            failing.append(m)
        elif m["headroom_pct"] is not None and abs(m["headroom_pct"]) <= margin_pct:
            fragile.append(m)
        else:
            safe.append(m)
    return failing, fragile, safe


def simulate_policy(
    metrics: List[Dict[str, Any]],
    factor: float,
) -> List[Dict[str, Any]]:
    """Re-evaluate metrics under a scaled threshold (factor * original)."""
    simulated: List[Dict[str, Any]] = []
    for m in metrics:
        new_threshold = m["threshold"] * factor
        new_margin = compute_margin(m["value"], new_threshold, m["direction"])
        simulated.append({
            **m,
            "threshold": round(new_threshold, 6),
            **new_margin,
        })
    return simulated


def build_analysis(
    evidence_dir: Path,
    margin_pct: float = 20.0,
) -> Dict[str, Any]:
    """Build complete sensitivity analysis."""
    metrics = extract_metrics(evidence_dir)
    failing, fragile, safe = classify_fragile(metrics, margin_pct)

    strict_sim = simulate_policy(metrics, 0.8)
    _, strict_fragile, _ = classify_fragile(strict_sim, margin_pct)
    strict_failing = [m for m in strict_sim if m["status"] == "FAIL"]

    relaxed_sim = simulate_policy(metrics, 1.2)
    relaxed_failing = [m for m in relaxed_sim if m["status"] == "FAIL"]

    return {
        "evidence_dir": str(evidence_dir),
        "margin_pct": margin_pct,
        "total_metrics": len(metrics),
        "failing_count": len(failing),
        "fragile_count": len(fragile),
        "safe_count": len(safe),
        "metrics": metrics,
        "failing": failing,
        "fragile": fragile,
        "safe": safe,
        "simulations": {
            "strict_0.8x": {
                "factor": 0.8,
                "description": "Thresholds tightened by 20%",
                "failing_count": len(strict_failing),
                "fragile_count": len(strict_fragile),
                "new_failures": [
                    m for m in strict_failing
                    if m["metric"] not in {f["metric"] for f in failing}
                ],
            },
            "relaxed_1.2x": {
                "factor": 1.2,
                "description": "Thresholds relaxed by 20%",
                "failing_count": len(relaxed_failing),
                "resolved": [
                    m["metric"] for m in metrics
                    if m["status"] == "FAIL"
                    and m["metric"] not in {f["metric"] for f in relaxed_failing}
                ],
            },
        },
    }


def to_markdown(analysis: Dict[str, Any]) -> str:
    """Render analysis as Markdown."""
    lines: List[str] = []
    lines.append("# Threshold Sensitivity Analysis")
    lines.append("")
    lines.append(f"**Evidence directory:** `{analysis['evidence_dir']}`")
    lines.append(f"**Fragility margin:** {analysis['margin_pct']}%")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| Total metrics | {analysis['total_metrics']} |")
    lines.append(f"| Failing | {analysis['failing_count']} |")
    lines.append(f"| Fragile | {analysis['fragile_count']} |")
    lines.append(f"| Safe | {analysis['safe_count']} |")
    lines.append("")

    if analysis["failing"]:
        lines.append("## Failing Metrics")
        lines.append("")
        lines.append("| Gate | Metric | Value | Threshold | Margin |")
        lines.append("|------|--------|-------|-----------|--------|")
        for m in analysis["failing"]:
            lines.append(
                f"| {m['gate']} | {m['metric']} | {m['value']:.4f} "
                f"| {m['threshold']:.4f} | {m['margin']:+.4f} |"
            )
        lines.append("")

    if analysis["fragile"]:
        lines.append("## Fragile Metrics (within margin)")
        lines.append("")
        lines.append("| Gate | Metric | Value | Threshold | Headroom |")
        lines.append("|------|--------|-------|-----------|----------|")
        for m in analysis["fragile"]:
            hp = f"{m['headroom_pct']:.1f}%" if m["headroom_pct"] is not None else "N/A"
            lines.append(
                f"| {m['gate']} | {m['metric']} | {m['value']:.4f} "
                f"| {m['threshold']:.4f} | {hp} |"
            )
        lines.append("")

    sims = analysis.get("simulations", {})
    strict = sims.get("strict_0.8x", {})
    relaxed = sims.get("relaxed_1.2x", {})
    lines.append("## Policy Simulations")
    lines.append("")
    lines.append(f"- **Strict (0.8x):** {strict.get('failing_count', 0)} failures "
                 f"({len(strict.get('new_failures', []))} new)")
    lines.append(f"- **Relaxed (1.2x):** {relaxed.get('failing_count', 0)} failures "
                 f"({len(relaxed.get('resolved', []))} resolved)")
    lines.append("")
    return "\n".join(lines)


def to_text(analysis: Dict[str, Any]) -> str:
    """Render analysis as plain text."""
    lines: List[str] = []
    lines.append("=== Threshold Sensitivity Analysis ===")
    lines.append(f"Evidence: {analysis['evidence_dir']}")
    lines.append(f"Margin:   {analysis['margin_pct']}%")
    lines.append(f"Total:    {analysis['total_metrics']} metrics")
    lines.append(f"Failing:  {analysis['failing_count']}")
    lines.append(f"Fragile:  {analysis['fragile_count']}")
    lines.append(f"Safe:     {analysis['safe_count']}")
    lines.append("")

    if analysis["failing"]:
        lines.append("--- FAILING ---")
        for m in analysis["failing"]:
            lines.append(
                f"  [{m['gate']}] {m['metric']}: "
                f"{m['value']:.4f} vs threshold {m['threshold']:.4f} "
                f"(margin {m['margin']:+.4f})"
            )
        lines.append("")

    if analysis["fragile"]:
        lines.append("--- FRAGILE ---")
        for m in analysis["fragile"]:
            hp = f"{m['headroom_pct']:.1f}%" if m["headroom_pct"] is not None else "N/A"
            lines.append(
                f"  [{m['gate']}] {m['metric']}: "
                f"{m['value']:.4f} vs threshold {m['threshold']:.4f} "
                f"(headroom {hp})"
            )
        lines.append("")

    sims = analysis.get("simulations", {})
    strict = sims.get("strict_0.8x", {})
    relaxed = sims.get("relaxed_1.2x", {})
    lines.append("--- SIMULATIONS ---")
    lines.append(f"  Strict (0.8x): {strict.get('failing_count', 0)} failures "
                 f"({len(strict.get('new_failures', []))} new)")
    lines.append(f"  Relaxed (1.2x): {relaxed.get('failing_count', 0)} failures "
                 f"({len(relaxed.get('resolved', []))} resolved)")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze gate metric sensitivity to threshold changes.",
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to the evidence directory containing gate reports.",
    )
    parser.add_argument(
        "--margin", type=float, default=20.0,
        help="Fragility margin percentage (default: 20).",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON.",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Output as Markdown.",
    )
    parser.add_argument(
        "--output", help="Write output to file instead of stdout.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()

    if not evidence_dir.is_dir():
        print(f"Error: evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 1

    analysis = build_analysis(evidence_dir, margin_pct=args.margin)

    if args.as_json:
        text = json.dumps(analysis, indent=2, ensure_ascii=False)
    elif args.markdown:
        text = to_markdown(analysis)
    else:
        text = to_text(analysis)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(text, encoding="utf-8")
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
