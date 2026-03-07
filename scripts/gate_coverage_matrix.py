#!/usr/bin/env python3
"""
Gate Coverage Matrix for ML Leakage Guard.

Scans an evidence directory against the full gate registry to produce
a coverage matrix showing which gates have been executed, their status,
and which gates are missing from the evidence.

Usage:
    python3 scripts/gate_coverage_matrix.py --evidence-dir evidence/
    python3 scripts/gate_coverage_matrix.py --evidence-dir evidence/ --json
    python3 scripts/gate_coverage_matrix.py --evidence-dir evidence/ --output matrix.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── JSON loader ─────────────────────────────────────────────────────────────

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on missing/invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


# ── Gate registry snapshot ──────────────────────────────────────────────────

def load_gate_registry() -> Dict[str, Dict[str, Any]]:
    """Load gate specs from _gate_registry if available, else use fallback."""
    try:
        from _gate_registry import GATE_REGISTRY
        registry: Dict[str, Dict[str, Any]] = {}
        for name, spec in GATE_REGISTRY.items():
            registry[name] = {
                "script": spec.script,
                "layer": spec.layer.value if hasattr(spec.layer, "value") else int(spec.layer),
                "layer_name": spec.layer.name if hasattr(spec.layer, "name") else str(spec.layer),
                "output_report": spec.report_output,
                "category": getattr(spec, "category", "unknown"),
            }
        return registry
    except ImportError:
        return {}


# ── Evidence scanning ───────────────────────────────────────────────────────

def scan_gate_reports(
    evidence_dir: Path,
    registry: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each registered gate, check if its report exists and extract status."""
    results: List[Dict[str, Any]] = []
    for gate_name, spec in sorted(registry.items()):
        report_file = spec.get("output_report", "")
        if not report_file:
            report_file = f"{gate_name}_report.json"
        report_path = evidence_dir / report_file
        report_data = load_json(report_path)

        entry: Dict[str, Any] = {
            "gate_name": gate_name,
            "layer": spec.get("layer"),
            "layer_name": spec.get("layer_name", ""),
            "category": spec.get("category", "unknown"),
            "report_file": report_file,
            "present": report_data is not None,
            "status": None,
            "failure_count": None,
            "warning_count": None,
        }
        if report_data is not None:
            entry["status"] = str(report_data.get("status", "unknown"))
            entry["failure_count"] = report_data.get("failure_count")
            entry["warning_count"] = report_data.get("warning_count")
        results.append(entry)
    return results


def compute_matrix_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from gate coverage entries."""
    total = len(entries)
    present = sum(1 for e in entries if e["present"])
    missing = total - present
    coverage_pct = round(100.0 * present / total, 1) if total > 0 else 0.0

    status_counts: Dict[str, int] = {}
    for e in entries:
        if e["present"]:
            s = e["status"] or "unknown"
            status_counts[s] = status_counts.get(s, 0) + 1

    # Per-layer breakdown
    layer_map: Dict[str, Dict[str, int]] = {}
    for e in entries:
        lname = e.get("layer_name") or "unknown"
        if lname not in layer_map:
            layer_map[lname] = {"total": 0, "present": 0, "pass": 0, "fail": 0}
        layer_map[lname]["total"] += 1
        if e["present"]:
            layer_map[lname]["present"] += 1
            if e["status"] == "pass":
                layer_map[lname]["pass"] += 1
            elif e["status"] == "fail":
                layer_map[lname]["fail"] += 1

    missing_gates = [e["gate_name"] for e in entries if not e["present"]]

    return {
        "total_gates": total,
        "present_gates": present,
        "missing_gates_count": missing,
        "coverage_percent": coverage_pct,
        "status_counts": status_counts,
        "per_layer": layer_map,
        "missing_gates": missing_gates,
    }


# ── Output formatting ───────────────────────────────────────────────────────

def to_json(entries: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """Format as JSON."""
    return json.dumps({"summary": summary, "gates": entries}, indent=2, ensure_ascii=False)


def to_text(entries: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    """Format as human-readable text."""
    lines: List[str] = []
    lines.append("=== Gate Coverage Matrix ===")
    lines.append(f"Coverage: {summary['present_gates']}/{summary['total_gates']} ({summary['coverage_percent']}%)")
    lines.append(f"Status: {', '.join(f'{k}={v}' for k, v in sorted(summary['status_counts'].items()))}")
    lines.append("")

    if summary["missing_gates"]:
        lines.append("--- Missing Gates ---")
        for g in summary["missing_gates"]:
            lines.append(f"  - {g}")
        lines.append("")

    lines.append("--- Per-Layer Breakdown ---")
    for layer_name, counts in sorted(summary["per_layer"].items()):
        lines.append(
            f"  {layer_name}: {counts['present']}/{counts['total']} "
            f"(pass={counts['pass']}, fail={counts['fail']})"
        )
    lines.append("")

    lines.append("--- Gate Details ---")
    for e in entries:
        status_str = e["status"] if e["present"] else "MISSING"
        f_count = e["failure_count"] if e["failure_count"] is not None else "-"
        w_count = e["warning_count"] if e["warning_count"] is not None else "-"
        lines.append(f"  [{e['layer_name']}] {e['gate_name']}: {status_str} (F={f_count}, W={w_count})")
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a gate coverage matrix from an evidence directory.",
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to the evidence directory.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output JSON instead of text.",
    )
    parser.add_argument(
        "--output", help="Write output to file (default: stdout).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()

    if not evidence_dir.is_dir():
        print(f"Error: evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 1

    registry = load_gate_registry()
    if not registry:
        print("Error: gate registry is empty or unavailable.", file=sys.stderr)
        return 1

    entries = scan_gate_reports(evidence_dir, registry)
    summary = compute_matrix_summary(entries)

    if args.json_output:
        output = to_json(entries, summary)
    else:
        output = to_text(entries, summary)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Coverage matrix written to: {out_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
