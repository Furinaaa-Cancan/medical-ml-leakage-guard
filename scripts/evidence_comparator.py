#!/usr/bin/env python3
"""
Evidence Comparator for ML Leakage Guard.

Compares two evidence directories (baseline vs current) side-by-side,
showing which gates improved, regressed, or stayed the same between runs.
Highlights metric deltas and new/resolved failures.

Usage:
    python3 scripts/evidence_comparator.py --baseline evidence_v1/ --current evidence_v2/
    python3 scripts/evidence_comparator.py --baseline evidence_v1/ --current evidence_v2/ --json
    python3 scripts/evidence_comparator.py --baseline evidence_v1/ --current evidence_v2/ --output diff.json --json
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


# ── Report scanning ─────────────────────────────────────────────────────────

def scan_reports(evidence_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan evidence directory for *_report.json files."""
    reports: Dict[str, Dict[str, Any]] = {}
    if not evidence_dir.is_dir():
        return reports
    for p in sorted(evidence_dir.glob("*_report.json")):
        data = load_json(p)
        if data is not None:
            stem = p.stem
            gate_name = data.get("gate_name")
            if not isinstance(gate_name, str) or not gate_name:
                gate_name = stem[:-7] if stem.endswith("_report") else stem
            reports[gate_name] = {
                "file": p.name,
                "status": str(data.get("status", "unknown")),
                "failure_count": data.get("failure_count", 0),
                "warning_count": data.get("warning_count", 0),
                "failures": data.get("failures", []),
                "execution_time_seconds": data.get("execution_time_seconds"),
            }
    return reports


# ── Comparison logic ────────────────────────────────────────────────────────

def _extract_codes(failures: Any) -> List[str]:
    """Extract failure codes from a failures list."""
    if not isinstance(failures, list):
        return []
    codes = []
    for f in failures:
        if isinstance(f, dict):
            code = f.get("code")
            if isinstance(code, str) and code:
                codes.append(code)
    return codes


def compare_gates(
    baseline: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compare gate results between baseline and current."""
    all_gates = sorted(set(baseline.keys()) | set(current.keys()))
    results: List[Dict[str, Any]] = []

    for gate_name in all_gates:
        b = baseline.get(gate_name)
        c = current.get(gate_name)

        entry: Dict[str, Any] = {"gate_name": gate_name}

        if b is None and c is not None:
            entry["change"] = "new"
            entry["current_status"] = c["status"]
            entry["baseline_status"] = None
        elif b is not None and c is None:
            entry["change"] = "removed"
            entry["current_status"] = None
            entry["baseline_status"] = b["status"]
        elif b is not None and c is not None:
            b_status = b["status"]
            c_status = c["status"]
            entry["baseline_status"] = b_status
            entry["current_status"] = c_status

            if b_status == "fail" and c_status == "pass":
                entry["change"] = "improved"
            elif b_status == "pass" and c_status == "fail":
                entry["change"] = "regressed"
            elif b_status == c_status:
                entry["change"] = "unchanged"
            else:
                entry["change"] = "changed"

            # Failure code diff
            b_codes = set(_extract_codes(b.get("failures")))
            c_codes = set(_extract_codes(c.get("failures")))
            entry["new_failures"] = sorted(c_codes - b_codes)
            entry["resolved_failures"] = sorted(b_codes - c_codes)

            # Failure/warning count deltas
            b_fc = b.get("failure_count", 0) or 0
            c_fc = c.get("failure_count", 0) or 0
            b_wc = b.get("warning_count", 0) or 0
            c_wc = c.get("warning_count", 0) or 0
            entry["failure_count_delta"] = c_fc - b_fc
            entry["warning_count_delta"] = c_wc - b_wc

            # Duration delta
            b_dur = b.get("execution_time_seconds")
            c_dur = c.get("execution_time_seconds")
            if isinstance(b_dur, (int, float)) and isinstance(c_dur, (int, float)):
                entry["duration_delta_seconds"] = round(c_dur - b_dur, 3)
        else:
            entry["change"] = "missing_both"

        results.append(entry)
    return results


def compute_comparison_summary(comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from gate comparisons."""
    change_counts: Dict[str, int] = {}
    for c in comparisons:
        ch = c.get("change", "unknown")
        change_counts[ch] = change_counts.get(ch, 0) + 1

    total_new_failures = sum(len(c.get("new_failures", [])) for c in comparisons)
    total_resolved = sum(len(c.get("resolved_failures", [])) for c in comparisons)
    improved = [c["gate_name"] for c in comparisons if c.get("change") == "improved"]
    regressed = [c["gate_name"] for c in comparisons if c.get("change") == "regressed"]

    return {
        "total_gates_compared": len(comparisons),
        "change_counts": change_counts,
        "improved_gates": improved,
        "regressed_gates": regressed,
        "total_new_failure_codes": total_new_failures,
        "total_resolved_failure_codes": total_resolved,
    }


# ── Output formatting ───────────────────────────────────────────────────────

def to_json(comparisons: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    return json.dumps({"summary": summary, "comparisons": comparisons}, indent=2, ensure_ascii=False)


def to_text(comparisons: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== Evidence Comparison ===")
    lines.append(f"Gates compared: {summary['total_gates_compared']}")
    lines.append(f"Changes: {', '.join(f'{k}={v}' for k, v in sorted(summary['change_counts'].items()))}")
    lines.append(f"New failure codes: {summary['total_new_failure_codes']}")
    lines.append(f"Resolved failure codes: {summary['total_resolved_failure_codes']}")
    lines.append("")

    if summary["improved_gates"]:
        lines.append("--- Improved Gates ---")
        for g in summary["improved_gates"]:
            lines.append(f"  + {g}")
        lines.append("")

    if summary["regressed_gates"]:
        lines.append("--- Regressed Gates ---")
        for g in summary["regressed_gates"]:
            lines.append(f"  - {g}")
        lines.append("")

    lines.append("--- Gate Details ---")
    for c in comparisons:
        change = c.get("change", "?")
        b_s = c.get("baseline_status") or "-"
        c_s = c.get("current_status") or "-"
        detail = f"  {c['gate_name']}: {b_s} → {c_s} [{change}]"

        extras = []
        if c.get("new_failures"):
            extras.append(f"new={','.join(c['new_failures'])}")
        if c.get("resolved_failures"):
            extras.append(f"resolved={','.join(c['resolved_failures'])}")
        fd = c.get("failure_count_delta")
        if fd is not None and fd != 0:
            extras.append(f"failures{'+'if fd>0 else ''}{fd}")
        if extras:
            detail += f" ({'; '.join(extras)})"
        lines.append(detail)
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two evidence directories and show gate-level diffs.",
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline evidence directory.")
    parser.add_argument("--current", required=True, help="Path to current evidence directory.")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output JSON.")
    parser.add_argument("--output", help="Write output to file (default: stdout).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline).expanduser().resolve()
    current_dir = Path(args.current).expanduser().resolve()

    for d, name in [(baseline_dir, "baseline"), (current_dir, "current")]:
        if not d.is_dir():
            print(f"Error: {name} directory not found: {d}", file=sys.stderr)
            return 1

    baseline_reports = scan_reports(baseline_dir)
    current_reports = scan_reports(current_dir)

    if not baseline_reports and not current_reports:
        print("Warning: no gate reports found in either directory.", file=sys.stderr)
        return 1

    comparisons = compare_gates(baseline_reports, current_reports)
    summary = compute_comparison_summary(comparisons)

    if args.json_output:
        output = to_json(comparisons, summary)
    else:
        output = to_text(comparisons, summary)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Comparison written to: {out_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
