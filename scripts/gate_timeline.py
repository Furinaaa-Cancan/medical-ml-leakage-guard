#!/usr/bin/env python3
"""
Gate Timeline Analyzer for ML Leakage Guard.

Reads gate reports from an evidence directory, extracts execution
timestamps and durations, identifies bottleneck gates, and produces
a timeline summary showing pipeline execution flow.

Usage:
    python3 scripts/gate_timeline.py --evidence-dir evidence/
    python3 scripts/gate_timeline.py --evidence-dir evidence/ --json
    python3 scripts/gate_timeline.py --evidence-dir evidence/ --output timeline.json --json
    python3 scripts/gate_timeline.py --evidence-dir evidence/ --top 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


# ── Gate entry extraction ───────────────────────────────────────────────────

def _parse_utc(ts: Any) -> Optional[datetime]:
    """Parse an ISO-8601 UTC timestamp string."""
    if not isinstance(ts, str):
        return None
    try:
        cleaned = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float if it is a finite number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        import math
        return float(value) if math.isfinite(value) else None
    return None


def extract_gate_entry(path: Path) -> Optional[Dict[str, Any]]:
    """Extract timeline-relevant data from a single gate report."""
    data = load_json(path)
    if data is None:
        return None

    gate_name = data.get("gate_name")
    if not isinstance(gate_name, str) or not gate_name:
        # Try to infer from filename
        stem = path.stem
        if stem.endswith("_report"):
            gate_name = stem[:-7]  # strip _report
        else:
            gate_name = stem

    timestamp = _parse_utc(data.get("execution_timestamp_utc"))
    duration = _safe_float(data.get("execution_time_seconds"))
    status = data.get("status", "unknown")
    failure_count = data.get("failure_count", 0)
    warning_count = data.get("warning_count", 0)

    return {
        "gate_name": gate_name,
        "file": path.name,
        "status": str(status),
        "timestamp_utc": timestamp.isoformat() if timestamp else None,
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "failure_count": int(failure_count) if isinstance(failure_count, (int, float)) else 0,
        "warning_count": int(warning_count) if isinstance(warning_count, (int, float)) else 0,
        "_timestamp_obj": timestamp,
    }


def scan_evidence_dir(evidence_dir: Path) -> List[Dict[str, Any]]:
    """Scan evidence directory for gate reports and extract timeline entries."""
    entries: List[Dict[str, Any]] = []
    if not evidence_dir.is_dir():
        return entries
    for p in sorted(evidence_dir.glob("*_report.json")):
        entry = extract_gate_entry(p)
        if entry is not None:
            entries.append(entry)
    return entries


# ── Analysis ────────────────────────────────────────────────────────────────

def sort_by_timestamp(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort entries by timestamp, putting None-timestamp entries at the end."""
    def _key(e: Dict[str, Any]) -> Tuple[int, str]:
        ts = e.get("_timestamp_obj")
        if ts is None:
            return (1, e["gate_name"])
        return (0, ts.isoformat())
    return sorted(entries, key=_key)


def compute_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute pipeline-level summary statistics."""
    total_gates = len(entries)
    durations = [e["duration_seconds"] for e in entries if e["duration_seconds"] is not None]
    timestamps = [e["_timestamp_obj"] for e in entries if e["_timestamp_obj"] is not None]

    statuses = {}
    for e in entries:
        s = e["status"]
        statuses[s] = statuses.get(s, 0) + 1

    total_duration = sum(durations) if durations else 0.0
    avg_duration = total_duration / len(durations) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    min_duration = min(durations) if durations else 0.0

    # Wall-clock span (first to last timestamp)
    wall_clock_span = None
    if len(timestamps) >= 2:
        earliest = min(timestamps)
        latest = max(timestamps)
        wall_clock_span = round((latest - earliest).total_seconds(), 3)

    return {
        "total_gates": total_gates,
        "status_counts": statuses,
        "total_duration_seconds": round(total_duration, 3),
        "average_duration_seconds": round(avg_duration, 3),
        "max_duration_seconds": round(max_duration, 3),
        "min_duration_seconds": round(min_duration, 3),
        "wall_clock_span_seconds": wall_clock_span,
        "gates_with_timestamp": len(timestamps),
        "gates_with_duration": len(durations),
    }


def find_bottlenecks(entries: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """Find the top-N slowest gates by duration."""
    with_duration = [e for e in entries if e["duration_seconds"] is not None]
    ranked = sorted(with_duration, key=lambda e: e["duration_seconds"], reverse=True)
    result = []
    for e in ranked[:top_n]:
        result.append({
            "gate_name": e["gate_name"],
            "duration_seconds": e["duration_seconds"],
            "status": e["status"],
        })
    return result


# ── Output formatting ───────────────────────────────────────────────────────

def _clean_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal keys from an entry for output."""
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def to_json(entries: List[Dict[str, Any]], summary: Dict[str, Any],
            bottlenecks: List[Dict[str, Any]]) -> str:
    """Format as JSON output."""
    output = {
        "summary": summary,
        "bottlenecks": bottlenecks,
        "timeline": [_clean_entry(e) for e in entries],
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def to_text(entries: List[Dict[str, Any]], summary: Dict[str, Any],
            bottlenecks: List[Dict[str, Any]]) -> str:
    """Format as human-readable text output."""
    lines: List[str] = []

    lines.append("=== Gate Timeline Analysis ===")
    lines.append(f"Total gates: {summary['total_gates']}")
    lines.append(f"Status: {', '.join(f'{k}={v}' for k, v in sorted(summary['status_counts'].items()))}")
    lines.append(f"Total duration: {summary['total_duration_seconds']:.3f}s")
    lines.append(f"Average duration: {summary['average_duration_seconds']:.3f}s")
    if summary["wall_clock_span_seconds"] is not None:
        lines.append(f"Wall-clock span: {summary['wall_clock_span_seconds']:.3f}s")
    lines.append("")

    if bottlenecks:
        lines.append("--- Bottleneck Gates (slowest) ---")
        for i, b in enumerate(bottlenecks, 1):
            lines.append(f"  {i}. {b['gate_name']}: {b['duration_seconds']:.3f}s [{b['status']}]")
        lines.append("")

    lines.append("--- Timeline ---")
    for e in entries:
        ts = e.get("timestamp_utc") or "no-timestamp"
        dur = f"{e['duration_seconds']:.3f}s" if e["duration_seconds"] is not None else "N/A"
        status = e["status"]
        name = e["gate_name"]
        lines.append(f"  [{ts}] {name}: {dur} [{status}]")
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze gate execution timeline from an evidence directory.",
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to the evidence directory containing gate reports.",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--output", help="Write output to file (default: stdout).",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of bottleneck gates to show (default: 5).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()

    if not evidence_dir.is_dir():
        print(f"Error: evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 1

    entries = scan_evidence_dir(evidence_dir)
    if not entries:
        print(f"Warning: no gate reports found in {evidence_dir}", file=sys.stderr)
        return 1

    sorted_entries = sort_by_timestamp(entries)
    summary = compute_summary(sorted_entries)
    bottlenecks = find_bottlenecks(sorted_entries, top_n=args.top)

    if args.json_output:
        output = to_json(sorted_entries, summary, bottlenecks)
    else:
        output = to_text(sorted_entries, summary, bottlenecks)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Timeline written to: {out_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
