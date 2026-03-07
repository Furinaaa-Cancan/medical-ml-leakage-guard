#!/usr/bin/env python3
"""
Scan stress-case diabetes configurations and summarize publication-grade feasibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_SCRIPT = REPO_ROOT / "experiments" / "authority-e2e" / "run_authority_e2e.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan stress diabetes feasibility across target modes and row caps.")
    parser.add_argument(
        "--target-modes",
        default="gt30,any,lt30",
        help="Comma-separated diabetes target modes to test.",
    )
    parser.add_argument(
        "--max-rows-options",
        default="20000,0",
        help="Comma-separated diabetes max_rows options (0 means full dataset).",
    )
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "experiments" / "authority-e2e" / "_feasibility_scan"),
        help="Directory for per-run authority summaries.",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "experiments" / "authority-e2e" / "stress_diabetes_feasibility_report.json"),
        help="Output feasibility report path.",
    )
    parser.add_argument(
        "--run-tag-prefix",
        default="feasibility-scan",
        help="Prefix for authority run-tag values.",
    )
    return parser.parse_args()


def parse_csv_tokens(raw: str) -> List[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_int_tokens(raw: str) -> List[int]:
    values: List[int] = []
    for token in parse_csv_tokens(raw):
        try:
            values.append(int(token))
        except Exception as exc:
            raise ValueError(f"Invalid integer token: {token}") from exc
    return values


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def run_case(
    mode: str,
    max_rows: int,
    summary_dir: Path,
    run_tag_prefix: str,
) -> Dict[str, Any]:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_tag = f"{run_tag_prefix}-{mode}-rows{max_rows}-{stamp}"
    summary_path = summary_dir / f"authority_summary.{mode}.rows{max_rows}.{stamp}.json"
    cache_path = summary_dir / f"stress_seed_search.{mode}.rows{max_rows}.{stamp}.json"
    selection_path = summary_dir / f"stress_seed_selection.{mode}.rows{max_rows}.{stamp}.json"

    cmd = [
        sys.executable,
        str(E2E_SCRIPT),
        "--include-stress-cases",
        "--stress-case-id",
        "uci-diabetes-130-readmission",
        "--diabetes-target-mode",
        mode,
        "--diabetes-max-rows",
        str(max_rows),
        "--summary-file",
        str(summary_path),
        "--stress-seed-cache-file",
        str(cache_path),
        "--stress-selection-file",
        str(selection_path),
        "--run-tag",
        run_tag,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    row: Dict[str, Any] = {}
    if summary_path.exists():
        summary = read_json(summary_path)
        for item in summary.get("results", []):
            if isinstance(item, dict) and str(item.get("case_id")) == "uci-diabetes-130-readmission":
                row = item
                break

    clinical_gap = row.get("clinical_floor_gap_summary") if isinstance(row, dict) else None
    min_margin = None
    if isinstance(clinical_gap, dict):
        raw_min_margin = clinical_gap.get("minimum_margin")
        if isinstance(raw_min_margin, (int, float)):
            min_margin = float(raw_min_margin)

    return {
        "target_mode": mode,
        "max_rows": int(max_rows),
        "run_tag": run_tag,
        "summary_file": str(summary_path),
        "return_code": int(proc.returncode),
        "status": row.get("status") if isinstance(row, dict) else None,
        "failure_code": row.get("failure_code") if isinstance(row, dict) else None,
        "root_failure_code_primary": row.get("root_failure_code_primary") if isinstance(row, dict) else None,
        "root_failure_codes": row.get("root_failure_codes") if isinstance(row, dict) else None,
        "minimum_floor_margin": min_margin,
        "metrics": row.get("metrics") if isinstance(row, dict) else None,
        "stderr_tail": proc.stderr[-1200:],
        "stdout_tail": proc.stdout[-1200:],
    }


def main() -> int:
    args = parse_args()
    target_modes = parse_csv_tokens(args.target_modes)
    if not target_modes:
        raise ValueError("target-modes must include at least one token.")
    max_rows_options = parse_int_tokens(args.max_rows_options)
    if not max_rows_options:
        raise ValueError("max-rows-options must include at least one integer.")

    summary_dir = Path(args.summary_dir).expanduser().resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for mode in target_modes:
        for max_rows in max_rows_options:
            rows.append(
                run_case(
                    mode=mode,
                    max_rows=max_rows,
                    summary_dir=summary_dir,
                    run_tag_prefix=str(args.run_tag_prefix).strip() or "feasibility-scan",
                )
            )

    passing = [row for row in rows if str(row.get("status")) == "pass"]
    best = None
    if rows:
        best = max(
            rows,
            key=lambda item: (
                1 if str(item.get("status")) == "pass" else 0,
                float(item.get("minimum_floor_margin")) if isinstance(item.get("minimum_floor_margin"), (int, float)) else -1e9,
            ),
        )

    report_payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO_ROOT),
        "target_modes": target_modes,
        "max_rows_options": max_rows_options,
        "scenario_count": len(rows),
        "passing_count": len(passing),
        "best_candidate": best,
        "results": rows,
        "overall_status": "pass" if passing else "fail",
        "failure_code": None if passing else "stress_case_clinical_feasibility_not_found",
    }

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report_payload, fh, ensure_ascii=True, indent=2)

    print(f"Feasibility report: {report_path}")
    print(json.dumps(report_payload, ensure_ascii=True, indent=2))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())

