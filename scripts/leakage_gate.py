#!/usr/bin/env python3
"""
Leakage gate for supervised prediction CSV splits.

Checks:
1. Row-level overlap across splits.
2. Entity ID overlap across splits.
3. Temporal ordering consistency (if time column is provided).
4. Suspicious feature names that often indicate target leakage.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from _gate_utils import add_issue, try_parse_time as _shared_try_parse_time, epoch_to_iso as _shared_epoch_to_iso
from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)


register_remediations({
    "io_error": "Verify the CSV file path exists and is readable. Check file encoding (expected UTF-8).",
    "column_mismatch": "Ensure all split CSV files have identical column headers. Regenerate splits from the same source.",
    "missing_target_column": "The target column specified by --target-col is missing from the split CSV. Check column names.",
    "suspicious_feature_names": "Features matching leakage patterns detected. Rename or remove columns that encode future/target information.",
    "row_overlap": "Identical rows found across splits. This indicates a split generation bug. Regenerate splits with proper deduplication.",
    "missing_id_columns": "ID columns specified by --id-cols are missing from the split CSV. Check column names.",
    "id_overlap": "Patient/entity IDs overlap between splits. Fix split generation to ensure strict ID separation.",
    "incomplete_id_rows": "Some rows have missing ID values. These were excluded from overlap checks. Verify data completeness.",
    "missing_time_column": "Time column specified by --time-col is missing. Check column names or omit --time-col.",
    "invalid_time_values": "Some time values couldn't be parsed. Check timestamp format consistency.",
    "no_parseable_time_values": "No valid timestamps found. Cannot perform temporal leakage checks.",
    "temporal_overlap": "Training data timestamps overlap with validation/test. Ensure strict temporal ordering in split boundaries.",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict anti-leakage checks on CSV splits.")
    parser.add_argument("--train", required=True, help="Path to training CSV.")
    parser.add_argument("--valid", help="Path to validation CSV.")
    parser.add_argument("--test", help="Path to test CSV.")
    parser.add_argument("--id-cols", default="", help="Comma-separated entity ID columns.")
    parser.add_argument("--time-col", help="Timestamp column for temporal leakage checks.")
    parser.add_argument("--target-col", help="Target column name.")
    parser.add_argument(
        "--ignore-cols",
        default="",
        help="Comma-separated columns to ignore in row-hash overlap checks.",
    )
    parser.add_argument(
        "--forbidden-feature-regex",
        default=r"\b(future|leak)\b|(?:^|_)(target|label|outcome)(?:_|$)",
        help="Regex for suspicious feature names. Default matches clearly leakage-indicative tokens (future, leak as whole words; target, label, outcome as underscore-delimited segments). Override with --forbidden-feature-regex for domain-specific patterns.",
    )
    parser.add_argument("--report", help="Optional path to write JSON report.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings in addition to hard failures.",
    )
    args = parser.parse_args()
    if not args.valid and not args.test:
        parser.error("Provide at least one of --valid or --test.")
    return args


def parse_csv(path: str, split_name: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name}: file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{split_name}: missing CSV header row.")

        headers = [h.strip() if h else "" for h in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw in reader:
            clean: Dict[str, str] = {}
            for k, v in raw.items():
                key = (k or "").strip()
                clean[key] = (v or "").strip()
            rows.append(clean)

    return {"path": path, "headers": headers, "rows": rows}


def parse_comma_set(raw: str) -> Set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def row_signature(row: Dict[str, str], ignore_cols: Set[str]) -> str:
    parts = []
    for col in sorted(row.keys()):
        if col in ignore_cols:
            continue
        parts.append(f"{col}={row.get(col, '')}")
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def try_parse_time(value: str) -> Optional[float]:
    return _shared_try_parse_time(value)


def bounds_for_time(rows: Iterable[Dict[str, str]], time_col: str) -> Dict[str, Any]:
    parsed: List[float] = []
    invalid = 0
    missing = 0
    for row in rows:
        raw = row.get(time_col, "").strip()
        if not raw:
            missing += 1
            continue
        ts = try_parse_time(raw)
        if ts is None:
            invalid += 1
            continue
        parsed.append(ts)

    if not parsed:
        return {
            "count": 0,
            "missing": missing,
            "invalid": invalid,
            "min": None,
            "max": None,
        }
    return {
        "count": len(parsed),
        "missing": missing,
        "invalid": invalid,
        "min": min(parsed),
        "max": max(parsed),
    }


def epoch_to_iso(ts: Optional[float]) -> Optional[str]:
    return _shared_epoch_to_iso(ts)




def main() -> int:
    args = parse_args()

    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    split_paths = [("train", args.train), ("valid", args.valid), ("test", args.test)]
    splits: Dict[str, Dict[str, Any]] = {}
    try:
        for name, path in split_paths:
            if path:
                splits[name] = parse_csv(path, name)
    except Exception as exc:
        add_issue(failures, "io_error", f"Failed to read CSV input for '{name}' split.", {"error": str(exc), "path": str(path)})
        return finish(args, splits, failures, warnings)

    ignore_cols = parse_comma_set(args.ignore_cols)
    id_cols = [c for c in (x.strip() for x in args.id_cols.split(",")) if c]
    feature_name_re = re.compile(args.forbidden_feature_regex, flags=re.IGNORECASE)

    # Column consistency.
    column_sets = {k: set(v["headers"]) for k, v in splits.items()}
    union_cols = set().union(*column_sets.values()) if column_sets else set()
    intersection_cols = set.intersection(*column_sets.values()) if column_sets else set()
    if union_cols != intersection_cols:
        add_issue(
            warnings,
            "column_mismatch",
            "Split files have non-identical column sets.",
            {
                "union_count": len(union_cols),
                "intersection_count": len(intersection_cols),
            },
        )

    if args.target_col:
        for split_name, split in splits.items():
            if args.target_col not in split["headers"]:
                add_issue(
                    failures,
                    "missing_target_column",
                    "Target column missing in split.",
                    {"split": split_name, "target_col": args.target_col},
                )

    # Suspicious headers.
    canonical_headers = splits["train"]["headers"] if "train" in splits else []
    suspicious = []
    for h in canonical_headers:
        if args.target_col and h == args.target_col:
            continue
        if feature_name_re.search(h):
            suspicious.append(h)
    if suspicious:
        add_issue(
            warnings,
            "suspicious_feature_names",
            "Feature names match leakage-prone patterns.",
            {"columns": suspicious},
        )

    # Row overlap.
    signature_sets: Dict[str, Set[str]] = {}
    for split_name, split in splits.items():
        signature_sets[split_name] = {
            row_signature(row, ignore_cols)
            for row in split["rows"]
        }

    for a, b in itertools.combinations(signature_sets.keys(), 2):
        overlap = signature_sets[a] & signature_sets[b]
        if overlap:
            add_issue(
                failures,
                "row_overlap",
                "Identical rows detected across splits.",
                {"pair": [a, b], "overlap_count": len(overlap)},
            )

    # Entity overlap.
    if id_cols:
        for split_name, split in splits.items():
            missing_cols = [c for c in id_cols if c not in split["headers"]]
            if missing_cols:
                add_issue(
                    failures,
                    "missing_id_columns",
                    "ID columns missing in split.",
                    {"split": split_name, "missing": missing_cols},
                )

        id_sets: Dict[str, Set[Tuple[str, ...]]] = {}
        for split_name, split in splits.items():
            keys: Set[Tuple[str, ...]] = set()
            null_key_rows = 0
            for row in split["rows"]:
                key: List[str] = []
                incomplete = False
                for col in id_cols:
                    val = row.get(col, "").strip()
                    if not val:
                        incomplete = True
                        break
                    key.append(val)
                if incomplete:
                    null_key_rows += 1
                    continue
                keys.add(tuple(key))
            id_sets[split_name] = keys
            if null_key_rows:
                add_issue(
                    warnings,
                    "incomplete_id_rows",
                    "Rows with missing ID columns were skipped in ID-overlap check.",
                    {"split": split_name, "skipped_rows": null_key_rows},
                )

        for a, b in itertools.combinations(id_sets.keys(), 2):
            overlap = id_sets[a] & id_sets[b]
            if overlap:
                add_issue(
                    failures,
                    "id_overlap",
                    "Entity IDs overlap across splits.",
                    {"pair": [a, b], "overlap_count": len(overlap), "id_cols": id_cols},
                )

    # Temporal ordering.
    time_bounds: Dict[str, Dict[str, Any]] = {}
    if args.time_col:
        for split_name, split in splits.items():
            if args.time_col not in split["headers"]:
                add_issue(
                    failures,
                    "missing_time_column",
                    "Time column missing in split.",
                    {"split": split_name, "time_col": args.time_col},
                )
                continue

            b = bounds_for_time(split["rows"], args.time_col)
            time_bounds[split_name] = b
            if b["invalid"] > 0:
                add_issue(
                    warnings,
                    "invalid_time_values",
                    "Some time values could not be parsed.",
                    {"split": split_name, "invalid_count": b["invalid"]},
                )
            if b["count"] == 0:
                add_issue(
                    failures,
                    "no_parseable_time_values",
                    "No parseable time values found for temporal checks.",
                    {"split": split_name},
                )

        def check_order(left: str, right: str) -> None:
            if left not in time_bounds or right not in time_bounds:
                return
            left_max = time_bounds[left]["max"]
            right_min = time_bounds[right]["min"]
            if left_max is None or right_min is None:
                return
            if left_max >= right_min:
                add_issue(
                    failures,
                    "temporal_overlap",
                    "Temporal boundary violation detected.",
                    {
                        "left_split": left,
                        "right_split": right,
                        "left_max": epoch_to_iso(left_max),
                        "right_min": epoch_to_iso(right_min),
                    },
                )

        check_order("train", "valid")
        check_order("train", "test")
        check_order("valid", "test")

    return finish(args, splits, failures, warnings, time_bounds=time_bounds)


def finish(
    args: argparse.Namespace,
    splits: Dict[str, Dict[str, Any]],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    time_bounds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> int:
    from _gate_utils import get_gate_elapsed, write_json as _write_json

    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"

    fi = [GateIssue.from_legacy(f, Severity.ERROR) for f in failures]
    wi = [GateIssue.from_legacy(w, Severity.WARNING) for w in warnings]
    for issue in fi + wi:
        if not issue.remediation:
            issue.remediation = get_remediation(issue.code)

    summary = {
        "rows_per_split": {k: len(v["rows"]) for k, v in splits.items()},
        "columns_per_split": {k: len(v["headers"]) for k, v in splits.items()},
        "time_bounds": {
            k: {
                "min": epoch_to_iso(v.get("min")),
                "max": epoch_to_iso(v.get("max")),
                "parsed_count": v.get("count"),
                "missing_count": v.get("missing"),
                "invalid_count": v.get("invalid"),
            }
            for k, v in (time_bounds or {}).items()
        },
    }

    input_files = {"train": str(Path(args.train).expanduser().resolve())}
    if args.valid:
        input_files["valid"] = str(Path(args.valid).expanduser().resolve())
    if args.test:
        input_files["test"] = str(Path(args.test).expanduser().resolve())

    report = build_report_envelope(
        gate_name="leakage_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary=summary,
        input_files=input_files,
    )

    if args.report:
        _write_json(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="leakage_gate",
        status=status,
        failures=fi,
        warnings=wi,
        strict=bool(args.strict),
        elapsed=get_gate_elapsed(),
    )

    return 2 if should_fail else 0


if __name__ == "__main__":
    from _gate_utils import start_gate_timer
    start_gate_timer()
    raise SystemExit(main())
