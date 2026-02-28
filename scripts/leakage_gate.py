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

from _gate_utils import add_issue


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
    s = value.strip()
    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        pass

    iso = s.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(iso).timestamp()
    except ValueError:
        pass

    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


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
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")




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
        add_issue(failures, "io_error", "Failed to read CSV input.", {"error": str(exc)})
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
            if left_max > right_min:
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
    should_fail = bool(failures) or (args.strict and bool(warnings))

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
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
    }

    if args.report:
        from _gate_utils import write_json as _write_json
        _write_json(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")

    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
