#!/usr/bin/env python3
"""
Dataset schema preflight with auto-mapping suggestions for train/valid/test splits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from _gate_utils import add_issue, write_json


TARGET_ALIASES = ["y", "label", "target", "outcome", "class", "readmitted", "event"]
PATIENT_ID_ALIASES = ["patient_id", "patientid", "subject_id", "person_id", "id"]
TIME_ALIASES = ["event_time", "index_time", "timestamp", "admit_time", "encounter_time", "date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight schema checks for medical train/valid/test splits or a single CSV file.")
    parser.add_argument("--train", default="", help="Path to train CSV.")
    parser.add_argument("--valid", default="", help="Path to valid CSV.")
    parser.add_argument("--test", default="", help="Path to test CSV.")
    parser.add_argument("--input-csv", default="", help="Path to a single complete CSV for pre-split quality checks.")
    parser.add_argument("--target-col", default="y", help="Preferred target column name.")
    parser.add_argument("--patient-id-col", default="patient_id", help="Preferred patient ID column name.")
    parser.add_argument("--time-col", default="event_time", help="Preferred index time column name.")
    parser.add_argument("--mapping-out", help="Optional output JSON path for resolved field mapping.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail when required columns need auto-mapping.")
    args = parser.parse_args()
    if not args.input_csv and not args.train:
        parser.error("Provide either --input-csv (single file mode) or --train/--valid/--test (split mode).")
    return args


def normalize_col(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")


def load_split(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Empty split CSV: {p}")
    return df


def resolve_column(
    preferred: str,
    aliases: Sequence[str],
    common_columns: Sequence[str],
) -> Tuple[Optional[str], str, List[str]]:
    common = list(common_columns)
    if preferred in common:
        return preferred, "exact", []
    norm_to_real: Dict[str, str] = {normalize_col(col): col for col in common}
    preferred_norm = normalize_col(preferred)
    if preferred_norm in norm_to_real:
        return norm_to_real[preferred_norm], "normalized", []
    candidates: List[str] = []
    for alias in aliases:
        alias_norm = normalize_col(alias)
        if alias_norm in norm_to_real:
            candidates.append(norm_to_real[alias_norm])
    # Keep order stable and unique.
    deduped = list(dict.fromkeys(candidates))
    if deduped:
        return deduped[0], "alias", deduped[1:]
    return None, "missing", []


def parse_binary_target(series: pd.Series) -> Tuple[Optional[np.ndarray], Optional[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = sorted(set(float(x) for x in numeric.tolist()))
        if values and all(v in {0.0, 1.0} for v in values):
            return numeric.to_numpy(dtype=int), None
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "negative": 0,
        "positive": 1,
    }
    mapped = lowered.map(mapping)
    if mapped.notna().all():
        return mapped.to_numpy(dtype=int), None
    example_values = sorted(set(lowered.dropna().tolist()))[:8]
    return None, f"non-binary values detected: {example_values}"


def missing_ratio(df: pd.DataFrame) -> float:
    total = float(df.shape[0] * max(1, df.shape[1]))
    if total <= 0:
        return 0.0
    return float(df.isna().sum().sum() / total)


def split_summary(df: pd.DataFrame, target_col: str, id_col: str, time_col: str) -> Dict[str, Any]:
    target, err = parse_binary_target(df[target_col])
    parsed_time = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    null_patient = int(df[id_col].isna().sum())
    unique_patient = int(df[id_col].nunique(dropna=True))
    stats = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_ratio": float(missing_ratio(df)),
        "patient_id_null_count": null_patient,
        "patient_id_unique_count": unique_patient,
        "time_parse_error_count": int(parsed_time.isna().sum()),
        "target_parse_error": err,
    }
    if target is not None and target.size > 0:
        stats["positive_count"] = int(np.sum(target == 1))
        stats["negative_count"] = int(np.sum(target == 0))
        stats["positive_rate"] = float(np.mean(target))
    else:
        stats["positive_count"] = None
        stats["negative_count"] = None
        stats["positive_rate"] = None
    return stats


def run_single_file_preflight(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> int:
    """Pre-split quality checks on a single CSV file."""
    csv_path = str(args.input_csv)
    try:
        df = load_split(csv_path)
    except Exception as exc:
        add_issue(failures, "input_csv_read_failed", f"Unable to read input CSV.", {"error": str(exc), "path": csv_path})
        return finish(args, failures, warnings, {}, None)

    columns = list(df.columns)
    print(f"[INFO] Input CSV: {csv_path} ({len(df)} rows, {len(columns)} columns)")

    resolved: Dict[str, Any] = {}
    for semantic_name, preferred, aliases in (
        ("target_col", args.target_col, TARGET_ALIASES),
        ("patient_id_col", args.patient_id_col, PATIENT_ID_ALIASES),
        ("index_time_col", args.time_col, TIME_ALIASES),
    ):
        selected, mode, alternates = resolve_column(preferred=preferred, aliases=aliases, common_columns=columns)
        resolved[semantic_name] = selected
        resolved[f"{semantic_name}_resolution"] = mode
        resolved[f"{semantic_name}_alternates"] = alternates
        if selected is None:
            add_issue(
                failures, "required_column_missing",
                "Unable to resolve required semantic column.",
                {"semantic": semantic_name, "preferred": preferred, "aliases": list(aliases)},
            )
        elif mode in {"normalized", "alias"}:
            add_issue(
                warnings, "column_auto_mapped",
                "Semantic column was auto-mapped; review before production use.",
                {"semantic": semantic_name, "preferred": preferred, "resolved": selected, "mode": mode},
            )

    target_col = resolved.get("target_col")
    patient_id_col = resolved.get("patient_id_col")
    time_col = resolved.get("index_time_col")

    file_stats: Dict[str, Any] = {}
    if all(isinstance(x, str) and x for x in (target_col, patient_id_col, time_col)):
        file_stats = split_summary(df, str(target_col), str(patient_id_col), str(time_col))
        file_stats["patient_id_unique_count"] = int(df[str(patient_id_col)].nunique(dropna=True))

        if isinstance(file_stats.get("target_parse_error"), str):
            add_issue(failures, "target_not_binary", "Target column is not parseable to binary 0/1.",
                      {"column": target_col, "error": file_stats["target_parse_error"]})
        if int(file_stats.get("patient_id_null_count", 0)) > 0:
            add_issue(failures, "patient_id_nulls_detected", "Patient ID column contains null values.",
                      {"column": patient_id_col, "null_count": file_stats["patient_id_null_count"]})
        time_err = int(file_stats.get("time_parse_error_count", 0))
        if time_err > 0:
            add_issue(warnings, "index_time_parse_issues", "Some index time values could not be parsed.",
                      {"column": time_col, "invalid_count": time_err})
        pos_rate = file_stats.get("positive_rate")
        if isinstance(pos_rate, (int, float)) and math.isfinite(float(pos_rate)):
            if float(pos_rate) <= 0.0 or float(pos_rate) >= 1.0:
                add_issue(warnings, "target_single_class", "Dataset contains only one class.",
                          {"positive_rate": float(pos_rate)})

        n_patients = file_stats.get("patient_id_unique_count", 0)
        if isinstance(n_patients, int) and n_patients < 6:
            add_issue(failures, "insufficient_patients",
                      f"Need at least 6 unique patients for 3-way split, found {n_patients}.",
                      {"patient_count": n_patients})

    if args.strict:
        for issue in warnings:
            if issue["code"] == "column_auto_mapped":
                add_issue(failures, "strict_auto_mapping_not_allowed",
                          "Strict mode requires explicit column names.", issue["details"])

    mapping_payload = None
    if not failures:
        mapping_payload = {
            "target_col": target_col,
            "patient_id_col": patient_id_col,
            "index_time_col": time_col,
            "label_col": target_col,
            "notes": "Generated by schema_preflight.py (single-file mode).",
        }

    summary = {
        "input_csv": str(Path(csv_path).expanduser().resolve()),
        "mode": "single_file",
        "column_count": len(columns),
        "resolved_mapping": resolved,
        "file_stats": file_stats,
        "suggested_request_patch": (
            {"label_col": target_col, "patient_id_col": patient_id_col, "index_time_col": time_col}
            if mapping_payload else None
        ),
    }
    return finish(args, failures, warnings, summary, mapping_payload)


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    # Single-file mode: pre-split quality checks
    if args.input_csv:
        return run_single_file_preflight(args, failures, warnings)

    split_dfs: Dict[str, Any] = {}
    split_items = [("train", args.train), ("valid", args.valid), ("test", args.test)]
    split_items = [(n, p) for n, p in split_items if p]
    if not split_items:
        add_issue(failures, "no_splits", "No split paths provided.", {})
        return finish(args, failures, warnings, {}, None)
    try:
        for _split_name, _split_path in split_items:
            split_dfs[_split_name] = load_split(_split_path)
    except Exception as exc:
        add_issue(
            failures,
            "split_read_failed",
            f"Unable to read '{_split_name}' split CSV.",
            {"error": str(exc), "path": str(_split_path)},
        )
        return finish(args, failures, warnings, {}, None)

    train_df = split_dfs.get("train")
    valid_df = split_dfs.get("valid")
    test_df = split_dfs.get("test")
    all_dfs = [df for df in [train_df, valid_df, test_df] if df is not None]
    if not all_dfs:
        add_issue(failures, "no_splits_loaded", "No split DataFrames loaded.", {})
        return finish(args, failures, warnings, {}, None)
    common_columns = sorted(set.intersection(*[set(df.columns) for df in all_dfs]))
    if not common_columns:
        add_issue(
            failures,
            "no_common_columns",
            "train/valid/test do not share any common columns.",
            {},
        )
        return finish(args, failures, warnings, {}, None)

    resolved: Dict[str, Any] = {}
    for semantic_name, preferred, aliases in (
        ("target_col", args.target_col, TARGET_ALIASES),
        ("patient_id_col", args.patient_id_col, PATIENT_ID_ALIASES),
        ("index_time_col", args.time_col, TIME_ALIASES),
    ):
        selected, mode, alternates = resolve_column(preferred=preferred, aliases=aliases, common_columns=common_columns)
        resolved[semantic_name] = selected
        resolved[f"{semantic_name}_resolution"] = mode
        resolved[f"{semantic_name}_alternates"] = alternates
        if selected is None:
            add_issue(
                failures,
                "required_column_missing",
                "Unable to resolve required semantic column across all splits.",
                {"semantic": semantic_name, "preferred": preferred, "aliases": list(aliases)},
            )
        elif mode in {"normalized", "alias"}:
            add_issue(
                warnings,
                "column_auto_mapped",
                "Semantic column was auto-mapped; review before production use.",
                {"semantic": semantic_name, "preferred": preferred, "resolved": selected, "mode": mode},
            )

    target_col = resolved.get("target_col")
    patient_id_col = resolved.get("patient_id_col")
    time_col = resolved.get("index_time_col")

    split_stats: Dict[str, Any] = {}
    if all(isinstance(x, str) and x for x in (target_col, patient_id_col, time_col)):
        for split_name, df in (("train", train_df), ("valid", valid_df), ("test", test_df)):
            if df is None:
                continue
            stats = split_summary(df, str(target_col), str(patient_id_col), str(time_col))
            split_stats[split_name] = stats
            if isinstance(stats.get("target_parse_error"), str):
                add_issue(
                    failures,
                    "target_not_binary",
                    "Target column is not parseable to binary 0/1.",
                    {"split": split_name, "column": target_col, "error": stats["target_parse_error"]},
                )
            if int(stats.get("patient_id_null_count", 0)) > 0:
                add_issue(
                    failures,
                    "patient_id_nulls_detected",
                    "Patient ID column contains null values.",
                    {"split": split_name, "column": patient_id_col, "null_count": stats["patient_id_null_count"]},
                )
            time_err = int(stats.get("time_parse_error_count", 0))
            if time_err > 0:
                add_issue(
                    failures,
                    "index_time_parse_failed",
                    "Index time column contains non-parseable timestamps.",
                    {"split": split_name, "column": time_col, "invalid_count": time_err},
                )
            pos_rate = stats.get("positive_rate")
            if isinstance(pos_rate, (int, float)) and math.isfinite(float(pos_rate)):
                if float(pos_rate) <= 0.0 or float(pos_rate) >= 1.0:
                    add_issue(
                        warnings,
                        "target_single_class_split",
                        "Split appears to contain only one class; strict training gates may fail.",
                        {"split": split_name, "positive_rate": float(pos_rate)},
                    )

    if args.strict:
        for issue in warnings:
            if issue["code"] == "column_auto_mapped":
                add_issue(
                    failures,
                    "strict_auto_mapping_not_allowed",
                    "Strict mode requires explicit semantic column names without auto-mapping.",
                    issue["details"],
                )

    mapping_payload = None
    if not failures:
        mapping_payload = {
            "target_col": target_col,
            "patient_id_col": patient_id_col,
            "index_time_col": time_col,
            "label_col": target_col,
            "notes": "Generated by schema_preflight.py.",
        }

    summary = {
        "split_paths": {
            name: str(Path(path).expanduser().resolve())
            for name, path in (("train", args.train), ("valid", args.valid), ("test", args.test))
            if path
        },
        "common_column_count": int(len(common_columns)),
        "resolved_mapping": resolved,
        "split_stats": split_stats,
        "suggested_request_patch": (
            {
                "label_col": target_col,
                "patient_id_col": patient_id_col,
                "index_time_col": time_col,
            }
            if mapping_payload is not None
            else None
        ),
    }
    return finish(args, failures, warnings, summary, mapping_payload)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
    mapping_payload: Optional[Dict[str, Any]],
) -> int:
    should_fail = bool(failures)
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": int(len(failures)),
        "warning_count": int(len(warnings)),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
    }

    if args.mapping_out and mapping_payload is not None:
        write_json(Path(args.mapping_out).expanduser().resolve(), mapping_payload)

    if args.report:
        write_json(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"Failures: {report['failure_count']} | Warnings: {report['warning_count']} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
