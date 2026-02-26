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


TARGET_ALIASES = ["y", "label", "target", "outcome", "class", "readmitted", "event"]
PATIENT_ID_ALIASES = ["patient_id", "patientid", "subject_id", "person_id", "id"]
TIME_ALIASES = ["event_time", "index_time", "timestamp", "admit_time", "encounter_time", "date"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight schema checks for medical train/valid/test splits.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", required=True, help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--target-col", default="y", help="Preferred target column name.")
    parser.add_argument("--patient-id-col", default="patient_id", help="Preferred patient ID column name.")
    parser.add_argument("--time-col", default="event_time", help="Preferred index time column name.")
    parser.add_argument("--mapping-out", help="Optional output JSON path for resolved field mapping.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail when required columns need auto-mapping.")
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


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


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    try:
        train_df = load_split(args.train)
        valid_df = load_split(args.valid)
        test_df = load_split(args.test)
    except Exception as exc:
        add_issue(
            failures,
            "split_read_failed",
            "Unable to read split CSV files.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, {}, None)

    common_columns = sorted(set(train_df.columns) & set(valid_df.columns) & set(test_df.columns))
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
            "train": str(Path(args.train).expanduser().resolve()),
            "valid": str(Path(args.valid).expanduser().resolve()),
            "test": str(Path(args.test).expanduser().resolve()),
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
