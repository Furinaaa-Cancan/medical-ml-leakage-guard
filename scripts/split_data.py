#!/usr/bin/env python3
"""
Medical-safe data splitting tool for ml-leakage-guard.

Splits a single CSV into train/valid/test with:
- Patient-level disjoint splits (no patient appears in multiple splits)
- Temporal ordering enforcement (train < valid < test when applicable)
- Prevalence and minimum sample size checks
- Auto-generated split_protocol.json compatible with split_protocol_gate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SUPPORTED_STRATEGIES = ("grouped_temporal", "grouped_random", "stratified_grouped")
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
MIN_ROWS_PER_SPLIT = 20
MIN_POSITIVE_PER_SPLIT = 10
MIN_NEGATIVE_PER_SPLIT = 10
MIN_PATIENTS_PER_SPLIT = 5
PREVALENCE_SHIFT_WARN_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the splitting tool.

    Returns:
        Parsed argument namespace with input path, output dir,
        column names, strategy, ratios, and seed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split a single CSV into train/valid/test with medical safety guarantees. "
            "Ensures patient-level disjoint splits, temporal ordering (when applicable), "
            "and minimum sample/prevalence checks per split."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to the complete CSV file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write train.csv, valid.csv, test.csv.",
    )
    parser.add_argument(
        "--patient-id-col",
        required=True,
        help="Column name for patient/entity ID (used for group-disjoint splitting).",
    )
    parser.add_argument(
        "--target-col",
        default="y",
        help="Binary target column name (default: y).",
    )
    parser.add_argument(
        "--time-col",
        default="",
        help="Index time column for temporal splitting (required for grouped_temporal strategy).",
    )
    parser.add_argument(
        "--strategy",
        default="grouped_temporal",
        choices=list(SUPPORTED_STRATEGIES),
        help=(
            "Splitting strategy. "
            "grouped_temporal: sort by time, then group-disjoint split (recommended for longitudinal data). "
            "grouped_random: group-disjoint random split (for cross-sectional data). "
            "stratified_grouped: group-disjoint split with stratification to preserve positive rate. "
            "(default: grouped_temporal)"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train set ratio (default: 0.6).")
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO, help="Valid set ratio (default: 0.2).")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test set ratio (default: 0.2).")
    parser.add_argument("--seed", type=int, default=20260228, help="Random seed for reproducible splitting (default: 20260228).")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument(
        "--split-protocol-out",
        default="",
        help="Path to write auto-generated split_protocol.json (default: <output-dir>/../configs/split_protocol.json).",
    )
    parser.add_argument(
        "--min-rows-per-split",
        type=int,
        default=MIN_ROWS_PER_SPLIT,
        help=f"Minimum rows per split (default: {MIN_ROWS_PER_SPLIT}).",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file for reproducibility traceability."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file or the CSV is empty.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input CSV not found: {p}")
    if not p.is_file():
        raise ValueError(f"Input path is not a file: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {p}")
    return df


def validate_columns(df: pd.DataFrame, patient_id_col: str, target_col: str, time_col: str) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        patient_id_col: Patient ID column name.
        target_col: Binary target column name.
        time_col: Time column name (empty string to skip).

    Raises:
        ValueError: If any required column is missing.
    """
    missing = []
    if patient_id_col not in df.columns:
        missing.append(f"patient_id_col '{patient_id_col}'")
    if target_col not in df.columns:
        missing.append(f"target_col '{target_col}'")
    if time_col and time_col not in df.columns:
        missing.append(f"time_col '{time_col}'")
    if missing:
        available = ", ".join(df.columns[:20].tolist())
        if len(df.columns) > 20:
            available += f", ... ({len(df.columns) - 20} more)"
        raise ValueError(
            f"Missing columns: {', '.join(missing)}. Available: {available}"
        )


def validate_binary_target(df: pd.DataFrame, target_col: str) -> int:
    """Validate that the target column contains only binary (0/1) values.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        Count of NaN target rows (which will be excluded).

    Raises:
        ValueError: If target contains non-numeric or non-binary values,
            or has no valid values.
    """
    na_count = int(df[target_col].isna().sum())
    if na_count > 0:
        print(
            f"[WARN] {na_count} rows have NaN target values in '{target_col}'; "
            "these rows will be EXCLUDED before splitting.",
            file=sys.stderr,
        )
    unique = set(df[target_col].dropna().unique())
    numeric_values = set()
    for v in unique:
        try:
            fv = float(v)
        except (ValueError, TypeError):
            raise ValueError(
                f"Target column '{target_col}' contains non-numeric value: {v}. "
                "Expected only 0/1."
            )
        if fv == 0.0:
            numeric_values.add(0)
        elif fv == 1.0:
            numeric_values.add(1)
        else:
            raise ValueError(
                f"Target column '{target_col}' contains non-binary value: {v} (={fv}). "
                "Expected only 0/1."
            )
    if not numeric_values:
        raise ValueError(f"Target column '{target_col}' has no valid values.")
    if len(numeric_values) < 2:
        print(
            f"[WARN] Target column '{target_col}' has only one class ({numeric_values}). "
            "Splitting may produce single-class splits.",
            file=sys.stderr,
        )
    return na_count


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios sum to ~1.0 and each is >= 0.05.

    Args:
        train_ratio: Training set proportion.
        valid_ratio: Validation set proportion.
        test_ratio: Test set proportion.

    Raises:
        ValueError: If ratios do not sum to ~1.0 or any ratio < 0.05.
    """
    total = train_ratio + valid_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"Split ratios must sum to ~1.0, got {train_ratio} + {valid_ratio} + {test_ratio} = {total}"
        )
    for name, ratio in [("train", train_ratio), ("valid", valid_ratio), ("test", test_ratio)]:
        if ratio < 0.05:
            raise ValueError(f"{name}_ratio must be >= 0.05, got {ratio}")


def get_patient_label(df: pd.DataFrame, patient_id_col: str, target_col: str) -> pd.DataFrame:
    """Compute per-patient majority label for stratified splitting.

    Args:
        df: Input DataFrame.
        patient_id_col: Patient ID column name.
        target_col: Binary target column name.

    Returns:
        DataFrame indexed by patient ID with columns 'mean', 'count',
        and 'label' (1 if mean >= 0.5, else 0).
    """
    patient_groups = df.groupby(patient_id_col)[target_col].agg(["mean", "count"])
    patient_groups["label"] = (patient_groups["mean"] >= 0.5).astype(int)
    return patient_groups


def _temp_col_name(df: pd.DataFrame, base: str = "__split_tmp__") -> str:
    """Generate a temporary column name that doesn't collide with existing columns."""
    name = base
    idx = 0
    while name in df.columns:
        idx += 1
        name = f"{base}{idx}"
    return name


def split_grouped_temporal(
    df: pd.DataFrame,
    patient_id_col: str,
    time_col: str,
    target_col: str,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by temporal order with patient-level disjoint groups.

    Patients are sorted by their earliest event time and assigned
    sequentially to train/valid/test. Deterministic (no seed needed).

    Args:
        df: Input DataFrame.
        patient_id_col: Patient ID column name.
        time_col: Time column name for temporal ordering.
        target_col: Binary target column name.
        train_ratio: Proportion of patients for training.
        valid_ratio: Proportion of patients for validation.

    Returns:
        Tuple of (train_df, valid_df, test_df).

    Raises:
        ValueError: If time_col is empty or fully unparseable.
    """
    if not time_col:
        raise ValueError("grouped_temporal strategy requires --time-col.")

    # Parse time column
    time_series = pd.to_datetime(df[time_col], errors="coerce")
    if time_series.isna().all():
        raise ValueError(
            f"Cannot parse any values in time column '{time_col}'. "
            "Use --strategy grouped_random if no time information is available."
        )
    na_count = int(time_series.isna().sum())
    if na_count > 0:
        print(
            f"[WARN] {na_count} rows have unparseable time values in '{time_col}'; "
            "these rows will be placed in the training set.",
            file=sys.stderr,
        )

    # Get earliest time per patient (collision-safe temp column)
    df = df.copy()
    tmp_col = _temp_col_name(df)
    df[tmp_col] = time_series
    patient_first_time = df.groupby(patient_id_col)[tmp_col].min().reset_index()
    _pmt_col = _temp_col_name(patient_first_time, "__patient_min_time__")
    patient_first_time.columns = [patient_id_col, _pmt_col]

    # Sort patients by their earliest time
    patient_first_time = patient_first_time.sort_values(_pmt_col, na_position="first")
    patients_ordered = patient_first_time[patient_id_col].tolist()

    n = len(patients_ordered)
    train_end = int(round(n * train_ratio))
    valid_end = int(round(n * (train_ratio + valid_ratio)))

    # Ensure at least 1 patient per split
    train_end = max(1, min(train_end, n - 2))
    valid_end = max(train_end + 1, min(valid_end, n - 1))

    train_patients = set(patients_ordered[:train_end])
    valid_patients = set(patients_ordered[train_end:valid_end])
    test_patients = set(patients_ordered[valid_end:])

    train_df = df[df[patient_id_col].isin(train_patients)].drop(columns=[tmp_col])
    valid_df = df[df[patient_id_col].isin(valid_patients)].drop(columns=[tmp_col])
    test_df = df[df[patient_id_col].isin(test_patients)].drop(columns=[tmp_col])

    return train_df, valid_df, test_df


def split_grouped_random(
    df: pd.DataFrame,
    patient_id_col: str,
    target_col: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by random patient-level grouping.

    Patients are shuffled randomly and assigned to train/valid/test.

    Args:
        df: Input DataFrame.
        patient_id_col: Patient ID column name.
        target_col: Binary target column name.
        train_ratio: Proportion of patients for training.
        valid_ratio: Proportion of patients for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, valid_df, test_df).
    """
    rng = np.random.default_rng(seed)
    patients = df[patient_id_col].unique().tolist()
    rng.shuffle(patients)

    n = len(patients)
    train_end = int(round(n * train_ratio))
    valid_end = int(round(n * (train_ratio + valid_ratio)))
    train_end = max(1, min(train_end, n - 2))
    valid_end = max(train_end + 1, min(valid_end, n - 1))

    train_patients = set(patients[:train_end])
    valid_patients = set(patients[train_end:valid_end])
    test_patients = set(patients[valid_end:])

    train_df = df[df[patient_id_col].isin(train_patients)]
    valid_df = df[df[patient_id_col].isin(valid_patients)]
    test_df = df[df[patient_id_col].isin(test_patients)]

    return train_df, valid_df, test_df


def split_stratified_grouped(
    df: pd.DataFrame,
    patient_id_col: str,
    target_col: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split with stratification by patient-level majority label.

    Patients are split per-class to preserve positive rate across
    train/valid/test while maintaining group-disjoint property.

    Args:
        df: Input DataFrame.
        patient_id_col: Patient ID column name.
        target_col: Binary target column name.
        train_ratio: Proportion of patients for training.
        valid_ratio: Proportion of patients for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, valid_df, test_df).
    """
    rng = np.random.default_rng(seed)
    patient_labels = get_patient_label(df, patient_id_col, target_col)

    train_patients: List[str] = []
    valid_patients: List[str] = []
    test_patients: List[str] = []

    for label_val in [0, 1]:
        group = patient_labels[patient_labels["label"] == label_val].index.tolist()
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        if n < 3:
            # Too few patients in this class to split 3 ways; put all in train
            # (downstream validate_splits will catch insufficient patients/samples)
            train_patients.extend(group)
            continue
        te = int(round(n * train_ratio))
        ve = int(round(n * (train_ratio + valid_ratio)))
        te = max(1, min(te, n - 2))
        ve = max(te + 1, min(ve, n - 1))

        train_patients.extend(group[:te])
        valid_patients.extend(group[te:ve])
        test_patients.extend(group[ve:])

    train_set = set(train_patients)
    valid_set = set(valid_patients)
    test_set = set(test_patients)

    train_df = df[df[patient_id_col].isin(train_set)]
    valid_df = df[df[patient_id_col].isin(valid_set)]
    test_df = df[df[patient_id_col].isin(test_set)]

    return train_df, valid_df, test_df


def validate_splits(
    splits: Dict[str, pd.DataFrame],
    patient_id_col: str,
    target_col: str,
    time_col: str,
    min_rows: int,
) -> List[Dict[str, Any]]:
    """Run post-split safety checks and return any issues found.

    Checks: minimum rows, minimum positive/negative counts, minimum
    patients, patient-level disjointness, temporal ordering, and
    prevalence shift.

    Args:
        splits: Dict mapping split name to DataFrame.
        patient_id_col: Patient ID column name.
        target_col: Binary target column name.
        time_col: Time column name (empty to skip temporal checks).
        min_rows: Minimum required rows per split.

    Returns:
        List of issue dicts with 'code', 'message', and context fields.
        Issues with level='warn' are non-blocking.
    """
    issues: List[Dict[str, Any]] = []

    # Check minimum rows
    for name, df in splits.items():
        if len(df) < min_rows:
            issues.append({
                "code": "insufficient_rows",
                "message": f"{name} split has {len(df)} rows (minimum: {min_rows}).",
                "split": name,
                "rows": len(df),
            })

    # Check minimum positive/negative per split
    for name, df in splits.items():
        target = df[target_col].astype(float)
        pos = int((target == 1.0).sum())
        neg = int((target == 0.0).sum())
        if pos < MIN_POSITIVE_PER_SPLIT:
            issues.append({
                "code": "insufficient_positive",
                "message": f"{name} split has {pos} positive samples (minimum: {MIN_POSITIVE_PER_SPLIT}).",
                "split": name,
                "positive_count": pos,
            })
        if neg < MIN_NEGATIVE_PER_SPLIT:
            issues.append({
                "code": "insufficient_negative",
                "message": f"{name} split has {neg} negative samples (minimum: {MIN_NEGATIVE_PER_SPLIT}).",
                "split": name,
                "negative_count": neg,
            })

    # Check minimum patients per split
    for name, df in splits.items():
        n_patients = int(df[patient_id_col].nunique())
        if n_patients < MIN_PATIENTS_PER_SPLIT:
            issues.append({
                "code": "insufficient_patients_in_split",
                "message": f"{name} split has {n_patients} unique patients (minimum: {MIN_PATIENTS_PER_SPLIT}).",
                "split": name,
                "patient_count": n_patients,
            })

    # Check patient-level disjoint
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            left_name = split_names[i]
            right_name = split_names[j]
            left_ids = set(splits[left_name][patient_id_col].astype(str).unique())
            right_ids = set(splits[right_name][patient_id_col].astype(str).unique())
            overlap = left_ids & right_ids
            if overlap:
                issues.append({
                    "code": "patient_overlap",
                    "message": f"Patient IDs overlap between {left_name} and {right_name} ({len(overlap)} patients).",
                    "left": left_name,
                    "right": right_name,
                    "overlap_count": len(overlap),
                })

    # Check temporal ordering (if time column available)
    if time_col and time_col in splits.get("train", pd.DataFrame()).columns:
        ordered_splits = [("train", "valid"), ("valid", "test")]
        for left_name, right_name in ordered_splits:
            if left_name not in splits or right_name not in splits:
                continue
            left_times = pd.to_datetime(splits[left_name][time_col], errors="coerce").dropna()
            right_times = pd.to_datetime(splits[right_name][time_col], errors="coerce").dropna()
            if left_times.empty or right_times.empty:
                continue
            if left_times.max() >= right_times.min():
                issues.append({
                    "code": "temporal_overlap",
                    "message": (
                        f"Temporal boundary violation: {left_name} (max: {left_times.max()}) "
                        f">= {right_name} (min: {right_times.min()}). "
                        "Multi-visit patients may span time boundaries; "
                        "temporal ordering is enforced at the patient level (earliest event time)."
                    ),
                    "left": left_name,
                    "right": right_name,
                })

    # Check prevalence shift between train and test
    if "train" in splits and "test" in splits:
        train_target = splits["train"][target_col].astype(float)
        test_target = splits["test"][target_col].astype(float)
        train_prev = float(train_target.mean()) if len(train_target) > 0 else 0.0
        test_prev = float(test_target.mean()) if len(test_target) > 0 else 0.0
        shift = abs(train_prev - test_prev)
        if shift > PREVALENCE_SHIFT_WARN_THRESHOLD:
            issues.append({
                "code": "prevalence_shift",
                "message": (
                    f"Prevalence shift between train ({train_prev:.3f}) and test ({test_prev:.3f}): "
                    f"delta={shift:.3f} exceeds threshold {PREVALENCE_SHIFT_WARN_THRESHOLD}. "
                    "Consider stratified_grouped strategy or verify temporal prevalence drift is expected."
                ),
                "level": "warn",
                "train_prevalence": round(train_prev, 4),
                "test_prevalence": round(test_prev, 4),
            })

    return issues


def split_summary(df: pd.DataFrame, target_col: str, time_col: str, patient_id_col: str) -> Dict[str, Any]:
    """Compute summary statistics for a single split.

    Args:
        df: Split DataFrame.
        target_col: Binary target column name.
        time_col: Time column name (empty to skip time range).
        patient_id_col: Patient ID column name.

    Returns:
        Dict with rows, patients, positive/negative counts,
        positive_rate, and optional time_min/time_max.
    """
    target = df[target_col].astype(float)
    pos = int((target == 1.0).sum())
    neg = int((target == 0.0).sum())
    total = pos + neg
    summary: Dict[str, Any] = {
        "rows": len(df),
        "patients": int(df[patient_id_col].nunique()),
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": round(pos / total, 4) if total > 0 else 0.0,
    }
    if time_col and time_col in df.columns:
        times = pd.to_datetime(df[time_col], errors="coerce").dropna()
        if not times.empty:
            summary["time_min"] = str(times.min().date())
            summary["time_max"] = str(times.max().date())
    return summary


def generate_split_protocol(
    strategy: str,
    patient_id_col: str,
    time_col: str,
    seed: int,
) -> Dict[str, Any]:
    """Generate a split_protocol.json compatible with split_protocol_gate.

    Args:
        strategy: Splitting strategy name.
        patient_id_col: Patient ID column name.
        time_col: Time column name (empty string if none).
        seed: Random seed used for splitting.

    Returns:
        Protocol dict with strategy, disjoint/temporal flags, and notes.
    """
    requires_temporal = strategy == "grouped_temporal"
    return {
        "split_strategy": strategy,
        "split_reference": f"auto-split-seed-{seed}",
        "id_col": patient_id_col,
        "index_time_col": time_col if time_col else "",
        "frozen_before_modeling": True,
        "requires_group_disjoint": True,
        "requires_temporal_order": requires_temporal,
        "allow_patient_overlap": False,
        "allow_time_overlap": False,
        "split_seed_locked": True,
        "notes": (
            f"Auto-generated by split_data.py using {strategy} strategy with seed {seed}. "
            "Patient-level disjoint splits enforced."
        ),
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a DataFrame to CSV via tmp + rename.

    Args:
        df: DataFrame to save.
        path: Target output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON file via tmp + rename.

    Args:
        path: Target output path.
        payload: Dict to serialize as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main() -> int:
    """Entry point for the split_data pipeline.

    Loads CSV, validates columns and target, splits data using the
    chosen strategy, runs safety checks, writes split files and
    protocol, and generates a split report.

    Returns:
        Exit code (0 for success, 2 for validation/split failure).
    """
    args = parse_args()

    # Validate ratios
    try:
        validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    # Load CSV
    try:
        df = load_csv(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns from {args.input}")

    # Validate columns
    time_col = args.time_col.strip()
    if args.strategy == "grouped_temporal" and not time_col:
        print(
            "[FAIL] --time-col is required for grouped_temporal strategy. "
            "Use --strategy grouped_random if no time column is available.",
            file=sys.stderr,
        )
        return 2

    try:
        validate_columns(df, args.patient_id_col, args.target_col, time_col)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    try:
        target_na_count = validate_binary_target(df, args.target_col)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    # Track true original row count before any exclusions
    raw_row_count = len(df)

    # C3: Reject rows with NaN patient_id
    pid_na_count = int(df[args.patient_id_col].isna().sum())
    if pid_na_count > 0:
        print(
            f"[WARN] {pid_na_count} rows have NaN patient IDs; excluding before split.",
            file=sys.stderr,
        )
        df = df.dropna(subset=[args.patient_id_col])

    # C4: Exclude rows with NaN target (recount after pid exclusion to avoid double-counting)
    target_na_after_pid = int(df[args.target_col].isna().sum())
    if target_na_after_pid > 0:
        df = df.dropna(subset=[args.target_col])

    rows_after_clean = len(df)
    total_excluded = raw_row_count - rows_after_clean
    if rows_after_clean == 0:
        print("[FAIL] No valid rows remain after excluding NaN patient_id/target.", file=sys.stderr)
        return 2

    n_patients = df[args.patient_id_col].nunique()
    print(f"[INFO] {n_patients} unique patients, {rows_after_clean} rows (excluded {total_excluded}), strategy: {args.strategy}")

    if n_patients < 6:
        print(
            f"[FAIL] Need at least 6 unique patients for 3-way split, found {n_patients}.",
            file=sys.stderr,
        )
        return 2

    # Split
    try:
        if args.strategy == "grouped_temporal":
            train_df, valid_df, test_df = split_grouped_temporal(
                df, args.patient_id_col, time_col, args.target_col,
                args.train_ratio, args.valid_ratio,
            )
        elif args.strategy == "grouped_random":
            train_df, valid_df, test_df = split_grouped_random(
                df, args.patient_id_col, args.target_col,
                args.train_ratio, args.valid_ratio, args.seed,
            )
        elif args.strategy == "stratified_grouped":
            train_df, valid_df, test_df = split_stratified_grouped(
                df, args.patient_id_col, args.target_col,
                args.train_ratio, args.valid_ratio, args.seed,
            )
        else:
            print(f"[FAIL] Unknown strategy: {args.strategy}", file=sys.stderr)
            return 2
    except (ValueError, KeyError) as exc:
        print(f"[FAIL] Splitting failed: {exc}", file=sys.stderr)
        return 2

    splits = {"train": train_df, "valid": valid_df, "test": test_df}

    # H2: Row count preservation assertion
    split_total = len(train_df) + len(valid_df) + len(test_df)
    if split_total != rows_after_clean:
        print(
            f"[FAIL] Row count mismatch: cleaned_input={rows_after_clean}, "
            f"train+valid+test={split_total} (lost {rows_after_clean - split_total} rows).",
            file=sys.stderr,
        )
        return 2

    # Validate splits
    issues = validate_splits(splits, args.patient_id_col, args.target_col, time_col, args.min_rows_per_split)
    hard_failures = [i for i in issues if i.get("level") != "warn"]
    warnings = [i for i in issues if i.get("level") == "warn"]

    for issue in hard_failures:
        print(f"[FAIL] {issue['message']}", file=sys.stderr)
    for issue in warnings:
        print(f"[WARN] {issue['message']}", file=sys.stderr)

    if hard_failures:
        print(
            f"\n[FAIL] {len(hard_failures)} safety check(s) failed. "
            "The input data may be too small or have insufficient class diversity.",
            file=sys.stderr,
        )
        return 2

    # Write split files
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    valid_path = output_dir / "valid.csv"
    test_path = output_dir / "test.csv"

    save_csv(train_df, train_path)
    save_csv(valid_df, valid_path)
    save_csv(test_df, test_path)

    # C1: Warn if non-temporal strategy may fail split_protocol_gate
    if args.strategy != "grouped_temporal":
        print(
            f"\n[WARN] Strategy '{args.strategy}' does not enforce temporal ordering. "
            "The downstream split_protocol_gate requires temporal order for publication-grade claims. "
            "Use --strategy grouped_temporal for full pipeline compatibility.",
            file=sys.stderr,
        )

    # Generate split_protocol.json
    protocol = generate_split_protocol(args.strategy, args.patient_id_col, time_col, args.seed)
    if args.split_protocol_out:
        protocol_path = Path(args.split_protocol_out).expanduser().resolve()
    else:
        configs_dir = output_dir.parent / "configs"
        protocol_path = configs_dir / "split_protocol.json"
    write_json(protocol_path, protocol)

    # Summary
    train_summary = split_summary(train_df, args.target_col, time_col, args.patient_id_col)
    valid_summary = split_summary(valid_df, args.target_col, time_col, args.patient_id_col)
    test_summary = split_summary(test_df, args.target_col, time_col, args.patient_id_col)

    print(f"\n[INFO] Split complete:")
    print(f"  Train: {train_path} ({train_summary['rows']} rows, {train_summary['patients']} patients, pos_rate={train_summary['positive_rate']:.3f})")
    print(f"  Valid: {valid_path} ({valid_summary['rows']} rows, {valid_summary['patients']} patients, pos_rate={valid_summary['positive_rate']:.3f})")
    print(f"  Test:  {test_path} ({test_summary['rows']} rows, {test_summary['patients']} patients, pos_rate={test_summary['positive_rate']:.3f})")
    print(f"  Protocol: {protocol_path}")

    # H3: Input file SHA256 fingerprint
    input_path = Path(args.input).expanduser().resolve()
    input_sha256 = file_sha256(input_path)

    # Build report
    report = {
        "contract_version": "split_report.v1",
        "status": "pass",
        "strategy": args.strategy,
        "seed": args.seed,
        "input_file": str(input_path),
        "input_sha256": input_sha256,
        "input_rows_raw": raw_row_count,
        "input_rows_clean": rows_after_clean,
        "input_rows_excluded": {
            "nan_patient_id": pid_na_count,
            "nan_target_after_pid_clean": target_na_after_pid,
            "total": total_excluded,
        },
        "input_patients": n_patients,
        "output_dir": str(output_dir),
        "files": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
            "split_protocol": str(protocol_path),
        },
        "splits": {
            "train": train_summary,
            "valid": valid_summary,
            "test": test_summary,
        },
        "safety_checks": {
            "patient_disjoint": not any(i["code"] == "patient_overlap" for i in issues),
            "temporal_order": not any(i["code"] == "temporal_overlap" for i in issues),
            "min_positive_per_split": MIN_POSITIVE_PER_SPLIT,
            "min_negative_per_split": MIN_NEGATIVE_PER_SPLIT,
            "min_rows_per_split": args.min_rows_per_split,
        },
        "warnings": warnings,
    }

    report_path: Optional[Path] = None
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
    else:
        evidence_dir = output_dir.parent / "evidence"
        report_path = evidence_dir / "split_report.json"

    write_json(report_path, report)
    print(f"  Report: {report_path}")

    print(f"\nStatus: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
