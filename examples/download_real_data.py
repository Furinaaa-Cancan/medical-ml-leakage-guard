#!/usr/bin/env python3
"""
Download and prepare real UCI medical datasets for ml-leakage-guard pipeline.

Produces a single CSV ready for:
  python3 scripts/mlgg.py split -- --input <csv> --patient-id-col patient_id ...

Available datasets:
  heart    - UCI Heart Disease (Cleveland), 303 rows, 13 features
  breast   - UCI Breast Cancer Wisconsin (Diagnostic), 569 rows, 30 features
  ckd      - UCI Chronic Kidney Disease, ~400 rows, 24 features

Usage:
  python3 examples/download_real_data.py heart
  python3 examples/download_real_data.py breast
  python3 examples/download_real_data.py ckd
  python3 examples/download_real_data.py heart --output /tmp/heart.csv

Then split and run the pipeline:
  python3 scripts/mlgg.py split -- \\
    --input examples/heart_disease.csv \\
    --output-dir /tmp/mlgg_heart/data \\
    --patient-id-col patient_id \\
    --target-col y \\
    --time-col event_time \\
    --strategy grouped_temporal
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RAW_DIR = REPO_ROOT / "experiments" / "authority-e2e" / "raw"

# UCI download URLs
URLS = {
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "breast": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
}


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {e.reason}")
        print(f"  URL may have moved. Check https://archive.ics.uci.edu/ml/datasets for updates.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  [ERROR] Network error: {e.reason}")
        print(f"  Check your internet connection or try again later.")
        sys.exit(1)
    size = dest.stat().st_size
    if size == 0:
        print(f"  [ERROR] Downloaded file is empty (0 bytes). URL may be invalid.")
        dest.unlink(missing_ok=True)
        sys.exit(1)
    print(f"  Saved to {dest} ({size:,} bytes)")


def add_patient_id_and_time(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add patient_id and synthetic event_time for pipeline compatibility.

    Shuffles rows first (so class ordering in original data doesn't create
    prevalence drift in temporal splits), then assigns unique timestamps.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["patient_id"] = [f"P{i:04d}" for i in range(1, n + 1)]
    # Spread across ~2 years with unique timestamps to avoid temporal boundary ties
    start = datetime(2023, 1, 1)
    total_minutes = 2 * 365 * 24 * 60  # ~2 years in minutes
    offsets = sorted(rng.choice(total_minutes, size=n, replace=False))
    df["event_time"] = [(start + timedelta(minutes=int(m))).strftime("%Y-%m-%d %H:%M") for m in offsets]
    return df


def prepare_heart(output: Path) -> None:
    """UCI Heart Disease (Cleveland) — 303 patients, 13 clinical features."""
    print("\n=== UCI Heart Disease (Cleveland) ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/heart+disease")
    print("  Rows: ~303 | Features: 13 | Task: predict heart disease presence")

    # Try local raw file first
    local = RAW_DIR / "heart_disease_processed.cleveland.data"
    if local.exists():
        print(f"  Using local file: {local}")
        raw_path = local
    else:
        raw_path = output.parent / ".heart_raw.data"
        download_file(URLS["heart"], raw_path)

    columns = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "oldpeak", "slope", "num_major_vessels", "thal", "goal"
    ]
    df = pd.read_csv(raw_path, header=None, names=columns, na_values="?")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Cleaned: {before} → {len(df)} rows (dropped {before - len(df)} with missing values)")

    # Binary target: goal > 0 means heart disease present
    df["y"] = (df["goal"] > 0).astype(int)
    df = df.drop(columns=["goal"])

    df = add_patient_id_and_time(df)

    # Reorder: patient_id, event_time, y, features...
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    # Clean up temp file if we downloaded
    if raw_path != local and raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive: {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")
    print(f"\n  Ready to use:")
    print(f"    python3 scripts/mlgg.py split -- \\")
    print(f"      --input {output} \\")
    print(f"      --output-dir /tmp/mlgg_heart/data \\")
    print(f"      --patient-id-col patient_id --target-col y --time-col event_time \\")
    print(f"      --strategy grouped_temporal")


def prepare_breast(output: Path) -> None:
    """UCI Breast Cancer Wisconsin (Diagnostic) — 569 patients, 30 features."""
    print("\n=== UCI Breast Cancer Wisconsin (Diagnostic) ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)")
    print("  Rows: 569 | Features: 30 | Task: predict malignant vs benign")

    local = RAW_DIR / "breast_cancer_wdbc.data"
    if local.exists():
        print(f"  Using local file: {local}")
        raw_path = local
    else:
        raw_path = output.parent / ".breast_raw.data"
        download_file(URLS["breast"], raw_path)

    # 30 real-valued features computed from cell nuclei images
    feature_names: List[str] = []
    for prefix in ["mean", "se", "worst"]:
        for feat in ["radius", "texture", "perimeter", "area", "smoothness",
                      "compactness", "concavity", "concave_points", "symmetry", "fractal_dim"]:
            feature_names.append(f"{prefix}_{feat}")

    columns = ["id", "diagnosis"] + feature_names
    df = pd.read_csv(raw_path, header=None, names=columns)

    # Binary target: M=malignant=1, B=benign=0
    df["y"] = (df["diagnosis"] == "M").astype(int)
    df = df.drop(columns=["id", "diagnosis"])

    df = add_patient_id_and_time(df, seed=43)

    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    # Clean up temp file if we downloaded
    if raw_path != local and raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (malignant): {pos} ({pos/len(df)*100:.1f}%) | Negative (benign): {neg}")
    print(f"\n  Ready to use:")
    print(f"    python3 scripts/mlgg.py split -- \\")
    print(f"      --input {output} \\")
    print(f"      --output-dir /tmp/mlgg_breast/data \\")
    print(f"      --patient-id-col patient_id --target-col y --time-col event_time \\")
    print(f"      --strategy grouped_temporal")


def prepare_ckd(output: Path) -> None:
    """UCI Chronic Kidney Disease — ~400 patients, 24 features."""
    print("\n=== UCI Chronic Kidney Disease ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease")
    print("  Rows: ~400 | Features: 24 | Task: predict CKD presence")

    local = RAW_DIR / "chronic_kidney_disease" / "Chronic_Kidney_Disease" / "chronic_kidney_disease.arff"
    if not local.exists():
        print(f"  [ERROR] Local file not found: {local}")
        print(f"  Please download from UCI and extract to: {RAW_DIR / 'chronic_kidney_disease'}")
        print(f"  Or use 'heart' or 'breast' datasets which support auto-download.")
        sys.exit(1)

    print(f"  Using local file: {local}")

    # Parse ARFF manually
    columns: List[str] = []
    data_lines: List[str] = []
    in_data = False
    with open(local, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.lower().startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    columns.append(parts[1].strip("'\""))
            elif line.lower() == "@data":
                in_data = True
            elif in_data and line and not line.startswith("%"):
                # Fix known CKD ARFF issues: embedded tabs (e.g. "\tno" → "no")
                # and trailing commas producing phantom empty fields
                line = line.replace("\t", "").rstrip(",")
                data_lines.append(line)

    n_cols = len(columns)
    # Filter lines with wrong field count (known CKD data quality issue)
    clean_lines = []
    skipped = 0
    for dl in data_lines:
        if len(dl.split(",")) == n_cols:
            clean_lines.append(dl)
        else:
            skipped += 1
    if skipped:
        print(f"  [INFO] Skipped {skipped} malformed row(s) with wrong field count.")
    csv_text = "\n".join(clean_lines)
    df = pd.read_csv(io.StringIO(csv_text), header=None, names=columns, na_values=["?"])

    # Target: class column — ckd=1, notckd=0
    target_col = columns[-1]  # 'class'
    df["y"] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ("ckd", "ckd.") else 0)
    df = df.drop(columns=[target_col])

    # Encode categorical features as binary/numeric before converting
    BINARY_MAPS: Dict[str, Dict[str, int]] = {
        "rbc": {"normal": 0, "abnormal": 1},
        "pc": {"normal": 0, "abnormal": 1},
        "pcc": {"notpresent": 0, "present": 1},
        "ba": {"notpresent": 0, "present": 1},
        "htn": {"no": 0, "yes": 1},
        "dm": {"no": 0, "yes": 1},
        "cad": {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe": {"no": 0, "yes": 1},
        "ane": {"no": 0, "yes": 1},
    }
    for col in df.columns:
        if col == "y":
            continue
        if col in BINARY_MAPS:
            col_map = BINARY_MAPS[col]
            df[col] = df[col].apply(
                lambda x, m=col_map: m.get(str(x).strip().lower()) if pd.notna(x) else np.nan
            )
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop columns with >50% missing
    thresh = len(df) * 0.5
    df = df.dropna(axis=1, thresh=int(thresh))

    # Drop rows with all-NaN features
    feature_cols = [c for c in df.columns if c != "y"]
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)

    df = add_patient_id_and_time(df, seed=44)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (CKD): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")
    print(f"\n  Ready to use:")
    print(f"    python3 scripts/mlgg.py split -- \\")
    print(f"      --input {output} \\")
    print(f"      --output-dir /tmp/mlgg_ckd/data \\")
    print(f"      --patient-id-col patient_id --target-col y --time-col event_time \\")
    print(f"      --strategy grouped_temporal")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare real UCI medical datasets for ml-leakage-guard pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        choices=["heart", "breast", "ckd", "all"],
        help="Which dataset to prepare: heart (303 rows), breast (569 rows), ckd (~400 rows), or all.",
    )
    parser.add_argument("--output", default="", help="Output CSV path (default: examples/<dataset>.csv).")
    args = parser.parse_args()

    examples_dir = SCRIPT_DIR
    examples_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        prepare_heart(examples_dir / "heart_disease.csv")
        prepare_breast(examples_dir / "breast_cancer.csv")
        prepare_ckd(examples_dir / "chronic_kidney_disease.csv")
    elif args.dataset == "heart":
        out = Path(args.output) if args.output else examples_dir / "heart_disease.csv"
        prepare_heart(out)
    elif args.dataset == "breast":
        out = Path(args.output) if args.output else examples_dir / "breast_cancer.csv"
        prepare_breast(out)
    elif args.dataset == "ckd":
        out = Path(args.output) if args.output else examples_dir / "chronic_kidney_disease.csv"
        prepare_ckd(out)

    print("\n✓ Done! Use the commands above to split and run the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
