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
import zipfile
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
    "hepatitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "spect": "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train",
    "spect_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test",
    "dermatology": "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
    "pima": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "mammographic": "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
    "framingham": "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv",
    "diabetes130_zip": "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
    "thyroid_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data",
    "thyroid_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data",
    "eeg_eye": "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff",
}


def download_file(url: str, dest: Path) -> None:
    print(f"  Downloading from {url} ...")
    try:
        resp = urllib.request.urlopen(url, timeout=60)
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code}: {e.reason}")
        print(f"  URL may have moved. Check https://archive.ics.uci.edu/ml/datasets for updates.")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"  [ERROR] Network error: {e.reason}")
        print(f"  Check your internet connection or try again later.")
        sys.exit(1)
    total = int(resp.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 8192
    with open(dest, "wb") as fh:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                print(f"\r  Progress: {pct}% ({downloaded:,}/{total:,} bytes)", end="", file=sys.stderr)
            else:
                print(f"\r  Downloaded: {downloaded:,} bytes", end="", file=sys.stderr)
    print(file=sys.stderr)  # newline after progress
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


def prepare_hepatitis(output: Path) -> None:
    """UCI Hepatitis — ~155 patients, 19 features."""
    print("\n=== UCI Hepatitis ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/hepatitis")
    print("  Rows: ~155 | Features: 19 | Task: predict survival")

    raw_path = output.parent / ".hepatitis_raw.data"
    download_file(URLS["hepatitis"], raw_path)

    columns = [
        "class", "age", "sex", "steroid", "antivirals", "fatigue", "malaise",
        "anorexia", "liver_big", "liver_firm", "spleen_palpable", "spiders",
        "ascites", "varices", "bilirubin", "alk_phosphate", "sgot",
        "albumin", "protime", "histology",
    ]
    df = pd.read_csv(raw_path, header=None, names=columns, na_values="?")

    # Binary target: class 1=die, 2=live → 1=die, 0=live
    df["y"] = (df["class"] == 1).astype(int)
    df = df.drop(columns=["class"])

    # Convert all to numeric
    for col in df.columns:
        if col != "y":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all", subset=[c for c in df.columns if c != "y"]).reset_index(drop=True)
    df = add_patient_id_and_time(df, seed=45)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (die): {pos} ({pos/len(df)*100:.1f}%) | Negative (live): {neg}")


def prepare_spect(output: Path) -> None:
    """UCI SPECT Heart — ~267 patients, 22 binary features."""
    print("\n=== UCI SPECT Heart ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/SPECT+Heart")
    print("  Rows: ~267 | Features: 22 | Task: predict cardiac SPECT diagnosis")

    train_path = output.parent / ".spect_train.data"
    test_path = output.parent / ".spect_test.data"
    download_file(URLS["spect"], train_path)
    download_file(URLS["spect_test"], test_path)

    columns = ["y"] + [f"F{i}" for i in range(1, 23)]
    df_train = pd.read_csv(train_path, header=None, names=columns)
    df_test = pd.read_csv(test_path, header=None, names=columns)
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = add_patient_id_and_time(df, seed=46)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    for p in [train_path, test_path]:
        if p.exists():
            p.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (abnormal): {pos} ({pos/len(df)*100:.1f}%) | Negative (normal): {neg}")


def prepare_dermatology(output: Path) -> None:
    """UCI Dermatology — ~366 patients, 34 features (binarized to erythemato-squamous vs rest)."""
    print("\n=== UCI Dermatology ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/dermatology")
    print("  Rows: ~366 | Features: 34 | Task: predict psoriasis (class 1 vs rest)")

    raw_path = output.parent / ".dermatology_raw.data"
    download_file(URLS["dermatology"], raw_path)

    columns = [
        "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
        "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
        "knee_elbow_involvement", "scalp_involvement", "family_history", "melanin_incontinence",
        "eosinophils_in_infiltrate", "pnl_infiltrate", "fibrosis_papillary_dermis",
        "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_rete_ridges",
        "elongation_rete_ridges", "thinning_suprapapillary_epidermis",
        "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
        "disappearance_granular_layer", "vacuolisation_basal_layer",
        "spongiosis", "saw_tooth_appearance", "follicular_horn_plug",
        "perifollicular_parakeratosis", "inflammatory_monoluclear_infiltrate",
        "band_like_infiltrate", "age", "class",
    ]
    df = pd.read_csv(raw_path, header=None, names=columns, na_values="?")

    # Binary target: class 1 (psoriasis) vs rest
    df["y"] = (df["class"] == 1).astype(int)
    df = df.drop(columns=["class"])

    for col in df.columns:
        if col != "y":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(how="all", subset=[c for c in df.columns if c != "y"]).reset_index(drop=True)
    df = add_patient_id_and_time(df, seed=47)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (psoriasis): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")


def prepare_pima(output: Path) -> None:
    """Pima Indians Diabetes — 768 patients, 8 features."""
    print("\n=== Pima Indians Diabetes ===")
    print("  Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    print("  Rows: 768 | Features: 8 | Task: predict diabetes onset")

    raw_path = output.parent / ".pima_raw.csv"
    download_file(URLS["pima"], raw_path)

    columns = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "y",
    ]
    df = pd.read_csv(raw_path, header=None, names=columns)

    # Replace biologically impossible zeros with NaN for certain columns
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        df.loc[df[col] == 0, col] = np.nan

    df = add_patient_id_and_time(df, seed=48)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (diabetes): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")


def prepare_mammographic(output: Path) -> None:
    """UCI Mammographic Mass — ~961 patients, 5 features."""
    print("\n=== UCI Mammographic Mass ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass")
    print("  Rows: ~961 | Features: 5 | Task: predict malignancy of mammographic mass")

    raw_path = output.parent / ".mammographic_raw.data"
    download_file(URLS["mammographic"], raw_path)

    columns = ["bi_rads", "age", "shape", "margin", "density", "severity"]
    df = pd.read_csv(raw_path, header=None, names=columns, na_values="?")

    # Binary target: severity (0=benign, 1=malignant)
    df["y"] = df["severity"].astype(float)
    df = df.drop(columns=["severity"])

    # Convert all to numeric
    for col in df.columns:
        if col != "y":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["y"]).reset_index(drop=True)
    df["y"] = df["y"].astype(int)

    df = add_patient_id_and_time(df, seed=49)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (malignant): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")


def prepare_thyroid(output: Path) -> None:
    """UCI Thyroid Disease (ANN) — 7,200 patients, 21 features."""
    print("\n=== UCI Thyroid Disease (ANN) ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/thyroid+disease")
    print("  Rows: ~7,200 | Features: 21 | Task: predict thyroid dysfunction")

    dfs = []
    for part, url_key in [("train", "thyroid_train"), ("test", "thyroid_test")]:
        raw_path = output.parent / f".thyroid_{part}_raw.data"
        download_file(URLS[url_key], raw_path)
        df = pd.read_csv(raw_path, sep=r"\s+", header=None)
        dfs.append(df)
        if raw_path.exists():
            raw_path.unlink()

    df = pd.concat(dfs, ignore_index=True)
    # 22 columns: 21 features + 1 target (last column)
    # Target: 1=normal, 2=hyperthyroid, 3=hypothyroid → binary: 1=normal(0), 2/3=abnormal(1)
    target_col_idx = df.shape[1] - 1
    feature_cols_raw = [f"thyroid_f{i}" for i in range(target_col_idx)]
    df.columns = feature_cols_raw + ["_target"]
    # Target: 1=normal, 2=hyperthyroid, 3=hypothyroid
    # Use hyperthyroid (class 2) as positive class — rare condition (~2.3%)
    df["y"] = (df["_target"] == 2).astype(int)
    df = df.drop(columns=["_target"])

    df = add_patient_id_and_time(df, seed=53)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (abnormal): {pos} ({pos/len(df)*100:.1f}%) | Negative (normal): {neg}")


def prepare_eeg_eye(output: Path) -> None:
    """UCI EEG Eye State — 14,980 observations, 14 EEG features."""
    print("\n=== UCI EEG Eye State ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State")
    print("  Rows: ~14,980 | Features: 14 | Task: predict eye state (open/closed) from EEG")

    raw_path = output.parent / ".eeg_eye_raw.arff"
    download_file(URLS["eeg_eye"], raw_path)

    # Parse ARFF
    columns: List[str] = []
    data_lines: List[str] = []
    in_data = False
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.lower().startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    columns.append(parts[1].strip("'\""))
            elif line.lower() == "@data":
                in_data = True
            elif in_data and line and not line.startswith("%"):
                data_lines.append(line)

    csv_text = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text), header=None, names=columns)

    # Last column is eyeDetection (0=open, 1=closed)
    target_col_name = columns[-1]
    df = df.rename(columns={target_col_name: "y"})
    df["y"] = df["y"].astype(int)

    # Rename EEG channels to cleaner names
    eeg_cols = [c for c in df.columns if c != "y"]
    rename_map = {old: f"eeg_{i+1}" for i, old in enumerate(eeg_cols)}
    df = df.rename(columns=rename_map)

    df = add_patient_id_and_time(df, seed=54)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (eye closed): {pos} ({pos/len(df)*100:.1f}%) | Negative (eye open): {neg}")


def prepare_framingham(output: Path) -> None:
    """Framingham Heart Study — 4,240 patients, 15 clinical features."""
    print("\n=== Framingham Heart Study ===")
    print("  Source: https://www.framinghamheartstudy.org/")
    print("  Rows: ~4,240 | Features: 15 | Task: predict 10-year coronary heart disease")

    raw_path = output.parent / ".framingham_raw.csv"
    download_file(URLS["framingham"], raw_path)

    df = pd.read_csv(raw_path)

    # Rename target
    df = df.rename(columns={"TenYearCHD": "y"})

    # Convert all to numeric
    for col in df.columns:
        if col != "y":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_patient_id_and_time(df, seed=51)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if raw_path.exists():
        raw_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (CHD): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")


def prepare_diabetes130(output: Path, max_rows: int = 10000) -> None:
    """UCI Diabetes 130-US Hospitals — up to 101,766 patients, 20+ features.

    Downloads ZIP from UCI, extracts diabetic_data.csv, binarizes readmission target,
    and optionally subsamples to max_rows for manageable play-mode training times.
    """
    n_label = f"{max_rows:,}" if max_rows > 0 else "all (~101K)"
    print(f"\n=== UCI Diabetes 130-US Hospitals (subsample: {n_label} rows) ===")
    print("  Source: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals")
    print(f"  Features: ~20 numeric/encoded | Task: predict hospital readmission <30 days")

    zip_path = output.parent / ".diabetes130_raw.zip"
    download_file(URLS["diabetes130_zip"], zip_path)

    with zipfile.ZipFile(zip_path) as z:
        with z.open("dataset_diabetes/diabetic_data.csv") as f:
            df = pd.read_csv(f)

    # Binary target: readmitted <30 days = 1, else = 0
    df["y"] = (df["readmitted"] == "<30").astype(int)

    # Select useful numeric/categorical columns
    keep_cols = [
        "y", "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ]
    # Encode age bracket as numeric midpoint
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_midpoint"] = df["age"].map(age_map)
    # Encode gender
    df["is_female"] = (df["gender"] == "Female").astype(int)
    # Encode race as dummies for top categories
    df["race_caucasian"] = (df["race"] == "Caucasian").astype(int)
    df["race_african_american"] = (df["race"] == "AfricanAmerican").astype(int)
    # A1C result
    df["A1Cresult_high"] = (df["A1Cresult"].isin([">7", ">8"])).astype(int)
    # Insulin change
    df["insulin_yes"] = (df["insulin"].isin(["Up", "Down", "Steady"])).astype(int)
    # Diabetic med changed
    df["change_yes"] = (df["change"] == "Ch").astype(int)
    # Discharge disposition (home=1)
    df["discharged_home"] = (df["discharge_disposition_id"] == 1).astype(int)

    extra_cols = [
        "age_midpoint", "is_female", "race_caucasian", "race_african_american",
        "A1Cresult_high", "insulin_yes", "change_yes", "discharged_home",
    ]
    final_cols = keep_cols + extra_cols
    df = df[[c for c in final_cols if c in df.columns]].copy()

    # Drop rows with all-NaN features
    feature_cols_check = [c for c in df.columns if c != "y"]
    df = df.dropna(subset=feature_cols_check, how="all").reset_index(drop=True)

    # Subsample if requested
    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=52).reset_index(drop=True)

    df = add_patient_id_and_time(df, seed=52)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    if zip_path.exists():
        zip_path.unlink()
    pos = int(df["y"].sum())
    neg = len(df) - pos
    print(f"  Output: {output}")
    print(f"  Rows: {len(df)} | Positive (readmit <30d): {pos} ({pos/len(df)*100:.1f}%) | Negative: {neg}")


def prepare_synth_large(output: Path, n_rows: int = 5000) -> None:
    """Synthetic large medical dataset — configurable row count, 15 features."""
    print(f"\n=== Synthetic Large Medical Dataset ({n_rows} rows) ===")
    print(f"  Rows: {n_rows} | Features: 15 | Task: predict adverse outcome (synthetic)")

    rng = np.random.default_rng(50)
    n = n_rows

    age = rng.normal(55, 15, n).clip(18, 95).astype(int)
    sex = rng.choice([0, 1], n)
    bmi = rng.normal(27, 5, n).clip(15, 50).round(1)
    bp_systolic = rng.normal(130, 20, n).clip(80, 200).astype(int)
    bp_diastolic = rng.normal(80, 12, n).clip(50, 120).astype(int)
    heart_rate = rng.normal(75, 12, n).clip(45, 130).astype(int)
    glucose = rng.normal(100, 30, n).clip(50, 300).round(1)
    cholesterol = rng.normal(200, 40, n).clip(100, 400).astype(int)
    creatinine = rng.lognormal(0.0, 0.4, n).clip(0.3, 5.0).round(2)
    hemoglobin = rng.normal(13.5, 2.0, n).clip(7, 18).round(1)
    platelets = rng.normal(250, 60, n).clip(50, 500).astype(int)
    wbc = rng.normal(7.0, 2.0, n).clip(2, 20).round(1)
    smoking = rng.choice([0, 1], n, p=[0.7, 0.3])
    diabetes_history = rng.choice([0, 1], n, p=[0.8, 0.2])
    family_history = rng.choice([0, 1], n, p=[0.75, 0.25])

    logit = (
        -3.5
        + 0.03 * (age - 55)
        + 0.4 * sex
        + 0.05 * (bmi - 27)
        + 0.02 * (bp_systolic - 130)
        + 0.01 * (glucose - 100)
        + 0.005 * (cholesterol - 200)
        + 0.3 * creatinine
        - 0.1 * (hemoglobin - 13.5)
        + 0.5 * smoking
        + 0.6 * diabetes_history
        + 0.3 * family_history
        + rng.normal(0, 0.5, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi,
        "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic,
        "heart_rate": heart_rate, "glucose": glucose,
        "cholesterol": cholesterol.astype(float), "creatinine": creatinine,
        "hemoglobin": hemoglobin, "platelets": platelets,
        "wbc": wbc, "smoking": smoking,
        "diabetes_history": diabetes_history,
        "family_history": family_history, "y": y,
    })
    # Add ~5% missing values to some columns
    for col in ["glucose", "cholesterol", "creatinine", "hemoglobin"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan
    df = add_patient_id_and_time(df, seed=50)
    feature_cols = [c for c in df.columns if c not in ("patient_id", "event_time", "y")]
    df = df[["patient_id", "event_time", "y"] + feature_cols]

    df.to_csv(output, index=False)
    pos = int(df["y"].sum())
    neg = n - pos
    print(f"  Output: {output}")
    print(f"  Rows: {n} | Positive: {pos} ({pos/n*100:.1f}%) | Negative: {neg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare real UCI medical datasets for ml-leakage-guard pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        choices=["heart", "breast", "ckd", "hepatitis", "spect", "dermatology", "pima", "mammographic", "thyroid", "eeg_eye", "framingham", "diabetes130", "all"],
        help="Dataset to prepare. framingham=4240 rows, diabetes130=10K subsample of 101K real hospital records.",
    )
    parser.add_argument("--output", default="", help="Output CSV path (default: examples/<dataset>.csv).")
    args = parser.parse_args()

    examples_dir = SCRIPT_DIR
    examples_dir.mkdir(parents=True, exist_ok=True)

    PREPARE = {
        "heart": ("heart_disease.csv", prepare_heart),
        "breast": ("breast_cancer.csv", prepare_breast),
        "ckd": ("chronic_kidney_disease.csv", prepare_ckd),
        "hepatitis": ("hepatitis.csv", prepare_hepatitis),
        "spect": ("spect_heart.csv", prepare_spect),
        "dermatology": ("dermatology.csv", prepare_dermatology),
        "pima": ("pima_diabetes.csv", prepare_pima),
        "mammographic": ("mammographic_mass.csv", prepare_mammographic),
        "thyroid": ("thyroid_disease.csv", prepare_thyroid),
        "eeg_eye": ("eeg_eye_state.csv", prepare_eeg_eye),
        "framingham": ("framingham_heart.csv", prepare_framingham),
        "diabetes130": ("diabetes130_readmission.csv", lambda o: prepare_diabetes130(o, max_rows=10000)),
    }

    if args.dataset == "all":
        for name, (default_file, fn) in PREPARE.items():
            fn(examples_dir / default_file)
    else:
        default_file, fn = PREPARE[args.dataset]
        out = Path(args.output) if args.output else examples_dir / default_file
        fn(out)

    print("\n✓ Done! Use the commands above to split and run the pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
