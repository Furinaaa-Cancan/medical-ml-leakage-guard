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
    "hepatitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "spect": "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train",
    "spect_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test",
    "dermatology": "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
    "pima": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "mammographic": "https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data",
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
        choices=["heart", "breast", "ckd", "hepatitis", "spect", "dermatology", "pima", "mammographic", "synth5k", "synth10k", "all"],
        help="Dataset to prepare. synth5k/synth10k = synthetic large datasets (5000/10000 rows).",
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
        "synth5k": ("synth_medical_5k.csv", lambda o: prepare_synth_large(o, n_rows=5000)),
        "synth10k": ("synth_medical_10k.csv", lambda o: prepare_synth_large(o, n_rows=10000)),
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
