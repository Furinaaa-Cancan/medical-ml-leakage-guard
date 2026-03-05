#!/usr/bin/env python3
"""
Generate reproducible offline demo medical binary-classification datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic leakage-safe medical demo splits for onboarding."
    )
    parser.add_argument("--project-root", required=True, help="Project root containing data/ directory.")
    parser.add_argument("--seed", type=int, default=20260227, help="Random seed for deterministic generation.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def random_dates(rng: np.random.Generator, start: str, end: str, n: int) -> np.ndarray:
    start_ts = np.datetime64(start)
    end_ts = np.datetime64(end)
    day_span = int((end_ts - start_ts).astype("timedelta64[D]").astype(int))
    offsets = rng.integers(0, max(1, day_span + 1), size=n, endpoint=False)
    return (start_ts + offsets.astype("timedelta64[D]")).astype("datetime64[D]")


def make_cohort(
    rng: np.random.Generator,
    n_rows: int,
    patient_id_start: int,
    date_start: str,
    date_end: str,
    shifts: Dict[str, float],
) -> pd.DataFrame:
    age = np.clip(rng.normal(60.0 + shifts.get("age", 0.0), 12.0, n_rows), 18.0, 90.0)
    sex_male = rng.binomial(1, np.clip(0.52 + shifts.get("sex_male", 0.0), 0.05, 0.95), n_rows)
    bmi = np.clip(rng.normal(27.0 + shifts.get("bmi", 0.0), 4.2, n_rows), 16.0, 45.0)
    systolic_bp = np.clip(rng.normal(128.0 + shifts.get("systolic_bp", 0.0), 16.0, n_rows), 80.0, 220.0)
    heart_rate = np.clip(rng.normal(78.0 + shifts.get("heart_rate", 0.0), 12.0, n_rows), 40.0, 170.0)
    wbc = np.clip(rng.normal(7.4 + shifts.get("wbc", 0.0), 1.8, n_rows), 2.0, 20.0)
    creatinine = np.clip(rng.normal(1.0 + shifts.get("creatinine", 0.0), 0.35, n_rows), 0.4, 4.0)
    lactate = np.clip(rng.normal(1.6 + shifts.get("lactate", 0.0), 0.7, n_rows), 0.2, 8.0)
    crp = np.clip(rng.normal(4.0 + shifts.get("crp", 0.0), 2.2, n_rows), 0.2, 40.0)
    comorbidity_index = np.clip(rng.poisson(2.0 + shifts.get("comorbidity_index", 0.0), n_rows), 0, 12)
    smoke_status = rng.binomial(1, np.clip(0.28 + shifts.get("smoke_status", 0.0), 0.02, 0.8), n_rows)

    logits = (
        -7.3
        + 0.2 * (age - 55.0)
        + 0.8 * sex_male
        + 0.15 * (bmi - 27.0)
        + 0.08 * (systolic_bp - 125.0)
        + 0.08 * (heart_rate - 75.0)
        + 1.3 * (wbc - 7.0)
        + 2.5 * (creatinine - 1.0)
        + 2.0 * (lactate - 1.5)
        + 0.3 * (crp - 4.0)
        + 1.1 * comorbidity_index
        + 0.7 * smoke_status
        + rng.normal(0.0, 0.03, n_rows)
    )
    prob = sigmoid(logits)
    y = rng.binomial(1, prob, n_rows)

    event_time = random_dates(rng=rng, start=date_start, end=date_end, n=n_rows)
    patient_ids = np.arange(patient_id_start, patient_id_start + n_rows, dtype=int)

    df = pd.DataFrame(
        {
            "patient_id": patient_ids.astype(str),
            "event_time": pd.to_datetime(event_time).strftime("%Y-%m-%d"),
            "y": y.astype(int),
            "age": np.round(age, 2),
            "sex_male": sex_male.astype(int),
            "bmi": np.round(bmi, 2),
            "systolic_bp": np.round(systolic_bp, 2),
            "heart_rate": np.round(heart_rate, 2),
            "wbc": np.round(wbc, 3),
            "creatinine": np.round(creatinine, 3),
            "lactate": np.round(lactate, 3),
            "crp": np.round(crp, 3),
            "comorbidity_index": comorbidity_index.astype(int),
            "smoke_status": smoke_status.astype(int),
        }
    )
    return df.sort_values(["event_time", "patient_id"]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def split_summary(df: pd.DataFrame) -> Dict[str, Any]:
    pos = int(df["y"].sum())
    rows = int(df.shape[0])
    return {
        "rows": rows,
        "positive_count": pos,
        "negative_count": int(rows - pos),
        "positive_rate": float(pos / rows) if rows > 0 else 0.0,
        "min_date": str(df["event_time"].min()),
        "max_date": str(df["event_time"].max()),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    train = make_cohort(
        rng=rng,
        n_rows=840,
        patient_id_start=100000,
        date_start="2024-01-01",
        date_end="2024-06-30",
        shifts={},
    )
    valid = make_cohort(
        rng=rng,
        n_rows=420,
        patient_id_start=200000,
        date_start="2024-07-01",
        date_end="2024-08-31",
        shifts={},
    )
    test = make_cohort(
        rng=rng,
        n_rows=320,
        patient_id_start=300000,
        date_start="2024-09-01",
        date_end="2024-10-31",
        shifts={"age": 0.3, "wbc": 0.03},
    )
    external_period = make_cohort(
        rng=rng,
        n_rows=220,
        patient_id_start=400000,
        date_start="2025-10-01",
        date_end="2025-12-31",
        shifts={"age": 0.7, "wbc": 0.1, "lactate": 0.03},
    )
    external_site = make_cohort(
        rng=rng,
        n_rows=220,
        patient_id_start=500000,
        date_start="2024-11-01",
        date_end="2025-01-31",
        shifts={"bmi": 0.4, "creatinine": 0.03, "smoke_status": 0.015},
    )

    paths = {
        "train": data_dir / "train.csv",
        "valid": data_dir / "valid.csv",
        "test": data_dir / "test.csv",
        "external_2025_q4": data_dir / "external_2025_q4.csv",
        "external_site_b": data_dir / "external_site_b.csv",
    }
    save_csv(train, paths["train"])
    save_csv(valid, paths["valid"])
    save_csv(test, paths["test"])
    save_csv(external_period, paths["external_2025_q4"])
    save_csv(external_site, paths["external_site_b"])

    report = {
        "status": "pass",
        "seed": seed,
        "project_root": str(project_root),
        "files": {name: str(path) for name, path in paths.items()},
        "summary": {
            "train": split_summary(train),
            "valid": split_summary(valid),
            "test": split_summary(test),
            "external_2025_q4": split_summary(external_period),
            "external_site_b": split_summary(external_site),
        },
    }

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else (project_root / "evidence" / "demo_dataset_report.json").resolve()
    )
    from _gate_utils import write_json as _write_report
    _write_report(report_path, report)

    print("Status: pass")
    print(f"ProjectRoot: {project_root}")
    print(f"TrainCSV: {paths['train']}")
    print(f"ValidCSV: {paths['valid']}")
    print(f"TestCSV: {paths['test']}")
    print(f"ExternalPeriodCSV: {paths['external_2025_q4']}")
    print(f"ExternalSiteCSV: {paths['external_site_b']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
