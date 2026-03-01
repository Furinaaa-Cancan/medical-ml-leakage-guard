#!/usr/bin/env python3
"""
Full Matrix Integration Test for ML Leakage Guard.

Test matrix: 3 datasets × 3 split strategies × 2 tuning strategies = 18 combinations.
Each combination: download data → split → train (2 minimal models) → verify output.

Usage:
    pytest tests/test_full_matrix.py -v --timeout=3600
    pytest tests/test_full_matrix.py -v -k "heart and grouped" --timeout=600
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
PYTHON = sys.executable

DATASETS = ["heart", "breast", "ckd"]
SPLIT_STRATEGIES = ["grouped_temporal", "grouped_random", "stratified_grouped"]
TUNING_STRATEGIES = ["fixed_grid", "random_subsample"]
MINIMAL_MODELS = "logistic_l1,logistic_l2"


def _run(cmd: List[str], cwd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a subprocess with timeout."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="session")
def dataset_cache(tmp_path_factory):
    """Download datasets once per session."""
    cache_dir = tmp_path_factory.mktemp("datasets")
    downloaded = {}
    for ds in DATASETS:
        csv_path = cache_dir / f"{ds}.csv"
        result = _run(
            [PYTHON, str(EXAMPLES_DIR / "download_real_data.py"), ds, "--output", str(csv_path)],
            cwd=str(cache_dir),
            timeout=120,
        )
        if result.returncode == 0 and csv_path.exists():
            downloaded[ds] = csv_path
        else:
            downloaded[ds] = None
    return downloaded


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("strategy", SPLIT_STRATEGIES)
@pytest.mark.parametrize("tuning", TUNING_STRATEGIES)
def test_full_pipeline(dataset, strategy, tuning, dataset_cache, tmp_path):
    """Run split → train → verify for a dataset/strategy/tuning combination."""
    csv_path = dataset_cache.get(dataset)
    if csv_path is None or not csv_path.exists():
        pytest.skip(f"Dataset {dataset} not available (download failed or requires local files)")

    work_dir = tmp_path / f"{dataset}_{strategy}_{tuning}"
    data_dir = work_dir / "data"
    evidence_dir = work_dir / "evidence"
    data_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)

    # Step 1: Split
    split_cmd = [
        PYTHON, str(SCRIPTS_DIR / "split_data.py"),
        "--input", str(csv_path),
        "--output-dir", str(data_dir),
        "--patient-id-col", "patient_id",
        "--target-col", "y",
        "--time-col", "event_time",
        "--strategy", strategy,
    ]
    result = _run(split_cmd, cwd=str(work_dir), timeout=120)
    assert result.returncode == 0, f"Split failed:\n{result.stderr}"

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    assert train_csv.exists(), "train.csv not created"
    assert test_csv.exists(), "test.csv not created"

    # Step 2: Train
    train_cmd = [
        PYTHON, str(SCRIPTS_DIR / "train_select_evaluate.py"),
        "--train", str(train_csv),
        "--test", str(test_csv),
        "--target-col", "y",
        "--patient-id-col", "patient_id",
        "--ignore-cols", "patient_id,event_time",
        "--model-pool", MINIMAL_MODELS,
        "--hyperparam-search", tuning,
        "--max-trials-per-family", "3",
        "--cv-splits", "3",
        "--model-selection-report-out", str(evidence_dir / "model_selection_report.json"),
        "--evaluation-report-out", str(evidence_dir / "evaluation_report.json"),
    ]
    # Add valid split if it exists
    valid_csv = data_dir / "valid.csv"
    if valid_csv.exists():
        train_cmd.extend(["--valid", str(valid_csv)])

    result = _run(train_cmd, cwd=str(work_dir), timeout=1200)
    assert result.returncode == 0, f"Training failed:\n{result.stderr[-2000:]}"

    # Step 3: Verify outputs
    eval_report = evidence_dir / "evaluation_report.json"
    ms_report = evidence_dir / "model_selection_report.json"

    assert eval_report.exists(), "evaluation_report.json not created"
    assert ms_report.exists(), "model_selection_report.json not created"

    # Validate JSON structure
    with eval_report.open() as f:
        eval_data = json.load(f)
    assert isinstance(eval_data, dict), "evaluation_report.json is not a dict"
    assert eval_data.get("model_id") is not None or eval_data.get("primary_metric") is not None, \
        "evaluation_report.json missing expected fields"

    with ms_report.open() as f:
        ms_data = json.load(f)
    assert isinstance(ms_data, dict), "model_selection_report.json is not a dict"


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize("dataset", ["heart"])
def test_single_smoke(dataset, dataset_cache, tmp_path):
    """Quick smoke test with a single dataset, default strategy."""
    csv_path = dataset_cache.get(dataset)
    if csv_path is None or not csv_path.exists():
        pytest.skip(f"Dataset {dataset} not available")

    data_dir = tmp_path / "data"
    evidence_dir = tmp_path / "evidence"
    data_dir.mkdir()
    evidence_dir.mkdir()

    # Split
    result = _run([
        PYTHON, str(SCRIPTS_DIR / "split_data.py"),
        "--input", str(csv_path),
        "--output-dir", str(data_dir),
        "--patient-id-col", "patient_id",
        "--target-col", "y",
        "--time-col", "event_time",
        "--strategy", "grouped_temporal",
    ], cwd=str(tmp_path))
    assert result.returncode == 0, f"Split failed: {result.stderr}"

    # Train with single model
    train_cmd = [
        PYTHON, str(SCRIPTS_DIR / "train_select_evaluate.py"),
        "--train", str(data_dir / "train.csv"),
        "--test", str(data_dir / "test.csv"),
        "--target-col", "y",
        "--patient-id-col", "patient_id",
        "--ignore-cols", "patient_id,event_time",
        "--model-pool", "logistic_l1",
        "--hyperparam-search", "fixed_grid",
        "--cv-splits", "3",
        "--model-selection-report-out", str(evidence_dir / "model_selection_report.json"),
        "--evaluation-report-out", str(evidence_dir / "evaluation_report.json"),
    ]
    valid_csv = data_dir / "valid.csv"
    if valid_csv.exists():
        train_cmd.extend(["--valid", str(valid_csv)])

    result = _run(train_cmd, cwd=str(tmp_path), timeout=300)
    assert result.returncode == 0, f"Training failed: {result.stderr[-1000:]}"
    assert (evidence_dir / "evaluation_report.json").exists()
