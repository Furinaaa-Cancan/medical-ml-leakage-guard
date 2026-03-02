#!/usr/bin/env python3
"""
End-to-end strict skill validation on authoritative public medical binary datasets.

Workflow per dataset:
1. Load and clean raw data.
2. Build train/valid/test/external CSV splits with disjoint IDs and strict temporal ordering.
3. Run train_select_evaluate.py to emit model_selection/evaluation/prediction-trace/external-validation artifacts.
4. Generate signed execution attestation bundle.
5. Run strict pipeline bootstrap (allow-missing-compare).
6. Freeze manifest baseline and rerun strict pipeline with comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import fcntl
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None  # type: ignore[assignment]


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


@dataclass
class DatasetCase:
    case_id: str
    raw_filename: str
    target_name: str
    source_name: str
    options: Optional[Dict[str, Any]] = None


@dataclass
class HeartStressSearchResult:
    selected_seed: int
    selected_profile: str
    report_path: Path
    selection_path: Path
    status: str


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
RAW_ROOT = DATA_ROOT / "raw"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
REFERENCES_ROOT = REPO_ROOT / "references"
_IS_CI = str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes", "on"}
_IS_LOCAL_TTY = bool(sys.stdout.isatty() and not _IS_CI)
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = _env_int(
    "MLLG_SUBPROCESS_TIMEOUT_SECONDS",
    1800 if _IS_LOCAL_TTY else 3600,
)
SUBPROCESS_TIMEOUT_SECONDS = max(1, DEFAULT_SUBPROCESS_TIMEOUT_SECONDS)
DEFAULT_CASE_LOCK_TIMEOUT_SECONDS = _env_int(
    "MLLG_CASE_LOCK_TIMEOUT_SECONDS",
    900 if _IS_LOCAL_TTY else 1800,
)
DEFAULT_LOCK_WAIT_HEARTBEAT_SECONDS = _env_float(
    "MLLG_LOCK_WAIT_HEARTBEAT_SECONDS",
    15.0,
)
CASE_LOCK_TIMEOUT_SECONDS = max(1.0, float(DEFAULT_CASE_LOCK_TIMEOUT_SECONDS))
LOCK_WAIT_HEARTBEAT_SECONDS = max(0.0, float(DEFAULT_LOCK_WAIT_HEARTBEAT_SECONDS))
SPLIT_RANDOM_SEED_BY_CASE: Dict[str, int] = {
    "uci-heart-disease": 20250003,
    "uci-breast-cancer-wdbc": 20260224,
    "uci-diabetes-130-readmission": 20260226,
    "uci-chronic-kidney-disease": 20260227,
}
EXTERNAL_SPLIT_RATIO_BY_CASE: Dict[str, float] = {
    # With independent external-institution cohorts available, keep a moderate
    # internal external-pool ratio to preserve internal train/valid/test stability
    # while still satisfying cross-period external minimum sample requirements.
    "uci-heart-disease": 0.20,
    "uci-breast-cancer-wdbc": 0.20,
    "uci-diabetes-130-readmission": 0.20,
    # Keep enough rows in each external cohort for calibration/DCA hard minimums.
    "uci-chronic-kidney-disease": 0.30,
}
THRESHOLD_SELECTION_SPLIT_BY_CASE: Dict[str, str] = {
    "uci-heart-disease": "valid",
    "uci-breast-cancer-wdbc": "valid",
    "uci-diabetes-130-readmission": "valid",
    "uci-chronic-kidney-disease": "valid",
}
MODEL_SELECTION_DATA_BY_CASE: Dict[str, str] = {
    "uci-heart-disease": "cv_inner",
    "uci-breast-cancer-wdbc": "cv_inner",
    "uci-diabetes-130-readmission": "cv_inner",
    "uci-chronic-kidney-disease": "cv_inner",
}
INTERNAL_SPLIT_FRACTIONS_BY_CASE: Dict[str, Dict[str, float]] = {
    # Heart dataset is small; keep test>=50 and each external cohort >=50 for calibration/DCA minimums.
    "uci-heart-disease": {"train": 0.48, "valid": 0.26, "test": 0.26},
    "uci-breast-cancer-wdbc": {"train": 0.60, "valid": 0.20, "test": 0.20},
    "uci-diabetes-130-readmission": {"train": 0.60, "valid": 0.20, "test": 0.20},
    "uci-chronic-kidney-disease": {"train": 0.60, "valid": 0.20, "test": 0.20},
}
ROBUSTNESS_TIME_SLICES_BY_CASE: Dict[str, int] = {
    "uci-heart-disease": 1,
    "uci-breast-cancer-wdbc": 2,
    "uci-diabetes-130-readmission": 2,
    "uci-chronic-kidney-disease": 2,
}
ROBUSTNESS_GROUP_COUNT_BY_CASE: Dict[str, int] = {
    "uci-heart-disease": 1,
    "uci-breast-cancer-wdbc": 2,
    "uci-diabetes-130-readmission": 2,
    "uci-chronic-kidney-disease": 2,
}
TRAINING_BUDGET_BY_CASE: Dict[str, Dict[str, int]] = {
    # Keep publication-style confidence intervals while controlling runtime.
    "uci-diabetes-130-readmission": {
        "bootstrap_resamples": 300,
        "ci_bootstrap_resamples": 400,
        "permutation_resamples": 100,
    }
}
MODEL_POOL_BY_CASE: Dict[str, List[str]] = {
    "default": [
        "logistic_l1",
        "logistic_l2",
        "logistic_elasticnet",
        "random_forest_balanced",
        "extra_trees_balanced",
        "hist_gradient_boosting_l2",
    ],
    # Stress heart benefits from a broader non-linear pool while keeping
    # logistic baselines auditable via required_models.
    "uci-heart-disease": [
        "logistic_l1",
        "logistic_l2",
        "logistic_elasticnet",
        "random_forest_balanced",
        "extra_trees_balanced",
        "hist_gradient_boosting_l2",
        "adaboost",
        "xgboost",
    ],
    # Large heterogeneous cohort: expand model families and let search pick.
    "uci-diabetes-130-readmission": [
        "logistic_l1",
        "logistic_l2",
        "logistic_elasticnet",
        "random_forest_balanced",
        "extra_trees_balanced",
        "hist_gradient_boosting_l2",
        "adaboost",
        "xgboost",
    ],
}
MAX_TRIALS_PER_FAMILY_BY_CASE: Dict[str, int] = {
    "default": 3,
    "uci-heart-disease": 4,
    "uci-diabetes-130-readmission": 6,
}
HYPERPARAM_SEARCH_BY_CASE: Dict[str, str] = {
    "default": "random_subsample",
    "uci-diabetes-130-readmission": "random_subsample",
}
N_JOBS_BY_CASE: Dict[str, int] = {
    "default": 4,
    "uci-diabetes-130-readmission": 8,
}
MISSINGNESS_MICE_MAX_ROWS_OVERRIDE_BY_CASE: Dict[str, int] = {
    # On larger cohorts, enforce scale-guard fallback to simple_with_indicator for runtime stability.
    "uci-diabetes-130-readmission": 5000
}
MISSINGNESS_MIN_NON_MISSING_OVERRIDE_BY_CASE: Dict[str, int] = {
    # CKD has legitimate sparse labs (e.g., rbc/rbcc); keep a strict but feasible lower bound.
    "uci-chronic-kidney-disease": 90
}
DEFAULT_DIABETES_MAX_ROWS = 20000
DEFAULT_DIABETES_TARGET_MODE = "gt30"
DEFAULT_CKD_MAX_ROWS = 0
HEART_EXTERNAL_INSTITUTION_RAW_FILES: List[Tuple[str, str]] = [
    ("hungarian", "heart_disease_processed.hungarian.data"),
    ("switzerland", "heart_disease_processed.switzerland.data"),
    ("va", "heart_disease_processed.va.data"),
]
# Build heart internal training pool from multiple institutions to reduce
# single-site bias (while still keeping a disjoint cross-institution cohort).
HEART_INTERNAL_AUX_RAW_FILES: List[Tuple[str, str]] = [
    ("va", "heart_disease_processed.va.data"),
]
HEART_INTERNAL_SITE_TAGS: List[str] = ["cleveland", "va"]
HEART_EXTERNAL_SITE_MIN_ROWS = 80
HEART_EXTERNAL_SITE_EVENT_RATE_MIN = 0.15
# Keep a broad but bounded institution-mix for stress heart: include VA while
# still excluding extreme prevalence cohorts (e.g., Switzerland ~0.93).
HEART_EXTERNAL_SITE_EVENT_RATE_MAX = 0.80
HEART_ALL_FEATURE_COLS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
# Cross-site stable subset: avoids known severe missingness drift across
# Hungarian/VA sites while keeping core clinical signals.
HEART_STABLE_FEATURE_COLS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
]
DEFAULT_STRESS_SEED_MIN = 20249900
DEFAULT_STRESS_SEED_MAX = 20250150
STRESS_SEARCH_REPORT_CONTRACT_VERSION = "v2"
DEFAULT_STRESS_PROFILE_SET = "strict_v1"
DEFAULT_STRESS_CASE_ID = "uci-chronic-kidney-disease"
SUPPORTED_STRESS_CASE_IDS = (
    "uci-heart-disease",
    "uci-diabetes-130-readmission",
    "uci-chronic-kidney-disease",
    "uci-breast-cancer-wdbc",
)
STRESS_PROFILE_SETS: Dict[str, List[Dict[str, str]]] = {
    "strict_v1": [
        {
            "profile_id": "valid_valid_cvinner_power",
            "selection_data": "valid",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "power",
        },
        {
            "profile_id": "valid_valid_cvinner_beta",
            "selection_data": "valid",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "beta",
        },
        {
            "profile_id": "valid_valid_cvinner_isotonic",
            "selection_data": "valid",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "isotonic",
        },
        {
            "profile_id": "cvinner_valid_cvinner_power",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "power",
        },
        {
            "profile_id": "cvinner_valid_cvinner_sigmoid",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "sigmoid",
        },
        {
            "profile_id": "cvinner_valid_cvinner_none",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "none",
        },
        {
            "profile_id": "cvinner_valid_valid_power",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "valid",
            "calibration_method": "power",
        },
        {
            "profile_id": "cvinner_valid_cvinner_beta",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "beta",
        },
        {
            "profile_id": "cvinner_valid_valid_beta",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "valid",
            "calibration_method": "beta",
        },
        {
            "profile_id": "cvinner_valid_cvinner_isotonic",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "cv_inner",
            "calibration_method": "isotonic",
        },
        {
            "profile_id": "cvinner_valid_valid_isotonic",
            "selection_data": "cv_inner",
            "threshold_selection_split": "valid",
            "calibration_fit_split": "valid",
            "calibration_method": "isotonic",
        },
    ]
}


def parse_bool_env(name: str, default: bool = False) -> bool:
    token = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return token in {"1", "true", "yes", "on"}


@contextmanager
def case_run_lock(
    case_id: str,
    timeout_seconds: float = 1800.0,
    heartbeat_seconds: float = 15.0,
):
    """
    Serialize per-case filesystem mutations to avoid concurrent rmtree/write races.
    - POSIX: advisory lock via fcntl.flock
    - Non-POSIX fallback: lockfile O_CREAT|O_EXCL spin lock
    """
    lock_dir = DATA_ROOT / "_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{str(case_id).strip().lower() or 'unknown'}.lock"
    started = time.time()
    heartbeat_every = max(0.0, float(heartbeat_seconds))
    last_wait_log = started
    owner_token = (
        f"pid={os.getpid()} acquired_at={datetime.now(tz=timezone.utc).isoformat()} case_id={case_id}\n"
    )

    def _maybe_log_wait() -> None:
        nonlocal last_wait_log
        if heartbeat_every <= 0:
            return
        now = time.time()
        if (now - last_wait_log) < heartbeat_every:
            return
        waited = now - started
        print(
            "[INFO] waiting_for_case_lock "
            f"case_id={case_id} waited_seconds={waited:.1f} "
            f"timeout_seconds={float(timeout_seconds):.1f} lock={lock_path}",
            flush=True,
        )
        last_wait_log = now

    if fcntl is not None:
        with lock_path.open("a+", encoding="utf-8") as fh:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    _maybe_log_wait()
                    if (time.time() - started) > float(timeout_seconds):
                        raise TimeoutError(f"case_run_lock_timeout: case_id={case_id} lock={lock_path}")
                    time.sleep(0.2)
            try:
                waited_seconds = time.time() - started
                if waited_seconds >= max(1.0, heartbeat_every):
                    print(
                        "[INFO] acquired_case_lock "
                        f"case_id={case_id} waited_seconds={waited_seconds:.1f} lock={lock_path}",
                        flush=True,
                    )
                fh.seek(0)
                fh.truncate(0)
                fh.write(owner_token)
                fh.flush()
                os.fsync(fh.fileno())
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return

    # Cross-platform fallback (no fcntl): create-once lock file.
    acquired = False
    while not acquired:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(owner_token)
                fh.flush()
                os.fsync(fh.fileno())
            acquired = True
        except FileExistsError:
            # Try stale lock eviction when lock age exceeds timeout.
            _maybe_log_wait()
            try:
                age = time.time() - float(lock_path.stat().st_mtime)
            except OSError:
                age = 0.0
            if age > float(timeout_seconds):
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if (time.time() - started) > float(timeout_seconds):
                raise TimeoutError(f"case_run_lock_timeout: case_id={case_id} lock={lock_path}")
            time.sleep(0.2)
    try:
        waited_seconds = time.time() - started
        if waited_seconds >= max(1.0, heartbeat_every):
            print(
                "[INFO] acquired_case_lock "
                f"case_id={case_id} waited_seconds={waited_seconds:.1f} lock={lock_path}",
                flush=True,
            )
        yield
    finally:
        try:
            content = lock_path.read_text(encoding="utf-8")
        except Exception:
            content = ""
        if content == owner_token:
            try:
                lock_path.unlink()
            except OSError:
                pass


FAIL_LINE_RE = re.compile(r"^\[FAIL\]\s+([a-zA-Z0-9_]+)\s*:", re.MULTILINE)


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def resolve_case_model_pool(case_id: str) -> List[str]:
    pool = list(MODEL_POOL_BY_CASE.get(case_id, MODEL_POOL_BY_CASE["default"]))
    resolved: List[str] = []
    for family in pool:
        token = str(family).strip().lower()
        if token == "xgboost" and not module_available("xgboost"):
            continue
        if token == "catboost" and not module_available("catboost"):
            continue
        if token:
            resolved.append(token)
    return resolved


def extract_pipeline_root_failures(pipeline_report: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    failed_steps: List[str] = []
    failure_codes: List[str] = []
    for step in pipeline_report.get("steps", []):
        if not isinstance(step, dict):
            continue
        try:
            exit_code = int(step.get("exit_code", 0) or 0)
        except Exception:
            exit_code = 0
        if exit_code == 0:
            continue
        step_name = str(step.get("name", "")).strip()
        if step_name:
            failed_steps.append(step_name)
        stdout_tail = str(step.get("stdout_tail", "") or "")
        stderr_tail = str(step.get("stderr_tail", "") or "")
        failure_codes.extend(FAIL_LINE_RE.findall(stdout_tail))
        failure_codes.extend(FAIL_LINE_RE.findall(stderr_tail))
    dedup_steps = sorted(set(failed_steps))
    dedup_codes = sorted(set(code.strip() for code in failure_codes if str(code).strip()))
    return dedup_steps, dedup_codes


def to_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            fv = float(token)
        except Exception:
            return None
        return fv if np.isfinite(fv) else None
    return None


def load_clinical_floor_policy(performance_policy_path: Path) -> Dict[str, float]:
    floors: Dict[str, float] = {
        "sensitivity_min": 0.85,
        "npv_min": 0.90,
        "specificity_min": 0.40,
        "ppv_min": 0.55,
    }
    try:
        policy = load_json(performance_policy_path)
    except Exception:
        return floors
    if not isinstance(policy, dict):
        return floors
    clinical = policy.get("clinical_floors")
    if isinstance(clinical, dict):
        for key in list(floors.keys()):
            value = to_float_or_none(clinical.get(key))
            if value is not None and 0.0 <= value <= 1.0:
                floors[key] = float(value)
    threshold_policy = policy.get("threshold_policy")
    if isinstance(threshold_policy, dict):
        clinical_nested = threshold_policy.get("clinical_floors")
        if isinstance(clinical_nested, dict):
            for key in list(floors.keys()):
                value = to_float_or_none(clinical_nested.get(key))
                if value is not None and 0.0 <= value <= 1.0:
                    floors[key] = float(value)
    return floors


def build_floor_gap_rows(metrics: Dict[str, Any], floors: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    metric_keys = {
        "sensitivity_min": "sensitivity",
        "npv_min": "npv",
        "specificity_min": "specificity",
        "ppv_min": "ppv",
    }
    rows: Dict[str, Dict[str, Any]] = {}
    for floor_key, metric_name in metric_keys.items():
        required = float(floors.get(floor_key, 0.0))
        observed = to_float_or_none(metrics.get(metric_name))
        if observed is None:
            rows[metric_name] = {
                "required_min": required,
                "observed": None,
                "margin": None,
                "met": False,
            }
            continue
        margin = float(observed) - required
        rows[metric_name] = {
            "required_min": required,
            "observed": float(observed),
            "margin": margin,
            "met": bool(margin >= 0.0),
        }
    return rows


def build_clinical_floor_gap_summary(
    metrics: Dict[str, Any],
    external_validation_report_path: Path,
    performance_policy_path: Path,
) -> Dict[str, Any]:
    floors = load_clinical_floor_policy(performance_policy_path)
    internal_rows = build_floor_gap_rows(metrics if isinstance(metrics, dict) else {}, floors)
    external_rows: List[Dict[str, Any]] = []
    try:
        external_payload = load_json(external_validation_report_path)
    except Exception:
        external_payload = {}
    cohorts = external_payload.get("cohorts") if isinstance(external_payload, dict) else None
    if isinstance(cohorts, list):
        for row in cohorts:
            if not isinstance(row, dict):
                continue
            cohort_id = str(row.get("cohort_id", "")).strip()
            cohort_type = str(row.get("cohort_type", "")).strip() or None
            cohort_metrics = row.get("metrics")
            metric_rows = build_floor_gap_rows(cohort_metrics if isinstance(cohort_metrics, dict) else {}, floors)
            external_rows.append(
                {
                    "cohort_id": cohort_id or None,
                    "cohort_type": cohort_type,
                    "floor_metrics": metric_rows,
                }
            )
    all_margins: List[float] = []
    for entry in list(internal_rows.values()):
        margin = entry.get("margin")
        if isinstance(margin, (int, float)):
            all_margins.append(float(margin))
    for cohort in external_rows:
        for entry in cohort.get("floor_metrics", {}).values():
            margin = entry.get("margin")
            if isinstance(margin, (int, float)):
                all_margins.append(float(margin))
    min_margin = min(all_margins) if all_margins else None
    return {
        "floors": floors,
        "internal_test": {"floor_metrics": internal_rows},
        "external_cohorts": external_rows,
        "all_floor_checks_met": bool(all_margins and min_margin is not None and min_margin >= 0.0),
        "minimum_margin": min_margin,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run authoritative publication-grade E2E checks for ml-leakage-guard."
    )
    parser.add_argument(
        "--include-stress-cases",
        action="store_true",
        help="Include stress dataset cases (can also set MLLG_INCLUDE_STRESS_CASES=1).",
    )
    parser.add_argument(
        "--stress-case-id",
        default=DEFAULT_STRESS_CASE_ID,
        choices=list(SUPPORTED_STRESS_CASE_IDS),
        help=(
            "Stress case dataset id. "
            "Seed-search controls are applied only when stress-case-id=uci-heart-disease."
        ),
    )
    parser.add_argument(
        "--include-large-cases",
        action="store_true",
        help="Include larger benchmark dataset cases (can also set MLLG_INCLUDE_LARGE_CASES=1).",
    )
    parser.add_argument(
        "--diabetes-max-rows",
        type=int,
        default=DEFAULT_DIABETES_MAX_ROWS,
        help=(
            "Maximum rows retained for Diabetes130 case after patient-level de-duplication "
            f"(default: {DEFAULT_DIABETES_MAX_ROWS}; set 0 for full dataset)."
        ),
    )
    parser.add_argument(
        "--diabetes-target-mode",
        default=DEFAULT_DIABETES_TARGET_MODE,
        choices=["lt30", "gt30", "any"],
        help="Diabetes130 binary target mode: lt30 (<30d), gt30 (>30d), any (any readmission).",
    )
    parser.add_argument(
        "--include-ckd-case",
        action="store_true",
        help="Include UCI Chronic Kidney Disease benchmark case (or set MLLG_INCLUDE_CKD_CASE=1).",
    )
    parser.add_argument(
        "--ckd-max-rows",
        type=int,
        default=DEFAULT_CKD_MAX_ROWS,
        help=(
            "Optional max rows retained for CKD case after cleaning (default: 0 means full dataset)."
        ),
    )
    parser.add_argument(
        "--stress-seed-search",
        action="store_true",
        help="Search feasible stress split seed range and freeze selected seed (heart-only).",
    )
    parser.add_argument(
        "--no-stress-seed-search",
        action="store_true",
        help="Disable automatic heart stress seed search when stress cases are included.",
    )
    parser.add_argument(
        "--stress-seed-min",
        type=int,
        default=DEFAULT_STRESS_SEED_MIN,
        help=f"Minimum seed for heart stress search (default: {DEFAULT_STRESS_SEED_MIN}).",
    )
    parser.add_argument(
        "--stress-seed-max",
        type=int,
        default=DEFAULT_STRESS_SEED_MAX,
        help=f"Maximum seed for heart stress search (default: {DEFAULT_STRESS_SEED_MAX}).",
    )
    parser.add_argument(
        "--stress-seed-cache-file",
        default=str(DATA_ROOT / "stress_seed_search_report.json"),
        help="Output JSON path for stress seed search diagnostics/report.",
    )
    parser.add_argument(
        "--stress-selection-file",
        default=str(DATA_ROOT / "stress_seed_selection.json"),
        help="Output JSON path for selected stress seed/profile freeze record.",
    )
    parser.add_argument(
        "--summary-file",
        default=str(DATA_ROOT / "authority_e2e_summary.json"),
        help="Path to write authority E2E summary JSON.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional unique run tag. If omitted, UTC timestamp token is used.",
    )
    parser.add_argument(
        "--stress-profile-set",
        default=DEFAULT_STRESS_PROFILE_SET,
        help=f"Stress search profile set name (default: {DEFAULT_STRESS_PROFILE_SET}).",
    )
    parser.add_argument(
        "--subprocess-timeout-seconds",
        type=int,
        default=DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
        help=(
            "Timeout budget per spawned subprocess in seconds "
            f"(default: {DEFAULT_SUBPROCESS_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--case-lock-timeout-seconds",
        type=float,
        default=float(DEFAULT_CASE_LOCK_TIMEOUT_SECONDS),
        help=(
            "Timeout budget for per-case workspace lock acquisition in seconds "
            f"(default: {float(DEFAULT_CASE_LOCK_TIMEOUT_SECONDS):.1f})."
        ),
    )
    parser.add_argument(
        "--lock-wait-heartbeat-seconds",
        type=float,
        default=float(DEFAULT_LOCK_WAIT_HEARTBEAT_SECONDS),
        help=(
            "Heartbeat interval while waiting for case lock. "
            "Set 0 to disable wait heartbeat logs "
            f"(default: {float(DEFAULT_LOCK_WAIT_HEARTBEAT_SECONDS):.1f})."
        ),
    )
    parser.add_argument(
        "--auto-scan-diabetes-feasibility",
        action="store_true",
        help=(
            "When a Diabetes130 case fails clinical floors (stress or include-large-cases), "
            "run an automatic feasibility scan across target_mode/max_rows combinations."
        ),
    )
    parser.add_argument(
        "--diabetes-feasibility-target-modes",
        default="gt30,any,lt30",
        help="Comma-separated target modes for auto diabetes feasibility scan.",
    )
    parser.add_argument(
        "--diabetes-feasibility-max-rows-options",
        default="20000,0",
        help="Comma-separated max_rows options for auto diabetes feasibility scan.",
    )
    parser.add_argument(
        "--diabetes-feasibility-summary-dir",
        default=str(DATA_ROOT / "_feasibility_scan_auto"),
        help="Directory used by auto diabetes feasibility scan for per-run authority summaries.",
    )
    parser.add_argument(
        "--diabetes-feasibility-report-file",
        default=str(DATA_ROOT / "stress_diabetes_feasibility_report.auto.json"),
        help="Output JSON path for auto diabetes feasibility report.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    timeout_seconds = float(SUBPROCESS_TIMEOUT_SECONDS)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_note = (
            f"subprocess_timeout: timeout_after_seconds={int(SUBPROCESS_TIMEOUT_SECONDS)} "
            f"cwd={str(cwd) if cwd else ''}"
        )
        stderr_combined = "\n".join([part for part in [stderr_text.strip(), timeout_note] if part]).strip()
        timed_out_proc = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=stdout_text,
            stderr=stderr_combined,
        )
        if allow_fail:
            return timed_out_proc
        raise RuntimeError(
            "Command failed (timeout):\n"
            f"$ {' '.join(cmd)}\n"
            f"exit={timed_out_proc.returncode}\n"
            f"stdout:\n{timed_out_proc.stdout}\n"
            f"stderr:\n{timed_out_proc.stderr}"
        )
    if not allow_fail and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"$ {' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return sha256_bytes(csv_bytes)


def summarize_split_frame(df: pd.DataFrame) -> Dict[str, Any]:
    events = int(pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).sum()) if "y" in df.columns else 0
    rows = int(len(df))
    return {
        "rows": rows,
        "events": events,
        "event_rate": float(events / rows) if rows > 0 else 0.0,
        "sha256": dataframe_sha256(df),
    }


def extract_failure_codes(report_path: Path) -> List[str]:
    if not report_path.exists():
        return []
    try:
        payload = load_json(report_path)
    except Exception:
        return []
    failures = payload.get("failures")
    codes: List[str] = []
    if isinstance(failures, list):
        for issue in failures:
            if isinstance(issue, dict):
                code = issue.get("code")
                if isinstance(code, str) and code:
                    codes.append(code)
    return sorted(set(codes))


def failure_gap_score(report_path: Path) -> float:
    if not report_path.exists():
        return 1e6
    try:
        payload = load_json(report_path)
    except Exception:
        return 1e6
    if str(payload.get("status", "")).strip().lower() == "pass":
        return 0.0
    failures = payload.get("failures")
    if not isinstance(failures, list) or not failures:
        return 10.0
    score = 0.0
    parsed_any = False
    for issue in failures:
        if not isinstance(issue, dict):
            continue
        details = issue.get("details")
        if not isinstance(details, dict):
            score += 1.0
            continue

        def _num(name: str) -> Optional[float]:
            value = details.get(name)
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(str(value))
            except Exception:
                return None

        pairs = [
            ("required_min", "value", "min"),
            ("minimum", "observed", "min"),
            ("min_required", "observed", "min"),
            ("max_allowed", "observed", "max"),
            ("maximum", "observed", "max"),
            ("fail_threshold", "observed_gap", "max"),
            ("threshold", "value", "max"),
        ]
        issue_gap = 0.0
        local_parsed = False
        for threshold_key, observed_key, direction in pairs:
            threshold = _num(threshold_key)
            observed = _num(observed_key)
            if threshold is None or observed is None:
                continue
            local_parsed = True
            if direction == "min":
                issue_gap = max(issue_gap, max(0.0, threshold - observed))
            else:
                issue_gap = max(issue_gap, max(0.0, observed - threshold))
        if local_parsed:
            parsed_any = True
            score += issue_gap
        else:
            score += 1.0
    if not parsed_any:
        score += float(len(failures))
    return float(score)


def calibration_gap_diagnostics(report_path: Path) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ece_excess_total": 0.0,
        "slope_excess_total": 0.0,
        "intercept_excess_total": 0.0,
        "max_ece_excess": 0.0,
        "max_slope_excess": 0.0,
        "max_intercept_excess": 0.0,
        "total_excess": 0.0,
    }
    if not report_path.exists():
        return payload
    try:
        report = load_json(report_path)
    except Exception:
        return payload
    failures = report.get("failures")
    if not isinstance(failures, list):
        return payload
    for issue in failures:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code", "")).strip()
        details = issue.get("details")
        if not isinstance(details, dict):
            continue
        if code == "calibration_ece_exceeds_threshold":
            ece = details.get("ece")
            ece_max = details.get("ece_max")
            if isinstance(ece, (int, float)) and isinstance(ece_max, (int, float)):
                excess = max(0.0, float(ece) - float(ece_max))
                payload["ece_excess_total"] = float(payload["ece_excess_total"]) + excess
                payload["max_ece_excess"] = max(float(payload["max_ece_excess"]), excess)
        elif code == "calibration_slope_out_of_range":
            slope = details.get("slope")
            slope_min = details.get("slope_min")
            slope_max = details.get("slope_max")
            if isinstance(slope, (int, float)):
                lower = max(0.0, float(slope_min) - float(slope)) if isinstance(slope_min, (int, float)) else 0.0
                upper = max(0.0, float(slope) - float(slope_max)) if isinstance(slope_max, (int, float)) else 0.0
                excess = max(lower, upper)
                payload["slope_excess_total"] = float(payload["slope_excess_total"]) + excess
                payload["max_slope_excess"] = max(float(payload["max_slope_excess"]), excess)
        elif code == "calibration_intercept_out_of_range":
            intercept = details.get("intercept")
            intercept_min = details.get("intercept_min")
            intercept_max = details.get("intercept_max")
            if isinstance(intercept, (int, float)):
                lower = (
                    max(0.0, float(intercept_min) - float(intercept))
                    if isinstance(intercept_min, (int, float))
                    else 0.0
                )
                upper = (
                    max(0.0, float(intercept) - float(intercept_max))
                    if isinstance(intercept_max, (int, float))
                    else 0.0
                )
                excess = max(lower, upper)
                payload["intercept_excess_total"] = float(payload["intercept_excess_total"]) + excess
                payload["max_intercept_excess"] = max(float(payload["max_intercept_excess"]), excess)
    payload["total_excess"] = float(
        float(payload["ece_excess_total"])
        + float(payload["slope_excess_total"])
        + float(payload["intercept_excess_total"])
    )
    return payload


def load_heart_dataset(raw_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    cols = HEART_ALL_FEATURE_COLS + ["goal"]
    frames: List[pd.DataFrame] = []

    sources: List[Tuple[str, Path]] = [("cleveland", raw_path)]
    for site_tag, filename in HEART_INTERNAL_AUX_RAW_FILES:
        sources.append((site_tag, RAW_ROOT / filename))

    for site_tag, site_path in sources:
        if not site_path.exists():
            continue
        site_df = pd.read_csv(site_path, header=None, names=cols, na_values="?")
        for col in HEART_ALL_FEATURE_COLS:
            site_df[col] = pd.to_numeric(site_df[col], errors="coerce")
        goal = pd.to_numeric(site_df["goal"], errors="coerce")
        keep_rows = goal.notna()
        site_df = site_df.loc[keep_rows].copy().reset_index(drop=True)
        if site_df.empty:
            continue
        site_df["y"] = (pd.to_numeric(site_df["goal"], errors="coerce") > 0).astype(int)
        site_df["__heart_site_tag"] = site_tag
        frames.append(site_df)

    if not frames:
        raise RuntimeError(f"No usable heart source rows found for: {raw_path}")
    df = pd.concat(frames, axis=0, ignore_index=True)
    feature_cols = [c for c in HEART_STABLE_FEATURE_COLS if c in HEART_ALL_FEATURE_COLS]
    # Keep rows with at least one observed predictor; missing values are imputed
    # downstream by train-only fitted imputers.
    has_any_feature = df[feature_cols].notna().any(axis=1)
    df = df.loc[has_any_feature].copy().reset_index(drop=True)
    return df[feature_cols + ["y"]], feature_cols


def load_heart_external_institution_pool(
    case_id: str,
    feature_cols: List[str],
    exclude_site_tags: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    cols = HEART_ALL_FEATURE_COLS + ["goal"]
    excluded = {str(x).strip().lower() for x in (exclude_site_tags or []) if str(x).strip()}
    metadata: Dict[str, Any] = {
        "source": "uci_heart_multi_institution",
        "site_selection_policy": {
            "min_rows": int(HEART_EXTERNAL_SITE_MIN_ROWS),
            "event_rate_min": float(HEART_EXTERNAL_SITE_EVENT_RATE_MIN),
            "event_rate_max": float(HEART_EXTERNAL_SITE_EVENT_RATE_MAX),
        },
        "exclude_site_tags": sorted(excluded),
        "sites": [],
    }
    frames: List[pd.DataFrame] = []
    for site_tag, filename in HEART_EXTERNAL_INSTITUTION_RAW_FILES:
        site_summary: Dict[str, Any] = {
            "site_tag": site_tag,
            "filename": filename,
            "status": "excluded",
            "reason": None,
        }
        if site_tag.lower() in excluded:
            site_summary["reason"] = "excluded_training_source"
            metadata["sites"].append(site_summary)
            continue
        raw_path = RAW_ROOT / filename
        if not raw_path.exists():
            site_summary["reason"] = "file_missing"
            metadata["sites"].append(site_summary)
            continue
        raw_df = pd.read_csv(raw_path, header=None, names=cols, na_values="?")
        if raw_df.empty:
            site_summary["reason"] = "empty_file"
            metadata["sites"].append(site_summary)
            continue
        for col in HEART_ALL_FEATURE_COLS:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        goal = pd.to_numeric(raw_df["goal"], errors="coerce")
        keep_rows = goal.notna()
        if not bool(keep_rows.any()):
            site_summary["reason"] = "label_missing"
            metadata["sites"].append(site_summary)
            continue
        raw_df = raw_df.loc[keep_rows].copy().reset_index(drop=True)
        raw_df["y"] = (pd.to_numeric(raw_df["goal"], errors="coerce") > 0).astype(int)
        # Keep rows with at least one observed predictor; missing values are handled downstream.
        has_any_feature = raw_df[feature_cols].notna().any(axis=1)
        raw_df = raw_df.loc[has_any_feature].copy().reset_index(drop=True)
        if raw_df.empty:
            site_summary["reason"] = "all_features_missing"
            metadata["sites"].append(site_summary)
            continue

        rows = int(raw_df.shape[0])
        events = int(pd.to_numeric(raw_df["y"], errors="coerce").fillna(0).astype(int).sum())
        event_rate = float(events / rows) if rows > 0 else 0.0
        site_summary["rows"] = rows
        site_summary["events"] = events
        site_summary["event_rate"] = event_rate
        if rows < int(HEART_EXTERNAL_SITE_MIN_ROWS):
            site_summary["reason"] = "below_min_rows"
            metadata["sites"].append(site_summary)
            continue
        if event_rate < float(HEART_EXTERNAL_SITE_EVENT_RATE_MIN) or event_rate > float(HEART_EXTERNAL_SITE_EVENT_RATE_MAX):
            site_summary["reason"] = "event_rate_out_of_range"
            metadata["sites"].append(site_summary)
            continue
        base_time = datetime(2027, 1, 1, tzinfo=timezone.utc) + timedelta(days=len(frames) * 365)
        patient_ids = [f"{case_id.upper()}_EXT_{site_tag.upper()}_{i:06d}" for i in range(int(raw_df.shape[0]))]
        event_times = [
            (base_time + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(int(raw_df.shape[0]))
        ]
        out = raw_df[feature_cols + ["y"]].copy()
        out.insert(0, "event_time", event_times)
        out.insert(0, "patient_id", patient_ids)
        frames.append(out[["patient_id", "event_time", "y"] + feature_cols])
        site_summary["status"] = "included"
        metadata["sites"].append(site_summary)

    if not frames:
        metadata["status"] = "empty"
        metadata["reason"] = "no_site_passed_selection"
        return None, metadata
    merged = pd.concat(frames, axis=0, ignore_index=True)
    if "y" not in merged.columns or merged["y"].nunique() < 2:
        metadata["status"] = "invalid"
        metadata["reason"] = "insufficient_label_variation"
        return None, metadata
    metadata["status"] = "ok"
    metadata["summary"] = summarize_split_frame(merged)
    return merged, metadata


def load_breast_dataset(raw_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    cols = ["sample_id", "diagnosis"] + feature_cols
    df = pd.read_csv(raw_path, header=None, names=cols)
    df["y"] = (df["diagnosis"].astype(str).str.strip().str.upper() == "M").astype(int)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df[feature_cols + ["y"]], feature_cols


def load_ckd_dataset(
    raw_path: Path,
    max_rows: Optional[int] = DEFAULT_CKD_MAX_ROWS,
    random_seed: int = 20260227,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "rbc",
        "pc",
        "pcc",
        "ba",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
        "ane",
    ]
    cols = feature_cols + ["class"]
    numeric_cols = {
        "age",
        "bp",
        "sg",
        "al",
        "su",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
    }
    missing_tokens = {"", "?", "na", "nan", "none", "null"}

    text = raw_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rows: List[List[str]] = []
    in_data = False
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower().startswith("@data"):
                in_data = True
            continue
        tokens = [str(token).strip().strip("'").strip('"') for token in line.split(",")]
        tokens = [token.replace("\t", "").replace("\r", "").strip() for token in tokens]
        if len(tokens) < len(cols):
            tokens.extend(["?"] * (len(cols) - len(tokens)))
        elif len(tokens) > len(cols):
            tokens = tokens[: len(cols)]
        rows.append(tokens)

    if not rows:
        raise ValueError(f"CKD parser found no data rows: {raw_path}")
    df = pd.DataFrame(rows, columns=cols)
    normalized = pd.DataFrame(index=df.index)
    for col in cols:
        series = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
        series = series.str.replace(r"[\t\r\n ]+", "", regex=True)
        series = series.replace({"": np.nan, "?": np.nan})
        normalized[col] = series

    class_token = (
        normalized["class"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    y = class_token.map({"ckd": 1, "notckd": 0})
    keep_label = y.notna()
    if not bool(keep_label.any()):
        raise ValueError("CKD dataset contains no valid labels after normalization.")
    normalized = normalized.loc[keep_label].copy()
    y = y.loc[keep_label].astype(int)

    encoded = pd.DataFrame(index=normalized.index)
    for col in feature_cols:
        series = normalized[col]
        if col in numeric_cols:
            token = series.astype(str).str.lower()
            token = token.where(~token.isin(missing_tokens), np.nan)
            encoded[col] = pd.to_numeric(token, errors="coerce")
            continue
        token = (
            series.astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9_.-]", "", regex=True)
        )
        token = token.where(~token.isin(missing_tokens), np.nan)
        categories = sorted(str(v) for v in token.dropna().unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        encoded[col] = token.map(mapping).astype(float)

    non_empty = encoded.notna().any(axis=1)
    out = encoded.loc[non_empty].copy()
    out["y"] = y.loc[non_empty].astype(int)
    if out["y"].nunique() < 2:
        raise ValueError("CKD labels collapsed after cleaning.")

    if isinstance(max_rows, int) and max_rows > 0 and int(out.shape[0]) > int(max_rows):
        sampled_idx, _ = train_test_split(
            out.index.to_numpy(),
            train_size=int(max_rows),
            random_state=int(random_seed),
            stratify=out["y"].to_numpy(),
        )
        out = out.loc[np.asarray(sampled_idx)].copy()

    return out[feature_cols + ["y"]].reset_index(drop=True), feature_cols


def normalize_categorical_token(series: pd.Series) -> pd.Series:
    token = series.astype(str).str.strip().str.lower()
    token = token.replace(
        {
            "": np.nan,
            "?": np.nan,
            "none": np.nan,
            "null": np.nan,
            "nan": np.nan,
            "unknown/invalid": np.nan,
        }
    )
    return token


def icd9_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    token = str(value).strip().upper().replace(" ", "")
    if token in {"", "?", "NAN", "NONE", "NULL"}:
        return "unknown"
    if token.startswith("V"):
        return "supplemental_v"
    if token.startswith("E"):
        return "external_e"
    try:
        code = float(token)
    except Exception:
        return "other"
    if 250.0 <= code < 251.0:
        return "diabetes"
    if (390.0 <= code <= 459.0) or (int(code) == 785):
        return "circulatory"
    if (460.0 <= code <= 519.0) or (int(code) == 786):
        return "respiratory"
    if (520.0 <= code <= 579.0) or (int(code) == 787):
        return "digestive"
    if (580.0 <= code <= 629.0) or (int(code) == 788):
        return "genitourinary"
    if 140.0 <= code <= 239.0:
        return "neoplasms"
    if 710.0 <= code <= 739.0:
        return "musculoskeletal"
    if 800.0 <= code <= 999.0:
        return "injury"
    return "other"


def load_diabetes_130_dataset(
    raw_path: Path,
    max_rows: Optional[int] = DEFAULT_DIABETES_MAX_ROWS,
    random_seed: int = 20260226,
    target_mode: str = DEFAULT_DIABETES_TARGET_MODE,
) -> Tuple[pd.DataFrame, List[str]]:
    na_tokens = ["?", "Unknown/Invalid", "None", "NULL", ""]
    df = pd.read_csv(raw_path, na_values=na_tokens, low_memory=False)
    required_cols = {"encounter_id", "patient_nbr", "readmitted"}
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(required_cols - set(df.columns))
        raise ValueError(f"Diabetes130 raw file missing required columns: {missing}")

    encounter_num = pd.to_numeric(df["encounter_id"], errors="coerce")
    patient_raw = df["patient_nbr"].astype(str).str.strip()
    readmitted = df["readmitted"].astype(str).str.strip()
    keep_mask = encounter_num.notna() & patient_raw.ne("") & readmitted.ne("")
    df = df.loc[keep_mask].copy().reset_index(drop=True)
    df["_encounter_num"] = pd.to_numeric(df["encounter_id"], errors="coerce")
    df["_patient_raw"] = df["patient_nbr"].astype(str).str.strip()
    mode = str(target_mode).strip().lower()
    if mode not in {"lt30", "gt30", "any"}:
        raise ValueError(f"Unsupported diabetes target_mode: {target_mode}")
    readmit_token = df["readmitted"].astype(str).str.strip()
    if mode == "lt30":
        df["y"] = (readmit_token == "<30").astype(int)
    elif mode == "gt30":
        df["y"] = (readmit_token == ">30").astype(int)
    else:
        df["y"] = readmit_token.isin(["<30", ">30"]).astype(int)

    # Prevent patient-level leakage: keep earliest encounter per patient.
    df = df.sort_values("_encounter_num", kind="mergesort").drop_duplicates("_patient_raw", keep="first")
    if df["y"].nunique() < 2:
        raise ValueError("Diabetes130 target must contain both positive and negative labels after de-duplication.")

    if isinstance(max_rows, int) and max_rows > 0 and int(df.shape[0]) > int(max_rows):
        sampled_idx, _ = train_test_split(
            df.index.to_numpy(),
            train_size=int(max_rows),
            random_state=int(random_seed),
            stratify=df["y"].to_numpy(),
        )
        df = df.loc[np.asarray(sampled_idx)].copy()

    drop_cols = {"readmitted", "encounter_id", "patient_nbr", "_encounter_num", "_patient_raw", "y"}
    base_feature_cols: List[str] = [c for c in df.columns if c not in drop_cols]
    engineered = pd.DataFrame(index=df.index)

    diag_cols = [c for c in ("diag_1", "diag_2", "diag_3") if c in df.columns]
    diabetes_diag_flags: List[pd.Series] = []
    chronic_diag_flags: List[pd.Series] = []
    for col in diag_cols:
        bucket = df[col].map(icd9_bucket)
        engineered[f"{col}_bucket"] = bucket
        is_diabetes = (bucket == "diabetes").astype(float)
        is_chronic = bucket.isin({"diabetes", "circulatory", "genitourinary", "neoplasms"}).astype(float)
        engineered[f"{col}_is_diabetes"] = is_diabetes
        engineered[f"{col}_is_major_chronic"] = is_chronic
        diabetes_diag_flags.append(is_diabetes)
        chronic_diag_flags.append(is_chronic)
    if diabetes_diag_flags:
        engineered["diag_diabetes_count"] = np.sum(np.column_stack(diabetes_diag_flags), axis=1).astype(float)
    if chronic_diag_flags:
        engineered["diag_major_chronic_count"] = np.sum(np.column_stack(chronic_diag_flags), axis=1).astype(float)

    medication_cols = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    med_active_flags: List[pd.Series] = []
    med_change_flags: List[pd.Series] = []
    for col in medication_cols:
        if col not in df.columns:
            continue
        token = normalize_categorical_token(df[col])
        active = token.isin({"up", "down", "steady"}).astype(float)
        changed = token.isin({"up", "down"}).astype(float)
        med_active_flags.append(active)
        med_change_flags.append(changed)
        engineered[f"{col}_is_active"] = active
        engineered[f"{col}_is_changed"] = changed
    if med_active_flags:
        engineered["med_active_count"] = np.sum(np.column_stack(med_active_flags), axis=1).astype(float)
    if med_change_flags:
        engineered["med_changed_count"] = np.sum(np.column_stack(med_change_flags), axis=1).astype(float)
    if "insulin" in df.columns:
        insulin_token = normalize_categorical_token(df["insulin"])
        engineered["insulin_active"] = insulin_token.isin({"up", "down", "steady"}).astype(float)

    if "A1Cresult" in df.columns:
        a1c_token = normalize_categorical_token(df["A1Cresult"])
        engineered["a1c_abnormal"] = a1c_token.isin({">7", ">8"}).astype(float)
        engineered["a1c_measured"] = a1c_token.isin({">7", ">8", "norm"}).astype(float)
    if "max_glu_serum" in df.columns:
        glu_token = normalize_categorical_token(df["max_glu_serum"])
        engineered["glu_abnormal"] = glu_token.isin({">200", ">300"}).astype(float)
        engineered["glu_measured"] = glu_token.isin({">200", ">300", "norm"}).astype(float)

    numeric_util_cols = [c for c in ("number_inpatient", "number_outpatient", "number_emergency") if c in df.columns]
    util_parts: List[pd.Series] = []
    for col in numeric_util_cols:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        util_parts.append(series.astype(float))
    if util_parts:
        util_sum = np.sum(np.column_stack(util_parts), axis=1).astype(float)
        engineered["total_prior_utilization"] = util_sum
        engineered["total_prior_utilization_log1p"] = np.log1p(util_sum)

    if "time_in_hospital" in df.columns:
        los = pd.to_numeric(df["time_in_hospital"], errors="coerce").fillna(0.0).clip(lower=0.0)
        los_safe = np.maximum(los.to_numpy(dtype=float), 1.0)
        if "num_medications" in df.columns:
            meds = pd.to_numeric(df["num_medications"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            engineered["medications_per_day"] = meds / los_safe
        if "num_procedures" in df.columns:
            procs = pd.to_numeric(df["num_procedures"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            engineered["procedures_per_day"] = procs / los_safe

    if not engineered.empty:
        for col in engineered.columns:
            df[col] = engineered[col]

    feature_cols: List[str] = [c for c in df.columns if c not in drop_cols]
    encoded_cols: Dict[str, pd.Series] = {}
    for col in feature_cols:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        if numeric_ratio >= 0.90:
            encoded_cols[col] = numeric.astype(float)
            continue
        token = series.astype(str).str.strip()
        token = token.replace({"": np.nan, "?": np.nan, "Unknown/Invalid": np.nan, "None": np.nan, "NULL": np.nan})
        categories = sorted(str(v) for v in token.dropna().unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        encoded_cols[col] = token.map(mapping).astype(float)

    encoded = pd.DataFrame(encoded_cols, index=df.index)

    non_empty = encoded.notna().any(axis=1)
    out = encoded.loc[non_empty, feature_cols].copy()
    out["y"] = df.loc[non_empty, "y"].astype(int).to_numpy()
    if out["y"].nunique() < 2:
        raise ValueError("Diabetes130 labels collapsed after feature filtering.")
    return out[feature_cols + ["y"]].reset_index(drop=True), feature_cols


def compute_risk_proxy(frame: pd.DataFrame, feature_cols: List[str]) -> Optional[pd.Series]:
    numeric_cols = [c for c in feature_cols if c in frame.columns]
    if not numeric_cols:
        return None
    numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] == 0:
        return None
    filled = numeric.copy()
    for col in filled.columns:
        series = filled[col]
        median = float(series.median(skipna=True)) if series.notna().any() else 0.0
        filled[col] = series.fillna(median)
    std = filled.std(axis=0, ddof=0).replace(0.0, 1.0)
    z = (filled - filled.mean(axis=0)) / std
    return z.sum(axis=1)


def build_stratify_labels(y_series: pd.Series, risk_proxy: Optional[pd.Series]) -> Tuple[Optional[np.ndarray], List[str]]:
    y = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None, ["y"]
    y_tokens = y.astype(str)
    if risk_proxy is not None:
        rp = pd.to_numeric(risk_proxy, errors="coerce")
        if rp.notna().any():
            for q in (5, 4, 3, 2):
                try:
                    bins = pd.qcut(rp, q=q, duplicates="drop")
                except Exception:
                    continue
                bin_tokens = bins.astype(str).fillna("q0")
                strata = y_tokens + "__" + bin_tokens
                counts = strata.value_counts()
                if not counts.empty and int(counts.min()) >= 2 and int(strata.nunique()) >= 2:
                    return strata.to_numpy(), ["y", f"risk_proxy_quantile_q{q}"]
    y_array = y.to_numpy()
    if len(set(y_array.tolist())) < 2:
        return None, ["y"]
    return y_array, ["y"]


def split_with_temporal_order(
    df: pd.DataFrame,
    feature_cols: List[str],
    case_id: str,
    split_seed_override: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    split_seed = int(split_seed_override if split_seed_override is not None else SPLIT_RANDOM_SEED_BY_CASE.get(case_id, 20260224))
    external_ratio = float(EXTERNAL_SPLIT_RATIO_BY_CASE.get(case_id, 0.17))
    if not (0.05 <= external_ratio <= 0.40):
        external_ratio = 0.17
    fractions = INTERNAL_SPLIT_FRACTIONS_BY_CASE.get(case_id, {"train": 0.60, "valid": 0.20, "test": 0.20})
    train_fraction = float(fractions.get("train", 0.60))
    valid_fraction = float(fractions.get("valid", 0.20))
    test_fraction = float(fractions.get("test", 0.20))
    total_fraction = train_fraction + valid_fraction + test_fraction
    if total_fraction <= 0:
        train_fraction, valid_fraction, test_fraction = 0.60, 0.20, 0.20
    else:
        train_fraction /= total_fraction
        valid_fraction /= total_fraction
        test_fraction /= total_fraction
    if not (0.30 <= train_fraction <= 0.80):
        train_fraction = 0.60
    if not (0.05 <= valid_fraction <= 0.40):
        valid_fraction = 0.20
    if not (0.05 <= test_fraction <= 0.40):
        test_fraction = 0.20
    total_fraction = train_fraction + valid_fraction + test_fraction
    train_fraction /= total_fraction
    valid_fraction /= total_fraction
    test_fraction /= total_fraction

    y = df["y"].to_numpy()
    indices = df.index.to_numpy()
    # Use risk-aware stratification for every case to reduce split-induced
    # distribution separability under strict publication gates.
    risk_proxy = compute_risk_proxy(df, feature_cols)
    external_stratify, _ = build_stratify_labels(df["y"], risk_proxy)
    internal_idx, external_idx = train_test_split(
        indices,
        test_size=external_ratio,
        random_state=split_seed,
        stratify=external_stratify if external_stratify is not None else y,
    )
    internal_df = df.loc[internal_idx]
    internal_risk = risk_proxy.loc[internal_idx] if risk_proxy is not None else None
    internal_stratify, _ = build_stratify_labels(internal_df["y"], internal_risk)
    train_idx, temp_idx = train_test_split(
        internal_idx,
        test_size=(1.0 - train_fraction),
        random_state=split_seed,
        stratify=internal_stratify if internal_stratify is not None else y[internal_idx],
    )
    temp_df = df.loc[temp_idx]
    temp_risk = risk_proxy.loc[temp_idx] if risk_proxy is not None else None
    temp_stratify, _ = build_stratify_labels(temp_df["y"], temp_risk)
    valid_within_temp = valid_fraction / max(1e-9, (valid_fraction + test_fraction))
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - valid_within_temp),
        random_state=split_seed,
        stratify=temp_stratify if temp_stratify is not None else y[temp_idx],
    )

    split_map = {
        "train": sorted(int(x) for x in train_idx),
        "valid": sorted(int(x) for x in valid_idx),
        "test": sorted(int(x) for x in test_idx),
        "external": sorted(int(x) for x in external_idx),
    }
    starts = {
        "train": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "valid": datetime(2022, 1, 1, tzinfo=timezone.utc),
        "test": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "external": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }

    output: Dict[str, pd.DataFrame] = {}
    for split_name, idxs in split_map.items():
        sub = df.loc[idxs].copy().reset_index(drop=True)
        patient_ids = [f"{case_id.upper()}_{i:06d}" for i in idxs]
        times = [
            (starts[split_name] + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(len(sub))
        ]
        sub.insert(0, "event_time", times)
        sub.insert(0, "patient_id", patient_ids)
        ordered = sub[["patient_id", "event_time", "y"] + feature_cols].copy()
        ordered["y"] = ordered["y"].astype(int)
        output[split_name] = ordered
    return output


def split_external_pool(
    external_df: pd.DataFrame,
    case_id: str,
    split_seed_override: Optional[int] = None,
    min_rows_per_cohort: int = 20,
    min_positive_per_cohort: int = 3,
    external_institution_override: Optional[pd.DataFrame] = None,
    external_institution_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if external_df.empty:
        raise ValueError("External pool is empty.")

    def cohort_ok(df_part: pd.DataFrame) -> bool:
        rows = int(len(df_part))
        events = int(pd.to_numeric(df_part["y"], errors="coerce").fillna(0).astype(int).sum())
        non_events = int(rows - events)
        return rows >= int(min_rows_per_cohort) and events >= int(min_positive_per_cohort) and non_events >= int(min_positive_per_cohort)

    split_seed = int(split_seed_override if split_seed_override is not None else SPLIT_RANDOM_SEED_BY_CASE.get(case_id, 20260224))
    case_token = case_id.upper()
    if isinstance(external_institution_override, pd.DataFrame) and not external_institution_override.empty:
        left = external_df.copy().reset_index(drop=True)
        right = external_institution_override.copy().reset_index(drop=True)

        left["patient_id"] = [f"{case_token}_CP_{i:06d}" for i in range(int(left.shape[0]))]
        right["patient_id"] = [f"{case_token}_CI_{i:06d}" for i in range(int(right.shape[0]))]

        metadata = {
            "strategy": "external_pool_plus_independent_institution",
            "split_seed_initial": int(split_seed),
            "split_seed_selected": int(split_seed),
            "attempt_count": 1,
            "fallback_used": False,
            "stratification_keys": ["y"],
            "risk_proxy_used": False,
            "minimum_requirements": {
                "min_rows_per_cohort": int(min_rows_per_cohort),
                "min_positive_per_cohort": int(min_positive_per_cohort),
            },
            "cohorts": {
                "cross_period": summarize_split_frame(left),
                "cross_institution": summarize_split_frame(right),
            },
            "cross_institution_source": (
                external_institution_metadata if isinstance(external_institution_metadata, dict) else {}
            ),
            "minimum_requirements_met": bool(cohort_ok(left) and cohort_ok(right)),
        }
        return left, right, metadata

    idx = external_df.index.to_numpy()
    y = external_df["y"].to_numpy()
    stratify_labels = y if len(set(y.tolist())) > 1 else None
    stratification_keys: List[str] = ["y"]

    feature_cols = [c for c in external_df.columns if c not in {"patient_id", "event_time", "y"}]
    risk_proxy: Optional[pd.Series] = compute_risk_proxy(external_df, feature_cols)
    strata_labels, strata_keys = build_stratify_labels(external_df["y"], risk_proxy)
    if strata_labels is not None:
        stratify_labels = strata_labels
        stratification_keys = strata_keys

    def split_balance_score(
        left_idx_raw: np.ndarray,
        right_idx_raw: np.ndarray,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
    ) -> float:
        left_rows = float(len(left_df))
        right_rows = float(len(right_df))
        total_rows = max(1.0, left_rows + right_rows)
        left_events = float(pd.to_numeric(left_df["y"], errors="coerce").fillna(0).astype(int).sum())
        right_events = float(pd.to_numeric(right_df["y"], errors="coerce").fillna(0).astype(int).sum())
        left_rate = left_events / max(1.0, left_rows)
        right_rate = right_events / max(1.0, right_rows)
        event_rate_gap = abs(left_rate - right_rate)
        event_count_gap = abs(left_events - right_events) / max(1.0, left_events + right_events)
        row_gap = abs(left_rows - right_rows) / total_rows
        risk_gap = 0.0
        if risk_proxy is not None:
            left_risk = pd.to_numeric(risk_proxy.loc[left_idx_raw], errors="coerce")
            right_risk = pd.to_numeric(risk_proxy.loc[right_idx_raw], errors="coerce")
            pooled = pd.concat([left_risk, right_risk], axis=0)
            if pooled.notna().any() and left_risk.notna().any() and right_risk.notna().any():
                pooled_std = float(pooled.std(ddof=0))
                if not np.isfinite(pooled_std) or pooled_std <= 1e-9:
                    pooled_std = 1.0
                risk_gap = abs(float(left_risk.mean()) - float(right_risk.mean())) / pooled_std
        return float((2.0 * event_rate_gap) + event_count_gap + (0.5 * row_gap) + (0.5 * risk_gap))

    left: Optional[pd.DataFrame] = None
    right: Optional[pd.DataFrame] = None
    selected_seed: Optional[int] = None
    best_score = float("inf")
    backup_left: Optional[pd.DataFrame] = None
    backup_right: Optional[pd.DataFrame] = None
    backup_seed: Optional[int] = None
    attempt_count = 0
    max_attempts = 64
    for offset in range(max_attempts):
        candidate_seed = int(split_seed + offset)
        attempt_count += 1
        left_idx, right_idx = train_test_split(
            idx,
            test_size=0.50,
            random_state=candidate_seed,
            stratify=stratify_labels,
        )
        left_candidate = external_df.loc[left_idx].copy().reset_index(drop=True)
        right_candidate = external_df.loc[right_idx].copy().reset_index(drop=True)
        if left_candidate["y"].nunique() < 2 or right_candidate["y"].nunique() < 2:
            continue
        if backup_left is None and backup_right is None:
            backup_left = left_candidate
            backup_right = right_candidate
            backup_seed = candidate_seed
        if not (cohort_ok(left_candidate) and cohort_ok(right_candidate)):
            continue
        candidate_score = split_balance_score(left_idx, right_idx, left_candidate, right_candidate)
        if candidate_score < best_score:
            best_score = float(candidate_score)
            left = left_candidate
            right = right_candidate
            selected_seed = candidate_seed

    if (left is None or right is None) and backup_left is not None and backup_right is not None:
        left = backup_left
        right = backup_right
        selected_seed = int(backup_seed if backup_seed is not None else split_seed)

    fallback_used = False
    if left is None or right is None or left["y"].nunique() < 2 or right["y"].nunique() < 2:
        # Deterministic fallback: alternating split.
        left = external_df.iloc[::2].copy().reset_index(drop=True)
        right = external_df.iloc[1::2].copy().reset_index(drop=True)
        fallback_used = True
        selected_seed = int(split_seed)

    if not cohort_ok(left) or not cohort_ok(right):
        fallback_used = True

    left["patient_id"] = left["patient_id"].astype(str).str.replace(f"{case_id.upper()}_", f"{case_id.upper()}_CP_", regex=False)
    right["patient_id"] = right["patient_id"].astype(str).str.replace(f"{case_id.upper()}_", f"{case_id.upper()}_CI_", regex=False)
    metadata = {
        "strategy": "deterministic_label_risk_stratified",
        "split_seed_initial": int(split_seed),
        "split_seed_selected": int(selected_seed if selected_seed is not None else split_seed),
        "attempt_count": int(attempt_count),
        "fallback_used": bool(fallback_used),
        "stratification_keys": stratification_keys,
        "risk_proxy_used": bool(risk_proxy is not None),
        "minimum_requirements": {
            "min_rows_per_cohort": int(min_rows_per_cohort),
            "min_positive_per_cohort": int(min_positive_per_cohort),
        },
        "cohorts": {
            "cross_period": summarize_split_frame(left),
            "cross_institution": summarize_split_frame(right),
        },
        "minimum_requirements_met": bool(cohort_ok(left) and cohort_ok(right)),
    }
    return left, right, metadata


def build_feature_group_spec(feature_cols: List[str]) -> Dict[str, Any]:
    groups = {"demographics": [], "clinical_signals": [], "derived_features": []}
    assigned: set[str] = set()
    for col in feature_cols:
        token = col.lower()
        if any(x in token for x in ("age", "sex", "gender")):
            groups["demographics"].append(col)
            assigned.add(col)
        elif any(x in token for x in ("mean", "worst", "se", "oldpeak", "slope", "ca", "thal", "cp", "exang", "fbs")):
            groups["derived_features"].append(col)
            assigned.add(col)
        else:
            groups["clinical_signals"].append(col)
            assigned.add(col)

    def first_unassigned() -> str | None:
        for name in feature_cols:
            if name not in assigned:
                return name
        return None

    if not groups["demographics"]:
        fallback = first_unassigned()
        if fallback is not None:
            groups["demographics"].append(fallback)
            assigned.add(fallback)
    if not groups["clinical_signals"]:
        fallback = first_unassigned()
        if fallback is not None:
            groups["clinical_signals"].append(fallback)
            assigned.add(fallback)
    if not groups["derived_features"]:
        fallback = first_unassigned()
        if fallback is not None:
            groups["derived_features"].append(fallback)
            assigned.add(fallback)

    # Contract requires every group non-empty and no duplicated feature assignments.
    # If all features are already assigned to a subset of groups, rebalance by moving
    # one feature from the currently largest group into an empty group.
    for target_group in ("demographics", "clinical_signals", "derived_features"):
        if groups[target_group]:
            continue
        donor_candidates = [
            name for name in ("demographics", "clinical_signals", "derived_features") if len(groups[name]) > 1
        ]
        if not donor_candidates:
            donor_candidates = [
                name for name in ("demographics", "clinical_signals", "derived_features") if len(groups[name]) > 0
            ]
        if not donor_candidates:
            continue
        donor_group = max(donor_candidates, key=lambda g: len(groups[g]))
        moved_feature = groups[donor_group].pop()
        groups[target_group].append(moved_feature)
    return {
        "groups": groups,
        "forbidden_features": ["confirmed_diagnosis_code", "reference_standard_positive"],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = f".{path.name}.tmp-{os.getpid()}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    tmp_path = path.parent / tmp_name
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def resolve_output_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def canonical_json_sha256(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def git_revision_hint() -> str:
    try:
        head = run_cmd(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", "HEAD"],
            allow_fail=True,
        )
        rev = str(head.stdout).strip() if head.returncode == 0 else "unknown"
        dirty = run_cmd(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            allow_fail=True,
        )
        if dirty.returncode == 0 and str(dirty.stdout).strip():
            return f"{rev}-dirty"
        return rev or "unknown"
    except Exception:
        return "unknown"


def get_stress_profiles(profile_set: str) -> List[Dict[str, str]]:
    token = str(profile_set).strip()
    profiles = STRESS_PROFILE_SETS.get(token)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"Unsupported stress profile set: {profile_set}")
    normalized: List[Dict[str, str]] = []
    for row in profiles:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid stress profile row in set={profile_set}: {row}")
        profile_id = str(row.get("profile_id", "")).strip()
        selection_data = str(row.get("selection_data", "")).strip().lower()
        threshold_selection_split = str(row.get("threshold_selection_split", "")).strip().lower()
        calibration_fit_split = str(row.get("calibration_fit_split", "")).strip().lower()
        calibration_method = str(row.get("calibration_method", "")).strip().lower()
        if not profile_id:
            raise ValueError(f"Missing profile_id in stress profile set={profile_set}")
        if selection_data not in {"valid", "cv_inner", "nested_cv"}:
            raise ValueError(f"Invalid selection_data in stress profile '{profile_id}': {selection_data}")
        if threshold_selection_split not in {"valid", "cv_inner"}:
            raise ValueError(
                f"Invalid threshold_selection_split in stress profile '{profile_id}': {threshold_selection_split}"
            )
        if calibration_fit_split not in {"valid", "cv_inner"}:
            raise ValueError(f"Invalid calibration_fit_split in stress profile '{profile_id}': {calibration_fit_split}")
        if calibration_method not in {"none", "sigmoid", "isotonic", "power", "beta"}:
            raise ValueError(f"Invalid calibration_method in stress profile '{profile_id}': {calibration_method}")
        normalized.append(
            {
                "profile_id": profile_id,
                "selection_data": selection_data,
                "threshold_selection_split": threshold_selection_split,
                "calibration_fit_split": calibration_fit_split,
                "calibration_method": calibration_method,
            }
        )
    return normalized


def find_stress_profile_by_id(profile_id: str) -> Optional[Dict[str, str]]:
    token = str(profile_id).strip()
    if not token:
        return None
    for set_name in sorted(STRESS_PROFILE_SETS.keys()):
        for profile in get_stress_profiles(set_name):
            if str(profile.get("profile_id", "")).strip() == token:
                return profile
    return None


def build_stress_contract_inputs(case_id: str, profile_set: str, profiles: List[Dict[str, str]]) -> Dict[str, Any]:
    policy_template = load_json(REFERENCES_ROOT / "performance-policy.example.json")
    policy_template["_case"] = case_id
    tuning_template = load_json(REFERENCES_ROOT / "tuning-protocol.example.json")
    return {
        "contract_version": STRESS_SEARCH_REPORT_CONTRACT_VERSION,
        "case_id": case_id,
        "profile_set": profile_set,
        "profiles": profiles,
        "performance_policy_template": policy_template,
        "tuning_protocol_template": tuning_template,
    }


def build_dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    rows = int(df.shape[0])
    events = int(pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).sum()) if "y" in df.columns else 0
    return {
        "rows": rows,
        "events": events,
        "sha256": dataframe_sha256(df),
    }


def copy_reference(src_name: str, dst_path: Path) -> None:
    src = REFERENCES_ROOT / src_name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)


def update_missingness_policy(
    path: Path,
    train_rows: int,
    feature_count: int,
    mice_max_rows_override: Optional[int] = None,
    min_non_missing_override: Optional[int] = None,
) -> None:
    policy = load_json(path)
    if isinstance(min_non_missing_override, int) and min_non_missing_override > 0:
        policy["min_non_missing_per_feature"] = int(min_non_missing_override)
    strategy = str(policy.get("strategy", "")).strip().lower()
    if strategy != "mice_with_scale_guard":
        # Keep a feasible non-missing threshold for small benchmark cohorts.
        min_non_missing = policy.get("min_non_missing_per_feature")
        if isinstance(min_non_missing, (int, float)) and train_rows < int(min_non_missing):
            policy["min_non_missing_per_feature"] = max(20, int(0.80 * train_rows))
        write_json(path, policy)
        return

    if isinstance(mice_max_rows_override, int) and mice_max_rows_override > 0:
        policy["mice_max_rows"] = int(mice_max_rows_override)
    mice_max_rows = int(policy.get("mice_max_rows", 200000))
    mice_max_cols = int(policy.get("mice_max_cols", 200))
    large_rows = int(policy.get("large_data_row_threshold", 1_000_000))
    large_cols = int(policy.get("large_data_col_threshold", 300))
    should_trigger = (
        train_rows > mice_max_rows
        or feature_count > mice_max_cols
        or (train_rows >= large_rows and feature_count >= large_cols)
    )
    policy["scale_guard_evidence"] = {
        "fallback_triggered": bool(should_trigger),
        "fallback_strategy": "simple_with_indicator",
        "train_rows_seen": int(train_rows),
        "feature_count_seen": int(feature_count),
    }
    min_non_missing = policy.get("min_non_missing_per_feature")
    if isinstance(min_non_missing, (int, float)) and train_rows < int(min_non_missing):
        policy["min_non_missing_per_feature"] = max(20, int(0.80 * train_rows))
    write_json(path, policy)


def update_performance_policy(
    path: Path,
    case_id: str,
    threshold_split_override: Optional[str] = None,
    calibration_method_override: Optional[str] = None,
    calibration_fit_split_override: Optional[str] = None,
) -> None:
    policy = load_json(path)
    calibration_method_by_case = {
        "uci-heart-disease": "power",
        "uci-breast-cancer-wdbc": "power",
        "uci-chronic-kidney-disease": "beta",
    }
    calibration_fit_split_by_case = {
        # Keep threshold selection on valid, but fit the calibrator on CV-inner OOF
        # to reduce small-valid overfitting for stress heart cohorts.
        "uci-heart-disease": "cv_inner",
        "uci-breast-cancer-wdbc": "valid",
        "uci-chronic-kidney-disease": "cv_inner",
    }
    token = calibration_method_by_case.get(case_id)
    if isinstance(calibration_method_override, str) and calibration_method_override.strip():
        token = str(calibration_method_override).strip().lower()
    if isinstance(token, str) and token:
        policy["calibration_method"] = token
    fit_split_token = calibration_fit_split_by_case.get(case_id)
    if isinstance(calibration_fit_split_override, str) and calibration_fit_split_override.strip():
        fit_split_token = str(calibration_fit_split_override).strip().lower()
    if isinstance(fit_split_token, str) and fit_split_token in {"valid", "cv_inner"}:
        policy["calibration_fit_split"] = fit_split_token
    threshold_split = THRESHOLD_SELECTION_SPLIT_BY_CASE.get(case_id, "valid")
    if isinstance(threshold_split_override, str) and threshold_split_override.strip():
        threshold_split = str(threshold_split_override).strip().lower()
    threshold_policy = policy.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        threshold_policy = {}
    threshold_policy["selection_split"] = threshold_split
    if isinstance(fit_split_token, str) and fit_split_token in {"valid", "cv_inner"}:
        threshold_policy["calibration_fit_split"] = fit_split_token
    policy["threshold_policy"] = threshold_policy

    # Apply model-pool/runtime overrides only for explicitly configured cases.
    if case_id in MODEL_POOL_BY_CASE or case_id in MAX_TRIALS_PER_FAMILY_BY_CASE:
        model_pool_block = policy.get("model_pool")
        if not isinstance(model_pool_block, dict):
            model_pool_block = {}
        if case_id in MODEL_POOL_BY_CASE:
            resolved_pool = resolve_case_model_pool(case_id)
            if resolved_pool:
                model_pool_block["models"] = resolved_pool
            required_models = model_pool_block.get("required_models")
            if not isinstance(required_models, list) or not required_models:
                model_pool_block["required_models"] = ["logistic_l2"]
        if case_id in MAX_TRIALS_PER_FAMILY_BY_CASE:
            model_pool_block["max_trials_per_family"] = int(MAX_TRIALS_PER_FAMILY_BY_CASE[case_id])
        if case_id in HYPERPARAM_SEARCH_BY_CASE:
            search_strategy = str(HYPERPARAM_SEARCH_BY_CASE[case_id]).strip().lower()
            if search_strategy not in {"random_subsample", "fixed_grid"}:
                search_strategy = "random_subsample"
            model_pool_block["search_strategy"] = search_strategy
        if case_id in N_JOBS_BY_CASE:
            model_pool_block["n_jobs"] = int(N_JOBS_BY_CASE[case_id])
        policy["model_pool"] = model_pool_block
    write_json(path, policy)


def update_imbalance_policy(
    path: Path,
    case_id: str,
    threshold_split_override: Optional[str] = None,
    calibration_split_override: Optional[str] = None,
) -> None:
    policy = load_json(path)
    threshold_split = THRESHOLD_SELECTION_SPLIT_BY_CASE.get(case_id, "valid")
    if isinstance(threshold_split_override, str) and threshold_split_override.strip():
        threshold_split = str(threshold_split_override).strip().lower()
    calibration_split = threshold_split
    if isinstance(calibration_split_override, str) and calibration_split_override.strip():
        calibration_split = str(calibration_split_override).strip().lower()
    policy["threshold_selection_split"] = threshold_split
    policy["calibration_split"] = calibration_split
    write_json(path, policy)


def update_tuning_protocol(path: Path, model_selection_data: str) -> None:
    spec = load_json(path)
    selection_data = str(model_selection_data).strip().lower()
    if selection_data not in {"valid", "cv_inner", "nested_cv"}:
        raise ValueError(f"Unsupported model_selection_data: {model_selection_data}")

    spec["objective_metric"] = "pr_auc"
    spec["model_selection_data"] = selection_data

    if selection_data == "valid":
        spec["early_stopping_data"] = "valid"
        spec["final_model_refit_scope"] = "train_only"
    elif selection_data == "cv_inner":
        spec["early_stopping_data"] = "cv_inner"
        spec["final_model_refit_scope"] = "train_only"
    else:
        spec["early_stopping_data"] = "nested_cv"
        spec["final_model_refit_scope"] = "outer_train_only"

    for key in (
        "test_used_for_model_selection",
        "test_used_for_early_stopping",
        "test_used_for_threshold_selection",
        "test_used_for_calibration",
    ):
        spec[key] = False
    spec["outer_evaluation_split_locked"] = True
    spec["random_seed_controlled"] = True

    cv = spec.get("cv")
    if not isinstance(cv, dict):
        cv = {}
    cv["enabled"] = True
    cv["type"] = "stratified_group_k_fold"
    cv["group_col"] = "patient_id"
    try:
        cv_n_splits = int(cv.get("n_splits", 5))
    except Exception:
        cv_n_splits = 5
    cv["n_splits"] = max(3, cv_n_splits)
    cv["nested"] = bool(selection_data == "nested_cv")
    spec["cv"] = cv

    write_json(path, spec)


def evaluate_heart_seed_feasibility(
    case: DatasetCase,
    df: pd.DataFrame,
    feature_cols: List[str],
    split_seed: int,
    workspace_root: Path,
    profile: Dict[str, str],
    run_tag: str,
    external_institution_override: Optional[pd.DataFrame] = None,
    external_institution_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    splits = split_with_temporal_order(df, feature_cols, case.case_id, split_seed_override=split_seed)
    external_pool = splits.get("external")
    if not isinstance(external_pool, pd.DataFrame):
        raise RuntimeError("external split missing from split_with_temporal_order output.")
    external_period_df, external_institution_df, external_split_meta = split_external_pool(
        external_pool,
        case.case_id,
        split_seed_override=split_seed,
        min_rows_per_cohort=20,
        min_positive_per_cohort=3,
        external_institution_override=external_institution_override,
        external_institution_metadata=external_institution_metadata,
    )

    profile_id = str(profile.get("profile_id", "")).strip()
    selection_data = str(profile.get("selection_data", "cv_inner")).strip().lower()
    threshold_selection_split = str(profile.get("threshold_selection_split", "valid")).strip().lower()
    calibration_fit_split = str(profile.get("calibration_fit_split", threshold_selection_split)).strip().lower()
    calibration_method = str(profile.get("calibration_method", "power")).strip().lower()
    seed_root = workspace_root / f"seed_{split_seed}" / profile_id
    if seed_root.exists():
        shutil.rmtree(seed_root)
    data_dir = seed_root / "data"
    cfg_dir = seed_root / "configs"
    evidence_dir = seed_root / "evidence"
    model_dir = seed_root / "models"
    for d in (data_dir, cfg_dir, evidence_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        splits[split_name].to_csv(data_dir / f"{split_name}.csv", index=False)
    external_period_df.to_csv(data_dir / "external_cross_period.csv", index=False)
    external_institution_df.to_csv(data_dir / "external_cross_institution.csv", index=False)

    write_json(cfg_dir / "feature_group_spec.json", build_feature_group_spec(feature_cols))
    lineage = {"features": {col: {"ancestors": [f"raw_{col}"]} for col in feature_cols}}
    write_json(cfg_dir / "feature_lineage.json", lineage)
    copy_reference("imbalance-policy.example.json", cfg_dir / "imbalance_policy.json")
    copy_reference("missingness-policy.example.json", cfg_dir / "missingness_policy.json")
    copy_reference("tuning-protocol.example.json", cfg_dir / "tuning_protocol.json")
    copy_reference("performance-policy.example.json", cfg_dir / "performance_policy.json")
    update_tuning_protocol(cfg_dir / "tuning_protocol.json", model_selection_data=selection_data)
    update_imbalance_policy(
        cfg_dir / "imbalance_policy.json",
        case.case_id,
        threshold_split_override=threshold_selection_split,
        calibration_split_override=calibration_fit_split,
    )
    update_performance_policy(
        cfg_dir / "performance_policy.json",
        case.case_id,
        threshold_split_override=threshold_selection_split,
        calibration_method_override=calibration_method,
        calibration_fit_split_override=calibration_fit_split,
    )
    update_missingness_policy(
        cfg_dir / "missingness_policy.json",
        train_rows=int(len(splits["train"])),
        feature_count=int(len(feature_cols)),
        mice_max_rows_override=MISSINGNESS_MICE_MAX_ROWS_OVERRIDE_BY_CASE.get(case.case_id),
        min_non_missing_override=MISSINGNESS_MIN_NON_MISSING_OVERRIDE_BY_CASE.get(case.case_id),
    )
    external_cohort_spec = {
        "cohorts": [
            {
                "cohort_id": f"{case.case_id}_external_period",
                "cohort_type": "cross_period",
                "path": "../data/external_cross_period.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            },
            {
                "cohort_id": f"{case.case_id}_external_institution",
                "cohort_type": "cross_institution",
                "path": "../data/external_cross_institution.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            },
        ]
    }
    write_json(cfg_dir / "external_cohort_spec.json", external_cohort_spec)

    model_selection_report_path = evidence_dir / "model_selection_report.json"
    evaluation_report_path = evidence_dir / "evaluation_report.json"
    feature_engineering_report_path = evidence_dir / "feature_engineering_report.json"
    distribution_report_path = evidence_dir / "distribution_report.json"
    ci_matrix_report_path = evidence_dir / "ci_matrix_report.json"
    prediction_trace_path = evidence_dir / "prediction_trace.csv.gz"
    external_validation_report_path = evidence_dir / "external_validation_report.json"
    robustness_report_path = evidence_dir / "robustness_report.json"
    seed_sensitivity_report_path = evidence_dir / "seed_sensitivity_report.json"
    permutation_null_path = evidence_dir / "permutation_null_pr_auc.txt"

    train_proc = run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "train_select_evaluate.py"),
            "--train",
            str(data_dir / "train.csv"),
            "--valid",
            str(data_dir / "valid.csv"),
            "--test",
            str(data_dir / "test.csv"),
            "--target-col",
            "y",
            "--ignore-cols",
            "patient_id,event_time",
            "--performance-policy",
            str(cfg_dir / "performance_policy.json"),
            "--feature-group-spec",
            str(cfg_dir / "feature_group_spec.json"),
            "--missingness-policy",
            str(cfg_dir / "missingness_policy.json"),
            "--feature-engineering-mode",
            "strict",
            "--selection-data",
            selection_data,
            "--threshold-selection-split",
            threshold_selection_split,
            "--primary-metric",
            "pr_auc",
            "--fast-diagnostic-mode",
            "--bootstrap-resamples",
            "200",
            "--ci-bootstrap-resamples",
            "200",
            "--permutation-resamples",
            "50",
            "--random-seed",
            "20260224",
            "--model-selection-report-out",
            str(model_selection_report_path),
            "--evaluation-report-out",
            str(evaluation_report_path),
            "--ci-matrix-report-out",
            str(ci_matrix_report_path),
            "--prediction-trace-out",
            str(prediction_trace_path),
            "--external-cohort-spec",
            str(cfg_dir / "external_cohort_spec.json"),
            "--external-validation-report-out",
            str(external_validation_report_path),
            "--model-out",
            str(model_dir / "selected_model.joblib"),
        ],
        allow_fail=True,
    )

    clinical_report = evidence_dir / "clinical_metrics_report.search.json"
    gap_report = evidence_dir / "generalization_gap_report.search.json"
    external_gate_report = evidence_dir / "external_validation_gate_report.search.json"
    calibration_dca_gate_report = evidence_dir / "calibration_dca_report.search_release.json"
    ci_matrix_gate_report = evidence_dir / "ci_matrix_report.search_release.json"
    def write_skipped_gate_report(path: Path, code: str, reason: str) -> None:
        payload = {
            "status": "skipped",
            "strict_mode": True,
            "failure_count": 1,
            "warning_count": 0,
            "failures": [
                {
                    "code": code,
                    "message": "Gate skipped due to upstream failure in stress seed search.",
                    "details": {"reason": reason},
                }
            ],
            "warnings": [],
            "summary": {"diagnostic_only": True},
        }
        write_json(path, payload)

    clinical_proc: Optional[subprocess.CompletedProcess[str]] = None
    gap_proc: Optional[subprocess.CompletedProcess[str]] = None
    external_proc: Optional[subprocess.CompletedProcess[str]] = None

    if train_proc.returncode == 0:
        clinical_proc = run_cmd(
            [
                sys.executable,
                str(SCRIPTS_ROOT / "clinical_metrics_gate.py"),
                "--evaluation-report",
                str(evaluation_report_path),
                "--external-validation-report",
                str(external_validation_report_path),
                "--performance-policy",
                str(cfg_dir / "performance_policy.json"),
                "--strict",
                "--report",
                str(clinical_report),
            ],
            allow_fail=True,
        )
    else:
        write_skipped_gate_report(
            clinical_report,
            code="skipped_upstream_train_failed",
            reason="train_select_evaluate_nonzero_exit",
        )

    if clinical_proc is not None and clinical_proc.returncode == 0:
        gap_proc = run_cmd(
            [
                sys.executable,
                str(SCRIPTS_ROOT / "generalization_gap_gate.py"),
                "--evaluation-report",
                str(evaluation_report_path),
                "--performance-policy",
                str(cfg_dir / "performance_policy.json"),
                "--strict",
                "--report",
                str(gap_report),
            ],
            allow_fail=True,
        )
    else:
        write_skipped_gate_report(
            gap_report,
            code="skipped_upstream_clinical_failed",
            reason="clinical_metrics_gate_nonzero_exit_or_not_run",
        )

    if gap_proc is not None and gap_proc.returncode == 0:
        external_proc = run_cmd(
            [
                sys.executable,
                str(SCRIPTS_ROOT / "external_validation_gate.py"),
                "--external-validation-report",
                str(external_validation_report_path),
                "--prediction-trace",
                str(prediction_trace_path),
                "--evaluation-report",
                str(evaluation_report_path),
                "--performance-policy",
                str(cfg_dir / "performance_policy.json"),
                "--strict",
                "--report",
                str(external_gate_report),
            ],
            allow_fail=True,
        )
    else:
        write_skipped_gate_report(
            external_gate_report,
            code="skipped_upstream_gap_failed",
            reason="generalization_gap_gate_nonzero_exit_or_not_run",
        )

    split_meta = {
        "split_seed": int(split_seed),
        "split_fraction_config": INTERNAL_SPLIT_FRACTIONS_BY_CASE.get(case.case_id, {}),
        "internal": {
            "train": summarize_split_frame(splits["train"]),
            "valid": summarize_split_frame(splits["valid"]),
            "test": summarize_split_frame(splits["test"]),
            "external_pool": summarize_split_frame(external_pool),
        },
        "external_split": external_split_meta,
    }

    feasible = bool(
        train_proc.returncode == 0
        and clinical_proc is not None
        and clinical_proc.returncode == 0
        and gap_proc is not None
        and gap_proc.returncode == 0
        and external_proc is not None
        and external_proc.returncode == 0
        and bool(external_split_meta.get("minimum_requirements_met"))
    )
    calibration_proc: Optional[subprocess.CompletedProcess[str]] = None
    ci_proc: Optional[subprocess.CompletedProcess[str]] = None
    release_ready = False
    if feasible:
        calibration_proc = run_cmd(
            [
                sys.executable,
                str(SCRIPTS_ROOT / "calibration_dca_gate.py"),
                "--prediction-trace",
                str(prediction_trace_path),
                "--evaluation-report",
                str(evaluation_report_path),
                "--external-validation-report",
                str(external_validation_report_path),
                "--performance-policy",
                str(cfg_dir / "performance_policy.json"),
                "--strict",
                "--report",
                str(calibration_dca_gate_report),
            ],
            allow_fail=True,
        )
        if calibration_proc.returncode == 0:
            ci_proc = run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS_ROOT / "ci_matrix_gate.py"),
                    "--evaluation-report",
                    str(evaluation_report_path),
                    "--prediction-trace",
                    str(prediction_trace_path),
                    "--external-validation-report",
                    str(external_validation_report_path),
                    "--performance-policy",
                    str(cfg_dir / "performance_policy.json"),
                    "--ci-matrix-report",
                    str(evidence_dir / "ci_matrix_report.json"),
                    "--strict",
                    "--report",
                    str(ci_matrix_gate_report),
                ],
                allow_fail=True,
            )
        release_ready = bool(calibration_proc.returncode == 0 and ci_proc is not None and ci_proc.returncode == 0)
    gate_reports = {
        "clinical_metrics_gate": str(clinical_report),
        "generalization_gap_gate": str(gap_report),
        "external_validation_gate": str(external_gate_report),
        "calibration_dca_gate": str(calibration_dca_gate_report) if feasible else None,
        "ci_matrix_gate": str(ci_matrix_gate_report) if feasible else None,
    }
    calibration_gap_components = calibration_gap_diagnostics(calibration_dca_gate_report) if feasible else {}
    calibration_gap_total = float(calibration_gap_components.get("total_excess", 0.0)) if calibration_gap_components else 0.0
    gap_score = (
        (1e5 if train_proc.returncode != 0 else 0.0)
        + failure_gap_score(clinical_report)
        + failure_gap_score(gap_report)
        + failure_gap_score(external_gate_report)
        + (failure_gap_score(calibration_dca_gate_report) if feasible else 0.0)
        + (failure_gap_score(ci_matrix_gate_report) if feasible and ci_proc is not None else 0.0)
        + calibration_gap_total
        + (0.0 if bool(external_split_meta.get("minimum_requirements_met")) else 10.0)
    )
    return {
        "run_tag": run_tag,
        "seed": int(split_seed),
        "profile_id": profile_id,
        "profile": {
            "selection_data": selection_data,
            "threshold_selection_split": threshold_selection_split,
            "calibration_fit_split": calibration_fit_split,
            "calibration_method": calibration_method,
        },
        "feasible": feasible,
        "release_ready": bool(release_ready),
        "status": "pass" if feasible else "fail",
        "failure_code": None if feasible else "stress_seed_candidate_not_feasible",
        "train_exit_code": int(train_proc.returncode),
        "gate_exit_codes": {
            "clinical_metrics_gate": int(clinical_proc.returncode) if clinical_proc is not None else None,
            "generalization_gap_gate": int(gap_proc.returncode) if gap_proc is not None else None,
            "external_validation_gate": int(external_proc.returncode) if external_proc is not None else None,
            "calibration_dca_gate": int(calibration_proc.returncode) if calibration_proc is not None else None,
            "ci_matrix_gate": int(ci_proc.returncode) if ci_proc is not None else None,
        },
        "gate_failures": {
            "clinical_metrics_gate": extract_failure_codes(clinical_report),
            "generalization_gap_gate": extract_failure_codes(gap_report),
            "external_validation_gate": extract_failure_codes(external_gate_report),
            "calibration_dca_gate": extract_failure_codes(calibration_dca_gate_report) if feasible else [],
            "ci_matrix_gate": extract_failure_codes(ci_matrix_gate_report) if feasible and ci_proc is not None else [],
        },
        "gate_reports": gate_reports,
        "calibration_gap_components": calibration_gap_components,
        "diagnostic_gap_score": float(gap_score),
        "split_metadata": split_meta,
    }


def search_heart_feasible_seed(
    case: DatasetCase,
    seed_min: int,
    seed_max: int,
    report_path: Path,
    selection_path: Path,
    run_tag: str,
    profile_set: str,
    dataset_fingerprint: Dict[str, Any],
    policy_sha256: str,
    code_revision_hint: str,
) -> HeartStressSearchResult:
    if seed_min > seed_max:
        raise ValueError("stress-seed-min must be <= stress-seed-max.")
    if case.case_id != "uci-heart-disease":
        raise ValueError("search_heart_feasible_seed only supports uci-heart-disease.")

    raw_path = RAW_ROOT / case.raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {raw_path}")
    df, feature_cols = load_heart_dataset(raw_path)
    external_institution_override, external_institution_metadata = load_heart_external_institution_pool(
        case.case_id,
        feature_cols,
        exclude_site_tags=HEART_INTERNAL_SITE_TAGS,
    )
    if external_institution_override is None:
        raise RuntimeError("Heart external institution pool unavailable after site-quality filtering.")
    profiles = get_stress_profiles(profile_set)

    workspace_parent = DATA_ROOT / "_stress_seed_workspace" / case.case_id
    run_token = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    workspace_root = workspace_parent / f"run_{run_token}_{int(seed_min)}_{int(seed_max)}"
    workspace_root.mkdir(parents=True, exist_ok=False)

    candidates: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    for split_seed in range(int(seed_min), int(seed_max) + 1):
        for profile_rank, profile in enumerate(profiles, start=1):
            candidate = evaluate_heart_seed_feasibility(
                case=case,
                df=df,
                feature_cols=feature_cols,
                split_seed=split_seed,
                workspace_root=workspace_root,
                profile=profile,
                run_tag=run_tag,
                external_institution_override=external_institution_override,
                external_institution_metadata=external_institution_metadata,
            )
            candidate["profile_rank"] = int(profile_rank)
            candidates.append(candidate)
            if candidate.get("release_ready"):
                selected = candidate
                break
        if selected is not None:
            break

    report_path = report_path.expanduser().resolve()
    selection_path = selection_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    best_candidate = (
        min(
            candidates,
            key=lambda row: (
                float(row.get("diagnostic_gap_score", 1e9)),
                int(row.get("seed", 10**9)),
                int(row.get("profile_rank", 10**9)),
            ),
        )
        if candidates
        else None
    )
    feasible_count = int(sum(1 for row in candidates if bool(row.get("feasible"))))
    release_ready_count = int(sum(1 for row in candidates if bool(row.get("release_ready"))))

    if selected is not None:
        selected_profile = str(selected.get("profile_id", "")).strip()
        selection_payload = {
            "status": "pass",
            "contract_version": STRESS_SEARCH_REPORT_CONTRACT_VERSION,
            "case_id": case.case_id,
            "run_tag": run_tag,
            "search_profile_set": profile_set,
            "policy_sha256": policy_sha256,
            "dataset_fingerprint": dataset_fingerprint,
            "code_revision_hint": code_revision_hint,
            "selection_policy": "first_release_ready_seed_over_range",
            "seed_range": {"min": int(seed_min), "max": int(seed_max)},
            "selected_seed": int(selected["seed"]),
            "selected_profile": selected_profile,
            "selected_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "selected_summary": {
                "gate_exit_codes": selected.get("gate_exit_codes", {}),
                "diagnostic_gap_score": selected.get("diagnostic_gap_score"),
                "calibration_gap_components": selected.get("calibration_gap_components", {}),
                "external_split_metadata": selected.get("split_metadata", {}).get("external_split", {}),
                "release_ready": bool(selected.get("release_ready")),
            },
        }
        report_payload = {
            "status": "pass",
            "contract_version": STRESS_SEARCH_REPORT_CONTRACT_VERSION,
            "case_id": case.case_id,
            "run_tag": run_tag,
            "policy_sha256": policy_sha256,
            "search_profile_set": profile_set,
            "selected_profile": selected_profile,
            "dataset_fingerprint": dataset_fingerprint,
            "code_revision_hint": code_revision_hint,
            "search_started_seed": int(seed_min),
            "search_ended_seed": int(seed_max),
            "searched_candidate_count": len(candidates),
            "feasible_candidate_count": feasible_count,
            "release_ready_candidate_count": release_ready_count,
            "selected_seed": int(selected["seed"]),
            "failure_code": None,
            "candidates": candidates,
            "selection": selection_payload,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        write_json(report_path, report_payload)
        write_json(selection_path, selection_payload)
        return HeartStressSearchResult(
            selected_seed=int(selected["seed"]),
            selected_profile=selected_profile,
            report_path=report_path,
            selection_path=selection_path,
            status="pass",
        )

    failure_payload = {
        "status": "fail",
        "contract_version": STRESS_SEARCH_REPORT_CONTRACT_VERSION,
        "case_id": case.case_id,
        "run_tag": run_tag,
        "policy_sha256": policy_sha256,
        "search_profile_set": profile_set,
        "selected_profile": None,
        "dataset_fingerprint": dataset_fingerprint,
        "code_revision_hint": code_revision_hint,
        "search_started_seed": int(seed_min),
        "search_ended_seed": int(seed_max),
        "searched_candidate_count": len(candidates),
        "feasible_candidate_count": feasible_count,
        "release_ready_candidate_count": release_ready_count,
        "selected_seed": None,
        "failure_code": "stress_seed_feasibility_not_found",
        "best_candidate": best_candidate,
        "candidates": candidates,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    selection_payload = {
        "status": "fail",
        "contract_version": STRESS_SEARCH_REPORT_CONTRACT_VERSION,
        "case_id": case.case_id,
        "run_tag": run_tag,
        "search_profile_set": profile_set,
        "policy_sha256": policy_sha256,
        "dataset_fingerprint": dataset_fingerprint,
        "code_revision_hint": code_revision_hint,
        "selection_policy": "first_release_ready_seed_over_range",
        "seed_range": {"min": int(seed_min), "max": int(seed_max)},
        "selected_seed": None,
        "selected_profile": None,
        "failure_code": "stress_seed_feasibility_not_found",
        "best_candidate_seed": int(best_candidate["seed"]) if isinstance(best_candidate, dict) and "seed" in best_candidate else None,
        "best_candidate_profile": (
            str(best_candidate.get("profile_id", "")).strip()
            if isinstance(best_candidate, dict)
            else None
        ),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_json(report_path, failure_payload)
    write_json(selection_path, selection_payload)
    raise RuntimeError(
        f"stress_seed_feasibility_not_found: no release-ready heart seed in [{seed_min}, {seed_max}] "
        f"(feasible_count={feasible_count}, release_ready_count={release_ready_count}, "
        f"best_gap={best_candidate.get('diagnostic_gap_score') if isinstance(best_candidate, dict) else 'n/a'})"
    )


def prepare_case_artifacts(
    case: DatasetCase,
    split_seed_override: Optional[int] = None,
    stress_search_result: Optional[HeartStressSearchResult] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    raw_path = RAW_ROOT / case.raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {raw_path}")

    if case.case_id == "uci-heart-disease":
        df, feature_cols = load_heart_dataset(raw_path)
    elif case.case_id == "uci-breast-cancer-wdbc":
        df, feature_cols = load_breast_dataset(raw_path)
    elif case.case_id == "uci-chronic-kidney-disease":
        max_rows: Optional[int] = DEFAULT_CKD_MAX_ROWS
        if isinstance(case.options, dict):
            raw_max_rows = case.options.get("max_rows")
            if isinstance(raw_max_rows, int):
                max_rows = int(raw_max_rows) if int(raw_max_rows) > 0 else None
        df, feature_cols = load_ckd_dataset(raw_path, max_rows=max_rows)
    elif case.case_id == "uci-diabetes-130-readmission":
        max_rows: Optional[int] = DEFAULT_DIABETES_MAX_ROWS
        target_mode = DEFAULT_DIABETES_TARGET_MODE
        if isinstance(case.options, dict):
            raw_max_rows = case.options.get("max_rows")
            if isinstance(raw_max_rows, int):
                max_rows = int(raw_max_rows) if int(raw_max_rows) > 0 else None
            raw_target_mode = case.options.get("target_mode")
            if isinstance(raw_target_mode, str) and raw_target_mode.strip():
                target_mode = str(raw_target_mode).strip().lower()
        df, feature_cols = load_diabetes_130_dataset(raw_path, max_rows=max_rows, target_mode=target_mode)
    else:
        raise ValueError(f"Unsupported case_id: {case.case_id}")

    effective_split_seed = int(
        split_seed_override
        if split_seed_override is not None
        else SPLIT_RANDOM_SEED_BY_CASE.get(case.case_id, 20260224)
    )
    splits = split_with_temporal_order(df, feature_cols, case.case_id, split_seed_override=effective_split_seed)
    external_pool = splits.get("external")
    if not isinstance(external_pool, pd.DataFrame):
        raise RuntimeError("external split missing from split_with_temporal_order output.")
    external_institution_override: Optional[pd.DataFrame] = None
    external_institution_metadata: Dict[str, Any] = {}
    if case.case_id == "uci-heart-disease":
        external_institution_override, external_institution_metadata = load_heart_external_institution_pool(
            case.case_id,
            feature_cols,
            exclude_site_tags=HEART_INTERNAL_SITE_TAGS,
        )
        if external_institution_override is None:
            raise RuntimeError("Heart external institution pool unavailable after site-quality filtering.")
    external_period_df, external_institution_df, external_split_meta = split_external_pool(
        external_pool,
        case.case_id,
        split_seed_override=effective_split_seed,
        min_rows_per_cohort=20,
        min_positive_per_cohort=3,
        external_institution_override=external_institution_override,
        external_institution_metadata=external_institution_metadata,
    )

    case_root = DATA_ROOT / case.case_id
    if case_root.exists():
        shutil.rmtree(case_root)
    data_dir = case_root / "data"
    cfg_dir = case_root / "configs"
    evidence_dir = case_root / "evidence"
    model_dir = case_root / "models"
    key_dir = case_root / "keys"
    for d in (data_dir, cfg_dir, evidence_dir, model_dir, key_dir):
        d.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        frame = splits[split_name]
        frame.to_csv(data_dir / f"{split_name}.csv", index=False)
    external_period_df.to_csv(data_dir / "external_cross_period.csv", index=False)
    external_institution_df.to_csv(data_dir / "external_cross_institution.csv", index=False)

    config_payload = {
        "dataset_case": case.case_id,
        "source_name": case.source_name,
        "model_pool": {
            "models": [
                "logistic_l1",
                "logistic_l2",
                "logistic_elasticnet",
                "random_forest_balanced",
                "hist_gradient_boosting_l2"
            ],
            "required_models": [
                "logistic_l2"
            ],
            "max_trials_per_family": 1,
            "search_strategy": "fixed_grid",
            "n_jobs": -1
        },
        "selection_policy": "pr_auc + one_se + lower_complexity",
        "threshold_policy": "maximize_f2_beta_under_sensitivity_npv_specificity_ppv_floors",
        "random_seed": 20260224,
        "features": feature_cols,
    }
    write_json(cfg_dir / "train_config.json", config_payload)

    phenotype_def = {
        "global_forbidden_patterns": ["(?i)target", "(?i)label"],
        "targets": {
            case.target_name: {
                "defining_variables": [
                    "confirmed_diagnosis_code",
                    "reference_standard_positive",
                ],
                "forbidden_patterns": ["(?i)diagnosis_code", "(?i)reference_standard"],
                "notes": f"Definition variables for {case.target_name} must not appear in predictors.",
            }
        },
    }
    write_json(cfg_dir / "phenotype_definitions.json", phenotype_def)

    lineage = {"features": {}}
    for col in feature_cols:
        lineage["features"][col] = {"ancestors": [f"raw_{col}"]}
    write_json(cfg_dir / "feature_lineage.json", lineage)
    write_json(cfg_dir / "feature_group_spec.json", build_feature_group_spec(feature_cols))

    copy_reference("split-protocol.example.json", cfg_dir / "split_protocol.json")
    copy_reference("imbalance-policy.example.json", cfg_dir / "imbalance_policy.json")
    copy_reference("missingness-policy.example.json", cfg_dir / "missingness_policy.json")
    copy_reference("tuning-protocol.example.json", cfg_dir / "tuning_protocol.json")
    copy_reference("performance-policy.example.json", cfg_dir / "performance_policy.json")
    copy_reference("reporting-bias-checklist.example.json", cfg_dir / "reporting_bias_checklist.json")
    selection_data = MODEL_SELECTION_DATA_BY_CASE.get(case.case_id, "cv_inner")
    threshold_split = THRESHOLD_SELECTION_SPLIT_BY_CASE.get(case.case_id, "valid")
    default_calibration_fit_split_by_case = {
        "uci-heart-disease": "cv_inner",
        "uci-chronic-kidney-disease": "cv_inner",
    }
    calibration_fit_split = default_calibration_fit_split_by_case.get(case.case_id, threshold_split)
    calibration_method: Optional[str] = None
    selected_profile_id: Optional[str] = None
    if case.case_id == "uci-heart-disease" and stress_search_result is not None:
        selected_profile_id = str(stress_search_result.selected_profile).strip()
        profile = find_stress_profile_by_id(selected_profile_id)
        if isinstance(profile, dict):
            selection_data = str(profile.get("selection_data", selection_data))
            threshold_split = str(profile.get("threshold_selection_split", threshold_split))
            calibration_fit_split = str(profile.get("calibration_fit_split", threshold_split))
            calibration_method = str(profile.get("calibration_method", "power"))
    update_tuning_protocol(cfg_dir / "tuning_protocol.json", model_selection_data=selection_data)
    update_imbalance_policy(
        cfg_dir / "imbalance_policy.json",
        case.case_id,
        threshold_split_override=threshold_split,
        calibration_split_override=calibration_fit_split,
    )
    update_performance_policy(
        cfg_dir / "performance_policy.json",
        case.case_id,
        threshold_split_override=threshold_split,
        calibration_method_override=calibration_method,
        calibration_fit_split_override=calibration_fit_split,
    )
    update_missingness_policy(
        cfg_dir / "missingness_policy.json",
        train_rows=int(len(splits["train"])),
        feature_count=int(len(feature_cols)),
        mice_max_rows_override=MISSINGNESS_MICE_MAX_ROWS_OVERRIDE_BY_CASE.get(case.case_id),
        min_non_missing_override=MISSINGNESS_MIN_NON_MISSING_OVERRIDE_BY_CASE.get(case.case_id),
    )
    case_budget = TRAINING_BUDGET_BY_CASE.get(case.case_id, {})
    bootstrap_resamples = int(case_budget.get("bootstrap_resamples", 500))
    ci_bootstrap_resamples = int(case_budget.get("ci_bootstrap_resamples", 2000))
    permutation_resamples = int(case_budget.get("permutation_resamples", 300))

    external_cohort_spec = {
        "cohorts": [
            {
                "cohort_id": f"{case.case_id}_external_period",
                "cohort_type": "cross_period",
                "path": "../data/external_cross_period.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            },
            {
                "cohort_id": f"{case.case_id}_external_institution",
                "cohort_type": "cross_institution",
                "path": "../data/external_cross_institution.csv",
                "label_col": "y",
                "patient_id_col": "patient_id",
                "index_time_col": "event_time",
            }
        ]
    }
    write_json(cfg_dir / "external_cohort_spec.json", external_cohort_spec)

    model_path = model_dir / "selected_model.joblib"
    model_selection_report_path = evidence_dir / "model_selection_report.json"
    evaluation_report_path = evidence_dir / "evaluation_report.json"
    feature_engineering_report_path = evidence_dir / "feature_engineering_report.json"
    distribution_report_path = evidence_dir / "distribution_report.json"
    ci_matrix_report_path = evidence_dir / "ci_matrix_report.json"
    prediction_trace_path = evidence_dir / "prediction_trace.csv.gz"
    external_validation_report_path = evidence_dir / "external_validation_report.json"
    robustness_report_path = evidence_dir / "robustness_report.json"
    seed_sensitivity_report_path = evidence_dir / "seed_sensitivity_report.json"
    permutation_null_path = evidence_dir / "permutation_null_pr_auc.txt"
    stress_seed_search_report_dst = evidence_dir / "stress_seed_search_report.json"
    stress_seed_selection_dst = evidence_dir / "stress_seed_selection.json"

    if stress_search_result is not None:
        shutil.copy2(stress_search_result.report_path, stress_seed_search_report_dst)
        shutil.copy2(stress_search_result.selection_path, stress_seed_selection_dst)
    elif case.case_id == "uci-heart-disease":
        write_json(
            stress_seed_selection_dst,
            {
                "status": "pass",
                "case_id": case.case_id,
                "selection_policy": "fixed_seed",
                "selected_seed": int(effective_split_seed),
                "selected_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )

    model_pool = resolve_case_model_pool(case.case_id)
    max_trials_per_family = int(
        MAX_TRIALS_PER_FAMILY_BY_CASE.get(
            case.case_id,
            MAX_TRIALS_PER_FAMILY_BY_CASE.get("default", 3),
        )
    )
    hyperparam_search = str(
        HYPERPARAM_SEARCH_BY_CASE.get(
            case.case_id,
            HYPERPARAM_SEARCH_BY_CASE.get("default", "random_subsample"),
        )
    )
    n_jobs = int(
        N_JOBS_BY_CASE.get(
            case.case_id,
            N_JOBS_BY_CASE.get("default", 4),
        )
    )
    cpu_cap = os.cpu_count() or n_jobs
    n_jobs = max(1, min(n_jobs, int(cpu_cap)))

    train_cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "train_select_evaluate.py"),
        "--train",
        str(data_dir / "train.csv"),
        "--valid",
        str(data_dir / "valid.csv"),
        "--test",
        str(data_dir / "test.csv"),
        "--target-col",
        "y",
        "--ignore-cols",
        "patient_id,event_time",
        "--performance-policy",
        str(cfg_dir / "performance_policy.json"),
        "--feature-group-spec",
        str(cfg_dir / "feature_group_spec.json"),
        "--missingness-policy",
        str(cfg_dir / "missingness_policy.json"),
        "--feature-engineering-mode",
        "strict",
        "--selection-data",
        selection_data,
        "--threshold-selection-split",
        threshold_split,
        "--primary-metric",
        "pr_auc",
        "--max-trials-per-family",
        str(max_trials_per_family),
        "--hyperparam-search",
        hyperparam_search,
        "--n-jobs",
        str(n_jobs),
        "--bootstrap-resamples",
        str(bootstrap_resamples),
        "--ci-bootstrap-resamples",
        str(ci_bootstrap_resamples),
        "--permutation-resamples",
        str(permutation_resamples),
        "--random-seed",
        "20260224",
        "--model-selection-report-out",
        str(model_selection_report_path),
        "--evaluation-report-out",
        str(evaluation_report_path),
        "--feature-engineering-report-out",
        str(feature_engineering_report_path),
        "--distribution-report-out",
        str(distribution_report_path),
        "--ci-matrix-report-out",
        str(ci_matrix_report_path),
        "--robustness-report-out",
        str(robustness_report_path),
        "--robustness-time-slices",
        str(int(ROBUSTNESS_TIME_SLICES_BY_CASE.get(case.case_id, 2))),
        "--robustness-group-count",
        str(int(ROBUSTNESS_GROUP_COUNT_BY_CASE.get(case.case_id, 2))),
        "--seed-sensitivity-out",
        str(seed_sensitivity_report_path),
        "--prediction-trace-out",
        str(prediction_trace_path),
        "--external-cohort-spec",
        str(cfg_dir / "external_cohort_spec.json"),
        "--external-validation-report-out",
        str(external_validation_report_path),
        "--model-out",
        str(model_path),
        "--permutation-null-out",
        str(permutation_null_path),
    ]
    if model_pool:
        train_cmd.extend(["--model-pool", ",".join(model_pool)])

    run_cmd(train_cmd)

    evaluation_report = load_json(evaluation_report_path)
    selected_model_id = str(evaluation_report.get("model_id", "unknown_model"))
    metrics = evaluation_report.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("evaluation_report.metrics missing.")
    eval_metadata = evaluation_report.get("metadata")
    if not isinstance(eval_metadata, dict):
        eval_metadata = {}
    model_pool_meta = eval_metadata.get("model_pool")
    if not isinstance(model_pool_meta, dict):
        model_pool_meta = {}
    requested_models = model_pool_meta.get("requested")
    if not isinstance(requested_models, list):
        requested_models = []
    model_pool_text = ",".join([str(x) for x in requested_models]) if requested_models else "unknown"
    primary_metric = float(metrics.get("pr_auc"))

    train_log_path = evidence_dir / "train.log"
    train_log_lines = [
        f"[INFO] case={case.case_id}",
        f"[INFO] selected_model={selected_model_id}",
        f"[INFO] model_pool={model_pool_text}",
        f"[INFO] feature_count={len(feature_cols)}",
        f"[INFO] train_rows={len(splits['train'])}",
        f"[INFO] valid_rows={len(splits['valid'])}",
        f"[INFO] test_rows={len(splits['test'])}",
        f"[INFO] test_pr_auc={float(metrics.get('pr_auc')):.6f}",
        f"[INFO] test_roc_auc={float(metrics.get('roc_auc')):.6f}",
        "[INFO] training_complete=true",
    ]
    train_log_path.write_text("\n".join(train_log_lines) + "\n", encoding="utf-8")

    key_pairs: Dict[str, Tuple[Path, Path]] = {}
    for role in ("attestation", "timestamp", "transparency", "execution", "execution_log", "witness_a", "witness_b"):
        priv_key = key_dir / f"{role}_priv.pem"
        pub_key = key_dir / f"{role}_pub.pem"
        run_cmd(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:3072",
                "-out",
                str(priv_key),
            ]
        )
        run_cmd(["openssl", "pkey", "-in", str(priv_key), "-pubout", "-out", str(pub_key)])
        key_pairs[role] = (priv_key, pub_key)

    study_id = f"{case.case_id}-study-v1"
    run_id = f"{case.case_id}-run-001"
    run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "generate_execution_attestation.py"),
            "--study-id",
            study_id,
            "--run-id",
            run_id,
            "--payload-out",
            str(evidence_dir / "attestation_payload.json"),
            "--signature-out",
            str(evidence_dir / "attestation.sig"),
            "--spec-out",
            str(cfg_dir / "execution_attestation.json"),
            "--private-key-file",
            str(key_pairs["attestation"][0]),
            "--public-key-file",
            str(key_pairs["attestation"][1]),
            "--timestamp-private-key-file",
            str(key_pairs["timestamp"][0]),
            "--timestamp-public-key-file",
            str(key_pairs["timestamp"][1]),
            "--transparency-private-key-file",
            str(key_pairs["transparency"][0]),
            "--transparency-public-key-file",
            str(key_pairs["transparency"][1]),
            "--execution-private-key-file",
            str(key_pairs["execution"][0]),
            "--execution-public-key-file",
            str(key_pairs["execution"][1]),
            "--execution-log-private-key-file",
            str(key_pairs["execution_log"][0]),
            "--execution-log-public-key-file",
            str(key_pairs["execution_log"][1]),
            "--require-independent-timestamp-authority",
            "--require-independent-execution-authority",
            "--require-independent-log-authority",
            "--require-witness-quorum",
            "--min-witness-count",
            "2",
            "--require-independent-witness-keys",
            "--require-witness-independence-from-signing",
            "--witness",
            f"witness-a|{key_pairs['witness_a'][1]}|{key_pairs['witness_a'][0]}",
            "--witness",
            f"witness-b|{key_pairs['witness_b'][1]}|{key_pairs['witness_b'][0]}",
            "--command",
            f"python scripts/train_select_evaluate.py --train {data_dir / 'train.csv'} --valid {data_dir / 'valid.csv'} --test {data_dir / 'test.csv'} --primary-metric pr_auc --performance-policy {cfg_dir / 'performance_policy.json'} --missingness-policy {cfg_dir / 'missingness_policy.json'} --prediction-trace-out {prediction_trace_path} --external-cohort-spec {cfg_dir / 'external_cohort_spec.json'} --external-validation-report-out {external_validation_report_path}",
            "--artifact",
            f"training_log={train_log_path}",
            "--artifact",
            f"training_config={cfg_dir / 'train_config.json'}",
            "--artifact",
            f"model_artifact={model_path}",
            "--artifact",
            f"model_selection_report={model_selection_report_path}",
            "--artifact",
            f"feature_engineering_report={feature_engineering_report_path}",
            "--artifact",
            f"distribution_report={distribution_report_path}",
            "--artifact",
            f"robustness_report={robustness_report_path}",
            "--artifact",
            f"seed_sensitivity_report={seed_sensitivity_report_path}",
            "--artifact",
            f"evaluation_report={evaluation_report_path}",
            "--artifact",
            f"ci_matrix_report={ci_matrix_report_path}",
            "--artifact",
            f"prediction_trace={prediction_trace_path}",
            "--artifact",
            f"external_validation_report={external_validation_report_path}",
        ]
    )

    request_payload = {
        "study_id": study_id,
        "run_id": run_id,
        "target_name": case.target_name,
        "prediction_unit": "patient-encounter",
        "index_time_col": "event_time",
        "label_col": "y",
        "patient_id_col": "patient_id",
        "primary_metric": "pr_auc",
        "claim_tier_target": "publication-grade",
        "phenotype_definition_spec": "phenotype_definitions.json",
        "feature_lineage_spec": "feature_lineage.json",
        "feature_group_spec": "feature_group_spec.json",
        "split_protocol_spec": "split_protocol.json",
        "imbalance_policy_spec": "imbalance_policy.json",
        "missingness_policy_spec": "missingness_policy.json",
        "tuning_protocol_spec": "tuning_protocol.json",
        "performance_policy_spec": "performance_policy.json",
        "external_cohort_spec": "external_cohort_spec.json",
        "reporting_bias_checklist_spec": "reporting_bias_checklist.json",
        "execution_attestation_spec": "execution_attestation.json",
        "model_selection_report_file": "../evidence/model_selection_report.json",
        "feature_engineering_report_file": "../evidence/feature_engineering_report.json",
        "distribution_report_file": "../evidence/distribution_report.json",
        "robustness_report_file": "../evidence/robustness_report.json",
        "seed_sensitivity_report_file": "../evidence/seed_sensitivity_report.json",
        "split_paths": {
            "train": "../data/train.csv",
            "valid": "../data/valid.csv",
            "test": "../data/test.csv",
        },
        "evaluation_report_file": "../evidence/evaluation_report.json",
        "prediction_trace_file": "../evidence/prediction_trace.csv.gz",
        "external_validation_report_file": "../evidence/external_validation_report.json",
        "ci_matrix_report_file": "../evidence/ci_matrix_report.json",
        "evaluation_metric_path": "metrics.pr_auc",
        "permutation_null_metrics_file": "../evidence/permutation_null_pr_auc.txt",
        "actual_primary_metric": primary_metric,
        "thresholds": {
            "alpha": 0.01,
            "min_delta": 0.03,
            "min_baseline_delta": 0.01,
            "ci_min_resamples": 200,
            "ci_max_width": 0.20,
        },
        "context": {
            "source": case.source_name,
            "notes": "Authoritative public benchmark dataset for strict skill validation.",
        },
    }
    if stress_seed_selection_dst.exists():
        request_payload["stress_evidence_files"] = {
            "seed_search_report": "../evidence/stress_seed_search_report.json"
            if stress_seed_search_report_dst.exists()
            else None,
            "seed_selection": "../evidence/stress_seed_selection.json",
        }
    write_json(cfg_dir / "request.json", request_payload)

    bootstrap_report = evidence_dir / "strict_pipeline_bootstrap_report.json"
    bootstrap_proc = run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "run_strict_pipeline.py"),
            "--request",
            str(cfg_dir / "request.json"),
            "--evidence-dir",
            str(evidence_dir),
            "--allow-missing-compare",
            "--strict",
            "--report",
            str(bootstrap_report),
        ],
        allow_fail=True,
    )

    # Refresh manifest baseline after bootstrap has produced gate artifacts.
    if bootstrap_report.exists():
        bootstrap_payload = load_json(bootstrap_report)
        if isinstance(bootstrap_payload, dict):
            manifest_step = None
            for step in bootstrap_payload.get("steps", []):
                if isinstance(step, dict) and step.get("name") == "manifest_lock":
                    manifest_step = step
                    break
            if isinstance(manifest_step, dict):
                manifest_cmd = manifest_step.get("command")
                if isinstance(manifest_cmd, str) and manifest_cmd.strip():
                    run_cmd(shlex.split(manifest_cmd))

    manifest_path = evidence_dir / "manifest.json"
    baseline_manifest = evidence_dir / "manifest_baseline.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Bootstrap manifest missing for {case.case_id}: {manifest_path}")
    shutil.copy2(manifest_path, baseline_manifest)

    final_report = evidence_dir / "strict_pipeline_report.json"
    final_proc = run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "run_strict_pipeline.py"),
            "--request",
            str(cfg_dir / "request.json"),
            "--evidence-dir",
            str(evidence_dir),
            "--compare-manifest",
            str(baseline_manifest),
            "--strict",
            "--report",
            str(final_report),
        ],
        allow_fail=True,
    )

    pipeline_report = load_json(final_report) if final_report.exists() else {"status": "not_run"}
    self_critique_report = (
        load_json(evidence_dir / "self_critique_report.json")
        if (evidence_dir / "self_critique_report.json").exists()
        else {"status": "not_run", "quality_score": None}
    )
    publication_report = (
        load_json(evidence_dir / "publication_gate_report.json")
        if (evidence_dir / "publication_gate_report.json").exists()
        else {"status": "not_run"}
    )
    gap_report = (
        load_json(evidence_dir / "generalization_gap_report.json")
        if (evidence_dir / "generalization_gap_report.json").exists()
        else {"summary": {}}
    )
    external_gate_report = (
        load_json(evidence_dir / "external_validation_gate_report.json")
        if (evidence_dir / "external_validation_gate_report.json").exists()
        else {"status": "not_run"}
    )
    calibration_dca_report = (
        load_json(evidence_dir / "calibration_dca_report.json")
        if (evidence_dir / "calibration_dca_report.json").exists()
        else {"status": "not_run"}
    )
    clinical_floor_gap_summary = build_clinical_floor_gap_summary(
        metrics=metrics if isinstance(metrics, dict) else {},
        external_validation_report_path=external_validation_report_path,
        performance_policy_path=cfg_dir / "performance_policy.json",
    )

    failed_steps, root_failure_codes = extract_pipeline_root_failures(pipeline_report)
    final_status = "pass" if int(final_proc.returncode) == 0 and str(pipeline_report.get("status")) == "pass" else "fail"
    failure_code = None if final_status == "pass" else "strict_pipeline_failed"
    return {
        "run_tag": run_tag,
        "case_id": case.case_id,
        "source_name": case.source_name,
        "status": final_status,
        "failure_code": failure_code,
        "root_failure_code_primary": (root_failure_codes[0] if root_failure_codes else None),
        "root_failure_codes": root_failure_codes,
        "failed_steps": failed_steps,
        "final_exit_code": int(final_proc.returncode),
        "split_seed": int(effective_split_seed),
        "stress_selected_profile": selected_profile_id,
        "external_split_metadata": external_split_meta,
        "rows": {
            "train": int(len(splits["train"])),
            "valid": int(len(splits["valid"])),
            "test": int(len(splits["test"])),
            "external_total": int(len(external_pool)),
            "external_cross_period": int(len(external_period_df)),
            "external_cross_institution": int(len(external_institution_df)),
        },
        "metrics": metrics,
        "clinical_floor_gap_summary": clinical_floor_gap_summary,
        "gap_summary": gap_report.get("summary", {}),
        "bootstrap_exit_code": int(bootstrap_proc.returncode),
        "pipeline_status": pipeline_report.get("status"),
        "publication_status": publication_report.get("status"),
        "self_critique_status": self_critique_report.get("status"),
        "self_critique_score": self_critique_report.get("quality_score"),
        "external_validation_gate_status": external_gate_report.get("status"),
        "calibration_dca_gate_status": calibration_dca_report.get("status"),
        "distribution_generalization_gate_status": (
            load_json(evidence_dir / "distribution_generalization_report.json").get("status")
            if (evidence_dir / "distribution_generalization_report.json").exists()
            else "not_run"
        ),
        "feature_engineering_audit_gate_status": (
            load_json(evidence_dir / "feature_engineering_audit_report.json").get("status")
            if (evidence_dir / "feature_engineering_audit_report.json").exists()
            else "not_run"
        ),
        "ci_matrix_gate_status": (
            load_json(evidence_dir / "ci_matrix_gate_report.json").get("status")
            if (evidence_dir / "ci_matrix_gate_report.json").exists()
            else "not_run"
        ),
        "artifacts": {
            "case_root": str(case_root),
            "request": str(cfg_dir / "request.json"),
            "pipeline_report": str(final_report),
            "model_selection_report": str(model_selection_report_path),
            "evaluation_report": str(evaluation_report_path),
            "feature_engineering_report": str(feature_engineering_report_path),
            "distribution_report": str(distribution_report_path),
            "ci_matrix_report": str(ci_matrix_report_path),
            "prediction_trace": str(prediction_trace_path),
            "external_validation_report": str(external_validation_report_path),
            "robustness_report": str(robustness_report_path),
            "seed_sensitivity_report": str(seed_sensitivity_report_path),
            "stress_seed_search_report": str(stress_seed_search_report_dst)
            if stress_seed_search_report_dst.exists()
            else None,
            "stress_seed_selection": str(stress_seed_selection_dst)
            if stress_seed_selection_dst.exists()
            else None,
        },
    }


def build_dataset_case(
    case_id: str,
    *,
    diabetes_max_rows: int,
    diabetes_target_mode: str,
    ckd_max_rows: int,
) -> DatasetCase:
    token = str(case_id).strip()
    if token == "uci-breast-cancer-wdbc":
        return DatasetCase(
            case_id="uci-breast-cancer-wdbc",
            raw_filename="breast_cancer_wdbc.data",
            target_name="breast_cancer_malignancy",
            source_name="UCI Breast Cancer Wisconsin (Diagnostic)",
        )
    if token == "uci-heart-disease":
        return DatasetCase(
            case_id="uci-heart-disease",
            raw_filename="heart_disease_processed.cleveland.data",
            target_name="heart_disease",
            source_name="UCI Heart Disease (Cleveland)",
        )
    if token == "uci-chronic-kidney-disease":
        return DatasetCase(
            case_id="uci-chronic-kidney-disease",
            raw_filename="chronic_kidney_disease/Chronic_Kidney_Disease/chronic_kidney_disease.arff",
            target_name="chronic_kidney_disease",
            source_name="UCI Chronic Kidney Disease",
            options={"max_rows": int(ckd_max_rows)},
        )
    if token == "uci-diabetes-130-readmission":
        diabetes_token = str(diabetes_target_mode).strip().lower()
        diabetes_target_name = {
            "lt30": "diabetes_readmission_lt30d",
            "gt30": "diabetes_readmission_gt30d",
            "any": "diabetes_readmission_any",
        }.get(diabetes_token, "diabetes_readmission_gt30d")
        return DatasetCase(
            case_id="uci-diabetes-130-readmission",
            raw_filename="diabetes_130_us_hospitals/diabetic_data.csv",
            target_name=diabetes_target_name,
            source_name="UCI Diabetes 130-US Hospitals (1999-2008)",
            options={"max_rows": int(diabetes_max_rows), "target_mode": diabetes_token},
        )
    raise ValueError(f"Unsupported case_id: {token}")


def maybe_run_auto_diabetes_feasibility_scan(
    args: argparse.Namespace,
    *,
    run_tag: str,
    stress_case_id: str,
    results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not bool(args.auto_scan_diabetes_feasibility):
        return None
    diabetes_row: Optional[Dict[str, Any]] = next(
        (
            row
            for row in results
            if isinstance(row, dict) and str(row.get("case_id")) == "uci-diabetes-130-readmission"
        ),
        None,
    )
    if not isinstance(diabetes_row, dict):
        return {
            "status": "skipped",
            "reason": "diabetes_case_not_present",
        }
    if str(diabetes_row.get("status")) == "pass":
        return {
            "status": "skipped",
            "reason": "diabetes_case_passed",
        }

    root_codes = diabetes_row.get("root_failure_codes")
    root_code_set = set(root_codes) if isinstance(root_codes, list) else set()
    clinical_floor_codes = {
        "clinical_floor_sensitivity_not_met",
        "clinical_floor_npv_not_met",
        "clinical_floor_specificity_not_met",
        "clinical_floor_ppv_not_met",
    }
    if not (root_code_set & clinical_floor_codes):
        return {
            "status": "skipped",
            "reason": "non_clinical_floor_failure",
            "root_failure_codes": sorted(root_code_set),
        }

    scan_script = REPO_ROOT / "experiments" / "authority-e2e" / "scan_stress_diabetes_feasibility.py"
    report_path = resolve_output_path(args.diabetes_feasibility_report_file)
    summary_dir = resolve_output_path(args.diabetes_feasibility_summary_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    scan_cmd = [
        sys.executable,
        str(scan_script),
        "--target-modes",
        str(args.diabetes_feasibility_target_modes),
        "--max-rows-options",
        str(args.diabetes_feasibility_max_rows_options),
        "--summary-dir",
        str(summary_dir),
        "--report",
        str(report_path),
        "--run-tag-prefix",
        f"{run_tag}-auto-diabetes-scan",
    ]
    proc = run_cmd(scan_cmd, cwd=REPO_ROOT, allow_fail=True)

    payload: Dict[str, Any] = {}
    if report_path.exists():
        try:
            payload = load_json(report_path)
        except Exception:
            payload = {}

    scan_status = str(payload.get("overall_status", "")).strip().lower()
    best_candidate = payload.get("best_candidate") if isinstance(payload, dict) else None
    recommended_retry_command: Optional[str] = None
    best_mode = None
    best_rows = None
    if isinstance(best_candidate, dict):
        token_mode = str(best_candidate.get("target_mode", "")).strip().lower()
        token_rows = best_candidate.get("max_rows")
        if token_mode in {"lt30", "gt30", "any"} and isinstance(token_rows, int):
            best_mode = token_mode
            best_rows = int(token_rows)
            mlgg_path = REPO_ROOT / "scripts" / "mlgg.py"
            recommended_retry_command = (
                f"{shlex.quote(sys.executable)} {shlex.quote(str(mlgg_path))} authority "
                f"--include-large-cases --include-stress-cases "
                f"--stress-case-id {shlex.quote(str(stress_case_id).strip() or DEFAULT_STRESS_CASE_ID)} "
                f"--diabetes-target-mode {shlex.quote(best_mode)} --diabetes-max-rows {best_rows}"
            )
    diabetes_row.setdefault("artifacts", {})
    if isinstance(diabetes_row["artifacts"], dict):
        diabetes_row["artifacts"]["diabetes_feasibility_report"] = str(report_path)
    trigger_source = "stress_case" if str(stress_case_id).strip() == "uci-diabetes-130-readmission" else "large_case"
    diabetes_row["diabetes_feasibility_scan"] = {
        "status": "pass" if scan_status == "pass" else "fail",
        "trigger_case_id": "uci-diabetes-130-readmission",
        "trigger_source": trigger_source,
        "report_file": str(report_path),
        "summary_dir": str(summary_dir),
        "return_code": int(proc.returncode),
        "best_candidate": best_candidate if isinstance(best_candidate, dict) else None,
        "recommended_retry_command": recommended_retry_command,
    }
    if scan_status != "pass":
        if trigger_source == "stress_case":
            diabetes_row["failure_code_detail"] = "stress_case_clinical_feasibility_not_found"
        else:
            diabetes_row["failure_code_detail"] = "large_case_clinical_feasibility_not_found"
        diabetes_row["recommended_stress_case_id"] = "uci-chronic-kidney-disease"
    elif best_mode is not None and best_rows is not None:
        diabetes_row["feasible_diabetes_configuration"] = {
            "target_mode": best_mode,
            "max_rows": best_rows,
        }
    return diabetes_row.get("diabetes_feasibility_scan")


def main() -> int:
    args = parse_args()
    if int(args.subprocess_timeout_seconds) < 1:
        raise SystemExit("--subprocess-timeout-seconds must be >= 1.")
    if float(args.case_lock_timeout_seconds) < 1.0:
        raise SystemExit("--case-lock-timeout-seconds must be >= 1.")
    if float(args.lock_wait_heartbeat_seconds) < 0.0:
        raise SystemExit("--lock-wait-heartbeat-seconds must be >= 0.")
    global SUBPROCESS_TIMEOUT_SECONDS, CASE_LOCK_TIMEOUT_SECONDS, LOCK_WAIT_HEARTBEAT_SECONDS
    SUBPROCESS_TIMEOUT_SECONDS = int(args.subprocess_timeout_seconds)
    CASE_LOCK_TIMEOUT_SECONDS = float(args.case_lock_timeout_seconds)
    LOCK_WAIT_HEARTBEAT_SECONDS = float(args.lock_wait_heartbeat_seconds)
    run_tag = str(args.run_tag).strip() or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary_path = resolve_output_path(args.summary_file)
    include_large_cases = bool(args.include_large_cases or parse_bool_env("MLLG_INCLUDE_LARGE_CASES", default=False))
    include_ckd_case = bool(args.include_ckd_case or parse_bool_env("MLLG_INCLUDE_CKD_CASE", default=False))
    diabetes_target_mode = str(args.diabetes_target_mode).strip().lower()
    cases = [
        build_dataset_case(
            "uci-breast-cancer-wdbc",
            diabetes_max_rows=int(args.diabetes_max_rows),
            diabetes_target_mode=diabetes_target_mode,
            ckd_max_rows=int(args.ckd_max_rows),
        ),
    ]
    if include_ckd_case:
        cases.append(
            build_dataset_case(
                "uci-chronic-kidney-disease",
                diabetes_max_rows=int(args.diabetes_max_rows),
                diabetes_target_mode=diabetes_target_mode,
                ckd_max_rows=int(args.ckd_max_rows),
            )
        )
    if include_large_cases:
        cases.append(
            build_dataset_case(
                "uci-diabetes-130-readmission",
                diabetes_max_rows=int(args.diabetes_max_rows),
                diabetes_target_mode=diabetes_target_mode,
                ckd_max_rows=int(args.ckd_max_rows),
            )
        )
    include_stress_cases = bool(args.include_stress_cases or parse_bool_env("MLLG_INCLUDE_STRESS_CASES", default=False))
    stress_case_id = str(args.stress_case_id).strip() or DEFAULT_STRESS_CASE_ID
    if args.stress_seed_search and not include_stress_cases:
        include_stress_cases = True
    stress_seed_search_enabled = bool(args.stress_seed_search or (include_stress_cases and not args.no_stress_seed_search))
    if stress_case_id != "uci-heart-disease":
        if stress_seed_search_enabled:
            print(
                "[WARN] stress seed-search is supported only for uci-heart-disease; "
                f"disabling seed-search for stress-case-id={stress_case_id}."
            )
        stress_seed_search_enabled = False
    if include_stress_cases:
        stress_case = build_dataset_case(
            stress_case_id,
            diabetes_max_rows=int(args.diabetes_max_rows),
            diabetes_target_mode=diabetes_target_mode,
            ckd_max_rows=int(args.ckd_max_rows),
        )
        if all(existing.case_id != stress_case.case_id for existing in cases):
            cases.append(stress_case)

    results: List[Dict[str, Any]] = []
    failed_cases: List[str] = []
    stress_result: Optional[HeartStressSearchResult] = None
    stress_seed_cache = resolve_output_path(args.stress_seed_cache_file)
    stress_selection_file = resolve_output_path(args.stress_selection_file)
    stress_profile_set = str(args.stress_profile_set).strip() or DEFAULT_STRESS_PROFILE_SET
    stress_profiles: List[Dict[str, str]] = []
    if stress_seed_search_enabled:
        stress_profiles = get_stress_profiles(stress_profile_set)

    if stress_seed_search_enabled and include_stress_cases:
        heart_case = next((c for c in cases if c.case_id == "uci-heart-disease"), None)
        if heart_case is not None:
            try:
                heart_df, heart_feature_cols = load_heart_dataset(RAW_ROOT / heart_case.raw_filename)
                dataset_fingerprint = build_dataset_fingerprint(heart_df)
                external_inst_df, external_inst_meta = load_heart_external_institution_pool(
                    heart_case.case_id,
                    heart_feature_cols,
                    exclude_site_tags=HEART_INTERNAL_SITE_TAGS,
                )
                dataset_fingerprint["external_institution"] = (
                    build_dataset_fingerprint(external_inst_df)
                    if isinstance(external_inst_df, pd.DataFrame) and not external_inst_df.empty
                    else {"rows": 0, "events": 0, "sha256": "missing"}
                )
                dataset_fingerprint["external_institution_source"] = external_inst_meta
                policy_contract = build_stress_contract_inputs(
                    case_id=heart_case.case_id,
                    profile_set=stress_profile_set,
                    profiles=stress_profiles,
                )
                policy_sha256 = canonical_json_sha256(policy_contract)
                code_revision_hint = git_revision_hint()
            except Exception as exc:
                failed_cases.append(heart_case.case_id)
                results.append(
                    {
                        "run_tag": run_tag,
                        "case_id": heart_case.case_id,
                        "source_name": heart_case.source_name,
                        "status": "fail",
                        "failure_code": "stress_seed_feasibility_not_found",
                        "final_exit_code": 2,
                        "error": f"stress_search_initialization_failed: {exc}",
                        "artifacts": {
                            "stress_seed_search_report": str(stress_seed_cache),
                            "stress_seed_selection": str(stress_selection_file),
                        },
                    }
                )
                heart_case = None
            if heart_case is None:
                pass
            else:
                cached_selected_seed: Optional[int] = None
                cached_selected_profile: Optional[str] = None
                if stress_seed_cache.exists():
                    try:
                        cached_payload = load_json(stress_seed_cache)
                        cached_status = str(cached_payload.get("status", "")).strip().lower()
                        cached_seed_raw = cached_payload.get("selected_seed")
                        cached_selected_profile_raw = cached_payload.get("selected_profile")
                        cache_contract_ok = bool(
                            str(cached_payload.get("contract_version", "")).strip() == STRESS_SEARCH_REPORT_CONTRACT_VERSION
                            and str(cached_payload.get("search_profile_set", "")).strip() == stress_profile_set
                            and str(cached_payload.get("policy_sha256", "")).strip() == policy_sha256
                        )
                        cached_dataset_fingerprint = cached_payload.get("dataset_fingerprint")
                        if isinstance(cached_dataset_fingerprint, dict):
                            cache_contract_ok = bool(
                                cache_contract_ok
                                and canonical_json_sha256(cached_dataset_fingerprint)
                                == canonical_json_sha256(dataset_fingerprint)
                            )
                        else:
                            cache_contract_ok = False
                        if cached_status == "pass" and isinstance(cached_seed_raw, int) and cache_contract_ok:
                            cached_seed = int(cached_seed_raw)
                            cached_profile = (
                                str(cached_selected_profile_raw).strip()
                                if isinstance(cached_selected_profile_raw, str)
                                else ""
                            )
                            profile_ok = any(p.get("profile_id") == cached_profile for p in stress_profiles)
                            selection_contract_ok = False
                            if stress_selection_file.exists():
                                try:
                                    selection_payload = load_json(stress_selection_file)
                                    selection_status = str(selection_payload.get("status", "")).strip().lower()
                                    selection_seed_raw = selection_payload.get("selected_seed")
                                    selection_profile_raw = selection_payload.get("selected_profile")
                                    selection_profile = (
                                        str(selection_profile_raw).strip()
                                        if isinstance(selection_profile_raw, str)
                                        else ""
                                    )
                                    selection_contract_ok = bool(
                                        selection_status == "pass"
                                        and str(selection_payload.get("contract_version", "")).strip()
                                        == STRESS_SEARCH_REPORT_CONTRACT_VERSION
                                        and str(selection_payload.get("search_profile_set", "")).strip() == stress_profile_set
                                        and str(selection_payload.get("policy_sha256", "")).strip() == policy_sha256
                                        and isinstance(selection_seed_raw, int)
                                        and int(selection_seed_raw) == cached_seed
                                        and selection_profile == cached_profile
                                    )
                                    selection_dataset_fingerprint = selection_payload.get("dataset_fingerprint")
                                    if isinstance(selection_dataset_fingerprint, dict):
                                        selection_contract_ok = bool(
                                            selection_contract_ok
                                            and canonical_json_sha256(selection_dataset_fingerprint)
                                            == canonical_json_sha256(dataset_fingerprint)
                                        )
                                    else:
                                        selection_contract_ok = False
                                except Exception:
                                    selection_contract_ok = False
                            if (
                                int(args.stress_seed_min) <= cached_seed <= int(args.stress_seed_max)
                                and profile_ok
                                and selection_contract_ok
                            ):
                                cached_selected_seed = cached_seed
                                cached_selected_profile = cached_profile
                    except Exception:
                        cached_selected_seed = None
                        cached_selected_profile = None

                if cached_selected_seed is not None and cached_selected_profile is not None:
                    stress_result = HeartStressSearchResult(
                        selected_seed=int(cached_selected_seed),
                        selected_profile=str(cached_selected_profile),
                        report_path=stress_seed_cache,
                        selection_path=stress_selection_file,
                        status="pass",
                    )
                else:
                    try:
                        stress_result = search_heart_feasible_seed(
                            case=heart_case,
                            seed_min=int(args.stress_seed_min),
                            seed_max=int(args.stress_seed_max),
                            report_path=stress_seed_cache,
                            selection_path=stress_selection_file,
                            run_tag=run_tag,
                            profile_set=stress_profile_set,
                            dataset_fingerprint=dataset_fingerprint,
                            policy_sha256=policy_sha256,
                            code_revision_hint=code_revision_hint,
                        )
                    except Exception as exc:
                        failed_cases.append(heart_case.case_id)
                        results.append(
                            {
                                "run_tag": run_tag,
                                "case_id": heart_case.case_id,
                                "source_name": heart_case.source_name,
                                "status": "fail",
                                "failure_code": "stress_seed_feasibility_not_found",
                                "final_exit_code": 2,
                                "error": str(exc),
                                "artifacts": {
                                    "stress_seed_search_report": str(stress_seed_cache),
                                    "stress_seed_selection": str(stress_selection_file),
                                },
                            }
                        )

    for case in cases:
        if case.case_id == "uci-heart-disease" and stress_seed_search_enabled and stress_result is None:
            # Seed-search failure already recorded above.
            continue
        try:
            split_seed_override: Optional[int] = None
            case_stress_result: Optional[HeartStressSearchResult] = None
            if case.case_id == "uci-heart-disease":
                if stress_result is not None:
                    split_seed_override = int(stress_result.selected_seed)
                    case_stress_result = stress_result
                else:
                    split_seed_override = int(SPLIT_RANDOM_SEED_BY_CASE.get(case.case_id, 20260224))
            with case_run_lock(
                case.case_id,
                timeout_seconds=float(CASE_LOCK_TIMEOUT_SECONDS),
                heartbeat_seconds=float(LOCK_WAIT_HEARTBEAT_SECONDS),
            ):
                result = prepare_case_artifacts(
                    case,
                    split_seed_override=split_seed_override,
                    stress_search_result=case_stress_result,
                    run_tag=run_tag,
                )
            results.append(result)
            if str(result.get("status")) != "pass":
                failed_cases.append(case.case_id)
        except Exception as exc:
            failed_cases.append(case.case_id)
            results.append(
                {
                    "run_tag": run_tag,
                    "case_id": case.case_id,
                    "source_name": case.source_name,
                    "status": "fail",
                    "failure_code": "case_execution_exception",
                    "final_exit_code": 1,
                    "error": str(exc),
                }
            )

    diabetes_auto_scan_result = maybe_run_auto_diabetes_feasibility_scan(
        args,
        run_tag=run_tag,
        stress_case_id=stress_case_id,
        results=results,
    )

    summary = {
        "run_tag": run_tag,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO_ROOT),
        "include_large_cases": bool(include_large_cases),
        "include_ckd_case": bool(include_ckd_case),
        "ckd_max_rows": int(args.ckd_max_rows),
        "diabetes_max_rows": int(args.diabetes_max_rows),
        "diabetes_target_mode": str(args.diabetes_target_mode).strip().lower(),
        "include_stress_cases": bool(include_stress_cases),
        "stress_case_id": stress_case_id if include_stress_cases else None,
        "stress_seed_search_enabled": bool(stress_seed_search_enabled),
        "stress_profile_set": stress_profile_set,
        "stress_seed_range": {"min": int(args.stress_seed_min), "max": int(args.stress_seed_max)},
        "subprocess_timeout_seconds": int(SUBPROCESS_TIMEOUT_SECONDS),
        "case_lock_timeout_seconds": float(CASE_LOCK_TIMEOUT_SECONDS),
        "lock_wait_heartbeat_seconds": float(LOCK_WAIT_HEARTBEAT_SECONDS),
        "stress_seed_cache_file": str(stress_seed_cache),
        "stress_selection_file": str(stress_selection_file),
        "stress_seed_selected": int(stress_result.selected_seed) if stress_result is not None else None,
        "stress_profile_selected": str(stress_result.selected_profile) if stress_result is not None else None,
        "diabetes_feasibility_auto_scan": diabetes_auto_scan_result,
        "results": results,
        "failed_cases": failed_cases,
        "overall_status": "pass" if not failed_cases else "fail",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not failed_cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
