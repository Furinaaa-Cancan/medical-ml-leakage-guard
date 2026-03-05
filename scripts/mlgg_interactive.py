#!/usr/bin/env python3
"""
Interactive terminal wizard for ml-leakage-guard core commands.

Supported commands:
- init
- workflow
- train
- authority
"""

from __future__ import annotations

import argparse
import csv as _csv_mod
import importlib.util
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _gate_utils import load_json_from_path as load_json, write_json as atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
PROFILE_CONTRACT_VERSION = "v1"
SUPPORTED_COMMANDS = ("init", "workflow", "train", "authority")
PROMPT_AUTO_ACCEPT_DEFAULTS = False

COMMAND_SCRIPT: Dict[str, Path] = {
    "init": SCRIPTS_ROOT / "init_project.py",
    "workflow": SCRIPTS_ROOT / "run_productized_workflow.py",
    "train": SCRIPTS_ROOT / "train_select_evaluate.py",
    "authority": EXPERIMENTS_ROOT / "run_authority_e2e.py",
}

TRAIN_CALIBRATION_CHOICES = ("none", "power", "sigmoid", "isotonic", "beta")
AUTHORITY_STRESS_CASE_CHOICES = (
    "uci-chronic-kidney-disease",
    "uci-heart-disease",
    "uci-diabetes-130-readmission",
    "uci-breast-cancer-wdbc",
)
TRAIN_MODEL_POOL_DEFAULT = (
    "logistic_l1,logistic_l2,logistic_elasticnet,"
    "random_forest_balanced,extra_trees_balanced,hist_gradient_boosting_l2,"
    "svm_linear,svm_rbf,soft_voting,weighted_voting,stacking"
)
TRAIN_OPTIONAL_BACKEND_HINTS = {
    "xgboost": "pip install xgboost",
    "catboost": "pip install catboost",
    "lightgbm": "pip install lightgbm",
    "tabpfn": "pip install tabpfn",
}
SPLIT_STRATEGY_CHOICES = ("grouped_temporal", "grouped_random", "stratified_grouped")
DATA_INPUT_MODES = ("pre_split", "single_csv")


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Interactive wizard for ml-leakage-guard core commands."
    )
    parser.add_argument(
        "--command",
        required=True,
        choices=list(SUPPORTED_COMMANDS),
        help="Target command to build and execute interactively.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for the target command.",
    )
    parser.add_argument(
        "--cwd",
        default=str(REPO_ROOT),
        help="Working directory used for command execution.",
    )
    parser.add_argument(
        "--profile-name",
        default="",
        help="Profile name for --load-profile / --save-profile.",
    )
    parser.add_argument(
        "--profile-dir",
        default="~/.mlgg/profiles",
        help="Profile directory (default: ~/.mlgg/profiles).",
    )
    parser.add_argument(
        "--save-profile",
        action="store_true",
        help="Save selected arguments into profile file.",
    )
    parser.add_argument(
        "--load-profile",
        action="store_true",
        help="Load default argument values from profile file before prompting.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print generated command but do not execute.",
    )
    parser.add_argument(
        "--accept-defaults",
        action="store_true",
        help="Auto-accept prompt defaults (non-interactive when defaults are available).",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def fail(message: str) -> int:
    print(f"[FAIL] {message}", file=sys.stderr)
    return 2


def normalize_path(raw: str) -> str:
    return str(Path(raw).expanduser().resolve(strict=False))


def infer_project_base_from_request(request_path: str) -> Path:
    request = Path(request_path).expanduser().resolve(strict=False)
    parent = request.parent
    if parent.name.lower() == "configs" and parent.parent != parent:
        return parent.parent
    return parent


def read_csv_columns(csv_path: str) -> List[str]:
    """Read column headers from a CSV file without loading the full dataset."""
    p = Path(csv_path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        reader = _csv_mod.reader(fh)
        headers = next(reader, None)
    return list(headers) if headers else []


def validate_binary_target(csv_path: str, col: str) -> Tuple[bool, str]:
    """Check that a column exists and contains only 0/1 values. Returns (ok, message)."""
    p = Path(csv_path).expanduser().resolve()
    try:
        with p.open("r", encoding="utf-8") as fh:
            reader = _csv_mod.DictReader(fh)
            if col not in (reader.fieldnames or []):
                return False, f"column '{col}' not found in {p.name}"
            unique_values: set = set()
            for row_idx, row in enumerate(reader):
                raw = str(row.get(col, "")).strip()
                unique_values.add(raw)
                if row_idx > 500:
                    break
        allowed = {"0", "1", "0.0", "1.0"}
        non_binary = unique_values - allowed - {""}
        if non_binary:
            samples = sorted(non_binary)[:5]
            return False, f"column '{col}' has non-binary values: {samples}"
        if not unique_values & {"0", "1", "0.0", "1.0"}:
            return False, f"column '{col}' appears empty or has no 0/1 values"
        return True, "binary target validated"
    except Exception as exc:
        return False, f"validation error: {exc}"


def prompt_column_choice(
    label: str,
    columns: List[str],
    default: str,
    required: bool = True,
    validate_fn: Optional[Any] = None,
) -> str:
    """Prompt user to select a column from available CSV columns."""
    if PROMPT_AUTO_ACCEPT_DEFAULTS:
        if default and default in columns:
            if validate_fn:
                ok, msg = validate_fn(default)
                if not ok:
                    print(f"  [WARN] Auto-accepted column '{default}': {msg}", file=sys.stderr)
            return default
        if default and columns:
            print(
                f"  [WARN] Default column '{default}' not found in CSV. "
                f"Available: {', '.join(columns[:10])}",
                file=sys.stderr,
            )
            if required:
                raise ValueError(
                    f"Column '{default}' not found in CSV and --accept-defaults is enabled. "
                    f"Override with --target-col or --patient-id-col."
                )
        return default

    print(f"\n{label}")
    print(f"  Available columns ({len(columns)}): {', '.join(columns[:30])}")
    if len(columns) > 30:
        print(f"  ... and {len(columns) - 30} more")

    while True:
        suffix = f" [default: {default}]" if default else ""
        try:
            raw = input(f"  Select column{suffix}: ").strip()
        except EOFError as exc:
            raise ValueError(
                "interactive stdin is not available; use --accept-defaults or --print-only."
            ) from exc
        value = raw if raw else default
        if required and not value:
            print("  [WARN] This field is required.")
            continue
        if value and value not in columns:
            print(f"  [WARN] Column '{value}' not found in CSV. Available: {', '.join(columns[:15])}")
            confirm = input("  Use anyway? (yes/no) [no]: ").strip().lower()
            if confirm not in ("yes", "y"):
                continue
        if validate_fn and value:
            ok, msg = validate_fn(value)
            if not ok:
                print(f"  [WARN] {msg}")
                confirm = input("  Use anyway? (yes/no) [no]: ").strip().lower()
                if confirm not in ("yes", "y"):
                    continue
        return str(value)


def compute_feature_summary(
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    max_rows: int = 2000,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-feature statistics for CLI feature display.

    Returns dict mapping column name to:
        is_numeric (bool), missing_pct (float), variance (float|None),
        corr_target (float|None), distinct (int).
    """
    import math as _math

    p = Path(csv_path).expanduser().resolve()
    n_rows = 0
    col_vals: Dict[str, List[Optional[float]]] = {c: [] for c in feature_cols}
    col_missing: Dict[str, int] = {c: 0 for c in feature_cols}
    col_numeric_ok: Dict[str, int] = {c: 0 for c in feature_cols}
    col_distinct: Dict[str, set] = {c: set() for c in feature_cols}
    target_vals: List[Optional[float]] = []

    try:
        with p.open("r", encoding="utf-8") as fh:
            reader = _csv_mod.DictReader(fh)
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                if not isinstance(row, dict):
                    continue
                n_rows += 1
                raw_t = str(row.get(target_col, "") or "").strip()
                t_val: Optional[float] = None
                if raw_t in ("0", "0.0"):
                    t_val = 0.0
                elif raw_t in ("1", "1.0"):
                    t_val = 1.0
                target_vals.append(t_val)
                for col in feature_cols:
                    raw = str(row.get(col, "") or "").strip()
                    if not raw:
                        col_missing[col] += 1
                        col_vals[col].append(None)
                        continue
                    if len(col_distinct[col]) < max_rows:
                        col_distinct[col].add(raw)
                    try:
                        fval = float(raw)
                        if _math.isfinite(fval):
                            col_vals[col].append(fval)
                            col_numeric_ok[col] += 1
                        else:
                            col_vals[col].append(None)
                    except (ValueError, TypeError):
                        col_vals[col].append(None)
    except Exception:
        return {}

    if n_rows == 0:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        non_empty = n_rows - col_missing[col]
        missing_pct = round(100.0 * float(col_missing[col]) / float(max(n_rows, 1)), 1)
        is_numeric = non_empty > 0 and col_numeric_ok[col] >= non_empty * 0.9
        distinct = len(col_distinct[col])
        variance: Optional[float] = None
        corr_target: Optional[float] = None

        numeric_values = [v for v in col_vals[col] if v is not None]
        if is_numeric and len(numeric_values) >= 2:
            mean = sum(numeric_values) / len(numeric_values)
            variance = sum((v - mean) ** 2 for v in numeric_values) / (len(numeric_values) - 1)
            paired = [
                (col_vals[col][i], target_vals[i])
                for i in range(min(len(col_vals[col]), len(target_vals)))
                if col_vals[col][i] is not None and target_vals[i] is not None
            ]
            if len(paired) >= 5:
                xs = [pp[0] for pp in paired]
                ys = [pp[1] for pp in paired]
                n_p = len(paired)
                mx = sum(xs) / n_p
                my = sum(ys) / n_p
                cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                var_x = sum((x - mx) ** 2 for x in xs)
                var_y = sum((y - my) ** 2 for y in ys)
                denom = (var_x * var_y) ** 0.5
                if denom > 1e-15:
                    corr_target = round(cov_xy / denom, 3)

        result[col] = {
            "is_numeric": is_numeric,
            "missing_pct": missing_pct,
            "variance": variance,
            "corr_target": corr_target,
            "distinct": distinct,
        }
    return result


def _format_feature_line(col: str, info: Dict[str, Any]) -> str:
    """Format a single feature column's statistics as a compact string."""
    parts = [f"{'num' if info.get('is_numeric') else 'cat':>3s}"]
    miss = info.get("missing_pct", 0.0)
    if miss > 0.05:
        parts.append(f"miss={miss}%")
    if info.get("is_numeric") and info.get("variance") is not None:
        v = info["variance"]
        if v < 1e-12:
            parts.append("CONST")
        elif v < 1e-6:
            parts.append("LOW-VAR")
        else:
            parts.append(f"var={v:.4g}")
    if info.get("corr_target") is not None:
        parts.append(f"corr={info['corr_target']:+.3f}")
    return f"    {col:<30s}  {' | '.join(parts)}"


def prompt_ignore_cols(
    columns: List[str],
    target_col: str,
    patient_id_col: str,
    default: str,
    csv_path: Optional[str] = None,
) -> str:
    """Prompt user to select columns to ignore (non-feature columns)."""
    if PROMPT_AUTO_ACCEPT_DEFAULTS:
        if columns:
            default_parts = [s.strip() for s in default.split(",") if s.strip()]
            existing = [c for c in default_parts if c in columns]
            dropped = [c for c in default_parts if c not in columns and c != target_col]
            if dropped:
                print(
                    f"  [WARN] Ignore-cols default adjusted: {dropped} not in CSV, removed.",
                    file=sys.stderr,
                )
            return ",".join(existing)
        return default

    default_set = set(s.strip() for s in default.split(",") if s.strip())
    feature_cols = [c for c in columns if c not in default_set and c != target_col]

    print(f"\n  Columns to IGNORE (non-feature, comma-separated)")
    print(f"  Current ignore list: {default}")
    print(f"  Resulting feature columns ({len(feature_cols)}):")

    feat_summary: Dict[str, Dict[str, Any]] = {}
    if csv_path and feature_cols:
        feat_summary = compute_feature_summary(csv_path, feature_cols, target_col)
    if feat_summary:
        print(f"  {'Column':<30s}  {'Type':>4s} | {'Miss%':>6s} | {'Variance':>10s} | {'Corr(y)':>8s}")
        print(f"  {'-' * 70}")
        for col in feature_cols:
            info = feat_summary.get(col)
            if info:
                print(_format_feature_line(col, info))
            else:
                print(f"    {col}")
    else:
        print(f"  {', '.join(feature_cols[:20])}")
        if len(feature_cols) > 20:
            print(f"  ... and {len(feature_cols) - 20} more")

    while True:
        try:
            raw = input(f"  Ignore columns [default: {default}]: ").strip()
        except EOFError as exc:
            raise ValueError(
                "interactive stdin is not available; use --accept-defaults or --print-only."
            ) from exc
        value = raw if raw else default
        ignore_set = set(s.strip() for s in value.split(",") if s.strip())
        missing = ignore_set - set(columns) - {target_col}
        if missing:
            print(f"  [WARN] These ignore columns not found in CSV: {sorted(missing)}")
        result_features = [c for c in columns if c not in ignore_set and c != target_col]
        print(f"  → {len(result_features)} feature columns will be used")
        if not result_features:
            print("  [WARN] No feature columns would remain!")
            continue
        return value


def infer_project_base_from_split_path(split_path: str) -> Path:
    split_file = Path(split_path).expanduser().resolve(strict=False)
    parent = split_file.parent
    if parent.name.lower() == "data" and parent.parent != parent:
        return parent.parent
    return parent


def _resolve_existing_csv(path_value: Any, base_dir: Path) -> Optional[str]:
    token = str(path_value).strip()
    if not token:
        return None
    raw_path = Path(token).expanduser()
    candidate = (
        raw_path.resolve(strict=False)
        if raw_path.is_absolute()
        else (base_dir / raw_path).resolve(strict=False)
    )
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return None


def infer_pre_split_paths_from_context() -> Optional[Dict[str, str]]:
    """Infer default train/valid/test CSVs for --accept-defaults pre-split mode."""
    cwd = Path.cwd().resolve(strict=False)

    # Prefer explicit split_paths from request.json when available.
    for request_path in (cwd / "configs" / "request.json", cwd / "request.json"):
        if not request_path.exists() or not request_path.is_file():
            continue
        try:
            payload = load_json(request_path)
        except Exception:
            continue
        split_paths = payload.get("split_paths")
        if not isinstance(split_paths, dict):
            continue
        resolved: Dict[str, str] = {}
        for key in ("train", "valid", "test"):
            path_token = split_paths.get(key)
            resolved_path = _resolve_existing_csv(path_token, request_path.parent)
            if not resolved_path:
                resolved = {}
                break
            resolved[key] = resolved_path
        if len(resolved) == 3:
            return resolved

    # Fallback: discover canonical data/*.csv under working directory.
    for base in (cwd / "data", cwd):
        train_path = (base / "train.csv").resolve(strict=False)
        valid_path = (base / "valid.csv").resolve(strict=False)
        test_path = (base / "test.csv").resolve(strict=False)
        if train_path.exists() and valid_path.exists() and test_path.exists():
            return {
                "train": str(train_path),
                "valid": str(valid_path),
                "test": str(test_path),
            }
    return None


def validate_profile_name(raw: str) -> str:
    name = str(raw).strip()
    if not name:
        raise ValueError("profile name is empty.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", name):
        raise ValueError(
            "profile name must match [A-Za-z0-9._-]{1,128}."
        )
    return name


def profile_file_path(profile_dir: str, profile_name: str) -> Path:
    return Path(profile_dir).expanduser().resolve(strict=False) / f"{profile_name}.json"


def prompt_text(label: str, default: str = "", required: bool = False) -> str:
    if PROMPT_AUTO_ACCEPT_DEFAULTS:
        value = str(default)
        if required and not value:
            raise ValueError(
                f"{label} requires a value when --accept-defaults is enabled."
            )
        return value
    while True:
        suffix = f" [default: {default}]" if default else ""
        try:
            raw = input(f"{label}{suffix}: ").strip()
        except EOFError as exc:
            raise ValueError(
                "interactive stdin is not available; use --accept-defaults or --print-only."
            ) from exc
        value = raw if raw else default
        if required and not value:
            print("[WARN] This field is required.")
            continue
        return str(value)


def prompt_path(
    label: str,
    default: str = "",
    required: bool = False,
    must_exist: bool = False,
    allow_empty: bool = False,
) -> str:
    while True:
        raw = prompt_text(label=label, default=default, required=required)
        value = raw.strip()
        if not value:
            if allow_empty:
                return ""
            if required:
                print("[WARN] This field is required.")
                continue
            return ""
        normalized = normalize_path(value)
        if must_exist and not Path(normalized).exists():
            print(
                f"[WARN] Path not found: {normalized}. "
                "Please provide an existing path."
            )
            continue
        return normalized


def prompt_int(
    label: str,
    default: int,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    while True:
        raw = prompt_text(label=label, default=str(default), required=True)
        try:
            value = int(raw)
        except Exception:
            print("[WARN] Please enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"[WARN] Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"[WARN] Value must be <= {max_value}.")
            continue
        return value


def prompt_choice(label: str, choices: Tuple[str, ...], default: str) -> str:
    if default not in choices:
        default = choices[0]
    if PROMPT_AUTO_ACCEPT_DEFAULTS:
        return default
    default_index = list(choices).index(default) + 1
    print(label)
    for idx, value in enumerate(choices, start=1):
        print(f"  {idx}. {value}")
    while True:
        try:
            raw = input(f"Select [1-{len(choices)}] (default {default_index}): ").strip()
        except EOFError as exc:
            raise ValueError(
                "interactive stdin is not available; use --accept-defaults or --print-only."
            ) from exc
        if not raw:
            return default
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(choices):
                return choices[picked - 1]
        if raw in choices:
            return raw
        print("[WARN] Invalid selection.")


def prompt_bool(label: str, default: bool = False) -> bool:
    default_token = "yes" if default else "no"
    picked = prompt_choice(label=label, choices=("yes", "no"), default=default_token)
    return picked == "yes"


def prompt_authority_stress_case(default: str) -> str:
    default_token = default if default in AUTHORITY_STRESS_CASE_CHOICES else AUTHORITY_STRESS_CASE_CHOICES[0]
    if PROMPT_AUTO_ACCEPT_DEFAULTS:
        return default_token
    print("Stress case ID")
    labels: Dict[str, str] = {
        "uci-chronic-kidney-disease": "recommended release-grade path",
        "uci-heart-disease": "advanced research/high-pressure path (may fail by design)",
        "uci-diabetes-130-readmission": "advanced large-cohort stress path",
        "uci-breast-cancer-wdbc": "advanced small-cohort stress path",
    }
    for idx, value in enumerate(AUTHORITY_STRESS_CASE_CHOICES, start=1):
        desc = labels.get(value, "")
        suffix = f" [{desc}]" if desc else ""
        print(f"  {idx}. {value}{suffix}")
    default_index = list(AUTHORITY_STRESS_CASE_CHOICES).index(default_token) + 1
    while True:
        raw = input(f"Select [1-{len(AUTHORITY_STRESS_CASE_CHOICES)}] (default {default_index}): ").strip()
        if not raw:
            return default_token
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(AUTHORITY_STRESS_CASE_CHOICES):
                return AUTHORITY_STRESS_CASE_CHOICES[picked - 1]
        if raw in AUTHORITY_STRESS_CASE_CHOICES:
            return raw
        print("[WARN] Invalid selection.")


def parse_command_overrides(command: str, passthrough: List[str]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    if command == "init":
        parser.add_argument("--project-root")
        parser.add_argument("--study-id")
        parser.add_argument("--target-name")
        parser.add_argument("--label-col")
        parser.add_argument("--patient-id-col")
        parser.add_argument("--index-time-col")
        parser.add_argument("--force", action="store_true", default=None)
    elif command == "workflow":
        parser.add_argument("--request")
        parser.add_argument("--evidence-dir")
        parser.add_argument("--compare-manifest")
        parser.add_argument("--allow-missing-compare", action="store_true", default=None)
        parser.add_argument("--continue-on-fail", action="store_true", default=None)
    elif command == "train":
        parser.add_argument("--input-csv")
        parser.add_argument("--split-strategy", choices=list(SPLIT_STRATEGY_CHOICES))
        parser.add_argument("--split-output-dir")
        parser.add_argument("--time-col")
        parser.add_argument("--train-ratio", type=float)
        parser.add_argument("--valid-ratio", type=float)
        parser.add_argument("--test-ratio", type=float)
        parser.add_argument("--train")
        parser.add_argument("--valid")
        parser.add_argument("--test")
        parser.add_argument("--target-col")
        parser.add_argument("--patient-id-col")
        parser.add_argument("--ignore-cols")
        parser.add_argument("--model-pool")
        parser.add_argument("--include-optional-models", action="store_true", default=None)
        parser.add_argument("--ensemble-top-k", type=int)
        parser.add_argument("--n-jobs", type=int)
        parser.add_argument("--calibration-method", choices=list(TRAIN_CALIBRATION_CHOICES))
        parser.add_argument("--feature-group-spec")
        parser.add_argument("--external-cohort-spec")
        parser.add_argument("--model-selection-report-out")
        parser.add_argument("--evaluation-report-out")
        parser.add_argument("--prediction-trace-out")
        parser.add_argument("--external-validation-report-out")
        parser.add_argument("--ci-matrix-report-out")
        parser.add_argument("--distribution-report-out")
        parser.add_argument("--feature-engineering-report-out")
        parser.add_argument("--robustness-report-out")
        parser.add_argument("--seed-sensitivity-out")
    elif command == "authority":
        parser.add_argument("--include-stress-cases", action="store_true", default=None)
        parser.add_argument("--stress-case-id", choices=list(AUTHORITY_STRESS_CASE_CHOICES))
        parser.add_argument("--stress-seed-search", action="store_true", default=None)
        parser.add_argument("--summary-file")
        parser.add_argument("--run-tag")
        parser.add_argument("--stress-profile-set")
    else:
        raise ValueError(f"Unsupported command: {command}")

    ns, unknown = parser.parse_known_args(passthrough)
    if unknown:
        raise ValueError(f"Unknown override arguments: {unknown}")
    return {k: v for k, v in vars(ns).items() if v is not None}


def profile_allowed_keys(command: str) -> Tuple[str, ...]:
    table: Dict[str, Tuple[str, ...]] = {
        "init": (
            "project_root",
            "study_id",
            "target_name",
            "label_col",
            "patient_id_col",
            "index_time_col",
            "force",
        ),
        "workflow": (
            "request",
            "evidence_dir",
            "compare_manifest",
            "allow_missing_compare",
            "continue_on_fail",
        ),
        "train": (
            "__data_mode__",
            "input_csv",
            "split_strategy",
            "split_output_dir",
            "time_col",
            "train_ratio",
            "valid_ratio",
            "test_ratio",
            "train",
            "valid",
            "test",
            "target_col",
            "patient_id_col",
            "ignore_cols",
            "model_pool",
            "include_optional_models",
            "ensemble_top_k",
            "n_jobs",
            "calibration_method",
            "feature_group_spec",
            "external_cohort_spec",
            "model_selection_report_out",
            "evaluation_report_out",
            "prediction_trace_out",
            "external_validation_report_out",
            "ci_matrix_report_out",
            "distribution_report_out",
            "feature_engineering_report_out",
            "robustness_report_out",
            "seed_sensitivity_out",
        ),
        "authority": (
            "include_stress_cases",
            "stress_case_id",
            "stress_seed_search",
            "summary_file",
            "run_tag",
            "stress_profile_set",
        ),
    }
    return table[command]


def validate_profile_values(command: str, values: Dict[str, Any]) -> None:
    def _check_type(key: str, expected: tuple[type, ...]) -> None:
        if key not in values:
            return
        value = values[key]
        if not isinstance(value, expected):
            exp = "/".join([t.__name__ for t in expected])
            raise ValueError(
                f"profile key '{key}' has invalid type: {type(value).__name__}; expected {exp}."
            )

    if command == "init":
        for key in ("project_root", "study_id", "target_name", "label_col", "patient_id_col", "index_time_col"):
            _check_type(key, (str,))
        _check_type("force", (bool,))
        return

    if command == "workflow":
        for key in ("request", "evidence_dir", "compare_manifest"):
            _check_type(key, (str,))
        _check_type("allow_missing_compare", (bool,))
        _check_type("continue_on_fail", (bool,))
        return

    if command == "train":
        for key in (
            "input_csv",
            "split_strategy",
            "split_output_dir",
            "time_col",
            "train",
            "valid",
            "test",
            "target_col",
            "patient_id_col",
            "model_pool",
            "calibration_method",
            "feature_group_spec",
            "external_cohort_spec",
            "model_selection_report_out",
            "evaluation_report_out",
            "prediction_trace_out",
            "external_validation_report_out",
            "ci_matrix_report_out",
            "distribution_report_out",
            "feature_engineering_report_out",
            "robustness_report_out",
            "seed_sensitivity_out",
        ):
            _check_type(key, (str,))
        _check_type("include_optional_models", (bool,))
        _check_type("n_jobs", (int,))
        if "calibration_method" in values:
            token = str(values["calibration_method"]).strip()
            if token not in TRAIN_CALIBRATION_CHOICES:
                raise ValueError(
                    "profile key 'calibration_method' has invalid value: "
                    f"{values['calibration_method']}. "
                    f"Expected one of: {', '.join(TRAIN_CALIBRATION_CHOICES)}."
                )
        return

    if command == "authority":
        _check_type("include_stress_cases", (bool,))
        _check_type("stress_seed_search", (bool,))
        for key in ("stress_case_id", "summary_file", "run_tag", "stress_profile_set"):
            _check_type(key, (str,))
        if "stress_case_id" in values:
            token = str(values["stress_case_id"]).strip()
            if token and token not in AUTHORITY_STRESS_CASE_CHOICES:
                raise ValueError(
                    "profile key 'stress_case_id' has invalid value: "
                    f"{values['stress_case_id']}. "
                    f"Expected one of: {', '.join(AUTHORITY_STRESS_CASE_CHOICES)}."
                )
        return

    raise ValueError(f"Unsupported command for profile validation: {command}")


def load_profile(
    profile_path: Path,
    command: str,
) -> Dict[str, Any]:
    payload = load_json(profile_path)
    required_root = (
        "contract_version",
        "command",
        "saved_at_utc",
        "argument_values",
        "python",
        "cwd",
    )
    for key in required_root:
        if key not in payload:
            raise ValueError(f"profile missing required field: {key}")
    if str(payload.get("contract_version")) != PROFILE_CONTRACT_VERSION:
        raise ValueError(
            "profile contract_version mismatch. "
            f"Expected {PROFILE_CONTRACT_VERSION}, got {payload.get('contract_version')}."
        )
    if str(payload.get("command")) != command:
        raise ValueError(
            f"profile command mismatch. Expected {command}, got {payload.get('command')}."
        )
    values = payload.get("argument_values")
    if not isinstance(values, dict):
        raise ValueError("profile.argument_values must be an object.")
    allowed = set(profile_allowed_keys(command))
    unknown = sorted(set(values.keys()) - allowed)
    if unknown:
        raise ValueError(
            "profile contains unknown argument keys: "
            + ", ".join(unknown)
        )
    validate_profile_values(command=command, values=values)
    return values


def save_profile(
    profile_path: Path,
    command: str,
    values: Dict[str, Any],
    python_bin: str,
    cwd: str,
) -> None:
    payload = {
        "contract_version": PROFILE_CONTRACT_VERSION,
        "command": command,
        "saved_at_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argument_values": values,
        "python": python_bin,
        "cwd": cwd,
    }
    atomic_write_json(profile_path, payload)


def merged_seed(
    key: str,
    default: Any,
    profile_values: Dict[str, Any],
    explicit_values: Dict[str, Any],
) -> Tuple[Any, str]:
    if key in explicit_values:
        return explicit_values[key], "cli"
    if key in profile_values:
        return profile_values[key], "profile"
    return default, "default"


def require_string(value: Any, field: str) -> str:
    out = str(value).strip()
    if not out:
        raise ValueError(f"{field} is required.")
    return out


def maybe_warn_optional_backends(model_pool: str, include_optional_models: bool) -> None:
    requested = [x.strip().lower() for x in str(model_pool).split(",") if x.strip()]
    if include_optional_models:
        requested.extend(["xgboost", "catboost", "lightgbm", "tabpfn"])
    requested = sorted(set(requested))
    for backend, install_hint in TRAIN_OPTIONAL_BACKEND_HINTS.items():
        if backend not in requested:
            continue
        if importlib.util.find_spec(backend) is None:
            print(
                f"[WARN] Optional backend '{backend}' is not installed. "
                f"Install with `{install_hint}` or remove '{backend}' from model pool."
            )


def collect_init_values(profile: Dict[str, Any], explicit: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    seed, source = merged_seed("project_root", "", profile, explicit)
    if source == "cli":
        values["project_root"] = require_string(seed, "project_root")
    else:
        values["project_root"] = prompt_path(
            label="Project root path",
            default=str(seed),
            required=True,
            must_exist=False,
        )

    for key, label, default in (
        ("study_id", "Study ID", "medical-prediction-v1"),
        ("target_name", "Target name", "disease_risk"),
        ("label_col", "Label column", "y"),
        ("patient_id_col", "Patient ID column", "patient_id"),
        ("index_time_col", "Index time column", "event_time"),
    ):
        seed, source = merged_seed(key, default, profile, explicit)
        if source == "cli":
            values[key] = require_string(seed, key)
        else:
            values[key] = prompt_text(label=label, default=str(seed), required=True)

    seed, source = merged_seed("force", False, profile, explicit)
    if source == "cli":
        values["force"] = bool(seed)
    else:
        values["force"] = prompt_bool("Overwrite existing config files (--force)?", default=bool(seed))
    return values


def collect_workflow_values(profile: Dict[str, Any], explicit: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    seed, source = merged_seed("request", "", profile, explicit)
    if source == "cli":
        request_path = normalize_path(require_string(seed, "request"))
        if not Path(request_path).exists():
            raise ValueError(f"request path not found: {request_path}")
        values["request"] = request_path
    else:
        values["request"] = prompt_path(
            label="Request JSON path",
            default=str(seed),
            required=True,
            must_exist=True,
        )
    project_base = infer_project_base_from_request(values["request"])
    baseline_manifest = project_base / "evidence" / "manifest_baseline.bootstrap.json"

    seed, source = merged_seed("evidence_dir", "evidence", profile, explicit)
    if source == "cli":
        values["evidence_dir"] = normalize_path(require_string(seed, "evidence_dir"))
    else:
        evidence_default = str(seed).strip()
        if not evidence_default or evidence_default == "evidence":
            evidence_default = str((project_base / "evidence").resolve())
        values["evidence_dir"] = prompt_path(
            label="Evidence output directory",
            default=evidence_default,
            required=True,
            must_exist=False,
        )

    seed, source = merged_seed("compare_manifest", "", profile, explicit)
    if source == "cli":
        value = str(seed).strip()
        if value:
            value = normalize_path(value)
            if not Path(value).exists():
                raise ValueError(f"compare manifest path not found: {value}")
        values["compare_manifest"] = value
    else:
        compare_default = str(seed).strip()
        if not compare_default and baseline_manifest.exists():
            compare_default = str(baseline_manifest.resolve())
        values["compare_manifest"] = prompt_path(
            label="Compare manifest path (optional)",
            default=compare_default,
            required=False,
            must_exist=True,
            allow_empty=True,
        )

    seed, source = merged_seed("allow_missing_compare", False, profile, explicit)
    if source == "cli":
        values["allow_missing_compare"] = bool(seed)
    else:
        default_allow_missing = bool(seed)
        if source == "default":
            # First run bootstrap should be easy: if no baseline manifest is provided/found,
            # default to allow missing compare.
            default_allow_missing = not bool(values["compare_manifest"])
        values["allow_missing_compare"] = prompt_bool(
            "Allow missing compare manifest for bootstrap run?",
            default=default_allow_missing,
        )

    seed, source = merged_seed("continue_on_fail", False, profile, explicit)
    if source == "cli":
        values["continue_on_fail"] = bool(seed)
    else:
        values["continue_on_fail"] = prompt_bool(
            "Enable strict pipeline diagnostic mode (--continue-on-fail)?",
            default=bool(seed),
        )
    return values


def run_auto_split(
    input_csv: str,
    output_dir: str,
    patient_id_col: str,
    target_col: str,
    time_col: str,
    strategy: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int = 20260228,
    print_only: bool = False,
) -> Tuple[str, str, str, List[str]]:
    """Run split_data.py and return (train_path, valid_path, test_path, cmd)."""
    split_script = str(SCRIPTS_ROOT / "split_data.py")
    cmd = [
        sys.executable, split_script,
        "--input", input_csv,
        "--output-dir", output_dir,
        "--patient-id-col", patient_id_col,
        "--target-col", target_col,
        "--strategy", strategy,
        "--train-ratio", str(train_ratio),
        "--valid-ratio", str(valid_ratio),
        "--test-ratio", str(test_ratio),
        "--seed", str(seed),
    ]
    if time_col:
        cmd.extend(["--time-col", time_col])

    out_dir = Path(output_dir).expanduser().resolve()
    train_path = str(out_dir / "train.csv")
    valid_path = str(out_dir / "valid.csv")
    test_path = str(out_dir / "test.csv")

    if print_only:
        return train_path, valid_path, test_path, cmd

    print(f"\n[INFO] Running auto-split: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise ValueError("Auto-split failed. Check error messages above.")
    if not Path(train_path).exists() or not Path(valid_path).exists() or not Path(test_path).exists():
        raise ValueError("Auto-split did not produce expected output files.")

    return train_path, valid_path, test_path, cmd


def collect_train_values(profile: Dict[str, Any], explicit: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    # Determine data input mode: single CSV or pre-split files
    input_csv_seed, input_csv_source = merged_seed("input_csv", "", profile, explicit)
    train_seed, train_source = merged_seed("train", "", profile, explicit)

    # Auto-detect mode: if --input-csv is provided via CLI, use single_csv mode
    if input_csv_source == "cli" and str(input_csv_seed).strip():
        data_mode = "single_csv"
    elif train_source == "cli" and str(train_seed).strip():
        data_mode = "pre_split"
    elif PROMPT_AUTO_ACCEPT_DEFAULTS:
        data_mode = "pre_split" if str(train_seed).strip() else "single_csv" if str(input_csv_seed).strip() else "pre_split"
    else:
        data_mode = prompt_choice(
            label="\nData input mode:",
            choices=("pre_split", "single_csv"),
            default="pre_split",
        )
        if data_mode == "pre_split":
            print("  → Using pre-split train/valid/test files.")
        else:
            print("  → Will auto-split a single CSV into train/valid/test with medical safety guarantees.")

    values["__data_mode__"] = data_mode

    if data_mode == "single_csv":
        # Collect single-file split parameters
        if input_csv_source == "cli":
            input_csv = normalize_path(require_string(input_csv_seed, "input_csv"))
            if not Path(input_csv).exists():
                raise ValueError(f"input_csv path not found: {input_csv}")
        else:
            input_csv = prompt_path(
                label="Complete CSV file path",
                default=str(input_csv_seed),
                required=True,
                must_exist=True,
            )
        values["input_csv"] = input_csv

        # Read columns for data-aware prompts
        input_columns: List[str] = []
        try:
            input_columns = read_csv_columns(input_csv)
            if input_columns:
                print(f"\n  [INFO] CSV has {len(input_columns)} columns: {', '.join(input_columns[:25])}")
                if len(input_columns) > 25:
                    print(f"         ... and {len(input_columns) - 25} more")
        except Exception as exc:
            print(f"  [WARN] Could not read CSV headers: {exc}", file=sys.stderr)

        # Target column
        seed_target, source_target = merged_seed("target_col", "y", profile, explicit)
        if source_target == "cli":
            values["target_col"] = require_string(seed_target, "target_col")
        elif input_columns:
            values["target_col"] = prompt_column_choice(
                label="  Select TARGET column (must be binary 0/1):",
                columns=input_columns,
                default=str(seed_target),
                required=True,
                validate_fn=lambda col: validate_binary_target(input_csv, col),
            )
        else:
            values["target_col"] = prompt_text(label="Target column", default=str(seed_target), required=True)

        # Patient ID column
        seed_pid, source_pid = merged_seed("patient_id_col", "patient_id", profile, explicit)
        if source_pid == "cli":
            values["patient_id_col"] = require_string(seed_pid, "patient_id_col")
        elif input_columns:
            values["patient_id_col"] = prompt_column_choice(
                label="  Select PATIENT ID column:",
                columns=input_columns,
                default=str(seed_pid),
                required=True,
            )
        else:
            values["patient_id_col"] = prompt_text(label="Patient ID column", default=str(seed_pid), required=True)

        if values["target_col"] == values["patient_id_col"]:
            msg = f"target_col and patient_id_col are both '{values['target_col']}'. These must be different columns."
            if PROMPT_AUTO_ACCEPT_DEFAULTS:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")

        # Time column
        seed_time, source_time = merged_seed("time_col", "event_time", profile, explicit)
        if source_time == "cli":
            values["time_col"] = str(seed_time).strip()
        elif input_columns:
            default_time = str(seed_time) if str(seed_time) in input_columns else ""
            if not default_time:
                for alias in ["event_time", "index_time", "timestamp", "admit_time", "date"]:
                    if alias in input_columns:
                        default_time = alias
                        break
            values["time_col"] = prompt_column_choice(
                label="  Select TIME column (for temporal splitting; leave empty for random split):",
                columns=input_columns,
                default=default_time,
                required=False,
            )
        else:
            values["time_col"] = prompt_text(label="Time column (optional)", default=str(seed_time), required=False)

        # Strategy
        seed_strat, source_strat = merged_seed("split_strategy", "", profile, explicit)
        if source_strat == "cli":
            values["split_strategy"] = str(seed_strat).strip()
        elif PROMPT_AUTO_ACCEPT_DEFAULTS:
            values["split_strategy"] = "grouped_temporal" if values.get("time_col") else "grouped_random"
        else:
            default_strat = "grouped_temporal" if values.get("time_col") else "grouped_random"
            values["split_strategy"] = prompt_choice(
                label="Split strategy:",
                choices=SPLIT_STRATEGY_CHOICES,
                default=str(seed_strat) if str(seed_strat) in SPLIT_STRATEGY_CHOICES else default_strat,
            )

        if values["split_strategy"] == "grouped_temporal" and not values.get("time_col"):
            if PROMPT_AUTO_ACCEPT_DEFAULTS:
                values["split_strategy"] = "grouped_random"
                print("  [INFO] No time column; auto-switched to grouped_random strategy.", file=sys.stderr)
            else:
                print("  [WARN] grouped_temporal requires a time column. Switching to grouped_random.")
                values["split_strategy"] = "grouped_random"

        # Ratios
        for ratio_key, ratio_label, ratio_default in [
            ("train_ratio", "Train ratio", 0.6),
            ("valid_ratio", "Valid ratio", 0.2),
            ("test_ratio", "Test ratio", 0.2),
        ]:
            seed_r, source_r = merged_seed(ratio_key, ratio_default, profile, explicit)
            if source_r == "cli":
                values[ratio_key] = float(seed_r)
            elif PROMPT_AUTO_ACCEPT_DEFAULTS:
                values[ratio_key] = float(seed_r)
            else:
                raw = prompt_text(label=ratio_label, default=str(seed_r), required=True)
                values[ratio_key] = float(raw)

        # Output dir
        seed_out, source_out = merged_seed("split_output_dir", "", profile, explicit)
        if source_out == "cli":
            values["split_output_dir"] = normalize_path(require_string(seed_out, "split_output_dir"))
        else:
            input_parent = Path(input_csv).expanduser().resolve().parent
            default_out = str(input_parent)
            if PROMPT_AUTO_ACCEPT_DEFAULTS:
                values["split_output_dir"] = default_out
            else:
                values["split_output_dir"] = prompt_path(
                    label="Split output directory",
                    default=str(seed_out) if str(seed_out).strip() else default_out,
                    required=True,
                    must_exist=False,
                )

        # Ignore columns — default dynamically from user-selected patient_id + time columns
        _dynamic_ignore_parts = [values["patient_id_col"]]
        if values.get("time_col"):
            _dynamic_ignore_parts.append(values["time_col"])
        _dynamic_ignore_default = ",".join(_dynamic_ignore_parts)
        seed_ignore, source_ignore = merged_seed("ignore_cols", _dynamic_ignore_default, profile, explicit)
        if source_ignore == "cli":
            values["ignore_cols"] = str(seed_ignore).strip()
        elif input_columns:
            values["ignore_cols"] = prompt_ignore_cols(
                columns=input_columns,
                target_col=values["target_col"],
                patient_id_col=values["patient_id_col"],
                default=str(seed_ignore),
                csv_path=values.get("input_csv"),
            )
        else:
            values["ignore_cols"] = prompt_text(
                label="Ignore columns CSV",
                default=str(seed_ignore),
                required=False,
            )

        # Set train/valid/test paths from split output (will be populated after split runs)
        out_dir = Path(values["split_output_dir"]).expanduser().resolve()
        values["train"] = str(out_dir / "train.csv")
        values["valid"] = str(out_dir / "valid.csv")
        values["test"] = str(out_dir / "test.csv")

        project_base = out_dir.parent if out_dir.name.lower() == "data" else out_dir
        default_evidence_dir = (project_base / "evidence").resolve()

    else:
        # Pre-split mode (original flow)
        inferred_splits = infer_pre_split_paths_from_context() if PROMPT_AUTO_ACCEPT_DEFAULTS else None
        for key, label in (
            ("train", "Train split CSV path"),
            ("valid", "Valid split CSV path"),
            ("test", "Test split CSV path"),
        ):
            seed, source = merged_seed(key, "", profile, explicit)
            if source == "cli":
                value = normalize_path(require_string(seed, key))
                if not Path(value).exists():
                    raise ValueError(f"{key} path not found: {value}")
                values[key] = value
            else:
                default_value = str(seed).strip()
                if not default_value and inferred_splits:
                    default_value = str(inferred_splits.get(key, "")).strip()
                if PROMPT_AUTO_ACCEPT_DEFAULTS and not default_value:
                    raise ValueError(
                        "No default pre-split files detected for --accept-defaults. "
                        "Expected data/train.csv, data/valid.csv, data/test.csv under "
                        f"working directory ({Path.cwd()}) or split_paths in configs/request.json. "
                        "Provide --train/--valid/--test or switch to --input-csv."
                    )
                values[key] = prompt_path(
                    label=label,
                    default=default_value,
                    required=True,
                    must_exist=True,
                )
        project_base = infer_project_base_from_split_path(values["train"])
        default_evidence_dir = (project_base / "evidence").resolve()

    if data_mode == "pre_split":
        # Read CSV columns for data-aware prompts
        train_columns: List[str] = []
        try:
            train_columns = read_csv_columns(values["train"])
            if train_columns:
                print(f"\n  [INFO] Train CSV has {len(train_columns)} columns: {', '.join(train_columns[:25])}")
                if len(train_columns) > 25:
                    print(f"         ... and {len(train_columns) - 25} more")
        except Exception as exc:
            print(f"  [WARN] Could not read train CSV headers: {exc}", file=sys.stderr)

        # Target column: show available columns + validate binary
        seed_target, source_target = merged_seed("target_col", "y", profile, explicit)
        if source_target == "cli":
            values["target_col"] = require_string(seed_target, "target_col")
        elif train_columns:
            values["target_col"] = prompt_column_choice(
                label="  Select TARGET column (must be binary 0/1):",
                columns=train_columns,
                default=str(seed_target),
                required=True,
                validate_fn=lambda col: validate_binary_target(values["train"], col),
            )
        else:
            values["target_col"] = prompt_text(label="Target column", default=str(seed_target), required=True)

        # Patient ID column: show available columns
        seed_pid, source_pid = merged_seed("patient_id_col", "patient_id", profile, explicit)
        if source_pid == "cli":
            values["patient_id_col"] = require_string(seed_pid, "patient_id_col")
        elif train_columns:
            values["patient_id_col"] = prompt_column_choice(
                label="  Select PATIENT ID column:",
                columns=train_columns,
                default=str(seed_pid),
                required=True,
            )
        else:
            values["patient_id_col"] = prompt_text(label="Patient ID column", default=str(seed_pid), required=True)

        # Cross-validate: target and patient_id must differ
        if values["target_col"] == values["patient_id_col"]:
            msg = (
                f"target_col and patient_id_col are both '{values['target_col']}'. "
                "These must be different columns."
            )
            if PROMPT_AUTO_ACCEPT_DEFAULTS:
                raise ValueError(msg)
            print(f"  [WARN] {msg}")

        # Ignore columns: show feature preview
        seed_ignore, source_ignore = merged_seed("ignore_cols", "patient_id,event_time", profile, explicit)
        if source_ignore == "cli":
            values["ignore_cols"] = str(seed_ignore).strip()
        elif train_columns:
            values["ignore_cols"] = prompt_ignore_cols(
                columns=train_columns,
                target_col=values["target_col"],
                patient_id_col=values["patient_id_col"],
                default=str(seed_ignore),
                csv_path=values.get("train"),
            )
        else:
            values["ignore_cols"] = prompt_text(
                label="Ignore columns CSV (non-feature columns to exclude)",
                default=str(seed_ignore),
                required=False,
            )

    seed, source = merged_seed("model_pool", TRAIN_MODEL_POOL_DEFAULT, profile, explicit)
    if source == "cli":
        values["model_pool"] = require_string(seed, "model_pool")
    else:
        values["model_pool"] = prompt_text(
            label="Model pool CSV",
            default=str(seed),
            required=True,
        )

    seed, source = merged_seed("include_optional_models", False, profile, explicit)
    if source == "cli":
        values["include_optional_models"] = bool(seed)
    else:
        values["include_optional_models"] = prompt_bool(
            "Include optional model backends when installed (--include-optional-models)?",
            default=bool(seed),
        )

    seed, source = merged_seed("ensemble_top_k", 0, profile, explicit)
    if source == "cli":
        values["ensemble_top_k"] = int(seed) if seed is not None else 0
    else:
        values["ensemble_top_k"] = prompt_int(
            label="Ensemble top-K (0=disabled; 3+ builds voting/stacking from top-K base models)",
            default=int(seed) if seed is not None else 0,
            min_value=0,
            max_value=20,
        )

    seed, source = merged_seed("n_jobs", 1, profile, explicit)
    if source == "cli":
        values["n_jobs"] = int(seed)
    else:
        values["n_jobs"] = prompt_int(
            label="CPU workers (--n-jobs, -1 means all cores)",
            default=int(seed),
            min_value=-1,
            max_value=256,
        )

    seed, source = merged_seed("calibration_method", "none", profile, explicit)
    if source == "cli":
        value = str(seed).strip()
        if value not in TRAIN_CALIBRATION_CHOICES:
            raise ValueError(
                "calibration_method must be one of: "
                + ", ".join(TRAIN_CALIBRATION_CHOICES)
            )
        values["calibration_method"] = value
    else:
        values["calibration_method"] = prompt_choice(
            label="Calibration method",
            choices=TRAIN_CALIBRATION_CHOICES,
            default=str(seed),
        )

    seed, source = merged_seed("feature_group_spec", "", profile, explicit)
    if source == "cli":
        value = str(seed).strip()
        if value:
            value = normalize_path(value)
            if not Path(value).exists():
                raise ValueError(f"feature_group_spec path not found: {value}")
        values["feature_group_spec"] = value
    else:
        values["feature_group_spec"] = prompt_path(
            label="Feature group spec JSON path (optional)",
            default=str(seed),
            required=False,
            must_exist=True,
            allow_empty=True,
        )

    seed, source = merged_seed("external_cohort_spec", "", profile, explicit)
    if source == "cli":
        value = str(seed).strip()
        if value:
            value = normalize_path(value)
            if not Path(value).exists():
                raise ValueError(f"external_cohort_spec path not found: {value}")
        values["external_cohort_spec"] = value
    else:
        values["external_cohort_spec"] = prompt_path(
            label="External cohort spec JSON path (optional)",
            default=str(seed),
            required=False,
            must_exist=True,
            allow_empty=True,
        )

    required_artifact_defaults = (
        (
            "model_selection_report_out",
            "Model selection report output",
            str(default_evidence_dir / "model_selection_report.json"),
        ),
        (
            "evaluation_report_out",
            "Evaluation report output",
            str(default_evidence_dir / "evaluation_report.json"),
        ),
        (
            "prediction_trace_out",
            "Prediction trace output",
            str(default_evidence_dir / "prediction_trace.csv.gz"),
        ),
        (
            "ci_matrix_report_out",
            "CI matrix report output",
            str(default_evidence_dir / "ci_matrix_report.json"),
        ),
        (
            "distribution_report_out",
            "Distribution report output",
            str(default_evidence_dir / "distribution_report.json"),
        ),
        (
            "robustness_report_out",
            "Robustness report output",
            str(default_evidence_dir / "robustness_report.json"),
        ),
        (
            "seed_sensitivity_out",
            "Seed sensitivity report output",
            str(default_evidence_dir / "seed_sensitivity_report.json"),
        ),
    )
    for key, label, default in required_artifact_defaults:
        seed, source = merged_seed(key, default, profile, explicit)
        if source == "cli":
            values[key] = normalize_path(require_string(seed, key))
        else:
            values[key] = prompt_path(
                label=label,
                default=str(seed),
                required=True,
                must_exist=False,
            )

    # external validation report must be paired with external cohort spec.
    seed, source = merged_seed(
        "external_validation_report_out",
        str(default_evidence_dir / "external_validation_report.json"),
        profile,
        explicit,
    )
    if values.get("external_cohort_spec"):
        if source == "cli":
            values["external_validation_report_out"] = normalize_path(
                require_string(seed, "external_validation_report_out")
            )
        else:
            values["external_validation_report_out"] = prompt_path(
                label="External validation report output",
                default=str(seed),
                required=True,
                must_exist=False,
            )
    else:
        if source == "cli" and str(seed).strip():
            raise ValueError(
                "external_validation_report_out requires external_cohort_spec."
            )
        values["external_validation_report_out"] = ""

    # feature engineering report requires feature_group_spec.
    seed, source = merged_seed(
        "feature_engineering_report_out",
        str(default_evidence_dir / "feature_engineering_report.json"),
        profile,
        explicit,
    )
    if values.get("feature_group_spec"):
        if source == "cli":
            values["feature_engineering_report_out"] = normalize_path(
                require_string(seed, "feature_engineering_report_out")
            )
        else:
            values["feature_engineering_report_out"] = prompt_path(
                label="Feature engineering report output",
                default=str(seed),
                required=True,
                must_exist=False,
            )
    else:
        if source == "cli" and str(seed).strip():
            raise ValueError(
                "feature_engineering_report_out requires feature_group_spec."
            )
        values["feature_engineering_report_out"] = ""

    if bool(values.get("external_cohort_spec")) != bool(values.get("external_validation_report_out")):
        raise ValueError(
            "external_cohort_spec and external_validation_report_out must be provided together."
        )
    if values.get("feature_engineering_report_out") and not values.get("feature_group_spec"):
        raise ValueError(
            "feature_engineering_report_out requires feature_group_spec."
        )

    maybe_warn_optional_backends(
        model_pool=values["model_pool"],
        include_optional_models=bool(values["include_optional_models"]),
    )
    return values


def collect_authority_values(profile: Dict[str, Any], explicit: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    seed, source = merged_seed("include_stress_cases", True, profile, explicit)
    if source == "cli":
        values["include_stress_cases"] = bool(seed)
    else:
        values["include_stress_cases"] = prompt_bool(
            "Include stress benchmark cases (--include-stress-cases)?",
            default=bool(seed),
        )

    seed, source = merged_seed("summary_file", str(EXPERIMENTS_ROOT / "authority_e2e_summary.json"), profile, explicit)
    if source == "cli":
        values["summary_file"] = normalize_path(require_string(seed, "summary_file"))
    else:
        values["summary_file"] = prompt_path(
            label="Authority summary output file",
            default=str(seed),
            required=True,
            must_exist=False,
        )

    seed, source = merged_seed("run_tag", "", profile, explicit)
    if source == "cli":
        values["run_tag"] = str(seed).strip()
    else:
        values["run_tag"] = prompt_text(
            label="Run tag (optional, leave empty for auto UTC tag)",
            default=str(seed),
            required=False,
        ).strip()

    seed, source = merged_seed("stress_profile_set", "strict_v1", profile, explicit)
    if source == "cli":
        values["stress_profile_set"] = require_string(seed, "stress_profile_set")
    else:
        values["stress_profile_set"] = prompt_text(
            label="Stress profile set",
            default=str(seed),
            required=True,
        )

    if values["include_stress_cases"]:
        seed, source = merged_seed("stress_case_id", AUTHORITY_STRESS_CASE_CHOICES[0], profile, explicit)
        if source == "cli":
            value = str(seed).strip()
            if value not in AUTHORITY_STRESS_CASE_CHOICES:
                raise ValueError(
                    "stress_case_id must be one of: "
                    + ", ".join(AUTHORITY_STRESS_CASE_CHOICES)
                )
            values["stress_case_id"] = value
        else:
            values["stress_case_id"] = prompt_authority_stress_case(default=str(seed))

        seed, source = merged_seed("stress_seed_search", False, profile, explicit)
        if source == "cli":
            values["stress_seed_search"] = bool(seed)
        else:
            default_seed_search = bool(seed)
            if source == "default" and values["stress_case_id"] == "uci-heart-disease":
                default_seed_search = True
            values["stress_seed_search"] = prompt_bool(
                "Enable stress seed search (--stress-seed-search)?",
                default=default_seed_search,
            )
        if values["stress_case_id"] == "uci-heart-disease":
            print(
                "[INFO] Selected heart stress path. This is an advanced research/high-pressure route; "
                "release-ready candidates are not guaranteed in every seed range."
            )
    else:
        values["stress_case_id"] = ""
        values["stress_seed_search"] = False
    return values


def collect_values(command: str, profile: Dict[str, Any], explicit: Dict[str, Any]) -> Dict[str, Any]:
    if command == "init":
        return collect_init_values(profile, explicit)
    if command == "workflow":
        return collect_workflow_values(profile, explicit)
    if command == "train":
        return collect_train_values(profile, explicit)
    if command == "authority":
        return collect_authority_values(profile, explicit)
    raise ValueError(f"Unsupported command: {command}")


def build_command(command: str, python_bin: str, values: Dict[str, Any]) -> List[str]:
    script = COMMAND_SCRIPT[command]
    if not script.exists():
        raise FileNotFoundError(f"Script not found for command '{command}': {script}")

    cmd: List[str] = [python_bin, str(script)]
    if command == "init":
        cmd.extend(["--project-root", str(values["project_root"])])
        cmd.extend(["--study-id", str(values["study_id"])])
        cmd.extend(["--target-name", str(values["target_name"])])
        cmd.extend(["--label-col", str(values["label_col"])])
        cmd.extend(["--patient-id-col", str(values["patient_id_col"])])
        cmd.extend(["--index-time-col", str(values["index_time_col"])])
        if bool(values.get("force", False)):
            cmd.append("--force")
    elif command == "workflow":
        cmd.extend(["--request", str(values["request"])])
        cmd.extend(["--evidence-dir", str(values["evidence_dir"])])
        if str(values.get("compare_manifest", "")).strip():
            cmd.extend(["--compare-manifest", str(values["compare_manifest"])])
        if bool(values.get("allow_missing_compare", False)):
            cmd.append("--allow-missing-compare")
        if bool(values.get("continue_on_fail", False)):
            cmd.append("--continue-on-fail")
        cmd.append("--strict")
    elif command == "train":
        cmd.extend(["--train", str(values["train"])])
        cmd.extend(["--valid", str(values["valid"])])
        cmd.extend(["--test", str(values["test"])])
        cmd.extend(["--target-col", str(values["target_col"])])
        cmd.extend(["--patient-id-col", str(values["patient_id_col"])])
        if str(values.get("ignore_cols", "")).strip():
            cmd.extend(["--ignore-cols", str(values["ignore_cols"])])
        cmd.extend(["--model-pool", str(values["model_pool"])])
        if bool(values.get("include_optional_models", False)):
            cmd.append("--include-optional-models")
        ensemble_top_k = int(values.get("ensemble_top_k", 0) or 0)
        if ensemble_top_k > 0:
            cmd.extend(["--ensemble-top-k", str(ensemble_top_k)])
        cmd.extend(["--n-jobs", str(int(values["n_jobs"]))])
        cmd.extend(["--calibration-method", str(values["calibration_method"])])
        if str(values.get("feature_group_spec", "")).strip():
            cmd.extend(["--feature-group-spec", str(values["feature_group_spec"])])
        if str(values.get("external_cohort_spec", "")).strip():
            cmd.extend(["--external-cohort-spec", str(values["external_cohort_spec"])])
        cmd.extend(["--model-selection-report-out", str(values["model_selection_report_out"])])
        cmd.extend(["--evaluation-report-out", str(values["evaluation_report_out"])])
        cmd.extend(["--prediction-trace-out", str(values["prediction_trace_out"])])
        if str(values.get("external_validation_report_out", "")).strip():
            cmd.extend(["--external-validation-report-out", str(values["external_validation_report_out"])])
        cmd.extend(["--ci-matrix-report-out", str(values["ci_matrix_report_out"])])
        cmd.extend(["--distribution-report-out", str(values["distribution_report_out"])])
        if str(values.get("feature_engineering_report_out", "")).strip():
            cmd.extend(["--feature-engineering-report-out", str(values["feature_engineering_report_out"])])
        cmd.extend(["--robustness-report-out", str(values["robustness_report_out"])])
        cmd.extend(["--seed-sensitivity-out", str(values["seed_sensitivity_out"])])
    elif command == "authority":
        cmd.extend(["--summary-file", str(values["summary_file"])])
        if str(values.get("run_tag", "")).strip():
            cmd.extend(["--run-tag", str(values["run_tag"])])
        if str(values.get("stress_profile_set", "")).strip():
            cmd.extend(["--stress-profile-set", str(values["stress_profile_set"])])
        if bool(values.get("include_stress_cases", False)):
            cmd.append("--include-stress-cases")
            if str(values.get("stress_case_id", "")).strip():
                cmd.extend(["--stress-case-id", str(values["stress_case_id"])])
            if bool(values.get("stress_seed_search", False)):
                cmd.append("--stress-seed-search")
    return cmd


def confirm_execute() -> bool:
    return prompt_bool("Execute this command now?", default=True)


def main() -> int:
    global PROMPT_AUTO_ACCEPT_DEFAULTS
    args, passthrough = parse_args()
    PROMPT_AUTO_ACCEPT_DEFAULTS = bool(args.accept_defaults)
    command = str(args.command)
    profile_name = str(args.profile_name).strip()
    profile_dir = str(args.profile_dir)
    cwd = Path(str(args.cwd)).expanduser().resolve(strict=False)
    python_bin = str(args.python).strip() or sys.executable
    if not cwd.exists() or not cwd.is_dir():
        return fail(f"--cwd path not found or not a directory: {cwd}")
    try:
        os.chdir(cwd)
    except Exception as exc:
        return fail(f"unable to change working directory to {cwd}: {exc}")

    if args.load_profile or args.save_profile:
        if not profile_name:
            return fail("--profile-name is required when using --load-profile or --save-profile.")
        try:
            profile_name = validate_profile_name(profile_name)
        except Exception as exc:
            return fail(f"invalid profile name: {exc}")

    profile_values: Dict[str, Any] = {}
    if args.load_profile:
        profile_path = profile_file_path(profile_dir=profile_dir, profile_name=profile_name)
        if not profile_path.exists():
            return fail(f"profile not found: {profile_path}")
        try:
            profile_values = load_profile(profile_path=profile_path, command=command)
        except Exception as exc:
            return fail(f"unable to load profile: {exc}")
        print(f"[INFO] Loaded profile: {profile_path}")

    try:
        explicit_values = parse_command_overrides(command=command, passthrough=passthrough)
    except Exception as exc:
        return fail(str(exc))

    if explicit_values:
        print("[INFO] CLI override keys: " + ", ".join(sorted(explicit_values.keys())))

    print(f"\n=== mlgg interactive wizard: {command} ===")
    try:
        values = collect_values(
            command=command,
            profile=profile_values,
            explicit=explicit_values,
        )
    except Exception as exc:
        return fail(f"interactive input validation failed: {exc}")

    if args.save_profile:
        profile_path = profile_file_path(profile_dir=profile_dir, profile_name=profile_name)
        try:
            save_profile(
                profile_path=profile_path,
                command=command,
                values=values,
                python_bin=python_bin,
                cwd=str(cwd),
            )
        except Exception as exc:
            return fail(f"unable to save profile: {exc}")
        print(f"[INFO] Saved profile: {profile_path}")

    # Handle auto-split for single_csv mode before building train command
    split_cmd: Optional[List[str]] = None
    if command == "train" and values.get("__data_mode__") == "single_csv":
        try:
            train_path, valid_path, test_path, split_cmd = run_auto_split(
                input_csv=values["input_csv"],
                output_dir=values["split_output_dir"],
                patient_id_col=values["patient_id_col"],
                target_col=values["target_col"],
                time_col=values.get("time_col", ""),
                strategy=values.get("split_strategy", "grouped_random"),
                train_ratio=float(values.get("train_ratio", 0.6)),
                valid_ratio=float(values.get("valid_ratio", 0.2)),
                test_ratio=float(values.get("test_ratio", 0.2)),
                print_only=bool(args.print_only),
            )
            values["train"] = train_path
            values["valid"] = valid_path
            values["test"] = test_path
        except Exception as exc:
            return fail(f"auto-split failed: {exc}")

    try:
        cmd = build_command(command=command, python_bin=python_bin, values=values)
    except Exception as exc:
        return fail(str(exc))

    if split_cmd:
        print("\nStep 1 — Auto-split command:")
        print(f"$ {shlex.join(split_cmd)}")
        print("\nStep 2 — Train command:")
    else:
        print("\nGenerated command:")
    print(f"$ {shlex.join(cmd)}")

    if args.print_only:
        print("[INFO] --print-only enabled, command not executed.")
        return 0

    try:
        proceed = confirm_execute()
    except ValueError as exc:
        return fail(str(exc))
    if not proceed:
        print("[INFO] Cancelled by user.")
        return 0

    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    return int(proc.returncode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
