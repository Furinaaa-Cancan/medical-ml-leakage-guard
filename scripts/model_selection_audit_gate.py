#!/usr/bin/env python3
"""
Fail-closed model-selection audit gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _gate_utils import add_issue, load_json_from_str as load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model-selection evidence for leakage-safe selection.")
    parser.add_argument("--model-selection-report", required=True, help="Path to model_selection_report.json.")
    parser.add_argument("--tuning-spec", required=True, help="Path to tuning protocol JSON.")
    parser.add_argument("--train", help="Optional train split CSV path for fingerprint verification.")
    parser.add_argument("--valid", help="Optional valid split CSV path for fingerprint verification.")
    parser.add_argument("--test", help="Optional test split CSV path for fingerprint verification.")
    parser.add_argument("--expected-primary-metric", default="pr_auc", help="Expected primary selection metric.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def canonical_metric_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def contains_test_token(value: str) -> bool:
    token = value.strip().lower()
    if not token:
        return False
    parts = [p for p in re.split(r"[^a-z0-9]+", token) if p]
    if "test" in parts:
        return True
    return any(p.startswith("test") or p.endswith("test") for p in parts if p not in {"latest", "attest"})


def finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def finite_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def finite_float_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or not value:
        return None
    out: List[float] = []
    for item in value:
        parsed = finite_float(item)
        if parsed is None:
            return None
        out.append(float(parsed))
    return out


def in_unit_interval(value: float, eps: float = 1e-12) -> bool:
    v = float(value)
    return (-eps) <= v <= (1.0 + eps)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path: Path) -> int:
    total = -1
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for total, _ in enumerate(fh):
            pass
    return max(total, 0)


def scan_candidate_for_test_usage(node: Any, path: str, hits: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            next_path = f"{path}.{key}" if path else key
            key_l = key.lower()
            if "test" in key_l and key_l not in {"test_used_for_model_selection"}:
                hits.append(next_path)
            scan_candidate_for_test_usage(value, next_path, hits)
        return
    if isinstance(node, list):
        for idx, value in enumerate(node):
            scan_candidate_for_test_usage(value, f"{path}[{idx}]", hits)


def extract_selection_tuple(
    candidate: Dict[str, Any], expected_metric: str
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[List[float]]]:
    metrics = candidate.get("selection_metrics")
    if not isinstance(metrics, dict):
        return None, None, None, None
    metric_block = metrics.get(expected_metric)
    if not isinstance(metric_block, dict):
        return None, None, None, None
    mean_value = finite_float(metric_block.get("mean"))
    std_value = finite_float(metric_block.get("std"))
    n_folds = finite_int(metric_block.get("n_folds"))
    fold_scores = finite_float_list(metric_block.get("fold_scores"))
    return mean_value, std_value, n_folds, fold_scores


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    try:
        model_selection = load_json(args.model_selection_report)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_model_selection_report",
            "Unable to parse model selection report JSON.",
            {"path": str(Path(args.model_selection_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        tuning = load_json(args.tuning_spec)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_tuning_spec",
            "Unable to parse tuning spec JSON.",
            {"path": str(Path(args.tuning_spec).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    expected_metric = canonical_metric_token(args.expected_primary_metric)
    if expected_metric != canonical_metric_token("pr_auc"):
        add_issue(
            warnings,
            "unexpected_primary_metric_override",
            "Model-selection audit expects pr_auc for publication-grade policy.",
            {"expected_primary_metric": args.expected_primary_metric},
        )

    selection_policy = model_selection.get("selection_policy")
    if not isinstance(selection_policy, dict):
        add_issue(
            failures,
            "missing_selection_policy",
            "model_selection_report must include selection_policy object.",
            {"migration_hint": "Include primary_metric/selection_data/one_se_rule policy fields."},
        )
        selection_policy = {}

    primary_metric = str(
        selection_policy.get("primary_metric", model_selection.get("primary_metric", ""))
    ).strip()
    if not primary_metric:
        add_issue(
            failures,
            "missing_selection_metric",
            "Selection policy must declare primary_metric.",
            {},
        )
    elif canonical_metric_token(primary_metric) != canonical_metric_token("pr_auc"):
        add_issue(
            failures,
            "selection_metric_mismatch",
            "Publication-grade selection metric must be pr_auc.",
            {"declared_primary_metric": primary_metric, "expected": "pr_auc"},
        )

    selection_data = str(selection_policy.get("selection_data", model_selection.get("selection_data", ""))).strip().lower()
    allowed_selection_data = {"valid", "cv_inner", "nested_cv"}
    if not selection_data:
        add_issue(
            failures,
            "missing_selection_data",
            "Selection policy must declare selection_data scope.",
            {},
        )
    elif selection_data not in allowed_selection_data:
        add_issue(
            failures,
            "invalid_selection_data",
            "Selection data must be valid/cv_inner/nested_cv.",
            {"selection_data": selection_data, "allowed": sorted(allowed_selection_data)},
        )
    if selection_data and contains_test_token(selection_data):
        add_issue(
            failures,
            "test_data_usage_detected",
            "Selection data scope references test split.",
            {"selection_data": selection_data},
        )

    test_used_for_model_selection = selection_policy.get(
        "test_used_for_model_selection", model_selection.get("test_used_for_model_selection")
    )
    if test_used_for_model_selection is True:
        add_issue(
            failures,
            "test_data_usage_detected",
            "test_used_for_model_selection must be false.",
            {"test_used_for_model_selection": test_used_for_model_selection},
        )

    if tuning.get("test_used_for_model_selection") is True:
        add_issue(
            failures,
            "test_data_usage_detected",
            "tuning_spec indicates test usage for model selection.",
            {"field": "test_used_for_model_selection"},
        )
    objective_metric = str(tuning.get("objective_metric", "")).strip()
    if objective_metric and canonical_metric_token(objective_metric) != canonical_metric_token("pr_auc"):
        add_issue(
            failures,
            "selection_metric_mismatch",
            "tuning_spec objective_metric must be pr_auc for this policy.",
            {"objective_metric": objective_metric, "expected": "pr_auc"},
        )

    model_selection_data = str(tuning.get("model_selection_data", "")).strip().lower()
    if model_selection_data and model_selection_data not in allowed_selection_data:
        add_issue(
            failures,
            "invalid_selection_data",
            "tuning_spec.model_selection_data must be valid/cv_inner/nested_cv.",
            {"model_selection_data": model_selection_data},
        )
    if model_selection_data and contains_test_token(model_selection_data):
        add_issue(
            failures,
            "test_data_usage_detected",
            "tuning_spec model_selection_data references test.",
            {"model_selection_data": model_selection_data},
        )
    if selection_data and model_selection_data and selection_data != model_selection_data:
        add_issue(
            failures,
            "selection_data_spec_mismatch",
            "model_selection_report.selection_policy.selection_data must match tuning_spec.model_selection_data.",
            {"selection_data_report": selection_data, "selection_data_tuning_spec": model_selection_data},
        )

    one_se_rule = selection_policy.get("one_se_rule")
    if one_se_rule is not True:
        add_issue(
            failures,
            "one_se_rule_not_enabled",
            "Model selection must use one-SE rule for complexity-aware selection.",
            {"one_se_rule": one_se_rule},
        )

    candidates = model_selection.get("candidates")
    if not isinstance(candidates, list):
        add_issue(
            failures,
            "missing_candidates",
            "model_selection_report.candidates must be an array.",
            {"migration_hint": "Provide candidate list with family/complexity_rank/selection_metrics."},
        )
        candidates = []

    if len(candidates) < 3:
        add_issue(
            failures,
            "candidate_pool_too_small",
            "Candidate model pool must contain at least three models.",
            {"candidate_count": len(candidates), "minimum_required": 3},
        )

    has_logistic = False
    suspicious_test_paths: List[str] = []
    candidate_rows: List[Dict[str, Any]] = []
    seen_model_ids: Dict[str, int] = {}
    selected_true_ids: List[str] = []
    for idx, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, dict):
            add_issue(
                failures,
                "invalid_candidate_entry",
                "Each candidate entry must be an object.",
                {"index": idx},
            )
            continue
        model_id = str(raw_candidate.get("model_id", "")).strip()
        family = str(raw_candidate.get("family", "")).strip().lower()
        complexity_rank = finite_int(raw_candidate.get("complexity_rank"))
        selected_flag = raw_candidate.get("selected")
        if selected_flag is not None and not isinstance(selected_flag, bool):
            add_issue(
                failures,
                "invalid_candidate_entry",
                "Candidate selected flag must be boolean when present.",
                {"candidate": model_id or f"index-{idx}", "selected": selected_flag},
            )
        if selected_flag is True and model_id:
            selected_true_ids.append(model_id)
        mean_value, std_value, n_folds, fold_scores = extract_selection_tuple(raw_candidate, "pr_auc")
        candidate_rows.append(
            {
                "index": idx,
                "model_id": model_id,
                "family": family,
                "complexity_rank": complexity_rank,
                "mean": mean_value,
                "std": std_value,
                "n_folds": n_folds,
                "fold_scores": fold_scores,
                "selected_flag": selected_flag,
            }
        )
        if model_id:
            seen_model_ids[model_id] = seen_model_ids.get(model_id, 0) + 1
        if "logistic" in family or "logistic" in model_id.lower():
            has_logistic = True
        scan_candidate_for_test_usage(raw_candidate, f"candidates[{idx}]", suspicious_test_paths)

    duplicate_model_ids = sorted([mid for mid, count in seen_model_ids.items() if count > 1])
    if duplicate_model_ids:
        add_issue(
            failures,
            "duplicate_candidate_model_id",
            "Candidate model_id values must be unique.",
            {"duplicate_model_ids": duplicate_model_ids},
        )

    if not has_logistic:
        add_issue(
            failures,
            "missing_logistic_baseline",
            "Candidate list must include an interpretable logistic baseline.",
            {},
        )

    if suspicious_test_paths:
        add_issue(
            failures,
            "test_data_usage_detected",
            "Candidate payload contains test-related fields; selection evidence must exclude test ranking artifacts.",
            {"paths": sorted(set(suspicious_test_paths))[:30]},
        )

    selected_model_id = str(model_selection.get("selected_model_id", "")).strip()
    if not selected_model_id:
        add_issue(
            failures,
            "missing_selected_model",
            "model_selection_report must include selected_model_id.",
            {},
        )
    elif selected_model_id not in seen_model_ids:
        add_issue(
            failures,
            "selected_model_not_in_candidates",
            "selected_model_id must reference one of the candidate model_id values.",
            {"selected_model_id": selected_model_id},
        )
    if selected_true_ids:
        if len(selected_true_ids) != 1:
            add_issue(
                failures,
                "invalid_selected_candidate_flags",
                "Exactly one candidate should be marked selected=true when selected flags are present.",
                {"selected_true_ids": selected_true_ids},
            )
        elif selected_model_id and selected_true_ids[0] != selected_model_id:
            add_issue(
                failures,
                "selected_model_flag_mismatch",
                "selected_model_id must match candidate selected=true marker.",
                {"selected_model_id": selected_model_id, "selected_true_id": selected_true_ids[0]},
            )

    valid_candidates: List[Dict[str, Any]] = []
    for row in candidate_rows:
        if (
            not row["model_id"]
            or row["complexity_rank"] is None
            or row["mean"] is None
            or row["std"] is None
            or row["n_folds"] is None
            or row["fold_scores"] is None
        ):
            add_issue(
                failures,
                "selection_replay_missing_fields",
                "Candidate is missing required replay fields for one-SE audit.",
                {
                    "candidate": row.get("model_id") or f"index-{row.get('index')}",
                    "required_fields": [
                        "model_id",
                        "complexity_rank",
                        "selection_metrics.pr_auc.mean/std/n_folds/fold_scores",
                    ],
                    "migration_hint": "Emit deterministic candidate fields to replay selection.",
                },
            )
            continue
        if int(row["complexity_rank"]) < 1:
            add_issue(
                failures,
                "invalid_candidate_complexity_rank",
                "complexity_rank must be integer >= 1.",
                {"candidate": row["model_id"], "complexity_rank": row["complexity_rank"]},
            )
            continue
        if not in_unit_interval(float(row["mean"])):
            add_issue(
                failures,
                "selection_metric_out_of_range",
                "selection_metrics.pr_auc.mean must be in [0,1].",
                {"candidate": row["model_id"], "mean": row["mean"]},
            )
            continue
        if not in_unit_interval(float(row["std"])):
            add_issue(
                failures,
                "selection_metric_out_of_range",
                "selection_metrics.pr_auc.std must be in [0,1].",
                {"candidate": row["model_id"], "std": row["std"]},
            )
            continue
        fold_scores = list(row["fold_scores"])
        if any(not in_unit_interval(float(x)) for x in fold_scores):
            add_issue(
                failures,
                "selection_metric_out_of_range",
                "selection_metrics.pr_auc.fold_scores values must be in [0,1].",
                {"candidate": row["model_id"], "fold_scores": fold_scores},
            )
            continue
        n_folds = int(row["n_folds"])
        if len(fold_scores) != n_folds:
            add_issue(
                failures,
                "selection_replay_fold_count_mismatch",
                "selection_metrics.pr_auc.n_folds must equal len(fold_scores).",
                {"candidate": row["model_id"], "n_folds": n_folds, "fold_score_count": len(fold_scores)},
            )
            continue
        if selection_data == "valid":
            if n_folds != 1:
                add_issue(
                    failures,
                    "selection_replay_invalid_n_folds",
                    "selection_data=valid requires n_folds=1.",
                    {"candidate": row["model_id"], "n_folds": n_folds},
                )
                continue
        elif n_folds < 2:
            add_issue(
                failures,
                "selection_replay_invalid_n_folds",
                "selection_metrics.pr_auc.n_folds must be >= 2 for CV-based selection.",
                {"candidate": row["model_id"], "n_folds": n_folds},
            )
            continue

        scores_arr = [float(x) for x in fold_scores]
        calc_mean = float(sum(scores_arr) / len(scores_arr))
        if len(scores_arr) > 1:
            mean = calc_mean
            calc_std = float(math.sqrt(sum((x - mean) ** 2 for x in scores_arr) / (len(scores_arr) - 1)))
        else:
            calc_std = 0.0
        if abs(calc_mean - float(row["mean"])) > 1e-9:
            add_issue(
                failures,
                "selection_metric_summary_mismatch",
                "selection_metrics.pr_auc.mean does not match fold_scores average.",
                {
                    "candidate": row["model_id"],
                    "reported_mean": row["mean"],
                    "computed_mean": calc_mean,
                },
            )
            continue
        if abs(calc_std - float(row["std"])) > 1e-9:
            add_issue(
                failures,
                "selection_metric_summary_mismatch",
                "selection_metrics.pr_auc.std does not match fold_scores sample std.",
                {
                    "candidate": row["model_id"],
                    "reported_std": row["std"],
                    "computed_std": calc_std,
                },
            )
            continue
        valid_candidates.append(row)

    fingerprint_summary: Dict[str, Any] = {}
    split_paths = {
        "train": args.train,
        "valid": args.valid,
        "test": args.test,
    }
    required_split_paths = {"train", "test"}
    if selection_data == "valid" or model_selection_data == "valid":
        required_split_paths.add("valid")
    if args.strict:
        for split_name in sorted(required_split_paths):
            raw_path = split_paths.get(split_name)
            if not isinstance(raw_path, str) or not raw_path.strip():
                add_issue(
                    failures,
                    "missing_split_file_for_fingerprint_verification",
                    "Strict mode requires split files for fingerprint verification.",
                    {"required_split": split_name, "provided_split_paths": {k: bool(v) for k, v in split_paths.items()}},
                )

    declared_fingerprints = model_selection.get("data_fingerprints")
    if args.strict and isinstance(declared_fingerprints, dict):
        for split_name in ("train", "valid", "test"):
            if split_name in declared_fingerprints:
                raw_path = split_paths.get(split_name)
                if not isinstance(raw_path, str) or not raw_path.strip():
                    add_issue(
                        failures,
                        "missing_split_file_for_fingerprint_verification",
                        "Declared fingerprint split cannot be verified without matching split CSV path.",
                        {"declared_split": split_name},
                    )
    if any(isinstance(v, str) and v.strip() for v in split_paths.values()):
        if not isinstance(declared_fingerprints, dict):
            add_issue(
                failures,
                "missing_data_fingerprints",
                "model_selection_report must include data_fingerprints for split provenance.",
                {"required_splits": [k for k, v in split_paths.items() if isinstance(v, str) and v.strip()]},
            )
        else:
            for split_name, raw_path in split_paths.items():
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                split_path = Path(raw_path).expanduser().resolve()
                if not split_path.exists() or not split_path.is_file():
                    add_issue(
                        failures,
                        "split_file_missing_for_fingerprint_verification",
                        "Split file for fingerprint verification is missing or invalid.",
                        {"split": split_name, "path": str(split_path)},
                    )
                    continue
                block = declared_fingerprints.get(split_name)
                if not isinstance(block, dict):
                    add_issue(
                        failures,
                        "missing_data_fingerprint_split",
                        "Missing fingerprint entry for split.",
                        {"split": split_name},
                    )
                    continue
                declared_sha = str(block.get("sha256", "")).strip().lower()
                declared_rows = finite_int(block.get("row_count"))
                if not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
                    add_issue(
                        failures,
                        "invalid_data_fingerprint",
                        "Fingerprint sha256 must be 64-char lowercase hex.",
                        {"split": split_name, "sha256": block.get("sha256")},
                    )
                    continue
                actual_sha = sha256_file(split_path).lower()
                actual_rows = csv_row_count(split_path)
                fingerprint_summary[split_name] = {
                    "path": str(split_path),
                    "declared_sha256": declared_sha,
                    "actual_sha256": actual_sha,
                    "declared_row_count": declared_rows,
                    "actual_row_count": actual_rows,
                }
                if declared_sha != actual_sha:
                    add_issue(
                        failures,
                        "data_fingerprint_mismatch",
                        "Split fingerprint sha256 does not match report.",
                        {"split": split_name, "declared_sha256": declared_sha, "actual_sha256": actual_sha},
                    )
                if declared_rows is None or int(declared_rows) != int(actual_rows):
                    add_issue(
                        failures,
                        "data_fingerprint_row_count_mismatch",
                        "Split row_count does not match report fingerprint block.",
                        {
                            "split": split_name,
                            "declared_row_count": declared_rows,
                            "actual_row_count": actual_rows,
                        },
                    )

    replay_summary: Dict[str, Any] = {}
    if valid_candidates:
        best = max(valid_candidates, key=lambda item: float(item["mean"]))
        best_mean = float(best["mean"])
        best_se = float(best["std"]) / math.sqrt(float(best["n_folds"]))
        one_se_threshold = best_mean - best_se
        eligible = [row for row in valid_candidates if float(row["mean"]) >= one_se_threshold - 1e-12]
        expected = sorted(
            eligible,
            key=lambda item: (
                int(item["complexity_rank"]),
                -float(item["mean"]),
                str(item["model_id"]),
            ),
        )[0]
        expected_model_id = str(expected["model_id"])

        if selected_model_id and selected_model_id != expected_model_id:
            add_issue(
                failures,
                "selection_not_reproducible",
                "Selected model is inconsistent with one-SE + simplicity replay.",
                {
                    "selected_model_id": selected_model_id,
                    "expected_model_id": expected_model_id,
                    "best_mean": best_mean,
                    "best_se": best_se,
                    "one_se_threshold": one_se_threshold,
                    "eligible_model_ids": [str(x["model_id"]) for x in eligible],
                },
            )

        replay_summary = {
            "best_model_id": str(best["model_id"]),
            "best_mean": best_mean,
            "best_se": best_se,
            "one_se_threshold": one_se_threshold,
            "eligible_model_ids": [str(x["model_id"]) for x in eligible],
            "replayed_selected_model_id": expected_model_id,
        }

    summary = {
        "model_selection_report": str(Path(args.model_selection_report).expanduser().resolve()),
        "tuning_spec": str(Path(args.tuning_spec).expanduser().resolve()),
        "candidate_count": len(candidates),
        "selected_model_id": selected_model_id or None,
        "selection_policy": {
            "primary_metric": primary_metric or None,
            "selection_data": selection_data or None,
            "test_used_for_model_selection": test_used_for_model_selection,
            "one_se_rule": one_se_rule,
        },
        "replay": replay_summary,
        "fingerprint_verification": fingerprint_summary,
    }
    return finish(args, failures, warnings, summary)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
    }

    if args.report:
        from _gate_utils import write_json as _write_report
        _write_report(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
