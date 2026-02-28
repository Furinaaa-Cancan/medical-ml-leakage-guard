#!/usr/bin/env python3
"""
Fail-closed distribution/generalization gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from _gate_utils import add_issue, load_json_from_str as load_json, to_float


SUPPORTED_EXTERNAL_TYPES = {"cross_period", "cross_institution"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate train/valid/test/external distribution drift and separability.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", required=True, help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--evaluation-report", help="Optional evaluation_report JSON to scope features to deployed set.")
    parser.add_argument("--external-validation-report", required=True, help="Path to external_validation_report.json.")
    parser.add_argument("--feature-group-spec", required=True, help="Path to feature_group_spec JSON.")
    parser.add_argument("--target-col", default="y", help="Binary target column.")
    parser.add_argument("--ignore-cols", default="patient_id,event_time", help="Comma-separated columns excluded from features.")
    parser.add_argument("--performance-policy", help="Optional performance_policy JSON.")
    parser.add_argument("--distribution-report", help="Optional distribution_report artifact to schema-check.")
    parser.add_argument("--report", help="Optional output gate report.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def parse_selected_features_from_evaluation_report(path: Optional[str]) -> List[str]:
    if not path:
        return []
    try:
        payload = load_json(path)
    except Exception as exc:
        print(f"[WARN] failed to load evaluation report for feature list: {exc}", file=sys.stderr)
        return []
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return []
    features = metadata.get("features")
    if not isinstance(features, list):
        return []
    out: List[str] = []
    for item in features:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if token:
            out.append(token)
    seen: set[str] = set()
    deduped: List[str] = []
    for name in out:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def parse_ignore_cols(raw: str, target_col: str) -> List[str]:
    out = {target_col}
    for token in str(raw).split(","):
        key = token.strip()
        if key:
            out.add(key)
    return sorted(out)


def load_split(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Split is empty: {p}")
    return df


def normalize_binary(series: pd.Series) -> Optional[np.ndarray]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(arr)):
        return None
    if not np.all(np.isin(arr, [0.0, 1.0])):
        return None
    return arr.astype(int)


def parse_thresholds(policy: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = {
        "split_classifier_auc_fail": 0.75,
        "split_classifier_auc_warn": 0.68,
        "top_feature_jsd_fail": 0.30,
        "top_feature_jsd_warn": 0.20,
        "high_shift_feature_fraction_fail": 0.30,
        "high_shift_feature_fraction_warn": 0.20,
        "missing_ratio_delta_fail": 0.20,
        "missing_ratio_delta_warn": 0.12,
        "prevalence_delta_fail": 0.20,
        "prevalence_delta_warn": 0.10,
    }
    if not isinstance(policy, dict):
        return out
    block = policy.get("distribution_thresholds_v2")
    if not isinstance(block, dict):
        return out
    for key in out:
        value = to_float(block.get(key))
        if value is None or value < 0.0:
            continue
        out[key] = float(value)
    return out


def build_external_paths(
    payload: Dict[str, Any],
    failures: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "external_validation_report must include non-empty cohorts list.",
            {},
        )
        return []
    out: List[Dict[str, Any]] = []
    for idx, cohort in enumerate(cohorts):
        if not isinstance(cohort, dict):
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "external_validation_report cohort entry must be object.",
                {"index": idx},
            )
            continue
        cohort_id = str(cohort.get("cohort_id", "")).strip()
        cohort_type = str(cohort.get("cohort_type", "")).strip().lower()
        fingerprint = cohort.get("data_fingerprint")
        path_value = fingerprint.get("path") if isinstance(fingerprint, dict) else None
        if not cohort_id:
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "external_validation_report cohort entry missing cohort_id.",
                {"index": idx},
            )
            continue
        if cohort_type not in SUPPORTED_EXTERNAL_TYPES:
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "external_validation_report cohort_type must be cross_period/cross_institution.",
                {"cohort_id": cohort_id, "cohort_type": cohort_type},
            )
            continue
        if not isinstance(path_value, str) or not path_value.strip():
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "external_validation_report cohort must include data_fingerprint.path.",
                {"cohort_id": cohort_id},
            )
            continue
        p = Path(path_value).expanduser().resolve()
        if not p.exists() or not p.is_file():
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "external cohort data path missing in data_fingerprint.path.",
                {"cohort_id": cohort_id, "path": str(p)},
            )
            continue
        out.append({"split_name": f"external:{cohort_id}", "cohort_id": cohort_id, "cohort_type": cohort_type, "path": str(p)})
    return out


def is_numeric_feature(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna().sum()
    return finite >= max(10, int(0.5 * len(series)))


def js_divergence_from_probs(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-12
    a = np.asarray(a, dtype=float) + eps
    b = np.asarray(b, dtype=float) + eps
    a = a / float(np.sum(a))
    b = b / float(np.sum(b))
    m = 0.5 * (a + b)
    kl_am = float(np.sum(a * np.log(a / m)))
    kl_bm = float(np.sum(b * np.log(b / m)))
    js = 0.5 * (kl_am + kl_bm)
    return float(js / math.log(2.0))


def feature_jsd(train: pd.Series, other: pd.Series) -> Optional[float]:
    train_missing = train.isna()
    other_missing = other.isna()
    train_non_missing = train[~train_missing]
    other_non_missing = other[~other_missing]
    if train_non_missing.empty or other_non_missing.empty:
        return None

    if is_numeric_feature(train) and is_numeric_feature(other):
        train_num = pd.to_numeric(train_non_missing, errors="coerce").dropna().to_numpy(dtype=float)
        other_num = pd.to_numeric(other_non_missing, errors="coerce").dropna().to_numpy(dtype=float)
        if train_num.size < 10 or other_num.size < 10:
            return None
        quantiles = np.linspace(0.0, 1.0, 11)
        bins = np.quantile(train_num, quantiles)
        bins = np.unique(bins)
        if bins.size < 3:
            return None
        train_hist, _ = np.histogram(train_num, bins=bins)
        other_hist, _ = np.histogram(other_num, bins=bins)
        return js_divergence_from_probs(train_hist, other_hist)

    train_cat = train_non_missing.astype(str)
    other_cat = other_non_missing.astype(str)
    keys = sorted(set(train_cat.unique().tolist()) | set(other_cat.unique().tolist()))
    if not keys:
        return None
    train_counts = train_cat.value_counts(normalize=True)
    other_counts = other_cat.value_counts(normalize=True)
    a = np.asarray([float(train_counts.get(k, 0.0)) for k in keys], dtype=float)
    b = np.asarray([float(other_counts.get(k, 0.0)) for k in keys], dtype=float)
    return js_divergence_from_probs(a, b)


def build_split_classifier_auc(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    features: Sequence[str],
    max_rows: int = 8000,
) -> Optional[float]:
    if not features:
        return None
    n = int(min(max_rows // 2, len(train_df), len(other_df)))
    if n < 20:
        return None
    tr = train_df.sample(n=n, random_state=7) if len(train_df) > n else train_df.copy()
    ot = other_df.sample(n=n, random_state=11) if len(other_df) > n else other_df.copy()
    y = np.concatenate([np.zeros(len(tr), dtype=int), np.ones(len(ot), dtype=int)])
    combined = pd.concat([tr[list(features)], ot[list(features)]], axis=0, ignore_index=True)
    processed = pd.DataFrame(index=combined.index)
    for feature in features:
        raw = combined[feature]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_fraction = float(numeric.notna().mean()) if len(numeric) else 0.0
        if numeric_fraction >= 0.60 and int(numeric.notna().sum()) >= 20:
            filled = numeric.fillna(float(numeric.median(skipna=True)) if numeric.notna().any() else 0.0)
            unique_count = int(filled.nunique(dropna=True))
            bin_count = int(min(10, max(2, unique_count)))
            try:
                processed[feature] = pd.qcut(filled, q=bin_count, duplicates="drop").astype(str)
            except Exception:
                processed[feature] = filled.round(3).astype(str)
        else:
            processed[feature] = raw.astype(str).fillna("__missing__")
    x = pd.get_dummies(processed, dummy_na=False)
    if x.shape[1] == 0:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        x.to_numpy(dtype=float),
        y,
        test_size=0.30,
        random_state=19,
        stratify=y,
    )
    model = LogisticRegression(max_iter=2000, solver="liblinear", C=0.5, class_weight=None)
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, y_score))


def build_missingness_classifier_auc(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    features: Sequence[str],
    max_rows: int = 8000,
) -> Optional[float]:
    if not features:
        return None
    n = int(min(max_rows // 2, len(train_df), len(other_df)))
    if n < 20:
        return None
    tr = train_df.sample(n=n, random_state=13) if len(train_df) > n else train_df.copy()
    ot = other_df.sample(n=n, random_state=17) if len(other_df) > n else other_df.copy()
    y = np.concatenate([np.zeros(len(tr), dtype=int), np.ones(len(ot), dtype=int)])
    tr_miss = tr[list(features)].isna().astype(int)
    ot_miss = ot[list(features)].isna().astype(int)
    x = pd.concat([tr_miss, ot_miss], axis=0, ignore_index=True).to_numpy(dtype=float)
    if x.shape[1] == 0:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=23,
        stratify=y,
    )
    model = LogisticRegression(max_iter=2000, solver="liblinear", C=0.5, class_weight=None)
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(y_test, y_score))


def safe_prevalence(df: pd.DataFrame, target_col: str) -> Optional[float]:
    if target_col not in df.columns:
        return None
    y = normalize_binary(df[target_col])
    if y is None or y.size == 0:
        return None
    return float(np.mean(y))


def group_drift_summary(feature_rows: List[Dict[str, Any]], groups: Dict[str, List[str]]) -> Dict[str, Any]:
    by_feature: Dict[str, Dict[str, Any]] = {str(row["feature"]): row for row in feature_rows if isinstance(row, dict)}
    out: Dict[str, Any] = {}
    for group_name, group_features in groups.items():
        jsd_values: List[float] = []
        missing_values: List[float] = []
        covered = 0
        for feature in group_features:
            row = by_feature.get(feature)
            if not isinstance(row, dict):
                continue
            covered += 1
            jsd = to_float(row.get("jsd"))
            missing_delta = to_float(row.get("missing_ratio_delta"))
            if jsd is not None:
                jsd_values.append(float(jsd))
            if missing_delta is not None:
                missing_values.append(float(missing_delta))
        out[group_name] = {
            "n_features_declared": len(group_features),
            "n_features_covered": covered,
            "mean_jsd": float(np.mean(jsd_values)) if jsd_values else None,
            "max_jsd": float(np.max(jsd_values)) if jsd_values else None,
            "mean_missing_ratio_delta": float(np.mean(missing_values)) if missing_values else None,
            "max_missing_ratio_delta": float(np.max(missing_values)) if missing_values else None,
        }
    return out


def validate_distribution_report_artifact(path: str, failures: List[Dict[str, Any]]) -> None:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "distribution_report path does not exist.",
            {"path": str(p)},
        )
        return
    try:
        payload = load_json(str(p))
    except Exception as exc:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "Unable to parse distribution_report JSON.",
            {"path": str(p), "error": str(exc)},
        )
        return
    if not isinstance(payload.get("schema_version"), str):
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "distribution_report must include schema_version string.",
            {"path": str(p)},
        )


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if args.distribution_report:
        validate_distribution_report_artifact(args.distribution_report, failures)
        if failures:
            return finish(args, failures, warnings, {})

    _split_dfs: Dict[str, Any] = {}
    try:
        for _sn, _sp in (("train", args.train), ("valid", args.valid), ("test", args.test)):
            _split_dfs[_sn] = load_split(_sp)
    except Exception as exc:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            f"Unable to load '{_sn}' split CSV.",
            {"error": str(exc), "path": str(_sp)},
        )
        return finish(args, failures, warnings, {})

    train_df = _split_dfs["train"]
    valid_df = _split_dfs["valid"]
    test_df = _split_dfs["test"]

    try:
        external_payload = load_json(args.external_validation_report)
    except Exception as exc:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "Unable to parse external_validation_report JSON.",
            {"path": str(Path(args.external_validation_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        feature_group_spec = load_json(args.feature_group_spec)
    except Exception as exc:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "Unable to parse feature_group_spec JSON.",
            {"path": str(Path(args.feature_group_spec).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    groups_raw = feature_group_spec.get("groups")
    if not isinstance(groups_raw, dict) or not groups_raw:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "feature_group_spec must include non-empty groups object.",
            {},
        )
        return finish(args, failures, warnings, {})
    groups: Dict[str, List[str]] = {}
    for key, values in groups_raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(values, list):
            continue
        clean = [str(x).strip() for x in values if isinstance(x, str) and str(x).strip()]
        if clean:
            groups[key.strip()] = clean
    if not groups:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "feature_group_spec groups has no usable features.",
            {},
        )
        return finish(args, failures, warnings, {})

    policy: Optional[Dict[str, Any]] = None
    if args.performance_policy:
        try:
            policy = load_json(args.performance_policy)
        except Exception as exc:
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "Unable to parse performance_policy JSON.",
                {"path": str(Path(args.performance_policy).expanduser()), "error": str(exc)},
            )
            return finish(args, failures, warnings, {})
    thresholds = parse_thresholds(policy)

    external_refs = build_external_paths(external_payload, failures)
    if failures:
        return finish(args, failures, warnings, {"thresholds": thresholds})

    ignore_cols = set(parse_ignore_cols(args.ignore_cols, args.target_col))
    selected_features = parse_selected_features_from_evaluation_report(args.evaluation_report)
    if selected_features:
        selected_set = set(selected_features)
        feature_candidates = [c for c in train_df.columns if c not in ignore_cols and c in selected_set]
    else:
        feature_candidates = [c for c in train_df.columns if c not in ignore_cols]
    if not feature_candidates:
        add_issue(
            failures,
            "distribution_report_schema_invalid",
            "No feature columns available after ignore-cols exclusion.",
            {
                "ignore_cols": sorted(ignore_cols),
                "selected_feature_count": int(len(selected_features)),
                "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve())
                if args.evaluation_report
                else None,
            },
        )
        return finish(args, failures, warnings, {"thresholds": thresholds})

    datasets: List[Tuple[str, str, pd.DataFrame]] = [
        ("valid", "internal", valid_df),
        ("test", "internal", test_df),
    ]
    for entry in external_refs:
        ext_df = load_split(entry["path"])
        datasets.append((entry["split_name"], entry["cohort_type"], ext_df))

    matrix_rows: List[Dict[str, Any]] = []
    external_drop_map: Dict[str, Dict[str, float]] = {}
    cohorts = external_payload.get("cohorts")
    if isinstance(cohorts, list):
        for row in cohorts:
            if not isinstance(row, dict):
                continue
            cohort_id = str(row.get("cohort_id", "")).strip()
            gap = row.get("transport_gap")
            if cohort_id and isinstance(gap, dict):
                external_drop_map[f"external:{cohort_id}"] = {
                    "pr_auc_drop": float(to_float(gap.get("pr_auc_drop_from_internal_test")) or 0.0),
                    "f2_beta_drop": float(to_float(gap.get("f2_beta_drop_from_internal_test")) or 0.0),
                    "brier_increase": float(to_float(gap.get("brier_increase_from_internal_test")) or 0.0),
                }

    for split_name, split_kind, other_df in datasets:
        common_features = [c for c in feature_candidates if c in other_df.columns]
        if not common_features:
            add_issue(
                failures,
                "distribution_report_schema_invalid",
                "No overlapping features between train and comparison split.",
                {"split": split_name},
            )
            continue

        feature_rows: List[Dict[str, Any]] = []
        jsd_values: List[float] = []
        high_shift_features = 0
        max_missing_delta = 0.0
        for feature in common_features:
            tr_col = train_df[feature]
            ot_col = other_df[feature]
            jsd = feature_jsd(tr_col, ot_col)
            train_missing = float(tr_col.isna().mean())
            other_missing = float(ot_col.isna().mean())
            missing_delta = abs(other_missing - train_missing)
            if jsd is not None and math.isfinite(jsd):
                jsd_values.append(float(jsd))
                if float(jsd) >= float(thresholds["top_feature_jsd_warn"]):
                    high_shift_features += 1
            max_missing_delta = max(max_missing_delta, float(missing_delta))
            feature_rows.append(
                {
                    "feature": feature,
                    "jsd": float(jsd) if jsd is not None and math.isfinite(jsd) else None,
                    "missing_ratio_delta": float(missing_delta),
                }
            )

        split_auc = build_split_classifier_auc(train_df, other_df, common_features)
        missingness_auc = build_missingness_classifier_auc(train_df, other_df, common_features)
        train_prev = safe_prevalence(train_df, args.target_col)
        other_prev = safe_prevalence(other_df, args.target_col)
        prevalence_delta = abs(other_prev - train_prev) if train_prev is not None and other_prev is not None else None
        top_jsd = float(np.max(jsd_values)) if jsd_values else 0.0
        high_shift_fraction = float(high_shift_features) / float(len(common_features)) if common_features else 0.0

        group_summary = group_drift_summary(feature_rows, groups)
        transport = external_drop_map.get(split_name)

        row = {
            "split": split_name,
            "split_kind": split_kind,
            "feature_count": int(len(common_features)),
            "top_feature_jsd": top_jsd,
            "high_shift_feature_fraction": high_shift_fraction,
            "max_missing_ratio_delta": float(max_missing_delta),
            "prevalence_delta": float(prevalence_delta) if prevalence_delta is not None else None,
            "split_classifier_auc": float(split_auc) if split_auc is not None else None,
            "missingness_pattern_auc": float(missingness_auc) if missingness_auc is not None else None,
            "group_drift_summary": group_summary,
            "transport_gap": transport,
            "top_shift_features": sorted(
                [r for r in feature_rows if isinstance(r.get("jsd"), (int, float))],
                key=lambda x: float(x.get("jsd", 0.0)),
                reverse=True,
            )[:15],
        }
        matrix_rows.append(row)

        if top_jsd > float(thresholds["top_feature_jsd_fail"]) or high_shift_fraction > float(
            thresholds["high_shift_feature_fraction_fail"]
        ) or max_missing_delta > float(thresholds["missing_ratio_delta_fail"]) or (
            prevalence_delta is not None and prevalence_delta > float(thresholds["prevalence_delta_fail"])
        ):
            add_issue(
                failures,
                "distribution_shift_exceeds_threshold",
                "Distribution shift exceeds publication-grade threshold.",
                {
                    "split": split_name,
                    "top_feature_jsd": top_jsd,
                    "high_shift_feature_fraction": high_shift_fraction,
                    "max_missing_ratio_delta": float(max_missing_delta),
                    "prevalence_delta": float(prevalence_delta) if prevalence_delta is not None else None,
                    "thresholds": {
                        "top_feature_jsd_fail": thresholds["top_feature_jsd_fail"],
                        "high_shift_feature_fraction_fail": thresholds["high_shift_feature_fraction_fail"],
                        "missing_ratio_delta_fail": thresholds["missing_ratio_delta_fail"],
                        "prevalence_delta_fail": thresholds["prevalence_delta_fail"],
                    },
                },
            )
        else:
            if top_jsd > float(thresholds["top_feature_jsd_warn"]) or high_shift_fraction > float(
                thresholds["high_shift_feature_fraction_warn"]
            ) or max_missing_delta > float(thresholds["missing_ratio_delta_warn"]) or (
                prevalence_delta is not None and prevalence_delta > float(thresholds["prevalence_delta_warn"])
            ):
                add_issue(
                    warnings,
                    "distribution_shift_warning",
                    "Distribution shift exceeds warning threshold.",
                    {"split": split_name},
                )

        if split_auc is not None and split_auc > float(thresholds["split_classifier_auc_fail"]):
            add_issue(
                failures,
                "split_separability_exceeds_threshold",
                "Split classifier AUC indicates high distribution separability.",
                {
                    "split": split_name,
                    "split_classifier_auc": float(split_auc),
                    "split_classifier_auc_fail": float(thresholds["split_classifier_auc_fail"]),
                },
            )
        elif split_auc is not None and split_auc > float(thresholds["split_classifier_auc_warn"]):
            add_issue(
                warnings,
                "split_separability_warning",
                "Split classifier AUC exceeds warning threshold.",
                {
                    "split": split_name,
                    "split_classifier_auc": float(split_auc),
                    "split_classifier_auc_warn": float(thresholds["split_classifier_auc_warn"]),
                },
            )

        if missingness_auc is not None and missingness_auc > float(thresholds["split_classifier_auc_fail"]):
            add_issue(
                failures,
                "missingness_pattern_shift_exceeds_threshold",
                "Missingness-pattern classifier AUC exceeds fail threshold.",
                {
                    "split": split_name,
                    "missingness_pattern_auc": float(missingness_auc),
                    "fail_threshold": float(thresholds["split_classifier_auc_fail"]),
                },
            )
        elif missingness_auc is not None and missingness_auc > float(thresholds["split_classifier_auc_warn"]):
            add_issue(
                warnings,
                "missingness_pattern_shift_warning",
                "Missingness-pattern classifier AUC exceeds warning threshold.",
                {
                    "split": split_name,
                    "missingness_pattern_auc": float(missingness_auc),
                    "warn_threshold": float(thresholds["split_classifier_auc_warn"]),
                },
            )

    summary = {
        "schema_version": "v4.0",
        "thresholds": thresholds,
        "feature_scope": {
            "mode": "evaluation_report_selected_features" if selected_features else "all_non_ignored_features",
            "selected_feature_count": int(len(selected_features)),
            "candidate_feature_count": int(len(feature_candidates)),
            "evaluation_report": str(Path(args.evaluation_report).expanduser().resolve())
            if args.evaluation_report
            else None,
        },
        "distribution_matrix": matrix_rows,
        "train_path": str(Path(args.train).expanduser().resolve()),
        "valid_path": str(Path(args.valid).expanduser().resolve()),
        "test_path": str(Path(args.test).expanduser().resolve()),
        "external_validation_report": str(Path(args.external_validation_report).expanduser().resolve()),
        "feature_group_spec": str(Path(args.feature_group_spec).expanduser().resolve()),
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
    raise SystemExit(main())
