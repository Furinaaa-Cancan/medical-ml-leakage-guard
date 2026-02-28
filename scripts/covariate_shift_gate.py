#!/usr/bin/env python3
"""
Fail-closed covariate-shift gate for publication-grade medical prediction workflows.

Goal:
- Detect split separability risk and unstable feature distributions that can inflate
  apparent performance or hide split-assignment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from _gate_utils import add_issue


MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a", "?"}
JSD_PSEUDO_COUNT = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate covariate shift between train and holdout splits.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--target-col", default="y", help="Target/label column name.")
    parser.add_argument(
        "--ignore-cols",
        default="",
        help="Comma-separated columns to ignore from feature shift audit (for example id/time columns).",
    )
    parser.add_argument(
        "--numeric-bins",
        type=int,
        default=20,
        help="Number of bins for numeric feature distributions.",
    )
    parser.add_argument(
        "--categorical-buckets",
        type=int,
        default=64,
        help="Hash bucket count for categorical feature distributions.",
    )
    parser.add_argument(
        "--numeric-detection-threshold",
        type=float,
        default=0.98,
        help="Required numeric-parse ratio to treat feature as numeric.",
    )
    parser.add_argument(
        "--high-shift-jsd",
        type=float,
        default=0.12,
        help="JSD threshold defining a high-shift feature.",
    )
    parser.add_argument(
        "--max-top-feature-jsd",
        type=float,
        default=0.35,
        help="Maximum allowed top-feature JSD before hard failure.",
    )
    parser.add_argument(
        "--max-mean-top10-jsd",
        type=float,
        default=0.18,
        help="Maximum allowed mean JSD across top-10 shifted features before hard failure.",
    )
    parser.add_argument(
        "--max-high-shift-feature-fraction",
        type=float,
        default=0.30,
        help="Maximum allowed fraction of high-shift features before hard failure.",
    )
    parser.add_argument(
        "--max-missing-ratio-delta",
        type=float,
        default=0.20,
        help="Maximum allowed missing-ratio delta before warning.",
    )
    parser.add_argument(
        "--warn-prevalence-delta",
        type=float,
        default=0.10,
        help="Warning threshold for split prevalence delta.",
    )
    parser.add_argument(
        "--max-prevalence-delta",
        type=float,
        default=0.25,
        help="Hard-failure threshold for split prevalence delta.",
    )
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=0,
        help="Optional row cap per split for auditing speed (0 means full file).",
    )
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def is_missing(raw: str) -> bool:
    return raw.strip().lower() in MISSING_TOKENS


def parse_ignore_cols(raw: str, target_col: str) -> Set[str]:
    out: Set[str] = {target_col}
    for token in raw.split(","):
        value = token.strip()
        if value:
            out.add(value)
    return out


def parse_binary_label(raw: str) -> Optional[int]:
    s = raw.strip()
    if not s:
        return None
    if s in {"0", "0.0"}:
        return 0
    if s in {"1", "1.0"}:
        return 1
    try:
        parsed = float(s)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    if parsed == 0.0:
        return 0
    if parsed == 1.0:
        return 1
    return None


def try_parse_float(raw: str) -> Optional[float]:
    s = raw.strip()
    if not s:
        return None
    try:
        parsed = float(s)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def validate_threshold_args(args: argparse.Namespace, failures: List[Dict[str, Any]]) -> None:
    ratio_fields = [
        "numeric_detection_threshold",
        "high_shift_jsd",
        "max_top_feature_jsd",
        "max_mean_top10_jsd",
        "max_high_shift_feature_fraction",
        "max_missing_ratio_delta",
        "warn_prevalence_delta",
        "max_prevalence_delta",
    ]
    for key in ratio_fields:
        value = float(getattr(args, key))
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            add_issue(
                failures,
                "invalid_threshold_range",
                "Threshold must be finite and within [0, 1].",
                {"field": key, "value": value},
            )

    if args.warn_prevalence_delta > args.max_prevalence_delta:
        add_issue(
            failures,
            "invalid_prevalence_threshold_order",
            "warn-prevalence-delta must be <= max-prevalence-delta.",
            {
                "warn_prevalence_delta": args.warn_prevalence_delta,
                "max_prevalence_delta": args.max_prevalence_delta,
            },
        )

    if args.numeric_bins < 2:
        add_issue(
            failures,
            "invalid_numeric_bins",
            "numeric-bins must be >= 2.",
            {"numeric_bins": args.numeric_bins},
        )
    if args.categorical_buckets < 2:
        add_issue(
            failures,
            "invalid_categorical_buckets",
            "categorical-buckets must be >= 2.",
            {"categorical_buckets": args.categorical_buckets},
        )
    if args.max_rows_per_split < 0:
        add_issue(
            failures,
            "invalid_max_rows_per_split",
            "max-rows-per-split must be >= 0.",
            {"max_rows_per_split": args.max_rows_per_split},
        )


def js_divergence(counts_a: List[int], counts_b: List[int]) -> Optional[float]:
    if len(counts_a) != len(counts_b) or not counts_a:
        return None

    total_a_raw = float(sum(counts_a))
    total_b_raw = float(sum(counts_b))
    if total_a_raw <= 0.0 or total_b_raw <= 0.0:
        return None

    n_bins = float(len(counts_a))
    total_a = total_a_raw + JSD_PSEUDO_COUNT * n_bins
    total_b = total_b_raw + JSD_PSEUDO_COUNT * n_bins

    log2 = math.log(2.0)
    kl_a = 0.0
    kl_b = 0.0
    for ca, cb in zip(counts_a, counts_b):
        pa = (float(ca) + JSD_PSEUDO_COUNT) / total_a
        pb = (float(cb) + JSD_PSEUDO_COUNT) / total_b
        m = 0.5 * (pa + pb)
        if pa > 0.0:
            kl_a += pa * math.log(pa / m)
        if pb > 0.0:
            kl_b += pb * math.log(pb / m)
    jsd = 0.5 * (kl_a + kl_b)
    return jsd / log2


def assign_numeric_bin(value: float, low: float, high: float, n_bins: int) -> int:
    if high <= low:
        return 0
    if value <= low:
        return 0
    if value >= high:
        return n_bins - 1
    ratio = (value - low) / (high - low)
    idx = int(ratio * float(n_bins))
    if idx < 0:
        return 0
    if idx >= n_bins:
        return n_bins - 1
    return idx


def categorical_bucket(value: str, bucket_count: int) -> int:
    digest = hashlib.sha256(value.strip().lower().encode("utf-8")).digest()
    as_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return as_int % bucket_count


def discover_feature_types(
    train_path: str,
    target_col: str,
    ignore_cols: Set[str],
    numeric_detection_threshold: float,
    max_rows: int,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train file not found: {train_path}")

    with open(train_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("train: missing CSV header.")
        headers = [(h or "").strip() for h in reader.fieldnames]
        if target_col not in headers:
            raise ValueError(f"train: missing target_col '{target_col}'.")

        features = [col for col in headers if col not in ignore_cols]
        if not features:
            raise ValueError("No usable feature columns remain after ignored-column exclusions.")

        meta: Dict[str, Dict[str, Any]] = {
            col: {"non_missing": 0, "numeric_parseable": 0, "numeric_min": None, "numeric_max": None}
            for col in features
        }

        row_count = 0
        sample_truncated = False
        for row in reader:
            row_count += 1
            if max_rows > 0 and row_count > max_rows:
                sample_truncated = True
                break
            for col in features:
                raw = row.get(col)
                value = "" if raw is None else str(raw)
                if is_missing(value):
                    continue
                m = meta[col]
                m["non_missing"] += 1
                parsed = try_parse_float(value)
                if parsed is None:
                    continue
                m["numeric_parseable"] += 1
                cur_min = m["numeric_min"]
                cur_max = m["numeric_max"]
                m["numeric_min"] = parsed if cur_min is None else min(float(cur_min), parsed)
                m["numeric_max"] = parsed if cur_max is None else max(float(cur_max), parsed)

    types: Dict[str, Dict[str, Any]] = {}
    for col in features:
        col_meta = meta[col]
        non_missing = int(col_meta["non_missing"])
        numeric_parseable = int(col_meta["numeric_parseable"])
        ratio = 0.0 if non_missing <= 0 else (numeric_parseable / float(non_missing))
        col_type = "numeric" if non_missing > 0 and ratio >= numeric_detection_threshold else "categorical"
        types[col] = {
            "type": col_type,
            "non_missing": non_missing,
            "numeric_parseable": numeric_parseable,
            "numeric_ratio": ratio,
            "numeric_min": col_meta["numeric_min"],
            "numeric_max": col_meta["numeric_max"],
        }

    discovery = {
        "path": str(Path(train_path).expanduser().resolve()),
        "row_count_profiled": row_count if not sample_truncated else max_rows,
        "sample_truncated": sample_truncated,
        "feature_count": len(features),
        "numeric_feature_count": sum(1 for f in types.values() if f["type"] == "numeric"),
        "categorical_feature_count": sum(1 for f in types.values() if f["type"] == "categorical"),
    }
    return features, types, discovery


def init_feature_counts(
    features: List[str],
    feature_types: Dict[str, Dict[str, Any]],
    numeric_bins: int,
    categorical_buckets: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for col in features:
        col_type = str(feature_types[col]["type"])
        bucket_count = numeric_bins if col_type == "numeric" else categorical_buckets
        out[col] = {
            "type": col_type,
            "counts": [0] * bucket_count,
            "missing_count": 0,
            "non_missing_count": 0,
            "invalid_numeric_count": 0,
        }
    return out


def profile_split(
    split_name: str,
    path: str,
    features: List[str],
    feature_types: Dict[str, Dict[str, Any]],
    numeric_bins: int,
    categorical_buckets: int,
    target_col: str,
    max_rows: int,
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{split_name}: file not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{split_name}: missing CSV header.")
        headers = [(h or "").strip() for h in reader.fieldnames]
        if target_col not in headers:
            raise ValueError(f"{split_name}: missing target_col '{target_col}'.")

        missing_features = [col for col in features if col not in headers]
        if missing_features:
            raise ValueError(f"{split_name}: missing required feature columns: {missing_features[:10]}")

        feature_counts = init_feature_counts(features, feature_types, numeric_bins, categorical_buckets)

        row_count = 0
        sample_truncated = False
        positive = 0
        negative = 0
        invalid_label_rows = 0

        for row in reader:
            row_count += 1
            if max_rows > 0 and row_count > max_rows:
                sample_truncated = True
                break

            label = parse_binary_label(str(row.get(target_col) or ""))
            if label is None:
                invalid_label_rows += 1
            elif label == 1:
                positive += 1
            else:
                negative += 1

            for col in features:
                stats = feature_counts[col]
                raw = row.get(col)
                value = "" if raw is None else str(raw)
                if is_missing(value):
                    stats["missing_count"] += 1
                    continue
                stats["non_missing_count"] += 1

                if stats["type"] == "numeric":
                    parsed = try_parse_float(value)
                    if parsed is None:
                        stats["invalid_numeric_count"] += 1
                        stats["missing_count"] += 1
                        continue
                    low = feature_types[col]["numeric_min"]
                    high = feature_types[col]["numeric_max"]
                    low_num = float(parsed if low is None else low)
                    high_num = float(parsed if high is None else high)
                    idx = assign_numeric_bin(parsed, low_num, high_num, numeric_bins)
                    stats["counts"][idx] += 1
                else:
                    idx = categorical_bucket(value, categorical_buckets)
                    stats["counts"][idx] += 1

    denom = positive + negative
    prevalence = None if denom <= 0 else (positive / float(denom))
    return {
        "path": str(Path(path).expanduser().resolve()),
        "row_count": row_count if not sample_truncated else max_rows,
        "sample_truncated": sample_truncated,
        "headers": headers,
        "positive_count": positive,
        "negative_count": negative,
        "invalid_label_rows": invalid_label_rows,
        "prevalence": prevalence,
        "feature_counts": feature_counts,
    }


def safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / float(denominator)


def evaluate_pair(
    train_stats: Dict[str, Any],
    other_stats: Dict[str, Any],
    features: List[str],
    pair_name: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    per_feature: Dict[str, Dict[str, Any]] = {}
    valid_jsd_values: List[float] = []
    max_missing_delta = 0.0

    train_rows = int(train_stats.get("row_count", 0) or 0)
    other_rows = int(other_stats.get("row_count", 0) or 0)

    for col in features:
        train_feature = train_stats["feature_counts"][col]
        other_feature = other_stats["feature_counts"][col]
        jsd = js_divergence(train_feature["counts"], other_feature["counts"])

        train_missing_ratio = safe_ratio(int(train_feature["missing_count"]), train_rows)
        other_missing_ratio = safe_ratio(int(other_feature["missing_count"]), other_rows)
        missing_delta = None
        if train_missing_ratio is not None and other_missing_ratio is not None:
            missing_delta = abs(train_missing_ratio - other_missing_ratio)
            max_missing_delta = max(max_missing_delta, missing_delta)

        per_feature[col] = {
            "jsd": jsd,
            "train_missing_ratio": train_missing_ratio,
            "other_missing_ratio": other_missing_ratio,
            "missing_ratio_delta": missing_delta,
            "other_split": pair_name.split("_vs_")[-1],
        }
        if jsd is not None and math.isfinite(jsd):
            valid_jsd_values.append(jsd)

    pair_summary = {
        "pair": pair_name,
        "feature_count": len(features),
        "features_with_finite_jsd": len(valid_jsd_values),
        "mean_jsd": statistics.fmean(valid_jsd_values) if valid_jsd_values else None,
        "max_jsd": max(valid_jsd_values) if valid_jsd_values else None,
        "max_missing_ratio_delta": max_missing_delta,
    }
    return per_feature, pair_summary


def top_n_feature_summary(
    feature_rows: List[Dict[str, Any]],
    key: str,
    n: int = 25,
) -> List[Dict[str, Any]]:
    sortable = [row for row in feature_rows if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
    sortable.sort(key=lambda item: float(item[key]), reverse=True)
    return sortable[:n]


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    validate_threshold_args(args, failures)
    if failures:
        return finish(args, failures, warnings, {}, [])

    ignore_cols = parse_ignore_cols(args.ignore_cols, args.target_col)

    try:
        features, feature_types, discovery = discover_feature_types(
            train_path=args.train,
            target_col=args.target_col,
            ignore_cols=ignore_cols,
            numeric_detection_threshold=args.numeric_detection_threshold,
            max_rows=args.max_rows_per_split,
        )
    except Exception as exc:
        add_issue(
            failures,
            "feature_type_discovery_failed",
            "Failed while discovering feature types/ranges from train split.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, {}, [])

    split_paths = {"train": args.train, "test": args.test}
    if args.valid:
        split_paths["valid"] = args.valid

    splits: Dict[str, Dict[str, Any]] = {}
    try:
        for split_name, path in split_paths.items():
            splits[split_name] = profile_split(
                split_name=split_name,
                path=path,
                features=features,
                feature_types=feature_types,
                numeric_bins=args.numeric_bins,
                categorical_buckets=args.categorical_buckets,
                target_col=args.target_col,
                max_rows=args.max_rows_per_split,
            )
    except Exception as exc:
        add_issue(
            failures,
            "split_profile_failed",
            "Failed to profile split distributions for covariate-shift audit.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, {"feature_discovery": discovery, "splits": {}}, [])

    for split_name, stats in splits.items():
        if int(stats.get("row_count", 0) or 0) <= 0:
            add_issue(
                failures,
                "empty_split",
                "Split must not be empty.",
                {"split": split_name},
            )
        invalid_label_rows = int(stats.get("invalid_label_rows", 0) or 0)
        if invalid_label_rows > 0:
            add_issue(
                failures,
                "invalid_labels",
                "Split contains non-binary/invalid labels.",
                {"split": split_name, "invalid_label_rows": invalid_label_rows},
            )
        if stats.get("sample_truncated"):
            add_issue(
                warnings,
                "split_profile_truncated",
                "Split profiling used row cap; covariate shift audit is sample-based.",
                {"split": split_name, "max_rows_per_split": args.max_rows_per_split},
            )

    if failures:
        return finish(args, failures, warnings, {"feature_discovery": discovery, "splits": summarize_splits(splits)}, [])

    pairs: List[Tuple[str, str]] = [("train", "test")]
    if "valid" in splits:
        pairs.insert(0, ("train", "valid"))

    pair_summaries: List[Dict[str, Any]] = []
    pair_feature_details: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for left, right in pairs:
        pair_name = f"{left}_vs_{right}"
        per_feature, pair_summary = evaluate_pair(splits[left], splits[right], features, pair_name)
        pair_feature_details[pair_name] = per_feature
        pair_summaries.append(pair_summary)

    feature_rows: List[Dict[str, Any]] = []
    finite_jsd_values: List[float] = []
    high_shift_features: List[str] = []
    max_missing_ratio_delta = 0.0

    for col in features:
        pair_metrics: Dict[str, Dict[str, Any]] = {}
        jsd_values: List[float] = []
        missing_deltas: List[float] = []
        for pair_name in pair_feature_details:
            details = pair_feature_details[pair_name][col]
            pair_metrics[pair_name] = {
                "jsd": details.get("jsd"),
                "missing_ratio_delta": details.get("missing_ratio_delta"),
                "train_missing_ratio": details.get("train_missing_ratio"),
                "other_missing_ratio": details.get("other_missing_ratio"),
            }
            jsd_val = details.get("jsd")
            if isinstance(jsd_val, (int, float)) and math.isfinite(float(jsd_val)):
                jsd_values.append(float(jsd_val))
            miss_val = details.get("missing_ratio_delta")
            if isinstance(miss_val, (int, float)) and math.isfinite(float(miss_val)):
                missing_deltas.append(float(miss_val))

        max_jsd = max(jsd_values) if jsd_values else None
        max_missing = max(missing_deltas) if missing_deltas else None
        if max_jsd is not None:
            finite_jsd_values.append(max_jsd)
            if max_jsd >= args.high_shift_jsd:
                high_shift_features.append(col)
        if max_missing is not None:
            max_missing_ratio_delta = max(max_missing_ratio_delta, max_missing)

        feature_rows.append(
            {
                "feature": col,
                "type": feature_types[col]["type"],
                "numeric_ratio_in_train": feature_types[col]["numeric_ratio"],
                "max_jsd": max_jsd,
                "max_missing_ratio_delta": max_missing,
                "pair_metrics": pair_metrics,
            }
        )

    if not finite_jsd_values:
        add_issue(
            failures,
            "no_finite_shift_metrics",
            "No finite feature shift metrics were computed.",
            {},
        )
        return finish(
            args,
            failures,
            warnings,
            {
                "feature_discovery": discovery,
                "splits": summarize_splits(splits),
                "pairs": pair_summaries,
            },
            feature_rows,
        )

    top_feature_jsd = max(finite_jsd_values)
    sorted_jsd = sorted(finite_jsd_values, reverse=True)
    mean_top10_jsd = statistics.fmean(sorted_jsd[: min(10, len(sorted_jsd))])
    feature_count = len(features)
    high_shift_fraction = len(high_shift_features) / float(feature_count) if feature_count > 0 else 0.0

    if top_feature_jsd > args.max_top_feature_jsd:
        add_issue(
            failures,
            "top_feature_shift_too_high",
            "Top-feature distribution shift exceeds hard threshold.",
            {"top_feature_jsd": top_feature_jsd, "max_top_feature_jsd": args.max_top_feature_jsd},
        )
    if mean_top10_jsd > args.max_mean_top10_jsd:
        add_issue(
            failures,
            "mean_top10_shift_too_high",
            "Mean shift among top features exceeds hard threshold.",
            {"mean_top10_jsd": mean_top10_jsd, "max_mean_top10_jsd": args.max_mean_top10_jsd},
        )
    if high_shift_fraction > args.max_high_shift_feature_fraction:
        add_issue(
            failures,
            "too_many_high_shift_features",
            "High-shift feature fraction exceeds threshold.",
            {
                "high_shift_feature_count": len(high_shift_features),
                "feature_count": feature_count,
                "high_shift_fraction": high_shift_fraction,
                "threshold": args.max_high_shift_feature_fraction,
                "high_shift_jsd": args.high_shift_jsd,
            },
        )
    if max_missing_ratio_delta > args.max_missing_ratio_delta:
        add_issue(
            warnings,
            "missingness_shift_exceeds_threshold",
            "Missingness shift across splits exceeds threshold.",
            {
                "max_missing_ratio_delta": max_missing_ratio_delta,
                "max_missing_ratio_delta_threshold": args.max_missing_ratio_delta,
            },
        )

    train_prev = splits["train"].get("prevalence")
    if isinstance(train_prev, (int, float)) and math.isfinite(float(train_prev)):
        for split_name in ("valid", "test"):
            if split_name not in splits:
                continue
            other_prev = splits[split_name].get("prevalence")
            if not isinstance(other_prev, (int, float)) or not math.isfinite(float(other_prev)):
                continue
            delta = abs(float(train_prev) - float(other_prev))
            if delta > args.max_prevalence_delta:
                add_issue(
                    failures,
                    "prevalence_shift_too_high",
                    "Outcome prevalence shift exceeds hard threshold.",
                    {
                        "reference_split": "train",
                        "other_split": split_name,
                        "train_prevalence": train_prev,
                        "other_prevalence": other_prev,
                        "delta": delta,
                        "max_prevalence_delta": args.max_prevalence_delta,
                    },
                )
            elif delta > args.warn_prevalence_delta:
                add_issue(
                    warnings,
                    "prevalence_shift_warning",
                    "Outcome prevalence shift exceeds warning threshold.",
                    {
                        "reference_split": "train",
                        "other_split": split_name,
                        "train_prevalence": train_prev,
                        "other_prevalence": other_prev,
                        "delta": delta,
                        "warn_prevalence_delta": args.warn_prevalence_delta,
                    },
                )

    summary = {
        "feature_discovery": discovery,
        "splits": summarize_splits(splits),
        "pairs": pair_summaries,
        "aggregates": {
            "feature_count": feature_count,
            "features_with_finite_jsd": len(finite_jsd_values),
            "top_feature_jsd": top_feature_jsd,
            "mean_top10_jsd": mean_top10_jsd,
            "high_shift_feature_count": len(high_shift_features),
            "high_shift_feature_fraction": high_shift_fraction,
            "max_missing_ratio_delta": max_missing_ratio_delta,
        },
        "thresholds": {
            "high_shift_jsd": args.high_shift_jsd,
            "max_top_feature_jsd": args.max_top_feature_jsd,
            "max_mean_top10_jsd": args.max_mean_top10_jsd,
            "max_high_shift_feature_fraction": args.max_high_shift_feature_fraction,
            "max_missing_ratio_delta": args.max_missing_ratio_delta,
            "warn_prevalence_delta": args.warn_prevalence_delta,
            "max_prevalence_delta": args.max_prevalence_delta,
        },
    }
    return finish(args, failures, warnings, summary, feature_rows)


def summarize_splits(splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split_name, stats in splits.items():
        out[split_name] = {
            "path": stats.get("path"),
            "row_count": stats.get("row_count"),
            "sample_truncated": stats.get("sample_truncated"),
            "positive_count": stats.get("positive_count"),
            "negative_count": stats.get("negative_count"),
            "invalid_label_rows": stats.get("invalid_label_rows"),
            "prevalence": stats.get("prevalence"),
        }
    return out


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
    feature_rows: List[Dict[str, Any]],
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
        "top_shift_features": top_n_feature_summary(feature_rows, key="max_jsd", n=25),
        "top_missingness_shift_features": top_n_feature_summary(feature_rows, key="max_missing_ratio_delta", n=25),
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
