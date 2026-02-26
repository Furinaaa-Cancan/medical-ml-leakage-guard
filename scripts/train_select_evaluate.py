#!/usr/bin/env python3
"""
Train, select, and evaluate binary models with leakage-safe model-selection evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/select/evaluate leakage-safe medical binary models.")
    parser.add_argument("--train", required=True, help="Path to train CSV.")
    parser.add_argument("--valid", required=True, help="Path to valid CSV.")
    parser.add_argument("--test", required=True, help="Path to test CSV.")
    parser.add_argument("--target-col", default="y", help="Target column.")
    parser.add_argument("--patient-id-col", default="patient_id", help="Patient ID column used for trace hashing.")
    parser.add_argument("--ignore-cols", default="patient_id,event_time", help="Comma-separated non-feature columns.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--missingness-policy", help="Optional missingness policy JSON path.")
    parser.add_argument("--selection-data", default="cv_inner", help="Model selection source (valid/cv_inner/nested_cv).")
    parser.add_argument("--threshold-selection-split", default="valid", help="Split used for threshold selection.")
    parser.add_argument(
        "--calibration-method",
        default="none",
        choices=["sigmoid", "isotonic", "power", "beta", "none"],
        help="Probability calibration method fit on leakage-safe split.",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="CV folds for candidate scoring.")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for F-beta threshold objective.")
    parser.add_argument("--sensitivity-floor", type=float, default=0.85, help="Minimum sensitivity for threshold choice.")
    parser.add_argument("--npv-floor", type=float, default=0.90, help="Minimum NPV for threshold choice.")
    parser.add_argument("--specificity-floor", type=float, default=0.40, help="Minimum specificity for threshold choice.")
    parser.add_argument("--ppv-floor", type=float, default=0.55, help="Minimum PPV for threshold choice.")
    parser.add_argument("--random-seed", type=int, default=20260225, help="Random seed.")
    parser.add_argument("--primary-metric", default="pr_auc", help="Primary optimization metric.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500, help="Bootstrap samples for CI.")
    parser.add_argument("--ci-bootstrap-resamples", type=int, default=2000, help="Bootstrap samples for CI matrix report.")
    parser.add_argument("--permutation-resamples", type=int, default=300, help="Permutation samples for null metric.")
    parser.add_argument(
        "--fast-diagnostic-mode",
        action="store_true",
        help=(
            "Skip expensive publication-style summary computations (e.g., CI matrix) "
            "for fast seed-search diagnostics."
        ),
    )
    parser.add_argument("--model-selection-report-out", required=True, help="Output model_selection_report.json.")
    parser.add_argument("--evaluation-report-out", required=True, help="Output evaluation_report.json.")
    parser.add_argument("--feature-group-spec", help="Optional feature_group_spec JSON path.")
    parser.add_argument("--feature-engineering-report-out", help="Optional feature_engineering_report JSON output path.")
    parser.add_argument(
        "--feature-engineering-mode",
        default="strict",
        choices=["strict", "moderate", "quick"],
        help="Feature engineering aggressiveness mode.",
    )
    parser.add_argument("--distribution-report-out", help="Optional distribution_report JSON output path.")
    parser.add_argument("--ci-matrix-report-out", help="Optional ci_matrix_report JSON output path.")
    parser.add_argument("--prediction-trace-out", help="Optional output prediction_trace CSV/CSV.GZ.")
    parser.add_argument("--external-cohort-spec", help="Optional external cohort spec JSON path.")
    parser.add_argument("--external-validation-report-out", help="Optional output external_validation_report.json.")
    parser.add_argument("--robustness-report-out", help="Optional output robustness_report.json.")
    parser.add_argument("--robustness-time-slices", type=int, default=4, help="Number of chronological slices for robustness report.")
    parser.add_argument("--robustness-group-count", type=int, default=4, help="Number of hash-based patient groups for robustness report.")
    parser.add_argument("--seed-sensitivity-out", help="Optional output seed_sensitivity_report.json.")
    parser.add_argument(
        "--seed-sensitivity-seeds",
        default="20260224,20260225,20260226,20260227,20260228",
        help="Comma-separated integer seeds for robustness sensitivity report.",
    )
    parser.add_argument("--model-out", help="Optional model artifact output path.")
    parser.add_argument("--permutation-null-out", help="Optional null metric output file (one value per line).")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_policy(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("performance policy JSON root must be object.")
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_ignore_cols(raw: str, target_col: str) -> List[str]:
    out: List[str] = [target_col]
    for token in raw.split(","):
        key = token.strip()
        if key:
            out.append(key)
    return sorted(set(out))


def parse_seed_list(raw: str, default_seed: int) -> List[int]:
    seeds: List[int] = []
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = int(item)
        except Exception:
            continue
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        seeds = [int(default_seed)]
    return seeds


def load_split(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Split is empty: {p}")
    return df


def load_external_cohort_spec(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("external cohort spec JSON root must be object.")
    return payload


def load_feature_group_spec(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("feature_group_spec JSON root must be object.")
    return payload


def normalize_feature_groups(payload: Dict[str, Any]) -> Tuple[Dict[str, List[str]], List[str]]:
    groups_raw = payload.get("groups")
    groups: Dict[str, List[str]] = {}
    if isinstance(groups_raw, dict):
        for key, values in groups_raw.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(values, list):
                continue
            clean = [str(x).strip() for x in values if isinstance(x, str) and str(x).strip()]
            if clean:
                groups[key.strip()] = clean
    forbidden_raw = payload.get("forbidden_features")
    forbidden = [str(x).strip() for x in forbidden_raw if isinstance(x, str) and str(x).strip()] if isinstance(forbidden_raw, list) else []
    return groups, sorted(set(forbidden))


def mode_config(mode: str) -> Dict[str, Any]:
    token = str(mode).strip().lower()
    if token == "quick":
        return {"max_missing_ratio": 0.80, "min_variance": 1e-10, "group_keep_ratio": 0.85, "stability_repeats": 25}
    if token == "moderate":
        return {"max_missing_ratio": 0.70, "min_variance": 1e-9, "group_keep_ratio": 0.70, "stability_repeats": 35}
    return {"max_missing_ratio": 0.60, "min_variance": 1e-8, "group_keep_ratio": 0.50, "stability_repeats": 45}


def select_features_by_filter(
    X_train: pd.DataFrame,
    features: Sequence[str],
    max_missing_ratio: float,
    min_variance: float,
) -> Tuple[List[str], Dict[str, Any]]:
    kept: List[str] = []
    dropped_missing: List[str] = []
    dropped_low_variance: List[str] = []
    for feature in features:
        series = X_train[feature]
        missing_ratio = float(series.isna().mean())
        if missing_ratio > float(max_missing_ratio):
            dropped_missing.append(feature)
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        variance = float(numeric.var(skipna=True)) if numeric.notna().any() else 0.0
        if variance <= float(min_variance):
            dropped_low_variance.append(feature)
            continue
        kept.append(feature)
    report = {
        "max_missing_ratio": float(max_missing_ratio),
        "min_variance": float(min_variance),
        "dropped_for_missingness": dropped_missing,
        "dropped_for_low_variance": dropped_low_variance,
        "kept_count": int(len(kept)),
    }
    return kept, report


def impute_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = pd.to_numeric(out[col], errors="coerce")
        median = float(series.median(skipna=True)) if series.notna().any() else 0.0
        out[col] = series.fillna(median)
    return out


def feature_stability_frequency(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    if not features:
        return {}
    rng = np.random.default_rng(seed)
    counts = {feature: 0 for feature in features}
    effective = 0
    y = np.asarray(y_train, dtype=int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if pos_idx.size == 0 or neg_idx.size == 0:
        return {feature: 0.0 for feature in features}

    for _ in range(int(repeats)):
        sample_pos = rng.choice(pos_idx, size=max(1, int(0.8 * pos_idx.size)), replace=True)
        sample_neg = rng.choice(neg_idx, size=max(1, int(0.8 * neg_idx.size)), replace=True)
        idx = np.concatenate([sample_pos, sample_neg], axis=0)
        rng.shuffle(idx)
        X_sub = impute_numeric_frame(X_train.iloc[idx][list(features)])
        y_sub = y[idx]
        try:
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.3,
                class_weight=None,
                max_iter=3000,
                random_state=seed,
            )
            model.fit(X_sub, y_sub)
            coef = np.asarray(model.coef_).reshape(-1)
            for feature, value in zip(features, coef):
                if abs(float(value)) > 1e-10:
                    counts[feature] += 1
            effective += 1
        except Exception:
            continue
    if effective <= 0:
        return {feature: 0.0 for feature in features}
    return {feature: float(counts[feature]) / float(effective) for feature in features}


def group_preselect_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    groups: Dict[str, List[str]],
    keep_ratio: float,
    stability_frequency: Dict[str, float],
) -> Tuple[List[str], Dict[str, Any]]:
    if not groups:
        return list(features), {"groups": {}, "fallback": "no_groups_provided"}

    y = np.asarray(y_train, dtype=float)
    report_groups: Dict[str, Any] = {}
    selected: List[str] = []
    feature_set = set(features)

    for group_name, group_features in groups.items():
        available = [f for f in group_features if f in feature_set]
        if not available:
            report_groups[group_name] = {
                "declared_count": len(group_features),
                "available_count": 0,
                "selected_count": 0,
                "selected_features": [],
            }
            continue

        scored: List[Tuple[str, float, float, float]] = []
        for feature in available:
            series = pd.to_numeric(X_train[feature], errors="coerce")
            if series.notna().any():
                corr = abs(float(series.fillna(series.median(skipna=True)).corr(pd.Series(y))))
                if not math.isfinite(corr):
                    corr = 0.0
            else:
                corr = 0.0
            freq = float(stability_frequency.get(feature, 0.0))
            combined = 0.70 * corr + 0.30 * freq
            scored.append((feature, combined, corr, freq))

        scored = sorted(scored, key=lambda x: (x[1], x[2], x[3], x[0]), reverse=True)
        keep_n = max(1, int(math.ceil(float(keep_ratio) * float(len(scored)))))
        keep_n = min(keep_n, len(scored))
        chosen = [item[0] for item in scored[:keep_n]]
        selected.extend(chosen)
        report_groups[group_name] = {
            "declared_count": len(group_features),
            "available_count": len(available),
            "selected_count": len(chosen),
            "selected_features": chosen,
            "ranked_features": [
                {
                    "feature": feature,
                    "combined_score": float(combined),
                    "corr_abs": float(corr),
                    "selection_frequency": float(freq),
                }
                for feature, combined, corr, freq in scored
            ],
        }

    dedup_selected = sorted(dict.fromkeys(selected))
    if not dedup_selected:
        dedup_selected = sorted(features)
    report = {"groups": report_groups, "selected_feature_count": len(dedup_selected)}
    return dedup_selected, report


def select_feature_columns(train_df: pd.DataFrame, ignore_cols: Sequence[str]) -> List[str]:
    ignore = set(ignore_cols)
    out = [c for c in train_df.columns if c not in ignore]
    if not out:
        raise ValueError("No feature columns remain after ignore-cols exclusion.")
    return out


def prepare_xy(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features[:10]}")
    X = df[list(feature_cols)].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(y)):
        raise ValueError("Target contains non-finite values.")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("Target must be binary (0/1).")
    return X, y.astype(int)


def build_imputer(imputation_strategy: str, seed: int) -> BaseEstimator:
    if imputation_strategy == "mice":
        return IterativeImputer(
            random_state=seed,
            max_iter=20,
            initial_strategy="median",
            sample_posterior=False,
        )
    return SimpleImputer(strategy="median", add_indicator=True)


def build_candidates(seed: int, imputation_strategy: str, class_weight: Optional[str]) -> List[Dict[str, Any]]:
    return [
        {
            "model_id": "logistic_l1",
            "family": "logistic_regression",
            "complexity_rank": 1,
            "estimator": Pipeline(
                steps=[
                    ("imputer", build_imputer(imputation_strategy, seed)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="l1",
                            max_iter=5000,
                            solver="liblinear",
                            C=0.3,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "logistic_l2",
            "family": "logistic_regression",
            "complexity_rank": 2,
            "estimator": Pipeline(
                steps=[
                    ("imputer", build_imputer(imputation_strategy, seed)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="l2",
                            max_iter=5000,
                            solver="liblinear",
                            C=1.0,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "logistic_elasticnet",
            "family": "logistic_regression",
            "complexity_rank": 3,
            "estimator": Pipeline(
                steps=[
                    ("imputer", build_imputer(imputation_strategy, seed)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="elasticnet",
                            l1_ratio=0.5,
                            max_iter=6000,
                            solver="saga",
                            C=0.8,
                            class_weight=class_weight,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "random_forest_balanced",
            "family": "random_forest",
            "complexity_rank": 4,
            "estimator": Pipeline(
                steps=[
                    ("imputer", build_imputer(imputation_strategy, seed)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=4,
                            min_samples_split=20,
                            min_samples_leaf=10,
                            class_weight=class_weight,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "hist_gradient_boosting_l2",
            "family": "hist_gradient_boosting",
            "complexity_rank": 5,
            "estimator": Pipeline(
                steps=[
                    ("imputer", build_imputer(imputation_strategy, seed)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.03,
                            max_depth=3,
                            max_iter=180,
                            l2_regularization=5.0,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
    ]


def predict_proba_1(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Estimator does not expose probability-like outputs.")


def fit_probability_calibrator(
    y_true: np.ndarray,
    proba_raw: np.ndarray,
    method: str,
    seed: int,
) -> Optional[Any]:
    token = str(method).strip().lower()
    if token in {"", "none"}:
        return None
    if token not in {"sigmoid", "isotonic", "power", "beta"}:
        raise ValueError(f"Unsupported calibration method: {method}")
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(proba_raw, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("Calibration arrays must be 1D and aligned.")
    if y.shape[0] < 20:
        return None
    if len(np.unique(y)) < 2:
        return None
    s = np.clip(s, 1e-6, 1.0 - 1e-6)
    if token == "isotonic":
        calibrator = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        calibrator.fit(s, y.astype(float))
        return calibrator
    if token == "power":
        # Smooth monotonic calibration that preserves ranking while controlling over/under-confidence.
        def calibration_ece_local(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
            n = int(labels.shape[0])
            if n <= 0:
                return 1.0
            requested_bins = max(2, int(n_bins))
            effective_bins = max(2, n // 10)
            bin_count = min(requested_bins, effective_bins)
            order = np.argsort(scores.astype(float))
            blocks = np.array_split(order, bin_count)
            total = 0.0
            for idx in blocks:
                count = int(idx.shape[0])
                if count == 0:
                    continue
                avg_score = float(np.mean(scores[idx]))
                avg_true = float(np.mean(labels[idx]))
                total += (count / n) * abs(avg_true - avg_score)
            return float(total)

        best: Optional[Tuple[float, float, float]] = None
        grid = np.linspace(0.70, 1.60, 181)
        for alpha in grid.tolist():
            calibrated = np.clip(np.power(s, float(alpha)), 1e-6, 1.0 - 1e-6)
            ece = calibration_ece_local(y, calibrated, n_bins=10)
            brier = float(brier_score_loss(y, calibrated))
            candidate = (float(ece), float(brier), float(alpha))
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        return {
            "kind": "power",
            "alpha": float(best[2]),
            "valid_ece": float(best[0]),
            "valid_brier": float(best[1]),
        }
    if token == "beta":
        beta_features = np.column_stack([np.log(s), np.log(1.0 - s)])
        beta_model = LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            random_state=int(seed),
        )
        beta_model.fit(beta_features, y)
        coef = np.asarray(beta_model.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(beta_model.intercept_, dtype=float).reshape(-1)[0])
        return {
            "kind": "beta",
            "coef_log_p": float(coef[0]) if coef.shape[0] >= 1 else 0.0,
            "coef_log_one_minus_p": float(coef[1]) if coef.shape[0] >= 2 else 0.0,
            "intercept": intercept,
        }
    calibrator = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=int(seed),
    )
    calibrator.fit(s.reshape(-1, 1), y)
    return calibrator


def apply_probability_calibrator(calibrator: Optional[Any], proba_raw: np.ndarray) -> np.ndarray:
    s = np.asarray(proba_raw, dtype=float)
    s = np.clip(s, 1e-6, 1.0 - 1e-6)
    if calibrator is None:
        return s
    if isinstance(calibrator, dict):
        kind = str(calibrator.get("kind", "")).strip().lower()
        if kind == "power":
            alpha = float(calibrator.get("alpha", 1.0))
            return np.clip(np.power(s, alpha), 1e-6, 1.0 - 1e-6)
        if kind == "beta":
            coef_log_p = float(calibrator.get("coef_log_p", 0.0))
            coef_log_one_minus_p = float(calibrator.get("coef_log_one_minus_p", 0.0))
            intercept = float(calibrator.get("intercept", 0.0))
            logits = (coef_log_p * np.log(s)) + (coef_log_one_minus_p * np.log(1.0 - s)) + intercept
            return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1.0 - 1e-6)
        raise ValueError(f"Unsupported calibrator kind: {kind}")
    if hasattr(calibrator, "predict_proba"):
        out = calibrator.predict_proba(s.reshape(-1, 1))
        if isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 2:
            return np.clip(np.asarray(out[:, 1], dtype=float), 1e-6, 1.0 - 1e-6)
    if hasattr(calibrator, "predict"):
        out = calibrator.predict(s)
        return np.clip(np.asarray(out, dtype=float), 1e-6, 1.0 - 1e-6)
    raise ValueError("Calibrator does not expose calibrated probabilities.")


def cv_score_pr_auc(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> Tuple[float, float, int, List[float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: List[float] = []
    for tr_idx, va_idx in splitter.split(X, y):
        y_val = y[va_idx]
        if len(np.unique(y_val)) < 2:
            continue
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        proba = predict_proba_1(model, X.iloc[va_idx])
        fold_scores.append(clip01(float(average_precision_score(y_val, proba))))
    if len(fold_scores) < 2:
        raise ValueError("Insufficient valid CV folds for PR-AUC scoring.")
    arr = np.asarray(fold_scores, dtype=float)
    return clip01(float(arr.mean())), clip01(float(arr.std(ddof=1))), int(arr.shape[0]), [clip01(float(x)) for x in fold_scores]


def choose_model_one_se(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = max(rows, key=lambda r: float(r["mean"]))
    best_se = float(best["std"]) / math.sqrt(float(best["n_folds"]))
    threshold = float(best["mean"]) - best_se
    eligible = [r for r in rows if float(r["mean"]) >= threshold - 1e-12]
    chosen = sorted(
        eligible,
        key=lambda r: (int(r["complexity_rank"]), -float(r["mean"]), str(r["model_id"])),
    )[0]
    return {
        "best_model_id": str(best["model_id"]),
        "best_mean": float(best["mean"]),
        "best_se": best_se,
        "one_se_threshold": threshold,
        "eligible_model_ids": [str(r["model_id"]) for r in eligible],
        "chosen_model_id": str(chosen["model_id"]),
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    tn = int(np.sum((y_true_i == 0) & (y_pred_i == 0)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, float(value))))


def metric_panel(y_true: np.ndarray, proba: np.ndarray, threshold: float, beta: float) -> Tuple[Dict[str, float], Dict[str, int]]:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_counts(y_true, y_pred)
    tp = float(cm["tp"])
    fp = float(cm["fp"])
    tn = float(cm["tn"])
    fn = float(cm["fn"])
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    npv = safe_ratio(tn, tn + fn)
    accuracy = safe_ratio(tp + tn, tp + fp + tn + fn)
    f1 = 0.0 if (precision + sensitivity) <= 0 else (2.0 * precision * sensitivity) / (precision + sensitivity)
    beta_sq = beta * beta
    f2 = 0.0 if ((beta_sq * precision) + sensitivity) <= 0 else ((1.0 + beta_sq) * precision * sensitivity) / (
        (beta_sq * precision) + sensitivity
    )
    roc_auc = float(roc_auc_score(y_true, proba))
    pr_auc = float(average_precision_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))
    metrics = {
        "accuracy": clip01(accuracy),
        "precision": clip01(precision),
        "ppv": clip01(precision),
        "npv": clip01(npv),
        "sensitivity": clip01(sensitivity),
        "specificity": clip01(specificity),
        "f1": clip01(f1),
        "f2_beta": clip01(f2),
        "roc_auc": clip01(roc_auc),
        "pr_auc": clip01(pr_auc),
        "brier": clip01(brier),
    }
    return metrics, cm


def choose_threshold(
    y_valid: np.ndarray,
    proba_valid: np.ndarray,
    beta: float,
    sensitivity_floor: float,
    npv_floor: float,
    specificity_floor: float,
    ppv_floor: float,
    guard_y: Optional[np.ndarray] = None,
    guard_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    quantiles = np.linspace(0.01, 0.99, 299)
    thresholds = sorted(set(float(np.quantile(proba_valid, q)) for q in quantiles) | {0.5})
    candidates: List[Dict[str, Any]] = []

    def floor_margin(metrics: Dict[str, float]) -> Dict[str, float]:
        sens_margin = float(metrics["sensitivity"]) - float(sensitivity_floor)
        npv_margin = float(metrics["npv"]) - float(npv_floor)
        spec_margin = float(metrics["specificity"]) - float(specificity_floor)
        ppv_margin = float(metrics["ppv"]) - float(ppv_floor)
        all_margins = [sens_margin, npv_margin, spec_margin, ppv_margin]
        return {
            "sensitivity": sens_margin,
            "npv": npv_margin,
            "specificity": spec_margin,
            "ppv": ppv_margin,
            "min": float(min(all_margins)),
            "primary_sum": float(sens_margin + npv_margin),
            "secondary_sum": float(spec_margin + ppv_margin),
        }

    for threshold in thresholds:
        metrics, cm = metric_panel(y_valid, proba_valid, threshold, beta=beta)
        margin = floor_margin(metrics)
        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
            "confusion_matrix": cm,
            "floor_margin": margin,
            "feasible": bool(
                metrics["sensitivity"] >= sensitivity_floor
                and metrics["npv"] >= npv_floor
                and metrics["specificity"] >= specificity_floor
                and metrics["ppv"] >= ppv_floor
            ),
        }
        candidates.append(candidate)

    selected: Optional[Dict[str, Any]] = None
    feasible = [c for c in candidates if bool(c.get("feasible"))]
    if feasible:
        threshold_pool = feasible
        if (
            guard_y is not None
            and guard_proba is not None
            and int(np.asarray(guard_y).shape[0]) == int(np.asarray(guard_proba).shape[0])
        ):
            for candidate in threshold_pool:
                guard_metrics, _ = metric_panel(
                    np.asarray(guard_y, dtype=int),
                    np.asarray(guard_proba, dtype=float),
                    float(candidate["threshold"]),
                    beta=beta,
                )
                candidate["guard_metrics"] = guard_metrics
                candidate["guard_floor_margin"] = floor_margin(guard_metrics)
                candidate["guard_feasible"] = bool(
                    guard_metrics["sensitivity"] >= sensitivity_floor
                    and guard_metrics["npv"] >= npv_floor
                    and guard_metrics["specificity"] >= specificity_floor
                    and guard_metrics["ppv"] >= ppv_floor
                )
            guard_feasible = [c for c in threshold_pool if bool(c.get("guard_feasible"))]
            if guard_feasible:
                selected = sorted(
                    guard_feasible,
                    key=lambda c: (
                        -float(c["guard_floor_margin"]["min"]),
                        -float(c["guard_floor_margin"]["primary_sum"]),
                        -float(c["guard_metrics"]["f2_beta"]),
                        -float(c["floor_margin"]["min"]),
                        -float(c["floor_margin"]["primary_sum"]),
                        -float(c["guard_floor_margin"]["secondary_sum"]),
                        -float(c["floor_margin"]["secondary_sum"]),
                        -float(c["guard_metrics"]["sensitivity"]),
                        -float(c["guard_metrics"]["npv"]),
                        -float(c["guard_metrics"]["specificity"]),
                        -float(c["guard_metrics"]["ppv"]),
                        float(c["threshold"]),
                    ),
                )[0]
            else:
                spec_target = min(1.0, float(specificity_floor) + 0.05)
                ppv_target = min(1.0, float(ppv_floor) + 0.03)
                guard_min_sensitivity = max(0.0, float(sensitivity_floor) - 0.05)
                guard_min_npv = max(0.0, float(npv_floor) - 0.15)
                stability_pool = [
                    c
                    for c in threshold_pool
                    if float(c["guard_metrics"]["sensitivity"]) >= guard_min_sensitivity
                    and float(c["guard_metrics"]["npv"]) >= guard_min_npv
                ]
                ranked_candidates = stability_pool if stability_pool else threshold_pool
                for candidate in ranked_candidates:
                    gm = candidate["guard_metrics"]
                    deficit_sens = max(0.0, float(sensitivity_floor) - float(gm["sensitivity"]))
                    deficit_npv = max(0.0, float(npv_floor) - float(gm["npv"]))
                    deficit_spec = max(0.0, float(specificity_floor) - float(gm["specificity"]))
                    deficit_ppv = max(0.0, float(ppv_floor) - float(gm["ppv"]))
                    deficits = [deficit_sens, deficit_npv, deficit_spec, deficit_ppv]
                    candidate["guard_deficit"] = {
                        "total": float(sum(deficits)),
                        "max": float(max(deficits)),
                        "sensitivity": float(deficit_sens),
                        "npv": float(deficit_npv),
                        "specificity": float(deficit_spec),
                        "ppv": float(deficit_ppv),
                    }
                    candidate["guard_target_gap"] = {
                        "specificity": abs(float(gm["specificity"]) - float(spec_target)),
                        "ppv": abs(float(gm["ppv"]) - float(ppv_target)),
                    }
                selected = sorted(
                    ranked_candidates,
                    key=lambda c: (
                        float(c["guard_deficit"]["sensitivity"]) + float(c["guard_deficit"]["npv"]),
                        float(c["guard_target_gap"]["specificity"]) + float(c["guard_target_gap"]["ppv"]),
                        -float(c["guard_metrics"]["f2_beta"]),
                        -float(c["guard_metrics"]["specificity"]),
                        -float(c["guard_metrics"]["ppv"]),
                        -float(c["guard_metrics"]["sensitivity"]),
                        -float(c["guard_metrics"]["npv"]),
                        float(c["threshold"]),
                    ),
                )[0]
        else:
            selected = sorted(
                threshold_pool,
                key=lambda c: (
                    -float(c["floor_margin"]["min"]),
                    -float(c["floor_margin"]["primary_sum"]),
                    -float(c["metrics"]["f2_beta"]),
                    -float(c["floor_margin"]["secondary_sum"]),
                    -float(c["metrics"]["sensitivity"]),
                    -float(c["metrics"]["npv"]),
                    -float(c["metrics"]["specificity"]),
                    -float(c["metrics"]["ppv"]),
                    float(c["threshold"]),
                ),
            )[0]
    elif candidates:
        # If no threshold fully satisfies floors on the selection split, pick the
        # clinically closest operating point instead of blindly maximizing F2.
        for candidate in candidates:
            m = candidate["metrics"]
            deficit_sens = max(0.0, float(sensitivity_floor) - float(m["sensitivity"]))
            deficit_npv = max(0.0, float(npv_floor) - float(m["npv"]))
            deficit_spec = max(0.0, float(specificity_floor) - float(m["specificity"]))
            deficit_ppv = max(0.0, float(ppv_floor) - float(m["ppv"]))
            deficits = [deficit_sens, deficit_npv, deficit_spec, deficit_ppv]
            candidate["constraint_deficit"] = {
                "total": float(sum(deficits)),
                "max": float(max(deficits)),
                "sensitivity": float(deficit_sens),
                "npv": float(deficit_npv),
                "specificity": float(deficit_spec),
                "ppv": float(deficit_ppv),
            }
        selected = sorted(
            candidates,
            key=lambda c: (
                float(c["constraint_deficit"]["total"]),
                float(c["constraint_deficit"]["max"]),
                float(c["constraint_deficit"]["specificity"]) + float(c["constraint_deficit"]["ppv"]),
                -float(c["metrics"]["specificity"]),
                -float(c["metrics"]["ppv"]),
                -float(c["metrics"]["sensitivity"]),
                -float(c["metrics"]["npv"]),
                -float(c["metrics"]["f2_beta"]),
                float(c["threshold"]),
            ),
        )[0]
    if selected is None:
        raise ValueError("Unable to choose threshold.")
    selection_ok = bool(selected["feasible"])
    guard_ok: Optional[bool] = None
    if "guard_feasible" in selected:
        guard_ok = bool(selected.get("guard_feasible"))
    overall_ok = bool(selection_ok and (guard_ok if guard_ok is not None else True))
    return {
        "selected_threshold": float(selected["threshold"]),
        "constraints_satisfied_selection_split": selection_ok,
        "constraints_satisfied_guard_split": guard_ok,
        "constraints_satisfied_overall": overall_ok,
        # Backward-compatible legacy field; explicitly aligned to overall status.
        "constraints_satisfied": overall_ok,
        "selected_metrics_on_valid": selected["metrics"],
        "selected_confusion_on_valid": selected["confusion_matrix"],
        "guard_metrics": selected.get("guard_metrics"),
        "sensitivity_floor": float(sensitivity_floor),
        "npv_floor": float(npv_floor),
        "specificity_floor": float(specificity_floor),
        "ppv_floor": float(ppv_floor),
    }


def metric_panel_robust(y_true: np.ndarray, proba: np.ndarray, threshold: float, beta: float) -> Dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_counts(y_true, y_pred)
    tp = float(cm["tp"])
    fp = float(cm["fp"])
    tn = float(cm["tn"])
    fn = float(cm["fn"])
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    beta_sq = beta * beta
    f2 = 0.0 if ((beta_sq * precision) + sensitivity) <= 0 else ((1.0 + beta_sq) * precision * sensitivity) / (
        (beta_sq * precision) + sensitivity
    )
    try:
        pr_auc = float(average_precision_score(y_true, proba))
    except Exception:
        pr_auc = safe_ratio(float(np.sum(y_true.astype(int) == 1)), float(y_true.shape[0]))
    try:
        brier = float(brier_score_loss(y_true, proba))
    except Exception:
        brier = 1.0
    pr_auc = min(1.0, max(0.0, float(pr_auc)))
    f2 = min(1.0, max(0.0, float(f2)))
    brier = min(1.0, max(0.0, float(brier)))
    return {
        "pr_auc": pr_auc,
        "f2_beta": f2,
        "brier": brier,
    }


def stable_group_index(value: str, n_groups: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % n_groups


def cv_oof_proba(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.full(shape=y.shape[0], fill_value=np.nan, dtype=float)
    for tr_idx, va_idx in splitter.split(X, y):
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        out[va_idx] = predict_proba_1(model, X.iloc[va_idx])
    if np.any(~np.isfinite(out)):
        raise ValueError("Failed to compute finite OOF probabilities for threshold selection.")
    return out


def load_missingness_policy(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("missingness policy JSON root must be object.")
    return payload


def resolve_imputation_plan(
    policy: Dict[str, Any],
    train_rows: int,
    feature_count: int,
) -> Dict[str, Any]:
    strategy = str(policy.get("strategy", "simple_with_indicator")).strip().lower()
    if not strategy:
        strategy = "simple_with_indicator"
    if strategy not in {"simple_with_indicator", "mice", "mice_with_scale_guard"}:
        strategy = "simple_with_indicator"

    plan: Dict[str, Any] = {
        "policy_strategy": strategy,
        "executed_strategy": "simple_with_indicator",
        "fit_scope": str(policy.get("imputer_fit_scope", "train_only")).strip().lower() or "train_only",
        "scale_guard": {
            "checked": strategy == "mice_with_scale_guard",
            "triggered": False,
            "fallback_strategy": None,
            "mice_max_rows": None,
            "mice_max_cols": None,
            "large_data_row_threshold": None,
            "large_data_col_threshold": None,
        },
    }

    if strategy == "mice":
        plan["executed_strategy"] = "mice"
        return plan

    if strategy != "mice_with_scale_guard":
        return plan

    mice_max_rows = policy.get("mice_max_rows", 200000)
    mice_max_cols = policy.get("mice_max_cols", 200)
    large_rows = policy.get("large_data_row_threshold", 1_000_000)
    large_cols = policy.get("large_data_col_threshold", 300)
    try:
        mice_max_rows_i = int(mice_max_rows)
        mice_max_cols_i = int(mice_max_cols)
        large_rows_i = int(large_rows)
        large_cols_i = int(large_cols)
    except Exception:
        mice_max_rows_i = 200000
        mice_max_cols_i = 200
        large_rows_i = 1_000_000
        large_cols_i = 300

    should_trigger = (
        train_rows > mice_max_rows_i
        or feature_count > mice_max_cols_i
        or (train_rows >= large_rows_i and feature_count >= large_cols_i)
    )
    plan["executed_strategy"] = "simple_with_indicator" if should_trigger else "mice"
    plan["scale_guard"] = {
        "checked": True,
        "triggered": bool(should_trigger),
        "fallback_strategy": "simple_with_indicator" if should_trigger else None,
        "mice_max_rows": mice_max_rows_i,
        "mice_max_cols": mice_max_cols_i,
        "large_data_row_threshold": large_rows_i,
        "large_data_col_threshold": large_cols_i,
    }
    return plan


def resolve_external_cohorts(
    external_spec: Dict[str, Any],
    external_spec_path: Optional[str],
    feature_cols: Sequence[str],
    default_target_col: str,
    default_patient_id_col: str,
) -> List[Dict[str, Any]]:
    cohorts = external_spec.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        return []

    base = Path(external_spec_path).expanduser().resolve().parent if external_spec_path else Path.cwd()
    out: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(cohorts):
        if not isinstance(entry, dict):
            raise ValueError(f"external cohort entry #{idx} must be object.")

        cohort_id = str(entry.get("cohort_id", "")).strip()
        if not cohort_id:
            raise ValueError(f"external cohort entry #{idx} missing cohort_id.")
        if cohort_id in seen_ids:
            raise ValueError(f"external cohort entry has duplicate cohort_id: {cohort_id}")
        seen_ids.add(cohort_id)

        cohort_type = str(entry.get("cohort_type", "")).strip().lower()
        if cohort_type not in {"cross_period", "cross_institution"}:
            raise ValueError(
                f"external cohort '{cohort_id}' must set cohort_type to cross_period or cross_institution."
            )

        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"external cohort '{cohort_id}' missing path.")
        data_path = Path(raw_path.strip()).expanduser()
        if not data_path.is_absolute():
            data_path = (base / data_path).resolve()
        else:
            data_path = data_path.resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"external cohort '{cohort_id}' path not found: {data_path}")

        target_col = str(entry.get("label_col", default_target_col)).strip() or default_target_col
        patient_id_col = str(entry.get("patient_id_col", default_patient_id_col)).strip() or default_patient_id_col

        cohort_df = load_split(str(data_path))
        X_ext, y_ext = prepare_xy(cohort_df, feature_cols=feature_cols, target_col=target_col)
        if patient_id_col in cohort_df.columns:
            patient_ids = cohort_df[patient_id_col].astype(str).tolist()
        else:
            patient_ids = [f"{cohort_id}_{i}" for i in range(int(cohort_df.shape[0]))]

        out.append(
            {
                "cohort_id": cohort_id,
                "cohort_type": cohort_type,
                "data_path": str(data_path),
                "frame": cohort_df,
                "target_col": target_col,
                "patient_id_col": patient_id_col,
                "X": X_ext,
                "y": y_ext,
                "patient_ids": patient_ids,
                "row_count": int(cohort_df.shape[0]),
            }
        )
    return out


def build_prediction_trace_rows(
    scope: str,
    cohort_id: str,
    cohort_type: str,
    patient_ids: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    model_id: str,
) -> pd.DataFrame:
    y_pred = (y_score >= threshold).astype(int)
    if len(patient_ids) != int(y_true.shape[0]):
        raise ValueError(f"prediction trace patient ID length mismatch for {scope}/{cohort_id}.")
    payload = {
        "scope": [scope] * int(y_true.shape[0]),
        "cohort_id": [cohort_id] * int(y_true.shape[0]),
        "cohort_type": [cohort_type] * int(y_true.shape[0]),
        "hashed_patient_id": [sha256_text(str(pid)) for pid in patient_ids],
        "y_true": y_true.astype(int),
        "y_score": y_score.astype(float),
        "y_pred": y_pred.astype(int),
        "selected_threshold": [float(threshold)] * int(y_true.shape[0]),
        "model_id": [model_id] * int(y_true.shape[0]),
    }
    return pd.DataFrame(payload)


def bootstrap_ci_pr_auc(y_true: np.ndarray, proba: np.ndarray, n_resamples: int, seed: int) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0])
    hits: List[float] = []
    max_attempts = max(5 * n_resamples, 2000)
    attempts = 0
    while len(hits) < n_resamples and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb = proba[idx]
        hits.append(float(average_precision_score(yb, pb)))
    if len(hits) < 200:
        raise ValueError(f"Insufficient bootstrap resamples for CI: {len(hits)}")
    arr = np.asarray(hits, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5]).tolist()
    return float(lo), float(hi), int(len(hits))


def stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if pos.size == 0 or neg.size == 0:
        return None
    sample_pos = rng.choice(pos, size=pos.size, replace=True)
    sample_neg = rng.choice(neg, size=neg.size, replace=True)
    idx = np.concatenate([sample_pos, sample_neg], axis=0)
    rng.shuffle(idx)
    return idx


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    beta: float,
    n_resamples: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], int]:
    rng = np.random.default_rng(seed)
    hits: Dict[str, List[float]] = {metric: [] for metric in ("accuracy", "precision", "ppv", "npv", "sensitivity", "specificity", "f1", "f2_beta", "roc_auc", "pr_auc", "brier")}
    attempts = 0
    max_attempts = max(5 * int(n_resamples), 8000)
    while len(hits["pr_auc"]) < int(n_resamples) and attempts < max_attempts:
        attempts += 1
        idx = stratified_bootstrap_indices(y_true, rng)
        if idx is None:
            break
        yb = y_true[idx]
        sb = y_score[idx]
        try:
            panel, _ = metric_panel(yb, sb, threshold, beta=beta)
        except Exception:
            continue
        if not all(isinstance(panel.get(k), (int, float)) and math.isfinite(float(panel.get(k))) for k in hits):
            continue
        for metric in hits:
            hits[metric].append(float(panel[metric]))
    effective = min((len(v) for v in hits.values()), default=0)
    summary: Dict[str, Dict[str, float]] = {}
    for metric, values in hits.items():
        arr = np.asarray(values[:effective], dtype=float)
        if arr.size == 0:
            summary[metric] = {"ci_lower": float("nan"), "ci_upper": float("nan"), "ci_width": float("nan")}
            continue
        lo, hi = np.percentile(arr, [2.5, 97.5]).tolist()
        summary[metric] = {
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "ci_width": float(hi - lo),
        }
    return summary, int(effective)


def js_divergence_from_probs(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-12
    a = np.asarray(a, dtype=float) + eps
    b = np.asarray(b, dtype=float) + eps
    a = a / float(np.sum(a))
    b = b / float(np.sum(b))
    m = 0.5 * (a + b)
    return float(0.5 * (np.sum(a * np.log(a / m)) + np.sum(b * np.log(b / m))) / math.log(2.0))


def feature_jsd(train: pd.Series, other: pd.Series) -> Optional[float]:
    tr = train.dropna()
    ot = other.dropna()
    if tr.empty or ot.empty:
        return None
    tr_num = pd.to_numeric(tr, errors="coerce")
    ot_num = pd.to_numeric(ot, errors="coerce")
    if tr_num.notna().sum() >= max(10, int(0.5 * len(tr))) and ot_num.notna().sum() >= max(10, int(0.5 * len(ot))):
        tr_arr = tr_num.dropna().to_numpy(dtype=float)
        ot_arr = ot_num.dropna().to_numpy(dtype=float)
        bins = np.unique(np.quantile(tr_arr, np.linspace(0.0, 1.0, 11)))
        if bins.size < 3:
            return None
        tr_hist, _ = np.histogram(tr_arr, bins=bins)
        ot_hist, _ = np.histogram(ot_arr, bins=bins)
        return js_divergence_from_probs(tr_hist, ot_hist)
    tr_cat = tr.astype(str)
    ot_cat = ot.astype(str)
    keys = sorted(set(tr_cat.unique().tolist()) | set(ot_cat.unique().tolist()))
    if not keys:
        return None
    tr_counts = tr_cat.value_counts(normalize=True)
    ot_counts = ot_cat.value_counts(normalize=True)
    a = np.asarray([float(tr_counts.get(k, 0.0)) for k in keys], dtype=float)
    b = np.asarray([float(ot_counts.get(k, 0.0)) for k in keys], dtype=float)
    return js_divergence_from_probs(a, b)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def summarize_seed_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarize empty metric list.")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "n": int(arr.size),
    }


def build_distribution_report(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_frames: List[Dict[str, Any]],
    target_col: str,
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    comparisons: List[Tuple[str, pd.DataFrame]] = [("valid", valid_df), ("test", test_df)]
    for ext in external_frames:
        split_name = f"external:{ext['cohort_id']}"
        comparisons.append((split_name, ext["frame"]))

    for split_name, frame in comparisons:
        jsd_values: List[float] = []
        missing_deltas: List[float] = []
        for feature in feature_cols:
            if feature not in frame.columns:
                continue
            jsd = feature_jsd(train_df[feature], frame[feature])
            if jsd is not None and math.isfinite(jsd):
                jsd_values.append(float(jsd))
            tr_missing = float(train_df[feature].isna().mean())
            ot_missing = float(frame[feature].isna().mean())
            missing_deltas.append(abs(ot_missing - tr_missing))

        y_train = pd.to_numeric(train_df[target_col], errors="coerce")
        y_other = pd.to_numeric(frame[target_col], errors="coerce")
        prev_train = float(y_train.mean()) if y_train.notna().any() else None
        prev_other = float(y_other.mean()) if y_other.notna().any() else None
        prevalence_delta = abs(prev_other - prev_train) if prev_train is not None and prev_other is not None else None
        rows.append(
            {
                "split": split_name,
                "feature_count": int(len(feature_cols)),
                "top_feature_jsd": float(max(jsd_values)) if jsd_values else 0.0,
                "mean_feature_jsd": float(np.mean(jsd_values)) if jsd_values else 0.0,
                "max_missing_ratio_delta": float(max(missing_deltas)) if missing_deltas else 0.0,
                "prevalence_delta": float(prevalence_delta) if prevalence_delta is not None else None,
            }
        )

    return {
        "schema_version": "v4.0",
        "distribution_matrix": rows,
        "metadata": {
            "train_rows": int(train_df.shape[0]),
            "valid_rows": int(valid_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "external_cohort_count": int(len(external_frames)),
        },
    }


def build_ci_matrix_report(
    split_payloads: Dict[str, Dict[str, Any]],
    external_payloads: List[Dict[str, Any]],
    beta: float,
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    split_metrics_ci: Dict[str, Any] = {}
    for idx, (split_name, payload) in enumerate(split_payloads.items()):
        y_true = payload["y_true"]
        y_score = payload["y_score"]
        threshold = float(payload["threshold"])
        point_metrics, _ = metric_panel(y_true, y_score, threshold, beta=beta)
        ci_summary, effective = bootstrap_metric_ci(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            beta=beta,
            n_resamples=int(n_resamples),
            seed=int(seed + idx * 17),
        )
        metrics_block: Dict[str, Any] = {}
        for metric_name, point_value in point_metrics.items():
            ci = ci_summary.get(metric_name, {})
            lo = float(ci.get("ci_lower", float("nan")))
            hi = float(ci.get("ci_upper", float("nan")))
            metrics_block[metric_name] = {
                "point": float(point_value),
                "ci_95": [lo, hi],
                "ci_width": float(hi - lo) if math.isfinite(lo) and math.isfinite(hi) else float("nan"),
                "n_resamples": int(effective),
            }
        split_metrics_ci[split_name] = {
            "row_count": int(len(y_true)),
            "positive_count": int(np.sum(np.asarray(y_true, dtype=int) == 1)),
            "selected_threshold": float(threshold),
            "metrics": metrics_block,
        }

    external_metrics_ci: Dict[str, Any] = {}
    for idx, payload in enumerate(external_payloads):
        cohort_id = str(payload["cohort_id"])
        y_true = payload["y_true"]
        y_score = payload["y_score"]
        threshold = float(payload["threshold"])
        point_metrics, _ = metric_panel(y_true, y_score, threshold, beta=beta)
        ci_summary, effective = bootstrap_metric_ci(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            beta=beta,
            n_resamples=int(n_resamples),
            seed=int(seed + 900 + idx * 29),
        )
        metrics_block: Dict[str, Any] = {}
        for metric_name, point_value in point_metrics.items():
            ci = ci_summary.get(metric_name, {})
            lo = float(ci.get("ci_lower", float("nan")))
            hi = float(ci.get("ci_upper", float("nan")))
            metrics_block[metric_name] = {
                "point": float(point_value),
                "ci_95": [lo, hi],
                "ci_width": float(hi - lo) if math.isfinite(lo) and math.isfinite(hi) else float("nan"),
                "n_resamples": int(effective),
            }
        external_metrics_ci[cohort_id] = {
            "cohort_type": str(payload["cohort_type"]),
            "row_count": int(len(y_true)),
            "positive_count": int(np.sum(np.asarray(y_true, dtype=int) == 1)),
            "selected_threshold": float(threshold),
            "metrics": metrics_block,
        }

    transport_drop_ci: Dict[str, Any] = {}
    internal_test = split_metrics_ci.get("test")
    if isinstance(internal_test, dict) and isinstance(internal_test.get("metrics"), dict):
        for cohort_id, ext in external_metrics_ci.items():
            ext_metrics = ext.get("metrics") if isinstance(ext, dict) else None
            if not isinstance(ext_metrics, dict):
                continue
            transport_drop_ci[cohort_id] = {
                "pr_auc_drop": {
                    "point": float(internal_test["metrics"]["pr_auc"]["point"] - ext_metrics["pr_auc"]["point"]),
                    "ci_95": None,
                    "ci_width": None,
                    "ci_note": "not_computed_point_estimate_only",
                    "n_resamples": int(ext_metrics["pr_auc"]["n_resamples"]),
                },
                "f2_beta_drop": {
                    "point": float(internal_test["metrics"]["f2_beta"]["point"] - ext_metrics["f2_beta"]["point"]),
                    "ci_95": None,
                    "ci_width": None,
                    "ci_note": "not_computed_point_estimate_only",
                    "n_resamples": int(ext_metrics["f2_beta"]["n_resamples"]),
                },
                "brier_increase": {
                    "point": float(ext_metrics["brier"]["point"] - internal_test["metrics"]["brier"]["point"]),
                    "ci_95": None,
                    "ci_width": None,
                    "ci_note": "not_computed_point_estimate_only",
                    "n_resamples": int(ext_metrics["brier"]["n_resamples"]),
                },
            }

    return {
        "status": "pass",
        "schema_version": "v4.0",
        "split_metrics_ci": {**split_metrics_ci, "external": external_metrics_ci},
        "transport_drop_ci": transport_drop_ci,
        "ci_quality_summary": {
            "required_resamples": int(n_resamples),
            "split_count": int(len(split_metrics_ci)),
            "external_count": int(len(external_metrics_ci)),
        },
    }


def main() -> int:
    args = parse_args()
    fast_diagnostic_mode = bool(args.fast_diagnostic_mode)
    if args.cv_splits < 3:
        raise SystemExit("--cv-splits must be >= 3.")
    if args.beta <= 0:
        raise SystemExit("--beta must be > 0.")
    if args.primary_metric.strip().lower() != "pr_auc":
        raise SystemExit("--primary-metric must be pr_auc for this strict workflow.")
    if bool(args.external_cohort_spec) != bool(args.external_validation_report_out):
        raise SystemExit("--external-cohort-spec and --external-validation-report-out must be provided together.")
    if args.external_cohort_spec and not args.prediction_trace_out:
        raise SystemExit("--prediction-trace-out is required when --external-cohort-spec is provided.")

    policy = load_policy(args.performance_policy)
    missingness_policy = load_missingness_policy(args.missingness_policy)
    external_spec = load_external_cohort_spec(args.external_cohort_spec)
    feature_group_spec = load_feature_group_spec(args.feature_group_spec)
    fe_mode_cfg = mode_config(args.feature_engineering_mode)
    threshold_policy = policy.get("threshold_policy") if isinstance(policy, dict) else None
    clinical_floors = policy.get("clinical_floors") if isinstance(policy, dict) else None
    if isinstance(threshold_policy, dict) and isinstance(threshold_policy.get("clinical_floors"), dict):
        clinical_floors = threshold_policy.get("clinical_floors")

    beta = float(policy.get("beta", args.beta)) if isinstance(policy.get("beta"), (int, float)) else float(args.beta)
    sensitivity_floor = (
        float(clinical_floors.get("sensitivity_min"))
        if isinstance(clinical_floors, dict) and isinstance(clinical_floors.get("sensitivity_min"), (int, float))
        else float(args.sensitivity_floor)
    )
    npv_floor = (
        float(clinical_floors.get("npv_min"))
        if isinstance(clinical_floors, dict) and isinstance(clinical_floors.get("npv_min"), (int, float))
        else float(args.npv_floor)
    )
    specificity_floor = (
        float(clinical_floors.get("specificity_min"))
        if isinstance(clinical_floors, dict) and isinstance(clinical_floors.get("specificity_min"), (int, float))
        else float(args.specificity_floor)
    )
    ppv_floor = (
        float(clinical_floors.get("ppv_min"))
        if isinstance(clinical_floors, dict) and isinstance(clinical_floors.get("ppv_min"), (int, float))
        else float(args.ppv_floor)
    )
    calibration_method = str(policy.get("calibration_method", args.calibration_method)).strip().lower()
    if calibration_method not in {"sigmoid", "isotonic", "power", "beta", "none"}:
        calibration_method = str(args.calibration_method).strip().lower()
    if calibration_method not in {"sigmoid", "isotonic", "power", "beta", "none"}:
        calibration_method = "none"
    selection_data = str(args.selection_data).strip().lower()
    if selection_data not in {"valid", "cv_inner"}:
        raise SystemExit("--selection-data in this trainer must be valid/cv_inner.")
    threshold_selection_split = str(args.threshold_selection_split).strip().lower()
    if threshold_selection_split not in {"valid", "cv_inner"}:
        raise SystemExit("--threshold-selection-split in this trainer must be valid/cv_inner.")
    calibration_fit_split = threshold_selection_split
    if isinstance(threshold_policy, dict):
        token = str(threshold_policy.get("calibration_fit_split", calibration_fit_split)).strip().lower()
        if token in {"valid", "cv_inner"}:
            calibration_fit_split = token
    token_top = str(policy.get("calibration_fit_split", calibration_fit_split)).strip().lower()
    if token_top in {"valid", "cv_inner"}:
        calibration_fit_split = token_top
    if args.feature_engineering_report_out and not args.feature_group_spec:
        raise SystemExit("--feature-group-spec is required when --feature-engineering-report-out is used.")

    train_df = load_split(args.train)
    valid_df = load_split(args.valid)
    test_df = load_split(args.test)
    ignore_cols = parse_ignore_cols(args.ignore_cols, args.target_col)
    base_feature_cols = select_feature_columns(train_df, ignore_cols)
    groups, forbidden_features = normalize_feature_groups(feature_group_spec)
    grouped_features = sorted({feature for values in groups.values() for feature in values})
    if grouped_features:
        stage0_features = [f for f in base_feature_cols if f in grouped_features and f not in set(forbidden_features)]
    else:
        stage0_features = [f for f in base_feature_cols if f not in set(forbidden_features)]
    if not stage0_features:
        stage0_features = [f for f in base_feature_cols if f not in set(forbidden_features)]
    if not stage0_features:
        raise SystemExit("No usable features remain after feature-group and forbidden-feature filtering.")

    X_train_stage0, y_train = prepare_xy(train_df, stage0_features, args.target_col)
    stage1_features, stage1_report = select_features_by_filter(
        X_train_stage0,
        stage0_features,
        max_missing_ratio=float(fe_mode_cfg["max_missing_ratio"]),
        min_variance=float(fe_mode_cfg["min_variance"]),
    )
    if not stage1_features:
        stage1_features = list(stage0_features)

    stability_frequency = feature_stability_frequency(
        X_train=X_train_stage0,
        y_train=y_train,
        features=stage1_features,
        repeats=int(fe_mode_cfg["stability_repeats"]),
        seed=int(args.random_seed),
    )
    selected_features, group_selection_report = group_preselect_features(
        X_train=X_train_stage0,
        y_train=y_train,
        features=stage1_features,
        groups=groups,
        keep_ratio=float(fe_mode_cfg["group_keep_ratio"]),
        stability_frequency=stability_frequency,
    )
    if not selected_features:
        selected_features = list(stage1_features)

    X_train, y_train = prepare_xy(train_df, selected_features, args.target_col)
    X_valid, y_valid = prepare_xy(valid_df, selected_features, args.target_col)
    X_test, y_test = prepare_xy(test_df, selected_features, args.target_col)
    external_cohorts = resolve_external_cohorts(
        external_spec=external_spec,
        external_spec_path=args.external_cohort_spec,
        feature_cols=selected_features,
        default_target_col=args.target_col,
        default_patient_id_col=args.patient_id_col,
    )
    if args.external_cohort_spec and not external_cohorts:
        raise SystemExit("external_cohort_spec must provide at least one external cohort entry.")
    if len(np.unique(y_valid)) < 2:
        raise SystemExit("valid split must contain both classes for threshold/model selection.")
    imputation = resolve_imputation_plan(
        missingness_policy,
        train_rows=int(X_train.shape[0]),
        feature_count=len(selected_features),
    )
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    minority_count = int(min(positive_count, negative_count))
    majority_count = int(max(positive_count, negative_count))
    imbalance_ratio = (float(majority_count) / float(minority_count)) if minority_count > 0 else float("inf")
    effective_class_weight: Optional[str] = "balanced" if imbalance_ratio >= 1.5 else None

    candidates = build_candidates(
        args.random_seed,
        str(imputation["executed_strategy"]),
        class_weight=effective_class_weight,
    )
    candidate_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        if selection_data == "cv_inner":
            mean_score, std_score, n_folds, fold_scores = cv_score_pr_auc(
                cand["estimator"], X_train, y_train, n_splits=args.cv_splits, seed=args.random_seed
            )
        else:
            model = clone(cand["estimator"])
            model.fit(X_train, y_train)
            valid_proba = predict_proba_1(model, X_valid)
            mean_score = float(average_precision_score(y_valid, valid_proba))
            std_score = 0.0
            n_folds = 1
            fold_scores = [mean_score]
        candidate_rows.append(
            {
                "model_id": cand["model_id"],
                "family": cand["family"],
                "complexity_rank": cand["complexity_rank"],
                "selection_metrics": {
                    "pr_auc": {
                        "mean": mean_score,
                        "std": std_score,
                        "n_folds": n_folds,
                        "fold_scores": [float(x) for x in fold_scores],
                    }
                },
                "selected": False,
            }
        )

    trace = choose_model_one_se(
        [
            {
                "model_id": row["model_id"],
                "complexity_rank": row["complexity_rank"],
                "mean": row["selection_metrics"]["pr_auc"]["mean"],
                "std": row["selection_metrics"]["pr_auc"]["std"],
                "n_folds": row["selection_metrics"]["pr_auc"]["n_folds"],
            }
            for row in candidate_rows
        ]
    )
    selected_model_id = str(trace["chosen_model_id"])
    for row in candidate_rows:
        row["selected"] = bool(row["model_id"] == selected_model_id)

    estimator_map = {cand["model_id"]: cand["estimator"] for cand in candidates}
    selected_estimator = clone(estimator_map[selected_model_id])
    selected_estimator.fit(X_train, y_train)
    if threshold_selection_split == "valid":
        threshold_y = y_valid
        threshold_proba_raw = predict_proba_1(selected_estimator, X_valid)
    else:
        threshold_y = y_train
        threshold_proba_raw = cv_oof_proba(
            estimator=selected_estimator,
            X=X_train,
            y=y_train,
            n_splits=args.cv_splits,
            seed=args.random_seed,
        )
    if calibration_fit_split == "valid":
        calibration_y = y_valid
        calibration_proba_raw = predict_proba_1(selected_estimator, X_valid)
    else:
        calibration_y = y_train
        calibration_proba_raw = cv_oof_proba(
            estimator=selected_estimator,
            X=X_train,
            y=y_train,
            n_splits=args.cv_splits,
            seed=args.random_seed,
        )
    calibrator = fit_probability_calibrator(
        y_true=calibration_y,
        proba_raw=calibration_proba_raw,
        method=calibration_method,
        seed=int(args.random_seed),
    )
    threshold_proba = apply_probability_calibrator(calibrator, threshold_proba_raw)
    guard_y: Optional[np.ndarray] = None
    guard_proba: Optional[np.ndarray] = None
    if threshold_selection_split == "cv_inner":
        guard_y = y_valid
        guard_proba = apply_probability_calibrator(calibrator, predict_proba_1(selected_estimator, X_valid))
    else:
        # When threshold selection is done on valid, use train OOF predictions as
        # an internal guard split to reduce valid-only threshold overfitting.
        guard_y = y_train
        guard_proba_raw = cv_oof_proba(
            estimator=selected_estimator,
            X=X_train,
            y=y_train,
            n_splits=args.cv_splits,
            seed=args.random_seed,
        )
        guard_proba = apply_probability_calibrator(calibrator, guard_proba_raw)
    threshold_info = choose_threshold(
        y_valid=threshold_y,
        proba_valid=threshold_proba,
        beta=beta,
        sensitivity_floor=sensitivity_floor,
        npv_floor=npv_floor,
        specificity_floor=specificity_floor,
        ppv_floor=ppv_floor,
        guard_y=guard_y,
        guard_proba=guard_proba,
    )
    selected_threshold = float(threshold_info["selected_threshold"])

    # Keep evaluation on the train-fitted selected model to avoid polluting valid split metrics.
    train_proba_raw = predict_proba_1(selected_estimator, X_train)
    valid_proba_raw = predict_proba_1(selected_estimator, X_valid)
    test_proba_raw = predict_proba_1(selected_estimator, X_test)
    train_proba = apply_probability_calibrator(calibrator, train_proba_raw)
    valid_proba = apply_probability_calibrator(calibrator, valid_proba_raw)
    test_proba = apply_probability_calibrator(calibrator, test_proba_raw)

    train_metrics, train_cm = metric_panel(y_train, train_proba, selected_threshold, beta=beta)
    valid_metrics, valid_cm = metric_panel(y_valid, valid_proba, selected_threshold, beta=beta)
    test_metrics, test_cm = metric_panel(y_test, test_proba, selected_threshold, beta=beta)

    if fast_diagnostic_mode:
        ci_lo = float("nan")
        ci_hi = float("nan")
        ci_n = 0
    else:
        ci_lo, ci_hi, ci_n = bootstrap_ci_pr_auc(
            y_true=y_test,
            proba=test_proba,
            n_resamples=int(args.bootstrap_resamples),
            seed=args.random_seed,
        )

    prevalence = float(np.mean(y_train))
    baseline_proba_test = np.full(shape=y_test.shape[0], fill_value=prevalence, dtype=float)
    prevalence_baseline = {
        "roc_auc": float(roc_auc_score(y_test, baseline_proba_test)),
        "pr_auc": float(average_precision_score(y_test, baseline_proba_test)),
        "brier": float(brier_score_loss(y_test, baseline_proba_test)),
    }

    baseline_logit = Pipeline(
        steps=[
            ("imputer", build_imputer(str(imputation["executed_strategy"]), args.random_seed)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="liblinear",
                    C=1.0,
                    class_weight=effective_class_weight,
                    random_state=args.random_seed,
                ),
            ),
        ]
    )
    baseline_logit.fit(X_train, y_train)
    baseline_logit_proba_test = predict_proba_1(baseline_logit, X_test)
    logistic_baseline = {
        "roc_auc": float(roc_auc_score(y_test, baseline_logit_proba_test)),
        "pr_auc": float(average_precision_score(y_test, baseline_logit_proba_test)),
        "brier": float(brier_score_loss(y_test, baseline_logit_proba_test)),
    }

    split_fingerprints = {
        "train": {
            "path": str(Path(args.train).expanduser().resolve()),
            "sha256": sha256_file(Path(args.train).expanduser().resolve()),
            "row_count": int(X_train.shape[0]),
        },
        "valid": {
            "path": str(Path(args.valid).expanduser().resolve()),
            "sha256": sha256_file(Path(args.valid).expanduser().resolve()),
            "row_count": int(X_valid.shape[0]),
        },
        "test": {
            "path": str(Path(args.test).expanduser().resolve()),
            "sha256": sha256_file(Path(args.test).expanduser().resolve()),
            "row_count": int(X_test.shape[0]),
        },
    }

    model_selection_report = {
        "status": "pass",
        "primary_metric": "pr_auc",
        "test_used_for_model_selection": False,
        "selection_policy": {
            "primary_metric": "pr_auc",
            "selection_data": selection_data,
            "one_se_rule": True,
            "complexity_tie_breaker": "prefer_lower_complexity_rank",
            "test_used_for_model_selection": False,
        },
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "selection_trace": trace,
        "selected_model_id": selected_model_id,
        "data_fingerprints": split_fingerprints,
        "imputation": imputation,
        "feature_engineering": {
            "mode": str(args.feature_engineering_mode),
            "selected_feature_count": int(len(selected_features)),
            "selected_features": selected_features,
            "selection_scope": "cv_inner_train_only" if selection_data == "cv_inner" else "train_only",
        },
    }

    evaluation_report = {
        "model_id": selected_model_id,
        "split": "test",
        "primary_metric": "pr_auc",
        "metrics": test_metrics,
        "split_metrics": {
            "train": {"metrics": train_metrics, "confusion_matrix": train_cm},
            "valid": {"metrics": valid_metrics, "confusion_matrix": valid_cm},
            "test": {"metrics": test_metrics, "confusion_matrix": test_cm},
        },
        "threshold_selection": {
            "selection_split": threshold_selection_split,
            "strategy": "maximize_fbeta_under_floors",
            "beta": beta,
            "calibration_method": calibration_method,
            "calibration_fit_split": calibration_fit_split,
            "selected_threshold": selected_threshold,
            "constraints": {
                "sensitivity_min": sensitivity_floor,
                "npv_min": npv_floor,
                "specificity_min": specificity_floor,
                "ppv_min": ppv_floor,
            },
            "constraints_satisfied_selection_split": bool(threshold_info["constraints_satisfied_selection_split"]),
            "constraints_satisfied_guard_split": (
                bool(threshold_info["constraints_satisfied_guard_split"])
                if threshold_info["constraints_satisfied_guard_split"] is not None
                else None
            ),
            "constraints_satisfied_overall": bool(threshold_info["constraints_satisfied_overall"]),
            # Backward-compatible alias kept for older consumers.
            "constraints_satisfied": bool(threshold_info["constraints_satisfied_overall"]),
            "selected_metrics_on_selection_split": threshold_info["selected_metrics_on_valid"],
            "selected_confusion_on_selection_split": threshold_info["selected_confusion_on_valid"],
            "selected_metrics_on_valid": threshold_info["selected_metrics_on_valid"]
            if threshold_selection_split == "valid"
            else None,
            "selected_confusion_on_valid": threshold_info["selected_confusion_on_valid"]
            if threshold_selection_split == "valid"
            else None,
            "selected_metrics_on_guard_split": threshold_info["guard_metrics"]
            if threshold_selection_split == "cv_inner"
            else None,
        },
        "uncertainty": {
            "metrics": {
                "pr_auc": (
                    {
                        "method": "bootstrap",
                        "n_resamples": ci_n,
                        "ci_95": [ci_lo, ci_hi],
                    }
                    if not fast_diagnostic_mode
                    else {
                        "method": "not_computed_fast_diagnostic",
                        "n_resamples": 0,
                        "ci_95": None,
                    }
                )
            }
        },
        "baselines": {
            "prevalence_model": {"metrics": prevalence_baseline},
            "logistic_regression_baseline": {"metrics": logistic_baseline},
        },
        "feature_engineering": {
            "provenance": {
                "mode": str(args.feature_engineering_mode),
                "stage0_candidate_count": int(len(stage0_features)),
                "stage1_filtered_count": int(len(stage1_features)),
                "selected_feature_count": int(len(selected_features)),
            }
        },
        "distribution_summary": {},
        "ci_matrix_ref": None,
        "transport_ci_ref": None,
        "metadata": {
            "feature_count": len(selected_features),
            "features": selected_features,
            "beta": beta,
            "selection_data": selection_data,
            "threshold_selection_split": threshold_selection_split,
            "calibration": {
                "method": calibration_method,
                "fit_split": calibration_fit_split,
                "enabled": calibration_method != "none",
                "fitted": calibrator is not None,
                "fit_rows": int(calibration_y.shape[0]),
                "details": calibrator if isinstance(calibrator, dict) else None,
            },
            "trainer_supported_selection_data": ["valid", "cv_inner"],
            "trainer_supported_threshold_selection_splits": ["valid", "cv_inner"],
            "trainer_supported_calibration_fit_splits": ["valid", "cv_inner"],
            "evaluation_model_fit_split": "train",
            "train_rows": int(X_train.shape[0]),
            "valid_rows": int(X_valid.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "data_fingerprints": split_fingerprints,
            "imputation": imputation,
            "imbalance": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "imbalance_ratio_majority_to_minority": (
                    float(imbalance_ratio) if math.isfinite(imbalance_ratio) else None
                ),
                "effective_class_weight": effective_class_weight if effective_class_weight is not None else "none",
                "class_weight_activation_threshold": 1.5,
            },
        },
    }

    def _patient_ids_or_fallback(df: pd.DataFrame, default_prefix: str) -> List[str]:
        if args.patient_id_col in df.columns:
            return df[args.patient_id_col].astype(str).tolist()
        return [f"{default_prefix}_{i}" for i in range(int(df.shape[0]))]

    prediction_trace_frames: List[pd.DataFrame] = []
    prediction_trace_frames.append(
        build_prediction_trace_rows(
            scope="train",
            cohort_id="internal_train",
            cohort_type="",
            patient_ids=_patient_ids_or_fallback(train_df, "train"),
            y_true=y_train,
            y_score=train_proba,
            threshold=selected_threshold,
            model_id=selected_model_id,
        )
    )
    prediction_trace_frames.append(
        build_prediction_trace_rows(
            scope="valid",
            cohort_id="internal_valid",
            cohort_type="",
            patient_ids=_patient_ids_or_fallback(valid_df, "valid"),
            y_true=y_valid,
            y_score=valid_proba,
            threshold=selected_threshold,
            model_id=selected_model_id,
        )
    )
    prediction_trace_frames.append(
        build_prediction_trace_rows(
            scope="test",
            cohort_id="internal_test",
            cohort_type="",
            patient_ids=_patient_ids_or_fallback(test_df, "test"),
            y_true=y_test,
            y_score=test_proba,
            threshold=selected_threshold,
            model_id=selected_model_id,
        )
    )

    external_validation_report: Optional[Dict[str, Any]] = None
    external_rows: List[Dict[str, Any]] = []
    external_score_cache: Dict[str, np.ndarray] = {}
    for cohort in external_cohorts:
        cohort_id = str(cohort["cohort_id"])
        cohort_type = str(cohort["cohort_type"])
        X_ext = cohort["X"]
        y_ext = cohort["y"]
        if len(np.unique(y_ext)) < 2:
            raise SystemExit(f"External cohort '{cohort_id}' must contain both classes for replay metrics.")
        proba_ext_raw = predict_proba_1(selected_estimator, X_ext)
        proba_ext = apply_probability_calibrator(calibrator, proba_ext_raw)
        external_score_cache[cohort_id] = np.asarray(proba_ext, dtype=float)
        metrics_ext, cm_ext = metric_panel(y_ext, proba_ext, selected_threshold, beta=beta)
        prediction_trace_frames.append(
            build_prediction_trace_rows(
                scope="external",
                cohort_id=cohort_id,
                cohort_type=cohort_type,
                patient_ids=cohort["patient_ids"],
                y_true=y_ext,
                y_score=proba_ext,
                threshold=selected_threshold,
                model_id=selected_model_id,
            )
        )
        pr_auc_drop = float(test_metrics["pr_auc"] - metrics_ext["pr_auc"])
        f2_drop = float(test_metrics["f2_beta"] - metrics_ext["f2_beta"])
        brier_increase = float(metrics_ext["brier"] - test_metrics["brier"])
        external_rows.append(
            {
                "cohort_id": cohort_id,
                "cohort_type": cohort_type,
                "row_count": int(cohort["row_count"]),
                "positive_count": int(np.sum(y_ext == 1)),
                "selected_threshold": float(selected_threshold),
                "metrics": metrics_ext,
                "confusion_matrix": cm_ext,
                "data_fingerprint": {
                    "path": str(cohort["data_path"]),
                    "sha256": sha256_file(Path(str(cohort["data_path"]))),
                    "row_count": int(cohort["row_count"]),
                },
                "transport_gap": {
                    "pr_auc_drop_from_internal_test": pr_auc_drop,
                    "f2_beta_drop_from_internal_test": f2_drop,
                    "brier_increase_from_internal_test": brier_increase,
                },
            }
        )

    if args.external_validation_report_out:
        external_validation_report = {
            "status": "pass",
            "model_id": selected_model_id,
            "primary_metric": "pr_auc",
            "internal_test_metrics": {
                "pr_auc": float(test_metrics["pr_auc"]),
                "f2_beta": float(test_metrics["f2_beta"]),
                "brier": float(test_metrics["brier"]),
            },
            "cohort_count": int(len(external_rows)),
            "cohorts": external_rows,
            "metadata": {
                "selected_threshold": float(selected_threshold),
                "beta": float(beta),
                "external_cohort_spec": str(Path(args.external_cohort_spec).expanduser().resolve())
                if args.external_cohort_spec
                else None,
            },
        }

    group_selection_frequency: Dict[str, float] = {}
    group_blocks = group_selection_report.get("groups") if isinstance(group_selection_report, dict) else None
    if isinstance(group_blocks, dict):
        for group_name, payload in group_blocks.items():
            if not isinstance(payload, dict):
                continue
            selected = payload.get("selected_features")
            if not isinstance(selected, list) or not selected:
                group_selection_frequency[group_name] = 0.0
                continue
            freq_values = [float(stability_frequency.get(str(f), 0.0)) for f in selected if isinstance(f, str)]
            group_selection_frequency[group_name] = float(np.mean(freq_values)) if freq_values else 0.0

    feature_engineering_report = {
        "status": "pass",
        "schema_version": "v4.0",
        "mode": str(args.feature_engineering_mode),
        "selection_scope": "cv_inner_train_only" if selection_data == "cv_inner" else "train_only",
        "data_scopes_used": ["train_only", "cv_inner_train_only"] if selection_data == "cv_inner" else ["train_only"],
        "feature_groups": groups,
        "forbidden_features": forbidden_features,
        "selected_features": selected_features,
        "stability": {
            "feature_selection_frequency": {k: float(v) for k, v in stability_frequency.items()},
            "group_selection_frequency": {k: float(v) for k, v in group_selection_frequency.items()},
            "repeats": int(fe_mode_cfg["stability_repeats"]),
        },
        "filters": stage1_report,
        "group_preselection": group_selection_report,
        "reproducibility": {
            "random_seed": int(args.random_seed),
            "cv": {"n_splits": int(args.cv_splits), "selection_data": selection_data},
            "selection_thresholds": {
                "max_missing_ratio": float(fe_mode_cfg["max_missing_ratio"]),
                "min_variance": float(fe_mode_cfg["min_variance"]),
                "group_keep_ratio": float(fe_mode_cfg["group_keep_ratio"]),
            },
            "retained_feature_list": selected_features,
            "selection_scope": "cv_inner_train_only" if selection_data == "cv_inner" else "train_only",
        },
    }

    compute_distribution_summary = bool(args.distribution_report_out) or not fast_diagnostic_mode
    distribution_report_payload: Optional[Dict[str, Any]] = None
    if compute_distribution_summary:
        external_frames = [{"cohort_id": c["cohort_id"], "frame": c["frame"]} for c in external_cohorts]
        distribution_report_payload = build_distribution_report(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            external_frames=external_frames,
            target_col=args.target_col,
            feature_cols=selected_features,
        )
        evaluation_report["distribution_summary"] = {
            "split_count": int(len(distribution_report_payload.get("distribution_matrix", []))),
            "schema_version": str(distribution_report_payload.get("schema_version", "v4.0")),
        }
    else:
        evaluation_report["distribution_summary"] = {
            "split_count": 0,
            "schema_version": "v4.0",
            "skipped_fast_diagnostic_mode": True,
        }

    compute_ci_matrix = bool(args.ci_matrix_report_out) or not fast_diagnostic_mode
    ci_matrix_report_payload: Optional[Dict[str, Any]] = None
    if compute_ci_matrix:
        ci_policy_block = policy.get("ci_policy") if isinstance(policy, dict) else None
        ci_resamples = (
            int(ci_policy_block.get("n_resamples"))
            if isinstance(ci_policy_block, dict) and isinstance(ci_policy_block.get("n_resamples"), int)
            else int(args.ci_bootstrap_resamples)
        )
        ci_resamples = max(200, int(ci_resamples))
        split_ci_payloads = {
            "train": {"y_true": y_train, "y_score": train_proba, "threshold": selected_threshold},
            "valid": {"y_true": y_valid, "y_score": valid_proba, "threshold": selected_threshold},
            "test": {"y_true": y_test, "y_score": test_proba, "threshold": selected_threshold},
        }
        external_ci_payloads = [
            {
                "cohort_id": str(c["cohort_id"]),
                "cohort_type": str(c["cohort_type"]),
                "y_true": c["y"],
                "y_score": external_score_cache.get(
                    str(c["cohort_id"]),
                    apply_probability_calibrator(calibrator, predict_proba_1(selected_estimator, c["X"])),
                ),
                "threshold": selected_threshold,
            }
            for c in external_cohorts
        ]
        ci_matrix_report_payload = build_ci_matrix_report(
            split_payloads=split_ci_payloads,
            external_payloads=external_ci_payloads,
            beta=beta,
            n_resamples=ci_resamples,
            seed=int(args.random_seed),
        )

    robustness_report: Optional[Dict[str, Any]] = None
    if args.robustness_report_out:
        requested_time_slices = max(1, int(args.robustness_time_slices))
        # Avoid unstable micro-slices on small test sets; keep each slice ~15 rows when possible.
        max_reliable_slices = max(1, int(X_test.shape[0] // 15))
        time_slices = max(1, min(requested_time_slices, max_reliable_slices))
        group_count = max(1, int(args.robustness_group_count))
        time_slices_block: Dict[str, Any] = {"slice_field": "event_time", "n_slices": time_slices, "slices": []}
        patient_groups_block: Dict[str, Any] = {
            "group_method": f"sha256(patient_id)%{group_count}",
            "n_groups": group_count,
            "groups": [],
        }

        time_values = pd.to_datetime(test_df["event_time"], errors="coerce", utc=True) if "event_time" in test_df.columns else None
        if time_values is None or bool(time_values.isna().any()):
            raise SystemExit("robustness report requires parseable event_time values in test split.")
        sorted_index = np.asarray(np.argsort(time_values.to_numpy()), dtype=int)
        for slice_idx, idx_block in enumerate(np.array_split(sorted_index, time_slices), start=1):
            if idx_block.size == 0:
                continue
            y_slice = y_test[idx_block]
            proba_slice = test_proba[idx_block]
            metrics_slice = metric_panel_robust(y_slice, proba_slice, selected_threshold, beta=beta)
            times_slice = time_values.iloc[idx_block]
            time_slices_block["slices"].append(
                {
                    "slice_id": f"t{slice_idx}",
                    "n": int(idx_block.size),
                    "positive_count": int(np.sum(y_slice == 1)),
                    "start_time_utc": str(times_slice.min().isoformat()),
                    "end_time_utc": str(times_slice.max().isoformat()),
                    "metrics": metrics_slice,
                }
            )

        if "patient_id" not in test_df.columns:
            raise SystemExit("robustness report requires patient_id column in test split.")
        patient_ids = test_df["patient_id"].astype(str).tolist()
        group_indices: Dict[int, List[int]] = {k: [] for k in range(group_count)}
        for idx, pid in enumerate(patient_ids):
            group_indices[stable_group_index(pid, group_count)].append(idx)
        for group_idx in range(group_count):
            idx_list = group_indices[group_idx]
            if not idx_list:
                continue
            idx_arr = np.asarray(idx_list, dtype=int)
            y_group = y_test[idx_arr]
            proba_group = test_proba[idx_arr]
            metrics_group = metric_panel_robust(y_group, proba_group, selected_threshold, beta=beta)
            patient_groups_block["groups"].append(
                {
                    "group_id": f"g{group_idx}",
                    "n": int(idx_arr.size),
                    "positive_count": int(np.sum(y_group == 1)),
                    "metrics": metrics_group,
                }
            )

        def summarize_block(rows: List[Dict[str, Any]]) -> Dict[str, float]:
            values = [float(r["metrics"]["pr_auc"]) for r in rows if isinstance(r, dict) and isinstance(r.get("metrics"), dict)]
            if not values:
                raise SystemExit("robustness report requires non-empty per-slice/group pr_auc values.")
            minimum = float(min(values))
            maximum = float(max(values))
            return {
                "pr_auc_min": minimum,
                "pr_auc_max": maximum,
                "pr_auc_range": float(maximum - minimum),
                "pr_auc_worst_drop_from_overall": float(test_metrics["pr_auc"] - minimum),
                "n_rows": int(len(values)),
            }

        robustness_report = {
            "status": "pass",
            "primary_metric": "pr_auc",
            "model_id": selected_model_id,
            "overall_test_metrics": {
                "pr_auc": float(test_metrics["pr_auc"]),
                "f2_beta": float(test_metrics["f2_beta"]),
                "brier": float(test_metrics["brier"]),
            },
            "time_slices": time_slices_block,
            "patient_hash_groups": patient_groups_block,
            "summary": {
                "time_slices": summarize_block(time_slices_block["slices"]),
                "patient_hash_groups": summarize_block(patient_groups_block["groups"]),
            },
            "metadata": {
                "selected_threshold": float(selected_threshold),
                "beta": float(beta),
                "selection_data": selection_data,
                "threshold_selection_split": threshold_selection_split,
                "requested_time_slices": int(requested_time_slices),
                "effective_time_slices": int(time_slices),
                "group_count": int(group_count),
            },
        }

    seed_sensitivity_report: Optional[Dict[str, Any]] = None
    if args.seed_sensitivity_out:
        seed_list = parse_seed_list(args.seed_sensitivity_seeds, default_seed=args.random_seed)
        seed_results: List[Dict[str, Any]] = []
        for seed in seed_list:
            seed_candidates = build_candidates(
                seed,
                str(imputation["executed_strategy"]),
                class_weight=effective_class_weight,
            )
            seed_estimator_map = {cand["model_id"]: cand["estimator"] for cand in seed_candidates}
            if selected_model_id not in seed_estimator_map:
                raise ValueError(f"Selected model_id not found in seeded candidate map: {selected_model_id}")
            seed_estimator = clone(seed_estimator_map[selected_model_id])
            seed_estimator.fit(X_train, y_train)

            if threshold_selection_split == "valid":
                threshold_y_seed = y_valid
                threshold_proba_seed_raw = predict_proba_1(seed_estimator, X_valid)
            else:
                threshold_y_seed = y_train
                threshold_proba_seed_raw = cv_oof_proba(
                    estimator=seed_estimator,
                    X=X_train,
                    y=y_train,
                    n_splits=args.cv_splits,
                    seed=seed,
                )
            if calibration_fit_split == "valid":
                calibration_y_seed = y_valid
                calibration_proba_seed_raw = predict_proba_1(seed_estimator, X_valid)
            else:
                calibration_y_seed = y_train
                calibration_proba_seed_raw = cv_oof_proba(
                    estimator=seed_estimator,
                    X=X_train,
                    y=y_train,
                    n_splits=args.cv_splits,
                    seed=seed,
                )
            calibrator_seed = fit_probability_calibrator(
                y_true=calibration_y_seed,
                proba_raw=calibration_proba_seed_raw,
                method=calibration_method,
                seed=int(seed),
            )
            threshold_proba_seed = apply_probability_calibrator(calibrator_seed, threshold_proba_seed_raw)
            guard_y_seed: Optional[np.ndarray] = None
            guard_proba_seed: Optional[np.ndarray] = None
            if threshold_selection_split == "cv_inner":
                guard_y_seed = y_valid
                guard_proba_seed = apply_probability_calibrator(
                    calibrator_seed,
                    predict_proba_1(seed_estimator, X_valid),
                )
            threshold_info_seed = choose_threshold(
                y_valid=threshold_y_seed,
                proba_valid=threshold_proba_seed,
                beta=beta,
                sensitivity_floor=sensitivity_floor,
                npv_floor=npv_floor,
                specificity_floor=specificity_floor,
                ppv_floor=ppv_floor,
                guard_y=guard_y_seed,
                guard_proba=guard_proba_seed,
            )
            threshold_seed = float(threshold_info_seed["selected_threshold"])
            test_proba_seed_raw = predict_proba_1(seed_estimator, X_test)
            test_proba_seed = apply_probability_calibrator(calibrator_seed, test_proba_seed_raw)
            test_metrics_seed, _ = metric_panel(y_test, test_proba_seed, threshold_seed, beta=beta)
            seed_results.append(
                {
                    "seed": int(seed),
                    "selected_threshold": threshold_seed,
                    "constraints_satisfied_selection_split": bool(
                        threshold_info_seed["constraints_satisfied_selection_split"]
                    ),
                    "constraints_satisfied_guard_split": (
                        bool(threshold_info_seed["constraints_satisfied_guard_split"])
                        if threshold_info_seed["constraints_satisfied_guard_split"] is not None
                        else None
                    ),
                    "constraints_satisfied_overall": bool(threshold_info_seed["constraints_satisfied_overall"]),
                    "constraints_satisfied": bool(threshold_info_seed["constraints_satisfied_overall"]),
                    "test_metrics": {
                        "pr_auc": float(test_metrics_seed["pr_auc"]),
                        "f2_beta": float(test_metrics_seed["f2_beta"]),
                        "brier": float(test_metrics_seed["brier"]),
                    },
                }
            )

        pr_auc_values = [float(row["test_metrics"]["pr_auc"]) for row in seed_results]
        f2_values = [float(row["test_metrics"]["f2_beta"]) for row in seed_results]
        brier_values = [float(row["test_metrics"]["brier"]) for row in seed_results]
        seed_sensitivity_report = {
            "status": "pass",
            "primary_metric": "pr_auc",
            "model_id": selected_model_id,
            "selection_data": selection_data,
            "threshold_selection_split": threshold_selection_split,
            "n_seed_runs": len(seed_results),
            "seeds": [int(x) for x in seed_list],
            "per_seed_results": seed_results,
            "summary": {
                "pr_auc": summarize_seed_metric(pr_auc_values),
                "f2_beta": summarize_seed_metric(f2_values),
                "brier": summarize_seed_metric(brier_values),
            },
            "metadata": {
                "beta": beta,
                "sensitivity_floor": sensitivity_floor,
                "npv_floor": npv_floor,
                "specificity_floor": specificity_floor,
                "ppv_floor": ppv_floor,
                "train_rows": int(X_train.shape[0]),
                "valid_rows": int(X_valid.shape[0]),
                "test_rows": int(X_test.shape[0]),
            },
        }

    model_selection_out = Path(args.model_selection_report_out).expanduser().resolve()
    evaluation_out = Path(args.evaluation_report_out).expanduser().resolve()

    prediction_trace_out: Optional[Path] = None
    if args.prediction_trace_out:
        prediction_trace_out = Path(args.prediction_trace_out).expanduser().resolve()
        ensure_parent(prediction_trace_out)
        prediction_trace_df = pd.concat(prediction_trace_frames, axis=0, ignore_index=True)
        prediction_trace_df.to_csv(prediction_trace_out, index=False)

    external_validation_out: Optional[Path] = None
    if args.external_validation_report_out and external_validation_report is not None:
        external_validation_out = Path(args.external_validation_report_out).expanduser().resolve()
        write_json(external_validation_out, external_validation_report)

    feature_engineering_out: Optional[Path] = None
    if args.feature_engineering_report_out:
        feature_engineering_out = Path(args.feature_engineering_report_out).expanduser().resolve()
        write_json(feature_engineering_out, feature_engineering_report)

    distribution_out: Optional[Path] = None
    if args.distribution_report_out and distribution_report_payload is not None:
        distribution_out = Path(args.distribution_report_out).expanduser().resolve()
        write_json(distribution_out, distribution_report_payload)

    ci_matrix_out: Optional[Path] = None
    if args.ci_matrix_report_out and ci_matrix_report_payload is not None:
        ci_matrix_out = Path(args.ci_matrix_report_out).expanduser().resolve()
        write_json(ci_matrix_out, ci_matrix_report_payload)

    evaluation_report["ci_matrix_ref"] = str(ci_matrix_out) if ci_matrix_out else None
    evaluation_report["transport_ci_ref"] = "transport_drop_ci" if ci_matrix_out else None
    if isinstance(evaluation_report.get("feature_engineering"), dict):
        provenance = evaluation_report["feature_engineering"].get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            evaluation_report["feature_engineering"]["provenance"] = provenance
        provenance["feature_group_spec"] = str(Path(args.feature_group_spec).expanduser().resolve()) if args.feature_group_spec else None
        provenance["feature_engineering_report"] = str(feature_engineering_out) if feature_engineering_out else None
        provenance["selected_features"] = selected_features

    evaluation_metadata = evaluation_report.get("metadata")
    if not isinstance(evaluation_metadata, dict):
        evaluation_metadata = {}
        evaluation_report["metadata"] = evaluation_metadata
    evaluation_metadata["prediction_trace_sha256"] = (
        sha256_file(prediction_trace_out) if prediction_trace_out and prediction_trace_out.exists() else None
    )
    evaluation_metadata["external_validation_report_sha256"] = (
        sha256_file(external_validation_out)
        if external_validation_out and external_validation_out.exists()
        else None
    )
    evaluation_metadata["external_cohort_count"] = int(len(external_rows))
    evaluation_metadata["feature_engineering_report_sha256"] = (
        sha256_file(feature_engineering_out)
        if feature_engineering_out and feature_engineering_out.exists()
        else None
    )
    evaluation_metadata["distribution_report_sha256"] = (
        sha256_file(distribution_out)
        if distribution_out and distribution_out.exists()
        else None
    )
    evaluation_metadata["ci_matrix_report_sha256"] = (
        sha256_file(ci_matrix_out) if ci_matrix_out and ci_matrix_out.exists() else None
    )
    threshold_block = evaluation_report.get("threshold_selection")
    threshold_fit_split = (
        str(threshold_block.get("calibration_fit_split")).strip().lower()
        if isinstance(threshold_block, dict) and threshold_block.get("calibration_fit_split") is not None
        else ""
    )
    calibration_meta = evaluation_metadata.get("calibration")
    metadata_fit_split = (
        str(calibration_meta.get("fit_split")).strip().lower()
        if isinstance(calibration_meta, dict) and calibration_meta.get("fit_split") is not None
        else ""
    )
    if threshold_fit_split and metadata_fit_split and threshold_fit_split != metadata_fit_split:
        raise SystemExit(
            "calibration_fit_split_mismatch: threshold_selection.calibration_fit_split "
            "must equal metadata.calibration.fit_split"
        )

    write_json(model_selection_out, model_selection_report)
    write_json(evaluation_out, evaluation_report)
    if args.robustness_report_out and robustness_report is not None:
        robustness_out = Path(args.robustness_report_out).expanduser().resolve()
        write_json(robustness_out, robustness_report)
    if args.seed_sensitivity_out and seed_sensitivity_report is not None:
        seed_sensitivity_out = Path(args.seed_sensitivity_out).expanduser().resolve()
        write_json(seed_sensitivity_out, seed_sensitivity_report)

    if args.model_out:
        model_out = Path(args.model_out).expanduser().resolve()
        ensure_parent(model_out)
        joblib.dump(selected_estimator, model_out)

    if args.permutation_null_out:
        rng = np.random.default_rng(args.random_seed)
        null_path = Path(args.permutation_null_out).expanduser().resolve()
        ensure_parent(null_path)
        permutation_resamples = int(args.permutation_resamples)
        if fast_diagnostic_mode:
            permutation_resamples = min(permutation_resamples, 0)
        with null_path.open("w", encoding="utf-8") as fh:
            for _ in range(max(0, int(permutation_resamples))):
                y_perm = rng.permutation(y_test)
                score = float(average_precision_score(y_perm, test_proba))
                fh.write(f"{score:.10f}\n")

    print(f"SelectedModel: {selected_model_id}")
    print(f"PrimaryMetric(pr_auc,test): {test_metrics['pr_auc']:.6f}")
    print(f"ModelSelectionReport: {model_selection_out}")
    print(f"EvaluationReport: {evaluation_out}")
    if prediction_trace_out is not None:
        print(f"PredictionTrace: {prediction_trace_out}")
    if external_validation_out is not None:
        print(f"ExternalValidationReport: {external_validation_out}")
    if args.robustness_report_out and robustness_report is not None:
        print(f"RobustnessReport: {Path(args.robustness_report_out).expanduser().resolve()}")
    if args.seed_sensitivity_out and seed_sensitivity_report is not None:
        print(f"SeedSensitivityReport: {Path(args.seed_sensitivity_out).expanduser().resolve()}")
    if feature_engineering_out is not None:
        print(f"FeatureEngineeringReport: {feature_engineering_out}")
    if distribution_out is not None:
        print(f"DistributionReport: {distribution_out}")
    if ci_matrix_out is not None:
        print(f"CIMatrixReport: {ci_matrix_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
