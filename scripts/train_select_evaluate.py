#!/usr/bin/env python3
"""
Train, select, and evaluate binary models with leakage-safe model-selection evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
    parser.add_argument("--ignore-cols", default="patient_id,event_time", help="Comma-separated non-feature columns.")
    parser.add_argument("--performance-policy", help="Optional performance policy JSON path.")
    parser.add_argument("--selection-data", default="cv_inner", help="Model selection source (valid/cv_inner/nested_cv).")
    parser.add_argument("--threshold-selection-split", default="valid", help="Split used for threshold selection.")
    parser.add_argument("--cv-splits", type=int, default=5, help="CV folds for candidate scoring.")
    parser.add_argument("--beta", type=float, default=2.0, help="Beta for F-beta threshold objective.")
    parser.add_argument("--sensitivity-floor", type=float, default=0.85, help="Minimum sensitivity for threshold choice.")
    parser.add_argument("--npv-floor", type=float, default=0.90, help="Minimum NPV for threshold choice.")
    parser.add_argument("--random-seed", type=int, default=20260225, help="Random seed.")
    parser.add_argument("--primary-metric", default="pr_auc", help="Primary optimization metric.")
    parser.add_argument("--bootstrap-resamples", type=int, default=500, help="Bootstrap samples for CI.")
    parser.add_argument("--permutation-resamples", type=int, default=300, help="Permutation samples for null metric.")
    parser.add_argument("--model-selection-report-out", required=True, help="Output model_selection_report.json.")
    parser.add_argument("--evaluation-report-out", required=True, help="Output evaluation_report.json.")
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


def parse_ignore_cols(raw: str, target_col: str) -> List[str]:
    out: List[str] = [target_col]
    for token in raw.split(","):
        key = token.strip()
        if key:
            out.append(key)
    return sorted(set(out))


def load_split(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"Split is empty: {p}")
    return df


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


def build_candidates(seed: int) -> List[Dict[str, Any]]:
    return [
        {
            "model_id": "logistic_l2",
            "family": "logistic_regression",
            "complexity_rank": 1,
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            solver="liblinear",
                            C=1.0,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "random_forest_balanced",
            "family": "random_forest",
            "complexity_rank": 2,
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=5,
                            class_weight="balanced_subsample",
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
            "complexity_rank": 3,
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_depth=6,
                            max_iter=300,
                            l2_regularization=1.0,
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


def cv_score_pr_auc(estimator: BaseEstimator, X: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int) -> Tuple[float, float, int]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: List[float] = []
    for tr_idx, va_idx in splitter.split(X, y):
        y_val = y[va_idx]
        if len(np.unique(y_val)) < 2:
            continue
        model = clone(estimator)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        proba = predict_proba_1(model, X.iloc[va_idx])
        fold_scores.append(float(average_precision_score(y_val, proba)))
    if len(fold_scores) < 2:
        raise ValueError("Insufficient valid CV folds for PR-AUC scoring.")
    arr = np.asarray(fold_scores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)), int(arr.shape[0])


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
        "accuracy": accuracy,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "f2_beta": f2,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }
    return metrics, cm


def choose_threshold(
    y_valid: np.ndarray,
    proba_valid: np.ndarray,
    beta: float,
    sensitivity_floor: float,
    npv_floor: float,
) -> Dict[str, Any]:
    quantiles = np.linspace(0.01, 0.99, 299)
    thresholds = sorted(set(float(np.quantile(proba_valid, q)) for q in quantiles) | {0.5})
    best_any: Optional[Dict[str, Any]] = None
    best_feasible: Optional[Dict[str, Any]] = None
    for threshold in thresholds:
        metrics, cm = metric_panel(y_valid, proba_valid, threshold, beta=beta)
        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
            "confusion_matrix": cm,
            "feasible": bool(metrics["sensitivity"] >= sensitivity_floor and metrics["npv"] >= npv_floor),
        }
        if best_any is None or (metrics["f2_beta"] > best_any["metrics"]["f2_beta"]):
            best_any = candidate
        if candidate["feasible"]:
            if best_feasible is None:
                best_feasible = candidate
            elif metrics["f2_beta"] > best_feasible["metrics"]["f2_beta"]:
                best_feasible = candidate
            elif metrics["f2_beta"] == best_feasible["metrics"]["f2_beta"]:
                if metrics["sensitivity"] > best_feasible["metrics"]["sensitivity"]:
                    best_feasible = candidate

    selected = best_feasible if best_feasible is not None else best_any
    if selected is None:
        raise ValueError("Unable to choose threshold.")
    return {
        "selected_threshold": float(selected["threshold"]),
        "constraints_satisfied": bool(selected["feasible"]),
        "selected_metrics_on_valid": selected["metrics"],
        "selected_confusion_on_valid": selected["confusion_matrix"],
        "sensitivity_floor": float(sensitivity_floor),
        "npv_floor": float(npv_floor),
    }


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def main() -> int:
    args = parse_args()
    if args.cv_splits < 3:
        raise SystemExit("--cv-splits must be >= 3.")
    if args.beta <= 0:
        raise SystemExit("--beta must be > 0.")
    if args.primary_metric.strip().lower() != "pr_auc":
        raise SystemExit("--primary-metric must be pr_auc for this strict workflow.")

    policy = load_policy(args.performance_policy)
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
    selection_data = str(args.selection_data).strip().lower()
    if selection_data not in {"valid", "cv_inner", "nested_cv"}:
        raise SystemExit("--selection-data must be valid/cv_inner/nested_cv.")
    threshold_selection_split = str(args.threshold_selection_split).strip().lower()
    if threshold_selection_split not in {"valid", "cv_inner", "nested_cv"}:
        raise SystemExit("--threshold-selection-split must be valid/cv_inner/nested_cv.")

    train_df = load_split(args.train)
    valid_df = load_split(args.valid)
    test_df = load_split(args.test)
    ignore_cols = parse_ignore_cols(args.ignore_cols, args.target_col)
    feature_cols = select_feature_columns(train_df, ignore_cols)
    X_train, y_train = prepare_xy(train_df, feature_cols, args.target_col)
    X_valid, y_valid = prepare_xy(valid_df, feature_cols, args.target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, args.target_col)

    candidates = build_candidates(args.random_seed)
    candidate_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        mean_score, std_score, n_folds = cv_score_pr_auc(
            cand["estimator"], X_train, y_train, n_splits=args.cv_splits, seed=args.random_seed
        )
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
    valid_proba_for_threshold = predict_proba_1(selected_estimator, X_valid)
    threshold_info = choose_threshold(
        y_valid=y_valid,
        proba_valid=valid_proba_for_threshold,
        beta=beta,
        sensitivity_floor=sensitivity_floor,
        npv_floor=npv_floor,
    )
    selected_threshold = float(threshold_info["selected_threshold"])

    # Keep evaluation on the train-fitted selected model to avoid polluting valid split metrics.
    train_proba = predict_proba_1(selected_estimator, X_train)
    valid_proba = predict_proba_1(selected_estimator, X_valid)
    test_proba = predict_proba_1(selected_estimator, X_test)

    train_metrics, train_cm = metric_panel(y_train, train_proba, selected_threshold, beta=beta)
    valid_metrics, valid_cm = metric_panel(y_valid, valid_proba, selected_threshold, beta=beta)
    test_metrics, test_cm = metric_panel(y_test, test_proba, selected_threshold, beta=beta)

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
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=4000,
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
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
            "selected_threshold": selected_threshold,
            "constraints": {
                "sensitivity_min": sensitivity_floor,
                "npv_min": npv_floor,
            },
            "constraints_satisfied": bool(threshold_info["constraints_satisfied"]),
            "selected_metrics_on_valid": threshold_info["selected_metrics_on_valid"],
            "selected_confusion_on_valid": threshold_info["selected_confusion_on_valid"],
        },
        "uncertainty": {
            "metrics": {
                "pr_auc": {
                    "method": "bootstrap",
                    "n_resamples": ci_n,
                    "ci_95": [ci_lo, ci_hi],
                }
            }
        },
        "baselines": {
            "prevalence_model": {"metrics": prevalence_baseline},
            "logistic_regression_baseline": {"metrics": logistic_baseline},
        },
        "metadata": {
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "beta": beta,
            "selection_data": selection_data,
            "threshold_selection_split": threshold_selection_split,
            "evaluation_model_fit_split": "train",
            "train_rows": int(X_train.shape[0]),
            "valid_rows": int(X_valid.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
    }

    model_selection_out = Path(args.model_selection_report_out).expanduser().resolve()
    evaluation_out = Path(args.evaluation_report_out).expanduser().resolve()
    write_json(model_selection_out, model_selection_report)
    write_json(evaluation_out, evaluation_report)

    if args.model_out:
        model_out = Path(args.model_out).expanduser().resolve()
        ensure_parent(model_out)
        joblib.dump(selected_estimator, model_out)

    if args.permutation_null_out:
        rng = np.random.default_rng(args.random_seed)
        null_path = Path(args.permutation_null_out).expanduser().resolve()
        ensure_parent(null_path)
        with null_path.open("w", encoding="utf-8") as fh:
            for _ in range(int(args.permutation_resamples)):
                y_perm = rng.permutation(y_test)
                score = float(average_precision_score(y_perm, test_proba))
                fh.write(f"{score:.10f}\n")

    print(f"SelectedModel: {selected_model_id}")
    print(f"PrimaryMetric(pr_auc,test): {test_metrics['pr_auc']:.6f}")
    print(f"ModelSelectionReport: {model_selection_out}")
    print(f"EvaluationReport: {evaluation_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
