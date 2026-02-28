"""Tests for scripts/train_select_evaluate.py — model training helpers.

Covers helper functions (parse_ignore_cols, parse_seed_list, canonical_model_name,
parse_model_pool_config, load_split, prepare_xy, select_feature_columns,
build_imputer, sha256_file, sha256_text, clip01, safe_ratio, confusion_counts,
metric_panel, choose_threshold, choose_model_one_se, predict_proba_1,
_family_grid, _family_base_complexity, _candidate_complexity_rank,
resolve_device, mode_config, select_features_by_filter,
build_candidates, cv_score_pr_auc, fit_probability_calibrator,
apply_probability_calibrator), and CLI integration.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import train_select_evaluate as tse


# ── pure helpers ─────────────────────────────────────────────────────────────

class TestParseIgnoreCols:
    def test_basic(self):
        result = tse.parse_ignore_cols("patient_id,event_time", "y")
        assert "y" in result
        assert "patient_id" in result
        assert "event_time" in result

    def test_target_always_included(self):
        result = tse.parse_ignore_cols("", "target")
        assert "target" in result

    def test_dedup(self):
        result = tse.parse_ignore_cols("y,y,y", "y")
        assert result.count("y") == 1

    def test_empty(self):
        result = tse.parse_ignore_cols("", "y")
        assert result == ["y"]


class TestParseSeedList:
    def test_basic(self):
        result = tse.parse_seed_list("1,2,3", 42)
        assert result == [1, 2, 3]

    def test_empty_fallback(self):
        result = tse.parse_seed_list("", 42)
        assert result == [42]

    def test_dedup(self):
        result = tse.parse_seed_list("1,1,2", 42)
        assert result == [1, 2]

    def test_invalid_tokens(self):
        result = tse.parse_seed_list("abc,1,xyz", 42)
        assert result == [1]


class TestCanonicalModelName:
    def test_alias(self):
        assert tse.canonical_model_name("lr_l1") == "logistic_l1"
        assert tse.canonical_model_name("rf") == "random_forest_balanced"
        assert tse.canonical_model_name("xgb") == "xgboost"

    def test_passthrough(self):
        assert tse.canonical_model_name("logistic_l2") == "logistic_l2"

    def test_case_insensitive(self):
        assert tse.canonical_model_name("LR_L1") == "logistic_l1"

    def test_hyphen_to_underscore(self):
        assert tse.canonical_model_name("lr-l1") == "logistic_l1"


class TestClip01:
    def test_normal(self):
        assert tse.clip01(0.5) == 0.5

    def test_below(self):
        assert tse.clip01(-0.1) == 0.0

    def test_above(self):
        assert tse.clip01(1.5) == 1.0

    def test_nan(self):
        assert tse.clip01(float("nan")) == 0.0

    def test_inf(self):
        assert tse.clip01(float("inf")) == 0.0


class TestSafeRatio:
    def test_normal(self):
        assert tse.safe_ratio(3.0, 6.0) == 0.5

    def test_zero_denom(self):
        assert tse.safe_ratio(1.0, 0.0) == 0.0

    def test_negative_denom(self):
        assert tse.safe_ratio(1.0, -1.0) == 0.0


class TestConfusionCounts:
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        cm = tse.confusion_counts(y_true, y_pred)
        assert cm["tp"] == 1
        assert cm["fn"] == 1
        assert cm["tn"] == 1
        assert cm["fp"] == 1

    def test_all_correct(self):
        y = np.array([1, 0, 1, 0])
        cm = tse.confusion_counts(y, y)
        assert cm["tp"] == 2
        assert cm["tn"] == 2
        assert cm["fp"] == 0
        assert cm["fn"] == 0


class TestSha256:
    def test_text(self):
        h = tse.sha256_text("hello")
        assert len(h) == 64
        assert h == tse.sha256_text("hello")

    def test_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h = tse.sha256_file(f)
        assert len(h) == 64


class TestModeConfig:
    def test_strict(self):
        cfg = tse.mode_config("strict")
        assert cfg["max_missing_ratio"] == 0.60

    def test_moderate(self):
        cfg = tse.mode_config("moderate")
        assert cfg["max_missing_ratio"] == 0.70

    def test_quick(self):
        cfg = tse.mode_config("quick")
        assert cfg["max_missing_ratio"] == 0.80

    def test_default_strict(self):
        cfg = tse.mode_config("unknown")
        assert cfg["max_missing_ratio"] == 0.60


class TestResolveDevice:
    def test_cpu(self):
        assert tse.resolve_device("cpu") == "cpu"

    def test_auto(self):
        result = tse.resolve_device("auto")
        assert result in {"cpu", "gpu", "mps"}


class TestSelectFeaturesByFilter:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 0, 0, 0], "c": [1.0, np.nan, np.nan, np.nan]})
        kept, report = tse.select_features_by_filter(df, ["a", "b", "c"], max_missing_ratio=0.60, min_variance=1e-8)
        assert "a" in kept
        assert "b" not in kept  # zero variance
        assert "c" not in kept  # 75% missing > 60% threshold


class TestFamilyGrid:
    def test_logistic_l1(self):
        grid = tse._family_grid("logistic_l1")
        assert len(grid) >= 1
        assert "C" in grid[0]

    def test_random_forest(self):
        grid = tse._family_grid("random_forest_balanced")
        assert len(grid) >= 1
        assert "n_estimators" in grid[0]

    def test_unsupported(self):
        with pytest.raises(ValueError):
            tse._family_grid("nonexistent_family")


class TestFamilyBaseComplexity:
    def test_ordering(self):
        assert tse._family_base_complexity("logistic_l1") < tse._family_base_complexity("random_forest_balanced")
        assert tse._family_base_complexity("random_forest_balanced") < tse._family_base_complexity("xgboost")

    def test_unknown(self):
        assert tse._family_base_complexity("unknown") == 99


class TestCandidateComplexityRank:
    def test_logistic(self):
        rank = tse._candidate_complexity_rank("logistic_l1", {"C": 1.0})
        assert isinstance(rank, int)
        assert rank > 0


# ── data loading ─────────────────────────────────────────────────────────────

class TestLoadSplit:
    def test_valid(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        loaded = tse.load_split(str(p))
        assert loaded.shape == (2, 2)

    def test_empty(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("a,b\n")
        with pytest.raises(ValueError, match="empty"):
            tse.load_split(str(p))


class TestPrepareXY:
    def test_valid(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "y": [0, 1, 0]})
        X, y = tse.prepare_xy(df, ["f1", "f2"], "y")
        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_missing_target(self):
        df = pd.DataFrame({"f1": [1, 2]})
        with pytest.raises(ValueError, match="Missing target"):
            tse.prepare_xy(df, ["f1"], "y")

    def test_non_binary_target(self):
        df = pd.DataFrame({"f1": [1, 2, 3], "y": [0, 1, 2]})
        with pytest.raises(ValueError, match="binary"):
            tse.prepare_xy(df, ["f1"], "y")


class TestSelectFeatureColumns:
    def test_basic(self):
        df = pd.DataFrame({"patient_id": [1], "y": [0], "f1": [1.0], "f2": [2.0]})
        cols = tse.select_feature_columns(df, ["patient_id", "y"])
        assert "f1" in cols
        assert "f2" in cols
        assert "patient_id" not in cols
        assert "y" not in cols

    def test_no_features(self):
        df = pd.DataFrame({"y": [0]})
        with pytest.raises(ValueError, match="No feature columns"):
            tse.select_feature_columns(df, ["y"])


class TestBuildImputer:
    def test_median(self):
        imp = tse.build_imputer("median", 42)
        assert imp is not None

    def test_mice(self):
        imp = tse.build_imputer("mice", 42)
        assert imp is not None


# ── metric panel & threshold ─────────────────────────────────────────────────

class TestMetricPanel:
    def test_perfect(self):
        y = np.array([1, 1, 0, 0])
        proba = np.array([0.9, 0.8, 0.1, 0.2])
        metrics, cm = tse.metric_panel(y, proba, threshold=0.5, beta=2.0)
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert cm["tp"] == 2
        assert cm["tn"] == 2

    def test_all_wrong(self):
        y = np.array([1, 1, 0, 0])
        proba = np.array([0.1, 0.2, 0.9, 0.8])
        metrics, cm = tse.metric_panel(y, proba, threshold=0.5, beta=2.0)
        assert metrics["sensitivity"] == 0.0
        assert metrics["specificity"] == 0.0


class TestChooseThreshold:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 200
        y = np.concatenate([np.ones(100), np.zeros(100)]).astype(int)
        proba = np.where(y == 1, rng.uniform(0.5, 0.99, n), rng.uniform(0.01, 0.5, n))
        result = tse.choose_threshold(
            y, proba, beta=2.0,
            sensitivity_floor=0.50, npv_floor=0.50,
            specificity_floor=0.30, ppv_floor=0.30,
        )
        assert "selected_threshold" in result
        assert "selected_metrics_on_valid" in result
        assert 0.0 <= result["selected_threshold"] <= 1.0


class TestChooseModelOneSe:
    def test_basic(self):
        rows = [
            {"model_id": "a", "mean": 0.85, "std": 0.03, "n_folds": 5, "complexity_rank": 100},
            {"model_id": "b", "mean": 0.90, "std": 0.04, "n_folds": 5, "complexity_rank": 200},
            {"model_id": "c", "mean": 0.88, "std": 0.02, "n_folds": 5, "complexity_rank": 50},
        ]
        result = tse.choose_model_one_se(rows)
        assert result["best_model_id"] == "b"
        assert "chosen_model_id" in result
        assert "one_se_threshold" in result


# ── CV scoring ───────────────────────────────────────────────────────────────

class TestCvScorePrAuc:
    def test_basic(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        n = 100
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = (X["f1"] + X["f2"] > 0).astype(int).values
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ])
        mean_score, std_score, n_folds, fold_scores = tse.cv_score_pr_auc(est, X, y, n_splits=3, seed=42)
        assert 0.0 <= mean_score <= 1.0
        assert n_folds >= 2
        assert len(fold_scores) == n_folds


# ── calibration ──────────────────────────────────────────────────────────────

class TestFitProbabilityCalibrator:
    def test_none(self):
        result = tse.fit_probability_calibrator(np.array([0, 1]), np.array([0.3, 0.7]), "none", 42)
        assert result is None

    def test_sigmoid(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
        proba = np.where(y == 1, rng.uniform(0.5, 0.95, 100), rng.uniform(0.05, 0.5, 100))
        result = tse.fit_probability_calibrator(y, proba, "sigmoid", 42)
        assert result is not None

    def test_isotonic(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
        proba = np.where(y == 1, rng.uniform(0.5, 0.95, 100), rng.uniform(0.05, 0.5, 100))
        result = tse.fit_probability_calibrator(y, proba, "isotonic", 42)
        assert result is not None

    def test_power(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
        proba = np.where(y == 1, rng.uniform(0.5, 0.95, 100), rng.uniform(0.05, 0.5, 100))
        result = tse.fit_probability_calibrator(y, proba, "power", 42)
        assert result is not None
        assert result["kind"] == "power"

    def test_beta(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
        proba = np.where(y == 1, rng.uniform(0.5, 0.95, 100), rng.uniform(0.05, 0.5, 100))
        result = tse.fit_probability_calibrator(y, proba, "beta", 42)
        assert result is not None
        assert result["kind"] == "beta"

    def test_too_few_samples(self):
        result = tse.fit_probability_calibrator(np.array([0, 1]), np.array([0.3, 0.7]), "sigmoid", 42)
        assert result is None


class TestApplyProbabilityCalibrator:
    def test_none(self):
        proba = np.array([0.3, 0.7])
        result = tse.apply_probability_calibrator(None, proba)
        np.testing.assert_array_almost_equal(result, proba, decimal=5)

    def test_power(self):
        calibrator = {"kind": "power", "alpha": 1.2}
        proba = np.array([0.5, 0.8])
        result = tse.apply_probability_calibrator(calibrator, proba)
        assert result.shape == (2,)
        assert all(0 < v < 1 for v in result)

    def test_beta(self):
        calibrator = {"kind": "beta", "coef_log_p": 1.0, "coef_log_one_minus_p": -0.5, "intercept": 0.0}
        proba = np.array([0.3, 0.7])
        result = tse.apply_probability_calibrator(calibrator, proba)
        assert result.shape == (2,)


# ── model pool config ────────────────────────────────────────────────────────

class TestParseModelPoolConfig:
    def _make_args(self, **kwargs):
        defaults = {
            "model_pool": "",
            "include_optional_models": False,
            "max_trials_per_family": 1,
            "hyperparam_search": "fixed_grid",
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_default(self):
        config = tse.parse_model_pool_config({}, self._make_args())
        assert "logistic_l2" in config["model_pool"]
        assert config["search_strategy"] in {"fixed_grid", "random_subsample"}

    def test_cli_override(self):
        config = tse.parse_model_pool_config({}, self._make_args(model_pool="logistic_l1,logistic_l2"))
        assert "logistic_l1" in config["model_pool"]
        assert "logistic_l2" in config["model_pool"]

    def test_include_optional(self):
        config = tse.parse_model_pool_config({}, self._make_args(include_optional_models=True))
        pool = config["model_pool"]
        assert "xgboost" in pool or "catboost" in pool

    def test_policy_override(self):
        policy = {"model_pool": {"models": ["logistic_l1"], "max_trials_per_family": 3}}
        config = tse.parse_model_pool_config(policy, self._make_args())
        assert "logistic_l1" in config["model_pool"]
        assert config["max_trials_per_family"] == 3


# ── build candidates ─────────────────────────────────────────────────────────

class TestBuildCandidates:
    def test_basic(self):
        config = {
            "model_pool": ["logistic_l2"],
            "required_models": ["logistic_l2"],
            "cli_models": [],
            "max_trials_per_family": 1,
            "search_strategy": "fixed_grid",
            "n_jobs": 1,
        }
        candidates, meta = tse.build_candidates(
            seed=42, sampling_seed=42, imputation_strategy="median",
            class_weight="balanced", model_pool_config=config,
        )
        assert len(candidates) >= 1
        assert candidates[0]["family"] == "logistic_regression"
        assert "estimator" in candidates[0]


# ── predict_proba_1 ──────────────────────────────────────────────────────────

class TestPredictProba1:
    def test_basic(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        n = 50
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = (X["f1"] > 0).astype(int).values
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ])
        est.fit(X, y)
        proba = tse.predict_proba_1(est, X)
        assert proba.shape == (n,)
        assert all(0 <= v <= 1 for v in proba)


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIMissingFile:
    def test_missing_train(self, tmp_path):
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_select_evaluate.py"),
            "--train", str(tmp_path / "nope.csv"),
            "--valid", str(tmp_path / "nope2.csv"),
            "--test", str(tmp_path / "nope3.csv"),
            "--model-selection-report-out", str(tmp_path / "ms.json"),
            "--evaluation-report-out", str(tmp_path / "eval.json"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
        assert proc.returncode != 0
