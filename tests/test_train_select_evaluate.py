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
import subprocess
import sys
from pathlib import Path

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

    def test_cli_override_honors_explicit_selection_without_hidden_baseline(self):
        config = tse.parse_model_pool_config({}, self._make_args(model_pool="random_forest_balanced"))
        assert config["requested_models"] == ["random_forest_balanced"]
        assert config["model_pool"] == ["random_forest_balanced"]
        assert config["required_models"] == []
        assert config["auto_added_required_models"] == []

    def test_include_optional(self, monkeypatch):
        # Patch XGBClassifier to a sentinel so the test doesn't depend on
        # whether optional packages are actually installed in this environment.
        monkeypatch.setattr(tse, "XGBClassifier", object())
        config = tse.parse_model_pool_config({}, self._make_args(include_optional_models=True))
        pool = config["model_pool"]
        optional_models = {"xgboost", "catboost", "lightgbm", "tabpfn"}
        assert any(name in pool for name in optional_models)

    def test_include_optional_filters_unavailable_backends(self, monkeypatch):
        monkeypatch.setattr(tse, "XGBClassifier", object())
        monkeypatch.setattr(tse, "CatBoostClassifier", None)
        monkeypatch.setattr(tse, "LGBMClassifier", None)
        monkeypatch.setattr(tse, "TabPFNClassifier", None)
        config = tse.parse_model_pool_config({}, self._make_args(include_optional_models=True, model_pool="logistic_l2"))
        pool = config["model_pool"]
        assert "xgboost" in pool
        assert "catboost" not in pool
        assert "lightgbm" not in pool
        assert "tabpfn" not in pool

    def test_policy_override(self):
        policy = {"model_pool": {"models": ["logistic_l1"], "max_trials_per_family": 3}}
        config = tse.parse_model_pool_config(policy, self._make_args())
        assert config["requested_models"] == ["logistic_l1"]
        assert "logistic_l1" in config["model_pool"]
        assert "logistic_l2" not in config["model_pool"]
        assert config["required_models"] == []
        assert config["max_trials_per_family"] == 3

    def test_policy_required_models_are_auto_added_explicitly(self):
        policy = {
            "model_pool": {
                "models": ["random_forest_balanced"],
                "required_models": ["logistic_l2"],
            }
        }
        config = tse.parse_model_pool_config(policy, self._make_args())
        assert config["requested_models"] == ["random_forest_balanced"]
        assert config["model_pool"] == ["random_forest_balanced", "logistic_l2"]
        assert config["required_models"] == ["logistic_l2"]
        assert config["auto_added_required_models"] == ["logistic_l2"]


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


# ── TRIPOD+AI / Top-conference helper functions ──────────────────────────────


class TestCalibrationAssessment:
    def test_basic(self):
        y = np.array([0]*50 + [1]*50)
        p = np.clip(y * 0.7 + 0.15 + np.random.RandomState(42).normal(0, 0.1, 100), 0.01, 0.99)
        result = tse._calibration_assessment(y, p)
        assert result["calibration_slope"] is not None
        assert result["calibration_intercept"] is not None
        assert result["expected_observed_ratio"] is not None
        assert result["ece"] is not None
        assert 0 <= result["ece"] <= 1

    def test_single_class(self):
        result = tse._calibration_assessment(np.ones(10), np.full(10, 0.9))
        assert result["calibration_slope"] is None


class TestDecisionCurveAnalysis:
    def test_basic(self):
        y = np.array([0]*50 + [1]*50)
        p = np.clip(y + np.random.RandomState(42).normal(0, 0.3, 100), 0.01, 0.99)
        result = tse._decision_curve_analysis(y, p)
        assert "thresholds" in result
        assert len(result["thresholds"]) == 19

    def test_custom_thresholds(self):
        y = np.array([0]*20 + [1]*20)
        p = np.random.RandomState(42).uniform(0, 1, 40)
        result = tse._decision_curve_analysis(y, p, thresholds=[0.2, 0.5, 0.8])
        assert len(result["thresholds"]) == 3


class TestSampleSizeAdequacy:
    def test_adequate(self):
        result = tse._sample_size_adequacy(n_events=200, n_features=10, n_total=500)
        assert result["adequacy"] == "adequate"
        assert result["events_per_variable"] == 20.0

    def test_insufficient(self):
        result = tse._sample_size_adequacy(n_events=5, n_features=10, n_total=50)
        assert result["adequacy"] == "insufficient"

    def test_zero_features(self):
        result = tse._sample_size_adequacy(n_events=100, n_features=0, n_total=200)
        assert result["events_per_variable"] == 100.0
        assert result["adequacy"] == "adequate"


class TestMulticollinearityCheck:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [1, 3, 2, 5, 4]})
        result = tse._multicollinearity_check(df)
        assert not result.get("skipped", False)
        assert "max_vif" in result

    def test_nan_input(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        result = tse._multicollinearity_check(df)
        assert result.get("skipped") is True

    def test_too_many_features(self):
        df = pd.DataFrame(np.random.RandomState(42).normal(0, 1, (10, 60)))
        result = tse._multicollinearity_check(df)
        assert result.get("skipped") is True


class TestNRI:
    def test_basic(self):
        y = np.array([0]*50 + [1]*50)
        p_new = np.clip(y * 0.8 + 0.1, 0.01, 0.99)
        p_ref = np.full(100, 0.5)
        result = tse._net_reclassification_improvement(y, p_new, p_ref, 0.5)
        assert result["nri_total"] > 0
        assert "continuous_nri_total" in result
        assert "reclassification_table" in result

    def test_identical_models(self):
        y = np.array([0]*20 + [1]*20)
        p = np.random.RandomState(42).uniform(0, 1, 40)
        result = tse._net_reclassification_improvement(y, p, p, 0.5)
        assert result["nri_total"] == 0.0


class TestDeLong:
    def test_significant(self):
        rng = np.random.RandomState(42)
        y = np.array([0]*100 + [1]*100)
        p_good = np.clip(y * 0.8 + 0.1, 0.01, 0.99)
        p_weak = np.clip(rng.uniform(0.3, 0.7, 200), 0.01, 0.99)
        result = tse._delong_test(y, p_good, p_weak)
        assert result["p_value"] is not None
        assert result["auc_a"] > result["auc_b"]

    def test_identical(self):
        y = np.array([0]*50 + [1]*50)
        p = np.random.RandomState(42).uniform(0, 1, 100)
        result = tse._delong_test(y, p, p)
        assert result["auc_diff"] == 0.0

    def test_too_few(self):
        result = tse._delong_test(np.array([1]), np.array([0.5]), np.array([0.3]))
        assert result["p_value"] is None


class TestMcNemar:
    def test_basic(self):
        y = np.array([0]*50 + [1]*50)
        pred_a = np.array([0]*40 + [1]*10 + [1]*45 + [0]*5)
        pred_b = np.array([0]*50 + [1]*50)
        result = tse._mcnemar_test(y, pred_a, pred_b)
        assert "chi2" in result
        assert "p_value" in result

    def test_identical(self):
        y = np.array([0]*20 + [1]*20)
        pred = np.array([0]*20 + [1]*20)
        result = tse._mcnemar_test(y, pred, pred)
        assert result["p_value"] == 1.0


class TestPredictionUncertainty:
    def test_basic(self):
        p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        result = tse._prediction_uncertainty(p)
        assert 0 <= result["entropy_mean"] <= 1
        assert result["entropy_max"] > 0.9

    def test_confident(self):
        p = np.array([0.01, 0.02, 0.98, 0.99])
        result = tse._prediction_uncertainty(p)
        assert result["high_uncertainty_fraction"] == 0.0


class TestSubgroupPerformance:
    def test_basic(self):
        y = np.array([0]*100 + [1]*100)
        p = np.clip(y * 0.7 + 0.15, 0.01, 0.99)
        X = pd.DataFrame({"gender": np.random.RandomState(42).choice([0, 1], 200)})
        result = tse._subgroup_performance(y, p, 0.5, 2.0, X)
        assert result["features_analyzed"] >= 1
        assert "gender" in result["subgroups"]

    def test_no_subgroup_cols(self):
        y = np.array([0]*50 + [1]*50)
        p = np.random.RandomState(42).uniform(0, 1, 100)
        X = pd.DataFrame({"continuous": np.random.RandomState(42).normal(0, 1, 100)})
        result = tse._subgroup_performance(y, p, 0.5, 2.0, X)
        assert result["features_analyzed"] == 0


class TestInferenceBenchmark:
    def test_basic(self):
        from sklearn.linear_model import LogisticRegression
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
        y = np.array([0, 0, 1, 1, 1])
        lr = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
        result = tse._inference_benchmark(lr, X)
        assert result["inference_latency_ms_per_sample"] is not None
        assert result["model_param_count"] is not None


class TestFeatureAblation:
    def test_basic(self):
        from sklearn.linear_model import LogisticRegression
        np.random.seed(42)
        X = pd.DataFrame({"important": np.random.normal(0, 1, 100), "noise": np.random.normal(0, 0.01, 100)})
        y = (X["important"] > 0).astype(int).values
        lr = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        result = tse._feature_ablation_study(lr, X, y, p, 0.5, 2.0)
        assert result["feature_count"] == 2
        assert result["top_features"][0]["feature"] == "important"


class TestErrorAnalysis:
    def test_basic(self):
        y = np.array([0]*50 + [1]*50)
        p = np.clip(y * 0.7 + 0.15 + np.random.RandomState(42).normal(0, 0.1, 100), 0.01, 0.99)
        X = pd.DataFrame({"f1": np.random.RandomState(42).normal(0, 1, 100)})
        result = tse._error_analysis(y, p, 0.5, X)
        assert result["true_positives"] is not None
        assert "feature_characterization" in result

    def test_no_errors(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        X = pd.DataFrame({"f1": [1, 2, 3, 4]})
        result = tse._error_analysis(y, p, 0.5, X)
        fp = result["false_positives"]
        fn = result["false_negatives"]
        assert (fp is None or fp["count"] == 0) and (fn is None or fn["count"] == 0)


class TestEnvironmentVersions:
    def test_basic(self):
        result = tse._environment_versions()
        assert "python" in result
        assert "numpy" in result
        assert "platform" in result


class TestPermutationImportance:
    def test_basic(self):
        from sklearn.linear_model import LogisticRegression
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": [2, 3, 4, 5, 6, 7, 8, 9]})
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        lr = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
        result = tse._permutation_importance_report(lr, X, y, scoring="accuracy", n_repeats=3, seed=42)
        assert "top_features" in result
        assert len(result["top_features"]) <= 2


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


class TestCLIImbalanceStrategyCandidates:
    @staticmethod
    def _write_split(path: Path, n_rows: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x1 = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        x2 = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        logits = 1.3 * x1 + 0.8 * x2 - 0.1
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(0.0, 1.0, size=n_rows) < prob).astype(int)
        if int(np.sum(y == 1)) == 0:
            y[0] = 1
        if int(np.sum(y == 0)) == 0:
            y[0] = 0
        base_time = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "patient_id": [f"P{idx:05d}" for idx in range(n_rows)],
                "event_time": [(base_time + pd.Timedelta(days=int(idx))).strftime("%Y-%m-%d %H:%M") for idx in range(n_rows)],
                "y": y.astype(int),
                "f1": x1.astype(float),
                "f2": x2.astype(float),
            }
        )
        df.to_csv(path, index=False)

    def test_candidates_probe_runs_and_writes_selection(self, tmp_path):
        train_csv = tmp_path / "train.csv"
        valid_csv = tmp_path / "valid.csv"
        test_csv = tmp_path / "test.csv"
        out_ms = tmp_path / "model_selection_report.json"
        out_eval = tmp_path / "evaluation_report.json"

        self._write_split(train_csv, n_rows=160, seed=101)
        self._write_split(valid_csv, n_rows=60, seed=102)
        self._write_split(test_csv, n_rows=60, seed=103)

        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "train_select_evaluate.py"),
            "--train",
            str(train_csv),
            "--valid",
            str(valid_csv),
            "--test",
            str(test_csv),
            "--target-col",
            "y",
            "--patient-id-col",
            "patient_id",
            "--ignore-cols",
            "patient_id,event_time",
            "--selection-data",
            "cv_inner",
            "--cv-splits",
            "3",
            "--model-pool",
            "logistic_l1,logistic_l2,logistic_elasticnet",
            "--max-trials-per-family",
            "1",
            "--hyperparam-search",
            "fixed_grid",
            "--imbalance-strategy-candidates",
            "auto,random_oversample,smote,adasyn",
            "--imbalance-selection-metric",
            "pr_auc",
            "--bootstrap-resamples",
            "30",
            "--permutation-resamples",
            "20",
            "--fast-diagnostic-mode",
            "--model-selection-report-out",
            str(out_ms),
            "--evaluation-report-out",
            str(out_eval),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0, proc.stderr
        assert out_ms.exists()
        assert out_eval.exists()

        ms = json.loads(out_ms.read_text(encoding="utf-8"))
        policy = ms.get("selection_policy", {})
        probe = policy.get("imbalance_probe", [])
        assert isinstance(probe, list) and len(probe) >= 1
        assert any(str(row.get("status")) == "pass" for row in probe)

        selected_strategy = str(policy.get("imbalance_strategy", ""))
        candidate_strategies = policy.get("imbalance_strategy_candidates", [])
        assert selected_strategy
        assert selected_strategy in candidate_strategies

        evaluation = json.loads(out_eval.read_text(encoding="utf-8"))
        metadata_imbalance = evaluation.get("metadata", {}).get("imbalance", {})
        assert str(metadata_imbalance.get("selected_strategy", "")) == selected_strategy
