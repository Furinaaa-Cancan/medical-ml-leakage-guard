#!/usr/bin/env python3
"""Lightweight smoke checks for pixel play-mode defaults.

Run:
    python3 scripts/test_play_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mlgg_pixel as play  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: List[str] = []


def assert_true(cond: bool, test_name: str, detail: str = "") -> None:
    if cond:
        print(f"  [{PASS}] {test_name}")
    else:
        print(f"  [{FAIL}] {test_name}" + (f": {detail}" if detail else ""))
        _failures.append(test_name)


def test_default_models_are_conservative_linear_pool() -> None:
    print("\n=== play: default model pool is conservative ===")
    expected = [0, 1, 2]
    assert_true(play.DEFAULT_MODELS == expected, "DEFAULT_MODELS index set is [0,1,2]")
    selected_names = [play.MODEL_POOL[idx][0] for idx in play.DEFAULT_MODELS]
    assert_true(
        selected_names == ["logistic_l1", "logistic_l2", "logistic_elasticnet"],
        "DEFAULT_MODELS resolve to logistic l1/l2/elasticnet",
    )


def test_readiness_reason_text_mapping_is_user_friendly() -> None:
    print("\n=== play: readiness reason code maps to user-friendly text ===")
    original_lang = play.LANG
    try:
        play.LANG = "en"
        msg_missing = play._readiness_reason_text("evaluation_report_missing")
        msg_parse = play._readiness_reason_text("evaluation_report_parse_error")
        msg_schema = play._readiness_reason_text("evaluation_report_schema_invalid")
        msg_unknown = play._readiness_reason_text("some_unknown_reason")
        assert_true("missing" in msg_missing.lower(), "missing-report reason text is user-friendly")
        assert_true("json" in msg_parse.lower(), "parse-error reason text is user-friendly")
        assert_true("core metrics" in msg_schema.lower(), "schema-invalid reason text is user-friendly")
        assert_true("unknown" in msg_unknown.lower(), "unknown reason text has fallback message")
    finally:
        play.LANG = original_lang


def test_split_strategy_order_is_source_aware() -> None:
    print("\n=== play: split strategy order is source-aware ===")
    dl = play.split_strategy_order_for_source("download")
    assert_true(dl[0] == "stratified_grouped", "download source default strategy is stratified_grouped")
    assert_true(sorted(dl) == sorted(["grouped_temporal", "stratified_grouped"]), "download strategy options complete")

    csv_src = play.split_strategy_order_for_source("csv")
    assert_true(csv_src[0] == "stratified_grouped", "csv source default strategy is stratified_grouped")
    assert_true(sorted(csv_src) == sorted(["grouped_temporal", "stratified_grouped"]), "csv strategy options complete")

    demo_src = play.split_strategy_order_for_source("demo")
    assert_true(demo_src[0] == "grouped_temporal", "demo source default strategy keeps grouped_temporal")
    assert_true(sorted(demo_src) == sorted(["grouped_temporal", "stratified_grouped"]), "demo strategy options complete")


def test_source_step_has_only_builtin_or_csv_paths() -> None:
    print("\n=== play: source step exposes only builtin dataset vs own csv ===")
    original_select = play.select
    captured = {}
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            captured["opts"] = list(opts)
            captured["descs"] = list(descs) if isinstance(descs, list) else []
            return 0

        play.select = fake_select  # type: ignore[assignment]
        state = {}
        result = play.step_source(state)
        assert_true(result is True, "step_source returns success")
        assert_true(len(captured.get("opts", [])) == 2, "step_source presents exactly two options")
        assert_true(state.get("source") == "download", "source option 0 maps to download builtin datasets")

        play.select = lambda opts, descs=None, title="", is_first=False: 1  # type: ignore[assignment]
        state2 = {}
        result2 = play.step_source(state2)
        assert_true(result2 is True, "step_source returns success for option 1")
        assert_true(state2.get("source") == "csv", "source option 1 maps to user csv")
    finally:
        play.select = original_select  # type: ignore[assignment]


def test_download_dataset_step_no_project_name_prompt() -> None:
    print("\n=== play: builtin dataset path no longer asks project name ===")
    original_select = play.select
    original_input_line = play._input_line
    called = {"count": 0}
    try:
        play.select = lambda opts, descs=None, title="", is_first=False: 0  # type: ignore[assignment]

        def fail_if_called(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("project-name prompt should not be called for builtin dataset path")

        play._input_line = fail_if_called  # type: ignore[assignment]
        state = {"source": "download"}
        result = play.step_dataset(state)
        assert_true(result is True, "step_dataset(download) completes without project-name prompt")
        assert_true(called["count"] == 0, "project-name prompt was not invoked")
        assert_true(
            str(state.get("out_dir", "")).endswith("heart_disease"),
            "download path keeps deterministic default output directory",
        )
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]


def test_download_dataset_menu_uses_stable_triplet() -> None:
    print("\n=== play: builtin dataset menu contains stable triplet only ===")
    original_select = play.select
    captured = {"opts": []}
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            captured["opts"] = list(opts)
            return 0

        play.select = fake_select  # type: ignore[assignment]
        state = {"source": "download"}
        result = play.step_dataset(state)
        assert_true(result is True, "step_dataset(download) succeeds")
        assert_true(len(captured["opts"]) == 3, "download dataset menu is restricted to 3 stable datasets")
    finally:
        play.select = original_select  # type: ignore[assignment]


def test_imbalance_step_supports_multiselect_and_metric() -> None:
    print("\n=== play: imbalance step supports multi-select with selection metric ===")
    original_multi_select = play.multi_select
    original_select = play.select
    try:
        play.multi_select = lambda *args, **kwargs: [0, 3, 6]  # auto, smote, adasyn  # type: ignore[assignment]
        play.select = lambda *args, **kwargs: 1  # choose roc_auc for strategy selection  # type: ignore[assignment]
        state = {"source": "csv"}
        result = play.step_imbalance(state)
        assert_true(result is True, "step_imbalance succeeds with multi-select")
        assert_true(
            state.get("imbalance_strategies") == ["auto", "smote", "adasyn"],
            "step_imbalance records selected strategy list",
        )
        assert_true(state.get("imbalance_strategy") == "auto", "step_imbalance keeps first strategy for compatibility")
        assert_true(
            state.get("imbalance_selection_metric") == "roc_auc",
            "step_imbalance stores strategy-selection metric",
        )
    finally:
        play.multi_select = original_multi_select  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_tuning_calibration_menu_includes_human_readable_descriptions() -> None:
    print("\n=== play: calibration menu includes brief method descriptions ===")
    original_select = play.select
    original_input_line = play._input_line
    captured = {"has_desc": False}
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("pick_tuning"):
                return 1  # random_subsample
            if title == play.t("pick_calib"):
                captured["has_desc"] = isinstance(descs, list) and len(descs) == 5 and all(bool(str(x).strip()) for x in descs)
                return 0  # none
            if title == play.t("pick_device"):
                return 1  # cpu
            return 0

        play.select = fake_select  # type: ignore[assignment]
        play._input_line = lambda *args, **kwargs: ""  # type: ignore[assignment]
        state = {"source": "csv", "_n_rows": 569}
        result = play.step_tuning(state)
        assert_true(result is True, "step_tuning succeeds")
        assert_true(captured["has_desc"], "calibration menu provides 5 non-empty descriptions")
        assert_true(state.get("calibration") == "none", "calibration selection captured")
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]


def test_tuning_optuna_input_accepts_back_token() -> None:
    print("\n=== play: tuning optuna custom input supports q/back token ===")
    original_select = play.select
    original_input_line = play._input_line
    try:
        optuna_menu_choices = ["custom", "preset"]

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("pick_tuning"):
                return 2  # optuna
            if title == play.t("pick_optuna_trials_preset"):
                mode = optuna_menu_choices.pop(0)
                if mode == "custom":
                    return len(opts) - 1
                return 0
            if title == play.t("pick_calib"):
                return 0
            if title == play.t("pick_device"):
                return 1
            return 0

        input_values = ["q"]  # q at custom optuna-trials prompt; should stay in optuna menu

        def fake_input_line(*args, **kwargs):
            return input_values.pop(0) if input_values else ""

        play.select = fake_select  # type: ignore[assignment]
        play._input_line = fake_input_line  # type: ignore[assignment]
        state = {"source": "csv", "_n_rows": 569}
        result = play.step_tuning(state)
        assert_true(result is True, "step_tuning succeeds after q/back from optuna custom prompt")
        assert_true(state.get("hyperparam_search") == "optuna", "q/back in custom prompt keeps optuna mode")
        assert_true(int(state.get("optuna_trials", 0)) == 20, "optuna preset can be chosen after returning from custom prompt")
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]


def test_tuning_max_trials_rejects_invalid_values() -> None:
    print("\n=== play: tuning max trials rejects invalid values and keeps interaction ===")
    original_select = play.select
    original_input_line = play._input_line
    original_notice = play._notice
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("pick_tuning"):
                return 1
            if title == play.t("pick_trials_preset"):
                return len(opts) - 1  # custom
            if title == play.t("pick_calib"):
                return 0
            if title == play.t("pick_device"):
                return 1
            return 0

        input_values = ["0", "abc", "12"]  # invalid, invalid, valid

        def fake_input_line(*args, **kwargs):
            return input_values.pop(0) if input_values else "12"

        play.select = fake_select  # type: ignore[assignment]
        play._input_line = fake_input_line  # type: ignore[assignment]
        play._notice = lambda *args, **kwargs: None  # type: ignore[assignment]
        state = {"source": "csv", "_n_rows": 569}
        result = play.step_tuning(state)
        assert_true(result is True, "step_tuning succeeds after invalid max_trials retries")
        assert_true(int(state.get("max_trials", 0)) == 12, "valid retry value is applied")
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]
        play._notice = original_notice  # type: ignore[assignment]


def test_tuning_trials_preset_exposes_quick_options() -> None:
    print("\n=== play: tuning max-trials uses preset options + custom ===")
    original_select = play.select
    captured = {"options": []}
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("pick_tuning"):
                return 1
            if title == play.t("pick_trials_preset"):
                captured["options"] = list(opts)
                return 0
            if title == play.t("pick_calib"):
                return 0
            if title == play.t("pick_device"):
                return 1
            return 0

        play.select = fake_select  # type: ignore[assignment]
        state = {"source": "csv", "_n_rows": 569}
        result = play.step_tuning(state)
        assert_true(result is True, "step_tuning succeeds with preset path")
        assert_true(len(captured["options"]) >= 6, "trials preset includes multiple quick options plus custom")
        assert_true(any(str(x).startswith("1") for x in captured["options"]), "trials preset includes low-try option")
        assert_true(any(str(x).startswith("50") for x in captured["options"]), "trials preset includes high-try option")
        assert_true(any("Custom" in str(x) or "\u81ea\u5b9a\u4e49" in str(x) for x in captured["options"]), "trials preset includes custom option")
    finally:
        play.select = original_select  # type: ignore[assignment]


def test_tuning_optuna_preset_exposes_quick_options() -> None:
    print("\n=== play: optuna trials uses preset options + custom ===")
    original_select = play.select
    captured = {"options": []}
    try:
        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("pick_tuning"):
                return 2
            if title == play.t("pick_optuna_trials_preset"):
                captured["options"] = list(opts)
                return 0
            if title == play.t("pick_trials_preset"):
                return 0
            if title == play.t("pick_calib"):
                return 0
            if title == play.t("pick_device"):
                return 1
            return 0

        play.select = fake_select  # type: ignore[assignment]
        state = {"source": "csv", "_n_rows": 569}
        result = play.step_tuning(state)
        assert_true(result is True, "step_tuning succeeds with optuna preset path")
        assert_true(len(captured["options"]) >= 5, "optuna preset includes multiple quick options plus custom")
        assert_true(any(str(x).startswith("20") for x in captured["options"]), "optuna preset includes 20")
        assert_true(any(str(x).startswith("50") for x in captured["options"]), "optuna preset includes 50")
        assert_true(any("Custom" in str(x) or "自定义" in str(x) for x in captured["options"]), "optuna preset includes custom option")
    finally:
        play.select = original_select  # type: ignore[assignment]


def test_advanced_custom_mode_is_fully_interactive_and_completes() -> None:
    print("\n=== play: advanced custom mode remains interactive and can finish ===")
    original_select = play.select
    original_input_line = play._input_line
    try:
        menu_sequence = [0, 1, 2, 3]  # edit ignore -> set n_jobs -> optional backends -> done

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("adv_ask"):
                return 1  # yes customize
            if title == play.t("adv_menu_title"):
                return menu_sequence.pop(0)
            if title == play.t("adv_ignore_mode_title"):
                return 1  # manual input
            if title == play.t("adv_njobs"):
                return 2  # 4 workers
            if title == play.t("adv_optional"):
                return 0  # keep optional backends in pool
            return 0

        play.select = fake_select  # type: ignore[assignment]
        play._input_line = lambda *args, **kwargs: "patient_id,event_time,site_id"  # type: ignore[assignment]
        state = {
            "source": "csv",
            "pid": "patient_id",
            "time": "event_time",
            "model_pool": "lightgbm,logistic_l2",
            "_model_labels": [play.t("m_lgbm"), play.t("m_logistic_l2")],
        }
        result = play.step_advanced(state)
        assert_true(result is True, "step_advanced custom mode completes")
        assert_true(state.get("ignore_cols") == "patient_id,event_time,site_id", "custom ignore_cols is applied")
        assert_true(int(state.get("n_jobs", 0)) == 4, "n_jobs preset selection is applied")
        assert_true(bool(state.get("include_optional_models")) is True, "optional backend flag is applied")
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]


def test_advanced_njobs_custom_rejects_invalid_values() -> None:
    print("\n=== play: advanced custom n_jobs rejects invalid values and retries ===")
    original_select = play.select
    original_input_line = play._input_line
    original_notice = play._notice
    try:
        adv_menu_sequence = [1, 1, 3]  # n_jobs -> n_jobs -> done

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("adv_ask"):
                return 1
            if title == play.t("adv_menu_title"):
                return adv_menu_sequence.pop(0)
            if title == play.t("adv_njobs"):
                return 4  # custom
            if title == play.t("adv_optional"):
                return 0
            return 0

        input_values = ["0", "4"]  # invalid then valid

        def fake_input_line(*args, **kwargs):
            return input_values.pop(0) if input_values else "4"

        play.select = fake_select  # type: ignore[assignment]
        play._input_line = fake_input_line  # type: ignore[assignment]
        play._notice = lambda *args, **kwargs: None  # type: ignore[assignment]
        state = {"source": "csv", "pid": "patient_id", "time": "event_time"}
        result = play.step_advanced(state)
        assert_true(result is True, "step_advanced completes after invalid n_jobs retry")
        assert_true(int(state.get("n_jobs", 0)) == 4, "n_jobs valid retry value is applied")
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._input_line = original_input_line  # type: ignore[assignment]
        play._notice = original_notice  # type: ignore[assignment]


def test_advanced_ignore_select_without_columns_uses_safe_defaults() -> None:
    print("\n=== play: advanced ignore editor falls back safely when columns unavailable ===")
    original_select = play.select
    original_notice = play._notice
    try:
        menu_sequence = [0, 3]  # edit ignore -> done

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("adv_ask"):
                return 1
            if title == play.t("adv_menu_title"):
                return menu_sequence.pop(0)
            if title == play.t("adv_ignore_mode_title"):
                return 0  # select from detected columns
            return 0

        play.select = fake_select  # type: ignore[assignment]
        play._notice = lambda *args, **kwargs: None  # type: ignore[assignment]
        state = {"source": "csv", "pid": "patient_id", "time": "event_time", "csv_path": "/tmp/does_not_exist.csv"}
        result = play.step_advanced(state)
        assert_true(result is True, "step_advanced completes with safe fallback")
        assert_true(
            state.get("ignore_cols") == "patient_id,event_time",
            "ignore_cols fallback keeps mandatory patient/time columns",
        )
    finally:
        play.select = original_select  # type: ignore[assignment]
        play._notice = original_notice  # type: ignore[assignment]


def test_advanced_optional_disable_removes_optional_models() -> None:
    print("\n=== play: advanced optional disable prunes optional models from model_pool ===")
    original_select = play.select
    try:
        menu_sequence = [2, 3]  # optional policy -> done

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("adv_ask"):
                return 1  # yes customize
            if title == play.t("adv_menu_title"):
                return menu_sequence.pop(0)
            if title == play.t("adv_optional"):
                return 1  # disable optional models
            return 0

        play.select = fake_select  # type: ignore[assignment]
        state = {
            "source": "csv",
            "pid": "patient_id",
            "time": "event_time",
            "model_pool": "catboost,logistic_l2",
            "_model_labels": [play.t("m_cat"), play.t("m_logistic_l2")],
            "include_optional_models": True,
        }
        result = play.step_advanced(state)
        assert_true(result is True, "step_advanced completes with optional-disable branch")
        assert_true(state.get("model_pool") == "logistic_l2", "optional model is removed from model_pool")
        assert_true(bool(state.get("include_optional_models")) is False, "include_optional_models flag is disabled")
    finally:
        play.select = original_select  # type: ignore[assignment]


def test_recommended_trials_respect_search_mode_and_rows() -> None:
    print("\n=== play: recommended max trials uses search mode + n_rows ===")
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "fixed_grid", "_n_rows": 300}) == 1,
        "fixed_grid always recommends 1",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "random_subsample", "_n_rows": 300}) == 8,
        "random search n<=500 recommends 8",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "random_subsample", "_n_rows": 1200}) == 12,
        "random search 500<n<=1500 recommends 12",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "random_subsample", "_n_rows": 5000}) == 20,
        "random search large n recommends 20",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "optuna", "_n_rows": 300}) == 20,
        "optuna n<=500 recommends 20",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "optuna", "_n_rows": 1200}) == 30,
        "optuna 500<n<=1500 recommends 30",
    )
    assert_true(
        play.recommended_max_trials({"hyperparam_search": "optuna", "_n_rows": 5000}) == 50,
        "optuna large n recommends 50",
    )


def test_strict_small_sample_profile_enforces_conservative_training_setup() -> None:
    print("\n=== play: strict small-sample profile enforces conservative setup ===")
    state = {
        "_strict_small_sample": True,
        "_strict_small_sample_max_rows": 1200,
        "_n_rows": 569,
        "model_pool": "logistic_l1,random_forest_balanced,hist_gradient_boosting_l2",
        "hyperparam_search": "optuna",
        "max_trials": 30,
        "calibration": "sigmoid",
    }
    result = play.apply_strict_small_sample_profile(state)
    assert_true(bool(result.get("active")), "strict small-sample profile is active for n<=threshold")
    assert_true(
        state.get("model_pool") == "logistic_l1",
        "strict small-sample profile filters model pool to linear-only selections",
    )
    assert_true(
        state.get("hyperparam_search") == "random_subsample",
        "strict small-sample profile disables optuna",
    )
    assert_true(
        int(state.get("max_trials", 999)) <= play.STRICT_SMALL_SAMPLE_MAX_TRIALS_CAP,
        "strict small-sample profile caps max_trials",
    )
    assert_true(
        state.get("calibration") == "power",
        "strict small-sample profile switches to conservative calibration",
    )


def test_strict_small_sample_profile_inactive_on_large_data() -> None:
    print("\n=== play: strict small-sample profile stays inactive on large data ===")
    state = {
        "_strict_small_sample": True,
        "_strict_small_sample_max_rows": 1200,
        "_n_rows": 5000,
        "model_pool": "logistic_l1,random_forest_balanced",
        "hyperparam_search": "random_subsample",
        "max_trials": 20,
        "calibration": "sigmoid",
    }
    result = play.apply_strict_small_sample_profile(state)
    assert_true(not bool(result.get("active")), "strict small-sample profile inactive for large n")
    assert_true(
        state.get("model_pool") == "logistic_l1,random_forest_balanced",
        "large-data profile does not force model-pool filtering",
    )


def test_step_run_failure_returns_fail_sentinel() -> None:
    print("\n=== play: step_run fail-closed sentinel on execution failure ===")
    original_spinner = play.run_spinner
    try:
        play.run_spinner = lambda *args, **kwargs: (2, "", "simulated failure")  # type: ignore[assignment]
        result = play.step_run({"source": "demo", "out_dir": "/tmp/mlgg_play_demo_fail"})
        assert_true(result is play.FAIL, "step_run returns FAIL sentinel when demo onboarding execution fails")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]


def test_step_run_prunes_unavailable_optional_model_backend() -> None:
    print("\n=== play: step_run prunes unavailable optional backends before train ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    original_backend_available = play.optional_backend_available
    original_select = play.select
    captured = {"train_cmd": None}
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            captured["train_cmd"] = list(cmd)
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]
        play.optional_backend_available = lambda family: False if family == "lightgbm" else True  # type: ignore[assignment]
        play.select = lambda *args, **kwargs: 1  # type: ignore[assignment]  # choose auto-downgrade in dependency resolver

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_prune_case",
            "csv_path": "/tmp/mlgg_play_prune_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "lightgbm,logistic_l2",
            "_model_labels": [play.t("m_lgbm"), play.t("m_logistic_l2")],
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run succeeds after pruning unavailable optional backend")
        assert_true(state.get("model_pool") == "logistic_l2", "state model_pool prunes unavailable lightgbm")
        train_cmd = captured["train_cmd"] or []
        joined = " ".join(str(x) for x in train_cmd)
        assert_true("--model-pool logistic_l2" in joined, "train command uses pruned model_pool")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_step_run_dependency_install_path_covers_optional_and_optuna() -> None:
    print("\n=== play: step_run can one-click install optional + optuna dependencies ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    original_backend_available = play.optional_backend_available
    original_optuna_available = play.optuna_backend_available
    original_select = play.select
    install_state = {"xgboost": False, "optuna": False}
    captured = {"train_cmd": None, "pip_cmd": None}
    try:
        def fake_spinner(cmd, label, cwd="", timeout=1800):  # type: ignore[override]
            text = " ".join(str(x) for x in cmd)
            if " -m pip install " in f" {text} ":
                captured["pip_cmd"] = list(cmd)
                install_state["xgboost"] = True
                install_state["optuna"] = True
                return (0, "", "")
            return (0, "", "")

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            captured["train_cmd"] = list(cmd)
            return (0, "", "")

        play.run_spinner = fake_spinner  # type: ignore[assignment]
        play.run_with_progress = fake_progress  # type: ignore[assignment]
        play.optional_backend_available = lambda family: install_state["xgboost"] if family == "xgboost" else True  # type: ignore[assignment]
        play.optuna_backend_available = lambda: bool(install_state["optuna"])  # type: ignore[assignment]
        play.select = lambda *args, **kwargs: 0  # type: ignore[assignment]  # choose auto-install

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_dep_install_case",
            "csv_path": "/tmp/mlgg_play_dep_install_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "xgboost,logistic_l2",
            "_model_labels": [play.t("m_xgb"), play.t("m_logistic_l2")],
            "include_optional_models": True,
            "hyperparam_search": "optuna",
            "optuna_trials": 20,
            "max_trials": 12,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run succeeds after one-click dependency install")
        assert_true(captured["pip_cmd"] is not None, "pip install command is executed")
        train_cmd = " ".join(str(x) for x in (captured["train_cmd"] or []))
        assert_true("--model-pool xgboost,logistic_l2" in train_cmd, "model pool is preserved after successful install")
        assert_true("--hyperparam-search optuna" in train_cmd, "optuna mode remains after successful install")
        assert_true("--include-optional-models" in train_cmd, "train command carries include-optional-models when enabled")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.optuna_backend_available = original_optuna_available  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_step_run_dependency_cancel_fails_closed() -> None:
    print("\n=== play: step_run dependency unresolved + cancel fails closed ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    original_backend_available = play.optional_backend_available
    original_select = play.select
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]
        play.run_with_progress = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]
        play.optional_backend_available = lambda family: False if family == "catboost" else True  # type: ignore[assignment]
        play.select = lambda *args, **kwargs: 2  # type: ignore[assignment]  # cancel

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_dep_cancel_case",
            "csv_path": "/tmp/mlgg_play_dep_cancel_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "catboost,logistic_l2",
            "_model_labels": [play.t("m_cat"), play.t("m_logistic_l2")],
            "include_optional_models": True,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
        }
        result = play.step_run(state)
        assert_true(result is play.FAIL, "step_run returns FAIL when user cancels dependency resolution")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_step_run_dependency_partial_install_then_downgrade() -> None:
    print("\n=== play: dependency partial install can downgrade and continue ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    original_backend_available = play.optional_backend_available
    original_select = play.select
    install_state = {"lightgbm": False}
    captured = {"train_cmd": None}
    try:
        def fake_spinner(cmd, label, cwd="", timeout=1800):  # type: ignore[override]
            text = " ".join(str(x) for x in cmd)
            if " -m pip install " in f" {text} ":
                pkg = str(cmd[-1])
                if pkg == "catboost":
                    return (1, "", "ERROR: No matching distribution found for catboost")
                if pkg == "lightgbm":
                    install_state["lightgbm"] = True
                    return (0, "", "")
                return (0, "", "")
            return (0, "", "")

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            captured["train_cmd"] = list(cmd)
            return (0, "", "")

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("dep_fix_title"):
                if opts and str(opts[0]) == play.t("dep_action_install"):
                    return 0  # first prompt: try install
                if opts and str(opts[0]) == play.t("dep_action_retry_failed"):
                    return 1  # partial-failure prompt: downgrade
            return 0

        play.run_spinner = fake_spinner  # type: ignore[assignment]
        play.run_with_progress = fake_progress  # type: ignore[assignment]
        play.optional_backend_available = lambda family: install_state["lightgbm"] if family == "lightgbm" else (False if family == "catboost" else True)  # type: ignore[assignment]
        play.select = fake_select  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_dep_partial_case",
            "csv_path": "/tmp/mlgg_play_dep_partial_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "catboost,lightgbm,logistic_l2",
            "_model_labels": [play.t("m_cat"), play.t("m_lgbm"), play.t("m_logistic_l2")],
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "include_optional_models": True,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run succeeds after partial install + downgrade")
        assert_true(state.get("model_pool") == "lightgbm,logistic_l2", "downgrade removes only unresolved optional backend")
        train_cmd = " ".join(str(x) for x in (captured["train_cmd"] or []))
        assert_true("--model-pool lightgbm,logistic_l2" in train_cmd, "train command keeps available optional backend")
        assert_true("--include-optional-models" in train_cmd, "train command keeps include-optional-models after partial downgrade")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_apply_dependency_downgrade_sets_fallback_when_all_optional_removed() -> None:
    print("\n=== play: dependency downgrade reports fallback when optional pool collapses ===")
    original_backend_available = play.optional_backend_available
    try:
        play.optional_backend_available = lambda family: False if family in {"catboost", "lightgbm"} else True  # type: ignore[assignment]
        state = {
            "model_pool": "catboost,lightgbm",
            "_model_labels": [play.t("m_cat"), play.t("m_lgbm")],
            "include_optional_models": True,
            "hyperparam_search": "fixed_grid",
        }
        issues = {
            "missing_optional": ["catboost", "lightgbm"],
            "optuna_missing": False,
            "has_issues": True,
        }
        result = play.apply_dependency_downgrade(state, issues)
        assert_true(result.get("kept_model_pool") == ["logistic_l2"], "downgrade falls back to logistic_l2 when optional pool empties")
        assert_true(bool(result.get("fallback_used", False)), "downgrade marks fallback_used=true when optional pool collapses")
    finally:
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]


def test_step_run_normalizes_stale_optional_flag_from_history() -> None:
    print("\n=== play: step_run normalizes stale include_optional_models flag ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    captured = {"train_cmd": None}
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            captured["train_cmd"] = list(cmd)
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "_from_history": True,  # simulate replay path where advanced step is skipped
            "out_dir": "/tmp/mlgg_play_history_optional_case",
            "csv_path": "/tmp/mlgg_play_history_optional_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": True,  # stale value from old run
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run succeeds in history replay path")
        assert_true(bool(state.get("include_optional_models")) is False, "stale optional flag is normalized to false")
        train_cmd = " ".join(str(x) for x in (captured["train_cmd"] or []))
        assert_true("--include-optional-models" not in train_cmd, "normalized run does not pass include-optional-models")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_export_cli_normalizes_stale_optional_flag() -> None:
    print("\n=== play: export cli normalizes stale optional flag ===")
    state = {
        "source": "csv",
        "out_dir": "/tmp/mlgg_export_optional_case",
        "csv_path": "/tmp/mlgg_export_optional_case_input.csv",
        "pid": "patient_id",
        "target": "y",
        "time": "event_time",
        "strategy": "stratified_grouped",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
        "test_ratio": 0.2,
        "validation_method": "holdout",
        "cv_folds": 5,
        "imbalance_strategies": ["auto"],
        "imbalance_selection_metric": "pr_auc",
        "model_pool": "logistic_l2",
        "_model_labels": [play.t("m_logistic_l2")],
        "include_optional_models": True,  # stale setting
        "hyperparam_search": "fixed_grid",
        "max_trials": 1,
        "calibration": "none",
        "device": "cpu",
        "n_jobs": 1,
    }
    play.normalize_optional_backend_state(state)
    cmd = play._export_cli(state)
    assert_true("--include-optional-models" not in cmd, "export cli omits include-optional-models after normalization")


def test_export_cli_keeps_optional_flag_when_model_pool_has_optional() -> None:
    print("\n=== play: export cli keeps optional flag for optional model pools ===")
    state = {
        "source": "csv",
        "out_dir": "/tmp/mlgg_export_optional_enabled_case",
        "csv_path": "/tmp/mlgg_export_optional_enabled_case_input.csv",
        "pid": "patient_id",
        "target": "y",
        "time": "event_time",
        "strategy": "stratified_grouped",
        "train_ratio": 0.6,
        "valid_ratio": 0.2,
        "test_ratio": 0.2,
        "validation_method": "holdout",
        "cv_folds": 5,
        "imbalance_strategies": ["auto"],
        "imbalance_selection_metric": "pr_auc",
        "model_pool": "xgboost,logistic_l2",
        "_model_labels": [play.t("m_xgb"), play.t("m_logistic_l2")],
        "include_optional_models": True,
        "hyperparam_search": "fixed_grid",
        "max_trials": 1,
        "calibration": "none",
        "device": "cpu",
        "n_jobs": 1,
    }
    play.normalize_optional_backend_state(state)
    cmd = play._export_cli(state)
    assert_true("--include-optional-models" in cmd, "export cli keeps include-optional-models for optional model pools")


def test_step_run_dependency_partial_install_retry_then_success() -> None:
    print("\n=== play: dependency partial install can retry failed package and succeed ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    original_backend_available = play.optional_backend_available
    original_select = play.select
    install_state = {"catboost": False, "lightgbm": False}
    captured = {"train_cmd": None}
    attempt = {"catboost": 0}
    try:
        def fake_spinner(cmd, label, cwd="", timeout=1800):  # type: ignore[override]
            text = " ".join(str(x) for x in cmd)
            if " -m pip install " in f" {text} ":
                pkg = str(cmd[-1])
                if pkg == "catboost":
                    attempt["catboost"] += 1
                    if attempt["catboost"] == 1:
                        return (1, "", "temporary install failure")
                    install_state["catboost"] = True
                    return (0, "", "")
                if pkg == "lightgbm":
                    install_state["lightgbm"] = True
                    return (0, "", "")
                return (0, "", "")
            return (0, "", "")

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            captured["train_cmd"] = list(cmd)
            return (0, "", "")

        def fake_select(opts, descs=None, title="", is_first=False):  # type: ignore[override]
            if title == play.t("dep_fix_title"):
                if opts and str(opts[0]) == play.t("dep_action_install"):
                    return 0  # first prompt: install all
                if opts and str(opts[0]) == play.t("dep_action_retry_failed"):
                    return 0  # retry failed package
            return 0

        play.run_spinner = fake_spinner  # type: ignore[assignment]
        play.run_with_progress = fake_progress  # type: ignore[assignment]
        play.optional_backend_available = lambda family: install_state.get(family, True) if family in {"catboost", "lightgbm"} else True  # type: ignore[assignment]
        play.select = fake_select  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_dep_retry_case",
            "csv_path": "/tmp/mlgg_play_dep_retry_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "catboost,lightgbm,logistic_l2",
            "_model_labels": [play.t("m_cat"), play.t("m_lgbm"), play.t("m_logistic_l2")],
            "include_optional_models": True,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run succeeds after retrying failed dependency package")
        assert_true(state.get("model_pool") == "catboost,lightgbm,logistic_l2", "model pool keeps optional backends after successful retry")
        train_cmd = " ".join(str(x) for x in (captured["train_cmd"] or []))
        assert_true("--model-pool catboost,lightgbm,logistic_l2" in train_cmd, "train command preserves full model pool after retry success")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.select = original_select  # type: ignore[assignment]


def test_step_run_fail_on_play_blockers_returns_fail() -> None:
    print("\n=== play: fail-on-play-blockers returns FAIL when readiness has blockers ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            out_dir = Path("/tmp/mlgg_play_fail_on_blockers_case")
            evidence = out_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            report = {
                "model_id": "logistic_l2",
                "metrics": {"pr_auc": 0.8012, "roc_auc": 0.8123, "f1": 0.7010},
                "threshold_selection": {"constraints_satisfied_overall": False},
                "overfitting_analysis": {"risk_level": "low"},
                "split_metrics": {"train": {"metrics": {}}, "valid": {"metrics": {}}, "test": {"metrics": {}}},
            }
            (evidence / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_fail_on_blockers_case",
            "csv_path": "/tmp/mlgg_play_fail_on_blockers_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": False,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "_fail_on_play_blockers": True,
        }
        result = play.step_run(state)
        assert_true(result is play.FAIL, "step_run returns FAIL when fail-on-play-blockers is enabled and blockers exist")
        blockers = list(state.get("_play_readiness_blockers", []))
        assert_true("threshold_constraints" in blockers, "threshold_constraints blocker is surfaced in state")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_step_run_fail_on_play_blockers_fails_when_readiness_unavailable() -> None:
    print("\n=== play: fail-on-play-blockers fails closed when quick-readiness is unavailable ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            out_dir = Path("/tmp/mlgg_play_readiness_unavailable_case")
            evidence = out_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            # Intentionally do not create evaluation_report.json.
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_readiness_unavailable_case",
            "csv_path": "/tmp/mlgg_play_readiness_unavailable_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": False,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "_fail_on_play_blockers": True,
        }
        result = play.step_run(state)
        assert_true(result is play.FAIL, "step_run returns FAIL when readiness report is missing and fail-on-play-blockers is enabled")
        assert_true(state.get("_play_readiness_evaluated") is False, "readiness evaluated state is false when report is missing")
        assert_true(state.get("_play_readiness_error") == "evaluation_report_missing", "missing readiness report reason is recorded")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_step_run_allows_readiness_unavailable_by_default_with_advisory() -> None:
    print("\n=== play: readiness unavailable remains non-blocking by default with advisory ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            out_dir = Path("/tmp/mlgg_play_readiness_unavailable_allowed_case")
            evidence = out_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            # Intentionally do not create evaluation_report.json.
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_readiness_unavailable_allowed_case",
            "csv_path": "/tmp/mlgg_play_readiness_unavailable_allowed_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": False,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "_fail_on_play_blockers": False,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run stays successful when readiness is unavailable and fail-on-play-blockers is disabled")
        assert_true(state.get("_play_readiness_evaluated") is False, "readiness evaluated state is false when report is missing")
        assert_true(state.get("_play_readiness_error") == "evaluation_report_missing", "missing readiness report reason is recorded")
        advisories = list(state.get("_play_readiness_advisories", []))
        assert_true("quick_readiness_unavailable" in advisories, "readiness-unavailable advisory is recorded for UI visibility")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_step_run_fail_on_play_blockers_fails_when_readiness_schema_invalid() -> None:
    print("\n=== play: fail-on-play-blockers fails closed when readiness report schema is invalid ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            out_dir = Path("/tmp/mlgg_play_readiness_invalid_schema_case")
            evidence = out_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            # Write an invalid report schema: no core metrics.
            report = {
                "model_id": "logistic_l2",
                "metrics": {},
                "threshold_selection": {},
            }
            (evidence / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_readiness_invalid_schema_case",
            "csv_path": "/tmp/mlgg_play_readiness_invalid_schema_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": False,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "_fail_on_play_blockers": True,
        }
        result = play.step_run(state)
        assert_true(result is play.FAIL, "step_run returns FAIL when readiness schema is invalid and fail-on-play-blockers is enabled")
        assert_true(state.get("_play_readiness_evaluated") is False, "readiness evaluated state is false for invalid schema")
        assert_true(state.get("_play_readiness_error") == "evaluation_report_schema_invalid", "invalid readiness report schema reason is recorded")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_step_run_allows_play_blockers_by_default() -> None:
    print("\n=== play: blockers do not fail run when fail-on-play-blockers is disabled ===")
    original_spinner = play.run_spinner
    original_progress = play.run_with_progress
    try:
        play.run_spinner = lambda *args, **kwargs: (0, "", "")  # type: ignore[assignment]

        def fake_progress(cmd, label, total=0, cwd="", timeout=3600):  # type: ignore[override]
            out_dir = Path("/tmp/mlgg_play_blockers_allowed_case")
            evidence = out_dir / "evidence"
            evidence.mkdir(parents=True, exist_ok=True)
            report = {
                "model_id": "logistic_l2",
                "metrics": {"pr_auc": 0.8012, "roc_auc": 0.8123, "f1": 0.7010},
                "threshold_selection": {"constraints_satisfied_overall": False},
                "overfitting_analysis": {"risk_level": "low"},
                "split_metrics": {"train": {"metrics": {}}, "valid": {"metrics": {}}, "test": {"metrics": {}}},
            }
            (evidence / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")
            return (0, "", "")

        play.run_with_progress = fake_progress  # type: ignore[assignment]

        state = {
            "source": "csv",
            "out_dir": "/tmp/mlgg_play_blockers_allowed_case",
            "csv_path": "/tmp/mlgg_play_blockers_allowed_case_input.csv",
            "pid": "patient_id",
            "target": "y",
            "time": "event_time",
            "strategy": "stratified_grouped",
            "train_ratio": 0.6,
            "valid_ratio": 0.2,
            "test_ratio": 0.2,
            "validation_method": "holdout",
            "cv_folds": 5,
            "imbalance_strategies": ["auto"],
            "imbalance_selection_metric": "pr_auc",
            "model_pool": "logistic_l2",
            "_model_labels": [play.t("m_logistic_l2")],
            "include_optional_models": False,
            "hyperparam_search": "fixed_grid",
            "max_trials": 1,
            "calibration": "none",
            "device": "cpu",
            "n_jobs": 1,
            "_fail_on_play_blockers": False,
        }
        result = play.step_run(state)
        assert_true(result is True, "step_run remains successful when fail-on-play-blockers is disabled")
        blockers = list(state.get("_play_readiness_blockers", []))
        assert_true("threshold_constraints" in blockers, "blockers are still recorded for UI visibility")
    finally:
        play.run_spinner = original_spinner  # type: ignore[assignment]
        play.run_with_progress = original_progress  # type: ignore[assignment]


def test_collect_runtime_dependency_issues_covers_all_optional_backends() -> None:
    print("\n=== play: runtime dependency issue collector covers all optional backends ===")
    original_backend_available = play.optional_backend_available
    original_optuna_available = play.optuna_backend_available
    try:
        missing_set = {"xgboost", "catboost", "lightgbm", "tabpfn"}
        play.optional_backend_available = lambda family: False if family in missing_set else True  # type: ignore[assignment]
        play.optuna_backend_available = lambda: False  # type: ignore[assignment]
        state = {
            "model_pool": "xgboost,catboost,lightgbm,tabpfn,logistic_l2",
            "hyperparam_search": "optuna",
        }
        issues = play.collect_runtime_dependency_issues(state)
        missing = set(str(x) for x in issues.get("missing_optional", []))
        assert_true(missing == missing_set, "all optional model backends are detected when unavailable")
        assert_true(bool(issues.get("optuna_missing")), "optuna missing flag is detected")
        assert_true(bool(issues.get("has_issues")), "has_issues is true when any dependency is missing")
    finally:
        play.optional_backend_available = original_backend_available  # type: ignore[assignment]
        play.optuna_backend_available = original_optuna_available  # type: ignore[assignment]


def test_wizard_exits_nonzero_when_run_step_fails() -> None:
    print("\n=== play: wizard returns exit code 2 when execution step fails ===")
    step_names = [
        "step_lang",
        "step_source",
        "step_dataset",
        "step_config",
        "step_split",
        "step_imbalance",
        "step_models",
        "step_tuning",
        "step_advanced",
        "step_confirm",
        "step_run",
    ]
    originals = {name: getattr(play, name) for name in step_names}
    try:
        for name in step_names[:-1]:
            setattr(play, name, lambda state: True)
        setattr(play, "step_run", lambda state: play.FAIL)
        rc = play.wizard(force_lang="en", dry_run=True)
        assert_true(rc == 2, "wizard returns 2 when step_run reports FAIL")
    finally:
        for name in step_names:
            setattr(play, name, originals[name])


def main() -> int:
    print("Running play smoke tests...")
    test_default_models_are_conservative_linear_pool()
    test_readiness_reason_text_mapping_is_user_friendly()
    test_source_step_has_only_builtin_or_csv_paths()
    test_download_dataset_step_no_project_name_prompt()
    test_download_dataset_menu_uses_stable_triplet()
    test_imbalance_step_supports_multiselect_and_metric()
    test_tuning_calibration_menu_includes_human_readable_descriptions()
    test_tuning_optuna_input_accepts_back_token()
    test_tuning_max_trials_rejects_invalid_values()
    test_tuning_trials_preset_exposes_quick_options()
    test_tuning_optuna_preset_exposes_quick_options()
    test_advanced_custom_mode_is_fully_interactive_and_completes()
    test_advanced_njobs_custom_rejects_invalid_values()
    test_advanced_ignore_select_without_columns_uses_safe_defaults()
    test_advanced_optional_disable_removes_optional_models()
    test_split_strategy_order_is_source_aware()
    test_recommended_trials_respect_search_mode_and_rows()
    test_strict_small_sample_profile_enforces_conservative_training_setup()
    test_strict_small_sample_profile_inactive_on_large_data()
    test_step_run_failure_returns_fail_sentinel()
    test_step_run_prunes_unavailable_optional_model_backend()
    test_step_run_dependency_install_path_covers_optional_and_optuna()
    test_step_run_dependency_cancel_fails_closed()
    test_step_run_dependency_partial_install_then_downgrade()
    test_apply_dependency_downgrade_sets_fallback_when_all_optional_removed()
    test_step_run_normalizes_stale_optional_flag_from_history()
    test_export_cli_normalizes_stale_optional_flag()
    test_export_cli_keeps_optional_flag_when_model_pool_has_optional()
    test_step_run_dependency_partial_install_retry_then_success()
    test_step_run_fail_on_play_blockers_returns_fail()
    test_step_run_fail_on_play_blockers_fails_when_readiness_unavailable()
    test_step_run_allows_readiness_unavailable_by_default_with_advisory()
    test_step_run_fail_on_play_blockers_fails_when_readiness_schema_invalid()
    test_step_run_allows_play_blockers_by_default()
    test_collect_runtime_dependency_issues_covers_all_optional_backends()
    test_wizard_exits_nonzero_when_run_step_fails()

    print(f"\n{'='*50}")
    if _failures:
        print(f"\033[31mFAILED {len(_failures)} test(s):\033[0m")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("\033[32mAll play smoke tests passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
