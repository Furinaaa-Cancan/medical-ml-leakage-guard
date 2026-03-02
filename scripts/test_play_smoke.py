#!/usr/bin/env python3
"""Lightweight smoke checks for pixel play-mode defaults.

Run:
    python3 scripts/test_play_smoke.py
"""

from __future__ import annotations

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


def test_split_strategy_order_is_source_aware() -> None:
    print("\n=== play: split strategy order is source-aware ===")
    dl = play.split_strategy_order_for_source("download")
    assert_true(dl[0] == "stratified_grouped", "download source default strategy is stratified_grouped")
    assert_true(sorted(dl) == sorted(["grouped_temporal", "grouped_random", "stratified_grouped"]), "download strategy options complete")

    csv_src = play.split_strategy_order_for_source("csv")
    assert_true(csv_src[0] == "grouped_temporal", "csv source default strategy keeps grouped_temporal")

    demo_src = play.split_strategy_order_for_source("demo")
    assert_true(demo_src[0] == "grouped_temporal", "demo source default strategy keeps grouped_temporal")


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
    test_source_step_has_only_builtin_or_csv_paths()
    test_split_strategy_order_is_source_aware()
    test_recommended_trials_respect_search_mode_and_rows()
    test_strict_small_sample_profile_enforces_conservative_training_setup()
    test_strict_small_sample_profile_inactive_on_large_data()
    test_step_run_failure_returns_fail_sentinel()
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
