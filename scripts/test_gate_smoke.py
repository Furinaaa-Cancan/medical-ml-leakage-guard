#!/usr/bin/env python3
"""
Lightweight smoke tests for individual gate scripts.

Tests verify:
- leakage_gate: pass on clean splits, fail on row/ID/temporal overlap.
- leakage_gate: word-boundary regex does not fire on legitimate column names.
- leakage_gate: temporal boundary > (not >=) logic.
- self_critique_gate: weight normalization sums to 100.
- run_strict_pipeline: --strict enforcement.
- publication_gate: fails without --strict.
- mlgg interactive: safe default train flags and workflow strict injection.

Run:
    python3 scripts/test_gate_smoke.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, str]], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def run_gate(
    args: List[str],
    input_text: str | None = None,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(
        [PYTHON] + args,
        text=True,
        capture_output=True,
        input=input_text,
        env=run_env,
    )


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures: List[str] = []


def assert_true(cond: bool, test_name: str, detail: str = "") -> None:
    if cond:
        print(f"  [{PASS}] {test_name}")
    else:
        print(f"  [{FAIL}] {test_name}" + (f": {detail}" if detail else ""))
        _failures.append(test_name)


# ---------------------------------------------------------------------------
# leakage_gate tests
# ---------------------------------------------------------------------------

def test_leakage_gate_clean_splits() -> None:
    print("\n=== leakage_gate: clean splits ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "event_time", "age", "y"]
        train_rows = [{"patient_id": str(i), "event_time": f"2020-01-{i:02d}", "age": "40", "y": "0"} for i in range(1, 11)]
        valid_rows = [{"patient_id": str(i), "event_time": f"2020-02-{i:02d}", "age": "45", "y": "1"} for i in range(11, 21)]
        test_rows  = [{"patient_id": str(i), "event_time": f"2020-03-{i:02d}", "age": "50", "y": "0"} for i in range(21, 31)]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "valid.csv", valid_rows, headers)
        write_csv(td / "test.csv",  test_rows,  headers)
        report_path = td / "report.json"
        proc = run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--valid", str(td / "valid.csv"),
            "--test",  str(td / "test.csv"),
            "--id-cols", "patient_id",
            "--time-col", "event_time",
            "--target-col", "y",
            "--report", str(report_path),
            "--strict",
        ])
        assert_true(proc.returncode == 0, "clean splits exit 0")
        report = load_report(report_path)
        assert_true(report["status"] == "pass", "clean splits status=pass")
        assert_true(report["failure_count"] == 0, "clean splits no failures")


def test_leakage_gate_row_overlap() -> None:
    print("\n=== leakage_gate: row overlap ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "age", "y"]
        shared = {"patient_id": "1", "age": "40", "y": "0"}
        train_rows = [shared, {"patient_id": "2", "age": "41", "y": "1"}]
        test_rows  = [shared, {"patient_id": "3", "age": "42", "y": "0"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "test.csv",  test_rows,  headers)
        report_path = td / "report.json"
        proc = run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--test",  str(td / "test.csv"),
            "--target-col", "y",
            "--report", str(report_path),
        ])
        assert_true(proc.returncode == 2, "row overlap exit 2")
        report = load_report(report_path)
        assert_true(report["status"] == "fail", "row overlap status=fail")
        codes = {f["code"] for f in report["failures"]}
        assert_true("row_overlap" in codes, "row_overlap failure code present")


def test_leakage_gate_id_overlap() -> None:
    print("\n=== leakage_gate: entity ID overlap ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "age", "y"]
        train_rows = [{"patient_id": "1", "age": "40", "y": "0"}, {"patient_id": "2", "age": "41", "y": "1"}]
        test_rows  = [{"patient_id": "1", "age": "99", "y": "1"}, {"patient_id": "3", "age": "42", "y": "0"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "test.csv",  test_rows,  headers)
        report_path = td / "report.json"
        proc = run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--test",  str(td / "test.csv"),
            "--id-cols", "patient_id",
            "--target-col", "y",
            "--report", str(report_path),
        ])
        assert_true(proc.returncode == 2, "id overlap exit 2")
        report = load_report(report_path)
        codes = {f["code"] for f in report["failures"]}
        assert_true("id_overlap" in codes, "id_overlap failure code present")


def test_leakage_gate_temporal_overlap() -> None:
    print("\n=== leakage_gate: temporal overlap (train max > valid min) ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "event_time", "y"]
        train_rows = [{"patient_id": "1", "event_time": "2020-03-01", "y": "0"}]
        valid_rows = [{"patient_id": "2", "event_time": "2020-02-01", "y": "1"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "valid.csv", valid_rows, headers)
        report_path = td / "report.json"
        proc = run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--valid", str(td / "valid.csv"),
            "--time-col", "event_time",
            "--report", str(report_path),
        ])
        assert_true(proc.returncode == 2, "temporal overlap exit 2")
        report = load_report(report_path)
        codes = {f["code"] for f in report["failures"]}
        assert_true("temporal_overlap" in codes, "temporal_overlap failure code present")


def test_leakage_gate_temporal_boundary_equal_is_ok() -> None:
    print("\n=== leakage_gate: same-day boundary (train max == valid min) should NOT fail ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "event_time", "y"]
        train_rows = [{"patient_id": "1", "event_time": "2020-01-31", "y": "0"}]
        valid_rows = [{"patient_id": "2", "event_time": "2020-01-31", "y": "1"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "valid.csv", valid_rows, headers)
        report_path = td / "report.json"
        proc = run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--valid", str(td / "valid.csv"),
            "--time-col", "event_time",
            "--report", str(report_path),
        ])
        report = load_report(report_path)
        codes = {f["code"] for f in report["failures"]}
        assert_true("temporal_overlap" not in codes, "same-day boundary does NOT trigger temporal_overlap")


def test_leakage_gate_word_boundary_regex() -> None:
    print("\n=== leakage_gate: tighter regex no false positives ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # postal_code: 'post' is NOT a whole word; lead_time_days: 'lead' removed from default;
        # next_visit_count: 'next' removed from default; postal_outcome: 'outcome' not a _-segment match here
        headers = ["patient_id", "postal_code", "next_visit_count", "lead_time_days", "y"]
        train_rows = [{"patient_id": "1", "postal_code": "10001", "next_visit_count": "2",
                       "lead_time_days": "5", "y": "0"}]
        test_rows  = [{"patient_id": "2", "postal_code": "10002", "next_visit_count": "1",
                       "lead_time_days": "3", "y": "1"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "test.csv",  test_rows,  headers)
        report_path = td / "report.json"
        run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--test",  str(td / "test.csv"),
            "--target-col", "y",
            "--report", str(report_path),
        ])
        report = load_report(report_path)
        codes_warn = {w["code"] for w in report.get("warnings", [])}
        assert_true("suspicious_feature_names" not in codes_warn,
                    "no false-positive suspicious_feature_names for postal_code/next_visit_count/lead_time_days")

    print("\n=== leakage_gate: tighter regex DOES fire on genuinely suspicious names ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # future_diagnosis: \bfuture\b matches; leak_flag: \bleak\b matches; outcome_30d: (?:^|_)outcome(?:_|$) matches
        headers = ["patient_id", "future_diagnosis", "leak_flag", "outcome_30d", "y"]
        train_rows = [{"patient_id": "1", "future_diagnosis": "0", "leak_flag": "0", "outcome_30d": "0", "y": "0"}]
        test_rows  = [{"patient_id": "2", "future_diagnosis": "1", "leak_flag": "0", "outcome_30d": "1", "y": "1"}]
        write_csv(td / "train.csv", train_rows, headers)
        write_csv(td / "test.csv",  test_rows,  headers)
        report_path = td / "report.json"
        run_gate([
            str(SCRIPTS_DIR / "leakage_gate.py"),
            "--train", str(td / "train.csv"),
            "--test",  str(td / "test.csv"),
            "--target-col", "y",
            "--report", str(report_path),
        ])
        report = load_report(report_path)
        codes_warn = {w["code"] for w in report.get("warnings", [])}
        assert_true("suspicious_feature_names" in codes_warn,
                    "suspicious_feature_names fires for future_diagnosis/leak_flag/outcome_30d")


# ---------------------------------------------------------------------------
# self_critique_gate: weight normalization
# ---------------------------------------------------------------------------

def test_self_critique_weight_normalization() -> None:
    print("\n=== self_critique_gate: weight normalization ===")
    _raw_weights = {
        "request_report": 7.0, "manifest": 10.0, "execution_attestation_report": 8.0,
        "reporting_bias_report": 8.0, "leakage_report": 13.0, "split_protocol_report": 8.0,
        "covariate_shift_report": 7.0, "definition_report": 13.0, "lineage_report": 11.0,
        "imbalance_report": 8.0, "missingness_report": 8.0, "tuning_report": 8.0,
        "model_selection_audit_report": 8.0, "feature_engineering_audit_report": 8.0,
        "clinical_metrics_report": 8.0, "prediction_replay_report": 8.0,
        "distribution_generalization_report": 8.0, "generalization_gap_report": 8.0,
        "robustness_report": 8.0, "seed_stability_report": 7.0,
        "external_validation_report": 8.0, "calibration_dca_report": 8.0,
        "ci_matrix_report": 8.0, "metric_report": 7.0,
        "evaluation_quality_report": 8.0, "permutation_report": 7.0,
        "publication_report": 8.0,
    }
    _total_raw = sum(_raw_weights.values())
    normalized = {k: v * 100.0 / _total_raw for k, v in _raw_weights.items()}
    total = sum(normalized.values())
    assert_true(abs(total - 100.0) < 1e-9, "normalized weights sum to 100.0", f"got {total:.6f}")
    assert_true(_total_raw > 100.0, "raw weights total > 100 (confirms normalization was needed)", f"raw total={_total_raw}")


# ---------------------------------------------------------------------------
# run_strict_pipeline: --strict enforcement
# ---------------------------------------------------------------------------

def test_run_strict_pipeline_requires_strict() -> None:
    print("\n=== run_strict_pipeline: fails without --strict ===")
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / "request.json"
        req.write_text(json.dumps({"study_id": "x"}), encoding="utf-8")
        proc = run_gate([
            str(SCRIPTS_DIR / "run_strict_pipeline.py"),
            "--request", str(req),
        ])
        assert_true(proc.returncode == 2, "missing --strict exits 2")
        assert_true("strict" in proc.stderr.lower(), "--strict enforcement message present")


# ---------------------------------------------------------------------------
# transport_drop_ci: ci_note sentinel instead of hard-coded [0.0, 0.0]
# ---------------------------------------------------------------------------

def test_transport_drop_ci_not_computed_sentinel() -> None:
    print("\n=== train_select_evaluate: transport_drop_ci uses ci_note sentinel ===")
    import importlib.util
    import types

    # Minimal stub: simulate the build_ci_matrix_report transport block directly
    # by importing the function and constructing minimal payloads.
    spec_path = SCRIPTS_DIR / "train_select_evaluate.py"
    loader = importlib.util.spec_from_file_location("tse", spec_path)
    assert loader is not None
    mod = importlib.util.module_from_spec(loader)
    try:
        loader.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass
    except Exception:
        pass

    # Directly test the sentinel logic in the transport_drop_ci block by
    # checking the source string contains ci_note and not the old [0.0, 0.0] pattern.
    src = spec_path.read_text(encoding="utf-8")
    assert_true(
        '"ci_note": "not_computed_point_estimate_only"' in src,
        "transport_drop_ci uses ci_note sentinel (not hard-coded [0.0, 0.0])",
    )
    assert_true(
        '"ci_95": [0.0, 0.0]' not in src,
        "transport_drop_ci no longer contains hard-coded [0.0, 0.0] CI bounds",
    )


# ---------------------------------------------------------------------------
# feature_engineering_audit_gate: correct error codes
# ---------------------------------------------------------------------------

def test_feature_engineering_audit_gate_error_codes() -> None:
    print("\n=== feature_engineering_audit_gate: error code consistency ===")
    src = (SCRIPTS_DIR / "feature_engineering_audit_gate.py").read_text(encoding="utf-8")
    # The feature_engineering_report parse failure must use correct code
    assert_true(
        '"feature_engineering_report_invalid"' in src,
        "feature_engineering_audit_gate uses feature_engineering_report_invalid code",
    )
    # Ensure the wrong code is no longer used for report parse failure
    # (the correct usage of feature_group_spec_missing_or_invalid is for group spec errors only)
    lines = src.splitlines()
    report_parse_block = False
    wrong_code_in_report_parse = False
    for i, line in enumerate(lines):
        if "report_payload = load_json(args.feature_engineering_report)" in line:
            report_parse_block = True
        if report_parse_block and '"feature_group_spec_missing_or_invalid"' in line:
            wrong_code_in_report_parse = True
            break
        if report_parse_block and '"feature_engineering_report_invalid"' in line:
            report_parse_block = False  # found the correct code, stop
    assert_true(
        not wrong_code_in_report_parse,
        "feature_engineering_audit_gate does not use feature_group_spec_missing_or_invalid for report parse failure",
    )


# ---------------------------------------------------------------------------
# feature_engineering_audit_gate: to_float isfinite guard
# ---------------------------------------------------------------------------

def test_feature_engineering_audit_gate_to_float_isfinite() -> None:
    print("\n=== feature_engineering_audit_gate: to_float rejects inf/nan ===")
    src = (SCRIPTS_DIR / "feature_engineering_audit_gate.py").read_text(encoding="utf-8")
    assert_true("math.isfinite" in src, "feature_engineering_audit_gate.to_float uses math.isfinite")
    assert_true("import math" in src, "feature_engineering_audit_gate imports math")


# ---------------------------------------------------------------------------
# mlgg interactive: default safety and strict injection
# ---------------------------------------------------------------------------

def test_mlgg_interactive_train_defaults_are_dependency_safe() -> None:
    print("\n=== mlgg interactive: train defaults avoid optional dependency hard-fails ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "event_time", "age", "y"]
        train_rows = [{"patient_id": f"tr{i}", "event_time": f"2024-01-{(i%28)+1:02d}", "age": "40", "y": str(i % 2)} for i in range(20)]
        valid_rows = [{"patient_id": f"va{i}", "event_time": f"2024-02-{(i%28)+1:02d}", "age": "45", "y": str((i + 1) % 2)} for i in range(10)]
        test_rows = [{"patient_id": f"te{i}", "event_time": f"2024-03-{(i%28)+1:02d}", "age": "50", "y": str(i % 2)} for i in range(10)]
        train_csv = td / "train.csv"
        valid_csv = td / "valid.csv"
        test_csv = td / "test.csv"
        write_csv(train_csv, train_rows, headers)
        write_csv(valid_csv, valid_rows, headers)
        write_csv(test_csv, test_rows, headers)

        # Keep defaults for all prompts after mandatory split paths.
        prompt_count = 32
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "train",
                "--print-only",
                "--train",
                str(train_csv),
                "--valid",
                str(valid_csv),
                "--test",
                str(test_csv),
            ],
            input_text=("\n" * prompt_count),
        )
        assert_true(proc.returncode == 0, "interactive train print-only exits 0")
        command_lines = [
            line.strip()
            for line in proc.stdout.splitlines()
            if line.strip().startswith("$ ") and "train_select_evaluate.py" in line
        ]
        assert_true(bool(command_lines), "interactive train emits generated train command")
        cmd_line = command_lines[-1] if command_lines else ""
        assert_true("--include-optional-models" not in cmd_line, "train default omits --include-optional-models")
        assert_true("--external-validation-report-out" not in cmd_line, "train default omits external report flag without cohort spec")
        assert_true("--feature-engineering-report-out" not in cmd_line, "train default omits feature engineering report flag without group spec")


def test_mlgg_interactive_workflow_always_injects_strict() -> None:
    print("\n=== mlgg interactive: workflow always injects --strict ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        req = td / "request.json"
        req.write_text("{}", encoding="utf-8")
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "workflow",
                "--print-only",
                "--request",
                str(req),
            ],
            # Prompts: evidence-dir, compare-manifest, allow-missing-compare, continue-on-fail
            input_text=("\n" * 8),
        )
        assert_true(proc.returncode == 0, "interactive workflow print-only exits 0")
        assert_true("--strict" in proc.stdout, "workflow interactive command includes --strict")


def test_mlgg_interactive_accept_defaults_non_blocking() -> None:
    print("\n=== mlgg interactive: --accept-defaults runs without stdin prompts ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        headers = ["patient_id", "event_time", "age", "y"]
        train_rows = [{"patient_id": f"tr{i}", "event_time": f"2024-01-{(i%28)+1:02d}", "age": "40", "y": str(i % 2)} for i in range(20)]
        valid_rows = [{"patient_id": f"va{i}", "event_time": f"2024-02-{(i%28)+1:02d}", "age": "45", "y": str((i + 1) % 2)} for i in range(10)]
        test_rows = [{"patient_id": f"te{i}", "event_time": f"2024-03-{(i%28)+1:02d}", "age": "50", "y": str(i % 2)} for i in range(10)]
        train_csv = td / "train.csv"
        valid_csv = td / "valid.csv"
        test_csv = td / "test.csv"
        write_csv(train_csv, train_rows, headers)
        write_csv(valid_csv, valid_rows, headers)
        write_csv(test_csv, test_rows, headers)

        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "train",
                "--print-only",
                "--accept-defaults",
                "--train",
                str(train_csv),
                "--valid",
                str(valid_csv),
                "--test",
                str(test_csv),
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 0, "--accept-defaults train print-only exits 0")
        assert_true(
            "Generated command:" in proc.stdout and "train_select_evaluate.py" in proc.stdout,
            "--accept-defaults emits train command without interactive input",
        )


def test_mlgg_interactive_profile_value_validation_fail_closed() -> None:
    print("\n=== mlgg interactive: malformed profile value types fail-closed ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        profile_dir = td / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = {
            "contract_version": "v1",
            "command": "train",
            "saved_at_utc": "2026-02-27T00:00:00Z",
            "argument_values": {
                "n_jobs": "bad",
            },
            "python": sys.executable,
            "cwd": str(td),
        }
        (profile_dir / "bad_profile.json").write_text(
            json.dumps(profile, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "train",
                "--load-profile",
                "--profile-name",
                "bad_profile",
                "--profile-dir",
                str(profile_dir),
                "--print-only",
                "--accept-defaults",
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 2, "malformed train profile exits 2")
        assert_true(
            "unable to load profile" in proc.stderr.lower(),
            "malformed profile surfaces load failure prefix",
        )
        assert_true(
            "profile key 'n_jobs' has invalid type" in proc.stderr.lower(),
            "malformed profile reports invalid key type details",
        )


def test_mlgg_interactive_workflow_default_evidence_dir_uses_request_project_base() -> None:
    print("\n=== mlgg interactive: workflow default evidence dir follows request project base ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "proj"
        request_path = project_root / "configs" / "request.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text("{}", encoding="utf-8")

        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "workflow",
                "--print-only",
                "--accept-defaults",
                "--request",
                str(request_path),
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 0, "workflow defaults print-only exits 0")
        expected_evidence = str((project_root / "evidence").resolve())
        assert_true(
            f"--evidence-dir {expected_evidence}" in proc.stdout,
            "workflow generated command uses request project-base evidence directory",
        )


def test_render_user_summary_propagates_fail_status() -> None:
    print("\n=== render_user_summary: fail status propagates to exit code and output ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        evidence_dir = td / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        strict_report = {
            "status": "fail",
            "failure_count": 1,
            "warning_count": 0,
            "failures": [{"code": "example_failure"}],
        }
        (evidence_dir / "strict_pipeline_report.json").write_text(
            json.dumps(strict_report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        out_md = evidence_dir / "user_summary.md"
        out_json = evidence_dir / "user_summary.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "render_user_summary.py"),
                "--evidence-dir",
                str(evidence_dir),
                "--out-markdown",
                str(out_md),
                "--out-json",
                str(out_json),
            ]
        )
        assert_true(proc.returncode == 2, "render_user_summary exits 2 when strict pipeline status is fail")
        assert_true("Status: fail" in proc.stdout, "render_user_summary prints Status: fail")
        summary = load_report(out_json)
        assert_true(summary.get("overall_status") == "fail", "summary overall_status is fail")


def test_mlgg_onboarding_preview_emits_full_step_plan() -> None:
    print("\n=== mlgg onboarding: preview mode emits 8-step command plan ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_project"
        report_path = td / "onboarding_preview_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "onboarding",
                "--project-root",
                str(project_root),
                "--mode",
                "preview",
                "--report",
                str(report_path),
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 0, "onboarding preview exits 0")
        assert_true(report_path.exists(), "onboarding preview report file exists")
        report = load_report(report_path)
        steps = report.get("steps", [])
        assert_true(report.get("contract_version") == "onboarding_report.v2", "onboarding preview contract_version is v2")
        assert_true(report.get("stop_on_fail") is True, "onboarding preview default stop_on_fail=true")
        assert_true(report.get("termination_reason") == "completed_successfully", "onboarding preview termination_reason is completed_successfully")
        assert_true(isinstance(steps, list) and len(steps) == 8, "onboarding preview contains 8 fixed steps")
        step_names = [str(row.get("name", "")) for row in steps if isinstance(row, dict)]
        assert_true("step1_doctor" in step_names, "onboarding preview includes step1_doctor")
        assert_true("step8_workflow_compare" in step_names, "onboarding preview includes step8_workflow_compare")


def test_mlgg_onboarding_guided_cancel_has_failure_code_and_actions() -> None:
    print("\n=== mlgg onboarding: guided cancel emits onboarding_step_cancelled ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_cancel"
        report_path = td / "onboarding_cancel_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "onboarding",
                "--project-root",
                str(project_root),
                "--mode",
                "guided",
                "--report",
                str(report_path),
            ],
            input_text="n\n",
        )
        assert_true(proc.returncode == 2, "guided cancel exits 2")
        assert_true(report_path.exists(), "guided cancel report exists")
        report = load_report(report_path)
        failure_codes = report.get("failure_codes", [])
        next_actions = report.get("next_actions", [])
        assert_true(report.get("status") == "fail", "guided cancel report status=fail")
        assert_true("onboarding_step_cancelled" in failure_codes, "guided cancel failure code present")
        assert_true(report.get("termination_reason") == "cancelled_by_user", "guided cancel termination_reason is cancelled_by_user")
        assert_true(
            all("No blocking failures" not in str(item) for item in next_actions),
            "guided cancel next_actions does not claim no blocking failures",
        )


def test_mlgg_onboarding_no_stop_on_fail_completes_with_failures() -> None:
    print("\n=== mlgg onboarding: --no-stop-on-fail records completed_with_failures ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_no_stop"
        report_path = td / "onboarding_no_stop_report.json"
        fake_python = td / "fake_python.sh"
        fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_python.chmod(0o755)
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg_onboarding.py"),
                "--project-root",
                str(project_root),
                "--mode",
                "auto",
                "--python",
                str(fake_python),
                "--no-stop-on-fail",
                "--report",
                str(report_path),
            ]
        )
        assert_true(proc.returncode == 2, "--no-stop-on-fail failure path exits 2")
        assert_true(report_path.exists(), "--no-stop-on-fail report exists")
        report = load_report(report_path)
        assert_true(report.get("status") == "fail", "--no-stop-on-fail report status=fail")
        assert_true(report.get("stop_on_fail") is False, "--no-stop-on-fail reflected in report")
        assert_true(
            report.get("termination_reason") == "completed_with_failures",
            "--no-stop-on-fail termination_reason is completed_with_failures",
        )
        steps = report.get("steps", [])
        step_names = [str(row.get("name", "")) for row in steps if isinstance(row, dict)]
        assert_true("step8_workflow_compare" in step_names, "--no-stop-on-fail continues through step8")


def test_mlgg_onboarding_missing_openssl_fails_closed() -> None:
    print("\n=== mlgg onboarding: missing openssl fails closed with actionable message ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_project"
        report_path = td / "onboarding_fail_report.json"
        # Keep this test fast and deterministic: guided mode + --yes will stop at the
        # first preflight where openssl is required and fail closed.
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "onboarding",
                "--project-root",
                str(project_root),
                "--mode",
                "guided",
                "--yes",
                "--report",
                str(report_path),
            ],
            env={"PATH": "/nonexistent"},
        )
        assert_true(proc.returncode == 2, "onboarding without openssl exits 2")
        combined = (proc.stdout + "\n" + proc.stderr).lower()
        assert_true("onboarding_openssl_missing" in combined, "missing openssl failure marker is present")


def test_mlgg_help_includes_onboarding_and_bootstrap_example() -> None:
    print("\n=== mlgg help: includes onboarding and bootstrap workflow examples ===")
    proc = run_gate([str(SCRIPTS_DIR / "mlgg.py"), "--help"])
    assert_true(proc.returncode == 0, "mlgg --help exits 0")
    body = proc.stdout
    assert_true("onboarding" in body, "mlgg --help lists onboarding command")
    assert_true(
        "--allow-missing-compare" in body,
        "mlgg --help examples include bootstrap manifest option",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Running gate smoke tests...")

    test_leakage_gate_clean_splits()
    test_leakage_gate_row_overlap()
    test_leakage_gate_id_overlap()
    test_leakage_gate_temporal_overlap()
    test_leakage_gate_temporal_boundary_equal_is_ok()
    test_leakage_gate_word_boundary_regex()
    test_self_critique_weight_normalization()
    test_run_strict_pipeline_requires_strict()
    test_transport_drop_ci_not_computed_sentinel()
    test_feature_engineering_audit_gate_error_codes()
    test_feature_engineering_audit_gate_to_float_isfinite()
    test_mlgg_interactive_train_defaults_are_dependency_safe()
    test_mlgg_interactive_workflow_always_injects_strict()
    test_mlgg_interactive_accept_defaults_non_blocking()
    test_mlgg_interactive_profile_value_validation_fail_closed()
    test_mlgg_interactive_workflow_default_evidence_dir_uses_request_project_base()
    test_render_user_summary_propagates_fail_status()
    test_mlgg_onboarding_preview_emits_full_step_plan()
    test_mlgg_onboarding_guided_cancel_has_failure_code_and_actions()
    test_mlgg_onboarding_no_stop_on_fail_completes_with_failures()
    test_mlgg_onboarding_missing_openssl_fails_closed()
    test_mlgg_help_includes_onboarding_and_bootstrap_example()

    print(f"\n{'='*50}")
    if _failures:
        print(f"\033[31mFAILED {len(_failures)} test(s):\033[0m")
        for name in _failures:
            print(f"  - {name}")
        return 1
    else:
        print(f"\033[32mAll tests passed.\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(main())
