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


def sha256_text_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_benchmark_registry(
    path: Path,
    *,
    dataset_file: Path,
    dataset_sha256: str,
    suite_rows: List[Dict[str, Any]],
) -> None:
    registry = {
        "contract_version": "benchmark_registry.v1",
        "description": "test registry",
        "stress_profile_set_default": "strict_v1",
        "cases": {
            "uci-breast-cancer-wdbc": {
                "role": "blocking",
                "split_seed": 20260224,
                "minimum_requirements": {"external_min_rows": 10, "external_min_events": 2},
            }
        },
        "dataset_fingerprints": {
            "uci-breast-cancer-wdbc": {
                "aggregate_sha256": "",
                "raw_files": [
                    {
                        "path": str(dataset_file),
                        "sha256": dataset_sha256,
                    }
                ],
            }
        },
        "profiles": {
            "quick": {"suites": suite_rows},
            "release": {"suites": suite_rows},
            "extended": {"suites": suite_rows},
        },
    }
    aggregate_row = f"{dataset_file}={dataset_sha256}"
    import hashlib

    registry["dataset_fingerprints"]["uci-breast-cancer-wdbc"]["aggregate_sha256"] = hashlib.sha256(
        aggregate_row.encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8")


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
        run_gate([
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
    uses_local_guard = ("math.isfinite" in src) and ("import math" in src)
    imports_shared_guard = ("from _gate_utils import" in src) and ("to_float" in src)
    assert_true(
        uses_local_guard or imports_shared_guard,
        "feature_engineering_audit_gate uses local finite-check guard or shared _gate_utils.to_float guard",
    )
    if imports_shared_guard:
        guard_src = (SCRIPTS_DIR / "_gate_utils.py").read_text(encoding="utf-8")
        assert_true("def to_float" in guard_src, "_gate_utils exposes to_float helper")
        assert_true("math.isfinite" in guard_src, "_gate_utils.to_float uses math.isfinite")


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
        expected_evidence_dir = str((td / "evidence").resolve())
        assert_true(
            f"--model-selection-report-out {expected_evidence_dir}/model_selection_report.json" in cmd_line,
            "train default output path is scoped to split project base (not repository root)",
        )


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
        assert_true(
            "--allow-missing-compare" in proc.stdout,
            "workflow interactive first-run default enables bootstrap compare bypass",
        )


def test_mlgg_interactive_authority_defaults_to_release_stress_path() -> None:
    print("\n=== mlgg interactive: authority defaults to CKD release stress path ===")
    proc = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "interactive",
            "--command",
            "authority",
            "--print-only",
            "--accept-defaults",
        ],
        input_text=None,
    )
    assert_true(proc.returncode == 0, "interactive authority print-only exits 0")
    body = proc.stdout
    assert_true("--include-stress-cases" in body, "authority interactive default includes stress cases")
    assert_true(
        "--stress-case-id uci-chronic-kidney-disease" in body,
        "authority interactive default selects CKD release stress case",
    )
    assert_true("--stress-seed-search" not in body, "authority interactive default does not enable seed search")


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


def test_mlgg_interactive_accept_defaults_auto_detects_presplit_data() -> None:
    print("\n=== mlgg interactive: --accept-defaults auto-detects pre-split data files ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        data_dir = td / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        headers = ["patient_id", "event_time", "age", "y"]
        train_rows = [{"patient_id": f"tr{i}", "event_time": f"2024-01-{(i%28)+1:02d}", "age": "40", "y": str(i % 2)} for i in range(20)]
        valid_rows = [{"patient_id": f"va{i}", "event_time": f"2024-02-{(i%28)+1:02d}", "age": "45", "y": str((i + 1) % 2)} for i in range(10)]
        test_rows = [{"patient_id": f"te{i}", "event_time": f"2024-03-{(i%28)+1:02d}", "age": "50", "y": str(i % 2)} for i in range(10)]
        train_csv = data_dir / "train.csv"
        valid_csv = data_dir / "valid.csv"
        test_csv = data_dir / "test.csv"
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
                "--cwd",
                str(td),
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 0, "auto-detected pre-split defaults exit 0")
        body = proc.stdout
        assert_true("--train " in body and str(train_csv.resolve()) in body, "auto-detected train split path is used")
        assert_true("--valid " in body and str(valid_csv.resolve()) in body, "auto-detected valid split path is used")
        assert_true("--test " in body and str(test_csv.resolve()) in body, "auto-detected test split path is used")


def test_mlgg_interactive_accept_defaults_missing_presplit_is_actionable() -> None:
    print("\n=== mlgg interactive: --accept-defaults missing pre-split defaults is actionable ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "interactive",
                "--command",
                "train",
                "--print-only",
                "--accept-defaults",
                "--cwd",
                str(td),
            ],
            input_text=None,
        )
        assert_true(proc.returncode == 2, "missing auto-detected pre-split defaults exits 2")
        stderr_text = proc.stderr + proc.stdout
        assert_true(
            "No default pre-split files detected for --accept-defaults" in stderr_text,
            "missing pre-split defaults emits explicit error marker",
        )
        assert_true(
            "--train/--valid/--test" in stderr_text and "--input-csv" in stderr_text,
            "missing pre-split defaults emits actionable override guidance",
        )


def test_mlgg_authority_wrapper_release_and_research_presets() -> None:
    print("\n=== mlgg wrapper: authority-release and authority-research-heart presets ===")
    release = run_gate([str(SCRIPTS_DIR / "mlgg.py"), "authority-release", "--dry-run"])
    assert_true(release.returncode == 0, "authority-release dry-run exits 0")
    assert_true("--include-stress-cases" in release.stdout, "authority-release injects include-stress-cases")
    assert_true(
        "--stress-case-id uci-chronic-kidney-disease" in release.stdout,
        "authority-release injects CKD stress-case-id",
    )

    research = run_gate([str(SCRIPTS_DIR / "mlgg.py"), "authority-research-heart", "--dry-run"])
    assert_true(research.returncode == 0, "authority-research-heart dry-run exits 0")
    assert_true("--include-stress-cases" in research.stdout, "authority-research-heart injects include-stress-cases")
    assert_true(
        "--stress-case-id uci-heart-disease" in research.stdout,
        "authority-research-heart injects heart stress-case-id",
    )
    assert_true("--stress-seed-search" in research.stdout, "authority-research-heart injects seed-search")


def test_mlgg_authority_wrapper_rejects_conflicting_route_flags() -> None:
    print("\n=== mlgg wrapper: preset commands reject conflicting route flags ===")
    release_conflict = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "authority-release",
            "--dry-run",
            "--stress-case-id",
            "uci-heart-disease",
        ]
    )
    assert_true(release_conflict.returncode == 2, "authority-release conflicting route flag exits 2")
    assert_true(
        "authority_preset_route_override_forbidden" in release_conflict.stderr,
        "authority-release conflict emits standard failure code",
    )

    research_conflict = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "authority-research-heart",
            "--dry-run",
            "--stress-case-id",
            "uci-chronic-kidney-disease",
        ]
    )
    assert_true(research_conflict.returncode == 2, "authority-research-heart conflicting route flag exits 2")
    assert_true(
        "authority_preset_route_override_forbidden" in research_conflict.stderr,
        "authority-research-heart conflict emits standard failure code",
    )
    research_conflict_json = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "authority-research-heart",
            "--dry-run",
            "--stress-case-id",
            "uci-chronic-kidney-disease",
            "--error-json",
        ]
    )
    assert_true(research_conflict_json.returncode == 2, "authority conflict with --error-json exits 2")
    stderr_lines = [line.strip() for line in research_conflict_json.stderr.splitlines() if line.strip()]
    payload = {}
    if stderr_lines:
        try:
            payload = json.loads(stderr_lines[-1])
        except Exception:
            payload = {}
    assert_true(
        payload.get("contract_version") == "mlgg_error.v1",
        "mlgg --error-json emits mlgg_error.v1 contract",
    )
    assert_true(
        payload.get("code") == "authority_preset_route_override_forbidden",
        "mlgg --error-json emits expected authority preset conflict code",
    )


def test_release_benchmark_contract_v2_fields_present() -> None:
    print("\n=== benchmark-suite: contract v2 required fields present ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        out = td / "benchmark_report.json"
        _artifacts_dir = td / "_benchmark_matrix_runs"  # noqa: F841 – used by subprocess
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--dry-run",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "3",
            ]
        )
        assert_true(proc.returncode == 0, "benchmark dry-run exits 0")
        report = load_report(out)
        assert_true(report.get("contract_version") == "release_benchmark_matrix.v2", "benchmark contract is v2")
        assert_true("status_reason" in report, "benchmark report includes status_reason")
        assert_true("failure_codes" in report and isinstance(report.get("failure_codes"), list), "benchmark report includes failure_codes")
        assert_true("all_failure_codes" in report and isinstance(report.get("all_failure_codes"), list), "benchmark report includes all_failure_codes")
        assert_true(report.get("repeat_count") == 3, "benchmark report includes repeat_count")
        assert_true("repeat_consistent" in report, "benchmark report includes repeat_consistent")
        assert_true("dataset_registry_sha256" in report, "benchmark report includes dataset_registry_sha256")
        assert_true("blocking_suite_ids" in report, "benchmark report includes blocking_suite_ids")
        assert_true("nonblocking_suite_ids" in report, "benchmark report includes nonblocking_suite_ids")
        assert_true("blocking_failure_codes" in report, "benchmark report includes blocking_failure_codes")
        assert_true("observational_failure_codes" in report, "benchmark report includes observational_failure_codes")
        assert_true(
            isinstance(report.get("authority_subprocess_timeout_seconds"), int),
            "benchmark report includes authority_subprocess_timeout_seconds",
        )
        assert_true(
            isinstance(report.get("authority_case_lock_timeout_seconds"), int),
            "benchmark report includes authority_case_lock_timeout_seconds",
        )
        assert_true(
            isinstance(report.get("authority_lock_wait_heartbeat_seconds"), int),
            "benchmark report includes authority_lock_wait_heartbeat_seconds",
        )


def test_release_benchmark_forwards_authority_timeout_and_lock_args() -> None:
    print("\n=== benchmark-suite: forwards authority timeout/lock args ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "authority_smoke_forwarding",
                    "name": "Authority forwarding smoke",
                    "kind": "authority",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        fake_python = td / "fake_python.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "out=\"\"\n"
            "log=\"\"\n"
            "sub_to=\"\"\n"
            "lock_to=\"\"\n"
            "lock_hb=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  if [ \"$prev\" = \"--subprocess-timeout-seconds\" ]; then sub_to=\"$arg\"; fi\n"
            "  if [ \"$prev\" = \"--case-lock-timeout-seconds\" ]; then lock_to=\"$arg\"; fi\n"
            "  if [ \"$prev\" = \"--lock-wait-heartbeat-seconds\" ]; then lock_hb=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "printf '{\"overall_status\":\"pass\",\"results\":[]}' > \"$out\"\n"
            "log=\"${out%.json}.args.txt\"\n"
            "printf 'subprocess=%s\\ncase_lock=%s\\nheartbeat=%s\\n' \"$sub_to\" \"$lock_to\" \"$lock_hb\" > \"$log\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        artifacts_dir = td / "_benchmark_matrix_runs"
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--artifacts-dir",
                str(artifacts_dir),
                "--python",
                str(fake_python),
                "--repeat",
                "1",
                "--authority-subprocess-timeout-seconds",
                "111",
                "--authority-case-lock-timeout-seconds",
                "222",
                "--authority-lock-wait-heartbeat-seconds",
                "7",
            ]
        )
        assert_true(proc.returncode == 0, "benchmark forwarding run exits 0")
        report = load_report(out)
        assert_true(report.get("overall_status") == "pass", "benchmark forwarding run status=pass")
        suite_runs = report.get("suite_runs", [])
        assert_true(isinstance(suite_runs, list) and len(suite_runs) == 1, "benchmark forwarding run emits one suite row")
        summary_file = Path(str((suite_runs[0] or {}).get("summary_file", "")))
        args_log = summary_file.with_name(f"{summary_file.stem}.args.txt")
        assert_true(args_log.exists(), "authority forwarding args log file exists")
        if args_log.exists():
            log_text = args_log.read_text(encoding="utf-8")
            assert_true("subprocess=111" in log_text, "forwarded subprocess timeout matches CLI input")
            assert_true("case_lock=222" in log_text, "forwarded case lock timeout matches CLI input")
            assert_true("heartbeat=7" in log_text, "forwarded lock heartbeat matches CLI input")


def test_mlgg_benchmark_passthrough_strips_separator_before_subcommand() -> None:
    print("\n=== mlgg benchmark-suite: passthrough strips leading -- separator ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "mlgg.py"),
                "benchmark-suite",
                "--profile",
                "quick",
                "--repeat",
                "1",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--",
                "--dry-run",
            ]
        )
        assert_true(proc.returncode == 0, "mlgg benchmark-suite passthrough dry-run exits 0")
        assert_true(out.exists(), "mlgg benchmark-suite passthrough creates output report")
        report = load_report(out)
        assert_true(report.get("overall_status") == "dry_run", "mlgg benchmark-suite passthrough keeps dry_run status")


def test_release_benchmark_blocking_failure_sets_standard_code() -> None:
    print("\n=== benchmark-suite: blocking failure emits benchmark_blocking_suite_failed ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        fake_python = td / "fake_python.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "printf '{\"overall_status\":\"fail\",\"passed_count\":0,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":false,\"observed_codes\":[\"scenario_fail\"]}]}' > \"$out\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "1",
                "--python",
                str(fake_python),
            ]
        )
        assert_true(proc.returncode == 2, "benchmark blocking failure exits 2")
        report = load_report(out)
        assert_true(report.get("overall_status") == "fail", "benchmark blocking failure report status=fail")
        assert_true(report.get("status_reason") == "benchmark_blocking_suite_failed", "benchmark blocking failure reason matches")
        codes = report.get("failure_codes", [])
        assert_true("benchmark_blocking_suite_failed" in codes, "benchmark blocking failure code present")


def test_release_benchmark_registry_mismatch_detected() -> None:
    print("\n=== benchmark-suite: registry mismatch triggers benchmark_registry_mismatch ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        registry_file = td / "benchmark_registry.bad.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256="0" * 64,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--dry-run",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
            ]
        )
        assert_true(proc.returncode == 2, "benchmark registry mismatch exits 2")
        report = load_report(out)
        assert_true(report.get("status_reason") == "benchmark_registry_mismatch", "registry mismatch status reason matches")
        assert_true("benchmark_registry_mismatch" in report.get("failure_codes", []), "registry mismatch failure code present")


def test_release_benchmark_registry_failure_contract_fields_present() -> None:
    print("\n=== benchmark-suite: registry failure still emits full v2 fields ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        out = td / "benchmark_report.json"
        missing_registry = td / "missing_registry.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "release",
                "--registry-file",
                str(missing_registry),
                "--output",
                str(out),
            ]
        )
        assert_true(proc.returncode == 2, "benchmark registry missing exits 2")
        report = load_report(out)
        assert_true(report.get("contract_version") == "release_benchmark_matrix.v2", "registry failure report keeps v2 contract")
        assert_true(report.get("status_reason") == "benchmark_registry_missing", "registry failure status reason matches")
        assert_true(isinstance(report.get("failure_codes"), list), "registry failure includes failure_codes list")
        assert_true(isinstance(report.get("all_failure_codes"), list), "registry failure includes all_failure_codes list")
        assert_true(isinstance(report.get("blocking_failure_codes"), list), "registry failure includes blocking_failure_codes list")
        assert_true(
            isinstance(report.get("observational_failure_codes"), list),
            "registry failure includes observational_failure_codes list",
        )


def test_release_benchmark_repeat_inconsistency_detected() -> None:
    print("\n=== benchmark-suite: repeat inconsistency triggers benchmark_repeat_inconsistent ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        counter = td / "counter.txt"
        fake_python = td / "fake_python_repeat.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"COUNTER='{counter}'\n"
            "count=0\n"
            "if [ -f \"$COUNTER\" ]; then count=$(cat \"$COUNTER\"); fi\n"
            "count=$((count+1))\n"
            "echo \"$count\" > \"$COUNTER\"\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "if [ \"$count\" -eq 1 ]; then\n"
            "  printf '{\"overall_status\":\"pass\",\"passed_count\":1,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' > \"$out\"\n"
            "else\n"
            "  printf '{\"overall_status\":\"fail\",\"passed_count\":0,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":false,\"observed_codes\":[\"scenario_fail\"]}]}' > \"$out\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "2",
                "--python",
                str(fake_python),
            ]
        )
        assert_true(proc.returncode == 2, "benchmark repeat inconsistency exits 2")
        report = load_report(out)
        assert_true(report.get("status_reason") == "benchmark_repeat_inconsistent", "repeat inconsistency status reason matches")
        assert_true(report.get("repeat_consistent") is False, "repeat consistency flag is false")
        assert_true("benchmark_repeat_inconsistent" in report.get("failure_codes", []), "repeat inconsistency code present")


def test_release_benchmark_repeat_detects_authority_metric_drift() -> None:
    print("\n=== benchmark-suite: repeat inconsistency catches authority metric drift ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "authority_release_core",
                    "name": "Authority release core",
                    "kind": "authority",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": ["uci-breast-cancer-wdbc"],
                }
            ],
        )
        counter = td / "counter_authority.txt"
        fake_python = td / "fake_python_authority_drift.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"COUNTER='{counter}'\n"
            "count=0\n"
            "if [ -f \"$COUNTER\" ]; then count=$(cat \"$COUNTER\"); fi\n"
            "count=$((count+1))\n"
            "echo \"$count\" > \"$COUNTER\"\n"
            "script=\"$1\"\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "if echo \"$script\" | grep -q 'run_authority_e2e.py'; then\n"
            "  if [ \"$count\" -eq 1 ]; then pr='0.910000'; else pr='0.780000'; fi\n"
            "  printf '{\"overall_status\":\"pass\",\"results\":[{\"case_id\":\"uci-breast-cancer-wdbc\",\"status\":\"pass\",\"metrics\":{\"pr_auc\":%s,\"roc_auc\":0.930000,\"f2_beta\":0.740000,\"brier\":0.120000}}]}' \"$pr\" > \"$out\"\n"
            "else\n"
            "  printf '{\"overall_status\":\"pass\",\"passed_count\":1,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' > \"$out\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "2",
                "--python",
                str(fake_python),
            ]
        )
        assert_true(proc.returncode == 2, "authority metric drift triggers repeat inconsistency exit 2")
        report = load_report(out)
        assert_true(
            report.get("status_reason") == "benchmark_repeat_inconsistent",
            "authority metric drift status reason is repeat inconsistency",
        )
        assert_true(report.get("repeat_consistent") is False, "authority metric drift sets repeat_consistent=false")
        assert_true(
            "benchmark_repeat_inconsistent" in report.get("failure_codes", []),
            "authority metric drift includes repeat inconsistency code",
        )


def test_release_benchmark_junit_marks_repeat_inconsistent_global_failure() -> None:
    print("\n=== benchmark-suite: junit marks global failure on repeat inconsistency ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "authority_release_core",
                    "name": "Authority release core",
                    "kind": "authority",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": ["uci-breast-cancer-wdbc"],
                }
            ],
        )
        counter = td / "counter_junit.txt"
        fake_python = td / "fake_python_authority_junit_drift.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"COUNTER='{counter}'\n"
            "count=0\n"
            "if [ -f \"$COUNTER\" ]; then count=$(cat \"$COUNTER\"); fi\n"
            "count=$((count+1))\n"
            "echo \"$count\" > \"$COUNTER\"\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "if [ \"$count\" -eq 1 ]; then pr='0.910000'; else pr='0.780000'; fi\n"
            "printf '{\"overall_status\":\"pass\",\"results\":[{\"case_id\":\"uci-breast-cancer-wdbc\",\"status\":\"pass\",\"metrics\":{\"pr_auc\":%s,\"roc_auc\":0.930000,\"f2_beta\":0.740000,\"brier\":0.120000}}]}' \"$pr\" > \"$out\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        junit = td / "benchmark.junit.xml"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "2",
                "--python",
                str(fake_python),
                "--emit-junit",
                str(junit),
            ]
        )
        assert_true(proc.returncode == 2, "repeat inconsistency exits 2 (junit test)")
        report = load_report(out)
        assert_true(
            report.get("status_reason") == "benchmark_repeat_inconsistent",
            "repeat inconsistency status reason matches (junit test)",
        )
        junit_text = junit.read_text(encoding="utf-8")
        assert_true("failures=\"1\"" in junit_text or "failures='1'" in junit_text, "junit reports one failure")
        assert_true("name=\"global\"" in junit_text or "name='global'" in junit_text, "junit contains global testcase")
        assert_true(
            "benchmark_repeat_inconsistent" in junit_text,
            "junit global failure message includes repeat inconsistency code",
        )


def test_release_benchmark_suite_timeout_is_fail_closed() -> None:
    print("\n=== benchmark-suite: suite timeout fails closed with standard code ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        fake_python = td / "fake_python_timeout.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "sleep 2\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "printf '{\"overall_status\":\"pass\",\"passed_count\":1,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' > \"$out\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "1",
                "--python",
                str(fake_python),
                "--suite-timeout-seconds",
                "1",
            ]
        )
        assert_true(proc.returncode == 2, "suite timeout exits 2")
        report = load_report(out)
        assert_true(report.get("overall_status") == "fail", "suite timeout yields fail status")
        assert_true(
            "benchmark_suite_timeout" in report.get("blocking_failure_codes", []),
            "suite timeout code appears in blocking failure codes",
        )
        assert_true(
            "benchmark_blocking_suite_failed" in report.get("failure_codes", []),
            "suite timeout still maps to blocking suite failed status code",
        )


def test_release_benchmark_repeat_detects_adversarial_count_drift() -> None:
    print("\n=== benchmark-suite: repeat inconsistency catches adversarial count drift ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        counter = td / "counter_adv_counts.txt"
        fake_python = td / "fake_python_adversarial_count_drift.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"COUNTER='{counter}'\n"
            "count=0\n"
            "if [ -f \"$COUNTER\" ]; then count=$(cat \"$COUNTER\"); fi\n"
            "count=$((count+1))\n"
            "echo \"$count\" > \"$COUNTER\"\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "printf '{\"overall_status\":\"pass\",\"passed_count\":%s,\"scenario_count\":%s,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' \"$count\" \"$count\" > \"$out\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "2",
                "--python",
                str(fake_python),
            ]
        )
        assert_true(proc.returncode == 2, "adversarial count drift triggers repeat inconsistency exit 2")
        report = load_report(out)
        assert_true(
            report.get("status_reason") == "benchmark_repeat_inconsistent",
            "adversarial count drift status reason is repeat inconsistency",
        )
        assert_true(report.get("repeat_consistent") is False, "adversarial count drift sets repeat_consistent=false")
        assert_true(
            "benchmark_repeat_inconsistent" in report.get("failure_codes", []),
            "adversarial count drift includes repeat inconsistency code",
        )


def test_release_benchmark_emits_observational_diagnostics_for_nonblocking_failures() -> None:
    print("\n=== benchmark-suite: non-blocking authority failure emits observational diagnostics ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                },
                {
                    "suite_id": "authority_release_extended",
                    "name": "Authority extended route (+ Diabetes130 large cohort)",
                    "kind": "authority",
                    "blocking": False,
                    "args": [],
                    "expected_case_ids": [
                        "uci-diabetes-130-readmission",
                    ],
                },
            ],
        )
        fake_python = td / "fake_python_diag.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "script=\"$1\"\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "if echo \"$script\" | grep -q 'run_authority_e2e.py'; then\n"
            "  printf '{\"overall_status\":\"fail\",\"results\":[{\"case_id\":\"uci-diabetes-130-readmission\",\"status\":\"fail\",\"failure_code\":\"strict_pipeline_failed\",\"root_failure_code_primary\":\"clinical_floor_npv_not_met\",\"root_failure_codes\":[\"clinical_floor_npv_not_met\",\"clinical_floor_ppv_not_met\"],\"clinical_floor_gap_summary\":{\"minimum_margin\":-0.19,\"internal_test\":{\"floor_metrics\":{\"npv\":{\"required_min\":0.9,\"observed\":0.81,\"margin\":-0.09,\"met\":false},\"ppv\":{\"required_min\":0.55,\"observed\":0.36,\"margin\":-0.19,\"met\":false}}},\"external_cohorts\":[]},\"artifacts\":{\"distribution_report\":\"/tmp/distribution_report.json\",\"external_validation_report\":\"/tmp/external_validation_report.json\"},\"diabetes_feasibility_scan\":{\"status\":\"pass\",\"recommended_retry_command\":\"python3 scripts/mlgg.py authority --include-large-cases --include-stress-cases --stress-case-id uci-chronic-kidney-disease --diabetes-target-mode lt30 --diabetes-max-rows 5000\",\"best_candidate\":{\"target_mode\":\"lt30\",\"max_rows\":5000}}}]}' > \"$out\"\n"
            "else\n"
            "  printf '{\"overall_status\":\"pass\",\"passed_count\":1,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' > \"$out\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        diag_out = td / "observational_diagnostics.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "1",
                "--python",
                str(fake_python),
                "--observational-diagnostics-out",
                str(diag_out),
            ]
        )
        assert_true(proc.returncode == 0, "benchmark with non-blocking authority failure still exits 0")
        report = load_report(out)
        assert_true(report.get("overall_status") == "pass", "overall status remains pass with non-blocking failure")
        assert_true(
            isinstance(report.get("failure_codes"), list) and len(report.get("failure_codes")) == 0,
            "top-level failure_codes remains empty when only non-blocking suites fail",
        )
        assert_true(
            isinstance(report.get("blocking_failure_codes"), list) and len(report.get("blocking_failure_codes")) == 0,
            "blocking_failure_codes remains empty",
        )
        assert_true(
            isinstance(report.get("observational_failure_codes"), list)
            and "clinical_floor_npv_not_met" in report.get("observational_failure_codes", []),
            "observational_failure_codes includes non-blocking clinical code",
        )
        obs_items = report.get("observational_diagnostics", [])
        assert_true(isinstance(obs_items, list) and len(obs_items) == 1, "observational_diagnostics contains one failed non-blocking suite")
        first = obs_items[0] if obs_items else {}
        assert_true(first.get("suite_id") == "authority_release_extended", "observational diagnostics suite id matches")
        assert_true(first.get("blocking") is False, "observational diagnostics marks blocking=false for non-blocking suite")
        assert_true(first.get("failure_code") == "strict_pipeline_failed", "observational diagnostics primary failure_code matches")
        case_rows = first.get("case_diagnostics", [])
        assert_true(isinstance(case_rows, list) and len(case_rows) == 1, "observational diagnostics contains failed case row")
        case_row = case_rows[0] if case_rows else {}
        assert_true(case_row.get("case_id") == "uci-diabetes-130-readmission", "diagnostic case id matches")
        scan_info = case_row.get("diabetes_feasibility_scan")
        assert_true(isinstance(scan_info, dict), "diagnostic case row includes diabetes feasibility scan payload")
        assert_true(
            str(scan_info.get("status")) == "pass",
            "diagnostic case row propagates diabetes feasibility scan status",
        )
        actions = first.get("recommended_actions", [])
        assert_true(
            any("scan-diabetes" in str(item) for item in actions),
            "observational diagnostics include diabetes feasibility scan recommendation",
        )
        assert_true(
            any("Apply feasible diabetes config candidate:" in str(item) for item in actions),
            "observational diagnostics include feasible diabetes retry command when available",
        )
        assert_true(diag_out.exists(), "observational diagnostics sidecar file exists")
        diag_payload = load_report(diag_out)
        assert_true(
            diag_payload.get("contract_version") == "release_benchmark_observational_diagnostics.v1",
            "observational diagnostics sidecar contract version matches",
        )
        suite_rows = report.get("suites", [])
        assert_true(isinstance(suite_rows, list) and len(suite_rows) >= 2, "benchmark report includes suite summary rows")
        failed_suite = next((row for row in suite_rows if row.get("suite_id") == "authority_release_extended"), {})
        assert_true(failed_suite.get("status") == "fail", "failed non-blocking suite preserved in suites summary")
        assert_true(failed_suite.get("failure_code") == "strict_pipeline_failed", "suite summary primary failure_code is populated")


def test_release_benchmark_repeat_outputs_are_not_overwritten() -> None:
    print("\n=== benchmark-suite: repeat outputs keep per-repeat summary files ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        dataset_file = td / "dataset.csv"
        dataset_file.write_text("x\n1\n", encoding="utf-8")
        dataset_sha = sha256_text_file(dataset_file)
        registry_file = td / "benchmark_registry.json"
        write_benchmark_registry(
            registry_file,
            dataset_file=dataset_file,
            dataset_sha256=dataset_sha,
            suite_rows=[
                {
                    "suite_id": "adversarial_fail_closed",
                    "name": "Adversarial fail-closed scenarios",
                    "kind": "adversarial",
                    "blocking": True,
                    "args": [],
                    "expected_case_ids": [],
                }
            ],
        )
        fake_python = td / "fake_python_repeat_outputs.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "out=\"\"\n"
            "prev=\"\"\n"
            "for arg in \"$@\"; do\n"
            "  if [ \"$prev\" = \"--output\" ] || [ \"$prev\" = \"--summary-file\" ]; then out=\"$arg\"; fi\n"
            "  prev=\"$arg\"\n"
            "done\n"
            "mkdir -p \"$(dirname \"$out\")\"\n"
            "printf '{\"overall_status\":\"pass\",\"passed_count\":1,\"scenario_count\":1,\"results\":[{\"name\":\"s1\",\"passed\":true,\"observed_codes\":[]}]}' > \"$out\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        out = td / "benchmark_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_release_benchmark_matrix.py"),
                "--profile",
                "quick",
                "--registry-file",
                str(registry_file),
                "--output",
                str(out),
                "--repeat",
                "2",
                "--python",
                str(fake_python),
            ]
        )
        assert_true(proc.returncode == 0, "benchmark repeat outputs run exits 0")
        report = load_report(out)
        suite_runs = report.get("suite_runs", [])
        summary_files = [
            str(row.get("summary_file", ""))
            for row in suite_runs
            if isinstance(row, dict) and str(row.get("suite_id")) == "adversarial_fail_closed"
        ]
        assert_true(len(summary_files) == 2, "two repeat rows present for adversarial suite")
        assert_true(
            len(set(summary_files)) == 2,
            "repeat summary files are unique per repeat index",
        )
        assert_true(any(".r1.json" in path for path in summary_files), "repeat summary includes r1 suffix")
        assert_true(any(".r2.json" in path for path in summary_files), "repeat summary includes r2 suffix")


def test_authority_e2e_gitignore_covers_runtime_outputs() -> None:
    print("\n=== authority-e2e: gitignore covers runtime output directories ===")
    ignore_file = SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / ".gitignore"
    text = ignore_file.read_text(encoding="utf-8")
    assert_true("_benchmark_matrix_runs/" in text, "authority-e2e gitignore includes _benchmark_matrix_runs/")
    assert_true("_locks/" in text, "authority-e2e gitignore includes _locks/")
    assert_true("uci-chronic-kidney-disease/" in text, "authority-e2e gitignore includes CKD runtime directory")
    assert_true(
        "uci-diabetes-130-readmission/" in text,
        "authority-e2e gitignore includes Diabetes130 runtime directory",
    )


def test_authority_e2e_run_cmd_has_subprocess_timeout_guard() -> None:
    print("\n=== authority-e2e: run_cmd has subprocess timeout guard ===")
    script = SCRIPTS_DIR.parent / "experiments" / "authority-e2e" / "run_authority_e2e.py"
    text = script.read_text(encoding="utf-8")
    assert_true("DEFAULT_SUBPROCESS_TIMEOUT_SECONDS" in text, "authority-e2e defines timeout default constant")
    assert_true("--subprocess-timeout-seconds" in text, "authority-e2e parser exposes subprocess timeout option")
    assert_true(
        "timeout=timeout_seconds" in text,
        "authority-e2e run_cmd applies timeout to subprocess.run",
    )
    assert_true("subprocess_timeout: timeout_after_seconds=" in text, "authority-e2e timeout path is explicitly reported")


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
        assert_true(
            "--allow-missing-compare" in proc.stdout,
            "workflow defaults include --allow-missing-compare when baseline manifest is missing",
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
        (evidence_dir / "dag_pipeline_report.json").write_text(
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


def _prepare_wrapper_minimal_project(project_root: Path) -> Path:
    configs = project_root / "configs"
    data = project_root / "data"
    evidence = project_root / "evidence"
    configs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (data / f"{split}.csv").write_text("patient_id,event_time,y\n1,2020-01-01,0\n", encoding="utf-8")
    request = {
        "split_paths": {
            "train": "../data/train.csv",
            "valid": "../data/valid.csv",
            "test": "../data/test.csv",
        },
        "label_col": "y",
        "patient_id_col": "patient_id",
        "index_time_col": "event_time",
    }
    request_path = configs / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=True, indent=2), encoding="utf-8")
    return request_path


def test_wrapper_stale_publication_report_does_not_trigger_bootstrap_retry() -> None:
    print("\n=== run_productized_workflow: stale publication report does not trigger bootstrap retry ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "proj"
        request_path = _prepare_wrapper_minimal_project(project_root)
        evidence = project_root / "evidence"
        stale_pub = evidence / "publication_gate_report.json"
        stale_manifest = evidence / "manifest.json"
        stale_pub.write_text(
            json.dumps({"status": "fail", "failures": [{"code": "manifest_comparison_missing"}]}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        stale_manifest.write_text(json.dumps({"stale": True}, ensure_ascii=True, indent=2), encoding="utf-8")
        old_ts = 946684800  # 2000-01-01 UTC
        os.utime(stale_pub, (old_ts, old_ts))
        os.utime(stale_manifest, (old_ts, old_ts))

        fake_python = td / "fake_py.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  */env_doctor.py) exit 0 ;;\n"
            "  */schema_preflight.py) exit 0 ;;\n"
            "  */run_strict_pipeline.py) exit 2 ;;\n"
            "  */render_user_summary.py) exit 0 ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        wrapper_report = evidence / "productized_workflow_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "run_productized_workflow.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence),
                "--strict",
                "--allow-missing-compare",
                "--python",
                str(fake_python),
                "--report",
                str(wrapper_report),
            ]
        )
        assert_true(proc.returncode == 2, "wrapper exits 2 when strict step fails without recoverable bootstrap")
        report = load_report(wrapper_report)
        step_names = [str(row.get("name", "")) for row in report.get("steps", []) if isinstance(row, dict)]
        assert_true(
            "run_strict_pipeline_with_bootstrap_baseline" not in step_names,
            "stale publication report does not trigger retry step",
        )
        assert_true(report.get("status") == "fail", "wrapper report status=fail when blocking step fails")
        assert_true(report.get("status_reason") == "blocking_step_failed", "status_reason is blocking_step_failed")
        assert_true(report.get("bootstrap_recovery_applied") is False, "bootstrap_recovery_applied remains false")


def test_wrapper_schema_preflight_failure_causes_fail_even_if_strict_pass() -> None:
    print("\n=== run_productized_workflow: schema preflight failure blocks pass even if strict passes ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "proj"
        request_path = _prepare_wrapper_minimal_project(project_root)
        evidence = project_root / "evidence"

        fake_python = td / "fake_py.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  */env_doctor.py) exit 0 ;;\n"
            "  */schema_preflight.py) exit 2 ;;\n"
            "  */run_strict_pipeline.py) exit 0 ;;\n"
            "  */render_user_summary.py) exit 0 ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        wrapper_report = evidence / "productized_workflow_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "run_productized_workflow.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence),
                "--strict",
                "--allow-missing-compare",
                "--python",
                str(fake_python),
                "--report",
                str(wrapper_report),
            ]
        )
        assert_true(proc.returncode == 2, "wrapper exits 2 when schema_preflight fails")
        report = load_report(wrapper_report)
        assert_true(report.get("status") == "fail", "wrapper status=fail")
        assert_true(report.get("status_reason") == "blocking_step_failed", "status_reason is blocking_step_failed")
        steps = {str(row.get("name", "")): row for row in report.get("steps", []) if isinstance(row, dict)}
        assert_true(str(steps.get("schema_preflight", {}).get("status")) == "fail", "schema_preflight step status=fail")
        assert_true(bool(steps.get("schema_preflight", {}).get("blocking")) is True, "schema_preflight is blocking")
        assert_true(str(steps.get("run_strict_pipeline", {}).get("status")) == "pass", "strict step may pass but wrapper still fails")


def test_wrapper_bootstrap_recovered_path_reports_pass_with_recovered_step() -> None:
    print("\n=== run_productized_workflow: bootstrap recovered path reports pass with recovered step ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "proj"
        request_path = _prepare_wrapper_minimal_project(project_root)
        evidence = project_root / "evidence"
        counter_file = td / "strict_counter.txt"

        fake_python = td / "fake_py.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            f"COUNTER_FILE='{counter_file}'\n"
            "if [ \"$1\" = \"\" ]; then exit 0; fi\n"
            "case \"$1\" in\n"
            "  */env_doctor.py) exit 0 ;;\n"
            "  */schema_preflight.py) exit 0 ;;\n"
            "  */render_user_summary.py) exit 0 ;;\n"
            "  */run_strict_pipeline.py)\n"
            "    count=0\n"
            "    if [ -f \"$COUNTER_FILE\" ]; then count=$(cat \"$COUNTER_FILE\"); fi\n"
            "    count=$((count+1))\n"
            "    echo \"$count\" > \"$COUNTER_FILE\"\n"
            "    evidence_dir=\"\"\n"
            "    prev=\"\"\n"
            "    for arg in \"$@\"; do\n"
            "      if [ \"$prev\" = \"--evidence-dir\" ]; then evidence_dir=\"$arg\"; fi\n"
            "      prev=\"$arg\"\n"
            "    done\n"
            "    if [ \"$count\" -eq 1 ]; then\n"
            "      if [ -n \"$evidence_dir\" ]; then\n"
            "        mkdir -p \"$evidence_dir\"\n"
            "        printf '{\"status\":\"fail\",\"failures\":[{\"code\":\"manifest_comparison_missing\"}]}' > \"$evidence_dir/publication_gate_report.json\"\n"
            "        printf '{\"manifest\":\"current\"}' > \"$evidence_dir/manifest.json\"\n"
            "      fi\n"
            "      exit 2\n"
            "    fi\n"
            "    exit 0 ;;\n"
            "  *) exit 0 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        wrapper_report = evidence / "productized_workflow_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "run_productized_workflow.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence),
                "--strict",
                "--allow-missing-compare",
                "--python",
                str(fake_python),
                "--report",
                str(wrapper_report),
            ]
        )
        assert_true(proc.returncode == 0, "wrapper exits 0 after successful bootstrap recovery")
        report = load_report(wrapper_report)
        assert_true(report.get("status") == "pass", "wrapper status=pass after recovery")
        assert_true(report.get("status_reason") == "bootstrap_recovered", "status_reason is bootstrap_recovered")
        assert_true(report.get("bootstrap_recovery_applied") is True, "bootstrap_recovery_applied=true")
        steps = {str(row.get("name", "")): row for row in report.get("steps", []) if isinstance(row, dict)}
        strict_step = steps.get("run_strict_pipeline", {})
        retry_step = steps.get("run_strict_pipeline_with_bootstrap_baseline", {})
        assert_true(str(strict_step.get("status")) == "recovered", "first strict step status=recovered")
        assert_true(bool(strict_step.get("blocking")) is False, "recovered strict step is non-blocking")
        assert_true(
            str(strict_step.get("recovered_by_step")) == "run_strict_pipeline_with_bootstrap_baseline",
            "recovered_by_step points to retry step",
        )
        assert_true(str(retry_step.get("status")) == "pass", "retry strict step status=pass")
        assert_true(bool(retry_step.get("blocking")) is True, "retry strict step remains blocking")


def test_wrapper_report_contract_v2_fields_present() -> None:
    print("\n=== run_productized_workflow: report contract v2 fields are present ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "proj"
        request_path = _prepare_wrapper_minimal_project(project_root)
        evidence = project_root / "evidence"

        fake_python = td / "fake_py.sh"
        fake_python.write_text(
            "#!/bin/sh\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        wrapper_report = evidence / "productized_workflow_report.json"
        proc = run_gate(
            [
                str(SCRIPTS_DIR / "run_productized_workflow.py"),
                "--request",
                str(request_path),
                "--evidence-dir",
                str(evidence),
                "--strict",
                "--allow-missing-compare",
                "--python",
                str(fake_python),
                "--report",
                str(wrapper_report),
            ]
        )
        assert_true(proc.returncode == 0, "wrapper exits 0 when all steps pass")
        report = load_report(wrapper_report)
        assert_true(report.get("contract_version") == "productized_workflow_report.v2", "contract version is v2")
        assert_true(
            report.get("status_reason") == "all_blocking_steps_passed",
            "status_reason is all_blocking_steps_passed for clean pass",
        )
        assert_true(isinstance(report.get("blocking_failure_count"), int), "blocking_failure_count is integer")
        assert_true(isinstance(report.get("recovered_failure_count"), int), "recovered_failure_count is integer")
        assert_true(isinstance(report.get("bootstrap_recovery_applied"), bool), "bootstrap_recovery_applied is bool")
        assert_true(
            report.get("bootstrap_recovery_source") is None or isinstance(report.get("bootstrap_recovery_source"), str),
            "bootstrap_recovery_source type is valid",
        )
        steps = report.get("steps", [])
        step_fields_ok = True
        for row in steps:
            if not isinstance(row, dict):
                step_fields_ok = False
                break
            if not {"status", "blocking", "recovered_by_step"}.issubset(set(row.keys())):
                step_fields_ok = False
                break
        assert_true(step_fields_ok, "all steps contain v2 fields: status/blocking/recovered_by_step")


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
        copy_ready = report.get("copy_ready_commands", {})
        assert_true(report.get("contract_version") == "onboarding_report.v2", "onboarding preview contract_version is v2")
        assert_true(report.get("stop_on_fail") is True, "onboarding preview default stop_on_fail=true")
        assert_true(report.get("termination_reason") == "completed_successfully", "onboarding preview termination_reason is completed_successfully")
        assert_true(report.get("preview_only") is True, "onboarding preview report marks preview_only=true")
        assert_true(report.get("display_status") == "preview", "onboarding preview display_status is preview")
        assert_true(isinstance(copy_ready, dict), "onboarding preview copy_ready_commands is object")
        assert_true("authority_release" in copy_ready, "onboarding preview copy_ready_commands includes authority_release")
        workflow_compare = str(copy_ready.get("workflow_compare", ""))
        assert_true(
            str((SCRIPTS_DIR / "mlgg.py").resolve()) in workflow_compare,
            "onboarding copy-ready commands use absolute mlgg.py path",
        )
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


def test_mlgg_onboarding_guided_without_stdin_fails_closed_with_clear_code() -> None:
    print("\n=== mlgg onboarding: guided mode without stdin fails closed clearly ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_no_stdin"
        report_path = td / "onboarding_no_stdin_report.json"
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
            input_text=None,
        )
        assert_true(proc.returncode == 2, "guided onboarding without stdin exits 2")
        combined = proc.stdout + "\n" + proc.stderr
        assert_true("Traceback" not in combined, "guided onboarding without stdin does not raise traceback")
        assert_true(report_path.exists(), "guided onboarding without stdin report exists")
        report = load_report(report_path)
        failure_codes = report.get("failure_codes", [])
        assert_true(
            "onboarding_interactive_input_unavailable" in failure_codes,
            "missing stdin path emits onboarding_interactive_input_unavailable",
        )


def test_mlgg_onboarding_guided_cancel_ignores_stale_report_codes() -> None:
    print("\n=== mlgg onboarding: stale historical report codes are ignored ===")
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        project_root = td / "demo_stale"
        evidence_dir = project_root / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / "old_report.json").write_text(
            json.dumps(
                {
                    "status": "fail",
                    "failures": [{"code": "stale_old_failure"}],
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
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
        assert_true(proc.returncode == 2, "guided cancel with stale report exits 2")
        report = load_report(report_path)
        failure_codes = report.get("failure_codes", [])
        assert_true("onboarding_step_cancelled" in failure_codes, "current run cancellation code present")
        assert_true("stale_old_failure" not in failure_codes, "stale historical failure code not included")


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


def test_mlgg_interactive_help_passthrough_requires_command_and_returns_script_help() -> None:
    print("\n=== mlgg interactive: -- --help without --command returns wizard help ===")
    proc = run_gate([str(SCRIPTS_DIR / "mlgg.py"), "interactive", "--", "--help"])
    assert_true(proc.returncode == 0, "interactive passthrough help exits 0")
    body = proc.stdout + "\n" + proc.stderr
    assert_true("Interactive wizard for ml-leakage-guard core commands." in body, "interactive help content is returned")


def test_mlgg_subcommand_direct_help_for_onboarding_and_interactive_train() -> None:
    print("\n=== mlgg help: direct subcommand --help routes to target script ===")
    onboarding_help = run_gate([str(SCRIPTS_DIR / "mlgg.py"), "onboarding", "--help"])
    assert_true(onboarding_help.returncode == 0, "onboarding --help exits 0")
    onboarding_body = onboarding_help.stdout + "\n" + onboarding_help.stderr
    assert_true("Guided novice onboarding for ml-leakage-guard." in onboarding_body, "onboarding direct help is routed")
    interactive_train_help = run_gate(
        [str(SCRIPTS_DIR / "mlgg.py"), "train", "--interactive", "--help"]
    )
    assert_true(interactive_train_help.returncode == 0, "train --interactive --help exits 0")
    interactive_body = interactive_train_help.stdout + "\n" + interactive_train_help.stderr
    assert_true(
        "Interactive wizard for ml-leakage-guard core commands." in interactive_body,
        "train --interactive --help routes to interactive wizard help",
    )


def test_mlgg_subcommand_direct_help_with_global_options() -> None:
    print("\n=== mlgg help: direct subcommand help still works with global options ===")
    onboarding_help = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "--python",
            sys.executable,
            "onboarding",
            "--help",
        ]
    )
    assert_true(onboarding_help.returncode == 0, "global --python + onboarding --help exits 0")
    onboarding_body = onboarding_help.stdout + "\n" + onboarding_help.stderr
    assert_true(
        "Guided novice onboarding for ml-leakage-guard." in onboarding_body,
        "global --python still routes to onboarding help",
    )
    interactive_help = run_gate(
        [
            str(SCRIPTS_DIR / "mlgg.py"),
            "--cwd",
            "/tmp",
            "train",
            "--interactive",
            "--help",
        ]
    )
    assert_true(interactive_help.returncode == 0, "global --cwd + train --interactive --help exits 0")
    interactive_body = interactive_help.stdout + "\n" + interactive_help.stderr
    assert_true(
        "Interactive wizard for ml-leakage-guard core commands." in interactive_body,
        "global --cwd still routes to interactive wizard help",
    )


def test_mlgg_onboarding_unknown_failure_only_when_no_specific_codes() -> None:
    print("\n=== mlgg onboarding: unknown failure code is fallback only ===")
    import importlib.util

    spec = importlib.util.spec_from_file_location("mlgg_onboarding_mod", SCRIPTS_DIR / "mlgg_onboarding.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        evidence = td / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "with_code_report.json").write_text(
            json.dumps({"status": "fail", "failures": [{"code": "real_code"}]}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        (evidence / "without_code_report.json").write_text(
            json.dumps({"status": "fail"}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        codes = mod.collect_failure_codes(td, min_mtime_epoch=0.0)  # type: ignore[attr-defined]
        assert_true("real_code" in codes, "specific failure code is collected")
        assert_true("onboarding_unknown_failure" not in codes, "unknown failure not added when specific code exists")


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
    assert_true("benchmark-suite" in body, "mlgg --help lists benchmark-suite command")
    assert_true("authority-release" in body, "mlgg --help lists authority-release command")
    assert_true("authority-research-heart" in body, "mlgg --help lists authority-research-heart command")
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
    test_mlgg_interactive_authority_defaults_to_release_stress_path()
    test_mlgg_interactive_accept_defaults_non_blocking()
    test_mlgg_authority_wrapper_release_and_research_presets()
    test_mlgg_authority_wrapper_rejects_conflicting_route_flags()
    test_release_benchmark_contract_v2_fields_present()
    test_release_benchmark_forwards_authority_timeout_and_lock_args()
    test_mlgg_benchmark_passthrough_strips_separator_before_subcommand()
    test_release_benchmark_blocking_failure_sets_standard_code()
    test_release_benchmark_registry_mismatch_detected()
    test_release_benchmark_registry_failure_contract_fields_present()
    test_release_benchmark_repeat_inconsistency_detected()
    test_release_benchmark_repeat_detects_authority_metric_drift()
    test_release_benchmark_junit_marks_repeat_inconsistent_global_failure()
    test_release_benchmark_suite_timeout_is_fail_closed()
    test_release_benchmark_repeat_detects_adversarial_count_drift()
    test_release_benchmark_emits_observational_diagnostics_for_nonblocking_failures()
    test_release_benchmark_repeat_outputs_are_not_overwritten()
    test_authority_e2e_gitignore_covers_runtime_outputs()
    test_authority_e2e_run_cmd_has_subprocess_timeout_guard()
    test_mlgg_interactive_profile_value_validation_fail_closed()
    test_mlgg_interactive_workflow_default_evidence_dir_uses_request_project_base()
    test_render_user_summary_propagates_fail_status()
    test_wrapper_stale_publication_report_does_not_trigger_bootstrap_retry()
    test_wrapper_schema_preflight_failure_causes_fail_even_if_strict_pass()
    test_wrapper_bootstrap_recovered_path_reports_pass_with_recovered_step()
    test_wrapper_report_contract_v2_fields_present()
    test_mlgg_onboarding_preview_emits_full_step_plan()
    test_mlgg_onboarding_guided_cancel_has_failure_code_and_actions()
    test_mlgg_onboarding_guided_without_stdin_fails_closed_with_clear_code()
    test_mlgg_onboarding_guided_cancel_ignores_stale_report_codes()
    test_mlgg_onboarding_no_stop_on_fail_completes_with_failures()
    test_mlgg_interactive_help_passthrough_requires_command_and_returns_script_help()
    test_mlgg_subcommand_direct_help_for_onboarding_and_interactive_train()
    test_mlgg_subcommand_direct_help_with_global_options()
    test_mlgg_onboarding_unknown_failure_only_when_no_specific_codes()
    test_mlgg_onboarding_missing_openssl_fails_closed()
    test_mlgg_help_includes_onboarding_and_bootstrap_example()

    print(f"\n{'='*50}")
    if _failures:
        print(f"\033[31mFAILED {len(_failures)} test(s):\033[0m")
        for name in _failures:
            print(f"  - {name}")
        return 1
    else:
        print("\033[32mAll tests passed.\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(main())
