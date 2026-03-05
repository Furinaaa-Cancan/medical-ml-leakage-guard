#!/usr/bin/env python3
"""
DAG-based pipeline executor for ml-leakage-guard gates.

Replaces the hardcoded sequential run_strict_pipeline.py with a declarative
DAG-driven executor that provides:

  - Automatic parallelism within dependency layers
  - Incremental execution (skip gates whose inputs haven't changed)
  - Single-gate and subset re-runs
  - Checkpoint/resume after failures
  - Rich terminal progress output with severity-aware issue display
  - Unified JSON pipeline report

Usage examples:

  # Full strict pipeline
  python run_dag_pipeline.py --request request.json --strict

  # Re-run only failed gates from last checkpoint
  python run_dag_pipeline.py --request request.json --strict --resume

  # Run a single gate (with its dependencies)
  python run_dag_pipeline.py --request request.json --strict --only calibration_dca_gate

  # Run a single gate without dependencies (assumes deps already passed)
  python run_dag_pipeline.py --request request.json --strict --only calibration_dca_gate --no-deps

  # List the DAG structure
  python run_dag_pipeline.py --show-dag

  # Dry-run: validate all inputs without executing gates
  python run_dag_pipeline.py --request request.json --strict --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from _gate_utils import (
    load_json_from_path as load_json,
    resolve_path,
    write_json,
)
from _gate_registry import (
    GATE_REGISTRY,
    GateLayer,
    GateSpec,
    get_execution_layers,
    get_runnable_subset,
    print_dag_summary,
    topological_sort,
    validate_dag,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_REPORT_VERSION = "dag_pipeline_report.v1"
CHECKPOINT_FILE = ".dag_checkpoint.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DAG-based pipeline executor for ml-leakage-guard gates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --request request.json --strict
  %(prog)s --request request.json --strict --resume
  %(prog)s --request request.json --strict --only calibration_dca_gate
  %(prog)s --show-dag
  %(prog)s --request request.json --dry-run
        """,
    )

    run_group = parser.add_argument_group("Execution")
    run_group.add_argument("--request", help="Path to request JSON.")
    run_group.add_argument(
        "--evidence-dir", default="evidence",
        help="Directory for gate artifacts and reports (default: evidence).",
    )
    run_group.add_argument("--strict", action="store_true", help="Run all gates in strict mode.")
    run_group.add_argument(
        "--python", default=sys.executable,
        help="Python executable for running gate scripts.",
    )
    run_group.add_argument("--report", help="Pipeline summary report JSON path.")

    subset_group = parser.add_argument_group("Subset execution")
    subset_group.add_argument(
        "--only", nargs="+", metavar="GATE",
        help="Run only these gates (plus dependencies unless --no-deps).",
    )
    subset_group.add_argument(
        "--no-deps", action="store_true",
        help="With --only, skip dependency gates (assume they already passed).",
    )
    subset_group.add_argument(
        "--skip", nargs="+", metavar="GATE",
        help="Skip these gates (and anything that depends on them).",
    )
    subset_group.add_argument(
        "--from-gate", metavar="GATE",
        help="Start execution from this gate (skip all preceding gates).",
    )

    resume_group = parser.add_argument_group("Resume / incremental")
    resume_group.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint, skipping already-passed gates.",
    )
    resume_group.add_argument(
        "--rerun-failed", action="store_true",
        help="Re-run only gates that failed in the last checkpoint.",
    )
    resume_group.add_argument(
        "--force", action="store_true",
        help="Ignore checkpoint and run everything fresh.",
    )

    parallel_group = parser.add_argument_group("Parallelism")
    parallel_group.add_argument(
        "--parallel", action="store_true",
        help="Run independent gates concurrently within each layer.",
    )
    parallel_group.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum concurrent gate processes (default: 4).",
    )

    mode_group = parser.add_argument_group("Mode")
    mode_group.add_argument(
        "--continue-on-fail", action="store_true",
        help="Continue executing after gate failures (diagnostic mode).",
    )
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Validate DAG and input files without running gates.",
    )
    mode_group.add_argument(
        "--show-dag", action="store_true",
        help="Print the DAG structure and exit.",
    )

    manifest_group = parser.add_argument_group("Manifest comparison")
    manifest_group.add_argument("--compare-manifest", help="Baseline manifest JSON path.")
    manifest_group.add_argument(
        "--allow-missing-compare", action="store_true",
        help="Allow first-run bootstrap without manifest baseline.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint(evidence_dir: Path) -> Dict[str, Any]:
    cp_path = evidence_dir / CHECKPOINT_FILE
    if not cp_path.exists():
        return {}
    try:
        with cp_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(evidence_dir: Path, state: Dict[str, Any]) -> None:
    write_json(evidence_dir / CHECKPOINT_FILE, state)


# ---------------------------------------------------------------------------
# Gate execution
# ---------------------------------------------------------------------------

def run_gate_subprocess(
    gate_name: str,
    cmd: List[str],
) -> Dict[str, Any]:
    """Run a single gate as a subprocess and return structured result."""
    t0 = _time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=3600)
        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        exit_code = 2
        stdout = ""
        stderr = f"TIMEOUT: {gate_name} exceeded 3600s subprocess limit."
    except Exception as exc:
        exit_code = 2
        stdout = ""
        stderr = f"EXCEPTION: {exc}"

    elapsed = _time.time() - t0
    return {
        "name": gate_name,
        "command": shlex.join(cmd),
        "exit_code": exit_code,
        "status": "pass" if exit_code == 0 else "fail",
        "execution_time_seconds": round(elapsed, 3),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def build_gate_command(
    spec: GateSpec,
    args: argparse.Namespace,
    scripts_dir: Path,
    evidence_dir: Path,
    normalized: Dict[str, Any],
    report_paths: Dict[str, Path],
    split_paths: Dict[str, str],
) -> List[str]:
    """Build the CLI command for a gate based on its spec and the normalized request."""
    cmd: List[str] = [args.python, str(scripts_dir / spec.script)]

    if spec.name == "request_contract_gate":
        cmd.extend(["--request", str(Path(args.request).expanduser().resolve())])
    elif spec.name == "manifest_lock":
        cmd.extend(_build_manifest_cmd(args, normalized, scripts_dir, evidence_dir, report_paths))
        cmd.extend(["--output", str(report_paths[spec.name])])
        if args.compare_manifest:
            cmd.extend(["--compare-with", str(resolve_path(Path.cwd(), args.compare_manifest))])
        return cmd
    elif spec.name in ("publication_gate", "self_critique_gate"):
        cmd.extend(_build_aggregation_cmd(spec.name, report_paths, args))
    else:
        cmd.extend(_build_standard_gate_cmd(spec, normalized, split_paths, report_paths))

    cmd.extend(["--report", str(report_paths[spec.name])])

    if args.strict:
        cmd.append("--strict")

    return cmd


def _build_standard_gate_cmd(
    spec: GateSpec,
    normalized: Dict[str, Any],
    split_paths: Dict[str, str],
    report_paths: Dict[str, Path],
) -> List[str]:
    """Build CLI args for a standard validation gate."""
    cmd: List[str] = []

    # Split file arguments (common across many gates)
    needs_splits = {
        "leakage_gate", "split_protocol_gate", "covariate_shift_gate",
        "definition_variable_guard", "feature_lineage_gate",
        "imbalance_policy_gate", "missingness_policy_gate",
        "distribution_generalization_gate", "model_selection_audit_gate",
    }

    if spec.name in needs_splits:
        if split_paths.get("train"):
            cmd.extend(["--train", split_paths["train"]])
        if split_paths.get("valid"):
            cmd.extend(["--valid", split_paths["valid"]])
        if split_paths.get("test"):
            cmd.extend(["--test", split_paths["test"]])

    # Gate-specific request input mappings
    for req_field, cli_flag in spec.request_inputs.items():
        value = normalized.get(req_field)
        if value is not None:
            cmd.extend([cli_flag, str(value)])

    # Report input mappings (cross-gate dependencies)
    for dep_gate, cli_flag in spec.report_inputs.items():
        if dep_gate in report_paths:
            cmd.extend([cli_flag, str(report_paths[dep_gate])])

    # Gate-specific extra arguments
    cmd.extend(_gate_specific_extras(spec.name, normalized, split_paths))

    return cmd


def _gate_specific_extras(
    gate_name: str,
    normalized: Dict[str, Any],
    split_paths: Dict[str, str],
) -> List[str]:
    """Return gate-specific CLI arguments not covered by spec mappings."""
    extras: List[str] = []
    id_col = str(normalized.get("patient_id_col", ""))
    time_col = str(normalized.get("index_time_col", ""))
    label_col = str(normalized.get("label_col", ""))
    target_name = str(normalized.get("target_name", ""))
    metric_name = str(normalized.get("primary_metric", ""))
    study_id = str(normalized.get("study_id", ""))
    run_id = str(normalized.get("run_id", ""))
    valid = split_paths.get("valid", "")

    if gate_name == "leakage_gate":
        extras.extend(["--id-cols", id_col, "--time-col", time_col, "--target-col", label_col])

    elif gate_name == "split_protocol_gate":
        extras.extend(["--id-col", id_col, "--time-col", time_col, "--target-col", label_col])

    elif gate_name == "covariate_shift_gate":
        extras.extend(["--target-col", label_col, "--ignore-cols", f"{id_col},{time_col}"])

    elif gate_name == "definition_variable_guard":
        extras.extend([
            "--target", target_name, "--target-col", label_col,
            "--ignore-cols", f"{id_col},{time_col}",
        ])

    elif gate_name == "feature_lineage_gate":
        extras.extend([
            "--target", target_name, "--target-col", label_col,
            "--ignore-cols", f"{id_col},{time_col}",
        ])

    elif gate_name == "imbalance_policy_gate":
        extras.extend(["--target-col", label_col])

    elif gate_name == "missingness_policy_gate":
        extras.extend(["--target-col", label_col, "--ignore-cols", f"{id_col},{time_col}"])

    elif gate_name == "tuning_leakage_gate":
        extras.extend(["--id-col", id_col])
        if valid:
            extras.append("--has-valid-split")

    elif gate_name == "model_selection_audit_gate":
        extras.extend(["--expected-primary-metric", metric_name])

    elif gate_name == "distribution_generalization_gate":
        extras.extend([
            "--target-col", label_col,
            "--ignore-cols", f"{id_col},{time_col}",
        ])

    elif gate_name == "execution_attestation_gate":
        extras.extend(["--study-id", study_id, "--run-id", run_id])

    elif gate_name == "metric_consistency_gate":
        extras.extend([
            "--required-evaluation-split", "test",
            "--metric-name", metric_name,
            "--expected", str(normalized.get("actual_primary_metric", "")),
        ])
        eval_metric_path = normalized.get("evaluation_metric_path")
        if isinstance(eval_metric_path, str) and eval_metric_path:
            extras.extend(["--metric-path", eval_metric_path])

    elif gate_name == "evaluation_quality_gate":
        thresholds = normalized.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        extras.extend([
            "--metric-name", metric_name,
            "--primary-metric", str(normalized.get("actual_primary_metric", "")),
            "--min-resamples", str(int(float(thresholds.get("ci_min_resamples", 200)))),
            "--min-baseline-delta", str(float(thresholds.get("min_baseline_delta", 0.0))),
            "--max-ci-width", str(float(thresholds.get("ci_max_width", 0.50))),
        ])
        eval_metric_path = normalized.get("evaluation_metric_path")
        if isinstance(eval_metric_path, str) and eval_metric_path:
            extras.extend(["--metric-path", eval_metric_path])

    elif gate_name == "permutation_significance_gate":
        thresholds = normalized.get("thresholds", {})
        if not isinstance(thresholds, dict):
            thresholds = {}
        extras.extend([
            "--metric-name", metric_name,
            "--actual", str(normalized.get("actual_primary_metric", "")),
            "--alpha", str(float(thresholds.get("alpha", 0.01))),
            "--min-delta", str(float(thresholds.get("min_delta", 0.03))),
        ])

    return extras


def _build_manifest_cmd(
    args: argparse.Namespace,
    normalized: Dict[str, Any],
    scripts_dir: Path,
    evidence_dir: Path,
    report_paths: Dict[str, Path],
) -> List[str]:
    """Build manifest_lock.py --inputs list."""
    cmd: List[str] = ["--inputs"]

    split_paths = normalized.get("split_paths", {})
    if isinstance(split_paths, dict):
        for key in ("train", "valid", "test"):
            val = split_paths.get(key)
            if isinstance(val, str) and val:
                cmd.append(val)

    path_fields = [
        "phenotype_definition_spec", "feature_lineage_spec", "split_protocol_spec",
        "imbalance_policy_spec", "missingness_policy_spec", "tuning_protocol_spec",
        "reporting_bias_checklist_spec", "performance_policy_spec", "feature_group_spec",
        "model_selection_report_file", "feature_engineering_report_file",
        "distribution_report_file", "robustness_report_file",
        "seed_sensitivity_report_file", "evaluation_report_file",
        "prediction_trace_file", "external_cohort_spec",
        "external_validation_report_file", "ci_matrix_report_file",
        "permutation_null_metrics_file", "execution_attestation_spec",
    ]
    for field in path_fields:
        val = normalized.get(field)
        if isinstance(val, str) and val:
            cmd.append(val)

    cmd.append(str(Path(args.request).expanduser().resolve()))

    for name in sorted(GATE_REGISTRY.keys()):
        spec = GATE_REGISTRY[name]
        cmd.append(str(scripts_dir / spec.script))

    cmd.append(str(scripts_dir / "run_dag_pipeline.py"))

    return cmd


def _build_aggregation_cmd(
    gate_name: str,
    report_paths: Dict[str, Path],
    args: argparse.Namespace,
) -> List[str]:
    """Build CLI args for publication_gate and self_critique_gate."""
    cmd: List[str] = []

    report_flag_map = {
        "request_contract_gate": "--request-report",
        "manifest_lock": "--manifest",
        "execution_attestation_gate": "--execution-attestation-report",
        "reporting_bias_gate": "--reporting-bias-report",
        "leakage_gate": "--leakage-report",
        "split_protocol_gate": "--split-protocol-report",
        "covariate_shift_gate": "--covariate-shift-report",
        "definition_variable_guard": "--definition-report",
        "feature_lineage_gate": "--lineage-report",
        "imbalance_policy_gate": "--imbalance-report",
        "missingness_policy_gate": "--missingness-report",
        "tuning_leakage_gate": "--tuning-report",
        "model_selection_audit_gate": "--model-selection-audit-report",
        "feature_engineering_audit_gate": "--feature-engineering-audit-report",
        "clinical_metrics_gate": "--clinical-metrics-report",
        "prediction_replay_gate": "--prediction-replay-report",
        "distribution_generalization_gate": "--distribution-generalization-report",
        "generalization_gap_gate": "--generalization-gap-report",
        "robustness_gate": "--robustness-report",
        "seed_stability_gate": "--seed-stability-report",
        "external_validation_gate": "--external-validation-report",
        "calibration_dca_gate": "--calibration-dca-report",
        "ci_matrix_gate": "--ci-matrix-report",
        "metric_consistency_gate": "--metric-report",
        "evaluation_quality_gate": "--evaluation-quality-report",
        "permutation_significance_gate": "--permutation-report",
    }

    if gate_name == "self_critique_gate":
        report_flag_map["publication_gate"] = "--publication-report"

    for dep_name, flag in report_flag_map.items():
        if dep_name in report_paths:
            cmd.extend([flag, str(report_paths[dep_name])])

    if gate_name == "self_critique_gate":
        cmd.extend(["--min-score", "95"])
        if getattr(args, "allow_missing_compare", False):
            cmd.append("--allow-missing-comparison")

    return cmd


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_layer_header(layer_idx: int, gate_names: List[str], parallel: bool) -> None:
    layer_name = GateLayer(layer_idx).name if layer_idx < len(GateLayer) else f"LAYER_{layer_idx}"
    mode = "parallel" if parallel and len(gate_names) > 1 else "sequential"
    color = _use_color()
    if color:
        print(f"\n\033[1;36m{'─' * 60}\033[0m")
        print(f"\033[1;36m  Layer {layer_idx} ({layer_name}) [{mode}] — {len(gate_names)} gate(s)\033[0m")
        print(f"\033[1;36m{'─' * 60}\033[0m")
    else:
        print(f"\n{'─' * 60}")
        print(f"  Layer {layer_idx} ({layer_name}) [{mode}] — {len(gate_names)} gate(s)")
        print(f"{'─' * 60}")


def print_gate_start(gate_name: str, cmd: List[str]) -> None:
    color = _use_color()
    if color:
        print(f"\n  \033[1m▶ {gate_name}\033[0m")
    else:
        print(f"\n  > {gate_name}")
    print(f"    $ {shlex.join(cmd)}")


def print_gate_result(result: Dict[str, Any]) -> None:
    color = _use_color()
    name = result["name"]
    status = result["status"]
    elapsed = result.get("execution_time_seconds", 0)

    if color:
        if status == "pass":
            icon = "\033[32m✓\033[0m"
        elif status == "skip":
            icon = "\033[33m⊘\033[0m"
        else:
            icon = "\033[1;31m✗\033[0m"
        print(f"    {icon} {name}  ({elapsed:.1f}s)")
    else:
        icon = "OK" if status == "pass" else ("SKIP" if status == "skip" else "FAIL")
        print(f"    [{icon}] {name}  ({elapsed:.1f}s)")

    stderr_tail = result.get("stderr_tail", "").strip()
    if status == "fail" and stderr_tail:
        for line in stderr_tail.split("\n")[-5:]:
            print(f"      {line}")


def print_pipeline_summary(steps: List[Dict[str, Any]], elapsed: float) -> None:
    color = _use_color()
    passed = sum(1 for s in steps if s["status"] == "pass")
    failed = sum(1 for s in steps if s["status"] == "fail")
    skipped = sum(1 for s in steps if s["status"] == "skip")
    total = len(steps)

    print(f"\n{'═' * 60}")
    if color:
        status_str = (
            "\033[1;32mALL PASSED\033[0m" if failed == 0
            else f"\033[1;31m{failed} FAILED\033[0m"
        )
        print(f"  Pipeline: {status_str}  ({passed}/{total} passed, {skipped} skipped, {elapsed:.1f}s)")
    else:
        status_str = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
        print(f"  Pipeline: {status_str}  ({passed}/{total} passed, {skipped} skipped, {elapsed:.1f}s)")
    print(f"{'═' * 60}")

    if failed > 0:
        print("\n  Failed gates:")
        for s in steps:
            if s["status"] == "fail":
                print(f"    ✗ {s['name']}")

    timed = sorted(
        [s for s in steps if s.get("execution_time_seconds", 0) > 0],
        key=lambda s: s["execution_time_seconds"],
        reverse=True,
    )
    if timed:
        print("\n  Slowest gates:")
        for s in timed[:5]:
            print(f"    {s['execution_time_seconds']:7.1f}s  {s['name']}")
    print()


# ---------------------------------------------------------------------------
# Main execution engine
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.show_dag:
        dag_errors = validate_dag()
        if dag_errors:
            for err in dag_errors:
                print(f"[DAG ERROR] {err}", file=sys.stderr)
            return 2
        print_dag_summary()
        return 0

    if not args.request:
        print("[FAIL] --request is required.", file=sys.stderr)
        return 2

    if not args.strict:
        print(
            "[FAIL] run_dag_pipeline.py requires --strict for publication-grade. "
            "Re-run with --strict.",
            file=sys.stderr,
        )
        return 2

    request_path = Path(args.request).expanduser().resolve()
    if not request_path.exists():
        print(f"[FAIL] Request file not found: {request_path}", file=sys.stderr)
        return 2

    scripts_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    evidence_dir = resolve_path(cwd, args.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Validate DAG
    dag_errors = validate_dag()
    if dag_errors:
        for err in dag_errors:
            print(f"[DAG ERROR] {err}", file=sys.stderr)
        return 2

    # Step 1: Run request_contract_gate to get normalized request
    report_paths: Dict[str, Path] = {}
    for name, spec in GATE_REGISTRY.items():
        if spec.report_output:
            report_paths[name] = evidence_dir / spec.report_output

    request_cmd = [
        args.python,
        str(scripts_dir / "request_contract_gate.py"),
        "--request", str(request_path),
        "--report", str(report_paths["request_contract_gate"]),
    ]
    if args.strict:
        request_cmd.append("--strict")

    steps: List[Dict[str, Any]] = []
    pipeline_t0 = _time.time()

    print_gate_start("request_contract_gate", request_cmd)
    result = run_gate_subprocess("request_contract_gate", request_cmd)
    steps.append(result)
    print_gate_result(result)

    if result["exit_code"] != 0:
        print("[FAIL] request_contract_gate failed. Cannot proceed.", file=sys.stderr)
        return _finalize(args, evidence_dir, steps, False, pipeline_t0)

    # Load normalized request
    request_report = load_json(report_paths["request_contract_gate"])
    normalized = request_report.get("normalized_request", {})
    if not isinstance(normalized, dict):
        print("[FAIL] request_contract_report missing normalized_request.", file=sys.stderr)
        return _finalize(args, evidence_dir, steps, False, pipeline_t0)

    claim_tier = str(normalized.get("claim_tier_target", ""))
    if claim_tier != "publication-grade":
        print(
            f"[FAIL] Only publication-grade supported (got: {claim_tier}).",
            file=sys.stderr,
        )
        return _finalize(args, evidence_dir, steps, False, pipeline_t0)

    split_paths_raw = normalized.get("split_paths", {})
    split_paths: Dict[str, str] = {}
    if isinstance(split_paths_raw, dict):
        for key in ("train", "valid", "test"):
            val = split_paths_raw.get(key)
            if isinstance(val, str) and val:
                split_paths[key] = val

    # Load checkpoint for resume
    checkpoint = load_checkpoint(evidence_dir) if (args.resume or args.rerun_failed) and not args.force else {}
    passed_gates: Set[str] = set(checkpoint.get("passed_gates", []))

    # Validate checkpoint: remove gates whose report files are missing
    if passed_gates:
        invalidated: List[str] = []
        for pg in sorted(passed_gates):
            if pg in report_paths and not report_paths[pg].exists():
                invalidated.append(pg)
        if invalidated:
            print(f"[WARN] Checkpoint reports missing for: {', '.join(invalidated)}. "
                  f"These gates will be re-run.", file=sys.stderr)
            passed_gates -= set(invalidated)

    # Determine which gates to run
    all_gates = topological_sort()
    gates_to_run = _compute_gates_to_run(args, all_gates, passed_gates)

    if args.dry_run:
        print("\n[DRY-RUN] Would execute these gates:")
        for g in gates_to_run:
            spec = GATE_REGISTRY.get(g)
            if spec:
                print(f"  Layer {spec.layer.value}: {g}")
        return 0

    # Execute gates layer by layer
    had_failure = False
    continue_on_fail = bool(args.continue_on_fail)
    newly_passed: Set[str] = set()

    execution_layers = get_execution_layers()
    for layer_idx, layer_gates in execution_layers:
        runnable_in_layer = [g for g in layer_gates if g in gates_to_run and g != "request_contract_gate"]
        if not runnable_in_layer:
            continue

        # Check if all dependencies have passed
        blocked: List[str] = []
        ready: List[str] = []
        for gate_name in runnable_in_layer:
            spec = GATE_REGISTRY[gate_name]
            deps_met = all(
                d in passed_gates or d in newly_passed or d not in gates_to_run
                for d in spec.depends_on
            )
            if deps_met:
                ready.append(gate_name)
            else:
                blocked.append(gate_name)

        if blocked and not continue_on_fail:
            for bg in blocked:
                steps.append({
                    "name": bg, "command": "", "exit_code": -1,
                    "status": "skip", "execution_time_seconds": 0,
                    "stdout_tail": "", "stderr_tail": "Skipped: dependency not met",
                })
            continue

        if not ready:
            continue

        use_parallel = args.parallel and len(ready) > 1
        print_layer_header(layer_idx, ready, use_parallel)

        if use_parallel:
            layer_results = _run_parallel(
                ready, args, scripts_dir, evidence_dir, normalized,
                report_paths, split_paths, args.max_workers,
            )
        else:
            layer_results = _run_sequential(
                ready, args, scripts_dir, evidence_dir, normalized,
                report_paths, split_paths,
            )

        for r in layer_results:
            steps.append(r)
            print_gate_result(r)
            if r["status"] == "pass":
                newly_passed.add(r["name"])
            elif r["status"] == "fail":
                had_failure = True
                if not continue_on_fail:
                    # Save checkpoint before early exit
                    _save_progress(evidence_dir, passed_gates | newly_passed, steps)
                    return _finalize(args, evidence_dir, steps, False, pipeline_t0)

    # Save final checkpoint
    all_passed = passed_gates | newly_passed
    _save_progress(evidence_dir, all_passed, steps)

    success = not had_failure
    return _finalize(args, evidence_dir, steps, success, pipeline_t0)


def _compute_gates_to_run(
    args: argparse.Namespace,
    all_gates: List[str],
    passed_gates: Set[str],
) -> List[str]:
    """Compute which gates should be executed based on CLI flags."""
    if args.only:
        include_deps = not args.no_deps
        return get_runnable_subset(args.only, include_dependencies=include_deps)

    if args.rerun_failed:
        return [g for g in all_gates if g not in passed_gates]

    if args.resume:
        return [g for g in all_gates if g not in passed_gates]

    gates = list(all_gates)

    if args.from_gate:
        try:
            idx = gates.index(args.from_gate)
            gates = gates[idx:]
        except ValueError:
            print(f"[WARN] --from-gate '{args.from_gate}' not found, running all.", file=sys.stderr)

    if args.skip:
        skip_set = set(args.skip)
        # Also skip anything that transitively depends on skipped gates
        from _gate_registry import get_dependents
        to_skip: Set[str] = set()
        queue = list(skip_set)
        while queue:
            current = queue.pop(0)
            if current in to_skip:
                continue
            to_skip.add(current)
            for dep in get_dependents(current):
                if dep not in to_skip:
                    queue.append(dep)
        gates = [g for g in gates if g not in to_skip]

    return gates


def _run_sequential(
    gate_names: List[str],
    args: argparse.Namespace,
    scripts_dir: Path,
    evidence_dir: Path,
    normalized: Dict[str, Any],
    report_paths: Dict[str, Path],
    split_paths: Dict[str, str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    failed = False
    for i, gate_name in enumerate(gate_names):
        if failed:
            results.append({
                "name": gate_name, "command": "", "exit_code": -1,
                "status": "skip", "execution_time_seconds": 0,
                "stdout_tail": "", "stderr_tail": "Skipped: earlier gate in layer failed",
            })
            continue
        spec = GATE_REGISTRY[gate_name]
        cmd = build_gate_command(spec, args, scripts_dir, evidence_dir, normalized, report_paths, split_paths)
        print_gate_start(gate_name, cmd)
        result = run_gate_subprocess(gate_name, cmd)
        results.append(result)
        if result["exit_code"] != 0 and not args.continue_on_fail:
            failed = True
    return results


def _run_parallel(
    gate_names: List[str],
    args: argparse.Namespace,
    scripts_dir: Path,
    evidence_dir: Path,
    normalized: Dict[str, Any],
    report_paths: Dict[str, Path],
    split_paths: Dict[str, str],
    max_workers: int,
) -> List[Dict[str, Any]]:
    tasks: List[Tuple[str, List[str]]] = []
    for gate_name in gate_names:
        spec = GATE_REGISTRY[gate_name]
        cmd = build_gate_command(spec, args, scripts_dir, evidence_dir, normalized, report_paths, split_paths)
        tasks.append((gate_name, cmd))

    results: List[Dict[str, Any]] = [{}] * len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        future_map = {
            pool.submit(run_gate_subprocess, name, cmd): i
            for i, (name, cmd) in enumerate(tasks)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()

    return results


def _save_progress(
    evidence_dir: Path,
    passed_gates: Set[str],
    steps: List[Dict[str, Any]],
) -> None:
    save_checkpoint(evidence_dir, {
        "passed_gates": sorted(passed_gates),
        "last_run_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "step_summary": [
            {"name": s["name"], "status": s["status"]}
            for s in steps
        ],
    })


def _finalize(
    args: argparse.Namespace,
    evidence_dir: Path,
    steps: List[Dict[str, Any]],
    success: bool,
    pipeline_t0: float,
) -> int:
    elapsed = _time.time() - pipeline_t0

    print_pipeline_summary(steps, elapsed)

    summary = {
        "contract_version": PIPELINE_REPORT_VERSION,
        "status": "pass" if success else "fail",
        "strict_mode": bool(args.strict),
        "diagnostic_only": bool(args.continue_on_fail),
        "publication_eligible": bool(args.strict and not args.continue_on_fail and success),
        "failure_count": sum(1 for s in steps if s.get("status") == "fail"),
        "pass_count": sum(1 for s in steps if s.get("status") == "pass"),
        "skip_count": sum(1 for s in steps if s.get("status") == "skip"),
        "total_execution_time_seconds": round(elapsed, 3),
        "steps": steps,
        "evidence_dir": str(evidence_dir),
    }

    out_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else (evidence_dir / "dag_pipeline_report.json")
    )
    write_json(out_path, summary)
    print(f"Pipeline report: {out_path}")

    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
