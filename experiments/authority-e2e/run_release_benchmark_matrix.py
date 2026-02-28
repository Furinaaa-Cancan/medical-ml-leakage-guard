#!/usr/bin/env python3
"""Run structured release benchmark matrix with registry lock and repeat checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
AUTHORITY_SCRIPT = EXPERIMENT_ROOT / "run_authority_e2e.py"
ADVERSARIAL_SCRIPT = EXPERIMENT_ROOT / "run_adversarial_gate_checks.py"
DEFAULT_REGISTRY_FILE = REPO_ROOT / "references" / "benchmark-registry.json"

FAIL_BENCHMARK_REGISTRY_MISSING = "benchmark_registry_missing"
FAIL_BENCHMARK_REGISTRY_MISMATCH = "benchmark_registry_mismatch"
FAIL_BENCHMARK_REPEAT_INCONSISTENT = "benchmark_repeat_inconsistent"
FAIL_BENCHMARK_BLOCKING_SUITE_FAILED = "benchmark_blocking_suite_failed"
FAIL_BENCHMARK_SUITE_TIMEOUT = "benchmark_suite_timeout"


@dataclass
class SuiteSpec:
    suite_id: str
    name: str
    kind: str
    blocking: bool
    command: List[str]
    summary_file: Path
    expected_case_ids: List[str]


def utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    )
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object root: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        token = str(raw).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def coerce_string_list(raw: Any) -> List[str]:
    if isinstance(raw, str):
        token = raw.strip()
        return [token] if token else []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if token:
            out.append(token)
    return out


def coerce_failure_codes(raw: Any, *, fallback: Optional[str] = None) -> List[str]:
    codes = coerce_string_list(raw)
    if isinstance(fallback, str) and fallback.strip():
        codes.append(fallback.strip())
    return dedup_keep_order(codes)


def parse_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run release benchmark matrix for ml-leakage-guard.")
    parser.add_argument(
        "--profile",
        choices=["quick", "release", "extended"],
        default="release",
        help=(
            "Benchmark profile: quick=release authority + adversarial, "
            "release=quick + large-case authority, extended=release + heart research stress."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENT_ROOT / "release_benchmark_matrix_summary.json"),
        help="Path to write matrix summary JSON.",
    )
    parser.add_argument(
        "--registry-file",
        default=str(DEFAULT_REGISTRY_FILE),
        help="Dataset benchmark registry JSON (contract: benchmark_registry.v1).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(EXPERIMENT_ROOT / "_benchmark_matrix_runs"),
        help="Directory for per-suite summary artifacts.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run nested scripts.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag. If omitted, UTC timestamp token is used.",
    )
    parser.add_argument(
        "--stress-seed-min",
        type=int,
        default=20250003,
        help="Heart research stress minimum seed (extended profile only).",
    )
    parser.add_argument(
        "--stress-seed-max",
        type=int,
        default=20250060,
        help="Heart research stress maximum seed (extended profile only).",
    )
    parser.add_argument(
        "--heart-blocking",
        action="store_true",
        help="Treat heart research suite as blocking in extended profile.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Repeat each suite run N times for stability consistency check (default: 3).",
    )
    parser.add_argument(
        "--fail-on-repeat-inconsistency",
        dest="fail_on_repeat_inconsistency",
        action="store_true",
        default=True,
        help="Fail closed when repeat runs produce inconsistent suite conclusions (default: true).",
    )
    parser.add_argument(
        "--no-fail-on-repeat-inconsistency",
        dest="fail_on_repeat_inconsistency",
        action="store_false",
        help="Do not fail when repeat conclusions are inconsistent (diagnostic mode only).",
    )
    parser.add_argument(
        "--emit-junit",
        default="",
        help="Optional JUnit XML output path for CI consumption.",
    )
    parser.add_argument(
        "--suite-timeout-seconds",
        type=int,
        default=7200,
        help="Per-suite subprocess timeout in seconds (default: 7200).",
    )
    parser.add_argument(
        "--observational-diagnostics-out",
        default="",
        help=(
            "Optional path to write non-blocking failure diagnostics JSON. "
            "If omitted, a sidecar file is written next to --output."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write dry-run report without execution.",
    )
    return parser.parse_args()


def _coerce_string_list(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list):
        return None
    values: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        token = item.strip()
        if not token:
            return None
        values.append(token)
    return values


def load_and_validate_registry(registry_file: Path, profile: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    if not registry_file.exists():
        return None, FAIL_BENCHMARK_REGISTRY_MISSING, f"Registry file not found: {registry_file}"
    try:
        registry = load_json(registry_file)
    except Exception as exc:
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"Registry parse error: {exc}"
    if str(registry.get("contract_version", "")).strip() != "benchmark_registry.v1":
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, "registry.contract_version must be benchmark_registry.v1."
    profiles = registry.get("profiles")
    if not isinstance(profiles, dict):
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, "registry.profiles must be an object."
    profile_payload = profiles.get(profile)
    if not isinstance(profile_payload, dict):
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry.profiles.{profile} missing."
    suites_raw = profile_payload.get("suites")
    if not isinstance(suites_raw, list) or not suites_raw:
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry.profiles.{profile}.suites must be non-empty list."
    for idx, row in enumerate(suites_raw):
        if not isinstance(row, dict):
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry suite row #{idx} must be object."
        if not isinstance(row.get("suite_id"), str) or not str(row.get("suite_id")).strip():
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry suite row #{idx} missing suite_id."
        kind = str(row.get("kind", "")).strip().lower()
        if kind not in {"authority", "adversarial"}:
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry suite {row.get('suite_id')} invalid kind={kind}."
        args = _coerce_string_list(row.get("args", []))
        if args is None:
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry suite {row.get('suite_id')} args must be string list."
        expected_case_ids = _coerce_string_list(row.get("expected_case_ids", []))
        if expected_case_ids is None:
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry suite {row.get('suite_id')} expected_case_ids must be string list."
    dataset_fps = registry.get("dataset_fingerprints")
    if not isinstance(dataset_fps, dict) or not dataset_fps:
        return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, "registry.dataset_fingerprints must be non-empty object."
    for case_id, case_row in dataset_fps.items():
        if not isinstance(case_id, str) or not case_id.strip() or not isinstance(case_row, dict):
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, "registry dataset_fingerprints rows must be objects keyed by case_id."
        raw_files = case_row.get("raw_files")
        if not isinstance(raw_files, list) or not raw_files:
            return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry dataset_fingerprints.{case_id}.raw_files must be non-empty list."
        for fp in raw_files:
            if not isinstance(fp, dict):
                return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry dataset_fingerprints.{case_id}.raw_files entries must be objects."
            path_token = fp.get("path")
            sha_token = fp.get("sha256")
            if not isinstance(path_token, str) or not path_token.strip() or not isinstance(sha_token, str) or len(sha_token.strip()) != 64:
                return None, FAIL_BENCHMARK_REGISTRY_MISMATCH, f"registry dataset_fingerprints.{case_id} invalid raw file fingerprint row."
    return registry, None, None


def validate_dataset_fingerprints(registry: Dict[str, Any], registry_file: Path) -> Tuple[bool, Optional[str]]:
    dataset_fps = registry.get("dataset_fingerprints", {})
    for case_id, case_row in dataset_fps.items():
        raw_files = case_row.get("raw_files", [])
        aggregate_rows: List[str] = []
        for fp in raw_files:
            raw_path = Path(str(fp.get("path", ""))).expanduser()
            if not raw_path.is_absolute():
                raw_path = (REPO_ROOT / raw_path).resolve()
            expected_sha = str(fp.get("sha256", "")).strip().lower()
            if not raw_path.exists():
                return False, f"registry raw file missing for case={case_id}: {raw_path}"
            observed_sha = sha256_file(raw_path).lower()
            if observed_sha != expected_sha:
                return (
                    False,
                    f"registry sha mismatch for case={case_id}: file={raw_path} expected={expected_sha} observed={observed_sha}",
                )
            try:
                path_token = str(raw_path.relative_to(REPO_ROOT))
            except Exception:
                path_token = str(raw_path)
            aggregate_rows.append(f"{path_token}={observed_sha}")
        aggregate_rows.sort()
        observed_aggregate = hashlib.sha256("\n".join(aggregate_rows).encode("utf-8")).hexdigest()
        expected_aggregate = str(case_row.get("aggregate_sha256", "")).strip().lower()
        if expected_aggregate and observed_aggregate != expected_aggregate:
            return (
                False,
                f"registry aggregate sha mismatch for case={case_id}: expected={expected_aggregate} observed={observed_aggregate}",
            )
    return True, None


def build_suites_from_registry(
    *,
    registry: Dict[str, Any],
    profile: str,
    run_tag: str,
    artifacts_dir: Path,
    python_bin: str,
    stress_seed_min: int,
    stress_seed_max: int,
    heart_blocking: bool,
) -> List[SuiteSpec]:
    profile_payload = registry["profiles"][profile]
    suites_raw = profile_payload["suites"]
    suites: List[SuiteSpec] = []
    replacement = {
        "__STRESS_SEED_MIN__": str(int(stress_seed_min)),
        "__STRESS_SEED_MAX__": str(int(stress_seed_max)),
    }
    for row in suites_raw:
        suite_id = str(row.get("suite_id", "")).strip()
        name = str(row.get("name", suite_id)).strip() or suite_id
        kind = str(row.get("kind", "")).strip().lower()
        blocking = parse_bool(row.get("blocking", True), default=True)
        if suite_id == "authority_research_heart" and kind == "authority":
            # explicit CLI override for extended profile heart route
            blocking = bool(heart_blocking)
        args_template = _coerce_string_list(row.get("args", [])) or []
        args_expanded = [replacement.get(token, token) for token in args_template]
        expected_case_ids = _coerce_string_list(row.get("expected_case_ids", [])) or []
        if kind == "authority":
            summary_file = artifacts_dir / f"{run_tag}.{suite_id}.authority_summary.json"
            command = [
                python_bin,
                str(AUTHORITY_SCRIPT),
                "--summary-file",
                str(summary_file),
                *args_expanded,
            ]
        else:
            summary_file = artifacts_dir / f"{run_tag}.{suite_id}.adversarial_summary.json"
            command = [
                python_bin,
                str(ADVERSARIAL_SCRIPT),
                "--output",
                str(summary_file),
                *args_expanded,
            ]
        suites.append(
            SuiteSpec(
                suite_id=suite_id,
                name=name,
                kind=kind,
                blocking=blocking,
                command=command,
                summary_file=summary_file,
                expected_case_ids=expected_case_ids,
            )
        )
    return suites


def parse_suite_outcome(kind: str, summary_file: Path, exit_code: int, expected_case_ids: List[str]) -> Dict[str, Any]:
    if not summary_file.exists():
        return {
            "status": "fail",
            "failure_reason": "summary_missing",
            "failure_codes": ["summary_missing"],
            "exit_code": int(exit_code),
            "overall_status": None,
        }
    try:
        payload = load_json(summary_file)
    except Exception as exc:
        return {
            "status": "fail",
            "failure_reason": "summary_parse_error",
            "failure_codes": ["summary_parse_error"],
            "exit_code": int(exit_code),
            "overall_status": None,
            "error": str(exc),
        }

    overall = str(payload.get("overall_status", "")).strip().lower()
    status = "pass" if exit_code == 0 and overall == "pass" else "fail"
    failure_codes: List[str] = []
    detail: Dict[str, Any] = {
        "status": status,
        "failure_reason": None if status == "pass" else "suite_failed",
        "failure_codes": failure_codes,
        "exit_code": int(exit_code),
        "overall_status": overall or None,
    }
    if kind == "adversarial":
        passed_count = int(payload.get("passed_count", 0) or 0)
        scenario_count = int(payload.get("scenario_count", 0) or 0)
        detail["passed_count"] = passed_count
        detail["scenario_count"] = scenario_count
        rows = payload.get("results")
        if isinstance(rows, list):
            failed_scenarios = [row for row in rows if isinstance(row, dict) and not bool(row.get("passed"))]
            detail["failed_scenarios"] = len(failed_scenarios)
            for row in failed_scenarios:
                codes = row.get("observed_codes")
                if isinstance(codes, list):
                    for code in codes:
                        if isinstance(code, str) and code.strip():
                            failure_codes.append(code.strip())
        if status == "pass" and passed_count != scenario_count:
            detail["status"] = "fail"
            detail["failure_reason"] = "adversarial_partial_pass"
            failure_codes.append("adversarial_partial_pass")
    elif kind == "authority":
        results = payload.get("results")
        if isinstance(results, list):
            detail["case_count"] = len(results)
            detail["failed_cases"] = [
                str(row.get("case_id"))
                for row in results
                if isinstance(row, dict) and str(row.get("status")) != "pass"
            ]
            actual_case_ids = sorted(
                str(row.get("case_id")).strip()
                for row in results
                if isinstance(row, dict) and isinstance(row.get("case_id"), str) and str(row.get("case_id")).strip()
            )
            detail["actual_case_ids"] = actual_case_ids
            if expected_case_ids:
                expected_sorted = sorted(expected_case_ids)
                detail["expected_case_ids"] = expected_sorted
                if actual_case_ids != expected_sorted:
                    detail["status"] = "fail"
                    detail["failure_reason"] = "authority_case_set_mismatch"
                    failure_codes.append("authority_case_set_mismatch")
            for row in results:
                if not isinstance(row, dict):
                    continue
                if str(row.get("status")) == "pass":
                    continue
                code = row.get("failure_code")
                if isinstance(code, str) and code.strip():
                    failure_codes.append(code.strip())
                code_primary = row.get("root_failure_code_primary")
                if isinstance(code_primary, str) and code_primary.strip():
                    failure_codes.append(code_primary.strip())
                root_codes = row.get("root_failure_codes")
                if isinstance(root_codes, list):
                    for token in root_codes:
                        if isinstance(token, str) and token.strip():
                            failure_codes.append(token.strip())
            metrics_rows: List[str] = []
            for row in results:
                if not isinstance(row, dict):
                    continue
                case_id = str(row.get("case_id", "")).strip()
                metrics = row.get("metrics")
                if not case_id or not isinstance(metrics, dict):
                    continue
                metric_items: List[str] = []
                for metric_key in ("pr_auc", "roc_auc", "f2_beta", "brier"):
                    value = metrics.get(metric_key)
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)):
                        metric_items.append(f"{metric_key}={float(value):.8f}")
                if metric_items:
                    metrics_rows.append(f"{case_id}|{'|'.join(metric_items)}")
            metrics_rows.sort()
            if metrics_rows:
                detail["case_metrics_fingerprint"] = hashlib.sha256(
                    "\n".join(metrics_rows).encode("utf-8")
                ).hexdigest()
            if detail.get("failed_cases") and not failure_codes:
                failure_codes.append("authority_case_failed")
    if detail["status"] != "pass" and not failure_codes:
        failure_codes.append("suite_failed")
    detail["failure_codes"] = dedup_keep_order(failure_codes)
    return detail


def suite_signature(row: Dict[str, Any]) -> str:
    def norm_str_list(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            token = str(item).strip()
            if token:
                out.append(token)
        return sorted(out)

    def norm_int(raw: Any) -> Optional[int]:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return int(raw)
        if isinstance(raw, float):
            return int(raw)
        if isinstance(raw, str):
            token = raw.strip()
            if not token:
                return None
            try:
                return int(token)
            except Exception:
                return None
        return None

    payload = {
        "status": str(row.get("status")),
        "overall_status": str(row.get("overall_status")),
        "failure_reason": str(row.get("failure_reason", "")),
        "failure_codes": sorted(str(x) for x in row.get("failure_codes", []) if isinstance(x, str)),
        "case_metrics_fingerprint": str(row.get("case_metrics_fingerprint", "")),
        "case_count": norm_int(row.get("case_count")),
        "failed_cases": norm_str_list(row.get("failed_cases")),
        "actual_case_ids": norm_str_list(row.get("actual_case_ids")),
        "expected_case_ids": norm_str_list(row.get("expected_case_ids")),
        "passed_count": norm_int(row.get("passed_count")),
        "scenario_count": norm_int(row.get("scenario_count")),
        "failed_scenarios": norm_int(row.get("failed_scenarios")),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def write_junit(
    path: Path,
    suite_runs: List[Dict[str, Any]],
    overall_status: str,
    failure_codes: List[str],
    status_reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tests = len(suite_runs) if suite_runs else 1
    failures = sum(1 for row in suite_runs if str(row.get("status")) not in {"pass", "dry_run"})
    needs_global_failure = bool(suite_runs and overall_status != "pass" and failures == 0)
    if needs_global_failure:
        tests += 1
        failures += 1
    root = ET.Element(
        "testsuite",
        attrib={
            "name": "mlgg.release_benchmark_matrix",
            "tests": str(tests),
            "failures": str(failures),
        },
    )
    if suite_runs:
        for row in suite_runs:
            case = ET.SubElement(
                root,
                "testcase",
                attrib={
                    "classname": "mlgg.benchmark",
                    "name": f"{row.get('suite_id')}.repeat{row.get('repeat_index')}",
                },
            )
            status = str(row.get("status", "")).strip().lower()
            if status not in {"pass", "dry_run"}:
                ET.SubElement(
                    case,
                    "failure",
                    attrib={"message": ",".join(row.get("failure_codes", []) or ["suite_failed"])},
                ).text = str(row.get("stderr_tail", ""))[:2000]
        if needs_global_failure:
            case = ET.SubElement(root, "testcase", attrib={"classname": "mlgg.benchmark", "name": "global"})
            ET.SubElement(
                case,
                "failure",
                attrib={"message": ",".join(failure_codes or [status_reason or "benchmark_failed"])},
            )
    else:
        case = ET.SubElement(root, "testcase", attrib={"classname": "mlgg.benchmark", "name": "global"})
        if overall_status != "pass":
            ET.SubElement(
                case,
                "failure",
                attrib={"message": ",".join(failure_codes or ["benchmark_failed"])},
            )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _to_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            return float(token)
        except Exception:
            return None
    return None


def extract_floor_violations(clinical_floor_gap_summary: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(clinical_floor_gap_summary, dict):
        return out

    def collect_rows(
        *,
        scope: str,
        cohort_id: Optional[str],
        cohort_type: Optional[str],
        metrics_payload: Any,
    ) -> None:
        if not isinstance(metrics_payload, dict):
            return
        for metric_name, metric_row in metrics_payload.items():
            if not isinstance(metric_row, dict):
                continue
            margin = _to_float_or_none(metric_row.get("margin"))
            met = bool(metric_row.get("met"))
            if margin is None:
                continue
            if met and margin >= 0.0:
                continue
            out.append(
                {
                    "scope": scope,
                    "cohort_id": cohort_id,
                    "cohort_type": cohort_type,
                    "metric": str(metric_name),
                    "required_min": _to_float_or_none(metric_row.get("required_min")),
                    "observed": _to_float_or_none(metric_row.get("observed")),
                    "margin": float(margin),
                }
            )

    internal = clinical_floor_gap_summary.get("internal_test")
    if isinstance(internal, dict):
        collect_rows(
            scope="internal_test",
            cohort_id="internal_test",
            cohort_type=None,
            metrics_payload=internal.get("floor_metrics"),
        )
    external_rows = clinical_floor_gap_summary.get("external_cohorts")
    if isinstance(external_rows, list):
        for row in external_rows:
            if not isinstance(row, dict):
                continue
            collect_rows(
                scope="external",
                cohort_id=str(row.get("cohort_id", "")).strip() or None,
                cohort_type=str(row.get("cohort_type", "")).strip() or None,
                metrics_payload=row.get("floor_metrics"),
            )
    out.sort(key=lambda row: float(row.get("margin", 0.0)))
    return out


def build_observational_diagnostics(suite_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []
    for suite in suite_rows:
        if bool(suite.get("blocking")):
            continue
        if str(suite.get("status", "")).strip().lower() == "pass":
            continue
        if str(suite.get("kind", "")).strip().lower() != "authority":
            continue

        summary_file = Path(str(suite.get("summary_file", "")).strip())
        summary_payload: Dict[str, Any] = {}
        parse_error: Optional[str] = None
        try:
            summary_payload = load_json(summary_file)
        except Exception as exc:
            parse_error = str(exc)

        case_diagnostics: List[Dict[str, Any]] = []
        recommended_actions: List[str] = []
        if parse_error is None:
            rows = summary_payload.get("results")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("status", "")).strip().lower() == "pass":
                        continue
                    case_id = str(row.get("case_id", "")).strip() or "unknown_case"
                    root_codes = coerce_failure_codes(
                        row.get("root_failure_codes"),
                        fallback=str(row.get("root_failure_code_primary", "")).strip() or None,
                    )
                    floor_violations = extract_floor_violations(row.get("clinical_floor_gap_summary"))
                    case_diag = {
                        "case_id": case_id,
                        "failure_code": row.get("failure_code"),
                        "root_failure_code_primary": row.get("root_failure_code_primary"),
                        "root_failure_codes": root_codes,
                        "minimum_floor_margin": (
                            _to_float_or_none((row.get("clinical_floor_gap_summary") or {}).get("minimum_margin"))
                            if isinstance(row.get("clinical_floor_gap_summary"), dict)
                            else None
                        ),
                        "floor_violations": floor_violations[:12],
                        "artifacts": row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {},
                    }
                    case_diagnostics.append(case_diag)

                    if any(code.startswith("clinical_floor_") for code in root_codes):
                        recommended_actions.append(
                            "Clinical floors not met: inspect floor_violations and re-run feasibility scan before changing benchmark role."
                        )
                        if case_id == "uci-diabetes-130-readmission":
                            recommended_actions.append(
                                "Run diabetes feasibility sweep: python3 scripts/mlgg.py scan-diabetes --target-modes gt30,any,lt30 --max-rows-options 20000,0"
                            )
                            recommended_actions.append(
                                "Review distribution_report and external_validation_report for transport stress before attempting floor-preserving model changes."
                            )

        command = suite.get("command")
        rerun_cmd = shlex.join(command) if isinstance(command, list) and command else None
        if rerun_cmd:
            recommended_actions.append(f"One-step rerun: {rerun_cmd}")
        if not recommended_actions:
            recommended_actions.append("Review summary_file and failed case root_failure_codes, then rerun the suite once for confirmation.")

        diagnostics.append(
            {
                "suite_id": str(suite.get("suite_id")),
                "suite_name": str(suite.get("name", "")),
                "kind": str(suite.get("kind", "")),
                "status": str(suite.get("status", "")),
                "summary_file": str(summary_file),
                "failure_codes": coerce_failure_codes(suite.get("failure_codes")),
                "repeat_statuses": suite.get("repeat_statuses", []),
                "case_diagnostics": case_diagnostics,
                "recommended_actions": dedup_keep_order(recommended_actions),
                "summary_parse_error": parse_error,
            }
        )
    return diagnostics


def main() -> int:
    args = parse_args()
    if int(args.suite_timeout_seconds) < 1:
        raise SystemExit("--suite-timeout-seconds must be >= 1.")
    if int(args.repeat) < 1:
        raise SystemExit("--repeat must be >= 1.")
    run_tag = str(args.run_tag).strip() or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = Path(str(args.output)).expanduser().resolve()
    registry_file = Path(str(args.registry_file)).expanduser().resolve()
    artifacts_dir = Path(str(args.artifacts_dir)).expanduser().resolve()
    python_bin = str(args.python).strip() or sys.executable
    registry, registry_fail_code, registry_fail_message = load_and_validate_registry(
        registry_file=registry_file,
        profile=str(args.profile),
    )

    registry_sha256: Optional[str] = None
    if registry is not None:
        registry_sha256 = canonical_json_sha256(registry)
        ok, message = validate_dataset_fingerprints(registry=registry, registry_file=registry_file)
        if not ok:
            registry_fail_code = FAIL_BENCHMARK_REGISTRY_MISMATCH
            registry_fail_message = message or "dataset fingerprint mismatch."

    if registry_fail_code is not None:
        diagnostics_path = (
            Path(str(args.observational_diagnostics_out)).expanduser().resolve()
            if str(args.observational_diagnostics_out).strip()
            else output_path.with_name(output_path.stem + ".observational_diagnostics.json")
        )
        report = {
            "contract_version": "release_benchmark_matrix.v2",
            "generated_at_utc": utc_now_text(),
            "repo_root": str(REPO_ROOT),
            "run_tag": run_tag,
            "profile": str(args.profile),
            "registry_file": str(registry_file),
            "dataset_registry_sha256": registry_sha256,
            "overall_status": "fail",
            "status_reason": registry_fail_code,
            "failure_codes": [registry_fail_code],
            "all_failure_codes": [registry_fail_code],
            "blocking_failure_codes": [],
            "observational_failure_codes": [],
            "repeat_count": int(args.repeat),
            "repeat_consistent": False,
            "blocking_suite_ids": [],
            "nonblocking_suite_ids": [],
            "blocking_failure_count": 0,
            "blocking_failures": [],
            "nonblocking_failure_count": 0,
            "nonblocking_failures": [],
            "observational_diagnostics_file": str(diagnostics_path),
            "observational_diagnostics": [],
            "suites": [],
            "suite_runs": [],
            "registry_error_message": registry_fail_message,
        }
        write_json_atomic(output_path, report)
        write_json_atomic(
            diagnostics_path,
            {
                "contract_version": "release_benchmark_observational_diagnostics.v1",
                "generated_at_utc": utc_now_text(),
                "overall_status": "not_available",
                "status_reason": registry_fail_code,
                "items": [],
            },
        )
        print(f"[FAIL] {registry_fail_code}: {registry_fail_message}")
        print(f"Release benchmark matrix summary: {output_path}")
        print(f"Observational diagnostics: {diagnostics_path}")
        if str(args.emit_junit).strip():
            write_junit(
                Path(str(args.emit_junit)).expanduser().resolve(),
                [],
                "fail",
                [registry_fail_code],
                registry_fail_code,
            )
        return 2

    assert registry is not None
    suites = build_suites_from_registry(
        registry=registry,
        profile=str(args.profile),
        run_tag=run_tag,
        artifacts_dir=artifacts_dir,
        python_bin=python_bin,
        stress_seed_min=int(args.stress_seed_min),
        stress_seed_max=int(args.stress_seed_max),
        heart_blocking=bool(args.heart_blocking),
    )

    suite_runs: List[Dict[str, Any]] = []
    suite_rows: List[Dict[str, Any]] = []  # final summary row per suite
    blocking_failures: List[str] = []
    nonblocking_failures: List[str] = []
    per_suite_rows: Dict[str, List[Dict[str, Any]]] = {suite.suite_id: [] for suite in suites}
    blocking_suite_ids = [suite.suite_id for suite in suites if suite.blocking]
    nonblocking_suite_ids = [suite.suite_id for suite in suites if not suite.blocking]

    for repeat_index in range(1, int(args.repeat) + 1):
        for suite in suites:
            command = list(suite.command)
            summary_file = suite.summary_file
            if int(args.repeat) > 1:
                summary_file = summary_file.with_name(f"{summary_file.stem}.r{repeat_index}{summary_file.suffix}")
                for flag in ("--summary-file", "--output"):
                    if flag in command:
                        pos = command.index(flag)
                        if pos + 1 < len(command):
                            command[pos + 1] = str(summary_file)
                        break
            if suite.kind == "authority":
                # make repeat runs explicit in downstream authority summary metadata.
                command.extend(["--run-tag", f"{run_tag}-{suite.suite_id}-r{repeat_index}"])
            print(f"$ {shlex.join(command)}")
            if args.dry_run:
                row = {
                    "repeat_index": repeat_index,
                    "suite_id": suite.suite_id,
                    "name": suite.name,
                    "kind": suite.kind,
                    "blocking": bool(suite.blocking),
                    "status": "dry_run",
                    "exit_code": None,
                    "overall_status": None,
                    "failure_reason": None,
                    "failure_codes": [],
                    "summary_file": str(summary_file),
                    "command": command,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
                suite_runs.append(row)
                per_suite_rows[suite.suite_id].append(row)
                continue

            summary_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    text=True,
                    capture_output=True,
                    timeout=float(args.suite_timeout_seconds),
                )
                outcome = parse_suite_outcome(
                    suite.kind,
                    summary_file,
                    int(proc.returncode),
                    suite.expected_case_ids,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_text = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr_text = exc.stderr if isinstance(exc.stderr, str) else ""
                outcome = {
                    "status": "fail",
                    "failure_reason": "suite_timeout",
                    "failure_codes": [FAIL_BENCHMARK_SUITE_TIMEOUT],
                    "exit_code": 124,
                    "overall_status": None,
                    "error": (
                        f"suite_timeout_seconds={int(args.suite_timeout_seconds)} "
                        f"elapsed={float(exc.timeout):.3f}"
                    ),
                }
                proc = subprocess.CompletedProcess(
                    args=command,
                    returncode=124,
                    stdout=stdout_text,
                    stderr=stderr_text,
                )
            row = {
                "repeat_index": repeat_index,
                "suite_id": suite.suite_id,
                "name": suite.name,
                "kind": suite.kind,
                "blocking": bool(suite.blocking),
                "status": outcome.get("status"),
                "exit_code": outcome.get("exit_code"),
                "overall_status": outcome.get("overall_status"),
                "failure_reason": outcome.get("failure_reason"),
                "failure_codes": outcome.get("failure_codes", []),
                "summary_file": str(summary_file),
                "command": command,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
            for key in (
                "case_count",
                "failed_cases",
                "actual_case_ids",
                "expected_case_ids",
                "passed_count",
                "scenario_count",
                "failed_scenarios",
                "case_metrics_fingerprint",
                "error",
            ):
                if key in outcome:
                    row[key] = outcome.get(key)
            suite_runs.append(row)
            per_suite_rows[suite.suite_id].append(row)

    inconsistent_suite_ids: List[str] = []
    for suite_id, rows in per_suite_rows.items():
        if not rows:
            continue
        sig0 = suite_signature(rows[0])
        if any(suite_signature(row) != sig0 for row in rows[1:]):
            inconsistent_suite_ids.append(suite_id)
        final_row = dict(rows[-1])
        final_row["repeat_statuses"] = [str(row.get("status")) for row in rows]
        final_row["repeat_failure_codes"] = [row.get("failure_codes", []) for row in rows]
        suite_rows.append(final_row)

    repeat_consistent = len(inconsistent_suite_ids) == 0
    if not args.dry_run:
        for row in suite_runs:
            if str(row.get("status")) == "pass":
                continue
            suite_id = str(row.get("suite_id", "")).strip()
            if bool(row.get("blocking")):
                blocking_failures.append(suite_id)
            else:
                nonblocking_failures.append(suite_id)
    blocking_failures = dedup_keep_order(blocking_failures)
    nonblocking_failures = dedup_keep_order(nonblocking_failures)

    all_failure_codes: List[str] = []
    blocking_failure_codes: List[str] = []
    observational_failure_codes: List[str] = []
    for row in suite_runs:
        if str(row.get("status")) == "pass":
            continue
        row_codes = coerce_failure_codes(row.get("failure_codes"))
        for code in row_codes:
            if isinstance(code, str) and code.strip():
                all_failure_codes.append(code.strip())
                if bool(row.get("blocking")):
                    blocking_failure_codes.append(code.strip())
                else:
                    observational_failure_codes.append(code.strip())

    failure_codes: List[str] = list(dedup_keep_order(blocking_failure_codes))

    if args.dry_run:
        overall_status = "dry_run"
        exit_code = 0
        status_reason = "commands_printed_only"
    elif bool(args.fail_on_repeat_inconsistency) and not repeat_consistent:
        overall_status = "fail"
        exit_code = 2
        status_reason = FAIL_BENCHMARK_REPEAT_INCONSISTENT
        all_failure_codes.append(FAIL_BENCHMARK_REPEAT_INCONSISTENT)
        failure_codes.append(FAIL_BENCHMARK_REPEAT_INCONSISTENT)
    elif blocking_failures:
        overall_status = "fail"
        exit_code = 2
        status_reason = FAIL_BENCHMARK_BLOCKING_SUITE_FAILED
        all_failure_codes.append(FAIL_BENCHMARK_BLOCKING_SUITE_FAILED)
        failure_codes.append(FAIL_BENCHMARK_BLOCKING_SUITE_FAILED)
    else:
        overall_status = "pass"
        exit_code = 0
        status_reason = "all_blocking_suites_passed"
    all_failure_codes = dedup_keep_order(all_failure_codes)
    failure_codes = dedup_keep_order(failure_codes)
    blocking_failure_codes = dedup_keep_order(blocking_failure_codes)
    observational_failure_codes = dedup_keep_order(observational_failure_codes)
    diagnostics_path = (
        Path(str(args.observational_diagnostics_out)).expanduser().resolve()
        if str(args.observational_diagnostics_out).strip()
        else output_path.with_name(output_path.stem + ".observational_diagnostics.json")
    )
    observational_diagnostics = build_observational_diagnostics(suite_rows if not args.dry_run else [])

    report: Dict[str, Any] = {
        "contract_version": "release_benchmark_matrix.v2",
        "generated_at_utc": utc_now_text(),
        "repo_root": str(REPO_ROOT),
        "run_tag": run_tag,
        "profile": str(args.profile),
        "registry_file": str(registry_file),
        "dataset_registry_sha256": registry_sha256,
        "overall_status": overall_status,
        "status_reason": status_reason,
        "failure_codes": failure_codes,
        "all_failure_codes": all_failure_codes,
        "blocking_failure_codes": blocking_failure_codes,
        "observational_failure_codes": observational_failure_codes,
        "repeat_count": int(args.repeat),
        "repeat_consistent": repeat_consistent,
        "inconsistent_suite_ids": sorted(inconsistent_suite_ids),
        "blocking_suite_ids": blocking_suite_ids,
        "nonblocking_suite_ids": nonblocking_suite_ids,
        "blocking_failure_count": len(blocking_failures),
        "blocking_failures": blocking_failures,
        "nonblocking_failure_count": len(nonblocking_failures),
        "nonblocking_failures": nonblocking_failures,
        "observational_diagnostics_file": str(diagnostics_path),
        "observational_diagnostics": observational_diagnostics,
        "suite_runs": suite_runs,
        "suites": suite_rows,
    }
    write_json_atomic(output_path, report)
    write_json_atomic(
        diagnostics_path,
        {
            "contract_version": "release_benchmark_observational_diagnostics.v1",
            "generated_at_utc": utc_now_text(),
            "repo_root": str(REPO_ROOT),
            "run_tag": run_tag,
            "profile": str(args.profile),
            "overall_status": overall_status,
            "status_reason": status_reason,
            "items": observational_diagnostics,
        },
    )
    if str(args.emit_junit).strip():
        junit_path = Path(str(args.emit_junit)).expanduser().resolve()
        write_junit(junit_path, suite_runs, overall_status, failure_codes, status_reason)
        print(f"JUnit summary: {junit_path}")
    print(f"Release benchmark matrix summary: {output_path}")
    print(f"Observational diagnostics: {diagnostics_path}")
    if overall_status == "fail" and blocking_failures:
        rerun_cmd = (
            "python3 scripts/mlgg.py benchmark-suite --profile "
            f"{args.profile} --repeat 1 --registry-file {registry_file}"
        )
        print("Blocking failures (priority):")
        for suite_id in blocking_failures:
            print(f"  - {suite_id}")
        print(f"One-step verify command: {rerun_cmd}")
    print(
        "overall_status="
        f"{overall_status} blocking_failures={len(blocking_failures)} "
        f"repeat_consistent={repeat_consistent}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
