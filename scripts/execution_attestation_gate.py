#!/usr/bin/env python3
"""
Fail-closed execution attestation gate for publication-grade medical prediction.

This gate verifies non-repudiation evidence by checking:
1. Signed attestation payload integrity (detached signature verification).
2. Hash integrity of execution artifacts (logs/config/model/evaluation).
3. Study/run identity consistency between request, spec, and signed payload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_SIGNING_METHODS = {"openssl-dgst-sha256"}
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
GIT_COMMIT_RE = re.compile(r"^[a-fA-F0-9]{7,40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify signed execution attestation and artifact integrity.")
    parser.add_argument("--attestation-spec", required=True, help="Path to execution attestation spec JSON.")
    parser.add_argument("--evaluation-report", required=True, help="Path to canonical evaluation report JSON.")
    parser.add_argument("--study-id", help="Expected study_id from request contract.")
    parser.add_argument("--run-id", help="Expected run_id from request contract or orchestrator.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def add_issue(bucket: List[Dict[str, Any]], code: str, message: str, details: Dict[str, Any]) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def parse_iso_ts(raw: str) -> Optional[dt.datetime]:
    value = raw.strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def require_str(obj: Dict[str, Any], key: str, failures: List[Dict[str, Any]], where: str) -> Optional[str]:
    value = obj.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    add_issue(
        failures,
        "invalid_field",
        "Required field must be a non-empty string.",
        {"where": where, "field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return None


def require_str_list(
    obj: Dict[str, Any],
    key: str,
    failures: List[Dict[str, Any]],
    where: str,
) -> List[str]:
    value = obj.get(key)
    if not isinstance(value, list):
        add_issue(
            failures,
            "invalid_field",
            "Required field must be a list of non-empty strings.",
            {"where": where, "field": key, "actual_type": type(value).__name__ if value is not None else None},
        )
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        else:
            add_issue(
                failures,
                "invalid_field",
                "List item must be a non-empty string.",
                {"where": where, "field": key},
            )
            return []
    return out


def load_json_obj(path: Path, failures: List[Dict[str, Any]], code_prefix: str) -> Optional[Dict[str, Any]]:
    if not path.exists():
        add_issue(
            failures,
            f"{code_prefix}_missing",
            "Required JSON file not found.",
            {"path": str(path)},
        )
        return None
    if not path.is_file():
        add_issue(
            failures,
            f"{code_prefix}_not_file",
            "Required JSON path is not a file.",
            {"path": str(path)},
        )
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        add_issue(
            failures,
            f"{code_prefix}_invalid_json",
            "Failed to parse JSON file.",
            {"path": str(path), "error": str(exc)},
        )
        return None
    if not isinstance(payload, dict):
        add_issue(
            failures,
            f"{code_prefix}_invalid_root",
            "JSON root must be an object.",
            {"path": str(path), "actual_type": type(payload).__name__},
        )
        return None
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


def verify_signature(
    payload_file: Path,
    signature_file: Path,
    public_key_file: Path,
    method: str,
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {
        "method": method,
        "payload_file": str(payload_file),
        "signature_file": str(signature_file),
        "public_key_file": str(public_key_file),
        "verified": False,
        "command": None,
        "stdout": "",
        "stderr": "",
    }

    if method not in ALLOWED_SIGNING_METHODS:
        add_issue(
            failures,
            "unsupported_signing_method",
            "Unsupported attestation signing method.",
            {"method": method, "allowed": sorted(ALLOWED_SIGNING_METHODS)},
        )
        return out

    cmd = [
        "openssl",
        "dgst",
        "-sha256",
        "-verify",
        str(public_key_file),
        "-signature",
        str(signature_file),
        str(payload_file),
    ]
    out["command"] = " ".join(cmd)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except FileNotFoundError:
        add_issue(
            failures,
            "openssl_not_available",
            "openssl command is required for signature verification but was not found.",
            {},
        )
        return out
    except Exception as exc:
        add_issue(
            failures,
            "signature_verification_error",
            "Signature verification process failed.",
            {"error": str(exc)},
        )
        return out

    out["stdout"] = (proc.stdout or "").strip()
    out["stderr"] = (proc.stderr or "").strip()
    if proc.returncode != 0:
        add_issue(
            failures,
            "signature_verification_failed",
            "Detached signature verification failed.",
            {"return_code": proc.returncode, "stdout": out["stdout"], "stderr": out["stderr"]},
        )
        return out

    out["verified"] = True
    return out


def parse_artifacts(
    artifacts: Any,
    payload_base: Path,
    required_names: List[str],
    eval_report: Path,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(artifacts, list) or not artifacts:
        add_issue(
            failures,
            "invalid_artifacts",
            "Signed payload must include non-empty artifacts list.",
            {"actual_type": type(artifacts).__name__ if artifacts is not None else None},
        )
        return {"artifact_count": 0, "required_names": required_names, "checked": []}

    seen_names: Set[str] = set()
    checked: List[Dict[str, Any]] = []
    found_required: Set[str] = set()

    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            add_issue(
                failures,
                "invalid_artifact_entry",
                "Artifact entry must be an object.",
                {"index": idx, "actual_type": type(item).__name__},
            )
            continue

        name = item.get("name")
        path_raw = item.get("path")
        digest = item.get("sha256")

        if not isinstance(name, str) or not name.strip():
            add_issue(failures, "invalid_artifact_name", "Artifact name must be non-empty string.", {"index": idx})
            continue
        name_clean = name.strip()
        if name_clean in seen_names:
            add_issue(
                failures,
                "duplicate_artifact_name",
                "Artifact names must be unique in signed payload.",
                {"name": name_clean},
            )
            continue
        seen_names.add(name_clean)

        if name_clean in required_names:
            found_required.add(name_clean)

        if not isinstance(path_raw, str) or not path_raw.strip():
            add_issue(
                failures,
                "invalid_artifact_path",
                "Artifact path must be non-empty string.",
                {"artifact": name_clean},
            )
            continue

        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest.strip()):
            add_issue(
                failures,
                "invalid_artifact_sha256",
                "Artifact sha256 must be 64-char hex string.",
                {"artifact": name_clean},
            )
            continue

        resolved_path = resolve_path(payload_base, path_raw.strip())
        if not resolved_path.exists():
            add_issue(
                failures,
                "artifact_file_missing",
                "Artifact file does not exist.",
                {"artifact": name_clean, "path": str(resolved_path)},
            )
            continue
        if not resolved_path.is_file():
            add_issue(
                failures,
                "artifact_path_not_file",
                "Artifact path must point to file.",
                {"artifact": name_clean, "path": str(resolved_path)},
            )
            continue

        actual_digest = sha256_file(resolved_path)
        if actual_digest.lower() != digest.strip().lower():
            add_issue(
                failures,
                "artifact_hash_mismatch",
                "Artifact sha256 does not match signed payload.",
                {
                    "artifact": name_clean,
                    "path": str(resolved_path),
                    "declared_sha256": digest.strip().lower(),
                    "actual_sha256": actual_digest.lower(),
                },
            )

        file_size = resolved_path.stat().st_size
        if name_clean.lower().endswith("log") and file_size < 64:
            add_issue(
                warnings,
                "short_log_artifact",
                "Log artifact is very small; execution trace may be insufficient.",
                {"artifact": name_clean, "path": str(resolved_path), "size_bytes": file_size},
            )

        checked.append(
            {
                "name": name_clean,
                "path": str(resolved_path),
                "sha256": actual_digest.lower(),
                "size_bytes": file_size,
            }
        )

    missing_required = [name for name in required_names if name not in found_required]
    if missing_required:
        add_issue(
            failures,
            "missing_required_artifacts",
            "Signed payload is missing required artifact names.",
            {"missing_required_artifacts": missing_required},
        )

    eval_artifact = next((x for x in checked if x["name"] == "evaluation_report"), None)
    if eval_artifact is None:
        add_issue(
            failures,
            "missing_evaluation_report_artifact",
            "Signed payload must include artifact named 'evaluation_report'.",
            {},
        )
    else:
        if Path(eval_artifact["path"]).resolve() != eval_report.resolve():
            add_issue(
                failures,
                "evaluation_report_path_mismatch",
                "evaluation_report artifact path must match requested evaluation report.",
                {"artifact_path": eval_artifact["path"], "expected_path": str(eval_report.resolve())},
            )

    return {
        "artifact_count": len(checked),
        "required_names": required_names,
        "checked": checked,
    }


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    spec_path = Path(args.attestation_spec).expanduser().resolve()
    eval_report = Path(args.evaluation_report).expanduser().resolve()

    if not eval_report.exists():
        add_issue(
            failures,
            "evaluation_report_missing",
            "Evaluation report file not found.",
            {"path": str(eval_report)},
        )
    elif not eval_report.is_file():
        add_issue(
            failures,
            "evaluation_report_not_file",
            "Evaluation report path must point to file.",
            {"path": str(eval_report)},
        )

    spec = load_json_obj(spec_path, failures, "attestation_spec")
    if spec is None:
        return finish(args, failures, warnings, {}, {})

    spec_study_id = require_str(spec, "study_id", failures, "attestation_spec")
    spec_run_id = require_str(spec, "run_id", failures, "attestation_spec")
    issued_at = require_str(spec, "issued_at_utc", failures, "attestation_spec")
    required_names = require_str_list(spec, "required_artifact_names", failures, "attestation_spec")

    issued_at_ts = None
    if issued_at is not None:
        issued_at_ts = parse_iso_ts(issued_at)
        if issued_at_ts is None:
            add_issue(
                failures,
                "invalid_issued_timestamp",
                "issued_at_utc must be ISO-8601 timestamp.",
                {"issued_at_utc": issued_at},
            )

    signing = spec.get("signing")
    if not isinstance(signing, dict):
        add_issue(
            failures,
            "invalid_signing_block",
            "attestation_spec.signing must be an object.",
            {"actual_type": type(signing).__name__ if signing is not None else None},
        )
        return finish(args, failures, warnings, {}, {})

    method = require_str(signing, "method", failures, "attestation_spec.signing")
    signature_file_raw = require_str(signing, "signature_file", failures, "attestation_spec.signing")
    public_key_file_raw = require_str(signing, "public_key_file", failures, "attestation_spec.signing")
    payload_file_raw = require_str(signing, "signed_payload_file", failures, "attestation_spec.signing")

    spec_base = spec_path.parent
    signature_file = resolve_path(spec_base, signature_file_raw) if signature_file_raw else None
    public_key_file = resolve_path(spec_base, public_key_file_raw) if public_key_file_raw else None
    payload_file = resolve_path(spec_base, payload_file_raw) if payload_file_raw else None

    for label, path in (
        ("signature_file", signature_file),
        ("public_key_file", public_key_file),
        ("signed_payload_file", payload_file),
    ):
        if path is None:
            continue
        if not path.exists():
            add_issue(
                failures,
                "signing_file_missing",
                "Signing material file is missing.",
                {"field": label, "path": str(path)},
            )
        elif not path.is_file():
            add_issue(
                failures,
                "signing_path_not_file",
                "Signing material path must point to a file.",
                {"field": label, "path": str(path)},
            )

    verification: Dict[str, Any] = {}
    if (
        method is not None
        and payload_file is not None
        and signature_file is not None
        and public_key_file is not None
        and not failures
    ):
        verification = verify_signature(
            payload_file=payload_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method=method,
            failures=failures,
        )

    payload = load_json_obj(payload_file, failures, "signed_payload") if payload_file is not None else None
    if payload is None:
        return finish(args, failures, warnings, {}, {"signature_verification": verification})

    payload_study_id = require_str(payload, "study_id", failures, "signed_payload")
    payload_run_id = require_str(payload, "run_id", failures, "signed_payload")
    payload_command = require_str(payload, "command", failures, "signed_payload")
    started_at = require_str(payload, "started_at_utc", failures, "signed_payload")
    finished_at = require_str(payload, "finished_at_utc", failures, "signed_payload")
    executor = require_str(payload, "executor", failures, "signed_payload")

    git_commit = payload.get("git_commit")
    if git_commit is not None:
        if not isinstance(git_commit, str) or not GIT_COMMIT_RE.fullmatch(git_commit.strip()):
            add_issue(
                warnings,
                "suspicious_git_commit_format",
                "git_commit is present but not formatted as 7-40 hex characters.",
                {"git_commit": git_commit},
            )

    if spec_study_id and payload_study_id and spec_study_id != payload_study_id:
        add_issue(
            failures,
            "study_id_mismatch",
            "study_id differs between attestation spec and signed payload.",
            {"spec_study_id": spec_study_id, "payload_study_id": payload_study_id},
        )
    if spec_run_id and payload_run_id and spec_run_id != payload_run_id:
        add_issue(
            failures,
            "run_id_mismatch",
            "run_id differs between attestation spec and signed payload.",
            {"spec_run_id": spec_run_id, "payload_run_id": payload_run_id},
        )
    if args.study_id and payload_study_id and args.study_id.strip() != payload_study_id:
        add_issue(
            failures,
            "request_study_id_mismatch",
            "Runtime expected study_id does not match signed payload.",
            {"expected_study_id": args.study_id.strip(), "payload_study_id": payload_study_id},
        )
    if args.run_id and payload_run_id and args.run_id.strip() != payload_run_id:
        add_issue(
            failures,
            "request_run_id_mismatch",
            "Runtime expected run_id does not match signed payload.",
            {"expected_run_id": args.run_id.strip(), "payload_run_id": payload_run_id},
        )

    started_ts = parse_iso_ts(started_at) if started_at else None
    finished_ts = parse_iso_ts(finished_at) if finished_at else None
    if started_at and started_ts is None:
        add_issue(
            failures,
            "invalid_started_timestamp",
            "started_at_utc must be ISO-8601 timestamp.",
            {"started_at_utc": started_at},
        )
    if finished_at and finished_ts is None:
        add_issue(
            failures,
            "invalid_finished_timestamp",
            "finished_at_utc must be ISO-8601 timestamp.",
            {"finished_at_utc": finished_at},
        )
    if started_ts and finished_ts and started_ts > finished_ts:
        add_issue(
            failures,
            "invalid_execution_time_window",
            "started_at_utc must be <= finished_at_utc.",
            {"started_at_utc": started_at, "finished_at_utc": finished_at},
        )
    if finished_ts and issued_at_ts and finished_ts > issued_at_ts:
        add_issue(
            failures,
            "invalid_attestation_issue_time",
            "issued_at_utc must be at or after finished_at_utc.",
            {"issued_at_utc": issued_at, "finished_at_utc": finished_at},
        )

    payload_digest_declared = spec.get("signed_payload_sha256")
    if payload_file is not None and isinstance(payload_digest_declared, str):
        payload_digest_actual = sha256_file(payload_file)
        if not SHA256_RE.fullmatch(payload_digest_declared.strip()):
            add_issue(
                failures,
                "invalid_signed_payload_sha256",
                "signed_payload_sha256 must be 64-char hex string.",
                {"signed_payload_sha256": payload_digest_declared},
            )
        elif payload_digest_actual.lower() != payload_digest_declared.strip().lower():
            add_issue(
                failures,
                "signed_payload_hash_mismatch",
                "signed_payload_sha256 does not match the payload file.",
                {
                    "declared_sha256": payload_digest_declared.strip().lower(),
                    "actual_sha256": payload_digest_actual.lower(),
                },
            )

    artifacts_summary = parse_artifacts(
        artifacts=payload.get("artifacts"),
        payload_base=payload_file.parent if payload_file is not None else spec_base,
        required_names=required_names,
        eval_report=eval_report,
        failures=failures,
        warnings=warnings,
    )

    summary = {
        "attestation_spec": str(spec_path),
        "evaluation_report": str(eval_report),
        "study_id": payload_study_id,
        "run_id": payload_run_id,
        "executor": executor,
        "command_present": payload_command is not None,
        "signature_verification": verification,
        "artifacts": artifacts_summary,
    }

    return finish(args, failures, warnings, summary, payload_metadata={
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "issued_at_utc": issued_at,
    })


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
    payload_metadata: Dict[str, Any],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
        "payload_metadata": payload_metadata,
    }

    if args.report:
        out = Path(args.report).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)

    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
