#!/usr/bin/env python3
"""
Fail-closed execution attestation gate for publication-grade medical prediction.

This gate verifies high-assurance non-repudiation evidence by checking:
1. Signed attestation payload integrity (detached signature verification).
2. Artifact hash integrity (logs/config/model/evaluation).
3. Study/run identity consistency between request, spec, payload, and records.
4. Key assurance policy (fingerprint, key length, age, expiry, revocation list).
5. Trusted timestamp record verification.
6. Transparency-log record verification.
7. Signed execution-log attestation verification against payload artifact hash.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_SIGNING_METHODS = {"openssl-dgst-sha256"}
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
GIT_COMMIT_RE = re.compile(r"^[a-fA-F0-9]{7,40}$")
PUBLIC_KEY_BITS_RE = re.compile(r"Public-Key:\s*\((\d+)\s*bit\)", re.IGNORECASE)


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


def require_str_list(obj: Dict[str, Any], key: str, failures: List[Dict[str, Any]], where: str) -> List[str]:
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


def require_bool(
    obj: Dict[str, Any], key: str, failures: List[Dict[str, Any]], where: str, default: Optional[bool] = None
) -> Optional[bool]:
    if key not in obj:
        return default
    value = obj.get(key)
    if isinstance(value, bool):
        return value
    add_issue(
        failures,
        "invalid_field",
        "Field must be boolean.",
        {"where": where, "field": key, "actual_type": type(value).__name__ if value is not None else None},
    )
    return default


def require_number(
    obj: Dict[str, Any], key: str, failures: List[Dict[str, Any]], where: str, default: Optional[float] = None
) -> Optional[float]:
    if key not in obj:
        return default
    value = obj.get(key)
    if isinstance(value, bool):
        value = None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        add_issue(
            failures,
            "invalid_field",
            "Field must be finite number.",
            {"where": where, "field": key, "actual_type": type(value).__name__ if value is not None else None},
        )
        return default
    if not math.isfinite(parsed):
        add_issue(
            failures,
            "invalid_field",
            "Field must be finite number.",
            {"where": where, "field": key, "value": value},
        )
        return default
    return parsed


def check_authority_not_revoked(
    role: str,
    authority_id: Optional[str],
    observed_fp: Optional[str],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    failures: List[Dict[str, Any]],
    extra_details: Optional[Dict[str, Any]] = None,
) -> None:
    if not revoked_key_ids and not revoked_key_fps:
        return

    authority_id_norm = authority_id.strip() if isinstance(authority_id, str) else ""
    observed_fp_norm = observed_fp.lower() if isinstance(observed_fp, str) else ""
    id_revoked = bool(authority_id_norm and authority_id_norm in revoked_key_ids)
    fp_revoked = bool(observed_fp_norm and observed_fp_norm in revoked_key_fps)
    if not id_revoked and not fp_revoked:
        return

    details: Dict[str, Any] = {
        "role": role,
        "authority_id": authority_id_norm or None,
        "public_key_fingerprint_sha256": observed_fp_norm or None,
        "revoked_by_key_id": bool(id_revoked),
        "revoked_by_fingerprint": bool(fp_revoked),
    }
    if isinstance(extra_details, dict):
        details.update(extra_details)
    add_issue(
        failures,
        f"{role}_key_revoked",
        f"{role} key is revoked by revocation list.",
        details,
    )


def enforce_publication_policy_requirements(
    key_assurance: Dict[str, Any],
    failures: List[Dict[str, Any]],
) -> None:
    policy = key_assurance.get("policy")
    if not isinstance(policy, dict):
        add_issue(
            failures,
            "publication_policy_missing",
            "Strict mode requires assurance_policy summary for publication-grade attestation.",
            {},
        )
        return

    required_true_flags = (
        "require_revocation_list",
        "require_timestamp_trust",
        "require_transparency_log",
        "require_transparency_log_signature",
        "require_execution_receipt",
        "require_execution_log_attestation",
        "require_independent_timestamp_authority",
        "require_independent_execution_authority",
        "require_independent_log_authority",
        "require_witness_quorum",
        "require_independent_witness_keys",
        "require_witness_independence_from_signing",
    )
    for field in required_true_flags:
        if policy.get(field) is not True:
            add_issue(
                failures,
                "publication_policy_disabled",
                "Strict publication policy requires assurance_policy flag to be true.",
                {"field": field, "value": policy.get(field)},
            )

    min_witness_count_raw = policy.get("min_witness_count")
    min_witness_count = None
    if isinstance(min_witness_count_raw, bool):
        min_witness_count_raw = None
    if isinstance(min_witness_count_raw, int):
        min_witness_count = int(min_witness_count_raw)
    elif isinstance(min_witness_count_raw, float) and math.isfinite(min_witness_count_raw) and float(min_witness_count_raw).is_integer():
        min_witness_count = int(min_witness_count_raw)
    if min_witness_count is None:
        add_issue(
            failures,
            "publication_min_witness_count_invalid",
            "Strict publication policy requires numeric assurance_policy.min_witness_count.",
            {"value": min_witness_count_raw},
        )
    elif min_witness_count < 2:
        add_issue(
            failures,
            "publication_min_witness_count_too_low",
            "Strict publication policy requires assurance_policy.min_witness_count >= 2.",
            {"min_witness_count": min_witness_count},
        )

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


def file_line_count(path: Path) -> Optional[int]:
    try:
        total = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in fh:
                total += 1
        return total
    except Exception:
        return None


def run_openssl(cmd: List[str], failures: List[Dict[str, Any]], code: str, message: str) -> Optional[subprocess.CompletedProcess]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except FileNotFoundError:
        add_issue(
            failures,
            "openssl_not_available",
            "openssl command is required but not available in PATH.",
            {},
        )
        return None
    except Exception as exc:
        add_issue(
            failures,
            code,
            message,
            {"error": str(exc)},
        )
        return None
    return proc


def verify_detached_signature(
    data_file: Path,
    signature_file: Path,
    public_key_file: Path,
    method: str,
    failures: List[Dict[str, Any]],
    scope: str,
) -> Dict[str, Any]:
    out = {
        "scope": scope,
        "method": method,
        "data_file": str(data_file),
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
            "Unsupported signing method.",
            {"scope": scope, "method": method, "allowed": sorted(ALLOWED_SIGNING_METHODS)},
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
        str(data_file),
    ]
    out["command"] = " ".join(cmd)
    proc = run_openssl(
        cmd,
        failures=failures,
        code="signature_verification_error",
        message="Detached signature verification process failed.",
    )
    if proc is None:
        return out

    out["stdout"] = (proc.stdout or "").strip()
    out["stderr"] = (proc.stderr or "").strip()
    if proc.returncode != 0:
        add_issue(
            failures,
            "signature_verification_failed",
            "Detached signature verification failed.",
            {"scope": scope, "return_code": proc.returncode, "stdout": out["stdout"], "stderr": out["stderr"]},
        )
        return out

    out["verified"] = True
    return out


def public_key_der_bytes(public_key_file: Path, failures: List[Dict[str, Any]]) -> Optional[bytes]:
    cmd = ["openssl", "pkey", "-pubin", "-in", str(public_key_file), "-outform", "DER"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=False)
    except FileNotFoundError:
        add_issue(
            failures,
            "openssl_not_available",
            "openssl command is required but not available in PATH.",
            {},
        )
        return None
    except Exception as exc:
        add_issue(
            failures,
            "public_key_parse_error",
            "Failed to parse public key for fingerprint.",
            {"error": str(exc)},
        )
        return None
    if proc.returncode != 0:
        add_issue(
            failures,
            "public_key_parse_error",
            "Failed to parse public key for fingerprint.",
            {"stderr": (proc.stderr or b"").decode("utf-8", errors="ignore").strip()},
        )
        return None
    return proc.stdout


def public_key_fingerprint_sha256(public_key_file: Path, failures: List[Dict[str, Any]]) -> Optional[str]:
    der = public_key_der_bytes(public_key_file, failures)
    if der is None:
        return None
    return hashlib.sha256(der).hexdigest()


def public_key_bits(public_key_file: Path, failures: List[Dict[str, Any]]) -> Optional[int]:
    cmd = ["openssl", "pkey", "-pubin", "-in", str(public_key_file), "-text", "-noout"]
    proc = run_openssl(
        cmd,
        failures=failures,
        code="public_key_parse_error",
        message="Failed to inspect public key metadata.",
    )
    if proc is None:
        return None
    if proc.returncode != 0:
        add_issue(
            failures,
            "public_key_parse_error",
            "Failed to inspect public key metadata.",
            {"stderr": (proc.stderr or "").strip()},
        )
        return None

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = PUBLIC_KEY_BITS_RE.search(text)
    if not match:
        add_issue(
            failures,
            "public_key_bits_not_found",
            "Unable to determine public key bit length from openssl output.",
            {},
        )
        return None
    try:
        return int(match.group(1))
    except ValueError:
        add_issue(
            failures,
            "public_key_bits_not_found",
            "Unable to parse public key bit length.",
            {"matched": match.group(1)},
        )
        return None


def collect_required_path(
    spec_base: Path, parent: Dict[str, Any], key: str, failures: List[Dict[str, Any]], where: str
) -> Optional[Path]:
    value = require_str(parent, key, failures, where)
    if value is None:
        return None
    path = resolve_path(spec_base, value)
    if not path.exists():
        add_issue(
            failures,
            "required_file_missing",
            "Required file is missing.",
            {"where": where, "field": key, "path": str(path)},
        )
        return None
    if not path.is_file():
        add_issue(
            failures,
            "required_path_not_file",
            "Required path must point to file.",
            {"where": where, "field": key, "path": str(path)},
        )
        return None
    return path


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


def validate_key_assurance(
    spec_base: Path,
    spec: Dict[str, Any],
    signing: Dict[str, Any],
    issued_at_ts: Optional[dt.datetime],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        add_issue(
            failures,
            "invalid_assurance_policy",
            "assurance_policy must be an object.",
            {"actual_type": type(policy).__name__},
        )
        policy = {}

    min_key_bits = require_number(policy, "min_signing_key_bits", failures, "assurance_policy", default=3072.0)
    max_key_age_days = require_number(policy, "max_signing_key_age_days", failures, "assurance_policy", default=180.0)
    require_revocation_list = require_bool(
        policy, "require_revocation_list", failures, "assurance_policy", default=True
    )
    require_timestamp_trust = require_bool(
        policy, "require_timestamp_trust", failures, "assurance_policy", default=True
    )
    require_transparency_log = require_bool(
        policy, "require_transparency_log", failures, "assurance_policy", default=True
    )
    require_transparency_log_signature = require_bool(
        policy, "require_transparency_log_signature", failures, "assurance_policy", default=True
    )
    require_execution_receipt = require_bool(
        policy, "require_execution_receipt", failures, "assurance_policy", default=True
    )
    require_execution_log_attestation = require_bool(
        policy, "require_execution_log_attestation", failures, "assurance_policy", default=True
    )
    require_independent_timestamp_authority = require_bool(
        policy, "require_independent_timestamp_authority", failures, "assurance_policy", default=False
    )
    require_independent_execution_authority = require_bool(
        policy,
        "require_independent_execution_authority",
        failures,
        "assurance_policy",
        default=False,
    )
    require_independent_log_authority = require_bool(
        policy,
        "require_independent_log_authority",
        failures,
        "assurance_policy",
        default=False,
    )
    require_witness_quorum = require_bool(
        policy, "require_witness_quorum", failures, "assurance_policy", default=False
    )
    min_witness_count = require_number(
        policy,
        "min_witness_count",
        failures,
        "assurance_policy",
        default=2.0 if require_witness_quorum else 0.0,
    )
    require_independent_witness_keys = require_bool(
        policy,
        "require_independent_witness_keys",
        failures,
        "assurance_policy",
        default=True if require_witness_quorum else False,
    )
    require_witness_independence_from_signing = require_bool(
        policy,
        "require_witness_independence_from_signing",
        failures,
        "assurance_policy",
        default=True if require_witness_quorum else False,
    )

    key_id = require_str(signing, "key_id", failures, "attestation_spec.signing")
    key_created_at_raw = require_str(signing, "key_created_at_utc", failures, "attestation_spec.signing")
    key_not_after_raw = require_str(signing, "key_not_after_utc", failures, "attestation_spec.signing")
    expected_fp = require_str(signing, "public_key_fingerprint_sha256", failures, "attestation_spec.signing")
    public_key_file = collect_required_path(
        spec_base, signing, "public_key_file", failures, "attestation_spec.signing"
    )

    key_created_at = parse_iso_ts(key_created_at_raw) if key_created_at_raw else None
    key_not_after = parse_iso_ts(key_not_after_raw) if key_not_after_raw else None

    if key_created_at_raw and key_created_at is None:
        add_issue(
            failures,
            "invalid_key_created_timestamp",
            "key_created_at_utc must be ISO-8601 timestamp.",
            {"key_created_at_utc": key_created_at_raw},
        )
    if key_not_after_raw and key_not_after is None:
        add_issue(
            failures,
            "invalid_key_expiry_timestamp",
            "key_not_after_utc must be ISO-8601 timestamp.",
            {"key_not_after_utc": key_not_after_raw},
        )
    if key_created_at and key_not_after and key_created_at > key_not_after:
        add_issue(
            failures,
            "invalid_key_validity_window",
            "key_created_at_utc must be <= key_not_after_utc.",
            {"key_created_at_utc": key_created_at_raw, "key_not_after_utc": key_not_after_raw},
        )

    observed_fp: Optional[str] = None
    observed_bits: Optional[int] = None
    if public_key_file is not None:
        observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
        observed_bits = public_key_bits(public_key_file, failures)

    if expected_fp and not SHA256_RE.fullmatch(expected_fp):
        add_issue(
            failures,
            "invalid_public_key_fingerprint",
            "public_key_fingerprint_sha256 must be 64-char hex string.",
            {"public_key_fingerprint_sha256": expected_fp},
        )
    elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
        add_issue(
            failures,
            "public_key_fingerprint_mismatch",
            "Public key fingerprint does not match attestation spec.",
            {"expected": expected_fp.lower(), "observed": observed_fp.lower()},
        )

    if observed_bits is not None and min_key_bits is not None and observed_bits < int(min_key_bits):
        add_issue(
            failures,
            "signing_key_too_weak",
            "Signing key bit length is below required threshold.",
            {"observed_bits": observed_bits, "required_min_bits": int(min_key_bits)},
        )

    if issued_at_ts and key_not_after and issued_at_ts > key_not_after:
        add_issue(
            failures,
            "signing_key_expired",
            "Attestation issued after key expiry.",
            {"issued_at_utc": issued_at_ts.isoformat(), "key_not_after_utc": key_not_after.isoformat()},
        )
    if issued_at_ts and key_created_at and issued_at_ts < key_created_at:
        add_issue(
            failures,
            "attestation_before_key_creation",
            "Attestation issued before signing key creation time.",
            {"issued_at_utc": issued_at_ts.isoformat(), "key_created_at_utc": key_created_at.isoformat()},
        )
    if (
        issued_at_ts
        and key_created_at
        and max_key_age_days is not None
        and (issued_at_ts - key_created_at).total_seconds() > float(max_key_age_days) * 86400.0
    ):
        add_issue(
            failures,
            "signing_key_rotation_overdue",
            "Signing key age exceeds configured max_signing_key_age_days.",
            {
                "max_signing_key_age_days": max_key_age_days,
                "key_created_at_utc": key_created_at.isoformat(),
                "issued_at_utc": issued_at_ts.isoformat(),
            },
        )

    revocation_list_path = None
    revoked_ids_clean: Set[str] = set()
    revoked_fps_clean: Set[str] = set()
    revocation_list_raw = signing.get("revocation_list_file")
    if isinstance(revocation_list_raw, str) and revocation_list_raw.strip():
        revocation_list_path = resolve_path(spec_base, revocation_list_raw.strip())
        revocation = load_json_obj(revocation_list_path, failures, "revocation_list")
        if revocation is not None:
            revoked_ids = revocation.get("revoked_key_ids", [])
            revoked_fps = revocation.get("revoked_public_key_fingerprints_sha256", [])
            if not isinstance(revoked_ids, list):
                add_issue(
                    failures,
                    "invalid_revocation_list",
                    "revoked_key_ids must be list.",
                    {"path": str(revocation_list_path)},
                )
                revoked_ids = []
            if not isinstance(revoked_fps, list):
                add_issue(
                    failures,
                    "invalid_revocation_list",
                    "revoked_public_key_fingerprints_sha256 must be list.",
                    {"path": str(revocation_list_path)},
                )
                revoked_fps = []

            revoked_ids_clean = {str(x).strip() for x in revoked_ids if isinstance(x, str) and str(x).strip()}
            revoked_fps_clean = {str(x).strip().lower() for x in revoked_fps if isinstance(x, str)}

            if key_id and key_id in revoked_ids_clean:
                add_issue(
                    failures,
                    "signing_key_revoked",
                    "Signing key_id is revoked.",
                    {"key_id": key_id, "revocation_list": str(revocation_list_path)},
                )
            if observed_fp and observed_fp.lower() in revoked_fps_clean:
                add_issue(
                    failures,
                    "signing_key_revoked",
                    "Signing public key fingerprint is revoked.",
                    {"public_key_fingerprint_sha256": observed_fp.lower(), "revocation_list": str(revocation_list_path)},
                )
    elif require_revocation_list:
        add_issue(
            failures,
            "missing_revocation_list",
            "assurance_policy requires signing revocation list file.",
            {},
        )

    return {
        "policy": {
            "min_signing_key_bits": min_key_bits,
            "max_signing_key_age_days": max_key_age_days,
            "require_revocation_list": bool(require_revocation_list),
            "require_timestamp_trust": bool(require_timestamp_trust),
            "require_transparency_log": bool(require_transparency_log),
            "require_transparency_log_signature": bool(require_transparency_log_signature),
            "require_execution_receipt": bool(require_execution_receipt),
            "require_execution_log_attestation": bool(require_execution_log_attestation),
            "require_independent_timestamp_authority": bool(require_independent_timestamp_authority),
            "require_independent_execution_authority": bool(require_independent_execution_authority),
            "require_independent_log_authority": bool(require_independent_log_authority),
            "require_witness_quorum": bool(require_witness_quorum),
            "min_witness_count": min_witness_count,
            "require_independent_witness_keys": bool(require_independent_witness_keys),
            "require_witness_independence_from_signing": bool(require_witness_independence_from_signing),
        },
        "key_id": key_id,
        "public_key_file": str(public_key_file) if public_key_file is not None else None,
        "public_key_fingerprint_observed_sha256": observed_fp,
        "public_key_bits_observed": observed_bits,
        "revocation_list_file": str(revocation_list_path) if revocation_list_path is not None else None,
        "revoked_key_ids": sorted(revoked_ids_clean),
        "revoked_public_key_fingerprints_sha256": sorted(revoked_fps_clean),
    }


def validate_timestamp_trust(
    spec_base: Path,
    spec: Dict[str, Any],
    payload_sha256: Optional[str],
    payload_study_id: Optional[str],
    payload_run_id: Optional[str],
    finished_ts: Optional[dt.datetime],
    issued_at_ts: Optional[dt.datetime],
    signing_fp: Optional[str],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    require_timestamp_trust = True
    require_independent_timestamp_authority = False
    if isinstance(policy, dict):
        require_timestamp_trust = bool(policy.get("require_timestamp_trust", True))
        require_independent_timestamp_authority = bool(policy.get("require_independent_timestamp_authority", False))

    block = spec.get("timestamp_trust")
    if block is None:
        if require_timestamp_trust:
            add_issue(
                failures,
                "missing_timestamp_trust",
                "assurance_policy requires timestamp_trust block.",
                {},
            )
        return {"present": False}
    if not isinstance(block, dict):
        add_issue(
            failures,
            "invalid_timestamp_trust",
            "timestamp_trust must be an object.",
            {"actual_type": type(block).__name__},
        )
        return {"present": True}

    mode = require_str(block, "mode", failures, "attestation_spec.timestamp_trust")
    if mode and mode != "signed_record":
        add_issue(
            failures,
            "unsupported_timestamp_mode",
            "Only 'signed_record' timestamp mode is currently supported.",
            {"mode": mode},
        )

    authority_id = require_str(block, "authority_id", failures, "attestation_spec.timestamp_trust")
    record_file = collect_required_path(
        spec_base, block, "record_file", failures, "attestation_spec.timestamp_trust"
    )
    signature_file = collect_required_path(
        spec_base, block, "signature_file", failures, "attestation_spec.timestamp_trust"
    )
    public_key_file = collect_required_path(
        spec_base, block, "public_key_file", failures, "attestation_spec.timestamp_trust"
    )
    expected_fp = require_str(block, "public_key_fingerprint_sha256", failures, "attestation_spec.timestamp_trust")

    verify_result = {}
    observed_fp = None
    if public_key_file is not None:
        observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
        if expected_fp and not SHA256_RE.fullmatch(expected_fp):
            add_issue(
                failures,
                "invalid_timestamp_public_key_fingerprint",
                "timestamp_trust.public_key_fingerprint_sha256 must be 64-char hex.",
                {"value": expected_fp},
            )
        elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
            add_issue(
                failures,
                "timestamp_public_key_fingerprint_mismatch",
                "Timestamp trust public key fingerprint mismatch.",
                {"expected": expected_fp.lower(), "observed": observed_fp.lower()},
            )

    if record_file and signature_file and public_key_file:
        verify_result = verify_detached_signature(
            data_file=record_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method="openssl-dgst-sha256",
            failures=failures,
            scope="timestamp_trust_record",
        )

    record = load_json_obj(record_file, failures, "timestamp_record") if record_file else None
    if record is not None:
        record_authority = require_str(record, "authority_id", failures, "timestamp_record")
        record_payload_sha = require_str(record, "payload_sha256", failures, "timestamp_record")
        record_ts_raw = require_str(record, "timestamp_utc", failures, "timestamp_record")
        record_study = require_str(record, "study_id", failures, "timestamp_record")
        record_run = require_str(record, "run_id", failures, "timestamp_record")

        if authority_id and record_authority and authority_id != record_authority:
            add_issue(
                failures,
                "timestamp_authority_mismatch",
                "timestamp record authority_id does not match attestation spec.",
                {"spec_authority_id": authority_id, "record_authority_id": record_authority},
            )
        if payload_sha256 and record_payload_sha and payload_sha256.lower() != record_payload_sha.lower():
            add_issue(
                failures,
                "timestamp_payload_hash_mismatch",
                "timestamp record payload_sha256 does not match signed payload.",
                {"payload_sha256": payload_sha256.lower(), "record_payload_sha256": record_payload_sha.lower()},
            )
        if payload_study_id and record_study and payload_study_id != record_study:
            add_issue(
                failures,
                "timestamp_study_id_mismatch",
                "timestamp record study_id mismatch.",
                {"payload_study_id": payload_study_id, "record_study_id": record_study},
            )
        if payload_run_id and record_run and payload_run_id != record_run:
            add_issue(
                failures,
                "timestamp_run_id_mismatch",
                "timestamp record run_id mismatch.",
                {"payload_run_id": payload_run_id, "record_run_id": record_run},
            )

        record_ts = parse_iso_ts(record_ts_raw) if record_ts_raw else None
        if record_ts_raw and record_ts is None:
            add_issue(
                failures,
                "invalid_timestamp_record_time",
                "timestamp_record.timestamp_utc must be ISO-8601 timestamp.",
                {"timestamp_utc": record_ts_raw},
            )
        if record_ts and finished_ts and record_ts < finished_ts:
            add_issue(
                failures,
                "timestamp_before_run_finished",
                "timestamp trust record precedes run finish time.",
                {"timestamp_utc": record_ts.isoformat(), "finished_at_utc": finished_ts.isoformat()},
            )
        if record_ts and issued_at_ts and record_ts > issued_at_ts:
            add_issue(
                failures,
                "timestamp_after_attestation_issue",
                "timestamp trust record occurs after issued_at_utc.",
                {"timestamp_utc": record_ts.isoformat(), "issued_at_utc": issued_at_ts.isoformat()},
            )

    if (
        require_independent_timestamp_authority
        and signing_fp
        and observed_fp
        and signing_fp.lower() == observed_fp.lower()
    ):
        add_issue(
            failures,
            "timestamp_authority_not_independent",
            "assurance_policy requires independent timestamp authority key.",
            {"signing_key_fingerprint": signing_fp.lower(), "timestamp_key_fingerprint": observed_fp.lower()},
        )
    check_authority_not_revoked(
        role="timestamp_authority",
        authority_id=authority_id,
        observed_fp=observed_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
    )
    return {
        "present": True,
        "mode": mode,
        "authority_id": authority_id,
        "record_file": str(record_file) if record_file else None,
        "signature_file": str(signature_file) if signature_file else None,
        "public_key_file": str(public_key_file) if public_key_file else None,
        "public_key_fingerprint_observed_sha256": observed_fp,
        "signature_verification": verify_result,
    }


def validate_transparency_log(
    spec_base: Path,
    spec: Dict[str, Any],
    payload_sha256: Optional[str],
    payload_study_id: Optional[str],
    payload_run_id: Optional[str],
    finished_ts: Optional[dt.datetime],
    issued_at_ts: Optional[dt.datetime],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    require_transparency_log = True
    require_transparency_log_signature = True
    if isinstance(policy, dict):
        require_transparency_log = bool(policy.get("require_transparency_log", True))
        require_transparency_log_signature = bool(policy.get("require_transparency_log_signature", True))

    block = spec.get("transparency_log")
    if block is None:
        if require_transparency_log:
            add_issue(
                failures,
                "missing_transparency_log",
                "assurance_policy requires transparency_log block.",
                {},
            )
        return {"present": False}
    if not isinstance(block, dict):
        add_issue(
            failures,
            "invalid_transparency_log",
            "transparency_log must be an object.",
            {"actual_type": type(block).__name__},
        )
        return {"present": True}

    log_id = require_str(block, "log_id", failures, "attestation_spec.transparency_log")
    record_file = collect_required_path(
        spec_base, block, "record_file", failures, "attestation_spec.transparency_log"
    )
    signature_file = None
    public_key_file = None
    expected_fp = None
    if require_transparency_log_signature:
        signature_file = collect_required_path(
            spec_base, block, "signature_file", failures, "attestation_spec.transparency_log"
        )
        public_key_file = collect_required_path(
            spec_base, block, "public_key_file", failures, "attestation_spec.transparency_log"
        )
        expected_fp = require_str(
            block, "public_key_fingerprint_sha256", failures, "attestation_spec.transparency_log"
        )

    observed_fp = None
    verify_result = {}
    if public_key_file is not None:
        observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
        if expected_fp and not SHA256_RE.fullmatch(expected_fp):
            add_issue(
                failures,
                "invalid_transparency_public_key_fingerprint",
                "transparency_log.public_key_fingerprint_sha256 must be 64-char hex.",
                {"value": expected_fp},
            )
        elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
            add_issue(
                failures,
                "transparency_public_key_fingerprint_mismatch",
                "Transparency log public key fingerprint mismatch.",
                {"expected": expected_fp.lower(), "observed": observed_fp.lower()},
            )

    if record_file and signature_file and public_key_file and require_transparency_log_signature:
        verify_result = verify_detached_signature(
            data_file=record_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method="openssl-dgst-sha256",
            failures=failures,
            scope="transparency_log_record",
        )

    record = load_json_obj(record_file, failures, "transparency_record") if record_file else None
    if record is not None:
        record_log_id = require_str(record, "log_id", failures, "transparency_record")
        entry_id = require_str(record, "entry_id", failures, "transparency_record")
        record_payload_sha = require_str(record, "payload_sha256", failures, "transparency_record")
        recorded_at_raw = require_str(record, "recorded_at_utc", failures, "transparency_record")
        record_study = require_str(record, "study_id", failures, "transparency_record")
        record_run = require_str(record, "run_id", failures, "transparency_record")

        if log_id and record_log_id and log_id != record_log_id:
            add_issue(
                failures,
                "transparency_log_id_mismatch",
                "Transparency record log_id does not match attestation spec.",
                {"spec_log_id": log_id, "record_log_id": record_log_id},
            )
        if payload_sha256 and record_payload_sha and payload_sha256.lower() != record_payload_sha.lower():
            add_issue(
                failures,
                "transparency_payload_hash_mismatch",
                "Transparency record payload_sha256 does not match signed payload.",
                {"payload_sha256": payload_sha256.lower(), "record_payload_sha256": record_payload_sha.lower()},
            )
        if payload_study_id and record_study and payload_study_id != record_study:
            add_issue(
                failures,
                "transparency_study_id_mismatch",
                "Transparency record study_id mismatch.",
                {"payload_study_id": payload_study_id, "record_study_id": record_study},
            )
        if payload_run_id and record_run and payload_run_id != record_run:
            add_issue(
                failures,
                "transparency_run_id_mismatch",
                "Transparency record run_id mismatch.",
                {"payload_run_id": payload_run_id, "record_run_id": record_run},
            )
        if entry_id is not None and len(entry_id.strip()) < 3:
            add_issue(
                failures,
                "invalid_transparency_entry_id",
                "Transparency entry_id must be at least 3 characters.",
                {"entry_id": entry_id},
            )

        recorded_at = parse_iso_ts(recorded_at_raw) if recorded_at_raw else None
        if recorded_at_raw and recorded_at is None:
            add_issue(
                failures,
                "invalid_transparency_record_time",
                "transparency_record.recorded_at_utc must be ISO-8601 timestamp.",
                {"recorded_at_utc": recorded_at_raw},
            )
        if recorded_at and finished_ts and recorded_at < finished_ts:
            add_issue(
                failures,
                "transparency_before_run_finished",
                "Transparency log record precedes run finish time.",
                {"recorded_at_utc": recorded_at.isoformat(), "finished_at_utc": finished_ts.isoformat()},
            )
        if recorded_at and issued_at_ts and recorded_at > issued_at_ts:
            add_issue(
                failures,
                "transparency_after_attestation_issue",
                "Transparency log record occurs after issued_at_utc.",
                {"recorded_at_utc": recorded_at.isoformat(), "issued_at_utc": issued_at_ts.isoformat()},
            )

    check_authority_not_revoked(
        role="transparency_log_authority",
        authority_id=log_id,
        observed_fp=observed_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
    )

    return {
        "present": True,
        "log_id": log_id,
        "record_file": str(record_file) if record_file else None,
        "signature_file": str(signature_file) if signature_file else None,
        "public_key_file": str(public_key_file) if public_key_file else None,
        "public_key_fingerprint_observed_sha256": observed_fp,
        "signature_verification": verify_result,
    }


def validate_execution_receipt(
    spec_base: Path,
    spec: Dict[str, Any],
    payload_sha256: Optional[str],
    payload_study_id: Optional[str],
    payload_run_id: Optional[str],
    payload_command: Optional[str],
    payload_executor: Optional[str],
    payload_exit_code: Optional[int],
    started_ts: Optional[dt.datetime],
    finished_ts: Optional[dt.datetime],
    issued_at_ts: Optional[dt.datetime],
    signing_fp: Optional[str],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    require_execution_receipt = True
    require_independent_execution_authority = False
    if isinstance(policy, dict):
        require_execution_receipt = bool(policy.get("require_execution_receipt", True))
        require_independent_execution_authority = bool(
            policy.get("require_independent_execution_authority", False)
        )

    block = spec.get("execution_receipt")
    if block is None:
        if require_execution_receipt:
            add_issue(
                failures,
                "missing_execution_receipt",
                "assurance_policy requires execution_receipt block.",
                {},
            )
        return {"present": False}
    if not isinstance(block, dict):
        add_issue(
            failures,
            "invalid_execution_receipt",
            "execution_receipt must be an object.",
            {"actual_type": type(block).__name__},
        )
        return {"present": True}

    authority_id = require_str(block, "authority_id", failures, "attestation_spec.execution_receipt")
    record_file = collect_required_path(
        spec_base, block, "record_file", failures, "attestation_spec.execution_receipt"
    )
    signature_file = collect_required_path(
        spec_base, block, "signature_file", failures, "attestation_spec.execution_receipt"
    )
    public_key_file = collect_required_path(
        spec_base, block, "public_key_file", failures, "attestation_spec.execution_receipt"
    )
    expected_fp = require_str(
        block, "public_key_fingerprint_sha256", failures, "attestation_spec.execution_receipt"
    )
    enforce_exit_code_zero = require_bool(
        block,
        "enforce_exit_code_zero",
        failures,
        "attestation_spec.execution_receipt",
        default=True,
    )

    observed_fp = None
    verify_result = {}
    if public_key_file is not None:
        observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
        if expected_fp and not SHA256_RE.fullmatch(expected_fp):
            add_issue(
                failures,
                "invalid_execution_receipt_public_key_fingerprint",
                "execution_receipt.public_key_fingerprint_sha256 must be 64-char hex.",
                {"value": expected_fp},
            )
        elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
            add_issue(
                failures,
                "execution_receipt_public_key_fingerprint_mismatch",
                "Execution receipt public key fingerprint mismatch.",
                {"expected": expected_fp.lower(), "observed": observed_fp.lower()},
            )

    if record_file and signature_file and public_key_file:
        verify_result = verify_detached_signature(
            data_file=record_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method="openssl-dgst-sha256",
            failures=failures,
            scope="execution_receipt_record",
        )

    record = load_json_obj(record_file, failures, "execution_receipt_record") if record_file else None
    exit_code_value: Optional[int] = None
    if record is not None:
        record_authority = require_str(record, "authority_id", failures, "execution_receipt_record")
        record_payload_sha = require_str(record, "payload_sha256", failures, "execution_receipt_record")
        record_study = require_str(record, "study_id", failures, "execution_receipt_record")
        record_run = require_str(record, "run_id", failures, "execution_receipt_record")
        record_command = require_str(record, "command", failures, "execution_receipt_record")
        record_executor = require_str(record, "executor", failures, "execution_receipt_record")
        record_started_raw = require_str(record, "started_at_utc", failures, "execution_receipt_record")
        record_finished_raw = require_str(record, "finished_at_utc", failures, "execution_receipt_record")
        record_issued_raw = require_str(record, "issued_at_utc", failures, "execution_receipt_record")

        raw_exit_code = record.get("exit_code")
        if isinstance(raw_exit_code, bool):
            raw_exit_code = None
        if isinstance(raw_exit_code, int):
            exit_code_value = int(raw_exit_code)
        elif isinstance(raw_exit_code, float) and math.isfinite(raw_exit_code) and float(raw_exit_code).is_integer():
            exit_code_value = int(raw_exit_code)
        else:
            add_issue(
                failures,
                "invalid_execution_receipt_exit_code",
                "execution_receipt_record.exit_code must be integer.",
                {"actual_value": raw_exit_code},
            )

        if authority_id and record_authority and authority_id != record_authority:
            add_issue(
                failures,
                "execution_receipt_authority_mismatch",
                "execution receipt authority_id does not match attestation spec.",
                {"spec_authority_id": authority_id, "record_authority_id": record_authority},
            )
        if payload_sha256 and record_payload_sha and payload_sha256.lower() != record_payload_sha.lower():
            add_issue(
                failures,
                "execution_receipt_payload_hash_mismatch",
                "execution receipt payload_sha256 does not match signed payload.",
                {"payload_sha256": payload_sha256.lower(), "record_payload_sha256": record_payload_sha.lower()},
            )
        if payload_study_id and record_study and payload_study_id != record_study:
            add_issue(
                failures,
                "execution_receipt_study_id_mismatch",
                "execution receipt study_id mismatch.",
                {"payload_study_id": payload_study_id, "record_study_id": record_study},
            )
        if payload_run_id and record_run and payload_run_id != record_run:
            add_issue(
                failures,
                "execution_receipt_run_id_mismatch",
                "execution receipt run_id mismatch.",
                {"payload_run_id": payload_run_id, "record_run_id": record_run},
            )
        if payload_command and record_command and payload_command.strip() != record_command.strip():
            add_issue(
                failures,
                "execution_receipt_command_mismatch",
                "execution receipt command does not match signed payload command.",
                {"payload_command": payload_command, "record_command": record_command},
            )
        if payload_executor and record_executor and payload_executor.strip() != record_executor.strip():
            add_issue(
                failures,
                "execution_receipt_executor_mismatch",
                "execution receipt executor does not match signed payload executor.",
                {"payload_executor": payload_executor, "record_executor": record_executor},
            )
        if payload_exit_code is not None and exit_code_value is not None and payload_exit_code != exit_code_value:
            add_issue(
                failures,
                "execution_receipt_exit_code_mismatch",
                "execution receipt exit_code does not match signed payload exit_code.",
                {"payload_exit_code": payload_exit_code, "record_exit_code": exit_code_value},
            )

        record_started = parse_iso_ts(record_started_raw) if record_started_raw else None
        record_finished = parse_iso_ts(record_finished_raw) if record_finished_raw else None
        record_issued = parse_iso_ts(record_issued_raw) if record_issued_raw else None
        if record_started_raw and record_started is None:
            add_issue(
                failures,
                "invalid_execution_receipt_started_time",
                "execution_receipt_record.started_at_utc must be ISO-8601 timestamp.",
                {"started_at_utc": record_started_raw},
            )
        if record_finished_raw and record_finished is None:
            add_issue(
                failures,
                "invalid_execution_receipt_finished_time",
                "execution_receipt_record.finished_at_utc must be ISO-8601 timestamp.",
                {"finished_at_utc": record_finished_raw},
            )
        if record_issued_raw and record_issued is None:
            add_issue(
                failures,
                "invalid_execution_receipt_issued_time",
                "execution_receipt_record.issued_at_utc must be ISO-8601 timestamp.",
                {"issued_at_utc": record_issued_raw},
            )
        if record_started and record_finished and record_started > record_finished:
            add_issue(
                failures,
                "invalid_execution_receipt_time_window",
                "execution receipt started_at_utc must be <= finished_at_utc.",
                {"started_at_utc": record_started_raw, "finished_at_utc": record_finished_raw},
            )
        if record_finished and record_issued and record_finished > record_issued:
            add_issue(
                failures,
                "invalid_execution_receipt_issue_time",
                "execution receipt issued_at_utc must be >= finished_at_utc.",
                {"finished_at_utc": record_finished_raw, "issued_at_utc": record_issued_raw},
            )
        if issued_at_ts and record_issued and record_issued > issued_at_ts:
            add_issue(
                failures,
                "execution_receipt_after_attestation_issue",
                "execution receipt issued_at_utc must not exceed attestation issued_at_utc.",
                {"receipt_issued_at_utc": record_issued.isoformat(), "attestation_issued_at_utc": issued_at_ts.isoformat()},
            )
        if started_ts and record_started and started_ts != record_started:
            add_issue(
                failures,
                "execution_receipt_started_time_mismatch",
                "execution receipt started_at_utc does not match signed payload.",
                {"payload_started_at_utc": started_ts.isoformat(), "receipt_started_at_utc": record_started.isoformat()},
            )
        if finished_ts and record_finished and finished_ts != record_finished:
            add_issue(
                failures,
                "execution_receipt_finished_time_mismatch",
                "execution receipt finished_at_utc does not match signed payload.",
                {
                    "payload_finished_at_utc": finished_ts.isoformat(),
                    "receipt_finished_at_utc": record_finished.isoformat(),
                },
            )

        if exit_code_value is not None:
            if enforce_exit_code_zero is not False and exit_code_value != 0:
                add_issue(
                    failures,
                    "execution_receipt_nonzero_exit_code",
                    "Execution receipt indicates non-zero exit_code.",
                    {"exit_code": exit_code_value},
                )
            elif enforce_exit_code_zero is False and exit_code_value != 0:
                add_issue(
                    warnings,
                    "execution_receipt_nonzero_exit_code_not_enforced",
                    "Execution receipt exit_code is non-zero while enforce_exit_code_zero=false.",
                    {"exit_code": exit_code_value},
                )

    if (
        require_independent_execution_authority
        and signing_fp
        and observed_fp
        and signing_fp.lower() == observed_fp.lower()
    ):
        add_issue(
            failures,
            "execution_authority_not_independent",
            "assurance_policy requires independent execution receipt authority key.",
            {"signing_key_fingerprint": signing_fp.lower(), "execution_key_fingerprint": observed_fp.lower()},
        )
    check_authority_not_revoked(
        role="execution_authority",
        authority_id=authority_id,
        observed_fp=observed_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
    )

    return {
        "present": True,
        "authority_id": authority_id,
        "record_file": str(record_file) if record_file else None,
        "signature_file": str(signature_file) if signature_file else None,
        "public_key_file": str(public_key_file) if public_key_file else None,
        "public_key_fingerprint_observed_sha256": observed_fp,
        "signature_verification": verify_result,
        "enforce_exit_code_zero": False if enforce_exit_code_zero is False else True,
        "exit_code": exit_code_value,
    }


def validate_execution_log_attestation(
    spec_base: Path,
    spec: Dict[str, Any],
    payload_sha256: Optional[str],
    payload_study_id: Optional[str],
    payload_run_id: Optional[str],
    issued_at_ts: Optional[dt.datetime],
    signing_fp: Optional[str],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    artifacts_summary: Dict[str, Any],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    require_execution_log_attestation = True
    require_independent_log_authority = False
    if isinstance(policy, dict):
        require_execution_log_attestation = bool(policy.get("require_execution_log_attestation", True))
        require_independent_log_authority = bool(policy.get("require_independent_log_authority", False))

    block = spec.get("execution_log_attestation")
    if block is None:
        if require_execution_log_attestation:
            add_issue(
                failures,
                "missing_execution_log_attestation",
                "assurance_policy requires execution_log_attestation block.",
                {},
            )
        return {"present": False}
    if not isinstance(block, dict):
        add_issue(
            failures,
            "invalid_execution_log_attestation",
            "execution_log_attestation must be an object.",
            {"actual_type": type(block).__name__},
        )
        return {"present": True}

    authority_id = require_str(block, "authority_id", failures, "attestation_spec.execution_log_attestation")
    artifact_name = require_str(block, "artifact_name", failures, "attestation_spec.execution_log_attestation")
    record_file = collect_required_path(
        spec_base, block, "record_file", failures, "attestation_spec.execution_log_attestation"
    )
    signature_file = collect_required_path(
        spec_base, block, "signature_file", failures, "attestation_spec.execution_log_attestation"
    )
    public_key_file = collect_required_path(
        spec_base, block, "public_key_file", failures, "attestation_spec.execution_log_attestation"
    )
    expected_fp = require_str(
        block, "public_key_fingerprint_sha256", failures, "attestation_spec.execution_log_attestation"
    )

    observed_fp = None
    verify_result: Dict[str, Any] = {}
    if public_key_file is not None:
        observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
        if expected_fp and not SHA256_RE.fullmatch(expected_fp):
            add_issue(
                failures,
                "invalid_execution_log_public_key_fingerprint",
                "execution_log_attestation.public_key_fingerprint_sha256 must be 64-char hex.",
                {"value": expected_fp},
            )
        elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
            add_issue(
                failures,
                "execution_log_public_key_fingerprint_mismatch",
                "Execution-log attestation public key fingerprint mismatch.",
                {"expected": expected_fp.lower(), "observed": observed_fp.lower()},
            )

    if record_file and signature_file and public_key_file:
        verify_result = verify_detached_signature(
            data_file=record_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method="openssl-dgst-sha256",
            failures=failures,
            scope="execution_log_record",
        )

    checked_artifacts = artifacts_summary.get("checked")
    if not isinstance(checked_artifacts, list):
        checked_artifacts = []

    artifact_entry: Optional[Dict[str, Any]] = None
    if artifact_name:
        for item in checked_artifacts:
            if isinstance(item, dict) and str(item.get("name")) == artifact_name:
                artifact_entry = item
                break
        if artifact_entry is None:
            add_issue(
                failures,
                "execution_log_artifact_not_found",
                "execution_log_attestation artifact_name not found in signed payload artifacts.",
                {"artifact_name": artifact_name},
            )

    record = load_json_obj(record_file, failures, "execution_log_record") if record_file else None
    line_count_value: Optional[int] = None
    if record is not None:
        record_authority = require_str(record, "authority_id", failures, "execution_log_record")
        record_payload_sha = require_str(record, "payload_sha256", failures, "execution_log_record")
        record_study = require_str(record, "study_id", failures, "execution_log_record")
        record_run = require_str(record, "run_id", failures, "execution_log_record")
        record_artifact_name = require_str(record, "artifact_name", failures, "execution_log_record")
        record_artifact_path = require_str(record, "artifact_path", failures, "execution_log_record")
        record_artifact_sha = require_str(record, "artifact_sha256", failures, "execution_log_record")
        record_issued_raw = require_str(record, "issued_at_utc", failures, "execution_log_record")

        raw_line_count = record.get("line_count")
        if isinstance(raw_line_count, bool):
            raw_line_count = None
        if isinstance(raw_line_count, int):
            line_count_value = int(raw_line_count)
        elif isinstance(raw_line_count, float) and math.isfinite(raw_line_count) and float(raw_line_count).is_integer():
            line_count_value = int(raw_line_count)
        elif raw_line_count is not None:
            add_issue(
                failures,
                "invalid_execution_log_line_count",
                "execution_log_record.line_count must be integer when present.",
                {"actual_value": raw_line_count},
            )

        if authority_id and record_authority and authority_id != record_authority:
            add_issue(
                failures,
                "execution_log_authority_mismatch",
                "execution-log record authority_id does not match attestation spec.",
                {"spec_authority_id": authority_id, "record_authority_id": record_authority},
            )
        if artifact_name and record_artifact_name and artifact_name != record_artifact_name:
            add_issue(
                failures,
                "execution_log_artifact_name_mismatch",
                "execution-log record artifact_name does not match attestation spec.",
                {"spec_artifact_name": artifact_name, "record_artifact_name": record_artifact_name},
            )
        if payload_sha256 and record_payload_sha and payload_sha256.lower() != record_payload_sha.lower():
            add_issue(
                failures,
                "execution_log_payload_hash_mismatch",
                "execution-log record payload_sha256 does not match signed payload.",
                {"payload_sha256": payload_sha256.lower(), "record_payload_sha256": record_payload_sha.lower()},
            )
        if payload_study_id and record_study and payload_study_id != record_study:
            add_issue(
                failures,
                "execution_log_study_id_mismatch",
                "execution-log record study_id mismatch.",
                {"payload_study_id": payload_study_id, "record_study_id": record_study},
            )
        if payload_run_id and record_run and payload_run_id != record_run:
            add_issue(
                failures,
                "execution_log_run_id_mismatch",
                "execution-log record run_id mismatch.",
                {"payload_run_id": payload_run_id, "record_run_id": record_run},
            )

        if record_artifact_sha and not SHA256_RE.fullmatch(record_artifact_sha):
            add_issue(
                failures,
                "invalid_execution_log_artifact_sha256",
                "execution_log_record.artifact_sha256 must be 64-char hex.",
                {"artifact_sha256": record_artifact_sha},
            )

        if artifact_entry is not None:
            artifact_entry_sha = str(artifact_entry.get("sha256", "")).lower()
            if record_artifact_sha and record_artifact_sha.lower() != artifact_entry_sha:
                add_issue(
                    failures,
                    "execution_log_artifact_hash_mismatch",
                    "execution-log record artifact_sha256 does not match signed artifact hash.",
                    {
                        "artifact_name": artifact_name,
                        "record_artifact_sha256": record_artifact_sha.lower(),
                        "signed_artifact_sha256": artifact_entry_sha,
                    },
                )

            artifact_entry_path = Path(str(artifact_entry.get("path"))).resolve()
            if record_artifact_path:
                rp = Path(record_artifact_path).expanduser()
                if not rp.is_absolute() and record_file is not None:
                    rp = (record_file.parent / rp).resolve()
                else:
                    rp = rp.resolve()
                if artifact_entry_path != rp:
                    add_issue(
                        failures,
                        "execution_log_artifact_path_mismatch",
                        "execution-log record artifact_path does not match signed artifact path.",
                        {
                            "artifact_name": artifact_name,
                            "record_artifact_path": str(rp),
                            "signed_artifact_path": str(artifact_entry_path),
                        },
                    )

                actual_line_count = file_line_count(artifact_entry_path)
                if line_count_value is not None and actual_line_count is not None and line_count_value != actual_line_count:
                    add_issue(
                        failures,
                        "execution_log_line_count_mismatch",
                        "execution-log record line_count does not match actual artifact line count.",
                        {
                            "artifact_name": artifact_name,
                            "record_line_count": line_count_value,
                            "actual_line_count": actual_line_count,
                        },
                    )
                elif line_count_value is not None and actual_line_count is None:
                    add_issue(
                        warnings,
                        "execution_log_line_count_not_verifiable",
                        "Unable to verify execution-log line_count due to unreadable artifact.",
                        {"artifact_name": artifact_name, "artifact_path": str(artifact_entry_path)},
                    )

        record_issued = parse_iso_ts(record_issued_raw) if record_issued_raw else None
        if record_issued_raw and record_issued is None:
            add_issue(
                failures,
                "invalid_execution_log_issued_time",
                "execution_log_record.issued_at_utc must be ISO-8601 timestamp.",
                {"issued_at_utc": record_issued_raw},
            )
        if issued_at_ts and record_issued and record_issued > issued_at_ts:
            add_issue(
                failures,
                "execution_log_after_attestation_issue",
                "execution-log record issued_at_utc must not exceed attestation issued_at_utc.",
                {"record_issued_at_utc": record_issued.isoformat(), "attestation_issued_at_utc": issued_at_ts.isoformat()},
            )

    if (
        require_independent_log_authority
        and signing_fp
        and observed_fp
        and signing_fp.lower() == observed_fp.lower()
    ):
        add_issue(
            failures,
            "execution_log_authority_not_independent",
            "assurance_policy requires independent execution-log authority key.",
            {"signing_key_fingerprint": signing_fp.lower(), "log_key_fingerprint": observed_fp.lower()},
        )
    check_authority_not_revoked(
        role="execution_log_authority",
        authority_id=authority_id,
        observed_fp=observed_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
    )

    return {
        "present": True,
        "authority_id": authority_id,
        "artifact_name": artifact_name,
        "record_file": str(record_file) if record_file else None,
        "signature_file": str(signature_file) if signature_file else None,
        "public_key_file": str(public_key_file) if public_key_file else None,
        "public_key_fingerprint_observed_sha256": observed_fp,
        "signature_verification": verify_result,
    }


def validate_witness_quorum(
    spec_base: Path,
    spec: Dict[str, Any],
    payload_sha256: Optional[str],
    payload_study_id: Optional[str],
    payload_run_id: Optional[str],
    finished_ts: Optional[dt.datetime],
    issued_at_ts: Optional[dt.datetime],
    signing_fp: Optional[str],
    revoked_key_ids: Set[str],
    revoked_key_fps: Set[str],
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    policy = spec.get("assurance_policy", {})
    require_witness_quorum = False
    min_witness_count = 0
    policy_min_witness_count = 0
    require_independent_witness_keys = False
    require_witness_independence_from_signing = False
    if isinstance(policy, dict):
        require_witness_quorum = bool(policy.get("require_witness_quorum", False))
        raw_min = policy.get("min_witness_count", 2 if require_witness_quorum else 0)
        if isinstance(raw_min, bool):
            raw_min = None
        if isinstance(raw_min, int):
            min_witness_count = int(raw_min)
        elif isinstance(raw_min, float) and math.isfinite(raw_min) and float(raw_min).is_integer():
            min_witness_count = int(raw_min)
        else:
            add_issue(
                failures,
                "invalid_witness_min_count",
                "assurance_policy.min_witness_count must be an integer.",
                {"value": raw_min},
            )
            min_witness_count = 0
        policy_min_witness_count = int(min_witness_count)
        require_independent_witness_keys = bool(policy.get("require_independent_witness_keys", require_witness_quorum))
        require_witness_independence_from_signing = bool(
            policy.get("require_witness_independence_from_signing", require_witness_quorum)
        )

    if require_witness_quorum and min_witness_count < 1:
        add_issue(
            failures,
            "invalid_witness_min_count",
            "assurance_policy.min_witness_count must be >= 1 when witness quorum is required.",
            {"min_witness_count": min_witness_count},
        )

    block = spec.get("witness_quorum")
    if block is None:
        if require_witness_quorum:
            add_issue(
                failures,
                "missing_witness_quorum",
                "assurance_policy requires witness_quorum block.",
                {},
            )
        return {"present": False, "required": bool(require_witness_quorum), "min_witness_count": min_witness_count}
    if not require_witness_quorum:
        add_issue(
            failures,
            "witness_quorum_policy_disabled",
            "witness_quorum block present but assurance_policy.require_witness_quorum is false.",
            {},
        )
    if not isinstance(block, dict):
        add_issue(
            failures,
            "invalid_witness_quorum",
            "witness_quorum must be an object.",
            {"actual_type": type(block).__name__},
        )
        return {"present": True, "required": bool(require_witness_quorum), "min_witness_count": min_witness_count}

    block_min_raw = block.get("min_witness_count")
    if block_min_raw is not None:
        if isinstance(block_min_raw, bool):
            block_min_raw = None
        if isinstance(block_min_raw, int):
            min_witness_count = int(block_min_raw)
        elif isinstance(block_min_raw, float) and math.isfinite(block_min_raw) and float(block_min_raw).is_integer():
            min_witness_count = int(block_min_raw)
        else:
            add_issue(
                failures,
                "invalid_witness_min_count",
                "witness_quorum.min_witness_count must be an integer when provided.",
                {"value": block_min_raw},
            )
    if require_witness_quorum and block_min_raw is not None and min_witness_count != policy_min_witness_count:
        add_issue(
            failures,
            "witness_min_count_mismatch",
            "witness_quorum.min_witness_count must match assurance_policy.min_witness_count.",
            {"policy_min_witness_count": policy_min_witness_count, "block_min_witness_count": min_witness_count},
        )

    records = block.get("records")
    if not isinstance(records, list) or not records:
        add_issue(
            failures,
            "missing_witness_records",
            "witness_quorum.records must be a non-empty list.",
            {"actual_type": type(records).__name__ if records is not None else None},
        )
        return {"present": True, "required": bool(require_witness_quorum), "min_witness_count": min_witness_count, "validated_witnesses": 0}

    validated: List[Dict[str, Any]] = []
    valid_fps: List[str] = []
    valid_authorities: List[str] = []
    effective_quorum_required = bool(require_witness_quorum or block_min_raw is not None)
    if effective_quorum_required and min_witness_count < 1:
        add_issue(
            failures,
            "invalid_witness_min_count",
            "Effective witness quorum requires min_witness_count >= 1.",
            {"min_witness_count": min_witness_count},
        )

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            add_issue(
                failures,
                "invalid_witness_record_spec",
                "witness_quorum.records items must be objects.",
                {"index": idx, "actual_type": type(item).__name__},
            )
            continue

        start_failures = len(failures)
        scope = f"attestation_spec.witness_quorum.records[{idx}]"
        authority_id = require_str(item, "authority_id", failures, scope)
        record_file = collect_required_path(spec_base, item, "record_file", failures, scope)
        signature_file = collect_required_path(spec_base, item, "signature_file", failures, scope)
        public_key_file = collect_required_path(spec_base, item, "public_key_file", failures, scope)
        expected_fp = require_str(item, "public_key_fingerprint_sha256", failures, scope)

        observed_fp = None
        verify_result: Dict[str, Any] = {}
        if public_key_file is not None:
            observed_fp = public_key_fingerprint_sha256(public_key_file, failures)
            if expected_fp and not SHA256_RE.fullmatch(expected_fp):
                add_issue(
                    failures,
                    "invalid_witness_public_key_fingerprint",
                    "witness_quorum public_key_fingerprint_sha256 must be 64-char hex.",
                    {"index": idx, "value": expected_fp},
                )
            elif expected_fp and observed_fp and expected_fp.lower() != observed_fp.lower():
                add_issue(
                    failures,
                    "witness_public_key_fingerprint_mismatch",
                    "Witness public key fingerprint mismatch.",
                    {"index": idx, "expected": expected_fp.lower(), "observed": observed_fp.lower()},
                )

        check_authority_not_revoked(
            role="witness_authority",
            authority_id=authority_id,
            observed_fp=observed_fp,
            revoked_key_ids=revoked_key_ids,
            revoked_key_fps=revoked_key_fps,
            failures=failures,
            extra_details={"index": idx},
        )

        if record_file and signature_file and public_key_file:
            verify_result = verify_detached_signature(
                data_file=record_file,
                signature_file=signature_file,
                public_key_file=public_key_file,
                method="openssl-dgst-sha256",
                failures=failures,
                scope=f"witness_record_{idx}",
            )

        record = load_json_obj(record_file, failures, f"witness_record_{idx}") if record_file else None
        if record is not None:
            rec_authority = require_str(record, "authority_id", failures, f"witness_record_{idx}")
            rec_payload_sha = require_str(record, "payload_sha256", failures, f"witness_record_{idx}")
            rec_study = require_str(record, "study_id", failures, f"witness_record_{idx}")
            rec_run = require_str(record, "run_id", failures, f"witness_record_{idx}")
            rec_attested = require_str(record, "attested_at_utc", failures, f"witness_record_{idx}")

            if authority_id and rec_authority and authority_id != rec_authority:
                add_issue(
                    failures,
                    "witness_authority_mismatch",
                    "Witness record authority_id mismatch.",
                    {"index": idx, "spec_authority_id": authority_id, "record_authority_id": rec_authority},
                )
            if payload_sha256 and rec_payload_sha and payload_sha256.lower() != rec_payload_sha.lower():
                add_issue(
                    failures,
                    "witness_payload_hash_mismatch",
                    "Witness record payload_sha256 does not match signed payload.",
                    {"index": idx, "payload_sha256": payload_sha256.lower(), "record_payload_sha256": rec_payload_sha.lower()},
                )
            if payload_study_id and rec_study and payload_study_id != rec_study:
                add_issue(
                    failures,
                    "witness_study_id_mismatch",
                    "Witness record study_id mismatch.",
                    {"index": idx, "payload_study_id": payload_study_id, "record_study_id": rec_study},
                )
            if payload_run_id and rec_run and payload_run_id != rec_run:
                add_issue(
                    failures,
                    "witness_run_id_mismatch",
                    "Witness record run_id mismatch.",
                    {"index": idx, "payload_run_id": payload_run_id, "record_run_id": rec_run},
                )

            attested_ts = parse_iso_ts(rec_attested) if rec_attested else None
            if rec_attested and attested_ts is None:
                add_issue(
                    failures,
                    "invalid_witness_attested_time",
                    "Witness record attested_at_utc must be ISO-8601 timestamp.",
                    {"index": idx, "attested_at_utc": rec_attested},
                )
            if finished_ts and attested_ts and attested_ts < finished_ts:
                add_issue(
                    failures,
                    "witness_before_execution_finished",
                    "Witness attested_at_utc must be >= execution finished_at_utc.",
                    {"index": idx, "attested_at_utc": attested_ts.isoformat(), "finished_at_utc": finished_ts.isoformat()},
                )
            if issued_at_ts and attested_ts and attested_ts > issued_at_ts:
                add_issue(
                    failures,
                    "witness_after_attestation_issue",
                    "Witness attested_at_utc must not exceed attestation issued_at_utc.",
                    {"index": idx, "attested_at_utc": attested_ts.isoformat(), "issued_at_utc": issued_at_ts.isoformat()},
                )

        is_valid = len(failures) == start_failures and bool(verify_result.get("verified"))
        if is_valid:
            valid_authorities.append(str(authority_id))
            if isinstance(observed_fp, str):
                valid_fps.append(observed_fp.lower())
        validated.append(
            {
                "index": idx,
                "authority_id": authority_id,
                "record_file": str(record_file) if record_file else None,
                "signature_file": str(signature_file) if signature_file else None,
                "public_key_file": str(public_key_file) if public_key_file else None,
                "public_key_fingerprint_observed_sha256": observed_fp,
                "signature_verification": verify_result,
                "validated": bool(is_valid),
            }
        )

    unique_fps = set(valid_fps)
    unique_authorities = {a for a in valid_authorities if a}
    validated_count = sum(1 for item in validated if item.get("validated"))

    if effective_quorum_required and validated_count < min_witness_count:
        add_issue(
            failures,
            "witness_quorum_not_met",
            "Validated witness count is below required minimum.",
            {"validated_count": validated_count, "min_witness_count": min_witness_count},
        )
    if require_independent_witness_keys and len(unique_fps) != len(valid_fps):
        add_issue(
            failures,
            "witness_keys_not_independent",
            "Witness signatures must use distinct public keys.",
            {"validated_witnesses": validated_count, "unique_key_count": len(unique_fps)},
        )
    if require_independent_witness_keys and len(unique_authorities) != len(valid_authorities):
        add_issue(
            failures,
            "witness_authorities_not_distinct",
            "Witness signatures must use distinct authority IDs.",
            {"validated_witnesses": validated_count, "unique_authority_count": len(unique_authorities)},
        )
    if require_witness_independence_from_signing and signing_fp:
        if any(fp == signing_fp.lower() for fp in unique_fps):
            add_issue(
                failures,
                "witness_key_matches_signing_key",
                "Witness keys must be independent from payload signing key.",
                {"signing_key_fingerprint": signing_fp.lower()},
            )

    return {
        "present": True,
        "required": bool(effective_quorum_required),
        "min_witness_count": int(min_witness_count),
        "validated_witnesses": validated_count,
        "witnesses": validated,
    }


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    spec_path = Path(args.attestation_spec).expanduser().resolve()
    eval_report = Path(args.evaluation_report).expanduser().resolve()
    spec_base = spec_path.parent

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
    issued_at_raw = require_str(spec, "issued_at_utc", failures, "attestation_spec")
    required_names = require_str_list(spec, "required_artifact_names", failures, "attestation_spec")

    issued_at_ts = parse_iso_ts(issued_at_raw) if issued_at_raw else None
    if issued_at_raw and issued_at_ts is None:
        add_issue(
            failures,
            "invalid_issued_timestamp",
            "issued_at_utc must be ISO-8601 timestamp.",
            {"issued_at_utc": issued_at_raw},
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

    signing_method = require_str(signing, "method", failures, "attestation_spec.signing")
    payload_file = collect_required_path(spec_base, signing, "signed_payload_file", failures, "attestation_spec.signing")
    signature_file = collect_required_path(spec_base, signing, "signature_file", failures, "attestation_spec.signing")
    public_key_file = collect_required_path(spec_base, signing, "public_key_file", failures, "attestation_spec.signing")

    signature_verification: Dict[str, Any] = {}
    if payload_file and signature_file and public_key_file and signing_method:
        signature_verification = verify_detached_signature(
            data_file=payload_file,
            signature_file=signature_file,
            public_key_file=public_key_file,
            method=signing_method,
            failures=failures,
            scope="attestation_payload",
        )

    payload = load_json_obj(payload_file, failures, "signed_payload") if payload_file else None
    if payload is None:
        return finish(args, failures, warnings, {"signature_verification": signature_verification}, {})

    payload_study_id = require_str(payload, "study_id", failures, "signed_payload")
    payload_run_id = require_str(payload, "run_id", failures, "signed_payload")
    payload_command = require_str(payload, "command", failures, "signed_payload")
    started_at_raw = require_str(payload, "started_at_utc", failures, "signed_payload")
    finished_at_raw = require_str(payload, "finished_at_utc", failures, "signed_payload")
    executor = require_str(payload, "executor", failures, "signed_payload")
    payload_exit_code: Optional[int] = None
    payload_exit_raw = payload.get("exit_code")
    if payload_exit_raw is not None:
        if isinstance(payload_exit_raw, bool):
            payload_exit_raw = None
        if isinstance(payload_exit_raw, int):
            payload_exit_code = int(payload_exit_raw)
        elif isinstance(payload_exit_raw, float) and math.isfinite(payload_exit_raw) and float(payload_exit_raw).is_integer():
            payload_exit_code = int(payload_exit_raw)
        else:
            add_issue(
                failures,
                "invalid_payload_exit_code",
                "signed_payload.exit_code must be integer when present.",
                {"actual_value": payload_exit_raw},
            )

    git_commit = payload.get("git_commit")
    if git_commit is not None and (
        not isinstance(git_commit, str) or not GIT_COMMIT_RE.fullmatch(git_commit.strip())
    ):
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

    started_ts = parse_iso_ts(started_at_raw) if started_at_raw else None
    finished_ts = parse_iso_ts(finished_at_raw) if finished_at_raw else None
    if started_at_raw and started_ts is None:
        add_issue(
            failures,
            "invalid_started_timestamp",
            "started_at_utc must be ISO-8601 timestamp.",
            {"started_at_utc": started_at_raw},
        )
    if finished_at_raw and finished_ts is None:
        add_issue(
            failures,
            "invalid_finished_timestamp",
            "finished_at_utc must be ISO-8601 timestamp.",
            {"finished_at_utc": finished_at_raw},
        )
    if started_ts and finished_ts and started_ts > finished_ts:
        add_issue(
            failures,
            "invalid_execution_time_window",
            "started_at_utc must be <= finished_at_utc.",
            {"started_at_utc": started_at_raw, "finished_at_utc": finished_at_raw},
        )
    if finished_ts and issued_at_ts and finished_ts > issued_at_ts:
        add_issue(
            failures,
            "invalid_attestation_issue_time",
            "issued_at_utc must be at or after finished_at_utc.",
            {"issued_at_utc": issued_at_raw, "finished_at_utc": finished_at_raw},
        )

    payload_digest_declared = spec.get("signed_payload_sha256")
    payload_digest_actual = sha256_file(payload_file) if payload_file else None
    if payload_file is not None:
        if not isinstance(payload_digest_declared, str):
            add_issue(
                failures,
                "missing_signed_payload_sha256",
                "attestation_spec must include signed_payload_sha256.",
                {},
            )
        elif not SHA256_RE.fullmatch(payload_digest_declared.strip()):
            add_issue(
                failures,
                "invalid_signed_payload_sha256",
                "signed_payload_sha256 must be 64-char hex string.",
                {"signed_payload_sha256": payload_digest_declared},
            )
        elif payload_digest_actual and payload_digest_actual.lower() != payload_digest_declared.strip().lower():
            add_issue(
                failures,
                "signed_payload_hash_mismatch",
                "signed_payload_sha256 does not match payload file.",
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

    key_assurance = validate_key_assurance(
        spec_base=spec_base,
        spec=spec,
        signing=signing,
        issued_at_ts=issued_at_ts,
        failures=failures,
        warnings=warnings,
    )
    revoked_key_ids_raw = key_assurance.get("revoked_key_ids")
    revoked_key_fps_raw = key_assurance.get("revoked_public_key_fingerprints_sha256")
    revoked_key_ids: Set[str] = (
        {str(x).strip() for x in revoked_key_ids_raw if isinstance(x, str) and str(x).strip()}
        if isinstance(revoked_key_ids_raw, list)
        else set()
    )
    revoked_key_fps: Set[str] = (
        {str(x).strip().lower() for x in revoked_key_fps_raw if isinstance(x, str) and str(x).strip()}
        if isinstance(revoked_key_fps_raw, list)
        else set()
    )
    if args.strict:
        enforce_publication_policy_requirements(key_assurance=key_assurance, failures=failures)
    signing_fp = key_assurance.get("public_key_fingerprint_observed_sha256")
    if signing_fp is not None and not isinstance(signing_fp, str):
        signing_fp = None

    timestamp_summary = validate_timestamp_trust(
        spec_base=spec_base,
        spec=spec,
        payload_sha256=payload_digest_actual,
        payload_study_id=payload_study_id,
        payload_run_id=payload_run_id,
        finished_ts=finished_ts,
        issued_at_ts=issued_at_ts,
        signing_fp=signing_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
        warnings=warnings,
    )

    transparency_summary = validate_transparency_log(
        spec_base=spec_base,
        spec=spec,
        payload_sha256=payload_digest_actual,
        payload_study_id=payload_study_id,
        payload_run_id=payload_run_id,
        finished_ts=finished_ts,
        issued_at_ts=issued_at_ts,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
    )
    execution_receipt_summary = validate_execution_receipt(
        spec_base=spec_base,
        spec=spec,
        payload_sha256=payload_digest_actual,
        payload_study_id=payload_study_id,
        payload_run_id=payload_run_id,
        payload_command=payload_command,
        payload_executor=executor,
        payload_exit_code=payload_exit_code,
        started_ts=started_ts,
        finished_ts=finished_ts,
        issued_at_ts=issued_at_ts,
        signing_fp=signing_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        failures=failures,
        warnings=warnings,
    )
    execution_log_summary = validate_execution_log_attestation(
        spec_base=spec_base,
        spec=spec,
        payload_sha256=payload_digest_actual,
        payload_study_id=payload_study_id,
        payload_run_id=payload_run_id,
        issued_at_ts=issued_at_ts,
        signing_fp=signing_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
        artifacts_summary=artifacts_summary,
        failures=failures,
        warnings=warnings,
    )
    witness_quorum_summary = validate_witness_quorum(
        spec_base=spec_base,
        spec=spec,
        payload_sha256=payload_digest_actual,
        payload_study_id=payload_study_id,
        payload_run_id=payload_run_id,
        finished_ts=finished_ts,
        issued_at_ts=issued_at_ts,
        signing_fp=signing_fp,
        revoked_key_ids=revoked_key_ids,
        revoked_key_fps=revoked_key_fps,
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
        "payload_exit_code": payload_exit_code,
        "signature_verification": signature_verification,
        "artifacts": artifacts_summary,
        "key_assurance": key_assurance,
        "timestamp_trust": timestamp_summary,
        "transparency_log": transparency_summary,
        "execution_receipt": execution_receipt_summary,
        "execution_log_attestation": execution_log_summary,
        "witness_quorum": witness_quorum_summary,
    }

    return finish(
        args,
        failures,
        warnings,
        summary,
        payload_metadata={
            "started_at_utc": started_at_raw,
            "finished_at_utc": finished_at_raw,
            "issued_at_utc": issued_at_raw,
        },
    )


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
