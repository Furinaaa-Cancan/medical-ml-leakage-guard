#!/usr/bin/env python3
"""
Generate and sign execution attestation artifacts for publication-grade review.

Outputs:
1. signed payload JSON (contains artifact hashes)
2. detached payload signature
3. key revocation list (bootstrapped if absent)
4. signed timestamp record
5. signed transparency-log record
6. signed execution receipt record
7. attestation spec consumed by execution_attestation_gate.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RUN_NONCE_RE = re.compile(r"^[a-fA-F0-9]{16,128}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create signed execution attestation bundle (payload + records + spec).")
    parser.add_argument("--study-id", required=True, help="Study identifier.")
    parser.add_argument("--run-id", required=True, help="Training run identifier.")
    parser.add_argument("--payload-out", required=True, help="Output JSON path for signed payload.")
    parser.add_argument("--spec-out", required=True, help="Output JSON path for attestation spec.")
    parser.add_argument("--signature-out", required=True, help="Output path for detached payload signature.")
    parser.add_argument("--public-key-file", required=True, help="Public key PEM used for payload verification.")
    parser.add_argument("--private-key-file", help="Private key PEM used for payload signing.")
    parser.add_argument("--skip-sign", action="store_true", help="Skip detached signatures (bootstrap scaffold only).")

    parser.add_argument("--key-id", default="attestation-key-default", help="Logical signing key identifier.")
    parser.add_argument("--key-created-at-utc", help="Signing key creation time ISO-8601 UTC.")
    parser.add_argument("--key-not-after-utc", help="Signing key expiry time ISO-8601 UTC.")
    parser.add_argument("--min-signing-key-bits", type=int, default=3072, help="assurance_policy.min_signing_key_bits.")
    parser.add_argument(
        "--max-signing-key-age-days", type=int, default=180, help="assurance_policy.max_signing_key_age_days."
    )
    parser.add_argument(
        "--revocation-list-file",
        help="JSON file with revoked key IDs/fingerprints. If absent, auto-created next to spec.",
    )

    parser.add_argument("--skip-timestamp-trust", action="store_true", help="Do not generate timestamp_trust block.")
    parser.add_argument(
        "--timestamp-authority-id",
        default="timestamp-authority-local",
        help="Authority ID in timestamp trust record.",
    )
    parser.add_argument("--timestamp-record-out", help="Output JSON path for timestamp record.")
    parser.add_argument("--timestamp-signature-out", help="Output path for timestamp record signature.")
    parser.add_argument("--timestamp-public-key-file", help="Public key PEM for timestamp record verification.")
    parser.add_argument("--timestamp-private-key-file", help="Private key PEM for timestamp record signing.")

    parser.add_argument("--skip-transparency-log", action="store_true", help="Do not generate transparency_log block.")
    parser.add_argument("--transparency-log-id", default="transparency-log-local", help="Transparency log identifier.")
    parser.add_argument("--transparency-record-out", help="Output JSON path for transparency record.")
    parser.add_argument("--transparency-signature-out", help="Output path for transparency record signature.")
    parser.add_argument("--transparency-public-key-file", help="Public key PEM for transparency record verification.")
    parser.add_argument("--transparency-private-key-file", help="Private key PEM for transparency record signing.")

    parser.add_argument("--skip-execution-receipt", action="store_true", help="Do not generate execution_receipt block.")
    parser.add_argument(
        "--execution-authority-id",
        default="execution-authority-local",
        help="Authority ID in execution receipt record.",
    )
    parser.add_argument("--execution-receipt-record-out", help="Output JSON path for execution receipt record.")
    parser.add_argument("--execution-receipt-signature-out", help="Output path for execution receipt signature.")
    parser.add_argument("--execution-public-key-file", help="Public key PEM for execution receipt verification.")
    parser.add_argument("--execution-private-key-file", help="Private key PEM for execution receipt signing.")
    parser.add_argument(
        "--execution-exit-code",
        type=int,
        default=0,
        help="Process exit code to record in execution receipt and payload (default: 0).",
    )

    parser.add_argument(
        "--require-independent-timestamp-authority",
        action="store_true",
        help="Set assurance_policy.require_independent_timestamp_authority=true in spec.",
    )
    parser.add_argument(
        "--require-independent-execution-authority",
        action="store_true",
        help="Set assurance_policy.require_independent_execution_authority=true in spec.",
    )
    parser.add_argument(
        "--require-independent-log-authority",
        action="store_true",
        help="Set assurance_policy.require_independent_log_authority=true in spec.",
    )
    parser.add_argument(
        "--require-witness-quorum",
        action="store_true",
        help="Set assurance_policy.require_witness_quorum=true in spec.",
    )
    parser.add_argument(
        "--min-witness-count",
        type=int,
        default=2,
        help="Minimum required witness signatures when quorum is required (default: 2).",
    )
    parser.add_argument(
        "--require-independent-witness-keys",
        action="store_true",
        help="Set assurance_policy.require_independent_witness_keys=true in spec.",
    )
    parser.add_argument(
        "--require-witness-independence-from-signing",
        action="store_true",
        help="Set assurance_policy.require_witness_independence_from_signing=true in spec.",
    )
    parser.add_argument(
        "--witness",
        action="append",
        default=[],
        help="Witness tuple AUTHORITY_ID|PUBLIC_KEY_FILE|PRIVATE_KEY_FILE. Repeat for multiple witnesses.",
    )

    parser.add_argument(
        "--skip-execution-log-attestation",
        action="store_true",
        help="Do not generate execution_log_attestation block.",
    )
    parser.add_argument(
        "--execution-log-authority-id",
        default="execution-log-authority-local",
        help="Authority ID in execution log attestation record.",
    )
    parser.add_argument(
        "--execution-log-artifact-name",
        default="training_log",
        help="Artifact name in signed payload to attest (default: training_log).",
    )
    parser.add_argument("--execution-log-record-out", help="Output JSON path for execution-log record.")
    parser.add_argument("--execution-log-signature-out", help="Output path for execution-log record signature.")
    parser.add_argument("--execution-log-public-key-file", help="Public key PEM for execution-log verification.")
    parser.add_argument("--execution-log-private-key-file", help="Private key PEM for execution-log signing.")

    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact in NAME=PATH form. Repeat for multiple artifacts.",
    )
    parser.add_argument(
        "--required-artifact",
        action="append",
        default=[],
        help="Artifact name that must exist during verification. Repeat as needed.",
    )
    parser.add_argument("--command", required=True, help="Training command line used for the run.")
    parser.add_argument("--executor", help="Executor identity, e.g. user@host. Default auto-derived.")
    parser.add_argument(
        "--run-nonce",
        help="Optional run nonce (hex). Default is auto-generated 128-bit hex for per-run binding.",
    )
    parser.add_argument("--git-commit", help="Git commit hash for training code.")
    parser.add_argument("--started-at-utc", help="Run start time ISO-8601 UTC.")
    parser.add_argument("--finished-at-utc", help="Run end time ISO-8601 UTC. Defaults to now.")
    parser.add_argument("--issued-at-utc", help="Attestation issue time ISO-8601 UTC. Defaults to finished-at-utc.")
    return parser.parse_args()


def iso_now_utc() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def iso_after_days(base_iso_utc: str, days: int) -> str:
    base = parse_iso_utc(base_iso_utc)
    if base is None:
        base = dt.datetime.now(tz=dt.timezone.utc)
    target = base + dt.timedelta(days=int(days))
    return target.isoformat().replace("+00:00", "Z")


def parse_iso_utc(raw: str) -> Optional[dt.datetime]:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_for_output(base: Path, path: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def parse_artifact(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --artifact '{raw}'. Expected NAME=PATH.")
    name, path = raw.split("=", 1)
    name_clean = name.strip()
    path_clean = path.strip()
    if not name_clean or not path_clean:
        raise ValueError(f"Invalid --artifact '{raw}'. NAME and PATH must be non-empty.")
    return name_clean, Path(path_clean).expanduser().resolve()


def parse_witness(raw: str) -> Tuple[str, Path, Path]:
    parts = [x.strip() for x in raw.split("|")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --witness '{raw}'. Expected AUTHORITY_ID|PUBLIC_KEY_FILE|PRIVATE_KEY_FILE."
        )
    authority_id, pub_raw, priv_raw = parts
    if not authority_id or not pub_raw or not priv_raw:
        raise ValueError(
            f"Invalid --witness '{raw}'. AUTHORITY_ID, PUBLIC_KEY_FILE and PRIVATE_KEY_FILE must be non-empty."
        )
    return (
        authority_id,
        Path(pub_raw).expanduser().resolve(),
        Path(priv_raw).expanduser().resolve(),
    )


def default_executor() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown-user"
    host = socket.gethostname() or "unknown-host"
    return f"{user}@{host}"


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")


def openssl_sign_file(private_key: Path, input_file: Path, output_sig: Path) -> None:
    cmd = [
        "openssl",
        "dgst",
        "-sha256",
        "-sign",
        str(private_key),
        "-out",
        str(output_sig),
        str(input_file),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("openssl command not found; required for detached signature creation.") from exc
    if proc.returncode != 0:
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"Detached signature creation failed.\nstdout={out}\nstderr={err}")


def public_key_der_bytes(public_key_file: Path) -> bytes:
    cmd = ["openssl", "pkey", "-pubin", "-in", str(public_key_file), "-outform", "DER"]
    try:
        proc = subprocess.run(cmd, text=False, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("openssl command not found; required for key fingerprinting.") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to parse public key DER.\nstderr={stderr}")
    return proc.stdout


def public_key_fingerprint_sha256(public_key_file: Path) -> str:
    return hashlib.sha256(public_key_der_bytes(public_key_file)).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    tmp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}"
    )
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            total += 1
    return total


def log_boundary_hashes(path: Path) -> Tuple[int, Optional[str], Optional[str]]:
    first_line: Optional[str] = None
    last_line: Optional[str] = None
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            total += 1
            normalized = raw_line.rstrip("\r\n")
            if first_line is None:
                first_line = normalized
            last_line = normalized
    first_hash = sha256_text(first_line) if first_line is not None else None
    last_hash = sha256_text(last_line) if last_line is not None else None
    return total, first_hash, last_hash


def ensure_revocation_file(path: Path) -> Dict[str, Any]:
    ensure_parent(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in revocation list {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"revocation_list_file must be JSON object: {path}")
        if "revoked_key_ids" not in loaded:
            loaded["revoked_key_ids"] = []
        if "revoked_public_key_fingerprints_sha256" not in loaded:
            loaded["revoked_public_key_fingerprints_sha256"] = []
    else:
        loaded = {
            "schema_version": "1.0",
            "revoked_key_ids": [],
            "revoked_public_key_fingerprints_sha256": [],
        }
    from _gate_utils import write_json as _write_revocation
    _write_revocation(path, loaded)
    return loaded


def main() -> int:
    args = parse_args()

    payload_out = Path(args.payload_out).expanduser().resolve()
    spec_out = Path(args.spec_out).expanduser().resolve()
    signature_out = Path(args.signature_out).expanduser().resolve()
    public_key = Path(args.public_key_file).expanduser().resolve()
    private_key = Path(args.private_key_file).expanduser().resolve() if args.private_key_file else None

    if not args.artifact:
        print("[FAIL] At least one --artifact NAME=PATH is required.", file=sys.stderr)
        return 2

    try:
        ensure_file(public_key, "public_key_file")
        if not args.skip_sign:
            if private_key is None:
                print("[FAIL] --private-key-file is required unless --skip-sign is set.", file=sys.stderr)
                return 2
            ensure_file(private_key, "private_key_file")
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    artifacts: List[Dict[str, Any]] = []
    names_seen = set()
    try:
        for raw in args.artifact:
            name, path = parse_artifact(raw)
            if name in names_seen:
                raise ValueError(f"Duplicate artifact name: {name}")
            names_seen.add(name)
            ensure_file(path, f"artifact '{name}'")
            artifacts.append(
                {
                    "name": name,
                    "abs_path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    if "evaluation_report" not in names_seen:
        print("[FAIL] Missing required artifact name 'evaluation_report'.", file=sys.stderr)
        return 2

    required_artifacts = [x.strip() for x in args.required_artifact if x and x.strip()]
    if not required_artifacts:
        required_artifacts = [
            "training_log",
            "training_config",
            "model_artifact",
            "model_selection_report",
            "robustness_report",
            "seed_sensitivity_report",
            "evaluation_report",
            "prediction_trace",
            "external_validation_report",
        ]

    missing_required = [x for x in required_artifacts if x not in names_seen]
    if missing_required:
        print(f"[FAIL] Missing required artifact names in --artifact list: {missing_required}", file=sys.stderr)
        return 2

    if args.min_witness_count < 1:
        print("[FAIL] --min-witness-count must be >= 1.", file=sys.stderr)
        return 2

    parsed_witnesses: List[Tuple[str, Path, Path]] = []
    try:
        for raw in args.witness:
            authority_id, pub_path, priv_path = parse_witness(raw)
            ensure_file(pub_path, f"witness public key ({authority_id})")
            if not args.skip_sign:
                ensure_file(priv_path, f"witness private key ({authority_id})")
            parsed_witnesses.append((authority_id, pub_path, priv_path))
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2

    require_witness_quorum = bool(args.require_witness_quorum or len(parsed_witnesses) > 0)
    require_independent_witness_keys = bool(args.require_independent_witness_keys or require_witness_quorum)
    require_witness_independence_from_signing = bool(
        args.require_witness_independence_from_signing or require_witness_quorum
    )
    # Default to publication-grade independent authorities when corresponding attestations are enabled.
    require_independent_timestamp_authority = bool(
        args.require_independent_timestamp_authority or not args.skip_timestamp_trust
    )
    require_independent_execution_authority = bool(
        args.require_independent_execution_authority or not args.skip_execution_receipt
    )
    require_independent_log_authority = bool(
        args.require_independent_log_authority or not args.skip_execution_log_attestation
    )
    if require_witness_quorum and not parsed_witnesses:
        print("[FAIL] Witness quorum required but no --witness entries provided.", file=sys.stderr)
        return 2
    if require_witness_quorum and args.min_witness_count > len(parsed_witnesses):
        print(
            f"[FAIL] min-witness-count ({args.min_witness_count}) exceeds provided witnesses ({len(parsed_witnesses)}).",
            file=sys.stderr,
        )
        return 2
    if require_witness_quorum and require_independent_witness_keys:
        seen_authorities: set[str] = set()
        seen_witness_fps: set[str] = set()
        for authority_id, witness_pub, _ in parsed_witnesses:
            authority_norm = authority_id.strip()
            if authority_norm in seen_authorities:
                print(
                    f"[FAIL] Duplicate witness authority_id detected under independent-witness policy: {authority_norm}",
                    file=sys.stderr,
                )
                return 2
            seen_authorities.add(authority_norm)

            try:
                witness_fp = public_key_fingerprint_sha256(witness_pub).lower()
            except Exception as exc:
                print(f"[FAIL] Unable to compute witness public key fingerprint: {exc}", file=sys.stderr)
                return 2

            if witness_fp in seen_witness_fps:
                print(
                    "[FAIL] Duplicate witness public key detected under independent-witness policy.",
                    file=sys.stderr,
                )
                return 2
            seen_witness_fps.add(witness_fp)

    execution_log_artifact_name = (
        args.execution_log_artifact_name.strip()
        if isinstance(args.execution_log_artifact_name, str) and args.execution_log_artifact_name.strip()
        else "training_log"
    )
    if not args.skip_execution_log_attestation and execution_log_artifact_name not in names_seen:
        print(
            f"[FAIL] execution-log artifact '{execution_log_artifact_name}' not found in --artifact list.",
            file=sys.stderr,
        )
        return 2

    artifacts_by_name: Dict[str, Dict[str, Any]] = {str(item["name"]): item for item in artifacts}

    started_at = args.started_at_utc.strip() if isinstance(args.started_at_utc, str) and args.started_at_utc.strip() else iso_now_utc()
    finished_at = args.finished_at_utc.strip() if isinstance(args.finished_at_utc, str) and args.finished_at_utc.strip() else iso_now_utc()
    issued_at = args.issued_at_utc.strip() if isinstance(args.issued_at_utc, str) and args.issued_at_utc.strip() else finished_at
    command_text = args.command.strip()
    if not command_text:
        print("[FAIL] --command must be a non-empty string.", file=sys.stderr)
        return 2
    command_sha256 = sha256_text(command_text)
    executor_value = args.executor.strip() if isinstance(args.executor, str) and args.executor.strip() else default_executor()
    run_nonce = args.run_nonce.strip() if isinstance(args.run_nonce, str) and args.run_nonce.strip() else secrets.token_hex(16)
    if not RUN_NONCE_RE.fullmatch(run_nonce):
        print(
            "[FAIL] --run-nonce must be 16-128 hex characters when provided.",
            file=sys.stderr,
        )
        return 2
    run_nonce = run_nonce.lower()
    key_created_at = (
        args.key_created_at_utc.strip()
        if isinstance(args.key_created_at_utc, str) and args.key_created_at_utc.strip()
        else started_at
    )
    key_not_after = (
        args.key_not_after_utc.strip()
        if isinstance(args.key_not_after_utc, str) and args.key_not_after_utc.strip()
        else iso_after_days(key_created_at, args.max_signing_key_age_days)
    )

    if parse_iso_utc(started_at) is None:
        print("[FAIL] started-at-utc must be ISO-8601 timestamp.", file=sys.stderr)
        return 2
    if parse_iso_utc(finished_at) is None:
        print("[FAIL] finished-at-utc must be ISO-8601 timestamp.", file=sys.stderr)
        return 2
    if parse_iso_utc(issued_at) is None:
        print("[FAIL] issued-at-utc must be ISO-8601 timestamp.", file=sys.stderr)
        return 2
    if parse_iso_utc(key_created_at) is None:
        print("[FAIL] key-created-at-utc must be ISO-8601 timestamp.", file=sys.stderr)
        return 2
    if parse_iso_utc(key_not_after) is None:
        print("[FAIL] key-not-after-utc must be ISO-8601 timestamp.", file=sys.stderr)
        return 2

    payload_base = payload_out.parent
    payload_artifacts: List[Dict[str, Any]] = []
    for item in artifacts:
        payload_artifacts.append(
            {
                "name": item["name"],
                "path": resolve_for_output(payload_base, Path(item["abs_path"])),
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )

    payload = {
        "schema_version": "1.1",
        "study_id": args.study_id.strip(),
        "run_id": args.run_id.strip(),
        "command": command_text,
        "command_sha256": command_sha256,
        "executor": executor_value,
        "run_nonce": run_nonce,
        "git_commit": args.git_commit.strip() if isinstance(args.git_commit, str) and args.git_commit.strip() else None,
        "signing_key_id": args.key_id.strip(),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "exit_code": int(args.execution_exit_code),
        "artifacts": payload_artifacts,
    }
    if payload["git_commit"] is None:
        payload.pop("git_commit", None)

    write_json(payload_out, payload)
    payload_sha256 = sha256_file(payload_out)

    if args.skip_sign:
        if signature_out.exists():
            signature_out.unlink()
        print("[WARN] --skip-sign enabled. Payload signature was not generated.", file=sys.stderr)
    else:
        ensure_parent(signature_out)
        try:
            openssl_sign_file(private_key=private_key, input_file=payload_out, output_sig=signature_out)  # type: ignore[arg-type]
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2

    spec_base = spec_out.parent
    revocation_file = (
        Path(args.revocation_list_file).expanduser().resolve()
        if isinstance(args.revocation_list_file, str) and args.revocation_list_file.strip()
        else (spec_base / "key_revocations.json").resolve()
    )
    try:
        ensure_revocation_file(revocation_file)
    except Exception as exc:
        print(f"[FAIL] Unable to initialize revocation list: {exc}", file=sys.stderr)
        return 2

    try:
        signing_fingerprint = public_key_fingerprint_sha256(public_key)
    except Exception as exc:
        print(f"[FAIL] Unable to compute signing public key fingerprint: {exc}", file=sys.stderr)
        return 2

    timestamp_block = None
    timestamp_outputs: Dict[str, str] = {}
    if not args.skip_timestamp_trust:
        timestamp_pub = (
            Path(args.timestamp_public_key_file).expanduser().resolve()
            if isinstance(args.timestamp_public_key_file, str) and args.timestamp_public_key_file.strip()
            else public_key
        )
        timestamp_priv = (
            Path(args.timestamp_private_key_file).expanduser().resolve()
            if isinstance(args.timestamp_private_key_file, str) and args.timestamp_private_key_file.strip()
            else private_key
        )
        timestamp_record_out = (
            Path(args.timestamp_record_out).expanduser().resolve()
            if isinstance(args.timestamp_record_out, str) and args.timestamp_record_out.strip()
            else (payload_out.parent / "attestation_timestamp_record.json").resolve()
        )
        timestamp_sig_out = (
            Path(args.timestamp_signature_out).expanduser().resolve()
            if isinstance(args.timestamp_signature_out, str) and args.timestamp_signature_out.strip()
            else (payload_out.parent / "attestation_timestamp_record.sig").resolve()
        )

        try:
            ensure_file(timestamp_pub, "timestamp_public_key_file")
            if not args.skip_sign:
                if timestamp_priv is None:
                    raise ValueError("timestamp_private_key_file is required when signatures are enabled.")
                ensure_file(timestamp_priv, "timestamp_private_key_file")
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2

        timestamp_record = {
            "schema_version": "1.0",
            "authority_id": args.timestamp_authority_id.strip(),
            "payload_sha256": payload_sha256,
            "timestamp_utc": issued_at,
            "study_id": args.study_id.strip(),
            "run_id": args.run_id.strip(),
            "run_nonce": run_nonce,
        }
        write_json(timestamp_record_out, timestamp_record)

        if args.skip_sign:
            if timestamp_sig_out.exists():
                timestamp_sig_out.unlink()
        else:
            ensure_parent(timestamp_sig_out)
            try:
                openssl_sign_file(private_key=timestamp_priv, input_file=timestamp_record_out, output_sig=timestamp_sig_out)  # type: ignore[arg-type]
            except Exception as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 2

        try:
            timestamp_fp = public_key_fingerprint_sha256(timestamp_pub)
        except Exception as exc:
            print(f"[FAIL] Unable to compute timestamp public key fingerprint: {exc}", file=sys.stderr)
            return 2

        timestamp_block = {
            "mode": "signed_record",
            "authority_id": args.timestamp_authority_id.strip(),
            "record_file": resolve_for_output(spec_base, timestamp_record_out),
            "signature_file": resolve_for_output(spec_base, timestamp_sig_out),
            "public_key_file": resolve_for_output(spec_base, timestamp_pub),
            "public_key_fingerprint_sha256": timestamp_fp,
        }
        timestamp_outputs = {
            "record": str(timestamp_record_out),
            "signature": str(timestamp_sig_out),
        }

    transparency_block = None
    transparency_outputs: Dict[str, str] = {}
    if not args.skip_transparency_log:
        transparency_pub = (
            Path(args.transparency_public_key_file).expanduser().resolve()
            if isinstance(args.transparency_public_key_file, str) and args.transparency_public_key_file.strip()
            else public_key
        )
        transparency_priv = (
            Path(args.transparency_private_key_file).expanduser().resolve()
            if isinstance(args.transparency_private_key_file, str) and args.transparency_private_key_file.strip()
            else private_key
        )
        transparency_record_out = (
            Path(args.transparency_record_out).expanduser().resolve()
            if isinstance(args.transparency_record_out, str) and args.transparency_record_out.strip()
            else (payload_out.parent / "attestation_transparency_record.json").resolve()
        )
        transparency_sig_out = (
            Path(args.transparency_signature_out).expanduser().resolve()
            if isinstance(args.transparency_signature_out, str) and args.transparency_signature_out.strip()
            else (payload_out.parent / "attestation_transparency_record.sig").resolve()
        )

        try:
            ensure_file(transparency_pub, "transparency_public_key_file")
            if not args.skip_sign:
                if transparency_priv is None:
                    raise ValueError("transparency_private_key_file is required when signatures are enabled.")
                ensure_file(transparency_priv, "transparency_private_key_file")
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2

        entry_id = f"{args.run_id.strip()}-{issued_at.replace(':', '').replace('-', '')}"
        transparency_record = {
            "schema_version": "1.0",
            "log_id": args.transparency_log_id.strip(),
            "entry_id": entry_id,
            "payload_sha256": payload_sha256,
            "recorded_at_utc": issued_at,
            "study_id": args.study_id.strip(),
            "run_id": args.run_id.strip(),
            "run_nonce": run_nonce,
        }
        write_json(transparency_record_out, transparency_record)

        if args.skip_sign:
            if transparency_sig_out.exists():
                transparency_sig_out.unlink()
        else:
            ensure_parent(transparency_sig_out)
            try:
                openssl_sign_file(
                    private_key=transparency_priv, input_file=transparency_record_out, output_sig=transparency_sig_out  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 2

        try:
            transparency_fp = public_key_fingerprint_sha256(transparency_pub)
        except Exception as exc:
            print(f"[FAIL] Unable to compute transparency public key fingerprint: {exc}", file=sys.stderr)
            return 2

        transparency_block = {
            "log_id": args.transparency_log_id.strip(),
            "record_file": resolve_for_output(spec_base, transparency_record_out),
            "signature_file": resolve_for_output(spec_base, transparency_sig_out),
            "public_key_file": resolve_for_output(spec_base, transparency_pub),
            "public_key_fingerprint_sha256": transparency_fp,
        }
        transparency_outputs = {
            "record": str(transparency_record_out),
            "signature": str(transparency_sig_out),
        }

    execution_receipt_block = None
    execution_receipt_outputs: Dict[str, str] = {}
    if not args.skip_execution_receipt:
        execution_pub = (
            Path(args.execution_public_key_file).expanduser().resolve()
            if isinstance(args.execution_public_key_file, str) and args.execution_public_key_file.strip()
            else public_key
        )
        execution_priv = (
            Path(args.execution_private_key_file).expanduser().resolve()
            if isinstance(args.execution_private_key_file, str) and args.execution_private_key_file.strip()
            else private_key
        )
        execution_record_out = (
            Path(args.execution_receipt_record_out).expanduser().resolve()
            if isinstance(args.execution_receipt_record_out, str) and args.execution_receipt_record_out.strip()
            else (payload_out.parent / "attestation_execution_receipt_record.json").resolve()
        )
        execution_sig_out = (
            Path(args.execution_receipt_signature_out).expanduser().resolve()
            if isinstance(args.execution_receipt_signature_out, str) and args.execution_receipt_signature_out.strip()
            else (payload_out.parent / "attestation_execution_receipt_record.sig").resolve()
        )

        try:
            ensure_file(execution_pub, "execution_public_key_file")
            if not args.skip_sign:
                if execution_priv is None:
                    raise ValueError("execution_private_key_file is required when signatures are enabled.")
                ensure_file(execution_priv, "execution_private_key_file")
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2

        execution_receipt_record = {
            "schema_version": "1.0",
            "authority_id": args.execution_authority_id.strip(),
            "payload_sha256": payload_sha256,
            "study_id": args.study_id.strip(),
            "run_id": args.run_id.strip(),
            "command": command_text,
            "command_sha256": command_sha256,
            "executor": payload["executor"],
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "issued_at_utc": issued_at,
            "exit_code": int(args.execution_exit_code),
            "run_nonce": run_nonce,
        }
        write_json(execution_record_out, execution_receipt_record)

        if args.skip_sign:
            if execution_sig_out.exists():
                execution_sig_out.unlink()
        else:
            ensure_parent(execution_sig_out)
            try:
                openssl_sign_file(
                    private_key=execution_priv, input_file=execution_record_out, output_sig=execution_sig_out  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 2

        try:
            execution_fp = public_key_fingerprint_sha256(execution_pub)
        except Exception as exc:
            print(f"[FAIL] Unable to compute execution receipt public key fingerprint: {exc}", file=sys.stderr)
            return 2

        execution_receipt_block = {
            "authority_id": args.execution_authority_id.strip(),
            "record_file": resolve_for_output(spec_base, execution_record_out),
            "signature_file": resolve_for_output(spec_base, execution_sig_out),
            "public_key_file": resolve_for_output(spec_base, execution_pub),
            "public_key_fingerprint_sha256": execution_fp,
            "enforce_exit_code_zero": True,
        }
        execution_receipt_outputs = {
            "record": str(execution_record_out),
            "signature": str(execution_sig_out),
        }

    execution_log_block = None
    execution_log_outputs: Dict[str, str] = {}
    if not args.skip_execution_log_attestation:
        log_artifact = artifacts_by_name.get(execution_log_artifact_name)
        if log_artifact is None:
            print(
                f"[FAIL] execution-log artifact '{execution_log_artifact_name}' not found in signed artifacts.",
                file=sys.stderr,
            )
            return 2

        execution_log_pub = (
            Path(args.execution_log_public_key_file).expanduser().resolve()
            if isinstance(args.execution_log_public_key_file, str) and args.execution_log_public_key_file.strip()
            else (
                Path(args.execution_public_key_file).expanduser().resolve()
                if isinstance(args.execution_public_key_file, str) and args.execution_public_key_file.strip()
                else public_key
            )
        )
        execution_log_priv = (
            Path(args.execution_log_private_key_file).expanduser().resolve()
            if isinstance(args.execution_log_private_key_file, str) and args.execution_log_private_key_file.strip()
            else (
                Path(args.execution_private_key_file).expanduser().resolve()
                if isinstance(args.execution_private_key_file, str) and args.execution_private_key_file.strip()
                else private_key
            )
        )
        execution_log_record_out = (
            Path(args.execution_log_record_out).expanduser().resolve()
            if isinstance(args.execution_log_record_out, str) and args.execution_log_record_out.strip()
            else (payload_out.parent / "attestation_execution_log_record.json").resolve()
        )
        execution_log_sig_out = (
            Path(args.execution_log_signature_out).expanduser().resolve()
            if isinstance(args.execution_log_signature_out, str) and args.execution_log_signature_out.strip()
            else (payload_out.parent / "attestation_execution_log_record.sig").resolve()
        )

        try:
            ensure_file(execution_log_pub, "execution_log_public_key_file")
            if not args.skip_sign:
                if execution_log_priv is None:
                    raise ValueError("execution_log_private_key_file is required when signatures are enabled.")
                ensure_file(execution_log_priv, "execution_log_private_key_file")
        except Exception as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 2

        log_path = Path(str(log_artifact["abs_path"]))
        line_count, first_line_sha256, last_line_sha256 = log_boundary_hashes(log_path)
        if line_count <= 0:
            print(
                f"[FAIL] execution-log artifact '{execution_log_artifact_name}' is empty; cannot attest training execution.",
                file=sys.stderr,
            )
            return 2
        execution_log_record = {
            "schema_version": "1.0",
            "authority_id": args.execution_log_authority_id.strip(),
            "payload_sha256": payload_sha256,
            "study_id": args.study_id.strip(),
            "run_id": args.run_id.strip(),
            "artifact_name": execution_log_artifact_name,
            "artifact_path": str(log_path),
            "artifact_sha256": str(log_artifact["sha256"]),
            "training_config_sha256": str(artifacts_by_name.get("training_config", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("training_config"), dict)
            else None,
            "model_artifact_sha256": str(artifacts_by_name.get("model_artifact", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("model_artifact"), dict)
            else None,
            "model_selection_report_sha256": str(artifacts_by_name.get("model_selection_report", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("model_selection_report"), dict)
            else None,
            "robustness_report_sha256": str(artifacts_by_name.get("robustness_report", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("robustness_report"), dict)
            else None,
            "seed_sensitivity_report_sha256": str(artifacts_by_name.get("seed_sensitivity_report", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("seed_sensitivity_report"), dict)
            else None,
            "evaluation_report_sha256": str(artifacts_by_name.get("evaluation_report", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("evaluation_report"), dict)
            else None,
            "prediction_trace_sha256": str(artifacts_by_name.get("prediction_trace", {}).get("sha256"))
            if isinstance(artifacts_by_name.get("prediction_trace"), dict)
            else None,
            "external_validation_report_sha256": str(
                artifacts_by_name.get("external_validation_report", {}).get("sha256")
            )
            if isinstance(artifacts_by_name.get("external_validation_report"), dict)
            else None,
            "line_count": line_count,
            "first_line_sha256": first_line_sha256,
            "last_line_sha256": last_line_sha256,
            "command": command_text,
            "command_sha256": command_sha256,
            "executor": executor_value,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "issued_at_utc": issued_at,
            "run_nonce": run_nonce,
        }
        write_json(execution_log_record_out, execution_log_record)

        if args.skip_sign:
            if execution_log_sig_out.exists():
                execution_log_sig_out.unlink()
        else:
            ensure_parent(execution_log_sig_out)
            try:
                openssl_sign_file(
                    private_key=execution_log_priv, input_file=execution_log_record_out, output_sig=execution_log_sig_out  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"[FAIL] {exc}", file=sys.stderr)
                return 2

        try:
            execution_log_fp = public_key_fingerprint_sha256(execution_log_pub)
        except Exception as exc:
            print(f"[FAIL] Unable to compute execution-log public key fingerprint: {exc}", file=sys.stderr)
            return 2

        execution_log_block = {
            "authority_id": args.execution_log_authority_id.strip(),
            "artifact_name": execution_log_artifact_name,
            "record_file": resolve_for_output(spec_base, execution_log_record_out),
            "signature_file": resolve_for_output(spec_base, execution_log_sig_out),
            "public_key_file": resolve_for_output(spec_base, execution_log_pub),
            "public_key_fingerprint_sha256": execution_log_fp,
        }
        execution_log_outputs = {
            "record": str(execution_log_record_out),
            "signature": str(execution_log_sig_out),
        }

    witness_quorum_block = None
    witness_outputs: List[Dict[str, str]] = []
    if require_witness_quorum:
        witness_records: List[Dict[str, str]] = []
        for idx, (authority_id, witness_pub, witness_priv) in enumerate(parsed_witnesses, start=1):
            record_out = (payload_out.parent / f"attestation_witness_record_{idx}.json").resolve()
            sig_out = (payload_out.parent / f"attestation_witness_record_{idx}.sig").resolve()
            witness_record = {
                "schema_version": "1.0",
                "authority_id": authority_id,
                "payload_sha256": payload_sha256,
                "study_id": args.study_id.strip(),
                "run_id": args.run_id.strip(),
                "attested_at_utc": issued_at,
                "run_nonce": run_nonce,
                "statement": "witnessed execution payload hash",
            }
            write_json(record_out, witness_record)

            if args.skip_sign:
                if sig_out.exists():
                    sig_out.unlink()
            else:
                ensure_parent(sig_out)
                try:
                    openssl_sign_file(private_key=witness_priv, input_file=record_out, output_sig=sig_out)
                except Exception as exc:
                    print(f"[FAIL] {exc}", file=sys.stderr)
                    return 2

            try:
                witness_fp = public_key_fingerprint_sha256(witness_pub)
            except Exception as exc:
                print(f"[FAIL] Unable to compute witness public key fingerprint: {exc}", file=sys.stderr)
                return 2

            witness_records.append(
                {
                    "authority_id": authority_id,
                    "record_file": resolve_for_output(spec_base, record_out),
                    "signature_file": resolve_for_output(spec_base, sig_out),
                    "public_key_file": resolve_for_output(spec_base, witness_pub),
                    "public_key_fingerprint_sha256": witness_fp,
                }
            )
            witness_outputs.append({"authority_id": authority_id, "record": str(record_out), "signature": str(sig_out)})

        witness_quorum_block = {
            "min_witness_count": int(args.min_witness_count),
            "records": witness_records,
        }

    # Publication-grade default: require all authority roles to be distinct.
    role_entries: List[Tuple[str, str, str]] = []

    def add_role_entry(role: str, authority_id: str, fingerprint: str) -> None:
        role_entries.append((role, authority_id.strip().lower(), fingerprint.strip().lower()))

    if isinstance(timestamp_block, dict):
        add_role_entry(
            "timestamp_trust",
            str(timestamp_block.get("authority_id", "")),
            str(timestamp_block.get("public_key_fingerprint_sha256", "")),
        )
    if isinstance(transparency_block, dict):
        add_role_entry(
            "transparency_log",
            str(transparency_block.get("log_id", "")),
            str(transparency_block.get("public_key_fingerprint_sha256", "")),
        )
    if isinstance(execution_receipt_block, dict):
        add_role_entry(
            "execution_receipt",
            str(execution_receipt_block.get("authority_id", "")),
            str(execution_receipt_block.get("public_key_fingerprint_sha256", "")),
        )
    if isinstance(execution_log_block, dict):
        add_role_entry(
            "execution_log_attestation",
            str(execution_log_block.get("authority_id", "")),
            str(execution_log_block.get("public_key_fingerprint_sha256", "")),
        )
    if isinstance(witness_quorum_block, dict):
        for idx, witness in enumerate(witness_quorum_block.get("records", [])):
            if isinstance(witness, dict):
                add_role_entry(
                    f"witness[{idx}]",
                    str(witness.get("authority_id", "")),
                    str(witness.get("public_key_fingerprint_sha256", "")),
                )

    empty_roles = [role for role, authority, fp in role_entries if not authority or not fp]
    if empty_roles:
        print(
            f"[FAIL] Missing authority_id or public_key_fingerprint_sha256 for roles: {sorted(set(empty_roles))}",
            file=sys.stderr,
        )
        return 2

    seen_authorities: Dict[str, List[str]] = {}
    seen_fingerprints: Dict[str, List[str]] = {}
    for role, authority, fp in role_entries:
        seen_authorities.setdefault(authority, []).append(role)
        seen_fingerprints.setdefault(fp, []).append(role)
    duplicate_authorities = {k: v for k, v in seen_authorities.items() if len(v) > 1}
    duplicate_fingerprints = {k: v for k, v in seen_fingerprints.items() if len(v) > 1}
    if duplicate_authorities or duplicate_fingerprints:
        print(
            "[FAIL] Authority roles are not distinct across timestamp/transparency/execution/log/witness attestations.",
            file=sys.stderr,
        )
        if duplicate_authorities:
            print(f"[DETAIL] duplicate authority IDs: {duplicate_authorities}", file=sys.stderr)
        if duplicate_fingerprints:
            print(f"[DETAIL] duplicate key fingerprints: {duplicate_fingerprints}", file=sys.stderr)
        return 2

    assurance_policy = {
        "min_signing_key_bits": int(args.min_signing_key_bits),
        "max_signing_key_age_days": int(args.max_signing_key_age_days),
        "require_revocation_list": True,
        "require_timestamp_trust": not args.skip_timestamp_trust,
        "require_transparency_log": not args.skip_transparency_log,
        "require_transparency_log_signature": not args.skip_transparency_log,
        "require_execution_receipt": not args.skip_execution_receipt,
        "require_execution_log_attestation": not args.skip_execution_log_attestation,
        "require_independent_timestamp_authority": bool(require_independent_timestamp_authority),
        "require_independent_execution_authority": bool(require_independent_execution_authority),
        "require_independent_log_authority": bool(require_independent_log_authority),
        "require_distinct_authority_roles": True,
        "require_witness_quorum": bool(require_witness_quorum),
        "min_witness_count": int(args.min_witness_count) if require_witness_quorum else 0,
        "require_independent_witness_keys": bool(require_independent_witness_keys),
        "require_witness_independence_from_signing": bool(require_witness_independence_from_signing),
    }

    spec: Dict[str, Any] = {
        "schema_version": "1.2",
        "study_id": args.study_id.strip(),
        "run_id": args.run_id.strip(),
        "issued_at_utc": issued_at,
        "required_artifact_names": required_artifacts,
        "signed_payload_sha256": payload_sha256,
        "assurance_policy": assurance_policy,
        "signing": {
            "method": "openssl-dgst-sha256",
            "key_id": args.key_id.strip(),
            "key_created_at_utc": key_created_at,
            "key_not_after_utc": key_not_after,
            "signed_payload_file": resolve_for_output(spec_base, payload_out),
            "signature_file": resolve_for_output(spec_base, signature_out),
            "public_key_file": resolve_for_output(spec_base, public_key),
            "public_key_fingerprint_sha256": signing_fingerprint,
            "revocation_list_file": resolve_for_output(spec_base, revocation_file),
        },
    }

    if timestamp_block is not None:
        spec["timestamp_trust"] = timestamp_block
    if transparency_block is not None:
        spec["transparency_log"] = transparency_block
    if execution_receipt_block is not None:
        spec["execution_receipt"] = execution_receipt_block
    if execution_log_block is not None:
        spec["execution_log_attestation"] = execution_log_block
    if witness_quorum_block is not None:
        spec["witness_quorum"] = witness_quorum_block

    write_json(spec_out, spec)

    print(f"Payload: {payload_out}")
    print(f"Payload signature: {signature_out}")
    print(f"Spec: {spec_out}")
    print(f"Revocation list: {revocation_file}")
    if timestamp_outputs:
        print(f"Timestamp record: {timestamp_outputs['record']}")
        print(f"Timestamp signature: {timestamp_outputs['signature']}")
    if transparency_outputs:
        print(f"Transparency record: {transparency_outputs['record']}")
        print(f"Transparency signature: {transparency_outputs['signature']}")
    if execution_receipt_outputs:
        print(f"Execution receipt record: {execution_receipt_outputs['record']}")
        print(f"Execution receipt signature: {execution_receipt_outputs['signature']}")
    if execution_log_outputs:
        print(f"Execution log record: {execution_log_outputs['record']}")
        print(f"Execution log signature: {execution_log_outputs['signature']}")
    for witness in witness_outputs:
        print(f"Witness ({witness['authority_id']}) record: {witness['record']}")
        print(f"Witness ({witness['authority_id']}) signature: {witness['signature']}")
    print(f"Artifacts hashed: {len(payload_artifacts)}")
    print(f"Required artifact names: {required_artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
