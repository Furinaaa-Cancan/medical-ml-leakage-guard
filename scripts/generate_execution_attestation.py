#!/usr/bin/env python3
"""
Generate and sign execution attestation artifacts for publication-grade review.

Outputs:
1. signed payload JSON (contains artifact hashes)
2. detached signature file
3. attestation spec JSON consumed by execution_attestation_gate.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create signed execution attestation spec + payload.")
    parser.add_argument("--study-id", required=True, help="Study identifier.")
    parser.add_argument("--run-id", required=True, help="Training run identifier.")
    parser.add_argument("--payload-out", required=True, help="Output JSON path for signed payload.")
    parser.add_argument("--spec-out", required=True, help="Output JSON path for attestation spec.")
    parser.add_argument("--signature-out", required=True, help="Output path for detached signature.")
    parser.add_argument("--public-key-file", required=True, help="Public key PEM used for verification.")
    parser.add_argument("--private-key-file", help="Private key PEM used for signing.")
    parser.add_argument("--skip-sign", action="store_true", help="Skip signature creation (bootstrap scaffold only).")
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
    parser.add_argument("--git-commit", help="Git commit hash for training code.")
    parser.add_argument("--started-at-utc", help="Run start time ISO-8601 UTC.")
    parser.add_argument("--finished-at-utc", help="Run end time ISO-8601 UTC. Defaults to now.")
    parser.add_argument("--issued-at-utc", help="Attestation issue time ISO-8601 UTC. Defaults to finished-at-utc.")
    return parser.parse_args()


def iso_now_utc() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


def default_executor() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown-user"
    host = socket.gethostname() or "unknown-host"
    return f"{user}@{host}"


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file: {path}")


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
        required_artifacts = ["training_log", "training_config", "model_artifact", "evaluation_report"]

    missing_required = [x for x in required_artifacts if x not in names_seen]
    if missing_required:
        print(
            f"[FAIL] Missing required artifact names in --artifact list: {missing_required}",
            file=sys.stderr,
        )
        return 2

    payload_out.parent.mkdir(parents=True, exist_ok=True)
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    signature_out.parent.mkdir(parents=True, exist_ok=True)

    started_at = args.started_at_utc.strip() if isinstance(args.started_at_utc, str) and args.started_at_utc.strip() else iso_now_utc()
    finished_at = args.finished_at_utc.strip() if isinstance(args.finished_at_utc, str) and args.finished_at_utc.strip() else iso_now_utc()
    issued_at = args.issued_at_utc.strip() if isinstance(args.issued_at_utc, str) and args.issued_at_utc.strip() else finished_at

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
        "schema_version": "1.0",
        "study_id": args.study_id.strip(),
        "run_id": args.run_id.strip(),
        "command": args.command.strip(),
        "executor": args.executor.strip() if isinstance(args.executor, str) and args.executor.strip() else default_executor(),
        "git_commit": args.git_commit.strip() if isinstance(args.git_commit, str) and args.git_commit.strip() else None,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "artifacts": payload_artifacts,
    }
    if payload["git_commit"] is None:
        payload.pop("git_commit", None)

    with payload_out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)

    if args.skip_sign:
        if signature_out.exists():
            signature_out.unlink()
        print("[WARN] --skip-sign enabled. Signature file was not generated.", file=sys.stderr)
    else:
        cmd = [
            "openssl",
            "dgst",
            "-sha256",
            "-sign",
            str(private_key),
            "-out",
            str(signature_out),
            str(payload_out),
        ]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True)
        except FileNotFoundError:
            print("[FAIL] openssl command not found; required for detached signature creation.", file=sys.stderr)
            return 2
        if proc.returncode != 0:
            print("[FAIL] Signature creation failed.", file=sys.stderr)
            if proc.stdout:
                print(proc.stdout, file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            return 2

    spec_base = spec_out.parent
    spec = {
        "schema_version": "1.0",
        "study_id": args.study_id.strip(),
        "run_id": args.run_id.strip(),
        "issued_at_utc": issued_at,
        "required_artifact_names": required_artifacts,
        "signed_payload_sha256": sha256_file(payload_out),
        "signing": {
            "method": "openssl-dgst-sha256",
            "signed_payload_file": resolve_for_output(spec_base, payload_out),
            "signature_file": resolve_for_output(spec_base, signature_out),
            "public_key_file": resolve_for_output(spec_base, public_key),
        },
    }

    with spec_out.open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=True, indent=2)

    print(f"Payload: {payload_out}")
    print(f"Signature: {signature_out}")
    print(f"Spec: {spec_out}")
    print(f"Artifacts hashed: {len(payload_artifacts)}")
    print(f"Required artifact names: {required_artifacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
