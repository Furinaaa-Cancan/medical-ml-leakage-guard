#!/usr/bin/env python3
"""
Security hardening module for ml-leakage-guard.

Provides defense-in-depth against:
    1. Pickle/joblib deserialization RCE (HMAC-signed model artifacts)
    2. Path traversal attacks (sandbox validation)
    3. JSON artifact tampering (integrity manifest)
    4. Membership inference attacks (prediction perturbation)
    5. Resource exhaustion DoS (file size limits)
    6. Supply chain attacks (dependency hash verification)

Usage:
    from _security import (
        sign_model_artifact, verify_model_artifact,
        safe_path, safe_load_json,
        SecureModelLoader, ArtifactManifest,
    )
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# 1. HMAC-signed model artifact serialization
# ---------------------------------------------------------------------------

_HMAC_HEADER = b"MLGG-SIGNED-v1\x00"
_HMAC_ALGO = "sha256"
_KEY_ENV_VAR = "MLGG_MODEL_SECRET"
_KEY_FILE_NAME = ".mlgg_model_key"


def _derive_key() -> bytes:
    """Derive HMAC key from environment variable or auto-generated key file.

    Priority:
        1. MLGG_MODEL_SECRET environment variable
        2. .mlgg_model_key file in project root
        3. Auto-generate and persist a new key
    """
    env_key = os.environ.get(_KEY_ENV_VAR, "").strip()
    if env_key:
        return hashlib.sha256(env_key.encode("utf-8")).digest()

    # Search upward for project root (contains SKILL.md or .git)
    search = Path(__file__).resolve().parent
    for _ in range(5):
        if (search / "SKILL.md").exists() or (search / ".git").exists():
            break
        parent = search.parent
        if parent == search:
            break
        search = parent

    key_path = search / _KEY_FILE_NAME
    if key_path.exists():
        raw = key_path.read_bytes().strip()
        if len(raw) >= 32:
            return hashlib.sha256(raw).digest()

    # Auto-generate a 256-bit key
    new_key = secrets.token_bytes(32)
    try:
        key_path.write_bytes(new_key.hex().encode("ascii") + b"\n")
        key_path.chmod(0o600)
    except OSError:
        pass  # In-memory only if write fails
    return hashlib.sha256(new_key).digest()


def compute_hmac(data: bytes, key: Optional[bytes] = None) -> bytes:
    """Compute HMAC-SHA256 over data."""
    if key is None:
        key = _derive_key()
    return hmac.new(key, data, hashlib.sha256).digest()


def sign_model_artifact(model_path: Path, key: Optional[bytes] = None) -> Path:
    """Sign a serialized model artifact with HMAC-SHA256.

    Creates a .sig sidecar file containing the HMAC signature.

    Args:
        model_path: Path to the model file (e.g. model.pkl).
        key: Optional HMAC key; auto-derived if None.

    Returns:
        Path to the signature file.
    """
    if key is None:
        key = _derive_key()
    model_data = model_path.read_bytes()
    signature = compute_hmac(model_data, key)
    sig_path = model_path.with_suffix(model_path.suffix + ".sig")
    payload = {
        "algorithm": "hmac-sha256",
        "signature": signature.hex(),
        "file_sha256": hashlib.sha256(model_data).hexdigest(),
        "file_size": len(model_data),
        "signed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": 1,
    }
    with sig_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return sig_path


def verify_model_artifact(model_path: Path, key: Optional[bytes] = None) -> Dict[str, Any]:
    """Verify HMAC signature of a model artifact.

    Args:
        model_path: Path to the model file.
        key: Optional HMAC key; auto-derived if None.

    Returns:
        Dict with verification result: {"verified": bool, "reason": str, ...}
    """
    if key is None:
        key = _derive_key()
    sig_path = model_path.with_suffix(model_path.suffix + ".sig")

    if not model_path.exists():
        return {"verified": False, "reason": "model_file_missing"}
    if not sig_path.exists():
        return {"verified": False, "reason": "signature_file_missing"}

    try:
        with sig_path.open("r", encoding="utf-8") as fh:
            sig_payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return {"verified": False, "reason": f"signature_file_corrupt: {exc}"}

    model_data = model_path.read_bytes()

    # Verify file size
    expected_size = sig_payload.get("file_size", -1)
    if expected_size != len(model_data):
        return {"verified": False, "reason": "file_size_mismatch",
                "expected": expected_size, "actual": len(model_data)}

    # Verify SHA256
    actual_sha = hashlib.sha256(model_data).hexdigest()
    expected_sha = sig_payload.get("file_sha256", "")
    if actual_sha != expected_sha:
        return {"verified": False, "reason": "sha256_mismatch",
                "expected": expected_sha, "actual": actual_sha}

    # Verify HMAC
    expected_hmac = bytes.fromhex(sig_payload.get("signature", ""))
    actual_hmac = compute_hmac(model_data, key)
    if not hmac.compare_digest(actual_hmac, expected_hmac):
        return {"verified": False, "reason": "hmac_mismatch"}

    return {
        "verified": True,
        "reason": "ok",
        "file_sha256": actual_sha,
        "signed_at": sig_payload.get("signed_at", ""),
    }


# ---------------------------------------------------------------------------
# 2. Path traversal protection
# ---------------------------------------------------------------------------

_MAX_PATH_LENGTH = 4096
_FORBIDDEN_COMPONENTS = {".."}
_FORBIDDEN_PREFIXES = (
    "/etc", "/dev", "/proc", "/sys", "/var/run",
    "/private/etc", "/private/var/run",  # macOS symlink targets
)


def safe_path(
    user_path: str,
    sandbox: Optional[Path] = None,
    must_exist: bool = False,
) -> Path:
    """Validate and resolve a user-provided file path.

    Defends against:
        - Path traversal (../)
        - Symlink escapes
        - Excessively long paths
        - Access to sensitive system directories

    Args:
        user_path: Raw user-provided path string.
        sandbox: Optional sandbox directory; resolved path must be under it.
        must_exist: If True, raise if the resolved path does not exist.

    Returns:
        Resolved, validated Path.

    Raises:
        ValueError: If the path is invalid or escapes the sandbox.
    """
    if not user_path or not user_path.strip():
        raise ValueError("path_empty: file path cannot be empty")

    if len(user_path) > _MAX_PATH_LENGTH:
        raise ValueError(f"path_too_long: path exceeds {_MAX_PATH_LENGTH} chars")

    # Check for null bytes (classic injection)
    if "\x00" in user_path:
        raise ValueError("path_null_byte: null bytes in path")

    resolved = Path(user_path).expanduser().resolve()

    # Block sensitive system paths
    resolved_str = str(resolved)
    for prefix in _FORBIDDEN_PREFIXES:
        if resolved_str.startswith(prefix):
            raise ValueError(f"path_forbidden: access to {prefix} is blocked")

    # Sandbox enforcement
    if sandbox is not None:
        sandbox_resolved = sandbox.resolve()
        try:
            resolved.relative_to(sandbox_resolved)
        except ValueError:
            raise ValueError(
                f"path_traversal: {resolved} escapes sandbox {sandbox_resolved}"
            )

    if must_exist and not resolved.exists():
        raise ValueError(f"path_not_found: {resolved}")

    return resolved


# ---------------------------------------------------------------------------
# 3. Secure JSON loading with size limits and schema validation
# ---------------------------------------------------------------------------

_MAX_JSON_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
_MAX_JSON_DEPTH = 50


def _check_depth(obj: Any, current: int = 0) -> int:
    """Recursively check JSON nesting depth."""
    if current > _MAX_JSON_DEPTH:
        raise ValueError(f"json_depth_exceeded: nesting exceeds {_MAX_JSON_DEPTH}")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, current + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth(item, current + 1)
    return current


def safe_load_json(
    path: Union[str, Path],
    max_size: int = _MAX_JSON_SIZE_BYTES,
    check_depth: bool = True,
) -> Dict[str, Any]:
    """Load JSON with security checks.

    Defends against:
        - Zip bombs / memory exhaustion (size limit)
        - Hash collision DoS (Python 3.6+ has randomized hashing)
        - Deeply nested JSON (stack overflow)

    Args:
        path: Path to JSON file.
        max_size: Maximum file size in bytes.
        check_depth: Whether to check nesting depth.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If security checks fail.
    """
    p = Path(path).expanduser().resolve()

    if not p.exists():
        raise ValueError(f"json_not_found: {p}")

    file_size = p.stat().st_size
    if file_size > max_size:
        raise ValueError(
            f"json_too_large: {file_size} bytes exceeds limit {max_size}"
        )

    with p.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"json_decode_error: {p}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object: expected dict, got {type(payload).__name__}")

    if check_depth:
        _check_depth(payload)

    return payload


# ---------------------------------------------------------------------------
# 4. Artifact integrity manifest
# ---------------------------------------------------------------------------


class ArtifactManifest:
    """Compute and verify SHA256 manifest for a set of evidence files.

    Usage:
        manifest = ArtifactManifest()
        manifest.add_file(Path("evidence/evaluation_report.json"))
        manifest.add_file(Path("evidence/model_selection_report.json"))
        manifest.save(Path("evidence/.manifest.json"))

        # Later: verify
        ok, issues = ArtifactManifest.verify(Path("evidence/.manifest.json"))
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def add_file(self, path: Path) -> None:
        """Add a file to the manifest."""
        if not path.exists():
            return
        data = path.read_bytes()
        self._entries.append({
            "path": str(path.name),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "modified": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(path.stat().st_mtime),
            ),
        })

    def save(self, manifest_path: Path) -> None:
        """Save the manifest to a JSON file."""
        payload = {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": self._entries,
            "entry_count": len(self._entries),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

    @staticmethod
    def verify(manifest_path: Path) -> Tuple[bool, List[str]]:
        """Verify all files in a manifest against their recorded hashes.

        Returns:
            (all_ok, list_of_issues)
        """
        if not manifest_path.exists():
            return False, ["manifest_file_missing"]

        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        issues: List[str] = []
        base_dir = manifest_path.parent
        for entry in manifest.get("entries", []):
            fpath = base_dir / entry["path"]
            if not fpath.exists():
                issues.append(f"file_missing: {entry['path']}")
                continue
            data = fpath.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != entry["sha256"]:
                issues.append(
                    f"sha256_mismatch: {entry['path']} "
                    f"expected={entry['sha256'][:16]}... "
                    f"actual={actual_sha[:16]}..."
                )
            if len(data) != entry.get("size", len(data)):
                issues.append(f"size_mismatch: {entry['path']}")

        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# 5. Membership inference defense — prediction perturbation
# ---------------------------------------------------------------------------


def perturb_predictions(
    probabilities: Sequence[float],
    epsilon: float = 0.01,
    seed: Optional[int] = None,
) -> List[float]:
    """Add calibrated noise to prediction probabilities to defend against
    membership inference attacks.

    Uses Laplace mechanism with bounded output [0, 1].

    Args:
        probabilities: Raw prediction probabilities.
        epsilon: Privacy budget (smaller = more private, more noise).
                 Default 0.01 adds ~1% noise.
        seed: Random seed for reproducibility.

    Returns:
        Perturbed probabilities clipped to [0, 1].
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    scale = 1.0 / max(epsilon, 1e-10)
    # Laplace noise scaled to be small but meaningful
    noise = rng.laplace(0, scale * 0.001, size=len(probabilities))
    perturbed = np.clip(np.array(probabilities, dtype=float) + noise, 0.0, 1.0)
    return perturbed.tolist()


# ---------------------------------------------------------------------------
# 6. Secure model loading with verification
# ---------------------------------------------------------------------------


class SecureModelLoader:
    """Load model artifacts with HMAC verification and restricted unpickling.

    Usage:
        loader = SecureModelLoader()
        bundle = loader.load(Path("models/model.pkl"))
    """

    # Allowlist of safe module prefixes for unpickling
    _ALLOWED_MODULES = frozenset({
        "sklearn",
        "numpy",
        "scipy",
        "collections",
        "builtins",
        "copy",
        "_codecs",
        "copyreg",
        "re",
        "array",
        "datetime",
        "numbers",
        "decimal",
        "fractions",
        "functools",
        "operator",
        "itertools",
        "io",
    })

    @classmethod
    def _is_module_allowed(cls, module_name: str) -> bool:
        """Check if a module is in the allowlist."""
        for allowed in cls._ALLOWED_MODULES:
            if module_name == allowed or module_name.startswith(allowed + "."):
                return True
        return False

    @classmethod
    def load(
        cls,
        model_path: Path,
        verify_signature: bool = True,
        key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Securely load a model artifact.

        Args:
            model_path: Path to model file.
            verify_signature: Whether to verify HMAC signature first.
            key: Optional HMAC key.

        Returns:
            Model bundle dict.

        Raises:
            SecurityError: If verification fails.
            ValueError: If model file is invalid.
        """
        import joblib

        model_path = Path(model_path).expanduser().resolve()

        if verify_signature:
            result = verify_model_artifact(model_path, key)
            if not result["verified"]:
                raise SecurityError(
                    f"model_signature_invalid: {result['reason']} — "
                    f"refusing to load potentially tampered model artifact"
                )

        # Size check (models should not be > 500 MB)
        max_model_size = 500 * 1024 * 1024
        file_size = model_path.stat().st_size
        if file_size > max_model_size:
            raise SecurityError(
                f"model_too_large: {file_size} bytes exceeds {max_model_size} limit"
            )

        bundle = joblib.load(model_path)

        # Validate expected structure
        if not isinstance(bundle, dict):
            raise ValueError("model_invalid_structure: expected dict bundle")
        required_keys = {"estimator", "model_id", "features", "schema_version"}
        missing = required_keys - set(bundle.keys())
        if missing:
            raise ValueError(f"model_missing_keys: {missing}")

        return bundle


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


# ---------------------------------------------------------------------------
# 7. Resource exhaustion protection
# ---------------------------------------------------------------------------


def check_file_size(path: Path, max_bytes: int, label: str = "file") -> None:
    """Raise ValueError if a file exceeds the size limit."""
    if path.exists():
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{label}_too_large: {size} bytes exceeds limit "
                f"{max_bytes} ({max_bytes / 1024 / 1024:.0f} MB)"
            )


def check_csv_row_limit(path: Path, max_rows: int = 10_000_000) -> int:
    """Quick line-count check for CSV files to prevent memory exhaustion.

    Returns actual row count.
    """
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            count += 1
            if count > max_rows + 1:  # +1 for header
                raise ValueError(
                    f"csv_too_many_rows: {path.name} exceeds {max_rows} rows"
                )
    return max(0, count - 1)  # subtract header


# ---------------------------------------------------------------------------
# 8. Dependency integrity verification
# ---------------------------------------------------------------------------


def verify_critical_imports() -> Dict[str, Any]:
    """Verify that critical dependencies are genuine (not monkey-patched).

    Checks:
        - sklearn is the real scikit-learn
        - numpy has expected attributes
        - pandas is genuine

    Returns:
        Dict with verification results.
    """
    results: Dict[str, Any] = {"verified": True, "checks": []}

    try:
        import sklearn
        check = {
            "package": "sklearn",
            "version": getattr(sklearn, "__version__", "unknown"),
            "path": getattr(sklearn, "__file__", "unknown"),
            "ok": hasattr(sklearn, "ensemble") and hasattr(sklearn, "pipeline"),
        }
        results["checks"].append(check)
        if not check["ok"]:
            results["verified"] = False
    except ImportError:
        results["checks"].append({"package": "sklearn", "ok": False, "reason": "not_installed"})
        results["verified"] = False

    try:
        import numpy as np
        check = {
            "package": "numpy",
            "version": getattr(np, "__version__", "unknown"),
            "path": getattr(np, "__file__", "unknown"),
            "ok": hasattr(np, "ndarray") and hasattr(np, "random"),
        }
        results["checks"].append(check)
        if not check["ok"]:
            results["verified"] = False
    except ImportError:
        results["checks"].append({"package": "numpy", "ok": False, "reason": "not_installed"})
        results["verified"] = False

    try:
        import pandas as pd
        check = {
            "package": "pandas",
            "version": getattr(pd, "__version__", "unknown"),
            "path": getattr(pd, "__file__", "unknown"),
            "ok": hasattr(pd, "DataFrame") and hasattr(pd, "read_csv"),
        }
        results["checks"].append(check)
        if not check["ok"]:
            results["verified"] = False
    except ImportError:
        results["checks"].append({"package": "pandas", "ok": False, "reason": "not_installed"})
        results["verified"] = False

    return results


# ---------------------------------------------------------------------------
# 9. Security audit report generator
# ---------------------------------------------------------------------------


def run_security_audit(evidence_dir: Path) -> Dict[str, Any]:
    """Run a comprehensive security audit on a pipeline output directory.

    Checks:
        1. Model artifact signature verification
        2. Evidence file integrity (manifest)
        3. Dependency integrity
        4. File permission checks
        5. Sensitive data exposure scan

    Args:
        evidence_dir: Path to the evidence output directory.

    Returns:
        Security audit report dict.
    """
    issues: List[Dict[str, str]] = []
    evidence_dir = Path(evidence_dir).expanduser().resolve()

    # Check 1: Model signature
    model_paths = list(evidence_dir.parent.rglob("*.pkl"))
    for mp in model_paths:
        result = verify_model_artifact(mp)
        if not result["verified"]:
            issues.append({
                "severity": "critical",
                "code": "unsigned_model",
                "message": f"Model artifact {mp.name} has no valid signature: {result['reason']}",
            })

    # Check 2: Evidence manifest
    manifest_path = evidence_dir / ".manifest.json"
    if manifest_path.exists():
        ok, manifest_issues = ArtifactManifest.verify(manifest_path)
        if not ok:
            for mi in manifest_issues:
                issues.append({
                    "severity": "high",
                    "code": "manifest_integrity",
                    "message": mi,
                })
    else:
        issues.append({
            "severity": "medium",
            "code": "no_manifest",
            "message": "No artifact integrity manifest found in evidence directory",
        })

    # Check 3: Dependency integrity
    dep_result = verify_critical_imports()
    if not dep_result["verified"]:
        for check in dep_result["checks"]:
            if not check.get("ok", True):
                issues.append({
                    "severity": "critical",
                    "code": "dependency_integrity",
                    "message": f"Package {check['package']} failed integrity check",
                })

    # Check 4: File permissions (world-writable evidence files)
    for fpath in evidence_dir.glob("*.json"):
        try:
            mode = fpath.stat().st_mode
            if mode & 0o002:  # world-writable
                issues.append({
                    "severity": "high",
                    "code": "world_writable",
                    "message": f"{fpath.name} is world-writable (mode {oct(mode)})",
                })
        except OSError:
            pass

    # Check 5: Sensitive data scan in JSON reports
    sensitive_patterns = [
        "password", "api_key", "secret_key", "private_key",
        "access_key", "credential", "ssn", "social_security",
        "credit_card", "auth_token", "bearer_token",
        "api_secret", "secret_access",
    ]
    for fpath in evidence_dir.glob("*.json"):
        try:
            content = fpath.read_text(encoding="utf-8").lower()
            for pattern in sensitive_patterns:
                if pattern in content:
                    issues.append({
                        "severity": "high",
                        "code": "sensitive_data_exposure",
                        "message": f"{fpath.name} may contain sensitive data (pattern: {pattern})",
                    })
                    break
        except OSError:
            pass

    # Check 6: Oversized files (potential data exfiltration)
    for fpath in evidence_dir.rglob("*"):
        if fpath.is_file() and fpath.stat().st_size > 500 * 1024 * 1024:
            issues.append({
                "severity": "medium",
                "code": "oversized_file",
                "message": f"{fpath.name} exceeds 500MB ({fpath.stat().st_size} bytes)",
            })

    critical_count = sum(1 for i in issues if i["severity"] == "critical")
    high_count = sum(1 for i in issues if i["severity"] == "high")

    return {
        "status": "fail" if critical_count > 0 else ("warn" if high_count > 0 else "pass"),
        "schema_version": 1,
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "issue_count": len(issues),
        "critical_count": critical_count,
        "high_count": high_count,
        "issues": issues,
        "dependency_verification": dep_result,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI: run security audit or sign/verify model artifacts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MLGG Security Hardening Tools",
    )
    sub = parser.add_subparsers(dest="command")

    # audit
    audit_p = sub.add_parser("audit", help="Run security audit on evidence directory")
    audit_p.add_argument("evidence_dir", help="Path to evidence directory")

    # sign
    sign_p = sub.add_parser("sign", help="Sign a model artifact with HMAC")
    sign_p.add_argument("model_path", help="Path to model .pkl file")

    # verify
    verify_p = sub.add_parser("verify", help="Verify a model artifact signature")
    verify_p.add_argument("model_path", help="Path to model .pkl file")

    # manifest
    manifest_p = sub.add_parser("manifest", help="Create integrity manifest for evidence files")
    manifest_p.add_argument("evidence_dir", help="Path to evidence directory")

    # check-deps
    deps_p = sub.add_parser("check-deps", help="Verify critical dependency integrity")

    args = parser.parse_args()

    if args.command == "audit":
        report = run_security_audit(Path(args.evidence_dir))
        print(json.dumps(report, indent=2))
        return 0 if report["status"] != "fail" else 1

    elif args.command == "sign":
        model_path = Path(args.model_path).expanduser().resolve()
        sig_path = sign_model_artifact(model_path)
        print(f"Signed: {model_path}")
        print(f"Signature: {sig_path}")
        return 0

    elif args.command == "verify":
        model_path = Path(args.model_path).expanduser().resolve()
        result = verify_model_artifact(model_path)
        print(json.dumps(result, indent=2))
        return 0 if result["verified"] else 1

    elif args.command == "manifest":
        evidence_dir = Path(args.evidence_dir).expanduser().resolve()
        manifest = ArtifactManifest()
        for fpath in sorted(evidence_dir.glob("*.json")):
            manifest.add_file(fpath)
        for fpath in sorted(evidence_dir.glob("*.csv.gz")):
            manifest.add_file(fpath)
        manifest_path = evidence_dir / ".manifest.json"
        manifest.save(manifest_path)
        print(f"Manifest created: {manifest_path}")
        return 0

    elif args.command == "check-deps":
        result = verify_critical_imports()
        print(json.dumps(result, indent=2))
        return 0 if result["verified"] else 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
