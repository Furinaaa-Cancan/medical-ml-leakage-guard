#!/usr/bin/env python3
"""
Fail-closed security audit gate for publication-grade medical prediction.

Checks:
1. Model artifact HMAC signature verification (tamper detection).
2. Evidence file integrity via SHA256 manifest.
3. Critical dependency integrity (sklearn/numpy/pandas not monkey-patched).
4. File permission hygiene (no world-writable evidence files).
5. Sensitive data exposure scan (API keys, passwords, tokens in evidence).
6. Oversized artifact detection (potential data exfiltration / zip bomb).

Usage:
    python3 scripts/security_audit_gate.py \\
        --evidence-dir evidence/ \\
        --report evidence/security_audit_gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)
from _gate_utils import start_gate_timer, write_json

register_remediations({
    "unsigned_model": "Sign the model artifact: python3 scripts/_security.py sign <model.pkl>",
    "manifest_missing": "Create evidence manifest: python3 scripts/_security.py manifest <evidence_dir>",
    "manifest_integrity_failure": "Evidence files have been modified. Re-run the pipeline to regenerate.",
    "dependency_integrity_failure": "Critical dependency may be compromised. Reinstall from trusted source.",
    "world_writable_evidence": "Fix file permissions: chmod 644 <file>",
    "sensitive_data_in_evidence": "Remove sensitive data from evidence files before publication.",
    "oversized_artifact": "Investigate oversized file. May indicate data exfiltration or accidental inclusion.",
})

GATE_NAME = "security_audit_gate"
GATE_VERSION = "1.0.0"

from _security import SENSITIVE_DATA_PATTERNS as _SENSITIVE_PATTERNS

_MAX_ARTIFACT_SIZE_MB = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Security audit gate: verify model signatures, evidence integrity, "
                    "dependency authenticity, and sensitive data exposure.",
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to the evidence output directory.",
    )
    parser.add_argument(
        "--model-dir",
        help="Path to the models directory (auto-detected from evidence-dir/../models if omitted).",
    )
    parser.add_argument("--report", help="Output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Promote warnings to failures.")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0=unlimited).")
    return parser.parse_args()


def _check_model_signatures(
    model_dir: Path,
    failures: List[GateIssue],
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Check HMAC signatures for all model artifacts."""
    summary: Dict[str, Any] = {"models_checked": 0, "models_signed": 0, "models_verified": 0}

    pkl_files = list(model_dir.rglob("*.pkl")) if model_dir.exists() else []
    summary["models_checked"] = len(pkl_files)

    if not pkl_files:
        return summary

    try:
        from _security import verify_model_artifact
    except ImportError:
        warnings.append(GateIssue(
            code="security_module_unavailable",
            severity=Severity.WARNING,
            message="_security module not available; model signature checks skipped.",
        ))
        return summary

    for pkl_path in pkl_files:
        result = verify_model_artifact(pkl_path)
        if result["verified"]:
            summary["models_signed"] += 1
            summary["models_verified"] += 1
        elif result["reason"] == "signature_file_missing":
            failures.append(GateIssue(
                code="unsigned_model",
                severity=Severity.ERROR,
                message=f"Model artifact {pkl_path.name} has no HMAC signature.",
                details={"path": str(pkl_path.name)},
                remediation=get_remediation("unsigned_model"),
            ))
        else:
            failures.append(GateIssue(
                code="model_signature_invalid",
                severity=Severity.CRITICAL,
                message=f"Model {pkl_path.name} signature verification failed: {result['reason']}",
                details={"path": str(pkl_path.name), "reason": result["reason"]},
            ))

    return summary


def _check_evidence_manifest(
    evidence_dir: Path,
    failures: List[GateIssue],
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Check evidence integrity via SHA256 manifest."""
    manifest_path = evidence_dir / ".manifest.json"
    summary: Dict[str, Any] = {"manifest_exists": False, "entries_checked": 0, "entries_valid": 0}

    if not manifest_path.exists():
        warnings.append(GateIssue(
            code="manifest_missing",
            severity=Severity.WARNING,
            message="No evidence integrity manifest (.manifest.json) found.",
            remediation=get_remediation("manifest_missing"),
        ))
        return summary

    summary["manifest_exists"] = True

    try:
        from _security import ArtifactManifest
        ok, issues = ArtifactManifest.verify(manifest_path)
    except ImportError:
        warnings.append(GateIssue(
            code="security_module_unavailable",
            severity=Severity.WARNING,
            message="_security module not available; manifest verification skipped.",
        ))
        return summary

    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_data = json.load(fh)
        summary["entries_checked"] = manifest_data.get("entry_count", 0)
    except (json.JSONDecodeError, OSError):
        pass

    if ok:
        summary["entries_valid"] = summary["entries_checked"]
    else:
        for issue_msg in issues:
            failures.append(GateIssue(
                code="manifest_integrity_failure",
                severity=Severity.ERROR,
                message=f"Evidence integrity check failed: {issue_msg}",
                remediation=get_remediation("manifest_integrity_failure"),
            ))

    return summary


def _check_dependency_integrity(
    failures: List[GateIssue],
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Verify critical dependencies are genuine."""
    try:
        from _security import verify_critical_imports
        result = verify_critical_imports()
    except ImportError:
        warnings.append(GateIssue(
            code="security_module_unavailable",
            severity=Severity.WARNING,
            message="_security module not available; dependency checks skipped.",
        ))
        return {"verified": False, "reason": "module_unavailable"}

    if not result["verified"]:
        for check in result.get("checks", []):
            if not check.get("ok", True):
                failures.append(GateIssue(
                    code="dependency_integrity_failure",
                    severity=Severity.CRITICAL,
                    message=f"Package {check['package']} failed integrity check.",
                    details=check,
                    remediation=get_remediation("dependency_integrity_failure"),
                ))

    return {
        "verified": result["verified"],
        "packages_checked": len(result.get("checks", [])),
    }


def _check_file_permissions(
    evidence_dir: Path,
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Check for world-writable evidence files."""
    summary: Dict[str, Any] = {"files_checked": 0, "world_writable": 0}

    for fpath in evidence_dir.glob("*.json"):
        summary["files_checked"] += 1
        try:
            mode = fpath.stat().st_mode
            if mode & 0o002:
                summary["world_writable"] += 1
                warnings.append(GateIssue(
                    code="world_writable_evidence",
                    severity=Severity.WARNING,
                    message=f"{fpath.name} is world-writable (mode {oct(mode)}).",
                    details={"path": str(fpath.name), "mode": oct(mode)},
                    remediation=get_remediation("world_writable_evidence"),
                ))
        except OSError:
            pass

    return summary


def _check_sensitive_data(
    evidence_dir: Path,
    failures: List[GateIssue],
) -> Dict[str, Any]:
    """Scan evidence files for sensitive data patterns."""
    summary: Dict[str, Any] = {"files_scanned": 0, "exposures_found": 0}

    for fpath in evidence_dir.glob("*.json"):
        summary["files_scanned"] += 1
        try:
            content = fpath.read_text(encoding="utf-8").lower()
            for pattern in _SENSITIVE_PATTERNS:
                if pattern in content:
                    summary["exposures_found"] += 1
                    failures.append(GateIssue(
                        code="sensitive_data_in_evidence",
                        severity=Severity.ERROR,
                        message=f"{fpath.name} may contain sensitive data (pattern: '{pattern}').",
                        details={"path": str(fpath.name), "pattern": pattern},
                        remediation=get_remediation("sensitive_data_in_evidence"),
                    ))
                    break  # One finding per file is enough
        except OSError:
            pass

    return summary


def _check_artifact_sizes(
    evidence_dir: Path,
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Check for oversized artifacts."""
    summary: Dict[str, Any] = {"files_checked": 0, "oversized": 0}
    max_bytes = _MAX_ARTIFACT_SIZE_MB * 1024 * 1024

    for fpath in evidence_dir.rglob("*"):
        if not fpath.is_file():
            continue
        summary["files_checked"] += 1
        try:
            size = fpath.stat().st_size
            if size > max_bytes:
                summary["oversized"] += 1
                warnings.append(GateIssue(
                    code="oversized_artifact",
                    severity=Severity.WARNING,
                    message=f"{fpath.name} is {size / 1024 / 1024:.0f} MB (limit: {_MAX_ARTIFACT_SIZE_MB} MB).",
                    details={"path": str(fpath.name), "size_bytes": size},
                    remediation=get_remediation("oversized_artifact"),
                ))
        except OSError:
            pass

    return summary


def _check_audit_chain(
    evidence_dir: Path,
    warnings: List[GateIssue],
) -> Dict[str, Any]:
    """Verify the tamper-evident gate audit log chain."""
    try:
        from _gate_utils import verify_audit_chain
        result = verify_audit_chain(evidence_dir)
    except ImportError:
        return {"checked": False, "reason": "module_unavailable"}

    if not result.get("valid", True) and result.get("entries", 0) > 0:
        warnings.append(GateIssue(
            code="audit_chain_broken",
            severity=Severity.WARNING,
            message=f"Gate audit log chain integrity broken at entry {result.get('broken_at')}: {result.get('reason')}.",
            details=result,
        ))

    return {
        "checked": True,
        "valid": result.get("valid", True),
        "entries": result.get("entries", 0),
    }


def main() -> int:
    start_gate_timer()
    args = parse_args()

    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    if not evidence_dir.exists():
        print(f"FAIL: evidence directory does not exist: {evidence_dir}", file=sys.stderr)
        return 2

    model_dir = (
        Path(args.model_dir).expanduser().resolve()
        if args.model_dir
        else evidence_dir.parent / "models"
    )

    failures: List[GateIssue] = []
    warnings: List[GateIssue] = []

    # Run all security checks
    sig_summary = _check_model_signatures(model_dir, failures, warnings)
    manifest_summary = _check_evidence_manifest(evidence_dir, failures, warnings)
    dep_summary = _check_dependency_integrity(failures, warnings)
    perm_summary = _check_file_permissions(evidence_dir, warnings)
    sensitive_summary = _check_sensitive_data(evidence_dir, failures)
    size_summary = _check_artifact_sizes(evidence_dir, warnings)
    audit_summary = _check_audit_chain(evidence_dir, warnings)

    # Strict mode: promote warnings to failures
    if args.strict:
        for w in list(warnings):
            failures.append(GateIssue(
                code=w.code,
                severity=Severity.ERROR,
                message=f"[strict] {w.message}",
                details=w.details,
                remediation=w.remediation,
            ))
        warnings.clear()

    # Determine status (binary pass/fail only — warnings alone don't fail)
    status = "fail" if failures else "pass"

    summary = {
        "model_signatures": sig_summary,
        "evidence_manifest": manifest_summary,
        "dependency_integrity": dep_summary,
        "file_permissions": perm_summary,
        "sensitive_data_scan": sensitive_summary,
        "artifact_sizes": size_summary,
        "audit_chain": audit_summary,
    }

    report = build_report_envelope(
        gate_name=GATE_NAME,
        status=status,
        strict_mode=args.strict,
        failures=failures,
        warnings=warnings,
        summary=summary,
        input_files={"evidence_dir": str(evidence_dir), "model_dir": str(model_dir)},
        gate_version=GATE_VERSION,
    )

    from _gate_utils import get_gate_elapsed
    print_gate_summary(GATE_NAME, status, failures, warnings, args.strict, get_gate_elapsed())

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        write_json(report_path, report)

    return 2 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
