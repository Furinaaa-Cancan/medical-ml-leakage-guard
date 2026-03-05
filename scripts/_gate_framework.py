"""
Unified gate framework for ml-leakage-guard.

Provides GateBase abstract class, standardized report envelope, severity
levels, remediation hint registry, and CLI helpers. Each gate script can
subclass GateBase for a consistent lifecycle while retaining full control
over its validation logic.

Backward-compatible: existing gates that do NOT subclass GateBase continue
to work unchanged. Migration is incremental and per-gate.
"""

from __future__ import annotations

import abc
import argparse
import enum
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from _gate_utils import (
    add_issue,
    add_timeout_argument,
    get_gate_elapsed,
    install_gate_timeout,
    start_gate_timer,
    write_json,
)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    """Issue severity levels, ordered from most to least critical."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {
            Severity.CRITICAL: 0,
            Severity.ERROR: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3,
        }[self]

    def __lt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank


# ---------------------------------------------------------------------------
# Structured issue
# ---------------------------------------------------------------------------

class GateIssue:
    """A single validation issue with severity and optional remediation."""

    __slots__ = (
        "code",
        "severity",
        "message",
        "details",
        "remediation",
        "source_file",
    )

    def __init__(
        self,
        code: str,
        severity: Severity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        remediation: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> None:
        self.code = code
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.remediation = remediation
        self.source_file = source_file

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }
        if self.remediation:
            d["remediation"] = self.remediation
        if self.source_file:
            d["source_file"] = self.source_file
        return d

    @staticmethod
    def from_legacy(
        legacy: Dict[str, Any],
        severity: Severity,
    ) -> "GateIssue":
        """Convert a legacy issue dict (code/message/details) to GateIssue."""
        return GateIssue(
            code=str(legacy.get("code", "unknown")),
            severity=severity,
            message=str(legacy.get("message", "")),
            details=legacy.get("details") if isinstance(legacy.get("details"), dict) else {},
        )


# ---------------------------------------------------------------------------
# Remediation hint registry
# ---------------------------------------------------------------------------

_REMEDIATION_REGISTRY: Dict[str, str] = {}


def register_remediation(code: str, hint: str) -> None:
    """Register a remediation hint for a failure code."""
    _REMEDIATION_REGISTRY[code] = hint


def get_remediation(code: str) -> Optional[str]:
    """Retrieve a registered remediation hint, or None."""
    return _REMEDIATION_REGISTRY.get(code)


def register_remediations(mapping: Dict[str, str]) -> None:
    """Bulk-register remediation hints."""
    _REMEDIATION_REGISTRY.update(mapping)


# ---------------------------------------------------------------------------
# Built-in remediation hints (cross-gate common codes)
# ---------------------------------------------------------------------------

_COMMON_REMEDIATIONS: Dict[str, str] = {
    "file_not_found": "Verify the file path in your request JSON. Ensure the file exists and is readable.",
    "invalid_json": "Fix JSON syntax errors in the input file. Use a JSON linter to validate.",
    "json_root_not_object": "Ensure the JSON file root is a {} object, not an array or primitive.",
    "missing_required_field": "Add the missing field to the input file. Check the gate documentation for required schema.",
    "hash_mismatch": "Re-generate the artifact. The file content has changed since the hash was recorded.",
    "metric_mismatch": "Re-run model evaluation to produce consistent metrics. Check for non-determinism in the pipeline.",
    "threshold_violation": "Adjust model/data to meet the threshold, or review whether the threshold in performance_policy is appropriate.",
    "strict_mode_warning_as_failure": "This warning is promoted to failure under --strict. Fix the underlying issue or run without --strict for exploratory mode.",
    "gate_timeout": "Increase --timeout value or optimize the gate's input data size.",
    "patient_id_overlap": "Remove overlapping patient IDs between train/valid/test splits. Check split_protocol_spec.",
    "temporal_leakage": "Ensure all training data timestamps precede validation/test timestamps. Review split boundaries.",
    "row_hash_overlap": "Deduplicate identical rows across splits. This likely indicates a split generation bug.",
    "feature_name_suspicious": "Rename or remove features matching the forbidden pattern. They may encode future information.",
    "signature_verification_failed": "Re-sign the artifact with the correct private key. Ensure the public key matches.",
    "key_revoked": "The signing key has been revoked. Re-sign with a non-revoked key.",
    "manifest_comparison_missing": "Provide --compare-manifest with a baseline manifest, or use --allow-missing-compare for first-run bootstrap.",
    "claim_tier_below_publication": "Set claim_tier_target to 'publication-grade' in request JSON for publication-grade validation.",
    "primary_metric_not_pr_auc": "Publication-grade requires primary_metric='pr_auc'. Update your request JSON.",
    "clinical_floor_below_baseline": "Increase clinical floor values to meet publication-grade minimums defined in PUBLICATION_POLICY_BASELINES.",
    "cross_period_missing": "Add at least one cross_period cohort to external validation. This is required for publication-grade.",
    "cross_institution_missing": "Add at least one cross_institution cohort to external validation. This is required for publication-grade.",
    "checklist_incomplete": "Complete all required TRIPOD+AI / PROBAST+AI / STARD-AI checklist items in your checklist spec.",
    "bias_risk_not_low": "Address bias risk factors until overall_bias_risk is 'low'. Review PROBAST+AI domains.",
    "seed_instability": "Model shows excessive variation across random seeds. Consider ensemble methods or more stable architectures.",
    "calibration_poor": "Recalibrate the model (e.g., Platt scaling, isotonic regression) to improve ECE and calibration slope.",
    "ci_width_excessive": "Confidence intervals are too wide. Increase bootstrap resamples or collect more data.",
    "permutation_not_significant": "Model performance is not statistically significant vs. permuted null. Review model validity.",
}

register_remediations(_COMMON_REMEDIATIONS)


# ---------------------------------------------------------------------------
# Report envelope
# ---------------------------------------------------------------------------

REPORT_ENVELOPE_VERSION = "2.0.0"


def build_report_envelope(
    gate_name: str,
    status: str,
    strict_mode: bool,
    failures: List[GateIssue],
    warnings: List[GateIssue],
    summary: Optional[Dict[str, Any]] = None,
    input_files: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    gate_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Build a standardized gate report envelope.

    All gate reports share this top-level structure, making downstream
    parsing (publication_gate, self_critique, render_user_summary)
    uniform and reliable.
    """
    now_utc = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

    failure_dicts = sorted(
        [f.to_dict() for f in failures],
        key=lambda d: Severity(d["severity"]).rank,
    )
    warning_dicts = sorted(
        [w.to_dict() for w in warnings],
        key=lambda d: Severity(d["severity"]).rank,
    )

    envelope: Dict[str, Any] = {
        "envelope_version": REPORT_ENVELOPE_VERSION,
        "gate_name": gate_name,
        "gate_version": gate_version,
        "status": status,
        "strict_mode": strict_mode,
        "execution_timestamp_utc": now_utc,
        "execution_time_seconds": round(get_gate_elapsed(), 3),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failure_dicts,
        "warnings": warning_dicts,
    }

    if summary is not None:
        envelope["summary"] = summary

    if input_files:
        envelope["input_files"] = input_files

    if extra:
        envelope.update(extra)

    return envelope


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all gates: --report, --strict, --timeout, --dry-run."""
    common = parser.add_argument_group("Common gate options")
    common.add_argument(
        "--report",
        help="Path to write the JSON gate report.",
    )
    common.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings to failures (required for publication-grade).",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and arguments only; do not run gate logic.",
    )
    add_timeout_argument(parser)


def add_input_file_argument(
    group: argparse._ArgumentGroup,
    flag: str,
    help_text: str,
    required: bool = True,
) -> None:
    """Add an input file argument with consistent naming."""
    group.add_argument(flag, required=required, help=help_text)


def validate_input_files(
    args: argparse.Namespace,
    file_args: Sequence[str],
) -> List[GateIssue]:
    """Pre-validate that all specified input file arguments point to existing files.

    Returns a list of GateIssue for any missing files.
    """
    issues: List[GateIssue] = []
    for arg_name in file_args:
        value = getattr(args, arg_name.lstrip("-").replace("-", "_"), None)
        if value is None:
            continue
        p = Path(str(value)).expanduser().resolve()
        if not p.exists():
            issues.append(GateIssue(
                code="file_not_found",
                severity=Severity.CRITICAL,
                message=f"Input file not found: {p}",
                details={"argument": arg_name, "path": str(p)},
                remediation=get_remediation("file_not_found"),
            ))
        elif not p.is_file():
            issues.append(GateIssue(
                code="path_not_file",
                severity=Severity.CRITICAL,
                message=f"Path is not a regular file: {p}",
                details={"argument": arg_name, "path": str(p)},
                remediation="Ensure the path points to a file, not a directory.",
            ))
    return issues


# ---------------------------------------------------------------------------
# Terminal output formatting
# ---------------------------------------------------------------------------

_SEVERITY_PREFIXES = {
    Severity.CRITICAL: "\033[1;31m[CRIT]\033[0m",
    Severity.ERROR: "\033[31m[FAIL]\033[0m",
    Severity.WARNING: "\033[33m[WARN]\033[0m",
    Severity.INFO: "\033[36m[INFO]\033[0m",
}

_SEVERITY_PREFIXES_PLAIN = {
    Severity.CRITICAL: "[CRIT]",
    Severity.ERROR: "[FAIL]",
    Severity.WARNING: "[WARN]",
    Severity.INFO: "[INFO]",
}


def _use_color() -> bool:
    """Decide whether to emit ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def format_issue_line(issue: GateIssue) -> str:
    """Format a single issue for terminal output."""
    prefixes = _SEVERITY_PREFIXES if _use_color() else _SEVERITY_PREFIXES_PLAIN
    prefix = prefixes.get(issue.severity, "[????]")
    line = f"{prefix} {issue.code}: {issue.message}"
    if issue.remediation:
        line += f"\n       \u2192 Fix: {issue.remediation}"
    return line


def print_gate_summary(
    gate_name: str,
    status: str,
    failures: List[GateIssue],
    warnings: List[GateIssue],
    strict: bool,
    elapsed: float,
) -> None:
    """Print a structured gate summary to stdout."""
    use_color = _use_color()

    if use_color:
        status_str = (
            "\033[32mPASS\033[0m" if status == "pass"
            else "\033[1;31mFAIL\033[0m"
        )
    else:
        status_str = status.upper()

    print(f"\n{'=' * 60}")
    print(f"Gate: {gate_name}")
    print(f"Status: {status_str}  |  Failures: {len(failures)}  |  Warnings: {len(warnings)}  |  Strict: {strict}  |  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    all_issues = sorted(
        [(issue, True) for issue in failures] + [(issue, False) for issue in warnings],
        key=lambda pair: pair[0].severity.rank,
    )

    if all_issues:
        print()
        for issue, _is_failure in all_issues:
            print(format_issue_line(issue))
        print()

    critical_count = sum(1 for f in failures if f.severity == Severity.CRITICAL)
    if critical_count > 0:
        print(f"  \u26a0  {critical_count} CRITICAL issue(s) require immediate attention.")
        print()


# ---------------------------------------------------------------------------
# GateBase abstract class
# ---------------------------------------------------------------------------

class GateBase(abc.ABC):
    """Abstract base class for all gate scripts.

    Subclasses implement:
        - ``gate_name`` class attribute
        - ``configure_parser()`` to add gate-specific arguments
        - ``run_checks()`` to execute validation logic
        - optionally ``build_summary()`` to add gate-specific report data

    The lifecycle is::

        gate = MyGate()
        exit_code = gate.execute(sys.argv[1:])
    """

    gate_name: str = "unknown_gate"
    gate_version: str = "1.0.0"

    # Subclasses list argument names that point to input files, for pre-validation.
    input_file_args: Sequence[str] = ()

    def __init__(self) -> None:
        self._failures: List[GateIssue] = []
        self._warnings: List[GateIssue] = []
        self._args: Optional[argparse.Namespace] = None
        self._input_files: Dict[str, str] = {}

    # -- Issue collection API --

    def add_failure(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: Severity = Severity.ERROR,
        remediation: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> None:
        hint = remediation or get_remediation(code)
        self._failures.append(GateIssue(
            code=code,
            severity=severity,
            message=message,
            details=details,
            remediation=hint,
            source_file=source_file,
        ))

    def add_warning(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: Severity = Severity.WARNING,
        remediation: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> None:
        hint = remediation or get_remediation(code)
        self._warnings.append(GateIssue(
            code=code,
            severity=severity,
            message=message,
            details=details,
            remediation=hint,
            source_file=source_file,
        ))

    def add_failure_legacy(
        self,
        bucket: List[Dict[str, Any]],
        code: str,
        message: str,
        details: Dict[str, Any],
    ) -> None:
        """Legacy-compatible: append to old-style list AND new framework."""
        add_issue(bucket, code, message, details)
        self.add_failure(code, message, details)

    def add_warning_legacy(
        self,
        bucket: List[Dict[str, Any]],
        code: str,
        message: str,
        details: Dict[str, Any],
    ) -> None:
        """Legacy-compatible: append to old-style list AND new framework."""
        add_issue(bucket, code, message, details)
        self.add_warning(code, message, details)

    # -- Parser setup --

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.get_description(),
            epilog=self.get_epilog(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        add_common_arguments(parser)
        self.configure_parser(parser)
        return parser

    def get_description(self) -> str:
        return f"Gate: {self.gate_name}"

    def get_epilog(self) -> Optional[str]:
        return None

    @abc.abstractmethod
    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add gate-specific arguments to the parser."""

    # -- Validation logic --

    @abc.abstractmethod
    def run_checks(self, args: argparse.Namespace) -> None:
        """Execute the gate's validation logic.

        Use self.add_failure() / self.add_warning() to record issues.
        """

    def build_summary(self, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
        """Return gate-specific summary data for the report. Override in subclass."""
        return None

    # -- Dry-run hook --

    def dry_run(self, args: argparse.Namespace) -> int:
        """Execute dry-run checks using already-validated file issues.

        Returns 0 if inputs are valid, 2 otherwise.
        """
        if self._failures:
            for issue in self._failures:
                print(format_issue_line(issue))
            return 2
        print(f"[DRY-RUN] {self.gate_name}: all input files validated OK.")
        return 0

    # -- Main execution lifecycle --

    def execute(self, argv: Optional[Sequence[str]] = None) -> int:
        """Full gate lifecycle: parse -> validate inputs -> run -> report -> exit."""
        parser = self.create_parser()
        self._args = parser.parse_args(argv)
        args = self._args

        start_gate_timer()

        if getattr(args, "timeout", 0) > 0:
            report_path = Path(args.report).expanduser().resolve() if args.report else None
            install_gate_timeout(args.timeout, report_path, self.gate_name)

        # Collect input file paths for the report
        for arg_name in self.input_file_args:
            attr = arg_name.lstrip("-").replace("-", "_")
            val = getattr(args, attr, None)
            if val is not None:
                self._input_files[attr] = str(Path(str(val)).expanduser().resolve())

        # Pre-validate input files
        file_issues = validate_input_files(args, self.input_file_args)
        for issue in file_issues:
            self._failures.append(issue)

        # Dry-run: stop after input validation
        if getattr(args, "dry_run", False):
            return self.dry_run(args)

        # Run actual checks only if no critical file issues
        critical_file_issues = [i for i in file_issues if i.severity == Severity.CRITICAL]
        if not critical_file_issues:
            self.run_checks(args)

        return self._finish(args)

    def _finish(self, args: argparse.Namespace) -> int:
        """Build report, print summary, write JSON, return exit code."""
        strict = getattr(args, "strict", False)
        should_fail = bool(self._failures) or (strict and bool(self._warnings))
        status = "fail" if should_fail else "pass"
        elapsed = get_gate_elapsed()

        summary = self.build_summary(args)

        report = build_report_envelope(
            gate_name=self.gate_name,
            status=status,
            strict_mode=strict,
            failures=self._failures,
            warnings=self._warnings,
            summary=summary,
            input_files=self._input_files if self._input_files else None,
            gate_version=self.gate_version,
        )

        if args.report:
            write_json(Path(args.report).expanduser().resolve(), report)

        print_gate_summary(
            gate_name=self.gate_name,
            status=status,
            failures=self._failures,
            warnings=self._warnings,
            strict=strict,
            elapsed=elapsed,
        )

        return 2 if should_fail else 0


# ---------------------------------------------------------------------------
# Adapter: wrap legacy finish() output into the new envelope format
# ---------------------------------------------------------------------------

def wrap_legacy_report(
    gate_name: str,
    legacy_report: Dict[str, Any],
    gate_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert a legacy gate report dict into the new envelope format.

    Useful for downstream consumers (publication_gate, render_user_summary)
    that need to handle both old and new format reports.
    """
    if "envelope_version" in legacy_report:
        return legacy_report

    now_utc = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

    raw_failures = legacy_report.get("failures", [])
    raw_warnings = legacy_report.get("warnings", [])

    failures = []
    for item in raw_failures:
        if isinstance(item, dict):
            code = str(item.get("code", "unknown"))
            entry: Dict[str, Any] = {
                "code": code,
                "severity": Severity.ERROR.value,
                "message": str(item.get("message", "")),
                "details": item.get("details", {}),
            }
            hint = get_remediation(code)
            if hint:
                entry["remediation"] = hint
            failures.append(entry)

    warnings = []
    for item in raw_warnings:
        if isinstance(item, dict):
            code = str(item.get("code", "unknown"))
            wentry: Dict[str, Any] = {
                "code": code,
                "severity": Severity.WARNING.value,
                "message": str(item.get("message", "")),
                "details": item.get("details", {}),
            }
            whint = get_remediation(code)
            if whint:
                wentry["remediation"] = whint
            warnings.append(wentry)

    envelope: Dict[str, Any] = {
        "envelope_version": REPORT_ENVELOPE_VERSION,
        "gate_name": gate_name,
        "gate_version": gate_version,
        "status": str(legacy_report.get("status", "unknown")),
        "strict_mode": bool(legacy_report.get("strict_mode", False)),
        "execution_timestamp_utc": now_utc,
        "execution_time_seconds": legacy_report.get("execution_time_seconds", 0),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }

    for key in ("summary", "normalized_request", "payload_metadata",
                "components", "quality_score", "recommendations",
                "actual_metric", "metrics"):
        if key in legacy_report:
            envelope[key] = legacy_report[key]

    return envelope


def load_gate_report(path: Path, gate_name: str) -> Dict[str, Any]:
    """Load a gate report JSON and normalize to envelope format.

    Handles both new-format (envelope_version present) and legacy reports.
    """
    from _gate_utils import load_json_from_path
    raw = load_json_from_path(path)
    return wrap_legacy_report(gate_name, raw)
