#!/usr/bin/env python3
"""
Environment self-check for ml-leakage-guard.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

from _gate_utils import add_issue, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Python/runtime dependencies for ml-leakage-guard.")
    parser.add_argument(
        "--require-optional-models",
        default="",
        help="Comma-separated optional backends that must be installed (xgboost,catboost,lightgbm,tabpfn,optuna).",
    )
    parser.add_argument("--strict", action="store_true", help="Treat optional warnings as failures.")
    parser.add_argument("--report", help="Optional output JSON report path.")
    return parser.parse_args()


def parse_required_optional(raw: str) -> List[str]:
    tokens = [part.strip().lower() for part in str(raw).split(",")]
    out: List[str] = []
    for token in tokens:
        if token and token not in out:
            out.append(token)
    return out


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    required_optional = parse_required_optional(args.require_optional_models)
    optional_packages = {"xgboost": "xgboost", "catboost": "catboost", "lightgbm": "lightgbm", "tabpfn": "tabpfn", "optuna": "optuna"}
    core_packages = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn", "joblib": "joblib"}

    package_status: Dict[str, Any] = {}
    for label, module_name in core_packages.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            package_status[label] = {"installed": True, "version": str(version)}
        except Exception as exc:
            package_status[label] = {"installed": False, "error": str(exc)}
            add_issue(
                failures,
                "core_dependency_missing",
                "Required package is not installed.",
                {"package": label, "module": module_name, "install_hint": f"pip install {label}"},
            )

    optional_status: Dict[str, Any] = {}
    for key, module_name in optional_packages.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            optional_status[key] = {"installed": True, "version": str(version)}
        except Exception:
            optional_status[key] = {
                "installed": False,
                "install_hint": f"pip install {module_name}",
            }
            if key in required_optional:
                add_issue(
                    failures,
                    "optional_backend_missing",
                    "Required optional model backend is not installed.",
                    {"backend": key, "install_hint": f"pip install {module_name}"},
                )
            else:
                add_issue(
                    warnings,
                    "optional_backend_not_installed",
                    "Optional backend not installed; related models will be unavailable.",
                    {"backend": key, "install_hint": f"pip install {module_name}"},
                )

    py_version = sys.version_info
    if (py_version.major, py_version.minor) < (3, 10):
        add_issue(
            failures,
            "python_version_unsupported",
            "Python >= 3.10 is required.",
            {"current": platform.python_version(), "required_min": "3.10"},
        )

    openssl_path = shutil.which("openssl")
    if openssl_path is None:
        add_issue(
            warnings,
            "openssl_not_found",
            "OpenSSL CLI not found; execution attestation key generation commands may fail.",
            {"install_hint": "Install openssl and ensure it is in PATH."},
        )
    git_path = shutil.which("git")
    if git_path is None:
        add_issue(
            warnings,
            "git_not_found",
            "git not found in PATH; reproducibility workflows may be limited.",
            {"install_hint": "Install git and ensure it is in PATH."},
        )

    cpu_count = os.cpu_count() or 1
    if cpu_count < 2:
        add_issue(
            warnings,
            "low_cpu_parallelism",
            "Detected only one CPU core; multi-core model runs will be limited.",
            {"cpu_count": cpu_count},
        )

    if args.strict and warnings:
        for warning in warnings:
            add_issue(
                failures,
                "strict_warning_promoted_to_failure",
                "Strict mode promotes environment warnings to failure.",
                {"source_code": warning["code"], "details": warning["details"]},
            )

    report = {
        "status": "fail" if failures else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": int(len(failures)),
        "warning_count": int(len(warnings)),
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": int(cpu_count),
            "core_packages": package_status,
            "optional_packages": optional_status,
            "required_optional_models": required_optional,
            "openssl_path": openssl_path,
            "git_path": git_path,
        },
    }

    if args.report:
        write_json(Path(args.report).expanduser().resolve(), report)

    print(f"Status: {report['status']}")
    print(f"Failures: {report['failure_count']} | Warnings: {report['warning_count']} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
