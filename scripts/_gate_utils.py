"""
Shared utility functions for ml-leakage-guard gate scripts.

This module consolidates common helper functions that are duplicated across
gate scripts. Each gate script remains independently runnable; importing
from this module is optional and does not change gate semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def add_issue(
    bucket: List[Dict[str, Any]],
    code: str,
    message: str,
    details: Dict[str, Any],
) -> None:
    """Append a structured issue dict to a bucket list."""
    bucket.append({"code": code, "message": message, "details": details})


def load_json_from_path(path: Path) -> Dict[str, Any]:
    """Load and validate a JSON object from a Path."""
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def load_json_from_str(path: str) -> Dict[str, Any]:
    """Load and validate a JSON object from a string path."""
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object.")
    return payload


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object from a string or Path."""
    if isinstance(path, Path):
        return load_json_from_path(path)
    return load_json_from_str(path)


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON object if the file exists, else return None."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON object to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{int(time.time() * 1_000_000)}"
    )
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(path)


def resolve_path(base: Path, value: str) -> Path:
    """Resolve a potentially relative path against a base directory."""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


_gate_start_time: Optional[float] = None


def start_gate_timer() -> None:
    """Record the gate start time for execution timing."""
    global _gate_start_time
    _gate_start_time = time.time()


def get_gate_elapsed() -> float:
    """Return elapsed seconds since start_gate_timer() was called."""
    if _gate_start_time is None:
        return 0.0
    return time.time() - _gate_start_time


def inject_execution_time(report: Dict[str, Any]) -> Dict[str, Any]:
    """Add execution_time_seconds to a gate report dict.

    Args:
        report: Gate report dict to augment.

    Returns:
        The same dict with execution_time_seconds added.
    """
    report["execution_time_seconds"] = round(get_gate_elapsed(), 3)
    return report


class GateTimeoutError(Exception):
    """Raised when a gate exceeds its configured timeout."""


def add_timeout_argument(parser: argparse.ArgumentParser) -> None:
    """Add a --timeout argument to a gate argparse parser.

    Args:
        parser: ArgumentParser to add the --timeout flag to.
    """
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Maximum execution time in seconds (0=unlimited).",
    )


def install_gate_timeout(
    timeout_seconds: int,
    report_path: Optional[Path] = None,
    gate_name: str = "unknown_gate",
) -> None:
    """Install a SIGALRM-based timeout for a gate script.

    When the timeout fires, a timeout report is written (if report_path
    is provided) and the process exits with code 2 (fail).

    Args:
        timeout_seconds: Seconds before timeout (0 = disabled).
        report_path: Path to write the timeout report JSON.
        gate_name: Name of the gate for the report.
    """
    if timeout_seconds <= 0:
        return
    if not hasattr(signal, "SIGALRM"):
        return

    def _handler(signum: int, frame: Any) -> None:
        payload: Dict[str, Any] = {
            "status": "fail",
            "gate": gate_name,
            "timeout_seconds": timeout_seconds,
            "issues": [
                {
                    "code": "gate_timeout",
                    "message": f"Gate exceeded {timeout_seconds}s timeout.",
                    "details": {"timeout_seconds": timeout_seconds},
                }
            ],
        }
        if report_path is not None:
            try:
                write_json(report_path, payload)
            except Exception:
                pass
        print(
            f"TIMEOUT: {gate_name} exceeded {timeout_seconds}s limit.",
            file=sys.stderr,
        )
        sys.exit(2)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)


def to_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, rejecting inf/nan and non-numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            parsed = float(token)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None
