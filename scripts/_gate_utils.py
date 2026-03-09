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


_MAX_JSON_FILE_SIZE = 100 * 1024 * 1024  # 100 MB safety limit


def _check_json_file_size(path: Path) -> None:
    """Raise ValueError if a JSON file exceeds the safety size limit."""
    try:
        size = path.stat().st_size
        if size > _MAX_JSON_FILE_SIZE:
            raise ValueError(
                f"JSON file too large: {path} is {size} bytes "
                f"(limit {_MAX_JSON_FILE_SIZE // (1024*1024)} MB)"
            )
    except OSError:
        pass  # File may not exist yet; let caller handle


def load_json_from_path(path: Path) -> Dict[str, Any]:
    """Load and validate a JSON object from a Path."""
    _check_json_file_size(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def load_json_from_str(path: str) -> Dict[str, Any]:
    """Load and validate a JSON object from a string path."""
    p = Path(path).expanduser().resolve()
    _check_json_file_size(p)
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc}") from exc
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
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
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


_FORBIDDEN_PATH_PREFIXES = [
    "/etc", "/private/etc", "/proc", "/sys", "/dev",
    "/var/run", "/boot", "/sbin",
]


def resolve_path(base: Path, value: str, sandbox: Optional[Path] = None) -> Path:
    """Resolve a potentially relative path against a base directory.

    Args:
        base: Base directory for relative paths.
        value: Raw path string to resolve.
        sandbox: If provided, the resolved path must be under this directory.

    Raises:
        ValueError: If the path contains null bytes or targets forbidden system paths.
    """
    if "\x00" in value:
        raise ValueError(f"Null byte in path: {value!r}")
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    resolved = str(p)
    for prefix in _FORBIDDEN_PATH_PREFIXES:
        if resolved.startswith(prefix + "/") or resolved == prefix:
            raise ValueError(f"Path targets forbidden system location: {p}")
    if sandbox is not None:
        sandbox_resolved = sandbox.resolve()
        if not resolved.startswith(str(sandbox_resolved)):
            raise ValueError(
                f"Path escapes sandbox: {p} is not under {sandbox_resolved}"
            )
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
            "gate_name": gate_name,
            "strict_mode": True,
            "timeout_seconds": timeout_seconds,
            "failures": [
                {
                    "code": "gate_timeout",
                    "message": f"Gate exceeded {timeout_seconds}s timeout.",
                    "details": {"timeout_seconds": timeout_seconds},
                }
            ],
            "warnings": [],
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


def try_parse_time(value: str) -> Optional[float]:
    """Parse a time string to epoch float, trying multiple formats.

    Args:
        value: Raw time string (ISO-8601, date, or numeric epoch).

    Returns:
        Epoch timestamp as float, or None if unparseable.
    """
    import datetime as _dt

    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    iso = s.replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except ValueError:
        pass
    formats = (
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    )
    for fmt in formats:
        try:
            return _dt.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def epoch_to_iso(ts: Optional[float]) -> Optional[str]:
    """Convert epoch timestamp to UTC ISO-8601 string.

    Args:
        ts: Epoch timestamp, or None.

    Returns:
        ISO-8601 string with Z suffix, or None.
    """
    import datetime as _dt

    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


# ---------------------------------------------------------------------------
# Shared numeric helpers used by multiple gate scripts
# ---------------------------------------------------------------------------


def canonical_metric_token(value: str) -> str:
    """Normalize a metric name to a canonical lowercase token for comparison."""
    import re
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_finite_number(value: Any) -> bool:
    """Check whether *value* is a finite int or float (excluding bool)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def to_int(value: Any) -> Optional[int]:
    """Safely convert *value* to int if it is integer-like, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def safe_ratio(num: float, den: float) -> float:
    """Return *num / den*, or 0.0 when *den* is non-positive."""
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def confusion_counts(
    y_true: "numpy.ndarray", y_pred: "numpy.ndarray"  # noqa: F821
) -> Dict[str, int]:
    """Compute TP/FP/TN/FN from binary label arrays.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).

    Returns:
        Dict with keys ``tp``, ``fp``, ``tn``, ``fn``.
    """
    import numpy as np  # local import to keep module lightweight

    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def normalize_binary(values: "pandas.Series") -> Optional["numpy.ndarray"]:  # noqa: F821
    """Coerce a pandas Series to a binary int ndarray, or None on failure.

    Returns None when the series contains non-finite or non-{0,1} values.
    """
    import numpy as np
    import pandas as pd

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(arr)):
        return None
    if not np.all(np.isin(arr, [0.0, 1.0])):
        return None
    return arr.astype(int)


def metric_panel(
    y_true: "numpy.ndarray",  # noqa: F821
    y_score: "numpy.ndarray",  # noqa: F821
    y_pred: "numpy.ndarray",  # noqa: F821
    beta: float,
) -> tuple:
    """Compute a standard binary-classification metric panel.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_score: Predicted probabilities in [0, 1].
        y_pred: Hard predictions (0/1).
        beta: Beta for F-beta score (typically 2.0).

    Returns:
        ``(metrics_dict, confusion_matrix_dict)`` tuple.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    cm = confusion_counts(y_true, y_pred)
    tp = float(cm["tp"])
    fp = float(cm["fp"])
    tn = float(cm["tn"])
    fn = float(cm["fn"])
    precision = safe_ratio(tp, tp + fp)
    sensitivity = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    npv = safe_ratio(tn, tn + fn)
    accuracy = safe_ratio(tp + tn, tp + fp + tn + fn)
    f1 = (
        0.0
        if (precision + sensitivity) <= 0
        else (2.0 * precision * sensitivity) / (precision + sensitivity)
    )
    beta_sq = beta * beta
    f2 = (
        0.0
        if ((beta_sq * precision) + sensitivity) <= 0
        else ((1.0 + beta_sq) * precision * sensitivity)
        / ((beta_sq * precision) + sensitivity)
    )
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "ppv": precision,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "f2_beta": f2,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
    }
    return metrics, cm


# ---------------------------------------------------------------------------
# Tamper-evident audit log for gate executions
# ---------------------------------------------------------------------------

_AUDIT_LOG_NAME = ".gate_audit.jsonl"


def _hmac_chain(prev_hash: str, entry_json: str) -> str:
    """Compute HMAC-SHA256 chain hash linking this entry to the previous."""
    import hashlib
    import hmac
    key = prev_hash.encode("utf-8")
    return hmac.new(key, entry_json.encode("utf-8"), hashlib.sha256).hexdigest()


def append_audit_entry(
    evidence_dir: Path,
    gate_name: str,
    status: str,
    failure_count: int = 0,
    warning_count: int = 0,
    execution_time: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a tamper-evident audit log entry for a gate execution.

    Each entry contains:
      - timestamp, gate name, status, counts
      - chain_hash: HMAC-SHA256 linking to previous entry (tamper detection)
      - pid and hostname for forensic tracing

    Args:
        evidence_dir: Directory containing the audit log.
        gate_name: Name of the executed gate.
        status: Gate result status (pass/fail).
        failure_count: Number of failures.
        warning_count: Number of warnings.
        execution_time: Execution duration in seconds.
        extra: Optional additional metadata.
    """
    import datetime as _dt
    import platform

    log_path = evidence_dir / _AUDIT_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Read last chain hash
    prev_hash = "0" * 64
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                prev_hash = last.get("chain_hash", prev_hash)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    entry: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "gate_name": gate_name,
        "status": status,
        "failure_count": failure_count,
        "warning_count": warning_count,
        "execution_time_seconds": round(execution_time, 3),
        "pid": os.getpid(),
        "hostname": platform.node(),
    }
    if extra:
        entry["extra"] = extra

    entry_json = json.dumps(entry, ensure_ascii=True, sort_keys=True)
    entry["chain_hash"] = _hmac_chain(prev_hash, entry_json)

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def verify_audit_chain(evidence_dir: Path) -> Dict[str, Any]:
    """Verify the integrity of the gate audit log chain.

    Returns:
        Dict with 'valid' (bool), 'entries' (int), 'broken_at' (int or None).
    """
    log_path = evidence_dir / _AUDIT_LOG_NAME
    if not log_path.exists():
        return {"valid": True, "entries": 0, "broken_at": None, "reason": "no_log"}

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {"valid": True, "entries": 0, "broken_at": None}

    prev_hash = "0" * 64
    for idx, line in enumerate(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return {"valid": False, "entries": len(lines), "broken_at": idx, "reason": "json_parse_error"}

        stored_hash = entry.pop("chain_hash", None)
        if stored_hash is None:
            return {"valid": False, "entries": len(lines), "broken_at": idx, "reason": "missing_chain_hash"}

        entry_json = json.dumps(entry, ensure_ascii=True, sort_keys=True)
        expected = _hmac_chain(prev_hash, entry_json)
        if stored_hash != expected:
            return {"valid": False, "entries": len(lines), "broken_at": idx, "reason": "chain_hash_mismatch"}
        prev_hash = stored_hash

    return {"valid": True, "entries": len(lines), "broken_at": None}
