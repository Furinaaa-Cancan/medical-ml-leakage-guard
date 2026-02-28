"""
Shared utility functions for ml-leakage-guard gate scripts.

This module consolidates common helper functions that are duplicated across
gate scripts. Each gate script remains independently runnable; importing
from this module is optional and does not change gate semantics.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def add_issue(
    bucket: List[Dict[str, Any]],
    code: str,
    message: str,
    details: Dict[str, Any],
) -> None:
    bucket.append({"code": code, "message": message, "details": details})


def load_json_from_path(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def load_json_from_str(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object.")
    return payload


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    if isinstance(path, Path):
        return load_json_from_path(path)
    return load_json_from_str(path)


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
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


def write_json_simple(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


def to_float(value: Any) -> Optional[float]:
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
