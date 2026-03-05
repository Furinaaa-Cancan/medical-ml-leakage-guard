#!/usr/bin/env python3
"""
Create deterministic evidence manifest for medical ML studies.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock dataset/config fingerprints into a manifest.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input file paths to fingerprint (datasets, configs, specs).",
    )
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--algo", default="sha256", choices=["sha256"], help="Hash algorithm.")
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Optional metadata key=value, repeatable.",
    )
    parser.add_argument(
        "--compare-with",
        help="Optional baseline manifest JSON to compare against (fail on mismatch).",
    )
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_summary(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"rows": None, "columns": None, "header": None, "header_sha256": None}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                summary["rows"] = 0
                return summary
            normalized = [h.strip() for h in header]
            summary["columns"] = len(normalized)
            summary["header"] = normalized
            summary["header_sha256"] = hashlib.sha256(
                "\x1f".join(normalized).encode("utf-8")
            ).hexdigest()

            row_count = 0
            for _ in reader:
                row_count += 1
            summary["rows"] = row_count
    except Exception as exc:  # pragma: no cover - defensive
        summary["error"] = str(exc)
    return summary


def parse_meta(meta_items: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for item in meta_items:
        if "=" not in item:
            raise ValueError(f"Invalid --meta item (expected key=value): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --meta key in item: {item}")
        meta[key] = value.strip()
    return meta


def digest_map(manifest: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return out
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        sha = entry.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and path and sha:
            out[path] = sha
    return out


def compare_manifest(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    current_map = digest_map(current)
    baseline_map = digest_map(baseline)

    current_paths = set(current_map.keys())
    baseline_paths = set(baseline_map.keys())
    missing_in_current = sorted(baseline_paths - current_paths)
    missing_in_baseline = sorted(current_paths - baseline_paths)

    hash_mismatches = []
    for path in sorted(current_paths & baseline_paths):
        if current_map[path] != baseline_map[path]:
            hash_mismatches.append(
                {
                    "path": path,
                    "current_sha256": current_map[path],
                    "baseline_sha256": baseline_map[path],
                }
            )

    matched = not missing_in_current and not missing_in_baseline and not hash_mismatches
    return {
        "matched": matched,
        "missing_in_current": missing_in_current,
        "missing_in_baseline": missing_in_baseline,
        "hash_mismatches": hash_mismatches,
        "current_file_count": len(current_map),
        "baseline_file_count": len(baseline_map),
    }


def main() -> int:
    args = parse_args()

    manifest: Dict[str, Any] = {
        "status": "pass",
        "created_at_utc": utc_now(),
        "hash_algorithm": args.algo,
        "files": [],
        "meta": {},
        "errors": [],
    }

    try:
        manifest["meta"] = parse_meta(args.meta)
    except ValueError as exc:
        manifest["status"] = "fail"
        manifest["errors"].append(str(exc))
        return write_and_exit(manifest, args.output, 2)

    for raw_path in args.inputs:
        path = Path(raw_path).expanduser().resolve()
        entry: Dict[str, Any] = {
            "input": raw_path,
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
        }

        if not path.exists():
            entry["error"] = "path_not_found"
            manifest["errors"].append(f"Missing input: {raw_path}")
            manifest["status"] = "fail"
            manifest["files"].append(entry)
            continue
        if not path.is_file():
            entry["error"] = "not_a_file"
            manifest["errors"].append(f"Not a file: {raw_path}")
            manifest["status"] = "fail"
            manifest["files"].append(entry)
            continue

        stat = path.stat()
        entry["size_bytes"] = stat.st_size
        entry["mtime_utc"] = dt.datetime.fromtimestamp(
            stat.st_mtime, tz=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        entry["sha256"] = file_sha256(path)

        if path.suffix.lower() == ".csv":
            entry["csv_summary"] = csv_summary(path)
        manifest["files"].append(entry)

    if args.compare_with:
        baseline_path = Path(args.compare_with).expanduser().resolve()
        manifest["comparison"] = {
            "baseline_manifest_path": str(baseline_path),
            "matched": False,
        }
        if not baseline_path.exists():
            manifest["status"] = "fail"
            manifest["errors"].append(f"Baseline manifest not found: {baseline_path}")
        else:
            try:
                with baseline_path.open("r", encoding="utf-8") as fh:
                    baseline_manifest = json.load(fh)
                if not isinstance(baseline_manifest, dict):
                    raise ValueError("Baseline manifest root must be object.")
                comparison = compare_manifest(manifest, baseline_manifest)
                manifest["comparison"].update(comparison)
                if not comparison["matched"]:
                    manifest["status"] = "fail"
                    manifest["errors"].append("Manifest comparison mismatch against baseline.")
            except Exception as exc:
                manifest["status"] = "fail"
                manifest["errors"].append(f"Failed to read baseline manifest: {exc}")

    output_path = Path(args.output).expanduser().resolve()
    from _gate_utils import write_json as _write_manifest
    _write_manifest(output_path, manifest)

    if manifest["status"] != "pass":
        print("Status: fail")
        for err in manifest["errors"]:
            print(f"[FAIL] {err}")
        return 2

    print("Status: pass")
    print(f"Files locked: {len(manifest['files'])}")
    print(f"Manifest: {output_path}")
    return 0


def write_and_exit(manifest: Dict[str, Any], output: str, code: int) -> int:
    output_path = Path(output).expanduser().resolve()
    from _gate_utils import write_json as _write_manifest
    _write_manifest(output_path, manifest)
    print(f"Status: {manifest.get('status', 'fail')}")
    for err in manifest.get("errors", []):
        print(f"[FAIL] {err}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
