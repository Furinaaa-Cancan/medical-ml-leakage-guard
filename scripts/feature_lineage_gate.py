#!/usr/bin/env python3
"""
Fail-closed lineage gate to detect derived leakage from disease-definition variables.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from _gate_framework import (
    GateIssue,
    Severity,
    build_report_envelope,
    get_remediation,
    print_gate_summary,
    register_remediations,
)
from _gate_utils import add_issue


register_remediations({
    "forbidden_feature_exact": "Feature matches a forbidden disease-definition variable. Remove it from the feature set.",
    "forbidden_feature_pattern": "Feature matches a forbidden pattern derived from disease definition. Remove or rename.",
    "lineage_missing": "Feature has no lineage entry. Add it to feature_lineage_spec or use --allow-missing-lineage.",
    "lineage_spec_missing": "Provide a valid feature_lineage_spec JSON.",
    "definition_spec_missing": "Provide a valid phenotype_definition_spec JSON.",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect lineage-level leakage from disease-definition variables."
    )
    parser.add_argument("--target", required=True, help="Target name in definition spec.")
    parser.add_argument("--definition-spec", required=True, help="Path to phenotype definition JSON.")
    parser.add_argument("--lineage-spec", required=True, help="Path to feature lineage JSON.")
    parser.add_argument("--train", required=True, help="Training CSV path.")
    parser.add_argument("--valid", help="Validation CSV path.")
    parser.add_argument("--test", help="Test CSV path.")
    parser.add_argument("--target-col", default="y", help="Target column name.")
    parser.add_argument("--ignore-cols", default="", help="Comma-separated non-feature columns.")
    parser.add_argument(
        "--allow-missing-lineage",
        action="store_true",
        help="Allow features without lineage entry in strict mode.",
    )
    parser.add_argument("--report", help="Optional output JSON report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def parse_comma_set(raw: str) -> Set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def read_csv_header(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"Missing header row: {path}")
    return [h.strip() for h in header]


def resolve_target_block(spec: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
    targets = spec.get("targets")
    if not isinstance(targets, dict):
        return None
    if target in targets and isinstance(targets[target], dict):
        return targets[target]
    lowered = target.lower()
    for key, value in targets.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
            return value
    return None


def list_from(obj: Dict[str, Any], key: str) -> List[str]:
    raw = obj.get(key, [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Field '{key}' must be a list.")
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def compile_patterns(patterns: Iterable[str]) -> Tuple[List[re.Pattern[str]], List[str]]:
    compiled: List[re.Pattern[str]] = []
    errors: List[str] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, flags=re.IGNORECASE))
        except re.error as exc:
            errors.append(f"Invalid regex '{pattern}': {exc}")
    return compiled, errors


def normalize_lineage_payload(raw: Dict[str, Any]) -> Dict[str, List[str]]:
    features = raw.get("features")
    source = features if isinstance(features, dict) else raw
    if not isinstance(source, dict):
        raise ValueError("Lineage spec must be an object or contain object field 'features'.")

    lineage: Dict[str, List[str]] = {}
    for feature, payload in source.items():
        if not isinstance(feature, str) or not feature.strip():
            continue
        ancestors: List[str] = []
        if isinstance(payload, list):
            ancestors = [str(x).strip() for x in payload if str(x).strip()]
        elif isinstance(payload, dict):
            raw_ancestors = payload.get("ancestors", [])
            if isinstance(raw_ancestors, list):
                ancestors = [str(x).strip() for x in raw_ancestors if str(x).strip()]
        elif isinstance(payload, str):
            if payload.strip():
                ancestors = [payload.strip()]
        lineage[feature.strip()] = ancestors
    return lineage


def build_lineage_key_index(lineage_map: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    index: Dict[str, str] = {}
    collisions: Dict[str, Set[str]] = {}
    for key in lineage_map.keys():
        nk = norm(key)
        existing = index.get(nk)
        if existing is None:
            index[nk] = key
            continue
        if existing != key:
            collisions.setdefault(nk, set()).update({existing, key})
    return index, {k: sorted(v) for k, v in collisions.items()}


def resolve_lineage_key(node: str, lineage_map: Dict[str, List[str]], key_index: Dict[str, str]) -> Optional[str]:
    if node in lineage_map:
        return node
    return key_index.get(norm(node))


def collect_transitive_candidates(
    feature: str,
    lineage_map: Dict[str, List[str]],
    key_index: Dict[str, str],
    max_depth: int = 256,
) -> Tuple[Set[str], List[List[str]], bool]:
    candidates: Set[str] = {feature}
    cycles: List[List[str]] = []
    overflow = False
    expanded: Set[str] = set()

    def dfs(node: str, path: List[str], depth: int) -> None:
        nonlocal overflow
        if depth > max_depth:
            overflow = True
            return
        if node in expanded:
            return
        resolved_node = resolve_lineage_key(node, lineage_map, key_index)
        ancestors = lineage_map.get(resolved_node, []) if resolved_node else []
        for anc in ancestors:
            anc_clean = anc.strip()
            if not anc_clean:
                continue
            resolved_anc = resolve_lineage_key(anc_clean, lineage_map, key_index)
            if resolved_anc:
                candidates.add(resolved_anc)
            else:
                candidates.add(anc_clean)
            if anc_clean in path:
                start = path.index(anc_clean)
                cycles.append(path[start:] + [anc_clean])
                continue
            dfs(resolved_anc or anc_clean, path + [anc_clean], depth + 1)
        expanded.add(node)

    resolved_feature = resolve_lineage_key(feature, lineage_map, key_index)
    if resolved_feature:
        dfs(resolved_feature, [feature], 0)
    return candidates, cycles, overflow


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    split_paths = {"train": args.train}
    if args.valid:
        split_paths["valid"] = args.valid
    if args.test:
        split_paths["test"] = args.test

    try:
        headers_by_split = {}
        for name, path in split_paths.items():
            headers_by_split[name] = read_csv_header(path)
    except Exception as exc:
        add_issue(failures, "input_error", f"Failed to read split headers for '{name}'.", {"error": str(exc), "path": str(path)})
        return finish(args, failures, warnings, {}, [], [], {}, [])

    header_sets = {name: set(cols) for name, cols in headers_by_split.items()}
    union_headers = set().union(*header_sets.values())
    intersection_headers = set.intersection(*header_sets.values()) if header_sets else set()
    if union_headers != intersection_headers:
        add_issue(
            warnings,
            "column_mismatch",
            "Split files have non-identical headers.",
            {"union_count": len(union_headers), "intersection_count": len(intersection_headers)},
        )

    def_spec_path = Path(args.definition_spec).expanduser().resolve()
    lineage_path = Path(args.lineage_spec).expanduser().resolve()

    try:
        with def_spec_path.open("r", encoding="utf-8") as fh:
            definition_spec = json.load(fh)
        if not isinstance(definition_spec, dict):
            raise ValueError("Definition spec root must be object.")
    except Exception as exc:
        add_issue(
            failures,
            "invalid_definition_spec",
            "Unable to load definition spec.",
            {"path": str(def_spec_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], {}, [])

    try:
        with lineage_path.open("r", encoding="utf-8") as fh:
            lineage_raw = json.load(fh)
        if not isinstance(lineage_raw, dict):
            raise ValueError("Lineage spec root must be object.")
        lineage_map = normalize_lineage_payload(lineage_raw)
        lineage_key_index, lineage_key_collisions = build_lineage_key_index(lineage_map)
    except Exception as exc:
        add_issue(
            failures,
            "invalid_lineage_spec",
            "Unable to load feature lineage spec.",
            {"path": str(lineage_path), "error": str(exc)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], {}, [])

    if lineage_key_collisions:
        add_issue(
            failures,
            "lineage_key_normalization_collision",
            "Lineage feature keys collide after normalization; canonical naming must be unique.",
            {"collisions": lineage_key_collisions},
        )

    target_block = resolve_target_block(definition_spec, args.target)
    if target_block is None:
        add_issue(
            failures,
            "target_not_found",
            "Target not found in definition spec.",
            {"target": args.target, "path": str(def_spec_path)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], lineage_map, [])

    try:
        global_forbidden_vars = list_from(definition_spec, "global_forbidden_variables")
        global_patterns = list_from(definition_spec, "global_forbidden_patterns")
        target_defining_vars = list_from(target_block, "defining_variables")
        target_forbidden_vars = list_from(target_block, "forbidden_variables")
        target_patterns = list_from(target_block, "forbidden_patterns")
    except ValueError as exc:
        add_issue(
            failures,
            "invalid_definition_spec",
            "Definition spec fields have invalid type.",
            {"error": str(exc)},
        )
        return finish(args, failures, warnings, headers_by_split, [], [], lineage_map, [])

    forbidden_exact = global_forbidden_vars + target_defining_vars + target_forbidden_vars
    forbidden_patterns = global_patterns + target_patterns
    forbidden_exact_norm = {norm(x): x for x in forbidden_exact}
    compiled_patterns, regex_errors = compile_patterns(forbidden_patterns)
    for err in regex_errors:
        add_issue(failures, "invalid_forbidden_pattern", "Invalid forbidden regex.", {"error": err})

    ignore_cols = parse_comma_set(args.ignore_cols)
    ignore_cols.add(args.target_col)
    checked_features = sorted([h for h in union_headers if h not in ignore_cols])

    if not checked_features:
        add_issue(
            warnings,
            "no_features_checked",
            "No predictor columns were checked after applying ignore columns.",
            {"ignored_columns": sorted(ignore_cols)},
        )

    missing_lineage: List[str] = []
    lineage_cycles: List[Dict[str, Any]] = []
    lineage_depth_overflow_features: List[str] = []
    exact_hits: List[Dict[str, Any]] = []
    pattern_hits: List[Dict[str, Any]] = []

    for feature in checked_features:
        resolved_feature = resolve_lineage_key(feature, lineage_map, lineage_key_index)
        if resolved_feature is None:
            missing_lineage.append(feature)
            candidates = [feature]
        else:
            transitive_candidates, cycles, overflow = collect_transitive_candidates(
                feature,
                lineage_map,
                lineage_key_index,
            )
            candidates = sorted(transitive_candidates)
            if cycles:
                lineage_cycles.append(
                    {
                        "feature": feature,
                        "cycles": cycles,
                    }
                )
            if overflow:
                lineage_depth_overflow_features.append(feature)

        for candidate in candidates:
            c_norm = norm(candidate)
            if c_norm in forbidden_exact_norm:
                exact_hits.append(
                    {
                        "feature": feature,
                        "candidate": candidate,
                        "matched_rule": forbidden_exact_norm[c_norm],
                    }
                )
            for pattern in compiled_patterns:
                if pattern.search(candidate):
                    pattern_hits.append(
                        {
                            "feature": feature,
                            "candidate": candidate,
                            "matched_pattern": pattern.pattern,
                        }
                    )

    if exact_hits:
        add_issue(
            failures,
            "lineage_definition_leakage",
            "Detected forbidden disease-definition ancestry in feature lineage.",
            {"hits": exact_hits},
        )
    if pattern_hits:
        add_issue(
            failures,
            "lineage_proxy_leakage",
            "Detected forbidden proxy patterns in feature lineage.",
            {"hits": pattern_hits},
        )

    if lineage_cycles:
        add_issue(
            failures,
            "lineage_cycle_detected",
            "Lineage spec contains cyclic dependencies.",
            {"features_with_cycles": lineage_cycles},
        )

    if lineage_depth_overflow_features:
        add_issue(
            failures,
            "lineage_depth_overflow",
            "Lineage traversal exceeded maximum depth; lineage graph may be malformed.",
            {"features": lineage_depth_overflow_features, "max_depth": 256},
        )

    if missing_lineage:
        issue = {
            "missing_count": len(missing_lineage),
            "missing_features": missing_lineage,
        }
        if args.strict and not args.allow_missing_lineage:
            add_issue(
                failures,
                "missing_lineage_entries",
                "Missing lineage entries for checked features in strict mode.",
                issue,
            )
        else:
            add_issue(
                warnings,
                "missing_lineage_entries",
                "Missing lineage entries for checked features.",
                issue,
            )

    if args.strict and not lineage_map:
        add_issue(
            failures,
            "empty_lineage_map",
            "Lineage map is empty in strict mode.",
            {},
        )

    return finish(
        args,
        failures,
        warnings,
        headers_by_split,
        sorted(forbidden_exact),
        forbidden_patterns,
        lineage_map,
        checked_features,
        lineage_key_index,
    )


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    headers_by_split: Dict[str, List[str]],
    forbidden_exact: List[str],
    forbidden_patterns: List[str],
    lineage_map: Dict[str, List[str]],
    checked_features: List[str],
    lineage_key_index: Optional[Dict[str, str]] = None,
) -> int:
    from _gate_utils import get_gate_elapsed, write_json as _write_report

    should_fail = bool(failures) or (args.strict and bool(warnings))
    status = "fail" if should_fail else "pass"
    key_index = lineage_key_index or {}

    fi = [GateIssue.from_legacy(f, Severity.ERROR) for f in failures]
    wi = [GateIssue.from_legacy(w, Severity.WARNING) for w in warnings]
    for issue in fi + wi:
        if not issue.remediation:
            issue.remediation = get_remediation(issue.code)

    input_files = {
        "definition_spec": str(Path(args.definition_spec).expanduser().resolve()),
        "lineage_spec": str(Path(args.lineage_spec).expanduser().resolve()),
        "train": str(Path(args.train).expanduser().resolve()),
    }
    if getattr(args, "valid", None):
        input_files["valid"] = str(Path(args.valid).expanduser().resolve())
    if getattr(args, "test", None):
        input_files["test"] = str(Path(args.test).expanduser().resolve())

    report = build_report_envelope(
        gate_name="feature_lineage_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=fi,
        warnings=wi,
        summary={
            "target": args.target,
            "splits": {k: {"column_count": len(v), "columns": v} for k, v in headers_by_split.items()},
            "forbidden_exact_count": len(forbidden_exact),
            "forbidden_pattern_count": len(forbidden_patterns),
            "lineage_feature_count": len(lineage_map),
            "checked_feature_count": len(checked_features),
            "lineage_coverage_ratio": (
                0.0
                if not checked_features
                else round(
                    sum(1 for f in checked_features if (f in lineage_map or norm(f) in key_index))
                    / float(len(checked_features)),
                    6,
                )
            ),
        },
        input_files=input_files,
    )

    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), report)

    print_gate_summary(
        gate_name="feature_lineage_gate",
        status=status,
        failures=fi,
        warnings=wi,
        strict=bool(args.strict),
        elapsed=get_gate_elapsed(),
    )

    return 2 if should_fail else 0


if __name__ == "__main__":
    from _gate_utils import start_gate_timer
    start_gate_timer()
    raise SystemExit(main())
