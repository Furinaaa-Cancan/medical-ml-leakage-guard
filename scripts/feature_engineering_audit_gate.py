#!/usr/bin/env python3
"""
Fail-closed feature engineering audit gate for publication-grade medical prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from _gate_utils import add_issue, load_json_from_str as load_json, to_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate feature engineering provenance/stability/reproducibility evidence.")
    parser.add_argument("--feature-group-spec", required=True, help="Path to feature_group_spec JSON.")
    parser.add_argument("--feature-engineering-report", required=True, help="Path to feature_engineering_report JSON.")
    parser.add_argument("--lineage-spec", required=True, help="Path to feature_lineage_spec JSON.")
    parser.add_argument("--tuning-spec", required=True, help="Path to tuning_protocol_spec JSON.")
    parser.add_argument("--report", help="Optional output gate report path.")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings.")
    return parser.parse_args()


def extract_groups(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    raw = spec.get("groups")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for group_name, features in raw.items():
        if not isinstance(group_name, str) or not group_name.strip():
            continue
        if not isinstance(features, list):
            continue
        clean = [str(x).strip() for x in features if isinstance(x, str) and str(x).strip()]
        if clean:
            out[group_name.strip()] = clean
    return out


def build_feature_to_group(groups: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group_name, features in groups.items():
        for feature in features:
            if feature not in mapping:
                mapping[feature] = group_name
    return mapping


def collect_forbidden_features(lineage_payload: Dict[str, Any]) -> Set[str]:
    forbidden: Set[str] = set()
    features = lineage_payload.get("features")
    if not isinstance(features, dict):
        return forbidden
    for feature_name, payload in features.items():
        if not isinstance(feature_name, str) or not feature_name.strip():
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("forbidden_for_modeling") is True:
            forbidden.add(feature_name.strip())
            continue
        ancestors = payload.get("ancestors")
        if isinstance(ancestors, list):
            joined = " ".join(str(x).lower() for x in ancestors if isinstance(x, str))
            if any(token in joined for token in ("target", "label", "outcome", "diagnosis")):
                forbidden.add(feature_name.strip())
    return forbidden


def main() -> int:
    args = parse_args()
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    try:
        group_spec = load_json(args.feature_group_spec)
    except Exception as exc:
        add_issue(
            failures,
            "feature_group_spec_missing_or_invalid",
            "Unable to parse feature_group_spec JSON.",
            {"path": str(Path(args.feature_group_spec).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    groups = extract_groups(group_spec)
    if not groups:
        add_issue(
            failures,
            "feature_group_spec_missing_or_invalid",
            "feature_group_spec must contain non-empty groups.",
            {"path": str(Path(args.feature_group_spec).expanduser())},
        )
        return finish(args, failures, warnings, {})
    feature_to_group = build_feature_to_group(groups)

    try:
        report_payload = load_json(args.feature_engineering_report)
    except Exception as exc:
        add_issue(
            failures,
            "feature_engineering_report_invalid",
            "Unable to parse feature_engineering_report JSON.",
            {"path": str(Path(args.feature_engineering_report).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        lineage_payload = load_json(args.lineage_spec)
    except Exception as exc:
        add_issue(
            failures,
            "feature_engineering_reproducibility_missing",
            "Unable to parse lineage_spec JSON for audit.",
            {"path": str(Path(args.lineage_spec).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    try:
        tuning_payload = load_json(args.tuning_spec)
    except Exception as exc:
        add_issue(
            failures,
            "feature_engineering_reproducibility_missing",
            "Unable to parse tuning_spec JSON for audit.",
            {"path": str(Path(args.tuning_spec).expanduser()), "error": str(exc)},
        )
        return finish(args, failures, warnings, {})

    selection_scope = str(report_payload.get("selection_scope", "")).strip().lower()
    if selection_scope not in ALLOWED_SELECTION_SCOPES:
        add_issue(
            failures,
            "feature_engineering_scope_violation",
            "feature_engineering_report.selection_scope must be train_only or cv_inner_train_only.",
            {"selection_scope": selection_scope, "allowed": sorted(ALLOWED_SELECTION_SCOPES)},
        )

    scopes_used = report_payload.get("data_scopes_used")
    if not isinstance(scopes_used, list) or not scopes_used:
        add_issue(
            failures,
            "feature_engineering_scope_violation",
            "feature_engineering_report.data_scopes_used must be non-empty list.",
            {"value": scopes_used},
        )
    else:
        scope_tokens = {str(x).strip().lower() for x in scopes_used if isinstance(x, str) and str(x).strip()}
        bad_tokens = sorted(token for token in scope_tokens if token in FORBIDDEN_SCOPE_TOKENS)
        if bad_tokens:
            add_issue(
                failures,
                "feature_selection_data_leakage",
                "Feature selection/engineering must not use valid/test/external scope.",
                {"forbidden_scopes_found": bad_tokens},
            )
        if "train_only" not in scope_tokens and "cv_inner_train_only" not in scope_tokens:
            add_issue(
                failures,
                "feature_engineering_scope_violation",
                "Feature engineering evidence must include train-only scope marker.",
                {"scopes": sorted(scope_tokens)},
            )

    tuning_selection_data = str(tuning_payload.get("model_selection_data", "")).strip().lower()
    if tuning_selection_data and tuning_selection_data not in {"cv_inner", "nested_cv", "valid"}:
        add_issue(
            failures,
            "feature_engineering_reproducibility_missing",
            "tuning_protocol_spec.model_selection_data must be valid/cv_inner/nested_cv.",
            {"model_selection_data": tuning_selection_data},
        )

    selected_features_raw = report_payload.get("selected_features")
    if not isinstance(selected_features_raw, list) or not selected_features_raw:
        add_issue(
            failures,
            "feature_engineering_reproducibility_missing",
            "feature_engineering_report.selected_features must be non-empty list.",
            {"selected_features": selected_features_raw},
        )
        selected_features: List[str] = []
    else:
        selected_features = [str(x).strip() for x in selected_features_raw if isinstance(x, str) and str(x).strip()]

    missing_grouped = [f for f in selected_features if f not in feature_to_group]
    if missing_grouped:
        add_issue(
            failures,
            "feature_group_spec_missing_or_invalid",
            "Every selected feature must be declared in feature_group_spec.",
            {"missing_features": missing_grouped[:50]},
        )

    forbidden_features = collect_forbidden_features(lineage_payload)
    leaked = sorted(f for f in selected_features if f in forbidden_features)
    if leaked:
        add_issue(
            failures,
            "feature_selection_data_leakage",
            "Selected feature set includes lineage-forbidden features.",
            {"forbidden_selected_features": leaked[:50]},
        )

    stability = report_payload.get("stability")
    if not isinstance(stability, dict):
        add_issue(
            failures,
            "feature_stability_evidence_missing",
            "feature_engineering_report.stability must be an object.",
            {},
        )
        feature_freq: Dict[str, Any] = {}
        group_freq: Dict[str, Any] = {}
    else:
        feature_freq = stability.get("feature_selection_frequency")
        group_freq = stability.get("group_selection_frequency")
        if not isinstance(feature_freq, dict) or not feature_freq:
            add_issue(
                failures,
                "feature_stability_evidence_missing",
                "stability.feature_selection_frequency must be non-empty object.",
                {},
            )
            feature_freq = {}
        if not isinstance(group_freq, dict) or not group_freq:
            add_issue(
                failures,
                "feature_stability_evidence_missing",
                "stability.group_selection_frequency must be non-empty object.",
                {},
            )
            group_freq = {}

    for feature in selected_features:
        value = to_float(feature_freq.get(feature)) if isinstance(feature_freq, dict) else None
        if value is None or not (0.0 <= value <= 1.0):
            add_issue(
                failures,
                "feature_stability_evidence_missing",
                "Selected feature is missing valid selection frequency evidence.",
                {"feature": feature, "selection_frequency": feature_freq.get(feature) if isinstance(feature_freq, dict) else None},
            )

    for group_name in groups:
        value = to_float(group_freq.get(group_name)) if isinstance(group_freq, dict) else None
        if value is None or not (0.0 <= value <= 1.0):
            add_issue(
                failures,
                "feature_stability_evidence_missing",
                "Feature group is missing valid group selection frequency evidence.",
                {"group": group_name, "selection_frequency": group_freq.get(group_name) if isinstance(group_freq, dict) else None},
            )

    reproducibility = report_payload.get("reproducibility")
    required_repro_fields = ("random_seed", "cv", "selection_thresholds", "retained_feature_list", "selection_scope")
    if not isinstance(reproducibility, dict):
        add_issue(
            failures,
            "feature_engineering_reproducibility_missing",
            "feature_engineering_report.reproducibility must be an object.",
            {},
        )
    else:
        for key in required_repro_fields:
            value = reproducibility.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                add_issue(
                    failures,
                    "feature_engineering_reproducibility_missing",
                    "feature_engineering_report.reproducibility missing required field.",
                    {"field": key},
                )
        retained = reproducibility.get("retained_feature_list")
        if isinstance(retained, list):
            retained_clean = [str(x).strip() for x in retained if isinstance(x, str) and str(x).strip()]
            if sorted(retained_clean) != sorted(selected_features):
                add_issue(
                    failures,
                    "feature_engineering_reproducibility_missing",
                    "reproducibility.retained_feature_list must match selected_features.",
                    {"retained_count": len(retained_clean), "selected_count": len(selected_features)},
                )

    summary = {
        "feature_group_spec": str(Path(args.feature_group_spec).expanduser().resolve()),
        "feature_engineering_report": str(Path(args.feature_engineering_report).expanduser().resolve()),
        "lineage_spec": str(Path(args.lineage_spec).expanduser().resolve()),
        "tuning_spec": str(Path(args.tuning_spec).expanduser().resolve()),
        "selection_scope": selection_scope,
        "selected_feature_count": len(selected_features),
        "group_count": len(groups),
    }
    return finish(args, failures, warnings, summary)


def finish(
    args: argparse.Namespace,
    failures: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> int:
    should_fail = bool(failures) or (args.strict and bool(warnings))
    report = {
        "status": "fail" if should_fail else "pass",
        "strict_mode": bool(args.strict),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures,
        "warnings": warnings,
        "summary": summary,
    }
    if args.report:
        out = Path(args.report).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)
    print(f"Status: {report['status']}")
    print(f"Failures: {len(failures)} | Warnings: {len(warnings)} | Strict: {args.strict}")
    for issue in failures:
        print(f"[FAIL] {issue['code']}: {issue['message']}")
    for issue in warnings:
        print(f"[WARN] {issue['code']}: {issue['message']}")
    return 2 if should_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
