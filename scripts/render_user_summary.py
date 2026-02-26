#!/usr/bin/env python3
"""
Render a user-facing summary (Markdown + JSON) from strict evidence artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_GATE_FILES = {
    "strict_pipeline": "strict_pipeline_report.json",
    "publication_gate": "publication_gate_report.json",
    "self_critique": "self_critique_report.json",
    "clinical_metrics": "clinical_metrics_report.json",
    "generalization_gap": "generalization_gap_report.json",
    "external_validation_gate": "external_validation_gate_report.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render user-friendly run summary from evidence artifacts.")
    parser.add_argument("--evidence-dir", required=True, help="Evidence directory containing gate reports.")
    parser.add_argument("--request", help="Optional request JSON path for study metadata.")
    parser.add_argument("--out-markdown", help="Output markdown path (default: evidence/user_summary.md).")
    parser.add_argument("--out-json", help="Output JSON path (default: evidence/user_summary.json).")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)
        fh.write("\n")
    tmp_path.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def summarize_gate(name: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"name": name, "status": "missing", "failure_count": None, "warning_count": None}
    return {
        "name": name,
        "status": str(payload.get("status", "unknown")),
        "failure_count": payload.get("failure_count"),
        "warning_count": payload.get("warning_count"),
    }


def get_top_failure_codes(payload: Optional[Dict[str, Any]], limit: int = 5) -> List[str]:
    if not isinstance(payload, dict):
        return []
    failures = payload.get("failures")
    if not isinstance(failures, list):
        return []
    out: List[str] = []
    for row in failures:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code", "")).strip()
        if code and code not in out:
            out.append(code)
        if len(out) >= limit:
            break
    return out


def extract_metrics(evaluation_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(evaluation_report, dict):
        return {}
    metrics = evaluation_report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics.get(key)
        for key in ("pr_auc", "roc_auc", "brier", "sensitivity", "specificity", "ppv", "npv", "f2_beta")
    }


def extract_gap_rows(gap_report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(gap_report, dict):
        return []
    summary = gap_report.get("summary")
    if not isinstance(summary, dict):
        return []
    rows = summary.get("gaps")
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "pair": f"{row.get('left_split')}->{row.get('right_split')}",
                "metric": row.get("metric"),
                "gap": row.get("directional_gap"),
                "warn": row.get("warn_threshold"),
                "fail": row.get("fail_threshold"),
            }
        )
    return out


def extract_external_summary(external_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(external_report, dict):
        return {"cohort_count": 0, "cohorts": []}
    cohorts = external_report.get("cohorts")
    if not isinstance(cohorts, list):
        cohorts = []
    out = []
    for row in cohorts:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        out.append(
            {
                "cohort_id": row.get("cohort_id"),
                "cohort_type": row.get("cohort_type"),
                "rows": row.get("row_count"),
                "events": row.get("positive_count"),
                "pr_auc": metrics.get("pr_auc"),
                "f2_beta": metrics.get("f2_beta"),
                "brier": metrics.get("brier"),
            }
        )
    return {"cohort_count": int(len(out)), "cohorts": out}


def to_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# ML Leakage Guard User Summary")
    lines.append("")
    lines.append(f"- Generated at: `{summary['generated_at_utc']}`")
    lines.append(f"- Study: `{summary.get('study_id')}`")
    lines.append(f"- Run: `{summary.get('run_id')}`")
    lines.append(f"- Overall status: `{summary.get('overall_status')}`")
    lines.append(f"- Publication gate: `{summary.get('publication_status')}`")
    lines.append(f"- Self-critique: `{summary.get('self_critique_status')}` (score: `{summary.get('self_critique_score')}`)")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- Selected model: `{summary.get('selected_model_id')}`")
    lines.append(f"- Primary metric: `{summary.get('primary_metric')}`")
    lines.append("")
    lines.append("## Key Test Metrics")
    metrics = summary.get("test_metrics", {})
    if isinstance(metrics, dict) and metrics:
        for key, value in metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No evaluation metrics found.")
    lines.append("")
    lines.append("## Generalization Gaps")
    gap_rows = summary.get("gap_rows", [])
    if isinstance(gap_rows, list) and gap_rows:
        for row in gap_rows:
            lines.append(
                f"- `{row.get('pair')}` `{row.get('metric')}` gap=`{row.get('gap')}` "
                f"(warn=`{row.get('warn')}`, fail=`{row.get('fail')}`)"
            )
    else:
        lines.append("- Gap report missing.")
    lines.append("")
    lines.append("## External Cohorts")
    external = summary.get("external_validation", {})
    if isinstance(external, dict) and external.get("cohort_count", 0) > 0:
        lines.append(f"- Cohort count: `{external.get('cohort_count')}`")
        for row in external.get("cohorts", []):
            lines.append(
                f"- `{row.get('cohort_id')}` ({row.get('cohort_type')}): "
                f"rows=`{row.get('rows')}`, events=`{row.get('events')}`, "
                f"pr_auc=`{row.get('pr_auc')}`, f2_beta=`{row.get('f2_beta')}`, brier=`{row.get('brier')}`"
            )
    else:
        lines.append("- External validation report missing.")
    lines.append("")
    lines.append("## Gate Health")
    for gate in summary.get("gate_status", []):
        lines.append(
            f"- `{gate.get('name')}`: `{gate.get('status')}` "
            f"(failures=`{gate.get('failure_count')}`, warnings=`{gate.get('warning_count')}`)"
        )
    lines.append("")
    lines.append("## Next Actions")
    next_actions = summary.get("next_actions", [])
    if next_actions:
        for idx, item in enumerate(next_actions, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("1. No blocking actions. Ready for publication-grade submission workflow.")
    lines.append("")
    return "\n".join(lines)


def derive_next_actions(summary: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    if summary.get("overall_status") != "pass":
        actions.append("Fix failing gates listed below and rerun strict pipeline.")
    failing_gates = [g for g in summary.get("gate_status", []) if str(g.get("status")) == "fail"]
    for gate in failing_gates[:3]:
        gate_name = str(gate.get("name"))
        codes = summary.get("top_failure_codes", {}).get(gate_name, [])
        if codes:
            actions.append(f"{gate_name}: address failure codes {', '.join(codes)}.")
        else:
            actions.append(f"{gate_name}: inspect report details and remediate.")
    if not actions:
        actions.append("Archive artifacts and lock manifest baseline for reproducibility.")
    return actions


def main() -> int:
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()
    request_payload = load_json(Path(args.request).expanduser().resolve()) if args.request else None

    strict_pipeline = load_json(evidence_dir / "strict_pipeline_report.json")
    publication = load_json(evidence_dir / "publication_gate_report.json")
    self_critique = load_json(evidence_dir / "self_critique_report.json")
    evaluation = load_json(evidence_dir / "evaluation_report.json")
    external_validation = load_json(evidence_dir / "external_validation_report.json")
    gap_report = load_json(evidence_dir / "generalization_gap_report.json")

    gate_payloads = {
        key: load_json(evidence_dir / file_name) for key, file_name in DEFAULT_GATE_FILES.items()
    }
    gate_status = [summarize_gate(name, payload) for name, payload in gate_payloads.items()]
    top_failure_codes = {name: get_top_failure_codes(payload) for name, payload in gate_payloads.items()}

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "study_id": request_payload.get("study_id") if isinstance(request_payload, dict) else None,
        "run_id": request_payload.get("run_id") if isinstance(request_payload, dict) else None,
        "overall_status": str((strict_pipeline or {}).get("status", "missing")),
        "publication_status": str((publication or {}).get("status", "missing")),
        "self_critique_status": str((self_critique or {}).get("status", "missing")),
        "self_critique_score": (self_critique or {}).get("quality_score"),
        "selected_model_id": (evaluation or {}).get("model_id"),
        "primary_metric": (evaluation or {}).get("primary_metric"),
        "test_metrics": extract_metrics(evaluation),
        "gap_rows": extract_gap_rows(gap_report),
        "external_validation": extract_external_summary(external_validation),
        "gate_status": gate_status,
        "top_failure_codes": top_failure_codes,
    }
    summary["next_actions"] = derive_next_actions(summary)

    markdown = to_markdown(summary)
    out_md = (
        Path(args.out_markdown).expanduser().resolve()
        if args.out_markdown
        else (evidence_dir / "user_summary.md").resolve()
    )
    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else (evidence_dir / "user_summary.json").resolve()
    )
    write_text(out_md, markdown)
    write_json(out_json, summary)

    print("Status: pass")
    print(f"UserSummaryMarkdown: {out_md}")
    print(f"UserSummaryJSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
