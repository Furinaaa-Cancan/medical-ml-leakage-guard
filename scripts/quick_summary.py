#!/usr/bin/env python3
"""
Quick Results Summary — one-command view of MLGG training output.

Reads evaluation_report.json, model_selection_report.json, and ci_matrix_report.json
from an output directory and prints a concise, color-coded terminal summary.

Usage:
    python3 scripts/quick_summary.py /path/to/output_dir
    python3 scripts/quick_summary.py --evidence /path/to/evidence/
    python3 scripts/quick_summary.py --eval evaluation_report.json

Examples:
    python3 scripts/quick_summary.py ~/Desktop/MLGG_Output/breast_cancer
    python3 scripts/quick_summary.py --json ~/Desktop/MLGG_Output/heart_disease
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── ANSI helpers ─────────────────────────────────────────────────────────────

_COLORS = {
    "R": "\033[91m", "G": "\033[92m", "Y": "\033[93m",
    "C": "\033[96m", "W": "\033[97m", "D": "\033[90m",
    "B": "\033[1m", "RST": "\033[0m",
}


def _s(color: str, text: str, bold: bool = False) -> str:
    prefix = _COLORS.get("B", "") if bold else ""
    return f"{prefix}{_COLORS.get(color, '')}{text}{_COLORS['RST']}"


def _box(title: str, lines: List[str], color: str = "C") -> None:
    max_w = max((len(l) for l in lines), default=20)
    max_w = max(max_w, len(title) + 4)
    border = "─" * (max_w + 2)
    print(f"  ┌{border}┐")
    print(f"  │ {_s(color, title, bold=True)}{' ' * (max_w - len(title))} │")
    print(f"  ├{border}┤")
    for line in lines:
        pad = max_w - len(line)
        print(f"  │ {line}{' ' * max(pad, 0)} │")
    print(f"  └{border}┘")


# ── JSON helpers ─────────────────────────────────────────────────────────────

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def fmt_metric(val: Any, ci: Any = None) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    text = f"{v:.4f}"
    if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None:
        try:
            text += f"  [{float(ci[0]):.4f}-{float(ci[1]):.4f}]"
        except (TypeError, ValueError):
            pass
    return text


def fmt_pct(val: Any) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)


# ── Main summary logic ───────────────────────────────────────────────────────

def print_key_metrics(eval_data: Dict[str, Any]) -> None:
    metrics = eval_data.get("metrics", {})
    unc = eval_data.get("uncertainty", {}).get("metrics", {})
    model_id = eval_data.get("model_id", "?")
    thresh_block = eval_data.get("threshold_selection", {})
    threshold = thresh_block.get("selected_threshold")

    lines = [f"  {'Model':<16} {model_id}"]
    if threshold is not None:
        lines.append(f"  {'Threshold':<16} {float(threshold):.4f}")
    lines.append("")

    for key, label in [
        ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
        ("f1", "F1"), ("f2_beta", "F-beta"),
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
        ("ppv", "PPV"), ("npv", "NPV"),
        ("brier", "Brier"),
    ]:
        val = metrics.get(key)
        if val is not None:
            ci = unc.get(key, {}).get("ci_95")
            lines.append(f"  {label:<16} {fmt_metric(val, ci)}")

    unc_method = eval_data.get("uncertainty", {}).get("method", "")
    unc_n = eval_data.get("uncertainty", {}).get("n_resamples", 0)
    if unc_method and unc_n:
        lines.append("")
        lines.append(f"  95% CI: {unc_method}, n={unc_n}")

    constraints = thresh_block.get("constraints_satisfied_overall")
    if constraints is not None:
        status = _s("G", "PASS") if constraints else _s("R", "FAIL")
        lines.append(f"  Constraints     {status}")

    _box("Key Metrics (Test Set)", lines)


def print_overfitting(eval_data: Dict[str, Any]) -> None:
    split_metrics = eval_data.get("split_metrics", {})
    if not isinstance(split_metrics, dict):
        return

    rows = []
    for key in ["pr_auc", "roc_auc", "f1", "brier"]:
        vals = {}
        for split_name in ["train", "valid", "test"]:
            block = split_metrics.get(split_name, {})
            m = block.get("metrics", {}) if isinstance(block, dict) else {}
            vals[split_name] = m.get(key)
        if vals.get("train") is not None and vals.get("test") is not None:
            gap = float(vals["train"]) - float(vals["test"])
            gap_str = f"{gap:+.4f}" if abs(gap) > 0.0001 else "+0.0000"
            row = f"  {key:<16}"
            for s_name in ["train", "valid", "test"]:
                v = vals.get(s_name)
                row += f"  {float(v):.4f}" if v is not None else "       —"
            row += f"  {gap_str}"
            rows.append(row)

    if rows:
        header = f"  {'':16}  {'Train':>7}  {'Valid':>7}  {'Test':>7}  {'Gap':>8}"
        lines = [header] + rows

        # Risk assessment
        test_pr = split_metrics.get("test", {}).get("metrics", {}).get("pr_auc")
        train_pr = split_metrics.get("train", {}).get("metrics", {}).get("pr_auc")
        if test_pr is not None and train_pr is not None:
            gap = float(train_pr) - float(test_pr)
            if gap > 0.10:
                lines.append("")
                lines.append(f"  Risk: {_s('R', 'HIGH')} — Overfitting detected (gap={gap:.4f})")
            elif gap > 0.05:
                lines.append("")
                lines.append(f"  Risk: {_s('Y', 'MODERATE')} — Some overfitting (gap={gap:.4f})")
            else:
                lines.append("")
                lines.append(f"  Risk: {_s('G', 'LOW')} — No overfitting detected")

        _box("Train / Valid / Test", lines)


def print_model_selection(ms_data: Dict[str, Any]) -> None:
    candidates = ms_data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return

    selected = ms_data.get("selected_model_id", "?")
    metric = ms_data.get("selection_metric", "pr_auc")
    lines = [
        f"  Candidates     {len(candidates)}",
        f"  Selected       {selected}",
        f"  Metric         {metric}",
        "",
    ]

    header = f"  {'#':>3} {'Model':<35} {'Mean':>8} {'Std':>8}"
    lines.append(header)

    top_n = min(10, len(candidates))
    for i, c in enumerate(candidates[:top_n]):
        mid = str(c.get("model_id", "?"))
        if len(mid) > 33:
            mid = mid[:30] + "..."
        mean = c.get("mean_score", c.get("score"))
        std = c.get("std_score", c.get("std"))
        marker = " *" if str(c.get("model_id")) == selected else ""
        mean_s = f"{float(mean):.4f}" if mean is not None else "    —"
        std_s = f"{float(std):.4f}" if std is not None else "    —"
        lines.append(f"  {i+1:>3} {mid:<35} {mean_s:>8} {std_s:>8}{marker}")

    if len(candidates) > top_n:
        lines.append(f"  ... and {len(candidates) - top_n} more")
    lines.append("")
    lines.append(f"  * = selected")

    _box(f"Model Selection ({len(candidates)} candidates)", lines)


def print_file_list(evidence_dir: Path) -> None:
    files = sorted(evidence_dir.glob("*.json")) + sorted(evidence_dir.glob("*.csv")) + sorted(evidence_dir.glob("*.sh"))
    if not files:
        return
    lines = [f"  {f.name}" for f in files[:15]]
    if len(files) > 15:
        lines.append(f"  ... and {len(files) - 15} more")
    _box("Evidence Files", lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick summary of MLGG training results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", nargs="?", default="",
                        help="MLGG output directory (contains evidence/ subfolder).")
    parser.add_argument("--evidence", default="",
                        help="Direct path to evidence directory.")
    parser.add_argument("--eval", default="",
                        help="Direct path to evaluation_report.json.")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted summary.")
    args = parser.parse_args()

    # Resolve evidence directory
    evidence_dir: Optional[Path] = None
    eval_path: Optional[Path] = None

    if args.eval:
        eval_path = Path(args.eval).expanduser().resolve()
        evidence_dir = eval_path.parent
    elif args.evidence:
        evidence_dir = Path(args.evidence).expanduser().resolve()
    elif args.output_dir:
        base = Path(args.output_dir).expanduser().resolve()
        if (base / "evidence").is_dir():
            evidence_dir = base / "evidence"
        elif base.is_dir():
            evidence_dir = base
    else:
        print("Error: provide output_dir, --evidence, or --eval", file=sys.stderr)
        return 1

    if evidence_dir is None or not evidence_dir.is_dir():
        print(f"Error: evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 1

    if eval_path is None:
        eval_path = evidence_dir / "evaluation_report.json"

    eval_data = load_json(eval_path)
    ms_data = load_json(evidence_dir / "model_selection_report.json")

    if eval_data is None:
        print(f"Error: evaluation_report.json not found in {evidence_dir}", file=sys.stderr)
        return 1

    if args.json:
        summary = {
            "model_id": eval_data.get("model_id"),
            "metrics": eval_data.get("metrics"),
            "split_metrics": eval_data.get("split_metrics"),
        }
        if ms_data:
            summary["candidates"] = len(ms_data.get("candidates", []))
            summary["selected"] = ms_data.get("selected_model_id")
        print(json.dumps(summary, indent=2))
        return 0

    print()
    print_key_metrics(eval_data)
    print()
    print_overfitting(eval_data)
    print()
    if ms_data:
        print_model_selection(ms_data)
        print()
    print_file_list(evidence_dir)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
