#!/usr/bin/env python3
"""
Performance Policy Generator for ML Leakage Guard.

Scans gate reports and evaluation artifacts in an evidence directory,
extracts observed metric values, and generates a recommended
``performance_policy.json`` with sensible thresholds.

Users can review and customize the generated policy before committing
it to their project.

Usage:
    python3 scripts/policy_generator.py --evidence-dir evidence/
    python3 scripts/policy_generator.py --evidence-dir evidence/ --margin 0.10
    python3 scripts/policy_generator.py --evidence-dir evidence/ --output policy.json
    python3 scripts/policy_generator.py --evidence-dir evidence/ --preset strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Defaults ────────────────────────────────────────────────────────────────

_DEFAULT_MARGIN = 0.15  # 15% headroom above observed worst case

_PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "lenient": {
        "description": "Relaxed thresholds for early-stage development",
        "margin_factor": 0.30,
    },
    "standard": {
        "description": "Balanced thresholds for pre-publication review",
        "margin_factor": 0.15,
    },
    "strict": {
        "description": "Tight thresholds for clinical deployment readiness",
        "margin_factor": 0.05,
    },
}


# ── JSON loader ─────────────────────────────────────────────────────────────

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on missing/invalid."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _is_finite(value: Any) -> bool:
    """Return True if value is a finite number (not bool)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _safe_get(data: Dict[str, Any], *keys: str) -> Optional[float]:
    """Walk nested dict by keys, return float or None."""
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    if _is_finite(current):
        return float(current)
    return None


# ── Metric extractors ──────────────────────────────────────────────────────

def extract_eval_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract key metrics from evaluation_report.json."""
    report = load_json(evidence_dir / "evaluation_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    # Try top-level metrics
    m = report.get("metrics", {})
    if isinstance(m, dict):
        for key in ["roc_auc", "pr_auc", "brier", "f2_beta", "accuracy"]:
            val = m.get(key)
            if _is_finite(val):
                metrics[key] = float(val)
    # Try split_metrics.test.metrics
    sm = report.get("split_metrics", {})
    if isinstance(sm, dict):
        test_m = sm.get("test", {})
        if isinstance(test_m, dict):
            inner = test_m.get("metrics", test_m)
            if isinstance(inner, dict):
                for key in ["roc_auc", "pr_auc", "brier", "f2_beta", "accuracy"]:
                    if key not in metrics:
                        val = inner.get(key)
                        if _is_finite(val):
                            metrics[key] = float(val)
    return metrics


def extract_robustness_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract robustness summary metrics."""
    report = load_json(evidence_dir / "robustness_gate_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    for bucket in ["time_slices", "patient_hash_groups"]:
        drop = _safe_get(report, "summary", "computed", bucket, "pr_auc_worst_drop_from_overall")
        rng = _safe_get(report, "summary", "computed", bucket, "pr_auc_range")
        if drop is not None:
            metrics[f"{bucket}_pr_auc_drop"] = drop
        if rng is not None:
            metrics[f"{bucket}_pr_auc_range"] = rng
    return metrics


def extract_generalization_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract generalization gap metrics."""
    report = load_json(evidence_dir / "generalization_gap_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    gap = _safe_get(report, "summary", "train_test_auc_gap")
    if gap is not None:
        metrics["train_test_auc_gap"] = gap
    return metrics


def extract_calibration_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract calibration metrics."""
    report = load_json(evidence_dir / "calibration_dca_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    brier = _safe_get(report, "summary", "brier_score")
    if brier is not None:
        metrics["brier_score"] = brier
    hl_p = _safe_get(report, "summary", "hosmer_lemeshow_p_value")
    if hl_p is not None:
        metrics["hosmer_lemeshow_p"] = hl_p
    return metrics


def extract_seed_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract seed stability metrics."""
    report = load_json(evidence_dir / "seed_stability_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    cv = _safe_get(report, "summary", "pr_auc_cv")
    if cv is not None:
        metrics["pr_auc_cv"] = cv
    return metrics


def extract_external_metrics(evidence_dir: Path) -> Dict[str, float]:
    """Extract external validation transport metrics."""
    report = load_json(evidence_dir / "external_validation_gate_report.json")
    if report is None:
        return {}
    metrics: Dict[str, float] = {}
    cohorts = report.get("summary", {}).get("replayed_cohorts", [])
    if isinstance(cohorts, list):
        drops = []
        for c in cohorts:
            if isinstance(c, dict):
                tg = c.get("transport_gap", {})
                if isinstance(tg, dict):
                    d = tg.get("pr_auc_drop_from_internal_test")
                    if _is_finite(d):
                        drops.append(float(d))
        if drops:
            metrics["max_cohort_pr_auc_drop"] = max(drops)
    return metrics


# ── Threshold derivation ───────────────────────────────────────────────────

def _apply_margin_lower(observed: float, margin: float) -> float:
    """For 'lower is better' metrics: threshold = observed * (1 + margin)."""
    return round(observed * (1.0 + margin), 4)


def _apply_margin_higher(observed: float, margin: float) -> float:
    """For 'higher is better' metrics: threshold = observed * (1 - margin)."""
    return round(observed * (1.0 - margin), 4)


def derive_policy(
    evidence_dir: Path,
    margin: float = _DEFAULT_MARGIN,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Derive a recommended performance policy from evidence.

    Returns (policy_dict, observed_metrics_dict).
    """
    eval_m = extract_eval_metrics(evidence_dir)
    rob_m = extract_robustness_metrics(evidence_dir)
    gen_m = extract_generalization_metrics(evidence_dir)
    cal_m = extract_calibration_metrics(evidence_dir)
    seed_m = extract_seed_metrics(evidence_dir)
    ext_m = extract_external_metrics(evidence_dir)

    observed = {**eval_m, **rob_m, **gen_m, **cal_m, **seed_m, **ext_m}

    policy: Dict[str, Any] = {
        "_generator": "policy_generator.py",
        "_margin": margin,
    }

    # Evaluation metric floors (higher is better → subtract margin)
    eval_floors: Dict[str, float] = {}
    for key in ["roc_auc", "pr_auc", "f2_beta"]:
        if key in eval_m:
            eval_floors[f"min_{key}"] = _apply_margin_higher(eval_m[key], margin)
    if "brier" in eval_m:
        eval_floors["max_brier"] = _apply_margin_lower(eval_m["brier"], margin)
    if eval_floors:
        policy["evaluation_metric_floors"] = eval_floors

    # Robustness thresholds (lower is better → add margin)
    rob_thresholds: Dict[str, Any] = {}
    for bucket in ["time_slices", "patient_hash_groups"]:
        bucket_t: Dict[str, float] = {}
        drop_key = f"{bucket}_pr_auc_drop"
        rng_key = f"{bucket}_pr_auc_range"
        if drop_key in rob_m:
            bucket_t["pr_auc_drop_fail"] = _apply_margin_lower(rob_m[drop_key], margin)
            bucket_t["pr_auc_drop_warn"] = _apply_margin_lower(rob_m[drop_key], margin * 0.5)
        if rng_key in rob_m:
            bucket_t["pr_auc_range_fail"] = _apply_margin_lower(rob_m[rng_key], margin)
            bucket_t["pr_auc_range_warn"] = _apply_margin_lower(rob_m[rng_key], margin * 0.5)
        if bucket_t:
            rob_thresholds[bucket] = bucket_t
    if rob_thresholds:
        policy["robustness_thresholds"] = rob_thresholds

    # Generalization gap
    if "train_test_auc_gap" in gen_m:
        policy["generalization_gap_thresholds"] = {
            "max_train_test_auc_gap": _apply_margin_lower(gen_m["train_test_auc_gap"], margin),
        }

    # Calibration
    cal_t: Dict[str, float] = {}
    if "brier_score" in cal_m:
        cal_t["max_brier_score"] = _apply_margin_lower(cal_m["brier_score"], margin)
    if "hosmer_lemeshow_p" in cal_m:
        cal_t["min_hl_p_value"] = _apply_margin_higher(cal_m["hosmer_lemeshow_p"], margin)
    if cal_t:
        policy["calibration_thresholds"] = cal_t

    # Seed stability
    if "pr_auc_cv" in seed_m:
        policy["seed_stability_thresholds"] = {
            "max_pr_auc_cv": _apply_margin_lower(seed_m["pr_auc_cv"], margin),
        }

    # External validation
    if "max_cohort_pr_auc_drop" in ext_m:
        policy["external_validation_thresholds"] = {
            "max_pr_auc_drop": _apply_margin_lower(ext_m["max_cohort_pr_auc_drop"], margin),
        }

    return policy, observed


# ── Output formatting ──────────────────────────────────────────────────────

def to_text(policy: Dict[str, Any], observed: Dict[str, Any], margin: float) -> str:
    """Render as human-readable text."""
    lines: List[str] = []
    lines.append("=== Generated Performance Policy ===")
    lines.append(f"Margin: {margin * 100:.0f}%")
    lines.append(f"Observed metrics: {len(observed)}")
    lines.append(f"Policy sections: {len([k for k in policy if not k.startswith('_')])}")
    lines.append("")

    if observed:
        lines.append("--- Observed Metrics ---")
        for k, v in sorted(observed.items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    for section_key in sorted(policy.keys()):
        if section_key.startswith("_"):
            continue
        lines.append(f"--- {section_key} ---")
        section = policy[section_key]
        if isinstance(section, dict):
            for k, v in sorted(section.items()):
                if isinstance(v, dict):
                    lines.append(f"  [{k}]")
                    for kk, vv in sorted(v.items()):
                        lines.append(f"    {kk}: {vv}")
                else:
                    lines.append(f"  {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a recommended performance_policy.json from evidence.",
    )
    parser.add_argument(
        "--evidence-dir", required=True,
        help="Path to the evidence directory containing gate reports.",
    )
    parser.add_argument(
        "--margin", type=float, default=None,
        help=f"Headroom margin fraction (default: {_DEFAULT_MARGIN}).",
    )
    parser.add_argument(
        "--preset", choices=list(_PRESET_PROFILES.keys()),
        help="Use a named preset profile (overrides --margin).",
    )
    parser.add_argument(
        "--output", help="Write policy JSON to file (default: stdout).",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Output human-readable text summary instead of JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    evidence_dir = Path(args.evidence_dir).expanduser().resolve()

    if not evidence_dir.is_dir():
        print(f"Error: evidence directory not found: {evidence_dir}", file=sys.stderr)
        return 1

    # Determine margin
    if args.preset:
        profile = _PRESET_PROFILES[args.preset]
        margin = float(profile["margin_factor"])
    elif args.margin is not None:
        margin = float(args.margin)
    else:
        margin = _DEFAULT_MARGIN

    policy, observed = derive_policy(evidence_dir, margin=margin)

    if args.preset:
        policy["_preset"] = args.preset
        policy["_preset_description"] = _PRESET_PROFILES[args.preset]["description"]

    if args.text:
        output = to_text(policy, observed, margin)
    else:
        output = json.dumps(policy, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Policy written to: {out_path}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
