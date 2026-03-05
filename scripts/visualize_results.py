#!/usr/bin/env python3
"""
Result Visualization Tool for ML Leakage Guard.

Generates publication-quality plots from evaluation_report.json and
prediction_trace.csv: ROC curve, PR curve, calibration curve,
decision curve analysis (DCA), and feature importance top-20.

Usage:
    python3 scripts/visualize_results.py \
        --evaluation-report evidence/evaluation_report.json \
        --prediction-trace evidence/prediction_trace.csv \
        --output-dir evidence/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1)

try:
    import pandas as pd
except ImportError:
    print("pandas is required: pip install pandas", file=sys.stderr)
    raise SystemExit(1)

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ── Style ──────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
}
DPI = 150
PRIMARY_COLOR = "#38bdf8"
SECONDARY_COLOR = "#a78bfa"
ACCENT_COLOR = "#22c55e"
WARN_COLOR = "#f97316"


def _apply_style() -> None:
    """Apply the dark theme to matplotlib."""
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate result visualization plots.")
    parser.add_argument("--evaluation-report", required=True, help="Path to evaluation_report.json.")
    parser.add_argument("--prediction-trace", help="Path to prediction_trace.csv.")
    parser.add_argument("--output-dir", default="evidence/plots", help="Output directory for PNG files.")
    parser.add_argument("--dpi", type=int, default=DPI, help="Plot DPI (default: 150).")
    return parser.parse_args()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out: Path, dpi: int) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PRIMARY_COLOR, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(out / "roc_curve.png"), dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out / 'roc_curve.png'}")


def plot_pr(y_true: np.ndarray, y_score: np.ndarray, out: Path, dpi: int) -> None:
    """Generate and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=SECONDARY_COLOR, lw=2, label=f"PR (AP = {ap:.3f})")
    ax.axhline(y=prevalence, color="#475569", lw=1, linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(out / "pr_curve.png"), dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out / 'pr_curve.png'}")


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, out: Path, dpi: int) -> None:
    """Generate and save calibration curve plot."""
    n_bins = 10
    try:
        fraction_pos, mean_predicted = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    except ValueError:
        print("  [SKIP] Calibration curve: insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_predicted, fraction_pos, color=ACCENT_COLOR, lw=2, marker="o", markersize=5, label="Model")
    ax.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(out / "calibration_curve.png"), dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out / 'calibration_curve.png'}")


def plot_dca(y_true: np.ndarray, y_score: np.ndarray, out: Path, dpi: int) -> None:
    """Generate and save Decision Curve Analysis plot."""
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    net_benefit_model = []
    net_benefit_all = []
    for t in thresholds:
        tp = int(np.sum((y_score >= t) & (y_true == 1)))
        fp = int(np.sum((y_score >= t) & (y_true == 0)))
        nb = (tp / n) - (fp / n) * (t / (1 - t)) if t < 1 else 0
        net_benefit_model.append(nb)
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1 else 0
        net_benefit_all.append(nb_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, net_benefit_model, color=PRIMARY_COLOR, lw=2, label="Model")
    ax.plot(thresholds, net_benefit_all, color=WARN_COLOR, lw=1, linestyle="--", label="Treat All")
    ax.axhline(y=0, color="#475569", lw=1, linestyle=":", label="Treat None")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(out / "dca_curve.png"), dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out / 'dca_curve.png'}")


def plot_feature_importance(
    report: Dict[str, Any], out: Path, dpi: int, top_k: int = 20
) -> None:
    """Generate and save feature importance bar chart from evaluation report."""
    fi = report.get("feature_importance")
    if not isinstance(fi, (dict, list)) or not fi:
        fi = report.get("feature_importances")
    if not isinstance(fi, (dict, list)) or not fi:
        print("  [SKIP] Feature importance: not available in report.")
        return

    if isinstance(fi, list):
        items = [(d.get("feature", f"f{i}"), float(d.get("importance", 0))) for i, d in enumerate(fi)]
    else:
        items = [(k, float(v)) for k, v in fi.items()]

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    items = items[:top_k]
    items.reverse()

    names = [x[0] for x in items]
    values = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    colors = [PRIMARY_COLOR if v >= 0 else WARN_COLOR for v in values]
    ax.barh(range(len(names)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance (Top {len(names)})")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(str(out / "feature_importance.png"), dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out / 'feature_importance.png'}")


def main() -> int:
    """Entry point for the visualization tool."""
    args = parse_args()
    _apply_style()

    report_path = Path(args.evaluation_report).expanduser().resolve()
    if not report_path.exists():
        print(f"Evaluation report not found: {report_path}", file=sys.stderr)
        return 1

    try:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in evaluation report: {exc}", file=sys.stderr)
        return 1
    if not isinstance(report, dict):
        print(f"Evaluation report is not a JSON object.", file=sys.stderr)
        return 1

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dpi = int(args.dpi)

    print(f"Output directory: {out}")

    # Try to load prediction trace for curve plots
    y_true: Optional[np.ndarray] = None
    y_score: Optional[np.ndarray] = None

    if args.prediction_trace:
        trace_path = Path(args.prediction_trace).expanduser().resolve()
        if trace_path.exists():
            df = pd.read_csv(trace_path)
            y_col = None
            score_col = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("y_true", "label", "target", "y"):
                    y_col = c
                if cl in ("y_score", "y_prob", "probability", "score", "predicted_probability"):
                    score_col = c
            if y_col and score_col:
                y_true = df[y_col].values.astype(float)
                y_score = df[score_col].values.astype(float)
                mask = np.isfinite(y_true) & np.isfinite(y_score)
                y_true = y_true[mask]
                y_score = y_score[mask]
                print(f"Loaded prediction trace: {len(y_true)} rows")
            else:
                print(f"  [WARN] Could not identify y_true/y_score columns in trace.")
        else:
            print(f"  [WARN] Prediction trace not found: {trace_path}")

    if y_true is not None and y_score is not None and len(y_true) > 10:
        plot_roc(y_true, y_score, out, dpi)
        plot_pr(y_true, y_score, out, dpi)
        plot_calibration(y_true, y_score, out, dpi)
        plot_dca(y_true, y_score, out, dpi)
    else:
        print("  [SKIP] Curve plots: no valid prediction trace available.")

    plot_feature_importance(report, out, dpi)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
