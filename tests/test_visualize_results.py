"""Unit tests for scripts/visualize_results.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import visualize_results as viz


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _make_report(tmp_path: Path, extra: Dict[str, Any] | None = None) -> Path:
    report = {"status": "pass", "metrics": {"roc_auc": 0.85}}
    if extra:
        report.update(extra)
    p = tmp_path / "evaluation_report.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


def _make_trace(tmp_path: Path, n: int = 100) -> Path:
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=n)
    y_score = np.clip(y_true * 0.6 + rng.normal(0.3, 0.15, n), 0, 1)
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
    p = tmp_path / "prediction_trace.csv"
    df.to_csv(p, index=False)
    return p


# ────────────────────────────────────────────────────────
# plot_roc
# ────────────────────────────────────────────────────────

class TestPlotRoc:
    def test_creates_png(self, tmp_path):
        rng = np.random.RandomState(1)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 3)
        y_score = np.clip(y_true * 0.7 + rng.normal(0.2, 0.1, len(y_true)), 0, 1)
        viz.plot_roc(y_true, y_score, tmp_path, dpi=72)
        assert (tmp_path / "roc_curve.png").exists()
        assert (tmp_path / "roc_curve.png").stat().st_size > 100


# ────────────────────────────────────────────────────────
# plot_pr
# ────────────────────────────────────────────────────────

class TestPlotPr:
    def test_creates_png(self, tmp_path):
        rng = np.random.RandomState(2)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0] * 3)
        y_score = np.clip(y_true * 0.7 + rng.normal(0.2, 0.1, len(y_true)), 0, 1)
        viz.plot_pr(y_true, y_score, tmp_path, dpi=72)
        assert (tmp_path / "pr_curve.png").exists()


# ────────────────────────────────────────────────────────
# plot_calibration
# ────────────────────────────────────────────────────────

class TestPlotCalibration:
    def test_creates_png(self, tmp_path):
        rng = np.random.RandomState(3)
        y_true = np.array([0, 0, 1, 1] * 25)
        y_score = np.clip(y_true * 0.6 + rng.normal(0.3, 0.15, len(y_true)), 0, 1)
        viz.plot_calibration(y_true, y_score, tmp_path, dpi=72)
        assert (tmp_path / "calibration_curve.png").exists()

    def test_skips_insufficient_data(self, tmp_path, capsys):
        viz.plot_calibration(np.array([1, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5]), tmp_path, dpi=72)
        captured = capsys.readouterr().out
        # Either skips or creates a degenerate plot — both are acceptable
        assert (tmp_path / "calibration_curve.png").exists() or "SKIP" in captured


# ────────────────────────────────────────────────────────
# plot_dca
# ────────────────────────────────────────────────────────

class TestPlotDca:
    def test_creates_png(self, tmp_path):
        rng = np.random.RandomState(4)
        y_true = np.array([0, 0, 1, 1] * 10)
        y_score = np.clip(y_true * 0.6 + rng.normal(0.3, 0.15, len(y_true)), 0, 1)
        viz.plot_dca(y_true, y_score, tmp_path, dpi=72)
        assert (tmp_path / "dca_curve.png").exists()


# ────────────────────────────────────────────────────────
# plot_feature_importance
# ────────────────────────────────────────────────────────

class TestPlotFeatureImportance:
    def test_dict_format(self, tmp_path):
        report = {"feature_importance": {"age": 0.3, "bp": 0.2, "creat": 0.1}}
        viz.plot_feature_importance(report, tmp_path, dpi=72)
        assert (tmp_path / "feature_importance.png").exists()

    def test_list_format(self, tmp_path):
        report = {"feature_importance": [
            {"feature": "age", "importance": 0.3},
            {"feature": "bp", "importance": 0.2},
        ]}
        viz.plot_feature_importance(report, tmp_path, dpi=72)
        assert (tmp_path / "feature_importance.png").exists()

    def test_fallback_key(self, tmp_path):
        report = {"feature_importances": {"x1": 0.5, "x2": 0.3}}
        viz.plot_feature_importance(report, tmp_path, dpi=72)
        assert (tmp_path / "feature_importance.png").exists()

    def test_missing_importance_skips(self, tmp_path, capsys):
        report = {"metrics": {"roc_auc": 0.9}}
        viz.plot_feature_importance(report, tmp_path, dpi=72)
        assert not (tmp_path / "feature_importance.png").exists()
        assert "SKIP" in capsys.readouterr().out

    def test_top_k(self, tmp_path):
        fi = {f"f{i}": float(i) for i in range(30)}
        report = {"feature_importance": fi}
        viz.plot_feature_importance(report, tmp_path, dpi=72, top_k=5)
        assert (tmp_path / "feature_importance.png").exists()


# ────────────────────────────────────────────────────────
# main (integration)
# ────────────────────────────────────────────────────────

class TestMain:
    def test_missing_report_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(tmp_path / "nope.json")],
        )
        assert viz.main() == 1

    def test_invalid_json_returns_1(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid", encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(bad)],
        )
        assert viz.main() == 1

    def test_non_dict_json_returns_1(self, tmp_path, monkeypatch):
        arr = tmp_path / "arr.json"
        arr.write_text("[1,2,3]", encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(arr)],
        )
        assert viz.main() == 1

    def test_report_only_succeeds(self, tmp_path, monkeypatch):
        report_path = _make_report(tmp_path)
        out_dir = tmp_path / "plots"
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(report_path),
             "--output-dir", str(out_dir)],
        )
        rc = viz.main()
        assert rc == 0
        assert out_dir.exists()

    def test_full_run_with_trace(self, tmp_path, monkeypatch):
        report_path = _make_report(
            tmp_path,
            extra={"feature_importance": {"age": 0.4, "bp": 0.2}},
        )
        trace_path = _make_trace(tmp_path, n=200)
        out_dir = tmp_path / "plots"
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(report_path),
             "--prediction-trace", str(trace_path),
             "--output-dir", str(out_dir), "--dpi", "72"],
        )
        rc = viz.main()
        assert rc == 0
        assert (out_dir / "roc_curve.png").exists()
        assert (out_dir / "pr_curve.png").exists()
        assert (out_dir / "dca_curve.png").exists()
        assert (out_dir / "feature_importance.png").exists()

    def test_trace_missing_columns_skips_curves(self, tmp_path, monkeypatch):
        report_path = _make_report(tmp_path)
        trace = tmp_path / "trace.csv"
        pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(trace, index=False)
        out_dir = tmp_path / "plots"
        monkeypatch.setattr(
            "sys.argv",
            ["viz", "--evaluation-report", str(report_path),
             "--prediction-trace", str(trace),
             "--output-dir", str(out_dir)],
        )
        rc = viz.main()
        assert rc == 0
        assert not (out_dir / "roc_curve.png").exists()
