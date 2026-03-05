"""Tests for scripts/prediction_replay_gate.py.

Covers helper functions (to_int, safe_ratio, confusion_counts, metric_panel,
parse_thresholds, normalize_binary, compare_metric), trace validation,
metric replay, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "prediction_replay_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import prediction_replay_gate as prg


# ── helper functions ─────────────────────────────────────────────────────────

class TestToInt:
    def test_int(self):
        assert prg.to_int(5) == 5

    def test_bool(self):
        assert prg.to_int(True) is None

    def test_float_integer(self):
        assert prg.to_int(5.0) == 5

    def test_float_non_integer(self):
        assert prg.to_int(5.5) is None

    def test_nan(self):
        assert prg.to_int(float("nan")) is None


class TestSafeRatio:
    def test_normal(self):
        assert prg.safe_ratio(10.0, 20.0) == 0.5

    def test_zero_denom(self):
        assert prg.safe_ratio(10.0, 0.0) == 0.0

    def test_negative_denom(self):
        assert prg.safe_ratio(10.0, -1.0) == 0.0


class TestConfusionCounts:
    def test_basic(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        cm = prg.confusion_counts(y_true, y_pred)
        assert cm["tp"] == 2
        assert cm["fn"] == 1
        assert cm["tn"] == 1
        assert cm["fp"] == 1


class TestNormalizeBinary:
    def test_valid(self):
        s = pd.Series([0, 1, 0, 1])
        result = prg.normalize_binary(s)
        assert result is not None
        assert list(result) == [0, 1, 0, 1]

    def test_non_binary(self):
        s = pd.Series([0, 1, 2])
        assert prg.normalize_binary(s) is None

    def test_nan(self):
        s = pd.Series([0, 1, None])
        assert prg.normalize_binary(s) is None


class TestParseThresholds:
    def test_defaults(self):
        result = prg.parse_thresholds(None)
        assert result["metric_tolerance"] == 1e-6
        assert result["beta"] == 2.0

    def test_custom(self):
        policy = {"prediction_replay_thresholds": {"metric_tolerance": 0.01, "beta": 1.0}}
        result = prg.parse_thresholds(policy)
        assert result["metric_tolerance"] == 0.01
        assert result["beta"] == 1.0


class TestMetricPanel:
    def test_perfect(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.1, 0.2])
        y_pred = np.array([1, 1, 0, 0])
        metrics, cm = prg.metric_panel(y_true, y_score, y_pred, beta=2.0)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert cm["tp"] == 2
        assert cm["tn"] == 2


# ── CLI integration ──────────────────────────────────────────────────────────

def _generate_split_data(n_pos, n_neg, threshold=0.5, rng=None):
    """Generate realistic y_true, y_score, y_pred for a split."""
    if rng is None:
        rng = np.random.RandomState(42)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    # Scores correlated with true labels
    y_score = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, len(y_true)), rng.beta(2, 5, len(y_true))),
        0.0, 1.0
    )
    y_pred = (y_score >= threshold).astype(int)
    return y_true, y_score, y_pred


def _build_trace_and_eval(tmp_path, threshold=0.5, rng=None):
    """Build matching prediction trace CSV and evaluation report JSON."""
    if rng is None:
        rng = np.random.RandomState(42)

    rows = []
    split_metrics_dict = {}
    for split_name, (n_pos, n_neg) in [("train", (40, 40)), ("valid", (20, 20)), ("test", (30, 30))]:
        y_true, y_score, y_pred = _generate_split_data(n_pos, n_neg, threshold, rng)
        for i in range(len(y_true)):
            rows.append({
                "scope": split_name,
                "cohort_id": "internal",
                "cohort_type": "internal",
                "hashed_patient_id": f"p_{split_name}_{i}",
                "y_true": int(y_true[i]),
                "y_score": float(y_score[i]),
                "y_pred": int(y_pred[i]),
                "selected_threshold": threshold,
                "model_id": "logistic_l2",
            })
        metrics, cm = prg.metric_panel(y_true, y_score, y_pred, beta=2.0)
        split_metrics_dict[split_name] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "n_samples": len(y_true),
        }

    trace_df = pd.DataFrame(rows)
    trace_path = tmp_path / "prediction_trace.csv"
    trace_df.to_csv(trace_path, index=False)

    test_metrics = dict(split_metrics_dict["test"]["metrics"])
    eval_report = {
        "split_metrics": split_metrics_dict,
        "metrics": test_metrics,
        "threshold_selection": {
            "selection_split": "valid",
            "selected_threshold": threshold,
        },
    }
    eval_path = tmp_path / "evaluation_report.json"
    eval_path.write_text(json.dumps(eval_report))

    return trace_path, eval_path


def _run_gate(tmp_path, trace_path=None, eval_path=None, strict=False, extra_args=None):
    if trace_path is None or eval_path is None:
        trace_path, eval_path = _build_trace_and_eval(tmp_path)

    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--evaluation-report", str(eval_path),
        "--prediction-trace", str(trace_path),
        "--report", str(report_path),
    ]
    if strict:
        cmd.append("--strict")
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {}


class TestCLIPass:
    def test_valid_pass(self, tmp_path):
        report = _run_gate(tmp_path)
        assert report["status"] == "pass"
        assert report["failure_count"] == 0

    def test_report_structure(self, tmp_path):
        report = _run_gate(tmp_path)
        assert "status" in report
        assert "strict_mode" in report
        assert "failures" in report
        assert "warnings" in report
        assert "summary" in report
        s = report["summary"]
        assert "splits" in s
        assert "trace_row_count" in s
        assert "thresholds" in s


class TestMissingFiles:
    def test_missing_eval_report(self, tmp_path):
        trace_path = tmp_path / "trace.csv"
        pd.DataFrame({"scope": ["test"], "y_true": [1]}).to_csv(trace_path, index=False)
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=tmp_path / "nope.json")
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_trace_missing" in codes

    def test_missing_trace(self, tmp_path):
        eval_path = tmp_path / "eval.json"
        eval_path.write_text("{}")
        report = _run_gate(tmp_path, trace_path=tmp_path / "nope.csv", eval_path=eval_path)
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_trace_missing" in codes


class TestTraceSchema:
    def test_missing_columns(self, tmp_path):
        trace_path, eval_path = _build_trace_and_eval(tmp_path)
        df = pd.read_csv(trace_path)
        df = df.drop(columns=["y_score"])
        trace_path2 = tmp_path / "trace2.csv"
        df.to_csv(trace_path2, index=False)
        report = _run_gate(tmp_path, trace_path=trace_path2, eval_path=eval_path)
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_trace_schema_invalid" in codes

    def test_non_binary_y_true(self, tmp_path):
        trace_path, eval_path = _build_trace_and_eval(tmp_path)
        df = pd.read_csv(trace_path)
        df.loc[0, "y_true"] = 2
        trace_path2 = tmp_path / "trace2.csv"
        df.to_csv(trace_path2, index=False)
        report = _run_gate(tmp_path, trace_path=trace_path2, eval_path=eval_path)
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_trace_schema_invalid" in codes

    def test_score_out_of_range(self, tmp_path):
        trace_path, eval_path = _build_trace_and_eval(tmp_path)
        df = pd.read_csv(trace_path)
        df.loc[0, "y_score"] = 1.5
        trace_path2 = tmp_path / "trace2.csv"
        df.to_csv(trace_path2, index=False)
        report = _run_gate(tmp_path, trace_path=trace_path2, eval_path=eval_path)
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_score_out_of_range" in codes


class TestMetricReplay:
    def test_metric_mismatch(self, tmp_path):
        trace_path, eval_path = _build_trace_and_eval(tmp_path)
        ev = json.loads(eval_path.read_text())
        ev["split_metrics"]["test"]["metrics"]["roc_auc"] = 0.999
        ev["metrics"]["roc_auc"] = 0.999
        eval_path.write_text(json.dumps(ev))
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path)
        codes = [f["code"] for f in report["failures"]]
        assert "prediction_metric_replay_mismatch" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True
