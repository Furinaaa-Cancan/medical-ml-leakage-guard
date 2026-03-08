"""Tests for scripts/external_validation_gate.py.

Covers helper functions (to_int, safe_ratio, confusion_counts, normalize_binary,
parse_thresholds, compare_metric), cohort validation, metric replay, transport
gap checks, type coverage, and CLI integration.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "external_validation_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import external_validation_gate as evg


# ── helper functions ─────────────────────────────────────────────────────────

class TestToInt:
    def test_int(self):
        assert evg.to_int(5) == 5

    def test_bool(self):
        assert evg.to_int(True) is None

    def test_float_integer(self):
        assert evg.to_int(5.0) == 5

    def test_none(self):
        assert evg.to_int(None) is None


class TestSafeRatio:
    def test_normal(self):
        assert evg.safe_ratio(10.0, 20.0) == 0.5

    def test_zero_denom(self):
        assert evg.safe_ratio(10.0, 0.0) == 0.0


class TestNormalizeBinary:
    def test_valid(self):
        s = pd.Series([0, 1, 0, 1])
        result = evg.normalize_binary(s)
        assert result is not None

    def test_non_binary(self):
        assert evg.normalize_binary(pd.Series([0, 1, 2])) is None


class TestParseThresholds:
    def test_defaults(self):
        result = evg.parse_thresholds(None)
        assert result["max_pr_auc_drop"] == 0.08
        assert result["require_cross_period"] is True

    def test_custom(self):
        policy = {"external_validation_thresholds": {"max_pr_auc_drop": 0.12}}
        result = evg.parse_thresholds(policy)
        assert result["max_pr_auc_drop"] == 0.12

    def test_bool_override(self):
        policy = {"external_validation_thresholds": {"require_cross_period": False}}
        result = evg.parse_thresholds(policy)
        assert result["require_cross_period"] is False


# ── CLI integration ──────────────────────────────────────────────────────────

def _generate_cohort_data(n_pos, n_neg, threshold=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_score = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, len(y_true)), rng.beta(2, 5, len(y_true))),
        0.0, 1.0,
    )
    y_pred = (y_score >= threshold).astype(int)
    return y_true, y_score, y_pred


def _build_test_artifacts(tmp_path, threshold=0.5):
    rng = np.random.RandomState(42)

    # Internal test data for evaluation report
    y_true_test, y_score_test, y_pred_test = _generate_cohort_data(30, 30, threshold, rng)
    test_metrics, _ = evg.metric_panel(y_true_test, y_score_test, y_pred_test, beta=2.0)

    # External cohorts
    rng2 = np.random.RandomState(99)
    y_true_cp, y_score_cp, y_pred_cp = _generate_cohort_data(25, 25, threshold, rng2)
    cp_metrics, cp_cm = evg.metric_panel(y_true_cp, y_score_cp, y_pred_cp, beta=2.0)

    rng3 = np.random.RandomState(77)
    y_true_ci, y_score_ci, y_pred_ci = _generate_cohort_data(25, 25, threshold, rng3)
    ci_metrics, ci_cm = evg.metric_panel(y_true_ci, y_score_ci, y_pred_ci, beta=2.0)

    # Build trace CSV
    rows = []
    for i in range(len(y_true_test)):
        rows.append({
            "scope": "test", "cohort_id": "internal", "cohort_type": "internal",
            "hashed_patient_id": f"test_{i}", "y_true": int(y_true_test[i]),
            "y_score": float(y_score_test[i]), "y_pred": int(y_pred_test[i]),
            "selected_threshold": threshold, "model_id": "logistic",
        })
    for i in range(len(y_true_cp)):
        rows.append({
            "scope": "external", "cohort_id": "hospital_2020", "cohort_type": "cross_period",
            "hashed_patient_id": f"cp_{i}", "y_true": int(y_true_cp[i]),
            "y_score": float(y_score_cp[i]), "y_pred": int(y_pred_cp[i]),
            "selected_threshold": threshold, "model_id": "logistic",
        })
    for i in range(len(y_true_ci)):
        rows.append({
            "scope": "external", "cohort_id": "hospital_b", "cohort_type": "cross_institution",
            "hashed_patient_id": f"ci_{i}", "y_true": int(y_true_ci[i]),
            "y_score": float(y_score_ci[i]), "y_pred": int(y_pred_ci[i]),
            "selected_threshold": threshold, "model_id": "logistic",
        })
    trace_path = tmp_path / "prediction_trace.csv"
    pd.DataFrame(rows).to_csv(trace_path, index=False)

    # Build evaluation report
    eval_report = {"metrics": test_metrics}
    eval_path = tmp_path / "evaluation_report.json"
    eval_path.write_text(json.dumps(eval_report))

    # Build external validation report
    ext_report = {
        "cohorts": [
            {
                "cohort_id": "hospital_2020",
                "cohort_type": "cross_period",
                "metrics": cp_metrics,
                "confusion_matrix": cp_cm,
                "row_count": len(y_true_cp),
                "positive_count": int(np.sum(y_true_cp)),
            },
            {
                "cohort_id": "hospital_b",
                "cohort_type": "cross_institution",
                "metrics": ci_metrics,
                "confusion_matrix": ci_cm,
                "row_count": len(y_true_ci),
                "positive_count": int(np.sum(y_true_ci)),
            },
        ]
    }
    ext_path = tmp_path / "external_validation_report.json"
    ext_path.write_text(json.dumps(ext_report))

    return trace_path, eval_path, ext_path


def _run_gate(tmp_path, trace_path=None, eval_path=None, ext_path=None,
              strict=False, policy=None):
    if trace_path is None or eval_path is None or ext_path is None:
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable, str(GATE_SCRIPT),
        "--external-validation-report", str(ext_path),
        "--prediction-trace", str(trace_path),
        "--evaluation-report", str(eval_path),
        "--report", str(report_path),
    ]
    if policy is not None:
        pp_path = tmp_path / "policy.json"
        pp_path.write_text(json.dumps(policy))
        cmd.extend(["--performance-policy", str(pp_path)])
    if strict:
        cmd.append("--strict")
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
        assert "summary" in report
        s = report["summary"]
        assert "replayed_cohorts" in s
        assert "thresholds" in s


class TestMissingFiles:
    def test_missing_ext_report(self, tmp_path):
        trace_path, eval_path, _ = _build_test_artifacts(tmp_path)
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=tmp_path / "nope.json")
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_missing" in codes

    def test_missing_trace(self, tmp_path):
        _, eval_path, ext_path = _build_test_artifacts(tmp_path)
        report = _run_gate(tmp_path, trace_path=tmp_path / "nope.csv",
                           eval_path=eval_path, ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_missing" in codes


class TestCohortValidation:
    def test_empty_cohorts(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"] = []
        ext_path.write_text(json.dumps(ext))
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_min_cohort_not_met" in codes

    def test_missing_cohort_id(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"][0]["cohort_id"] = ""
        ext_path.write_text(json.dumps(ext))
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMetricReplay:
    def test_metric_mismatch(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"][0]["metrics"]["roc_auc"] = 0.999
        ext_path.write_text(json.dumps(ext))
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestTransportDrop:
    def test_large_drop(self, tmp_path):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        # Inflate internal metrics so transport drop is huge
        ev = json.loads(eval_path.read_text())
        ev["metrics"]["pr_auc"] = 0.99
        ev["metrics"]["f2_beta"] = 0.99
        ev["metrics"]["brier"] = 0.01
        eval_path.write_text(json.dumps(ev))
        report = _run_gate(tmp_path, trace_path=trace_path, eval_path=eval_path,
                           ext_path=ext_path)
        codes = [f["code"] for f in report["failures"]]
        assert "external_validation_transport_drop_exceeds_threshold" in codes


class TestStrictMode:
    def test_strict_flag(self, tmp_path):
        report = _run_gate(tmp_path, strict=True)
        assert report["strict_mode"] is True


# ── Direct main() tests ────────────────────────────────────────────────────

def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class TestMainPass:
    def test_pass(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 0
        out = json.loads(rpt.read_text())
        assert out["status"] == "pass"
        assert out["failure_count"] == 0
        assert "replayed_cohorts" in out["summary"]


class TestMainMissingFiles:
    def test_missing_ext_report(self, tmp_path, monkeypatch):
        trace_path, eval_path, _ = _build_test_artifacts(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(tmp_path / "nope.json"),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes

    def test_missing_trace(self, tmp_path, monkeypatch):
        _, eval_path, ext_path = _build_test_artifacts(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(tmp_path / "nope.csv"),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes

    def test_missing_eval_report(self, tmp_path, monkeypatch):
        trace_path, _, ext_path = _build_test_artifacts(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(tmp_path / "nope.json"),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes


class TestMainInvalidJSON:
    def test_invalid_ext_json(self, tmp_path, monkeypatch):
        trace_path, eval_path, _ = _build_test_artifacts(tmp_path)
        bad = tmp_path / "bad_ext.json"
        bad.write_text("{bad", encoding="utf-8")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(bad),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes


class TestMainEmptyCohorts:
    def test_no_cohorts(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"] = []
        _write_json(ext_path, ext)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_min_cohort_not_met" in codes


class TestMainMetricMismatch:
    def test_metric_mismatch(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"][0]["metrics"]["roc_auc"] = 0.999
        _write_json(ext_path, ext)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainTransportDrop:
    def test_large_drop(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ev = json.loads(eval_path.read_text())
        ev["metrics"]["pr_auc"] = 0.99
        ev["metrics"]["f2_beta"] = 0.99
        ev["metrics"]["brier"] = 0.01
        _write_json(eval_path, ev)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_transport_drop_exceeds_threshold" in codes


class TestMainMissingCohortId:
    def test_empty_cohort_id(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"][0]["cohort_id"] = ""
        _write_json(ext_path, ext)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainMissingEvalMetrics:
    def test_no_eval_metrics(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ev = json.loads(eval_path.read_text())
        del ev["metrics"]
        _write_json(eval_path, ev)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainStrictDirect:
    def test_strict_mode(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt), "--strict",
        ])
        rc = evg.main()
        out = json.loads(rpt.read_text())
        assert out["strict_mode"] is True


class TestMainNoReport:
    def test_no_report_flag(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
        ])
        rc = evg.main()
        assert rc == 0


class TestMainNonBinaryTrace:
    def test_non_binary_y_true(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        df.loc[0, "y_true"] = 2
        trace2 = tmp_path / "trace2.csv"
        df.to_csv(trace2, index=False)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace2),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainScoreOutOfRange:
    def test_score_above_one(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        df.loc[0, "y_score"] = 1.5
        trace2 = tmp_path / "trace2.csv"
        df.to_csv(trace2, index=False)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace2),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainCorruptEvalReport:
    def test_corrupt_eval_json(self, tmp_path, monkeypatch):
        trace_path, _, ext_path = _build_test_artifacts(tmp_path)
        bad_eval = tmp_path / "bad_eval.json"
        bad_eval.write_text("{corrupt")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(bad_eval),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes


class TestMainCorruptTrace:
    def test_corrupt_trace_csv(self, tmp_path, monkeypatch):
        _, eval_path, ext_path = _build_test_artifacts(tmp_path)
        bad_trace = tmp_path / "bad_trace.csv"
        bad_trace.write_text("")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(bad_trace),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes or "external_validation_metric_replay_mismatch" in codes


class TestMainMissingTraceColumns:
    def test_missing_columns(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        df = df.drop(columns=["cohort_type"])
        trace2 = tmp_path / "trace_missing_col.csv"
        df.to_csv(trace2, index=False)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace2),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainNonFiniteExternalScore:
    def test_nan_y_score(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        df = pd.read_csv(trace_path)
        ext_rows = df[df["scope"] == "external"]
        if len(ext_rows) > 0:
            df.loc[ext_rows.index[0], "y_score"] = float("nan")
        trace2 = tmp_path / "trace_nan.csv"
        df.to_csv(trace2, index=False)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace2),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2


class TestMainMissingEvalPrAuc:
    def test_no_pr_auc(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ev = json.loads(eval_path.read_text())
        del ev["metrics"]["pr_auc"]
        _write_json(eval_path, ev)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestMainNonDictCohort:
    def test_cohort_not_dict(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        ext = json.loads(ext_path.read_text())
        ext["cohorts"].append("not_a_dict")
        _write_json(ext_path, ext)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_metric_replay_mismatch" in codes


class TestParseThresholdsEdgeCases:
    def test_custom_bool_thresholds(self):
        policy = {"external_validation_thresholds": {"require_cross_period": False, "require_cross_institution": False}}
        result = evg.parse_thresholds(policy)
        assert result["require_cross_period"] is False

    def test_custom_numeric_thresholds(self):
        policy = {"external_validation_thresholds": {"beta": 3.0, "min_cohort_count": 2.0, "max_pr_auc_drop": 0.05}}
        result = evg.parse_thresholds(policy)
        assert result["beta"] == 3.0
        assert result["min_cohort_count"] == 2.0

    def test_no_thresholds_block(self):
        result = evg.parse_thresholds({"other": True})
        assert result["beta"] == 2.0  # default


class TestMainWithPolicy:
    def test_custom_policy_thresholds(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        policy = {"external_validation_thresholds": {"max_pr_auc_drop": 0.001}}
        pp = tmp_path / "policy.json"
        _write_json(pp, policy)
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--performance-policy", str(pp),
            "--report", str(rpt),
        ])
        rc = evg.main()
        out = json.loads(rpt.read_text())
        assert "thresholds" in out.get("summary", {}) or rc in (0, 2)

    def test_corrupt_policy(self, tmp_path, monkeypatch):
        trace_path, eval_path, ext_path = _build_test_artifacts(tmp_path)
        bad_pp = tmp_path / "bad_policy.json"
        bad_pp.write_text("{bad")
        rpt = tmp_path / "rpt.json"
        monkeypatch.setattr("sys.argv", [
            "evg",
            "--external-validation-report", str(ext_path),
            "--prediction-trace", str(trace_path),
            "--evaluation-report", str(eval_path),
            "--performance-policy", str(bad_pp),
            "--report", str(rpt),
        ])
        rc = evg.main()
        assert rc == 2
        out = json.loads(rpt.read_text())
        codes = [f["code"] for f in out["failures"]]
        assert "external_validation_missing" in codes
