"""Tests for scripts/quick_summary.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import quick_summary as qs


def _write_json(path: Path, data) -> str:
    path.write_text(json.dumps(data))
    return str(path)


def _good_eval():
    return {
        "model_id": "logistic_l2__t01",
        "threshold_selection": {"selected_threshold": 0.5, "constraints_satisfied_overall": True},
        "metrics": {"roc_auc": 0.85, "pr_auc": 0.80, "f1": 0.75, "brier": 0.15,
                    "accuracy": 0.82, "sensitivity": 0.78, "specificity": 0.86,
                    "ppv": 0.80, "npv": 0.84, "f2_beta": 0.76},
        "uncertainty": {
            "method": "bootstrap", "n_resamples": 500,
            "metrics": {"roc_auc": {"ci_95": [0.80, 0.90]}, "pr_auc": {"ci_95": [0.75, 0.85]}},
        },
        "split_metrics": {
            "train": {"metrics": {"pr_auc": 0.88, "roc_auc": 0.90, "f1": 0.82, "brier": 0.12}},
            "valid": {"metrics": {"pr_auc": 0.83, "roc_auc": 0.87, "f1": 0.78, "brier": 0.14}},
            "test": {"metrics": {"pr_auc": 0.80, "roc_auc": 0.85, "f1": 0.75, "brier": 0.15}},
        },
    }


def _good_ms():
    return {
        "selected_model_id": "logistic_l2__t01",
        "selection_metric": "pr_auc",
        "candidates": [
            {"model_id": "logistic_l2__t01", "mean_score": 0.82, "std_score": 0.03},
            {"model_id": "logistic_l1__t01", "mean_score": 0.80, "std_score": 0.04},
            {"model_id": "svm_rbf__t01", "mean_score": 0.79, "std_score": 0.05},
        ],
    }


class TestLoadJson:
    def test_valid(self, tmp_path):
        p = tmp_path / "test.json"
        _write_json(p, {"a": 1})
        assert qs.load_json(p) == {"a": 1}

    def test_missing(self, tmp_path):
        assert qs.load_json(tmp_path / "nope.json") is None

    def test_corrupt(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{bad")
        assert qs.load_json(p) is None

    def test_non_dict(self, tmp_path):
        p = tmp_path / "arr.json"
        p.write_text("[1,2,3]")
        assert qs.load_json(p) is None


class TestFmtMetric:
    def test_none(self):
        assert qs.fmt_metric(None) == "—"

    def test_float(self):
        assert "0.8500" in qs.fmt_metric(0.85)

    def test_with_ci(self):
        result = qs.fmt_metric(0.85, [0.80, 0.90])
        assert "0.8000" in result and "0.9000" in result

    def test_bad_value(self):
        assert qs.fmt_metric("bad") == "bad"


class TestFmtPct:
    def test_none(self):
        assert qs.fmt_pct(None) == "—"

    def test_float(self):
        assert "35.0%" in qs.fmt_pct(0.35)


class TestMainDirect:
    def test_pass_with_output_dir(self, tmp_path, monkeypatch):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", _good_eval())
        _write_json(ev_dir / "model_selection_report.json", _good_ms())
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path)])
        rc = qs.main()
        assert rc == 0

    def test_pass_with_evidence_flag(self, tmp_path, monkeypatch):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", _good_eval())
        monkeypatch.setattr("sys.argv", ["qs", "--evidence", str(ev_dir)])
        rc = qs.main()
        assert rc == 0

    def test_pass_with_eval_flag(self, tmp_path, monkeypatch):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", _good_eval())
        monkeypatch.setattr("sys.argv", ["qs", "--eval", str(ev_dir / "evaluation_report.json")])
        rc = qs.main()
        assert rc == 0

    def test_json_mode(self, tmp_path, monkeypatch, capsys):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", _good_eval())
        _write_json(ev_dir / "model_selection_report.json", _good_ms())
        monkeypatch.setattr("sys.argv", ["qs", "--json", str(tmp_path)])
        rc = qs.main()
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["model_id"] == "logistic_l2__t01"
        assert data["candidates"] == 3

    def test_missing_eval(self, tmp_path, monkeypatch):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path)])
        rc = qs.main()
        assert rc == 1

    def test_no_args(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["qs"])
        rc = qs.main()
        assert rc == 1

    def test_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path / "nope")])
        rc = qs.main()
        assert rc == 1

    def test_overfitting_high(self, tmp_path, monkeypatch, capsys):
        ev = _good_eval()
        ev["split_metrics"]["train"]["metrics"]["pr_auc"] = 0.99
        ev["split_metrics"]["test"]["metrics"]["pr_auc"] = 0.60
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", ev)
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path)])
        rc = qs.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "HIGH" in out

    def test_no_model_selection(self, tmp_path, monkeypatch):
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", _good_eval())
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path)])
        rc = qs.main()
        assert rc == 0

    def test_constraints_fail(self, tmp_path, monkeypatch, capsys):
        ev = _good_eval()
        ev["threshold_selection"]["constraints_satisfied_overall"] = False
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        _write_json(ev_dir / "evaluation_report.json", ev)
        monkeypatch.setattr("sys.argv", ["qs", str(tmp_path)])
        rc = qs.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_flat_evidence_dir(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "evaluation_report.json", _good_eval())
        monkeypatch.setattr("sys.argv", ["qs", "--evidence", str(tmp_path)])
        rc = qs.main()
        assert rc == 0
