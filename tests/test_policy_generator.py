"""Tests for scripts/policy_generator.py.

Covers helpers (load_json, _is_finite, _safe_get), metric extractors,
threshold derivation, presets, output formatting, and CLI integration
via direct main().
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import policy_generator as pg


def _write_json(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


# ── helpers ──────────────────────────────────────────────────────────────────


class TestLoadJson:
    def test_valid(self, tmp_path):
        _write_json(tmp_path / "r.json", {"a": 1})
        assert pg.load_json(tmp_path / "r.json") == {"a": 1}

    def test_missing(self, tmp_path):
        assert pg.load_json(tmp_path / "nope.json") is None

    def test_invalid(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert pg.load_json(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        _write_json(tmp_path / "arr.json", [1, 2])
        assert pg.load_json(tmp_path / "arr.json") is None


class TestIsFinite:
    def test_int(self):
        assert pg._is_finite(5) is True

    def test_float(self):
        assert pg._is_finite(0.5) is True

    def test_bool(self):
        assert pg._is_finite(True) is False

    def test_nan(self):
        assert pg._is_finite(float("nan")) is False

    def test_none(self):
        assert pg._is_finite(None) is False


class TestSafeGet:
    def test_nested(self):
        d = {"a": {"b": {"c": 0.5}}}
        assert pg._safe_get(d, "a", "b", "c") == 0.5

    def test_missing(self):
        assert pg._safe_get({"a": 1}, "b") is None

    def test_non_dict(self):
        assert pg._safe_get({"a": "text"}, "a", "b") is None

    def test_non_finite(self):
        assert pg._safe_get({"a": float("nan")}, "a") is None


# ── margin helpers ───────────────────────────────────────────────────────────


class TestApplyMargin:
    def test_lower_is_better(self):
        # observed=0.10, margin=0.15 → threshold = 0.10 * 1.15 = 0.115
        result = pg._apply_margin_lower(0.10, 0.15)
        assert abs(result - 0.115) < 0.001

    def test_higher_is_better(self):
        # observed=0.90, margin=0.15 → threshold = 0.90 * 0.85 = 0.765
        result = pg._apply_margin_higher(0.90, 0.15)
        assert abs(result - 0.765) < 0.001


# ── metric extractors ───────────────────────────────────────────────────────


def _make_eval_report(roc_auc=0.88, pr_auc=0.82, brier=0.12):
    return {
        "metrics": {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier},
    }


def _make_robustness_report(ts_drop=0.05, ts_range=0.10):
    return {
        "summary": {
            "computed": {
                "time_slices": {
                    "pr_auc_worst_drop_from_overall": ts_drop,
                    "pr_auc_range": ts_range,
                },
            }
        }
    }


class TestExtractEvalMetrics:
    def test_top_level(self, tmp_path):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        m = pg.extract_eval_metrics(tmp_path)
        assert "roc_auc" in m
        assert abs(m["roc_auc"] - 0.88) < 0.001

    def test_split_metrics_fallback(self, tmp_path):
        _write_json(tmp_path / "evaluation_report.json", {
            "split_metrics": {"test": {"metrics": {"roc_auc": 0.85}}}
        })
        m = pg.extract_eval_metrics(tmp_path)
        assert abs(m["roc_auc"] - 0.85) < 0.001

    def test_missing(self, tmp_path):
        assert pg.extract_eval_metrics(tmp_path) == {}


class TestExtractRobustnessMetrics:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        m = pg.extract_robustness_metrics(tmp_path)
        assert "time_slices_pr_auc_drop" in m

    def test_missing(self, tmp_path):
        assert pg.extract_robustness_metrics(tmp_path) == {}


class TestExtractGeneralizationMetrics:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "generalization_gap_report.json", {
            "summary": {"train_test_auc_gap": 0.03}
        })
        m = pg.extract_generalization_metrics(tmp_path)
        assert abs(m["train_test_auc_gap"] - 0.03) < 0.001

    def test_missing(self, tmp_path):
        assert pg.extract_generalization_metrics(tmp_path) == {}


class TestExtractCalibrationMetrics:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "calibration_dca_report.json", {
            "summary": {"brier_score": 0.15, "hosmer_lemeshow_p_value": 0.35}
        })
        m = pg.extract_calibration_metrics(tmp_path)
        assert "brier_score" in m
        assert "hosmer_lemeshow_p" in m

    def test_missing(self, tmp_path):
        assert pg.extract_calibration_metrics(tmp_path) == {}


class TestExtractSeedMetrics:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "seed_stability_report.json", {
            "summary": {"pr_auc_cv": 0.02}
        })
        m = pg.extract_seed_metrics(tmp_path)
        assert abs(m["pr_auc_cv"] - 0.02) < 0.001

    def test_missing(self, tmp_path):
        assert pg.extract_seed_metrics(tmp_path) == {}


class TestExtractExternalMetrics:
    def test_with_cohorts(self, tmp_path):
        _write_json(tmp_path / "external_validation_gate_report.json", {
            "summary": {
                "replayed_cohorts": [
                    {"transport_gap": {"pr_auc_drop_from_internal_test": 0.05}},
                    {"transport_gap": {"pr_auc_drop_from_internal_test": 0.08}},
                ]
            }
        })
        m = pg.extract_external_metrics(tmp_path)
        assert abs(m["max_cohort_pr_auc_drop"] - 0.08) < 0.001

    def test_missing(self, tmp_path):
        assert pg.extract_external_metrics(tmp_path) == {}


# ── derive_policy ────────────────────────────────────────────────────────────


class TestDerivePolicy:
    def test_full(self, tmp_path):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        _write_json(tmp_path / "generalization_gap_report.json", {
            "summary": {"train_test_auc_gap": 0.03}
        })
        policy, observed = pg.derive_policy(tmp_path, margin=0.15)
        assert "evaluation_metric_floors" in policy
        assert "robustness_thresholds" in policy
        assert "generalization_gap_thresholds" in policy
        assert len(observed) > 0

    def test_empty_dir(self, tmp_path):
        policy, observed = pg.derive_policy(tmp_path)
        assert observed == {}
        # Only _generator and _margin keys
        assert len([k for k in policy if not k.startswith("_")]) == 0

    def test_margin_affects_thresholds(self, tmp_path):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        pol_tight, _ = pg.derive_policy(tmp_path, margin=0.05)
        pol_loose, _ = pg.derive_policy(tmp_path, margin=0.30)
        tight_floor = pol_tight["evaluation_metric_floors"]["min_roc_auc"]
        loose_floor = pol_loose["evaluation_metric_floors"]["min_roc_auc"]
        assert tight_floor > loose_floor  # tighter margin → higher floor


# ── to_text ──────────────────────────────────────────────────────────────────


class TestToText:
    def test_structure(self, tmp_path):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        policy, observed = pg.derive_policy(tmp_path)
        text = pg.to_text(policy, observed, 0.15)
        assert "Generated Performance Policy" in text
        assert "Observed Metrics" in text
        assert "roc_auc" in text


# ── direct main() CLI tests ─────────────────────────────────────────────────


class TestMainPass:
    def test_json_stdout(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path),
        ])
        rc = pg.main()
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "evaluation_metric_floors" in data

    def test_json_output_file(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        out = tmp_path / "policy.json"
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--output", str(out),
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(out.read_text())
        assert "evaluation_metric_floors" in data

    def test_text_output(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--text",
        ])
        rc = pg.main()
        assert rc == 0
        captured = capsys.readouterr()
        assert "Generated Performance Policy" in captured.out


class TestMainMissingDir:
    def test_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path / "nope"),
        ])
        rc = pg.main()
        assert rc == 1


class TestMainPreset:
    def test_strict_preset(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--preset", "strict",
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["_preset"] == "strict"
        assert data["_margin"] == 0.05

    def test_lenient_preset(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--preset", "lenient",
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["_preset"] == "lenient"
        assert data["_margin"] == 0.30


class TestMainCustomMargin:
    def test_custom_margin(self, tmp_path, monkeypatch, capsys):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--margin", "0.25",
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["_margin"] == 0.25


class TestMainEmptyDir:
    def test_empty_evidence(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path),
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # Only internal keys
        real_keys = [k for k in data if not k.startswith("_")]
        assert len(real_keys) == 0


class TestMainAllReports:
    def test_comprehensive(self, tmp_path, monkeypatch):
        _write_json(tmp_path / "evaluation_report.json", _make_eval_report())
        _write_json(tmp_path / "robustness_gate_report.json", _make_robustness_report())
        _write_json(tmp_path / "generalization_gap_report.json", {
            "summary": {"train_test_auc_gap": 0.03}
        })
        _write_json(tmp_path / "calibration_dca_report.json", {
            "summary": {"brier_score": 0.15, "hosmer_lemeshow_p_value": 0.35}
        })
        _write_json(tmp_path / "seed_stability_report.json", {
            "summary": {"pr_auc_cv": 0.02}
        })
        _write_json(tmp_path / "external_validation_gate_report.json", {
            "summary": {
                "replayed_cohorts": [
                    {"transport_gap": {"pr_auc_drop_from_internal_test": 0.06}},
                ]
            }
        })
        out = tmp_path / "policy.json"
        monkeypatch.setattr("sys.argv", [
            "pg", "--evidence-dir", str(tmp_path), "--output", str(out),
        ])
        rc = pg.main()
        assert rc == 0
        data = json.loads(out.read_text())
        assert "evaluation_metric_floors" in data
        assert "robustness_thresholds" in data
        assert "generalization_gap_thresholds" in data
        assert "calibration_thresholds" in data
        assert "seed_stability_thresholds" in data
        assert "external_validation_thresholds" in data
