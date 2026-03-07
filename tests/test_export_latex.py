#!/usr/bin/env python3
"""Unit tests for scripts/export_latex.py."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import export_latex as el
from export_latex import (
    _escape,
    _fmt,
    _fmt_ci,
    _load,
    table_ci_matrix,
    table_external,
    table_model_selection,
    table_performance,
)


# ── _fmt ──────────────────────────────────────────────────────


class TestFmt:
    def test_none(self) -> None:
        assert _fmt(None) == "---"

    def test_float(self) -> None:
        assert _fmt(0.12345, 3) == "0.123"

    def test_int(self) -> None:
        assert _fmt(1, 2) == "1.00"

    def test_zero(self) -> None:
        assert _fmt(0.0, 4) == "0.0000"

    def test_string_numeric(self) -> None:
        assert _fmt("0.5", 2) == "0.50"

    def test_string_non_numeric(self) -> None:
        assert _fmt("abc") == "abc"

    def test_default_dp(self) -> None:
        assert _fmt(0.123456789) == "0.123"


# ── _fmt_ci ───────────────────────────────────────────────────


class TestFmtCi:
    def test_both_present(self) -> None:
        assert _fmt_ci(0.7, 0.9, 2) == "[0.70, 0.90]"

    def test_low_none(self) -> None:
        assert _fmt_ci(None, 0.9) == ""

    def test_high_none(self) -> None:
        assert _fmt_ci(0.7, None) == ""

    def test_both_none(self) -> None:
        assert _fmt_ci(None, None) == ""


# ── _load ─────────────────────────────────────────────────────


class TestLoad:
    def test_none_path(self) -> None:
        assert _load(None) is None

    def test_empty_string(self) -> None:
        assert _load("") is None

    def test_valid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "r.json"
        p.write_text('{"status": "pass"}', encoding="utf-8")
        assert _load(str(p)) == {"status": "pass"}

    def test_missing_file(self, tmp_path: Path) -> None:
        assert _load(str(tmp_path / "no_file.json")) is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{broken", encoding="utf-8")
        assert _load(str(p)) is None

    def test_non_dict_json(self, tmp_path: Path) -> None:
        p = tmp_path / "arr.json"
        p.write_text("[1,2]", encoding="utf-8")
        assert _load(str(p)) is None


# ── _escape ───────────────────────────────────────────────────


class TestEscape:
    def test_underscore(self) -> None:
        assert _escape("a_b") == r"a\_b"

    def test_ampersand(self) -> None:
        assert _escape("a&b") == r"a\&b"

    def test_percent(self) -> None:
        assert _escape("50%") == r"50\%"

    def test_hash(self) -> None:
        assert _escape("#1") == r"\#1"

    def test_dollar(self) -> None:
        assert _escape("$x") == r"\$x"

    def test_no_special(self) -> None:
        assert _escape("hello") == "hello"

    def test_multiple_specials(self) -> None:
        result = _escape("a_b&c%d#e$f")
        assert result == r"a\_b\&c\%d\#e\$f"


# ── table_performance ─────────────────────────────────────────


class TestTablePerformance:
    def test_dict_splits(self) -> None:
        report = {
            "splits": {
                "test": {"roc_auc": 0.9, "pr_auc": 0.85, "brier": 0.1, "accuracy": 0.88}
            }
        }
        tex = table_performance(report, 3)
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex
        assert "0.900" in tex
        assert "0.850" in tex
        assert "test" in tex

    def test_list_splits(self) -> None:
        report = {
            "splits": [
                {"split": "train", "roc_auc": 0.95, "pr_auc": 0.92, "brier": 0.05, "accuracy": 0.93}
            ]
        }
        tex = table_performance(report, 3)
        assert "train" in tex
        assert "0.950" in tex

    def test_empty_splits(self) -> None:
        report: Dict[str, Any] = {}
        tex = table_performance(report, 3)
        assert "\\begin{table}" in tex
        assert "\\end{table}" in tex

    def test_top_level_fallback(self) -> None:
        report = {"test_roc_auc": 0.88}
        tex = table_performance(report, 3)
        assert "0.880" in tex

    def test_alternate_key_names(self) -> None:
        report = {
            "per_split_metrics": {
                "valid": {"auroc": 0.87, "auprc": 0.82, "brier_score": 0.12}
            }
        }
        tex = table_performance(report, 3)
        assert "0.870" in tex
        assert "0.820" in tex


# ── table_model_selection ─────────────────────────────────────


class TestTableModelSelection:
    def test_with_candidates(self) -> None:
        report = {
            "candidates": [
                {"family": "logistic", "cv_score": 0.82, "complexity_rank": 1, "selected": True},
                {"family": "xgboost", "cv_score": 0.85, "complexity_rank": 3, "selected": False},
            ]
        }
        tex = table_model_selection(report, 3)
        assert "\\begin{table}" in tex
        assert "logistic" in tex
        assert "\\checkmark" in tex

    def test_empty_candidates(self) -> None:
        assert table_model_selection({"candidates": []}, 3) == ""

    def test_no_candidates_key(self) -> None:
        assert table_model_selection({}, 3) == ""

    def test_max_10_candidates(self) -> None:
        cands = [{"model_id": f"m{i}", "cv_score": 0.5 + i * 0.01} for i in range(15)]
        report = {"candidates": cands}
        tex = table_model_selection(report, 3)
        assert tex.count("\\\\") <= 14  # header + 10 rows + midrule lines


# ── table_external ────────────────────────────────────────────


class TestTableExternal:
    def test_with_cohorts(self) -> None:
        report = {
            "cohorts": [
                {"name": "hospital_A", "roc_auc": 0.87, "pr_auc": 0.80, "brier": 0.15}
            ]
        }
        tex = table_external(report, 3)
        assert "hospital" in tex
        assert "0.870" in tex

    def test_empty_cohorts(self) -> None:
        tex = table_external({"cohorts": []}, 3)
        assert "\\begin{table}" in tex

    def test_alternate_key(self) -> None:
        report = {
            "external_cohorts": [
                {"cohort_name": "ext1", "auroc": 0.75, "auprc": 0.70, "brier_score": 0.2}
            ]
        }
        tex = table_external(report, 3)
        assert "ext1" in tex


# ── table_ci_matrix ───────────────────────────────────────────


class TestTableCiMatrix:
    def test_with_matrix(self) -> None:
        report = {
            "ci_matrix": [
                {"metric": "pr_auc", "split": "test", "point_estimate": 0.85, "ci_lower": 0.78, "ci_upper": 0.91}
            ]
        }
        tex = table_ci_matrix(report, 3)
        assert "pr" in tex
        assert "0.850" in tex
        assert "[0.780, 0.910]" in tex

    def test_empty_matrix(self) -> None:
        assert table_ci_matrix({"ci_matrix": []}, 3) == ""

    def test_no_matrix_key(self) -> None:
        assert table_ci_matrix({}, 3) == ""

    def test_alternate_key(self) -> None:
        report = {
            "confidence_intervals": [
                {"metric": "roc_auc", "split": "valid", "value": 0.9, "lower": 0.85, "upper": 0.95}
            ]
        }
        tex = table_ci_matrix(report, 3)
        assert "roc" in tex
        assert "[0.850, 0.950]" in tex


# ── Direct main() tests ─────────────────────────────────────────

import json


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class TestMainBasic:
    def test_eval_only(self, tmp_path, monkeypatch):
        eval_rpt = {
            "splits": {
                "test": {"roc_auc": 0.9, "pr_auc": 0.85, "brier": 0.1, "accuracy": 0.88}
            }
        }
        ev = tmp_path / "eval.json"
        _write_json(ev, eval_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev), "--output", str(out),
        ])
        rc = el.main()
        assert rc == 0
        assert out.exists()
        tex = out.read_text()
        assert "\\begin{table}" in tex
        assert "0.900" in tex

    def test_custom_decimal_places(self, tmp_path, monkeypatch):
        eval_rpt = {"splits": {"test": {"roc_auc": 0.876543}}}
        ev = tmp_path / "eval.json"
        _write_json(ev, eval_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev), "--output", str(out),
            "--decimal-places", "5",
        ])
        rc = el.main()
        assert rc == 0
        tex = out.read_text()
        assert "0.87654" in tex


class TestMainMissingEval:
    def test_missing_eval_report(self, tmp_path, monkeypatch):
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(tmp_path / "nope.json"),
            "--output", str(out),
        ])
        rc = el.main()
        assert rc == 1
        assert not out.exists()


class TestMainWithModelSelection:
    def test_model_selection_included(self, tmp_path, monkeypatch):
        eval_rpt = {"splits": {"test": {"roc_auc": 0.9}}}
        ms_rpt = {
            "candidates": [
                {"family": "logistic", "cv_score": 0.82, "complexity_rank": 1, "selected": True},
                {"family": "xgboost", "cv_score": 0.85, "complexity_rank": 3, "selected": False},
            ]
        }
        ev = tmp_path / "eval.json"
        ms = tmp_path / "ms.json"
        _write_json(ev, eval_rpt)
        _write_json(ms, ms_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev),
            "--model-selection-report", str(ms),
            "--output", str(out),
        ])
        rc = el.main()
        assert rc == 0
        tex = out.read_text()
        assert "logistic" in tex
        assert "\\checkmark" in tex


class TestMainWithExternal:
    def test_external_included(self, tmp_path, monkeypatch):
        eval_rpt = {"splits": {"test": {"roc_auc": 0.9}}}
        ext_rpt = {
            "cohorts": [
                {"name": "hospital_A", "roc_auc": 0.87, "pr_auc": 0.80, "brier": 0.15}
            ]
        }
        ev = tmp_path / "eval.json"
        ext = tmp_path / "ext.json"
        _write_json(ev, eval_rpt)
        _write_json(ext, ext_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev),
            "--external-report", str(ext),
            "--output", str(out),
        ])
        rc = el.main()
        assert rc == 0
        tex = out.read_text()
        assert "hospital" in tex


class TestMainWithCI:
    def test_ci_matrix_included(self, tmp_path, monkeypatch):
        eval_rpt = {"splits": {"test": {"roc_auc": 0.9}}}
        ci_rpt = {
            "ci_matrix": [
                {"metric": "pr_auc", "split": "test", "point_estimate": 0.85,
                 "ci_lower": 0.78, "ci_upper": 0.91}
            ]
        }
        ev = tmp_path / "eval.json"
        ci = tmp_path / "ci.json"
        _write_json(ev, eval_rpt)
        _write_json(ci, ci_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev),
            "--ci-matrix-report", str(ci),
            "--output", str(out),
        ])
        rc = el.main()
        assert rc == 0
        tex = out.read_text()
        assert "[0.780, 0.910]" in tex


class TestMainAllReports:
    def test_all_reports(self, tmp_path, monkeypatch):
        eval_rpt = {"splits": {"test": {"roc_auc": 0.9, "pr_auc": 0.85}}}
        ms_rpt = {"candidates": [{"family": "lr", "cv_score": 0.8, "selected": True}]}
        ext_rpt = {"cohorts": [{"name": "ext1", "roc_auc": 0.87}]}
        ci_rpt = {"ci_matrix": [{"metric": "roc_auc", "split": "test",
                                  "point_estimate": 0.9, "ci_lower": 0.85, "ci_upper": 0.95}]}
        ev = tmp_path / "eval.json"
        ms = tmp_path / "ms.json"
        ext = tmp_path / "ext.json"
        ci = tmp_path / "ci.json"
        _write_json(ev, eval_rpt)
        _write_json(ms, ms_rpt)
        _write_json(ext, ext_rpt)
        _write_json(ci, ci_rpt)
        out = tmp_path / "tables.tex"
        monkeypatch.setattr("sys.argv", [
            "el", "--evaluation-report", str(ev),
            "--model-selection-report", str(ms),
            "--external-report", str(ext),
            "--ci-matrix-report", str(ci),
            "--output", str(out),
        ])
        rc = el.main()
        assert rc == 0
        tex = out.read_text()
        assert "tab:performance" in tex
        assert "tab:model_selection" in tex
        assert "tab:external" in tex
        assert "tab:ci" in tex
