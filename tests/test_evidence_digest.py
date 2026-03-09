"""Unit tests for scripts/evidence_digest.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evidence_digest import (
    _fmt_float,
    _fmt_pct,
    _get,
    _load,
    extract_digest,
    main as digest_main,
    to_markdown,
)


# ── helpers ──────────────────────────────────────────────────


def _write(directory: Path, filename: str, data: dict) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


# ── _load ────────────────────────────────────────────────────


class TestLoad:
    def test_missing(self, tmp_path):
        assert _load(tmp_path / "nope.json") is None

    def test_valid(self, tmp_path):
        _write(tmp_path, "r.json", {"status": "pass"})
        assert _load(tmp_path / "r.json") == {"status": "pass"}

    def test_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert _load(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        (tmp_path / "arr.json").write_text("[1]", encoding="utf-8")
        assert _load(tmp_path / "arr.json") is None


# ── _get ─────────────────────────────────────────────────────


class TestGet:
    def test_nested(self):
        d = {"a": {"b": {"c": 42}}}
        assert _get(d, "a", "b", "c") == 42

    def test_missing_key(self):
        assert _get({"a": 1}, "b", default="x") == "x"

    def test_none_input(self):
        assert _get(None, "a", default="x") == "x"

    def test_non_dict_intermediate(self):
        assert _get({"a": "string"}, "a", "b", default="x") == "x"


# ── _fmt_pct / _fmt_float ───────────────────────────────────


class TestFormatters:
    def test_fmt_pct_float(self):
        assert _fmt_pct(0.123) == "12.3%"

    def test_fmt_pct_none(self):
        assert _fmt_pct(None) == "—"

    def test_fmt_float_normal(self):
        assert _fmt_float(0.85) == "0.8500"

    def test_fmt_float_none(self):
        assert _fmt_float(None) == "—"

    def test_fmt_float_custom_decimals(self):
        assert _fmt_float(0.85, 2) == "0.85"


# ── extract_digest ───────────────────────────────────────────


class TestExtractDigest:
    def test_empty_dir(self, tmp_path):
        d = extract_digest(tmp_path)
        assert d["schema_version"] == "evidence_digest.v1"
        assert d["pipeline_status"] == "unknown"
        assert d["gates"]["missing"] == 29
        assert d["metrics"] == {}

    def test_with_pipeline_report(self, tmp_path):
        _write(tmp_path, "dag_pipeline_report.json", {"status": "pass"})
        d = extract_digest(tmp_path)
        assert d["pipeline_status"] == "pass"

    def test_with_eval_metrics(self, tmp_path):
        _write(tmp_path, "evaluation_report.json", {
            "roc_auc": 0.92, "pr_auc": 0.88, "sensitivity": 0.85,
        })
        d = extract_digest(tmp_path)
        assert d["metrics"]["roc_auc"] == 0.92
        assert d["metrics"]["pr_auc"] == 0.88

    def test_with_model_selection(self, tmp_path):
        _write(tmp_path, "model_selection_audit_report.json", {
            "status": "pass",
            "summary": {"selected_model_name": "GradientBoosting", "candidate_count": 5},
        })
        d = extract_digest(tmp_path)
        assert d["model"]["selected"] == "GradientBoosting"
        assert d["model"]["candidates_evaluated"] == 5

    def test_with_split_data(self, tmp_path):
        _write(tmp_path, "split_protocol_report.json", {
            "status": "pass",
            "summary": {
                "splits": {
                    "train": {"row_count": 1000, "id_count": 800, "prevalence": 0.15},
                    "test": {"row_count": 200, "id_count": 160, "prevalence": 0.14},
                },
            },
        })
        d = extract_digest(tmp_path)
        assert d["splits"]["train"]["rows"] == 1000
        assert d["splits"]["test"]["prevalence"] == 0.14

    def test_publication_status(self, tmp_path):
        _write(tmp_path, "publication_gate_report.json", {"status": "pass"})
        d = extract_digest(tmp_path)
        assert d["publication_status"] == "pass"

    def test_gate_counts(self, tmp_path):
        _write(tmp_path, "leakage_report.json", {"status": "pass"})
        _write(tmp_path, "split_protocol_report.json", {"status": "fail"})
        d = extract_digest(tmp_path)
        assert d["gates"]["passed"] == 1
        assert d["gates"]["failed"] == 1
        assert d["gates"]["missing"] == 27

    def test_calibration_ece(self, tmp_path):
        _write(tmp_path, "calibration_dca_report.json", {
            "status": "pass", "summary": {"ece": 0.032},
        })
        d = extract_digest(tmp_path)
        assert d["calibration_ece"] == 0.032

    def test_fallback_strict_pipeline(self, tmp_path):
        _write(tmp_path, "strict_pipeline_report.json", {"status": "fail"})
        d = extract_digest(tmp_path)
        assert d["pipeline_status"] == "fail"


# ── to_markdown ──────────────────────────────────────────────


class TestToMarkdown:
    def test_basic_structure(self, tmp_path):
        d = extract_digest(tmp_path)
        md = to_markdown(d)
        assert "# Evidence Digest" in md
        assert "Gate Summary" in md

    def test_with_metrics(self, tmp_path):
        _write(tmp_path, "evaluation_report.json", {"roc_auc": 0.92})
        d = extract_digest(tmp_path)
        md = to_markdown(d)
        assert "Key Metrics" in md
        assert "roc_auc" in md
        assert "0.9200" in md

    def test_with_splits(self, tmp_path):
        _write(tmp_path, "split_protocol_report.json", {
            "status": "pass",
            "summary": {"splits": {"train": {"row_count": 100, "id_count": 80, "prevalence": 0.2}}},
        })
        d = extract_digest(tmp_path)
        md = to_markdown(d)
        assert "Data Splits" in md
        assert "train" in md

    def test_with_model(self, tmp_path):
        _write(tmp_path, "model_selection_audit_report.json", {
            "status": "pass",
            "summary": {"selected_model_name": "XGBoost", "candidate_count": 3},
        })
        d = extract_digest(tmp_path)
        md = to_markdown(d)
        assert "Model" in md
        assert "XGBoost" in md


# ── main() CLI ───────────────────────────────────────────────


class TestDigestMain:
    def test_missing_dir_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "digest", "--evidence-dir", str(tmp_path / "nope"),
        ])
        assert digest_main() == 1

    def test_basic_markdown(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", [
            "digest", "--evidence-dir", str(tmp_path),
        ])
        rc = digest_main()
        assert rc == 0
        assert "# Evidence Digest" in capsys.readouterr().out

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", [
            "digest", "--evidence-dir", str(tmp_path), "--json",
        ])
        rc = digest_main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["schema_version"] == "evidence_digest.v1"

    def test_file_output(self, tmp_path, monkeypatch):
        out = tmp_path / "digest.md"
        monkeypatch.setattr("sys.argv", [
            "digest", "--evidence-dir", str(tmp_path), "--output", str(out),
        ])
        rc = digest_main()
        assert rc == 0
        assert out.exists()
        assert "# Evidence Digest" in out.read_text()

    def test_json_file_output(self, tmp_path, monkeypatch):
        out = tmp_path / "digest.json"
        monkeypatch.setattr("sys.argv", [
            "digest", "--evidence-dir", str(tmp_path),
            "--json", "--output", str(out),
        ])
        rc = digest_main()
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["gates"]["total"] == 29
