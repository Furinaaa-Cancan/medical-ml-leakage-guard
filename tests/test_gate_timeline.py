"""Tests for scripts/gate_timeline.py.

Covers helpers (load_json, _parse_utc, _safe_float), gate entry extraction,
directory scanning, analysis (sorting, summary, bottlenecks), output formatting,
and CLI integration via direct main().
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import gate_timeline as gt


def _write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_gate_report(
    gate_name: str = "test_gate",
    status: str = "pass",
    timestamp: str = "2025-01-15T10:00:00Z",
    duration: float = 1.5,
    failures: int = 0,
    warnings: int = 0,
) -> dict:
    return {
        "gate_name": gate_name,
        "status": status,
        "execution_timestamp_utc": timestamp,
        "execution_time_seconds": duration,
        "failure_count": failures,
        "warning_count": warnings,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


class TestLoadJson:
    def test_valid(self, tmp_path):
        _write_json(tmp_path / "r.json", {"a": 1})
        assert gt.load_json(tmp_path / "r.json") == {"a": 1}

    def test_missing(self, tmp_path):
        assert gt.load_json(tmp_path / "nope.json") is None

    def test_invalid(self, tmp_path):
        (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
        assert gt.load_json(tmp_path / "bad.json") is None

    def test_non_dict(self, tmp_path):
        _write_json(tmp_path / "arr.json", [1, 2])
        assert gt.load_json(tmp_path / "arr.json") is None


class TestParseUtc:
    def test_z_suffix(self):
        dt = gt._parse_utc("2025-01-15T10:00:00Z")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 1
        assert dt.hour == 10

    def test_offset_suffix(self):
        dt = gt._parse_utc("2025-01-15T10:00:00+00:00")
        assert dt is not None

    def test_invalid(self):
        assert gt._parse_utc("not-a-date") is None

    def test_none(self):
        assert gt._parse_utc(None) is None

    def test_non_string(self):
        assert gt._parse_utc(12345) is None


class TestSafeFloat:
    def test_int(self):
        assert gt._safe_float(5) == 5.0

    def test_float(self):
        assert gt._safe_float(1.5) == 1.5

    def test_bool(self):
        assert gt._safe_float(True) is None

    def test_nan(self):
        assert gt._safe_float(float("nan")) is None

    def test_inf(self):
        assert gt._safe_float(float("inf")) is None

    def test_none(self):
        assert gt._safe_float(None) is None

    def test_string(self):
        assert gt._safe_float("1.5") is None


# ── extract_gate_entry ──────────────────────────────────────────────────────


class TestExtractGateEntry:
    def test_full_report(self, tmp_path):
        p = _write_json(tmp_path / "test_gate_report.json", _make_gate_report())
        e = gt.extract_gate_entry(p)
        assert e is not None
        assert e["gate_name"] == "test_gate"
        assert e["status"] == "pass"
        assert e["duration_seconds"] == 1.5
        assert e["timestamp_utc"] is not None

    def test_missing_file(self, tmp_path):
        assert gt.extract_gate_entry(tmp_path / "nope.json") is None

    def test_infer_gate_name(self, tmp_path):
        p = _write_json(tmp_path / "leakage_gate_report.json", {
            "status": "pass",
            "execution_time_seconds": 0.5,
        })
        e = gt.extract_gate_entry(p)
        assert e["gate_name"] == "leakage_gate"

    def test_no_timestamp(self, tmp_path):
        p = _write_json(tmp_path / "gate_report.json", {
            "gate_name": "my_gate",
            "status": "fail",
            "execution_time_seconds": 2.0,
        })
        e = gt.extract_gate_entry(p)
        assert e["timestamp_utc"] is None
        assert e["duration_seconds"] == 2.0

    def test_no_duration(self, tmp_path):
        p = _write_json(tmp_path / "gate_report.json", {
            "gate_name": "my_gate",
            "status": "pass",
            "execution_timestamp_utc": "2025-01-15T10:00:00Z",
        })
        e = gt.extract_gate_entry(p)
        assert e["duration_seconds"] is None

    def test_infer_name_no_report_suffix(self, tmp_path):
        p = _write_json(tmp_path / "custom_report.json", {
            "status": "pass",
        })
        e = gt.extract_gate_entry(p)
        assert e["gate_name"] == "custom"


# ── scan_evidence_dir ───────────────────────────────────────────────────────


class TestScanEvidenceDir:
    def test_normal(self, tmp_path):
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report(gate_name="a", duration=1.0))
        _write_json(tmp_path / "gate_b_report.json", _make_gate_report(gate_name="b", duration=2.0))
        entries = gt.scan_evidence_dir(tmp_path)
        assert len(entries) == 2

    def test_empty_dir(self, tmp_path):
        assert gt.scan_evidence_dir(tmp_path) == []

    def test_non_dir(self, tmp_path):
        assert gt.scan_evidence_dir(tmp_path / "nope") == []

    def test_ignores_non_report(self, tmp_path):
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report(gate_name="a"))
        _write_json(tmp_path / "config.json", {"key": "val"})
        entries = gt.scan_evidence_dir(tmp_path)
        assert len(entries) == 1

    def test_ignores_invalid_json(self, tmp_path):
        _write_json(tmp_path / "gate_a_report.json", _make_gate_report(gate_name="a"))
        (tmp_path / "gate_b_report.json").write_text("{bad", encoding="utf-8")
        entries = gt.scan_evidence_dir(tmp_path)
        assert len(entries) == 1


# ── sort_by_timestamp ──────────────────────────────────────────────────────


class TestSortByTimestamp:
    def test_sorts_by_time(self):
        entries = [
            {"gate_name": "b", "_timestamp_obj": datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)},
            {"gate_name": "a", "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)},
        ]
        result = gt.sort_by_timestamp(entries)
        assert result[0]["gate_name"] == "a"
        assert result[1]["gate_name"] == "b"

    def test_none_timestamp_last(self):
        entries = [
            {"gate_name": "b", "_timestamp_obj": None},
            {"gate_name": "a", "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)},
        ]
        result = gt.sort_by_timestamp(entries)
        assert result[0]["gate_name"] == "a"
        assert result[1]["gate_name"] == "b"


# ── compute_summary ────────────────────────────────────────────────────────


class TestComputeSummary:
    def test_full(self):
        entries = [
            {
                "gate_name": "a", "status": "pass", "duration_seconds": 1.0,
                "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            },
            {
                "gate_name": "b", "status": "fail", "duration_seconds": 3.0,
                "_timestamp_obj": datetime(2025, 1, 15, 10, 0, 10, tzinfo=timezone.utc),
            },
        ]
        s = gt.compute_summary(entries)
        assert s["total_gates"] == 2
        assert s["status_counts"] == {"pass": 1, "fail": 1}
        assert s["total_duration_seconds"] == 4.0
        assert s["average_duration_seconds"] == 2.0
        assert s["max_duration_seconds"] == 3.0
        assert s["min_duration_seconds"] == 1.0
        assert s["wall_clock_span_seconds"] == 10.0

    def test_empty(self):
        s = gt.compute_summary([])
        assert s["total_gates"] == 0
        assert s["total_duration_seconds"] == 0.0
        assert s["wall_clock_span_seconds"] is None

    def test_single_timestamp(self):
        entries = [
            {
                "gate_name": "a", "status": "pass", "duration_seconds": 1.0,
                "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            },
        ]
        s = gt.compute_summary(entries)
        assert s["wall_clock_span_seconds"] is None

    def test_no_durations(self):
        entries = [
            {"gate_name": "a", "status": "pass", "duration_seconds": None, "_timestamp_obj": None},
        ]
        s = gt.compute_summary(entries)
        assert s["total_duration_seconds"] == 0.0
        assert s["gates_with_duration"] == 0


# ── find_bottlenecks ───────────────────────────────────────────────────────


class TestFindBottlenecks:
    def test_top_n(self):
        entries = [
            {"gate_name": "fast", "duration_seconds": 0.5, "status": "pass"},
            {"gate_name": "slow", "duration_seconds": 5.0, "status": "pass"},
            {"gate_name": "medium", "duration_seconds": 2.0, "status": "fail"},
        ]
        result = gt.find_bottlenecks(entries, top_n=2)
        assert len(result) == 2
        assert result[0]["gate_name"] == "slow"
        assert result[1]["gate_name"] == "medium"

    def test_no_durations(self):
        entries = [
            {"gate_name": "a", "duration_seconds": None, "status": "pass"},
        ]
        assert gt.find_bottlenecks(entries) == []

    def test_empty(self):
        assert gt.find_bottlenecks([]) == []


# ── output formatting ──────────────────────────────────────────────────────


class TestToJson:
    def test_structure(self):
        entries = [
            {
                "gate_name": "a", "file": "a_report.json", "status": "pass",
                "timestamp_utc": "2025-01-15T10:00:00+00:00",
                "duration_seconds": 1.0, "failure_count": 0, "warning_count": 0,
                "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            },
        ]
        summary = gt.compute_summary(entries)
        bottlenecks = gt.find_bottlenecks(entries)
        result = json.loads(gt.to_json(entries, summary, bottlenecks))
        assert "summary" in result
        assert "timeline" in result
        assert "bottlenecks" in result
        # Internal keys should be stripped
        assert "_timestamp_obj" not in result["timeline"][0]


class TestToText:
    def test_structure(self):
        entries = [
            {
                "gate_name": "a", "status": "pass",
                "timestamp_utc": "2025-01-15T10:00:00+00:00",
                "duration_seconds": 1.5,
                "_timestamp_obj": datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc),
            },
        ]
        summary = gt.compute_summary(entries)
        bottlenecks = gt.find_bottlenecks(entries)
        text = gt.to_text(entries, summary, bottlenecks)
        assert "Gate Timeline Analysis" in text
        assert "Total gates: 1" in text
        assert "1.500s" in text


# ── CLI integration (direct main()) ────────────────────────────────────────


def _populate_evidence(tmp_path: Path) -> Path:
    """Create an evidence directory with multiple gate reports."""
    ev = tmp_path / "evidence"
    ev.mkdir()
    _write_json(ev / "schema_preflight_report.json", _make_gate_report(
        gate_name="schema_preflight", status="pass",
        timestamp="2025-01-15T10:00:00Z", duration=0.8,
    ))
    _write_json(ev / "leakage_gate_report.json", _make_gate_report(
        gate_name="leakage_gate", status="pass",
        timestamp="2025-01-15T10:00:02Z", duration=3.2,
    ))
    _write_json(ev / "robustness_gate_report.json", _make_gate_report(
        gate_name="robustness_gate", status="fail",
        timestamp="2025-01-15T10:00:06Z", duration=12.5,
        failures=2,
    ))
    return ev


class TestMainTextOutput:
    def test_text_stdout(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gt", "--evidence-dir", str(ev)])
        rc = gt.main()
        assert rc == 0
        text = capsys.readouterr().out
        assert "Gate Timeline Analysis" in text
        assert "robustness_gate" in text
        assert "Total gates: 3" in text


class TestMainJsonOutput:
    def test_json_stdout(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", ["gt", "--evidence-dir", str(ev), "--json"])
        rc = gt.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["total_gates"] == 3
        assert len(data["timeline"]) == 3
        assert len(data["bottlenecks"]) == 3


class TestMainOutputFile:
    def test_write_to_file(self, tmp_path, monkeypatch):
        ev = _populate_evidence(tmp_path)
        out = tmp_path / "timeline.json"
        monkeypatch.setattr("sys.argv", [
            "gt", "--evidence-dir", str(ev), "--json", "--output", str(out),
        ])
        rc = gt.main()
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["summary"]["total_gates"] == 3


class TestMainMissingDir:
    def test_missing_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["gt", "--evidence-dir", str(tmp_path / "nope")])
        rc = gt.main()
        assert rc == 1


class TestMainEmptyDir:
    def test_empty_evidence(self, tmp_path, monkeypatch):
        ev = tmp_path / "evidence"
        ev.mkdir()
        monkeypatch.setattr("sys.argv", ["gt", "--evidence-dir", str(ev)])
        rc = gt.main()
        assert rc == 1


class TestMainTopN:
    def test_custom_top(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "gt", "--evidence-dir", str(ev), "--json", "--top", "1",
        ])
        rc = gt.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert len(data["bottlenecks"]) == 1
        assert data["bottlenecks"][0]["gate_name"] == "robustness_gate"


class TestMainBottleneckOrder:
    def test_bottleneck_sorted_desc(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "gt", "--evidence-dir", str(ev), "--json",
        ])
        rc = gt.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        durations = [b["duration_seconds"] for b in data["bottlenecks"]]
        assert durations == sorted(durations, reverse=True)


class TestMainTimelineOrder:
    def test_timeline_sorted_by_timestamp(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "gt", "--evidence-dir", str(ev), "--json",
        ])
        rc = gt.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        timestamps = [e["timestamp_utc"] for e in data["timeline"]]
        assert timestamps == sorted(timestamps)


class TestMainWallClockSpan:
    def test_wall_clock_computed(self, tmp_path, monkeypatch, capsys):
        ev = _populate_evidence(tmp_path)
        monkeypatch.setattr("sys.argv", [
            "gt", "--evidence-dir", str(ev), "--json",
        ])
        rc = gt.main()
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["wall_clock_span_seconds"] == 6.0
