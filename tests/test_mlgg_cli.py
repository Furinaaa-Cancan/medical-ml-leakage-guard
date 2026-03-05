"""Tests for scripts/mlgg.py command routing.

Covers helper functions (_extract_option_value, _find_subcommand,
passthrough_contains_flag, emit_fail, build_parser), subcommand dispatch,
--dry-run, --interactive routing, preset blocked flags, unknown subcommand,
and --help forwarding.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "mlgg.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import mlgg


# ── helper functions ─────────────────────────────────────────────────────────

class TestExtractOptionValue:
    def test_basic(self):
        assert mlgg._extract_option_value(["--python", "/usr/bin/python3"], "--python", "default") == "/usr/bin/python3"

    def test_equals_form(self):
        assert mlgg._extract_option_value(["--python=/usr/bin/python3"], "--python", "default") == "/usr/bin/python3"

    def test_missing(self):
        assert mlgg._extract_option_value(["--other", "val"], "--python", "fallback") == "fallback"

    def test_empty_value(self):
        assert mlgg._extract_option_value(["--python", ""], "--python", "default") == "default"


class TestFindSubcommand:
    def test_found(self):
        result = mlgg._find_subcommand(["onboarding", "--help"])
        assert result is not None
        assert result[1] == "onboarding"

    def test_not_found(self):
        assert mlgg._find_subcommand(["--help"]) is None

    def test_multiple(self):
        result = mlgg._find_subcommand(["--flag", "train", "--test"])
        assert result is not None
        assert result[1] == "train"


class TestPassthroughContainsFlag:
    def test_present(self):
        assert mlgg.passthrough_contains_flag(["--include-stress-cases", "--other"], "--include-stress-cases")

    def test_equals(self):
        assert mlgg.passthrough_contains_flag(["--stress-case-id=abc"], "--stress-case-id")

    def test_absent(self):
        assert not mlgg.passthrough_contains_flag(["--other"], "--include-stress-cases")


class TestBuildParser:
    def test_returns_parser(self):
        parser = mlgg.build_parser()
        assert parser is not None

    def test_known_subcommands(self):
        parser = mlgg.build_parser()
        args, _ = parser.parse_known_args(["doctor"])
        assert args.subcommand == "doctor"


# ── CLI integration ──────────────────────────────────────────────────────────

class TestCLIDryRun:
    def test_doctor_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "doctor", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "env_doctor" in proc.stdout

    def test_preflight_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "preflight", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "schema_preflight" in proc.stdout

    def test_split_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "split", "--dry-run", "--", "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0

    def test_train_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "train", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0

    def test_interactive_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "interactive", "--command", "init", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0

    def test_play_dry_run(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "play", "--dry-run"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0


class TestCLIUnknown:
    def test_unknown_subcommand(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "nonexistent"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode != 0


class TestCLIPresetBlocked:
    def test_authority_release_blocked(self):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "authority-release", "--dry-run",
            "--include-stress-cases",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "override" in proc.stderr.lower() or "blocked" in proc.stderr.lower() or "forbidden" in proc.stderr.lower()


class TestCLIErrorJson:
    def test_error_json_output(self):
        cmd = [
            sys.executable, str(GATE_SCRIPT),
            "authority-release", "--error-json",
            "--include-stress-cases",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2
        assert "mlgg_error.v1" in proc.stderr


class TestCLIHelp:
    def test_main_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "onboarding" in proc.stdout

    def test_doctor_help(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "doctor", "--help"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0


class TestCLIInteractiveRouting:
    def test_interactive_missing_command(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "interactive"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 2


class TestCLIPassthrough:
    def test_passthrough_forwarded(self):
        cmd = [sys.executable, str(GATE_SCRIPT), "doctor", "--dry-run", "--", "--report", "/tmp/r.json"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SCRIPTS_DIR))
        assert proc.returncode == 0
        assert "--report" in proc.stdout
