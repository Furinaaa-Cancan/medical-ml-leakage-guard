"""Tests for scripts/export_review_prompt.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
GATE_SCRIPT = SCRIPTS_DIR / "export_review_prompt.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import export_review_prompt as erp


# ── helper ───────────────────────────────────────────────────────────────────

def _load_standard():
    return erp.load_json(erp.REVIEW_STANDARD_PATH)


# ── get_criteria_for_level ────────────────────────────────────────────────────

class TestGetCriteriaForLevel:
    def test_quick_subset_of_standard(self):
        standard = _load_standard()
        quick = erp.get_criteria_for_level(standard["dimensions"], "quick")
        standard_crits = erp.get_criteria_for_level(standard["dimensions"], "standard")
        # Quick must be a strict subset
        quick_ids = {c["criterion"]["id"] for c in quick}
        standard_ids = {c["criterion"]["id"] for c in standard_crits}
        assert quick_ids.issubset(standard_ids)

    def test_standard_subset_of_comprehensive(self):
        standard = _load_standard()
        std = erp.get_criteria_for_level(standard["dimensions"], "standard")
        comp = erp.get_criteria_for_level(standard["dimensions"], "comprehensive")
        std_ids = {c["criterion"]["id"] for c in std}
        comp_ids = {c["criterion"]["id"] for c in comp}
        assert std_ids.issubset(comp_ids)

    def test_quick_has_criteria(self):
        standard = _load_standard()
        quick = erp.get_criteria_for_level(standard["dimensions"], "quick")
        assert len(quick) > 0

    def test_comprehensive_max_criteria(self):
        standard = _load_standard()
        comp = erp.get_criteria_for_level(standard["dimensions"], "comprehensive")
        # comprehensive must have the most
        quick = erp.get_criteria_for_level(standard["dimensions"], "quick")
        assert len(comp) >= len(quick)


# ── render_markdown_prompt ────────────────────────────────────────────────────

class TestRenderMarkdownPrompt:
    def test_returns_string(self):
        standard = _load_standard()
        result = erp.render_markdown_prompt(
            standard=standard,
            level="quick",
            journal_data=None,
            journal_name=None,
            include_literature=False,
            lit_kb=None,
        )
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_criteria_heading(self):
        standard = _load_standard()
        result = erp.render_markdown_prompt(
            standard=standard, level="standard",
            journal_data=None, journal_name=None,
            include_literature=False, lit_kb=None,
        )
        assert "## Criteria" in result or "##" in result

    def test_contains_role_section(self):
        standard = _load_standard()
        result = erp.render_markdown_prompt(
            standard=standard, level="quick",
            journal_data=None, journal_name=None,
            include_literature=False, lit_kb=None,
        )
        assert "Your Role" in result or "peer reviewer" in result.lower()

    def test_journal_section_included(self):
        standard = _load_standard()
        journal_standards = erp.load_json(erp.JOURNAL_STANDARDS_PATH)
        journal_data = journal_standards.get("journals", {}).get("nature_medicine")
        result = erp.render_markdown_prompt(
            standard=standard, level="comprehensive",
            journal_data=journal_data, journal_name="nature_medicine",
            include_literature=False, lit_kb=None,
        )
        assert "Nature Medicine" in result or "nature_medicine" in result.lower()

    def test_literature_section_included(self):
        standard = _load_standard()
        lit_kb = erp.load_json(erp.LITERATURE_KB_PATH)
        result = erp.render_markdown_prompt(
            standard=standard, level="standard",
            journal_data=None, journal_name=None,
            include_literature=True, lit_kb=lit_kb,
        )
        assert "Literature" in result or "References" in result

    def test_literature_not_included_by_default(self):
        standard = _load_standard()
        result = erp.render_markdown_prompt(
            standard=standard, level="quick",
            journal_data=None, journal_name=None,
            include_literature=False, lit_kb=None,
        )
        assert "Key Literature References" not in result


# ── render_json_prompt ────────────────────────────────────────────────────────

class TestRenderJsonPrompt:
    def test_returns_valid_json(self):
        standard = _load_standard()
        result = erp.render_json_prompt(
            standard=standard, level="standard",
            journal_data=None, journal_name=None,
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_required_keys(self):
        standard = _load_standard()
        result = erp.render_json_prompt(
            standard=standard, level="quick",
            journal_data=None, journal_name=None,
        )
        parsed = json.loads(result)
        assert "criteria" in parsed
        assert "review_level" in parsed
        assert "total_criteria" in parsed

    def test_criteria_count_matches(self):
        standard = _load_standard()
        criteria_flat = erp.get_criteria_for_level(standard["dimensions"], "standard")
        result = erp.render_json_prompt(
            standard=standard, level="standard",
            journal_data=None, journal_name=None,
        )
        parsed = json.loads(result)
        assert parsed["total_criteria"] == len(criteria_flat)

    def test_journal_section_in_json(self):
        standard = _load_standard()
        journal_standards = erp.load_json(erp.JOURNAL_STANDARDS_PATH)
        journal_data = journal_standards.get("journals", {}).get("jama")
        result = erp.render_json_prompt(
            standard=standard, level="quick",
            journal_data=journal_data, journal_name="jama",
        )
        parsed = json.loads(result)
        assert "target_journal" in parsed


# ── CLI integration ───────────────────────────────────────────────────────────

def _run_cli(*args):
    cmd = [sys.executable, str(GATE_SCRIPT)] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPTS_DIR))


class TestCLI:
    def test_quick_stdout(self):
        result = _run_cli("--level", "quick")
        assert result.returncode == 0
        assert len(result.stdout) > 100

    def test_standard_stdout(self):
        result = _run_cli("--level", "standard")
        assert result.returncode == 0
        assert "Criteria" in result.stdout or "criteria" in result.stdout.lower()

    def test_comprehensive_stdout(self):
        result = _run_cli("--level", "comprehensive")
        assert result.returncode == 0
        assert len(result.stdout) > len(_run_cli("--level", "quick").stdout)

    def test_json_format(self):
        result = _run_cli("--level", "quick", "--format", "json")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "criteria" in parsed

    def test_output_to_file(self, tmp_path):
        out = tmp_path / "prompt.md"
        result = _run_cli("--level", "quick", "--output", str(out))
        assert result.returncode == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_journal_flag(self):
        result = _run_cli("--level", "standard", "--journal", "nature_medicine")
        assert result.returncode == 0
        assert "Nature Medicine" in result.stdout

    def test_include_literature(self):
        result = _run_cli("--level", "quick", "--include-literature")
        assert result.returncode == 0
        assert "Literature" in result.stdout or "References" in result.stdout

    def test_json_with_journal(self):
        result = _run_cli("--level", "quick", "--format", "json", "--journal", "jama")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "target_journal" in parsed
