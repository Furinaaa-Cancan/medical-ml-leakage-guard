#!/usr/bin/env python3
"""
export_review_prompt.py — Export MLGG review criteria as a portable LLM prompt.

Renders the mlgg-review-standard.json into a self-contained system prompt that
any LLM (Claude, GPT-4, Gemini, etc.) can use to review a medical ML paper.
No local deployment required — users paste the output into their LLM of choice.

Usage:
    python3 scripts/export_review_prompt.py --level quick
    python3 scripts/export_review_prompt.py --level standard --output review_prompt.md
    python3 scripts/export_review_prompt.py --level comprehensive --format json
    python3 scripts/export_review_prompt.py --level standard --journal nature_medicine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REFERENCES_DIR = Path(__file__).parent.parent / "references"
REVIEW_STANDARD_PATH = REFERENCES_DIR / "mlgg-review-standard.json"
JOURNAL_STANDARDS_PATH = REFERENCES_DIR / "journal-rigor-standards.json"
LITERATURE_KB_PATH = REFERENCES_DIR / "literature-knowledge-base.json"

LEVEL_LABELS = {
    "quick": "Quick (18 red-line criteria, ~5 min)",
    "standard": "Standard (53 criteria, ~30 min)",
    "comprehensive": "Comprehensive (76 criteria, ~2 hr)",
}

LEVEL_HIERARCHY = {"quick": 0, "standard": 1, "comprehensive": 2}

SEVERITY_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MLGG review criteria as a portable LLM prompt."
    )
    parser.add_argument(
        "--level",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Review depth: quick (18 red-line), standard (53), comprehensive (76 criteria).",
    )
    parser.add_argument(
        "--journal",
        choices=["nature_medicine", "lancet_digital_health", "jama", "bmj", "npj_digital_medicine"],
        default=None,
        help="Optional: target journal. Adds journal-specific mandatory requirements section.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (for pasting into LLM chat) or json (for API use).",
    )
    parser.add_argument(
        "--include-literature",
        action="store_true",
        help="Append key literature citations to the prompt.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_criteria_for_level(
    dimensions: Dict[str, Any], level: str
) -> List[Dict[str, Any]]:
    """Return all criteria at or below the given review level."""
    max_rank = LEVEL_HIERARCHY[level]
    result = []
    for dim_key, dim in dimensions.items():
        for criterion in dim["criteria"]:
            crit_rank = LEVEL_HIERARCHY.get(criterion.get("level", "comprehensive"), 2)
            if crit_rank <= max_rank:
                result.append({"dim_key": dim_key, "dim": dim, "criterion": criterion})
    return result


def render_markdown_prompt(
    standard: Dict[str, Any],
    level: str,
    journal_data: Optional[Dict[str, Any]],
    journal_name: Optional[str],
    include_literature: bool,
    lit_kb: Optional[Dict[str, Any]],
) -> str:
    dimensions = standard["dimensions"]
    criteria_flat = get_criteria_for_level(dimensions, level)
    total_weight = sum(d["weight"] for d in dimensions.values())

    journal_section = ""
    if journal_data and journal_name:
        full_name = journal_data.get("full_name", journal_name)
        min_score = journal_data.get("min_score", "N/A")
        required_gates = journal_data.get("required_gates", [])
        ml_req = journal_data.get("ml_prediction_requirements", {})
        mandatory = ml_req.get("mandatory", [])
        journal_section = f"""
## Target Journal: {full_name}

**Minimum acceptable score**: {min_score}/100
**Required gates**: {', '.join(f'`{g}`' for g in required_gates)}

### Mandatory Requirements for {full_name}

| # | Requirement | Gate | Dimension |
|---|-------------|------|-----------|
"""
        for i, req in enumerate(mandatory, 1):
            gate = req.get("gate") or "manual review"
            dim = req.get("dimension", "—")
            journal_section += f"| {i} | {req['requirement']} | `{gate}` | D{dim} |\n"
        journal_section += "\n> **IMPORTANT**: If any mandatory requirement is absent, mark it as FAIL regardless of other scores.\n"

    # Build criteria by dimension
    dim_sections = []
    for dim_key, dim in dimensions.items():
        dim_criteria = [
            c for c in criteria_flat if c["dim_key"] == dim_key
        ]
        if not dim_criteria:
            continue

        weight = dim["weight"]
        dim_text = f"\n### D{dim['id']} — {dim['name']} ({dim['name_zh']})  ·  weight: {weight}/{total_weight}\n\n"
        for entry in dim_criteria:
            c = entry["criterion"]
            sev_icon = SEVERITY_EMOJI.get(c.get("severity", "MEDIUM"), "🟡")
            level_tag = c.get("level", "").upper()
            crit_text = (
                f"**{c['id']}** [{level_tag}] {sev_icon} {c['text']}\n"
                f"- *Chinese*: {c.get('text_zh', '')}\n"
                f"- *Pass condition*: {c.get('pass_condition', 'See standard')}\n"
                f"- *What to look for*: {c.get('evidence_required', 'See gate report')}\n\n"
            )
            dim_text += crit_text
        dim_sections.append(dim_text)

    lit_section = ""
    if include_literature and lit_kb:
        entries = lit_kb.get("entries", [])
        # Only include entries relevant to the level
        lit_section = "\n## Key Literature References\n\n"
        for e in entries[:20]:
            lit_section += (
                f"- **{e['id']}** [{e.get('year','')}] {e['title'][:80]}... "
                f"(*{e.get('journal','')}*, IF≈{e.get('impact_factor','')})\n"
            )

    n_criteria = len(criteria_flat)
    prompt = f"""# MLGG Medical ML Paper Review — {LEVEL_LABELS[level]}

## Your Role

You are a rigorous peer reviewer for top-tier medical journals (Nature Medicine, JAMA, Lancet Digital Health, BMJ). Your task is to evaluate the provided medical machine learning paper using the structured criteria below.

You apply the **MLGG (ML Leakage Guard) Review Standard v{standard.get('version', '1.0')}**, which covers {n_criteria} criteria across {len(dimensions)} dimensions totalling {total_weight} points.
{journal_section}
## Scoring Instructions

For EACH criterion:
1. **verdict**: `PASS` / `FAIL` / `UNCLEAR` / `NOT_APPLICABLE`
2. **evidence_quote**: Exact sentence(s) from the paper supporting your verdict (required for PASS; required for FAIL to show what's missing)
3. **confidence**: 0.0–1.0 (how certain you are given the available text)
4. **note**: Optional brief explanation (required when verdict is UNCLEAR or NOT_APPLICABLE)

**Scoring rules**:
- PASS: full points for criterion
- UNCLEAR: half points
- FAIL: 0 points (if CRITICAL severity → flag as blocking)
- NOT_APPLICABLE: excluded from denominator

## Criteria
{"".join(dim_sections)}
## Required Output Format

Respond with a single JSON object following this exact schema:

```json
{{
  "paper_title": "...",
  "review_level": "{level}",
  "reviewer_model": "your-model-name",
  "review_date": "YYYY-MM-DD",
  "dimensions": {{
    "data_integrity": {{
      "score": 0,
      "max_score": {next((d['weight'] for d in dimensions.values() if d['id']==1), 12)},
      "criteria": [
        {{
          "id": "D1.1",
          "verdict": "PASS",
          "evidence_quote": "exact sentence from paper",
          "confidence": 0.95,
          "note": ""
        }}
      ]
    }}
  }},
  "total_score": 0,
  "max_possible_score": {total_weight},
  "publication_grade": "top-journal | solid | major-issues | not-publishable",
  "critical_failures": ["D2.1 — no temporal split documented", "..."],
  "blocking_issues": [],
  "top_strengths": [],
  "top_weaknesses": [],
  "recommended_fixes": [],
  "summary": "2-3 sentence overall assessment"
}}
```

**Grade thresholds**: top-journal ≥90 · solid 75–89 · major-issues 60–74 · not-publishable <60
{lit_section}
## Now review the paper below

---

[PASTE PAPER TEXT OR METHODS SECTION HERE]
"""
    return prompt


def render_json_prompt(
    standard: Dict[str, Any],
    level: str,
    journal_data: Optional[Dict[str, Any]],
    journal_name: Optional[str],
) -> str:
    dimensions = standard["dimensions"]
    criteria_flat = get_criteria_for_level(dimensions, level)

    criteria_list = []
    for entry in criteria_flat:
        c = entry["criterion"]
        criteria_list.append(
            {
                "id": c["id"],
                "dimension": entry["dim"]["name"],
                "weight": entry["dim"]["weight"],
                "level": c["level"],
                "severity": c.get("severity"),
                "text": c["text"],
                "pass_condition": c.get("pass_condition"),
                "evidence_required": c.get("evidence_required"),
            }
        )

    payload: Dict[str, Any] = {
        "prompt_version": "mlgg-review-prompt.v1",
        "review_level": level,
        "total_criteria": len(criteria_list),
        "dimensions": [
            {"id": dim["id"], "name": dim["name"], "weight": dim["weight"]}
            for dim in dimensions.values()
        ],
        "criteria": criteria_list,
        "system_prompt": (
            "You are a rigorous peer reviewer for top-tier medical journals. "
            "Evaluate the provided paper using the criteria list. "
            "For each criterion output: verdict (PASS/FAIL/UNCLEAR/NOT_APPLICABLE), "
            "evidence_quote, confidence (0-1), note. "
            "Return a single JSON with all verdicts and a total_score."
        ),
        "output_schema": {
            "paper_title": "string",
            "review_level": level,
            "verdicts": [{"id": "D1.1", "verdict": "PASS", "evidence_quote": "...", "confidence": 0.9, "note": ""}],
            "total_score": "number",
            "publication_grade": "top-journal | solid | major-issues | not-publishable",
            "critical_failures": [],
            "summary": "string",
        },
    }
    if journal_data and journal_name:
        payload["target_journal"] = {
            "key": journal_name,
            "full_name": journal_data.get("full_name"),
            "min_score": journal_data.get("min_score"),
            "required_gates": journal_data.get("required_gates", []),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    standard = load_json(REVIEW_STANDARD_PATH)
    journal_standards = load_json(JOURNAL_STANDARDS_PATH)
    lit_kb = load_json(LITERATURE_KB_PATH) if args.include_literature else None

    journal_data: Optional[Dict[str, Any]] = None
    if args.journal:
        journal_data = journal_standards.get("journals", {}).get(args.journal)

    if args.format == "markdown":
        output = render_markdown_prompt(
            standard=standard,
            level=args.level,
            journal_data=journal_data,
            journal_name=args.journal,
            include_literature=args.include_literature,
            lit_kb=lit_kb,
        )
    else:
        output = render_json_prompt(
            standard=standard,
            level=args.level,
            journal_data=journal_data,
            journal_name=args.journal,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        ext = ".md" if args.format == "markdown" else ".json"
        final_path = out_path if out_path.suffix else out_path.with_suffix(ext)
        if final_path != out_path:
            out_path.rename(final_path)
            out_path = final_path
        n_criteria = len(get_criteria_for_level(standard["dimensions"], args.level))
        journal_tag = f" ({args.journal})" if args.journal else ""
        print(f"Exported {n_criteria}-criteria {args.level} review prompt{journal_tag} → {out_path}", file=sys.stderr)
        print(f"Paste the contents of {out_path} into any LLM (Claude, GPT-4, Gemini) followed by your paper text.", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
