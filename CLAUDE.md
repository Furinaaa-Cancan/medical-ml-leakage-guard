# CLAUDE.md — Agent Operating Protocol for ML Leakage Guard

> This file instructs Claude Code (and compatible agents) how to operate within this repository.

## Identity

You are operating inside **ml-leakage-guard**, a publication-grade medical binary classification framework with 31 fail-closed anti-data-leakage gates. Your primary role is to help researchers produce top-journal-quality (Nature Medicine, Lancet Digital Health, JAMA, BMJ) prediction evidence with machine-verifiable rigor.

## Quick Dispatch

Read `SKILL.md` first — it contains the full operational playbook including:
- Intent → command mapping table
- 31-gate pipeline execution order
- Input/output contracts
- Medical non-negotiable rules

## Core Principles

1. **Fail-Closed First**: Never silently pass. Any ambiguity → fail + explain.
2. **Evidence Over Claims**: Every assertion must have a machine-checkable artifact.
3. **No Data Leakage**: Never fit on test, never tune on test, never peek at test.
4. **Self-Improving**: When you encounter a new error pattern, append it to `references/error-knowledge-base.json`.
5. **Quantitative Judgment**: Use the 10-dimension scoring rubric (see below) when evaluating any ML project.

## Agent Workflow Modes

### Mode A: Build New Project (从零构建)
When user says "帮我搭建一个预测项目" / "build a prediction project":
1. `python3 scripts/mlgg.py onboarding --mode auto`
2. Guide through data preparation → splitting → training → audit → publication gate
3. Target: all 31 gates pass in strict mode

### Mode B: Audit Existing Project (审计他人项目)
When user says "帮我审查这个项目" / "evaluate this ML project":
1. Run `python3 scripts/audit_external_project.py --project-dir <dir>`
2. Generate quantitative scores across 10 dimensions
3. Produce remediation plan with priority ordering

### Mode C: Incremental Fix (增量修复)
When user says "这个 gate 失败了" / "fix this failure":
1. Read the gate report JSON
2. Look up error code in `references/error-knowledge-base.json`
3. Apply fix → re-run gate → verify pass

### Mode D: Batch Review (批量评审)
When user says "帮我批量评审这些项目" / "batch review these projects":
1. Prepare a manifest JSON (see `references/batch-manifest.example.json`)
2. Run `python3 scripts/mlgg.py batch-review --manifest <manifest.json> --target-journal nature_medicine`
3. Output: comparison matrix, cross-cutting gap analysis, aggregated remediation priorities
4. Use `--format markdown` for human-readable or `--summary-csv` for spreadsheet analysis

### Literature Query Protocol
When you need to find literature support for a review criterion or gate:
1. Read `references/literature-knowledge-base.json`
2. Search by `category`, `gates_implementing`, or `dimensions_affected`
3. Cite entries by their `LIT-NNN` ID in review reports
4. Only add entries meeting quality criteria: IF>10 journal, EQUATOR/Cochrane guideline, or PRISMA systematic review

### MLGG Review Standard
When performing structured reviews, reference `references/mlgg-review-standard.json`:
- **Quick review** (5 min): 18 critical red-line criteria
- **Standard review** (30 min): 53 cumulative criteria for conference/journal submissions
- **Comprehensive review** (2 hr): 76 cumulative criteria for Nature Medicine / JAMA / BMJ level

## 12-Dimension Scoring Rubric (100-point scale)

Used for quantitative evaluation of any medical ML project:

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| 1. Data Integrity | 12 | Split isolation, patient-level disjoint, temporal ordering, no row overlap |
| 2. Leakage Prevention | 15 | No target leakage, no definition-variable leakage, no lineage leakage, no post-index features |
| 3. Pipeline Isolation | 12 | Preprocessor train-only fit, imputer isolation, resampling train-only |
| 4. Model Selection Rigor | 10 | Candidate pool ≥3, one-SE rule, no test peeking, required baseline |
| 5. Statistical Validity | 12 | Bootstrap CI, permutation test, calibration, DCA, metric consistency |
| 6. Generalization Evidence | 10 | Train-test gap, external cohort, transport-drop CI, seed stability |
| 7. Clinical Completeness | 7 | Full metric panel, confusion matrix consistency, threshold feasibility |
| 8. Reporting Standards | 7 | TRIPOD+AI, PROBAST+AI, STARD-AI, exclusion criteria, limitation documentation |
| 9. Reproducibility | 6 | Seed logging, version tracking, execution attestation, manifest lock |
| 10. Security & Provenance | 3 | Model signing, artifact integrity, sensitive data protection |
| 11. Fairness & Equity | 3 | Equalized odds gap, disparate impact ratio, subgroup performance minimums |
| 12. Sample Size Adequacy | 3 | EPV ≥10, shrinkage factor ≥0.90, minimum 100 events/non-events |

**Score interpretation**:
- 90-100: Publication-grade (顶刊级)
- 75-89: Solid but gaps remain (需补充)
- 60-74: Major issues (重大缺陷)
- <60: Not publishable (不可发表)

## Error Knowledge Base Protocol

When you encounter a new error during any operation:
1. Check if it exists in `references/error-knowledge-base.json`
2. If not, append a new entry with: `{code, symptom, root_cause, fix, prevention, first_seen, gate}`
3. This creates a self-improving diagnostic system

## Code Quality Standards

- All Python files: type annotations, Google-style docstrings
- All gate scripts: uniform CLI contract (`--report`, `--strict`, exit 0/2)
- All gate scripts: `finish()` must use `bool(failures) or (args.strict and bool(warnings))`
- All `to_float()`: must include `math.isfinite` guard
- Tests: pytest style, ≥85% coverage per gate, use `tmp_path` fixture

## File Layout

```
scripts/          # Gate scripts, training, orchestrators, analysis tools
tests/            # 2900+ pytest tests
examples/         # Dataset downloaders
experiments/      # Authority E2E benchmarks
references/       # JSON templates, checklists, knowledge base
docs/             # Architecture and review docs
.github/workflows # CI/CD
```

## Key Commands

```bash
# New project
python3 scripts/mlgg.py play

# Strict audit
python3 scripts/mlgg.py workflow --request configs/request.json --strict

# Audit external project
python3 scripts/audit_external_project.py --project-dir /path/to/project

# Run tests
python3 -m pytest tests/ -q --tb=short

# Check environment
python3 scripts/mlgg.py doctor
```
