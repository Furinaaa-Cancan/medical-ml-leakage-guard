# External Benchmark Comparison (2026-02-24)

## Data Sources Used for Comparison
- Deepchecks open-source framework and leakage-related checks:
  - [deepchecks/deepchecks](https://github.com/deepchecks/deepchecks)
  - [Deepchecks tabular leakage checks docs](https://docs.deepchecks.com/stable/tabular/auto_checks/model_evaluation/plot_data_leakage_report.html)
- DSSG randomization framework for leakage/falsification:
  - [dssg/randomize_your_data](https://github.com/dssg/randomize_your_data)
- Reporting / bias-assessment standards for medical prediction:
  - [TRIPOD+AI (PubMed)](https://pubmed.ncbi.nlm.nih.gov/38754780/)
  - [PROBAST+AI (PubMed)](https://pubmed.ncbi.nlm.nih.gov/40411595/)
  - [STARD-AI (PubMed)](https://pubmed.ncbi.nlm.nih.gov/39898396/)

## Where This Skill Was Already Strong
- Strict fail-closed gating for leakage, split protocol, definition-variable leakage, transitive lineage leakage, imbalance, missingness/imputation, tuning isolation, metric source consistency, permutation falsification, reproducibility lock, and self-critique.
- Publication-grade path already required machine-readable evidence artifacts.

## Gap Identified from External Comparison
- Missing explicit gate for train-vs-holdout covariate separability risk:
  - Deepchecks-style train/test drift signals were not hard-gated.
  - Randomization/adversarial-split thinking existed conceptually but not as dedicated deterministic artifact.

## Implemented Improvement in This Revision
- Added `scripts/covariate_shift_gate.py`.
- Added new strict artifact: `evidence/covariate_shift_report.json`.
- Integrated the new gate into:
  - `scripts/run_strict_pipeline.py`
  - `scripts/publication_gate.py`
  - `scripts/self_critique_gate.py`
  - `SKILL.md` workflow, output contract, strict rules, and script inventory
  - `references/top-tier-rigor-checklist.md`
  - `references/report-template.md`
  - `agents/openai.yaml` default prompt

## Implemented Improvement (Execution Non-Repudiation)
- Added `scripts/execution_attestation_gate.py` to verify detached signatures and signed artifact hashes.
- Added `scripts/generate_execution_attestation.py` to improve personal-user UX for payload/signature/spec generation.
- Added strict artifact: `evidence/execution_attestation_report.json`.
- Integrated attestation gate into strict pipeline + publication/self-critique aggregate gates.
- Updated request contract to require `run_id` + `execution_attestation_spec` for publication-grade claims.
- Added key assurance controls: fingerprint pinning, minimum key bits, key age/expiry checks, and revocation-list enforcement.
- Added trusted timestamp record verification and transparency-log record verification.
- Added signed execution-log attestation verification (`execution_log_attestation`) that binds `training_log` hash to signed payload hash and run identity.
- Added independent-authority policy enforcement for execution-log attestation.

## Remaining Recommended Upgrades (Next Iteration)
- Add machine-checkable TRIPOD+AI / PROBAST+AI compliance gate (reporting + risk-of-bias checklist JSON).
- Add external-validation-specific gate for site/time transport checks when multi-center data are available.
