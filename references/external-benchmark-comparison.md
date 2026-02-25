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
- Attestation and supply-chain integrity standards:
  - [SLSA requirements](https://slsa.dev/spec/v1.0/requirements)
  - [in-toto getting started](https://in-toto.io/docs/getting-started/)
  - [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)

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
- Added cross-role authority distinctness enforcement (`require_distinct_authority_roles`) so timestamp/transparency/execution/execution-log/witness roles cannot reuse authority IDs or signing keys.
- Added publication-gate contract checks that require `authority_role_distinctness` summary to be enforced and passing.

## Implemented Improvement (Witness Quorum)
- Added `witness_quorum` support in execution attestation spec and strict gate.
- Added signed witness-record generation in `scripts/generate_execution_attestation.py`.
- Added fail-closed quorum checks in `scripts/execution_attestation_gate.py`:
  - minimum validated witness count
  - witness signature verification
  - witness payload/study/run binding checks
  - independent witness key/authority checks
  - witness key independence from payload signing key
- Added manifest lock coverage for witness quorum files in `scripts/run_strict_pipeline.py`.
- Added adversarial scenario coverage for witness quorum tampering in `experiments/authority-e2e/run_adversarial_gate_checks.py`.

## Implemented Improvement (Model Selection + Clinical Metrics + Overfitting Gap)
- Added `scripts/model_selection_audit_gate.py`:
  - candidate pool size gate (`>=3`) with required logistic baseline
  - strict non-test model-selection scope checks
  - deterministic one-SE + simplicity replay validation
- Added `scripts/clinical_metrics_gate.py`:
  - required clinical panel enforcement (accuracy/precision/PPV/NPV/sensitivity/specificity/F1/F2-beta/ROC-AUC/PR-AUC/Brier)
  - precision==PPV and confusion-matrix formula consistency checks
- Added `scripts/generalization_gap_gate.py`:
  - train/valid/test directional gap thresholds with warning/fail tiers
- Added policy artifact `references/performance-policy.example.json`.
- Integrated all three gates into strict pipeline + publication gate + self-critique gate.

## Comparison vs SLSA / in-toto / Sigstore (Current Position)
- Current skill already provides:
  - Detached-signature verification for execution payload and authority records.
  - Explicit multi-authority evidence (timestamp, transparency, execution receipt, execution log, witness quorum).
  - Threshold witness validation with identity/hash/time binding checks.
  - Fail-closed adversarial scenarios for tampering and policy downgrade attempts.
- Still weaker than full external attestation ecosystems on:
  - Publicly auditable transparency inclusion proofs (Sigstore/Rekor style monitorable log ecosystem).
  - Platform-trust-boundary guarantees where provenance is generated outside tenant control (SLSA L3-style control-plane guarantees).
  - Standardized interop attestation envelope compatibility (in-toto statement/provenance interchange).

## Remaining Recommended Upgrades (Next Iteration)
- Add machine-checkable TRIPOD+AI / PROBAST+AI compliance gate (reporting + risk-of-bias checklist JSON).
- Add external-validation-specific gate for site/time transport checks when multi-center data are available.
- Add optional Sigstore/cosign verification mode for attestation artifacts with Rekor inclusion checks.
- Add optional in-toto/SLSA provenance ingestion + verification mode for execution evidence interchange.
