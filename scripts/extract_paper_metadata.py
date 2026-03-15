"""extract_paper_metadata.py — LLM-powered structured extraction from medical ML papers.

Reads paper text (abstract, PMC full-text XML) and uses an LLM to extract
structured fields into the paper's metadata.json.  Supports two providers:

  Provider       Model              API key env var      Cost (est.)
  ─────────────────────────────────────────────────────────────────
  deepseek       deepseek-chat      DEEPSEEK_API_KEY     ~$0.001/paper
  claude         claude-opus-4-6    ANTHROPIC_API_KEY    ~$0.05/paper

Extracted fields (automatically filled — no manual work needed):
  - study_design:   outcome, prediction_horizon, setting, multicenter, ext. validation
  - dataset:        source_type/name, n_patients, events, split_strategy, sizes
  - model:          model_type, n_candidates, hyperparameter tuning, feature selection
  - performance_metrics: AUROC±CI, sensitivity, specificity, PPV, NPV, Brier, DCA
  - reporting_standards: TRIPOD+AI, PROBAST+AI, code/data availability
  - leakage_risk_assessment: 8 leakage gate checks from methodology description

Usage:
    # DeepSeek (default, cheapest)
    export DEEPSEEK_API_KEY=sk-...
    python3 scripts/extract_paper_metadata.py --paper-dir papers/nature_medicine/cardiovascular/smith_2023_af_ehr
    python3 scripts/extract_paper_metadata.py --all --output-dir papers/

    # Claude (more accurate leakage risk assessment)
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 scripts/extract_paper_metadata.py --all --provider claude

    # Batch API (Claude only, async, half price)
    python3 scripts/extract_paper_metadata.py --all --provider claude --batch

    # Filter by journal / domain
    python3 scripts/extract_paper_metadata.py --all --journal nature_medicine --domain cardiovascular

    # Force re-extract / abstract only
    python3 scripts/extract_paper_metadata.py --all --force --abstract-only

Requirements:
    pip install openai>=1.0.0          # for DeepSeek
    pip install anthropic>=0.40.0      # for Claude (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    print("Missing dependency: pydantic")
    print("Install with: pip install pydantic>=2.0")
    sys.exit(1)

# Optional provider SDKs — imported lazily in main()
_anthropic_mod: Any = None
_openai_mod: Any = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_metadata")

# ---------------------------------------------------------------------------
# Pydantic extraction schemas
# ---------------------------------------------------------------------------

class StudyDesignOut(BaseModel):
    prediction_type: Optional[str] = Field(None, description="binary_classification / multiclass / regression / survival")
    outcome: Optional[str] = Field(None, description="Primary outcome variable being predicted")
    prediction_unit: Optional[str] = Field(None, description="patient / admission / visit / encounter")
    prediction_horizon: Optional[str] = Field(None, description="e.g. '30-day mortality', 'in-hospital', '1-year'")
    setting: Optional[str] = Field(None, description="Emergency department / ICU / inpatient / outpatient / community")
    is_multicenter: Optional[bool] = Field(None, description="True if data from multiple hospitals/sites")
    has_external_validation: Optional[bool] = Field(None, description="True if tested on a truly separate external cohort")
    external_cohort_description: Optional[str] = Field(None, description="Brief description of external validation cohort if present")


class DatasetOut(BaseModel):
    source_type: Optional[str] = Field(None, description="EHR_single_center | EHR_multicenter | public_dataset | registry | biobank | claims_data | mixed")
    source_name: Optional[str] = Field(None, description="Name of database or institution (e.g. MIMIC-IV, UK Biobank, Mayo Clinic)")
    n_patients_total: Optional[int] = Field(None, description="Total number of patients/subjects in the full dataset")
    n_events_positive: Optional[int] = Field(None, description="Number of positive outcomes (events)")
    n_events_negative: Optional[int] = Field(None, description="Number of negative outcomes (non-events)")
    prevalence_pct: Optional[float] = Field(None, description="Event prevalence as percentage (0-100)")
    split_strategy: Optional[str] = Field(None, description="random | temporal | site_based | not_reported")
    train_n: Optional[int] = Field(None, description="Number of samples in training set")
    valid_n: Optional[int] = Field(None, description="Number of samples in validation set (if separate)")
    test_n: Optional[int] = Field(None, description="Number of samples in test/holdout set")
    features_n: Optional[int] = Field(None, description="Number of input features/predictors")
    has_missing_data: Optional[bool] = Field(None, description="True if missing data was present and needed handling")
    missing_data_strategy: Optional[str] = Field(None, description="e.g. multiple imputation, mean imputation, complete case, not mentioned")


class ModelOut(BaseModel):
    model_type: Optional[str] = Field(None, description="Primary model: logistic_regression | random_forest | xgboost | lightgbm | deep_learning | ensemble | other")
    n_candidate_models: Optional[int] = Field(None, description="How many model types/architectures were compared")
    selection_criterion: Optional[str] = Field(None, description="How the final model was selected (e.g. AUROC, Brier score, one-SE rule)")
    hyperparameter_tuning: Optional[str] = Field(None, description="Method: grid search, random search, Bayesian, cross-validation, not mentioned")
    tuning_set: Optional[str] = Field(None, description="validation_only | train_validation | test_used | not_reported")
    feature_selection_method: Optional[str] = Field(None, description="LASSO, recursive feature elimination, importance threshold, none, not reported")
    preprocessing_pipeline: Optional[str] = Field(None, description="Brief description of preprocessing steps")


class PerformanceMetricsOut(BaseModel):
    primary_metric: Optional[str] = Field(None, description="The paper's stated primary performance metric")
    test_auroc: Optional[float] = Field(None, description="AUROC on the held-out test set (0-1 scale)")
    test_auroc_ci_lower: Optional[float] = Field(None, description="95% CI lower bound for test AUROC")
    test_auroc_ci_upper: Optional[float] = Field(None, description="95% CI upper bound for test AUROC")
    test_auprc: Optional[float] = Field(None, description="AUPRC (average precision) on test set if reported")
    test_sensitivity: Optional[float] = Field(None, description="Sensitivity / recall on test set (0-1)")
    test_specificity: Optional[float] = Field(None, description="Specificity on test set (0-1)")
    test_ppv: Optional[float] = Field(None, description="Positive predictive value on test set (0-1)")
    test_npv: Optional[float] = Field(None, description="Negative predictive value on test set (0-1)")
    test_f1: Optional[float] = Field(None, description="F1 score on test set (0-1)")
    test_brier_score: Optional[float] = Field(None, description="Brier score on test set (lower=better, 0-1)")
    calibration_method: Optional[str] = Field(None, description="Calibration method: Platt scaling, isotonic regression, not done, not reported")
    calibration_reported: Optional[bool] = Field(None, description="True if calibration plot or HL test reported")
    dca_reported: Optional[bool] = Field(None, description="True if decision curve analysis was performed")
    bootstrap_ci_reported: Optional[bool] = Field(None, description="True if bootstrap confidence intervals were used")
    n_bootstrap_resamples: Optional[int] = Field(None, description="Number of bootstrap resamples if reported")
    external_auroc: Optional[float] = Field(None, description="AUROC on external validation cohort if present")
    external_auroc_ci_lower: Optional[float] = Field(None, description="95% CI lower for external AUROC")
    external_auroc_ci_upper: Optional[float] = Field(None, description="95% CI upper for external AUROC")


class ReportingStandardsOut(BaseModel):
    tripod_ai_claimed: Optional[bool] = Field(None, description="True if paper claims TRIPOD or TRIPOD+AI adherence")
    probast_ai_claimed: Optional[bool] = Field(None, description="True if paper claims PROBAST or PROBAST+AI adherence")
    stard_ai_claimed: Optional[bool] = Field(None, description="True if STARD-AI or STARD mentioned")
    equator_guideline_cited: Optional[str] = Field(None, description="Which EQUATOR guideline is cited if any")
    limitation_section: Optional[bool] = Field(None, description="True if a limitations section is present")
    code_availability: Optional[str] = Field(None, description="public_github | on_request | not_available | not_mentioned")
    data_availability: Optional[str] = Field(None, description="public | on_request | restricted | not_available | not_mentioned")


class LeakageRiskOut(BaseModel):
    patient_level_split_confirmed: Optional[bool] = Field(
        None,
        description="True if the paper explicitly confirms patient-level (not encounter-level) train/test split — i.e., all records from one patient stay in the same split"
    )
    temporal_split_confirmed: Optional[bool] = Field(
        None,
        description="True if training data precedes test data in time (temporal split), avoiding look-ahead bias"
    )
    preprocessing_fit_on_train_only: Optional[bool] = Field(
        None,
        description="True if the paper states normalisation, imputation, or encoding was fit ONLY on training data"
    )
    tuning_used_test_data: Optional[bool] = Field(
        None,
        description="True (BAD) if there is evidence hyperparameter tuning used the held-out test set"
    )
    target_leakage_risk: Optional[str] = Field(
        None,
        description="low | medium | high | cannot_assess — whether any features likely encode the outcome being predicted"
    )
    post_index_feature_risk: Optional[str] = Field(
        None,
        description="low | medium | high | cannot_assess — whether features collected after the prediction index date are included"
    )
    phenotype_definition_overlap_risk: Optional[str] = Field(
        None,
        description="low | medium | high | cannot_assess — whether the outcome definition uses variables also in the feature set"
    )
    notes: str = Field(
        default="",
        description="Key observations about methodology that inform the leakage assessment"
    )


class ExtractionResult(BaseModel):
    study_design: StudyDesignOut
    dataset: DatasetOut
    model: ModelOut
    performance_metrics: PerformanceMetricsOut
    reporting_standards: ReportingStandardsOut
    leakage_risk: LeakageRiskOut
    extraction_confidence: str = Field(description="high | medium | low — how confident based on available text")
    extraction_notes: str = Field(description="Caveats: what was unclear, what required inference")


# ---------------------------------------------------------------------------
# System prompt (shared / cache-eligible)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert medical AI/ML methodologist performing structured extraction from clinical prediction model papers.

Your task is to extract specific methodology and results fields from the provided paper text (abstract, methods, or results sections).

EXTRACTION RULES:
1. Extract ONLY what is explicitly stated. If a value is not reported, return null.
2. For numeric values: use the held-out TEST set values, not training or validation. If multiple test sets exist, use the primary/internal test set for the main fields.
3. For AUROC and CI: values must be in 0-1 range. If reported as percentages (e.g., 0.87 = 87%), keep as 0-1.
4. For leakage risk assessment:
   - patient_level_split_confirmed: must be EXPLICITLY stated. Null if only implied.
   - preprocessing_fit_on_train_only: True only if Methods explicitly say preprocessing was fit on training data. If paper doesn't mention this, return null (not False).
   - tuning_used_test_data: True if there is clear evidence of test set contamination in tuning. Otherwise null or False.
   - target_leakage_risk: Assess whether any features could directly encode the outcome. "High" means obvious concern (e.g., using diagnosis codes that define the outcome as features). "Medium" means plausible concern. "Low" means no obvious concern.
5. extraction_confidence: "high" if key fields (AUROC, sample size, split strategy) are clearly stated; "medium" if some required inference; "low" if only abstract available or poor reporting.
6. In extraction_notes: document what was unclear, what required inference, and any methodological concerns worth flagging for MLGG audit.

Be precise and systematic. This data will be used for automated MLGG framework validation."""


# ---------------------------------------------------------------------------
# Full-text fetchers
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mlgg-metadata-extractor/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_pmc_fulltext(pmcid: str) -> str:
    """Fetch PMC full-text XML and return Methods + Results sections as plain text."""
    clean_id = pmcid.upper().replace("PMC", "")
    if not clean_id.isdigit():
        log.warning("Invalid PMCID: %s", pmcid)
        return ""

    params = urllib.parse.urlencode({
        "db": "pmc",
        "id": clean_id,
        "rettype": "xml",
        "retmode": "xml",
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    try:
        raw = _http_get(url, timeout=30)
    except Exception as exc:
        log.warning("PMC fetch failed for %s: %s", pmcid, exc)
        return ""

    return _extract_pmc_sections(raw)


def _extract_pmc_sections(raw: bytes) -> str:
    """Extract Methods, Results, and Discussion sections from PMC XML."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""

    target_keywords = {
        "methods", "method", "materials and methods", "patients and methods",
        "statistical analysis", "statistical methods", "study design", "data collection",
        "study population", "results", "outcomes", "performance", "discussion",
        "model development", "model evaluation", "model performance",
    }

    sections: list[str] = []

    def _extract_text(el: ET.Element) -> str:
        return " ".join(el.itertext()).strip()

    for sec in root.findall(".//sec"):
        title_el = sec.find("title")
        title = title_el.text.strip().lower() if title_el is not None and title_el.text else ""
        is_target = any(kw in title for kw in target_keywords)
        if is_target or not title:
            text = _extract_text(sec)
            if len(text) > 100:
                header = f"\n## {title_el.text.strip() if title_el is not None and title_el.text else 'Section'}\n"
                sections.append(header + text[:8000])  # cap per section

    # Fallback: grab all body text
    if not sections:
        body = root.find(".//body")
        if body is not None:
            sections.append(_extract_text(body)[:20000])

    combined = "\n\n".join(sections)
    return combined[:25000]  # cap total


# ---------------------------------------------------------------------------
# Paper text assembler
# ---------------------------------------------------------------------------
def assemble_paper_text(metadata: dict[str, Any], abstract_only: bool = False) -> str:
    """Build the text to send to Claude from available sources."""
    bib = metadata.get("bibliographic", {})
    auto = metadata.get("auto_classification", {})

    title = bib.get("title", "")
    abstract = auto.get("abstract_snippet", "") or ""
    pmcid = bib.get("pmcid", "")
    source = metadata.get("_source", "")

    lines: list[str] = []
    if title:
        lines.append(f"TITLE: {title}")
    if bib.get("journal"):
        lines.append(f"JOURNAL: {bib['journal']} ({bib.get('year', '')})")
    if bib.get("doi"):
        lines.append(f"DOI: {bib['doi']}")
    if abstract:
        lines.append(f"\nABSTRACT:\n{abstract}")

    if not abstract_only and pmcid:
        log.info("  Fetching PMC full-text for %s…", pmcid)
        full_text = _fetch_pmc_fulltext(pmcid)
        if full_text:
            lines.append(f"\n\nFULL TEXT (Methods + Results):\n{full_text}")
            time.sleep(0.35)  # NCBI rate limit
        else:
            log.info("  Full-text not available, using abstract only")

    if not abstract and len(lines) <= 3:
        return ""  # not enough to extract from

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON schema for DeepSeek prompt injection
# ---------------------------------------------------------------------------
def _extraction_schema_text() -> str:
    """Return a compact JSON schema description for use in the DeepSeek prompt."""
    return """{
  "study_design": {
    "prediction_type": "string|null  (binary_classification/multiclass/regression/survival)",
    "outcome": "string|null",
    "prediction_unit": "string|null  (patient/admission/visit/encounter)",
    "prediction_horizon": "string|null  (e.g. '30-day mortality')",
    "setting": "string|null  (ICU/ED/inpatient/outpatient/community)",
    "is_multicenter": "boolean|null",
    "has_external_validation": "boolean|null",
    "external_cohort_description": "string|null"
  },
  "dataset": {
    "source_type": "string|null  (EHR_single_center/EHR_multicenter/public_dataset/registry/biobank/claims_data/mixed)",
    "source_name": "string|null",
    "n_patients_total": "integer|null",
    "n_events_positive": "integer|null",
    "n_events_negative": "integer|null",
    "prevalence_pct": "number|null  (0-100)",
    "split_strategy": "string|null  (random/temporal/site_based/not_reported)",
    "train_n": "integer|null",
    "valid_n": "integer|null",
    "test_n": "integer|null",
    "features_n": "integer|null",
    "has_missing_data": "boolean|null",
    "missing_data_strategy": "string|null"
  },
  "model": {
    "model_type": "string|null  (logistic_regression/random_forest/xgboost/lightgbm/deep_learning/ensemble/other)",
    "n_candidate_models": "integer|null",
    "selection_criterion": "string|null",
    "hyperparameter_tuning": "string|null",
    "tuning_set": "string|null  (validation_only/train_validation/test_used/not_reported)",
    "feature_selection_method": "string|null",
    "preprocessing_pipeline": "string|null"
  },
  "performance_metrics": {
    "primary_metric": "string|null",
    "test_auroc": "number|null  (0-1 scale)",
    "test_auroc_ci_lower": "number|null",
    "test_auroc_ci_upper": "number|null",
    "test_auprc": "number|null",
    "test_sensitivity": "number|null  (0-1)",
    "test_specificity": "number|null  (0-1)",
    "test_ppv": "number|null  (0-1)",
    "test_npv": "number|null  (0-1)",
    "test_f1": "number|null  (0-1)",
    "test_brier_score": "number|null",
    "calibration_method": "string|null",
    "calibration_reported": "boolean|null",
    "dca_reported": "boolean|null",
    "bootstrap_ci_reported": "boolean|null",
    "n_bootstrap_resamples": "integer|null",
    "external_auroc": "number|null",
    "external_auroc_ci_lower": "number|null",
    "external_auroc_ci_upper": "number|null"
  },
  "reporting_standards": {
    "tripod_ai_claimed": "boolean|null",
    "probast_ai_claimed": "boolean|null",
    "stard_ai_claimed": "boolean|null",
    "equator_guideline_cited": "string|null",
    "limitation_section": "boolean|null",
    "code_availability": "string|null  (public_github/on_request/not_available/not_mentioned)",
    "data_availability": "string|null  (public/on_request/restricted/not_available/not_mentioned)"
  },
  "leakage_risk": {
    "patient_level_split_confirmed": "boolean|null",
    "temporal_split_confirmed": "boolean|null",
    "preprocessing_fit_on_train_only": "boolean|null",
    "tuning_used_test_data": "boolean|null  (true = BAD, contamination detected)",
    "target_leakage_risk": "string|null  (low/medium/high/cannot_assess)",
    "post_index_feature_risk": "string|null  (low/medium/high/cannot_assess)",
    "phenotype_definition_overlap_risk": "string|null  (low/medium/high/cannot_assess)",
    "notes": "string  (methodology observations, never null)"
  },
  "extraction_confidence": "string  (high/medium/low)",
  "extraction_notes": "string  (caveats and ambiguities)"
}"""


def _build_user_prompt(paper_text: str) -> str:
    return (
        "Please extract structured metadata from the following medical prediction paper.\n\n"
        f"{paper_text}\n\n"
        "Extract all fields you can. Return null for fields not stated or unclear.\n"
        "For leakage_risk fields, base your assessment on the described methodology.\n"
        "Return ONLY valid JSON matching the schema — no markdown, no prose."
    )


# ---------------------------------------------------------------------------
# DeepSeek extraction (OpenAI-compatible API)
# ---------------------------------------------------------------------------
def extract_with_deepseek(
    client: Any,  # openai.OpenAI
    paper_text: str,
    paper_id: str,
    model: str = "deepseek-chat",
) -> ExtractionResult | None:
    """Call DeepSeek via OpenAI-compatible API with JSON mode."""
    schema_text = _extraction_schema_text()
    system = (
        SYSTEM_PROMPT
        + f"\n\nReturn your answer as a single JSON object matching this schema exactly:\n{schema_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _build_user_prompt(paper_text)},
            ],
        )
        raw_text = response.choices[0].message.content or ""
        raw = json.loads(raw_text)
        # leakage_risk.notes must not be None
        if "leakage_risk" in raw and raw["leakage_risk"].get("notes") is None:
            raw["leakage_risk"]["notes"] = ""
        return ExtractionResult(**raw)
    except json.JSONDecodeError as exc:
        log.error("[%s] JSON decode failed: %s", paper_id, exc)
        return None
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "rate" in msg.lower():
            log.warning("[%s] Rate limited, waiting 30s…", paper_id)
            time.sleep(30)
            return extract_with_deepseek(client, paper_text, paper_id, model)
        log.error("[%s] DeepSeek extraction failed: %s", paper_id, exc)
        return None


# ---------------------------------------------------------------------------
# Claude extraction (Anthropic SDK)
# ---------------------------------------------------------------------------
def extract_with_claude(
    client: Any,  # anthropic.Anthropic
    paper_text: str,
    paper_id: str,
    model: str = "claude-opus-4-6",
) -> ExtractionResult | None:
    """Call Claude with adaptive thinking + structured output. Returns None on failure."""
    anthropic = _anthropic_mod
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": _build_user_prompt(paper_text)}],
            output_format=ExtractionResult,
        )
        return response.parsed_output
    except anthropic.BadRequestError as exc:
        log.error("[%s] Bad request: %s", paper_id, exc)
        return None
    except anthropic.RateLimitError:
        log.warning("[%s] Rate limited, waiting 60s…", paper_id)
        time.sleep(60)
        return extract_with_claude(client, paper_text, paper_id, model)
    except Exception as exc:
        log.error("[%s] Claude extraction failed: %s", paper_id, exc)
        return None


# ---------------------------------------------------------------------------
# metadata.json merger
# ---------------------------------------------------------------------------
def merge_extraction(
    metadata: dict[str, Any],
    result: ExtractionResult,
    force: bool = False,
) -> dict[str, Any]:
    """Merge extracted values into metadata dict. Only fills null/empty fields unless force=True."""
    today = date.today().isoformat()

    def _merge_section(target: dict[str, Any], extracted: dict[str, Any]) -> None:
        for key, new_val in extracted.items():
            if key.startswith("_"):
                continue
            if new_val is None:
                continue
            existing = target.get(key)
            if force or existing is None or existing == "" or existing == []:
                target[key] = new_val

    _merge_section(metadata["study_design"], result.study_design.model_dump())
    _merge_section(metadata["dataset"], result.dataset.model_dump())
    _merge_section(metadata["model"], result.model.model_dump())
    _merge_section(metadata["performance_metrics"], result.performance_metrics.model_dump())
    _merge_section(metadata["reporting_standards"], result.reporting_standards.model_dump())

    # leakage_risk_assessment section
    leak = metadata.setdefault("leakage_risk_assessment", {})
    risk_dump = result.leakage_risk.model_dump()
    # Notes: always append (don't overwrite)
    new_notes = risk_dump.pop("notes", "")
    existing_notes = leak.get("notes", "")
    if new_notes:
        combined_notes = (existing_notes + "\n" + new_notes).strip() if existing_notes else new_notes
        leak["notes"] = combined_notes
    _merge_section(leak, risk_dump)

    # Audit metadata
    audit = metadata.setdefault("mlgg_audit", {})
    audit["_last_extraction_date"] = today
    audit["_extraction_confidence"] = result.extraction_confidence
    audit["_extraction_notes"] = result.extraction_notes

    # Reviewer notes: mark as auto-extracted
    rev = metadata.setdefault("reviewer_notes", {})
    if rev.get("added_by") == "fetch_papers.py (auto)" or not rev.get("added_by"):
        rev["added_by"] = "extract_paper_metadata.py (auto)"
        rev["added_date"] = today

    return metadata


# ---------------------------------------------------------------------------
# Single paper processor
# ---------------------------------------------------------------------------
def process_paper(
    paper_dir: Path,
    extractor: Any,  # callable(paper_text, paper_id) -> ExtractionResult | None
    provider_label: str,
    abstract_only: bool,
    force: bool,
    dry_run: bool,
) -> bool:
    """Extract metadata for one paper directory. Returns True on success."""
    metadata_path = paper_dir / "metadata.json"
    if not metadata_path.exists():
        log.warning("No metadata.json in %s, skipping", paper_dir)
        return False

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Skip if already extracted (unless force)
    if not force:
        audit = metadata.get("mlgg_audit", {})
        if audit.get("_last_extraction_date"):
            log.info("  Already extracted (%s), skipping. Use --force to re-extract.", audit["_last_extraction_date"])
            return True

    paper_text = assemble_paper_text(metadata, abstract_only=abstract_only)
    if not paper_text.strip():
        log.warning("  No usable text for %s, skipping", paper_dir.name)
        return False

    if dry_run:
        log.info("  [DRY-RUN] Would extract %d chars via %s", len(paper_text), provider_label)
        return True

    log.info("  Extracting via %s (text: %d chars)…", provider_label, len(paper_text))
    result = extractor(paper_text, paper_dir.name)
    if result is None:
        return False

    merged = merge_extraction(metadata, result, force=force)

    # Write back
    with open(metadata_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    conf = result.extraction_confidence
    auroc = result.performance_metrics.test_auroc
    risk = result.leakage_risk.target_leakage_risk
    log.info(
        "  Done [%s]. confidence=%s | AUROC=%s | leakage_risk=%s",
        provider_label,
        conf,
        f"{auroc:.3f}" if auroc else "n/a",
        risk or "n/a",
    )
    return True


# ---------------------------------------------------------------------------
# Batch API mode
# ---------------------------------------------------------------------------
def build_batch_requests(
    paper_dirs: list[Path],
    abstract_only: bool,
) -> list[tuple[str, str]]:
    """Build (paper_dir_str, prompt) pairs for Batch API submission."""
    requests = []
    for paper_dir in paper_dirs:
        metadata_path = paper_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        text = assemble_paper_text(metadata, abstract_only=abstract_only)
        if not text.strip():
            continue
        requests.append((str(paper_dir), text))
    return requests


def submit_batch(
    client: Any,  # anthropic.Anthropic
    paper_dirs: list[Path],
    abstract_only: bool,
    force: bool,
) -> str | None:
    """Submit a Batch API job. Returns batch ID."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pairs = build_batch_requests(paper_dirs, abstract_only)
    if not pairs:
        log.warning("No papers with usable text for batch submission")
        return None

    log.info("Submitting batch of %d papers…", len(pairs))
    batch_requests = []

    for paper_dir_str, paper_text in pairs:
        prompt = (
            "Please extract structured metadata from the following medical prediction paper.\n\n"
            f"{paper_text}\n\n"
            "Extract all fields you can from the text above. Return null for fields not mentioned or unclear.\n"
            "For leakage_risk fields, base your assessment on the described methodology."
        )
        # Use folder name as custom_id (sanitise)
        custom_id = Path(paper_dir_str).name[:64].replace(" ", "_")
        batch_requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model="claude-opus-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ),
        ))

    try:
        batch = client.messages.batches.create(requests=batch_requests)
        log.info("Batch submitted: %s", batch.id)
        return batch.id
    except Exception as exc:
        log.error("Batch submission failed: %s", exc)
        return None


def poll_and_apply_batch(
    client: Any,  # anthropic.Anthropic
    batch_id: str,
    paper_dirs: list[Path],
    force: bool,
) -> None:
    """Poll batch until complete, then apply results to metadata.json files."""
    log.info("Polling batch %s…", batch_id)

    # Build dir lookup by custom_id
    dir_by_id: dict[str, Path] = {}
    for pd in paper_dirs:
        custom_id = pd.name[:64].replace(" ", "_")
        dir_by_id[custom_id] = pd

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        log.info(
            "  Status: %s | processing=%d succeeded=%d errored=%d",
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(30)

    succeeded = 0
    failed = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            log.warning("  [%s] %s", result.custom_id, result.result.type)
            failed += 1
            continue

        paper_dir = dir_by_id.get(result.custom_id)
        if not paper_dir:
            log.warning("  [%s] No matching paper dir found", result.custom_id)
            continue

        metadata_path = paper_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        if not text:
            failed += 1
            continue

        try:
            raw = json.loads(text)
            extracted = ExtractionResult(**raw)
        except Exception as exc:
            log.warning("  [%s] Parse failed: %s", result.custom_id, exc)
            failed += 1
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)
        merged = merge_extraction(metadata, extracted, force=force)
        with open(metadata_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        succeeded += 1

    log.info("Batch complete: %d succeeded, %d failed", succeeded, failed)


# ---------------------------------------------------------------------------
# Paper discovery
# ---------------------------------------------------------------------------
JOURNAL_DIRS = {
    "nature_medicine", "lancet_digital_health", "jama", "bmj",
    "npj_digital_medicine", "specialist_journals",
}
DOMAIN_DIRS = {
    "cardiovascular", "oncology", "diabetes", "kidney_disease",
    "sepsis_icu", "neurology", "respiratory", "infectious_disease", "other",
}


def discover_papers(
    output_dir: Path,
    journal_filter: str | None = None,
    domain_filter: str | None = None,
    force: bool = False,
) -> list[Path]:
    """Return list of paper directories to process."""
    papers: list[Path] = []

    journals = {journal_filter} if journal_filter else JOURNAL_DIRS
    for journal in journals:
        journal_path = output_dir / journal
        if not journal_path.exists():
            continue
        domains = {domain_filter} if domain_filter else DOMAIN_DIRS
        for domain in domains:
            domain_path = journal_path / domain
            if not domain_path.exists():
                continue
            for paper_dir in sorted(domain_path.iterdir()):
                if not paper_dir.is_dir():
                    continue
                if paper_dir.name.startswith("."):
                    continue
                meta = paper_dir / "metadata.json"
                if not meta.exists():
                    continue
                if not force:
                    with open(meta) as f:
                        m = json.load(f)
                    if m.get("mlgg_audit", {}).get("_last_extraction_date"):
                        continue  # already done
                papers.append(paper_dir)

    return papers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-powered structured extraction from medical ML papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper-dir", type=Path, help="Extract a single paper directory")
    mode.add_argument("--all", action="store_true", help="Extract all unprocessed papers")
    mode.add_argument(
        "--poll-batch",
        metavar="BATCH_ID",
        help="Poll a Claude Batch API job and apply results (Claude provider only)",
    )

    p.add_argument(
        "--provider",
        choices=["deepseek", "claude"],
        default=None,
        help="LLM provider: deepseek (default, cheap) or claude (accurate)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override model name (deepseek default: deepseek-chat; claude default: claude-opus-4-6)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("papers"), help="Root papers/ directory")
    p.add_argument("--journal", help="Filter by journal tier (e.g. nature_medicine)")
    p.add_argument("--domain", help="Filter by disease domain (e.g. cardiovascular)")
    p.add_argument("--abstract-only", action="store_true", help="Use abstract only, skip full-text fetch")
    p.add_argument("--force", action="store_true", help="Re-extract even if already done")
    p.add_argument("--dry-run", action="store_true", help="Print what would be done without calling API")
    p.add_argument("--batch", action="store_true", help="Use Claude Batch API (async, half price, Claude only)")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers for inline mode (default: 1)")
    return p


def _setup_provider(args: argparse.Namespace) -> tuple[str, Any, Any, str]:
    """Resolve provider, build client, return (provider, extractor_fn, label)."""
    global _anthropic_mod, _openai_mod

    # Auto-detect provider from available env vars
    provider = args.provider
    if provider is None:
        if os.environ.get("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "claude"
        else:
            provider = "deepseek"  # will fail below with clear message

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: DEEPSEEK_API_KEY not set.")
            print("  export DEEPSEEK_API_KEY=sk-...")
            print("  Get it at: https://platform.deepseek.com/api_keys")
            sys.exit(1)
        try:
            import openai as _openai_import
            _openai_mod = _openai_import
        except ImportError:
            print("ERROR: openai package not installed.")
            print("  pip install openai>=1.0.0")
            sys.exit(1)
        model = args.model or "deepseek-chat"
        client = _openai_mod.OpenAI(
            api_key=api_key or "dummy",
            base_url="https://api.deepseek.com",
        )
        extractor = lambda text, pid: extract_with_deepseek(client, text, pid, model)
        label = f"DeepSeek ({model})"

    else:  # claude
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not args.dry_run:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            print("  export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)
        try:
            import anthropic as _anthropic_import
            _anthropic_mod = _anthropic_import
        except ImportError:
            print("ERROR: anthropic package not installed.")
            print("  pip install anthropic>=0.40.0")
            sys.exit(1)
        model = args.model or "claude-opus-4-6"
        client = _anthropic_mod.Anthropic(api_key=api_key or "dummy")
        extractor = lambda text, pid: extract_with_claude(client, text, pid, model)
        label = f"Claude ({model})"

    return provider, client, extractor, label


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    provider, client, extractor, label = _setup_provider(args)
    log.info("Provider: %s", label)

    # ── Single paper ──────────────────────────────────────────────────────
    if args.paper_dir:
        if not args.paper_dir.exists():
            log.error("Paper directory not found: %s", args.paper_dir)
            return 1
        log.info("Processing: %s", args.paper_dir.name)
        ok = process_paper(
            args.paper_dir,
            extractor,
            label,
            abstract_only=args.abstract_only,
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0 if ok else 1

    # ── Poll batch (Claude only) ──────────────────────────────────────────
    if args.poll_batch:
        if provider != "claude":
            log.error("--poll-batch is only supported with --provider claude")
            return 1
        papers = discover_papers(args.output_dir, args.journal, args.domain, force=True)
        poll_and_apply_batch(client, args.poll_batch, papers, args.force)
        return 0

    # ── All papers ────────────────────────────────────────────────────────
    papers = discover_papers(args.output_dir, args.journal, args.domain, force=args.force)
    log.info("Found %d papers to process", len(papers))
    if not papers:
        log.info("Nothing to do.")
        return 0

    # Batch API mode (Claude only)
    if args.batch and not args.dry_run:
        if provider != "claude":
            log.error("--batch requires --provider claude (DeepSeek has no batch API)")
            return 1
        batch_id = submit_batch(client, papers, args.abstract_only, args.force)
        if batch_id:
            print(f"\nBatch submitted: {batch_id}")
            print(
                f"Poll with: python3 scripts/extract_paper_metadata.py"
                f" --poll-batch {batch_id} --provider claude --output-dir {args.output_dir}"
            )
        return 0

    # Inline mode
    succeeded = 0
    failed = 0
    for i, paper_dir in enumerate(papers, 1):
        log.info("[%d/%d] %s", i, len(papers), paper_dir.name)
        ok = process_paper(
            paper_dir,
            extractor,
            label,
            abstract_only=args.abstract_only,
            force=args.force,
            dry_run=args.dry_run,
        )
        if ok:
            succeeded += 1
        else:
            failed += 1
        if not args.dry_run and i < len(papers):
            time.sleep(1.0 if provider == "deepseek" else 1.5)

    log.info("Done. %d succeeded, %d failed", succeeded, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
