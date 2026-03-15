"""fetch_papers.py — Multi-source academic paper crawler for MLGG framework validation.

Queries PubMed, PMC Open Access, Semantic Scholar, OpenAlex, and arXiv;
normalises results into a common schema; auto-classifies into journal-tier ×
disease-domain folders; and writes a pre-filled metadata.json for each paper.

Usage examples:
    # Basic PubMed search → auto-classify → write metadata.json
    python3 scripts/fetch_papers.py \\
        --query "machine learning clinical prediction" \\
        --sources pubmed semantic_scholar \\
        --max-results 50 \\
        --output-dir papers/

    # Multi-keyword search with PDF download (open-access only)
    python3 scripts/fetch_papers.py \\
        --query "deep learning atrial fibrillation EHR" \\
        --sources pubmed pmc arxiv \\
        --max-results 30 \\
        --output-dir papers/ \\
        --download-pdf

    # Dry-run: print what would be created without writing files
    python3 scripts/fetch_papers.py \\
        --query "XGBoost sepsis prediction ICU" \\
        --sources pubmed openalex \\
        --max-results 20 \\
        --output-dir papers/ \\
        --dry-run

    # Add a rate-limit email for PubMed (recommended for large fetches)
    python3 scripts/fetch_papers.py \\
        --query "random forest diabetes prediction" \\
        --sources pubmed \\
        --max-results 200 \\
        --output-dir papers/ \\
        --email your@email.com
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_papers")

# ---------------------------------------------------------------------------
# Journal → tier folder mapping
# ---------------------------------------------------------------------------
JOURNAL_TIER_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"nature medicine", re.I), "nature_medicine"),
    (re.compile(r"lancet digital health", re.I), "lancet_digital_health"),
    (re.compile(r"\bjama\b|jama (internal|cardiology|network|oncology|pediatrics|psychiatry|surgery)", re.I), "jama"),
    (re.compile(r"\bbmj\b|british medical journal|bmj open|bmj quality", re.I), "bmj"),
    (re.compile(r"npj digital medicine|npj.digital", re.I), "npj_digital_medicine"),
]
TIER_FALLBACK = "specialist_journals"

# ---------------------------------------------------------------------------
# Disease domain classification — ordered by specificity
# ---------------------------------------------------------------------------
DISEASE_DOMAIN_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(
        r"atrial fibrillat|heart fail|cardiac arrest|myocardial infarct|"
        r"coronary artery|cardiovascular|cardiomyopathy|aortic|"
        r"ventricular|endocarditis|peripheral artery|stroke|cerebrovascular",
        re.I), "cardiovascular"),
    (re.compile(
        r"cancer|tumor|tumour|carcinoma|malignant|neoplasm|oncolog|"
        r"lymphoma|leukemia|leukaemia|melanoma|glioma|sarcoma|metastas",
        re.I), "oncology"),
    (re.compile(
        r"diabet|glucose|insulin|glycemi|glycaemi|HbA1c|hyperglycemi|"
        r"hypoglycemi|pancrea",
        re.I), "diabetes"),
    (re.compile(
        r"kidney|renal|acute kidney injury|AKI|chronic kidney|CKD|"
        r"dialysis|nephro|glomerulo|proteinuri",
        re.I), "kidney_disease"),
    (re.compile(
        r"sepsis|septic shock|intensive care|ICU|critically ill|"
        r"organ failure|mechanical ventilation|vasopressor",
        re.I), "sepsis_icu"),
    (re.compile(
        r"alzheimer|dementia|parkinson|seizure|epilepsy|neurodegenerative|"
        r"cognitive impairment|multiple sclerosis|neuropathy|stroke predict",
        re.I), "neurology"),
    (re.compile(
        r"COPD|asthma|pneumonia|respiratory|pulmonary|lung|COVID|"
        r"ARDS|bronchitis|emphysema|obstructive sleep apnea",
        re.I), "respiratory"),
    (re.compile(
        r"infect|bacteria|viral|HIV|AIDS|tuberculosis|antimicrobial|"
        r"pathogen|bloodstream|bacteremia|fungal|malaria",
        re.I), "infectious_disease"),
]
DOMAIN_FALLBACK = "other"


def classify_journal(journal: str) -> str:
    """Map a journal name string to a tier folder name."""
    for pattern, folder in JOURNAL_TIER_MAP:
        if pattern.search(journal):
            return folder
    return TIER_FALLBACK


def classify_disease(title: str, abstract: str, mesh_terms: list[str]) -> str:
    """Map paper content to a disease domain folder."""
    haystack = " ".join([title, abstract] + mesh_terms)
    for pattern, domain in DISEASE_DOMAIN_RULES:
        if pattern.search(haystack):
            return domain
    return DOMAIN_FALLBACK


# ---------------------------------------------------------------------------
# Common Paper schema
# ---------------------------------------------------------------------------
def _empty_paper() -> dict[str, Any]:
    return {
        "title": "",
        "authors": [],
        "journal": "",
        "year": None,
        "doi": "",
        "pmid": "",
        "pmcid": "",
        "url": "",
        "abstract": "",
        "mesh_terms": [],
        "keywords": [],
        "source": "",
        "is_open_access": False,
        "pdf_url": "",
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, retries: int = 3, delay: float = 1.5, timeout: int = 20) -> bytes:
    """HTTP GET with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "mlgg-paper-fetcher/1.0 (research purposes)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            log.warning("GET failed (%s), retrying in %.0fs…", exc, delay)
            time.sleep(delay)
            delay *= 2
    return b""  # unreachable


# ---------------------------------------------------------------------------
# Source: PubMed (NCBI E-utilities)
# ---------------------------------------------------------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _pubmed_search(query: str, max_results: int, email: str) -> list[str]:
    """Return list of PMIDs matching *query*."""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "tool": "mlgg-fetch",
        "email": email or "mlgg@example.com",
    })
    url = f"{EUTILS_BASE}/esearch.fcgi?{params}"
    data = json.loads(_get(url))
    return data.get("esearchresult", {}).get("idlist", [])


def _pubmed_fetch_batch(pmids: list[str], email: str) -> list[dict[str, Any]]:
    """Fetch full records for a list of PMIDs; returns list of paper dicts."""
    if not pmids:
        return []
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
        "tool": "mlgg-fetch",
        "email": email or "mlgg@example.com",
    })
    url = f"{EUTILS_BASE}/efetch.fcgi?{params}"
    raw = _get(url)
    return _parse_pubmed_xml(raw)


def _parse_pubmed_xml(raw: bytes) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        log.error("PubMed XML parse error: %s", exc)
        return papers

    for article in root.findall(".//PubmedArticle"):
        p = _empty_paper()
        p["source"] = "pubmed"

        # PMID
        pmid_el = article.find(".//PMID")
        p["pmid"] = pmid_el.text if pmid_el is not None else ""

        # Title
        title_el = article.find(".//ArticleTitle")
        p["title"] = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract
        abstract_parts = article.findall(".//AbstractText")
        p["abstract"] = " ".join("".join(el.itertext()) for el in abstract_parts).strip()

        # Journal
        journal_el = article.find(".//Journal/Title")
        p["journal"] = journal_el.text.strip() if journal_el is not None else ""

        # Year
        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")
        if year_el is not None and year_el.text:
            m = re.search(r"\d{4}", year_el.text)
            p["year"] = int(m.group()) if m else None

        # Authors
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                p["authors"].append(f"{last} {fore}".strip())

        # DOI
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi":
                p["doi"] = id_el.text or ""
            if id_el.get("IdType") == "pmc":
                p["pmcid"] = id_el.text or ""

        # MeSH
        for mesh_el in article.findall(".//MeshHeading/DescriptorName"):
            p["mesh_terms"].append(mesh_el.text or "")

        # URL
        if p["doi"]:
            p["url"] = f"https://doi.org/{p['doi']}"
        elif p["pmid"]:
            p["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"

        papers.append(p)
    return papers


def fetch_pubmed(query: str, max_results: int, email: str) -> list[dict[str, Any]]:
    """Search PubMed and return paper dicts."""
    log.info("[PubMed] Searching: %s (max %d)", query, max_results)
    pmids = _pubmed_search(query, max_results, email)
    log.info("[PubMed] Found %d PMIDs", len(pmids))
    if not pmids:
        return []

    # Fetch in batches of 100
    papers: list[dict[str, Any]] = []
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        log.info("[PubMed] Fetching records %d–%d…", i + 1, i + len(batch))
        papers.extend(_pubmed_fetch_batch(batch, email))
        time.sleep(0.4)  # NCBI rate limit: 3 req/s without API key

    return papers


# ---------------------------------------------------------------------------
# Source: PMC Open Access
# ---------------------------------------------------------------------------
def fetch_pmc(query: str, max_results: int, email: str) -> list[dict[str, Any]]:
    """Search PMC OA subset — wraps PubMed fetch with filter."""
    oa_query = f"({query}) AND (open access[filter])"
    log.info("[PMC] Searching OA subset: %s", oa_query)
    papers = fetch_pubmed(oa_query, max_results, email)
    for p in papers:
        p["source"] = "pmc"
        if p["pmcid"]:
            p["is_open_access"] = True
            p["pdf_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{p['pmcid']}/pdf/"
    return papers


# ---------------------------------------------------------------------------
# Source: Semantic Scholar
# ---------------------------------------------------------------------------
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "title,authors,year,externalIds,venue,abstract,isOpenAccess,openAccessPdf"
)


def fetch_semantic_scholar(query: str, max_results: int, _email: str) -> list[dict[str, Any]]:
    """Search Semantic Scholar public API."""
    log.info("[S2] Searching: %s (max %d)", query, max_results)
    papers: list[dict[str, Any]] = []
    limit = min(max_results, 100)
    offset = 0

    while len(papers) < max_results:
        params = urllib.parse.urlencode({
            "query": query,
            "fields": S2_FIELDS,
            "limit": limit,
            "offset": offset,
        })
        url = f"{S2_BASE}/paper/search?{params}"
        try:
            raw = json.loads(_get(url))
        except Exception as exc:
            log.warning("[S2] Request failed: %s", exc)
            break

        batch = raw.get("data", [])
        if not batch:
            break

        for item in batch:
            p = _empty_paper()
            p["source"] = "semantic_scholar"
            p["title"] = item.get("title", "")
            p["year"] = item.get("year")
            p["journal"] = item.get("venue", "")
            p["abstract"] = item.get("abstract", "") or ""
            p["authors"] = [a.get("name", "") for a in item.get("authors", [])]
            ext = item.get("externalIds", {})
            p["doi"] = ext.get("DOI", "")
            p["pmid"] = ext.get("PubMed", "")
            p["is_open_access"] = item.get("isOpenAccess", False)
            oa_pdf = item.get("openAccessPdf")
            if oa_pdf and isinstance(oa_pdf, dict):
                p["pdf_url"] = oa_pdf.get("url", "")
            if p["doi"]:
                p["url"] = f"https://doi.org/{p['doi']}"
            papers.append(p)

        offset += len(batch)
        if offset >= raw.get("total", 0):
            break
        time.sleep(0.5)

    return papers[:max_results]


# ---------------------------------------------------------------------------
# Source: OpenAlex
# ---------------------------------------------------------------------------
OA_BASE = "https://api.openalex.org"


def fetch_openalex(query: str, max_results: int, email: str) -> list[dict[str, Any]]:
    """Search OpenAlex API."""
    log.info("[OpenAlex] Searching: %s (max %d)", query, max_results)
    papers: list[dict[str, Any]] = []
    per_page = min(max_results, 200)
    page = 1

    while len(papers) < max_results:
        params = urllib.parse.urlencode({
            "search": query,
            "filter": "type:article",
            "select": "title,authorships,publication_year,primary_location,doi,abstract_inverted_index,open_access,ids",
            "per-page": per_page,
            "page": page,
            "mailto": email or "mlgg@example.com",
        })
        url = f"{OA_BASE}/works?{params}"
        try:
            raw = json.loads(_get(url))
        except Exception as exc:
            log.warning("[OpenAlex] Request failed: %s", exc)
            break

        results = raw.get("results", [])
        if not results:
            break

        for item in results:
            p = _empty_paper()
            p["source"] = "openalex"
            p["title"] = item.get("title", "") or ""
            p["year"] = item.get("publication_year")

            # Journal
            loc = item.get("primary_location") or {}
            source = loc.get("source") or {}
            p["journal"] = source.get("display_name", "")

            # Authors
            for au in item.get("authorships", []):
                author = au.get("author", {})
                p["authors"].append(author.get("display_name", ""))

            # DOI
            doi_raw = item.get("doi", "") or ""
            p["doi"] = doi_raw.replace("https://doi.org/", "")

            # Abstract from inverted index
            inv = item.get("abstract_inverted_index") or {}
            if inv:
                p["abstract"] = _reconstruct_abstract(inv)

            # Open access
            oa = item.get("open_access") or {}
            p["is_open_access"] = oa.get("is_oa", False)
            p["pdf_url"] = oa.get("oa_url", "") or ""

            if p["doi"]:
                p["url"] = f"https://doi.org/{p['doi']}"

            # PMID via ids
            ids = item.get("ids") or {}
            pmid_raw = ids.get("pmid", "") or ""
            p["pmid"] = pmid_raw.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/")

            papers.append(p)

        page += 1
        if len(results) < per_page:
            break
        time.sleep(0.3)

    return papers[:max_results]


def _reconstruct_abstract(inv: dict[str, list[int]]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    max_pos = max((pos for positions in inv.values() for pos in positions), default=-1)
    if max_pos < 0:
        return ""
    words: list[str] = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                words[pos] = word
    return " ".join(w for w in words if w)


# ---------------------------------------------------------------------------
# Source: arXiv
# ---------------------------------------------------------------------------
ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = "cat:cs.LG OR cat:stat.ML OR cat:q-bio.QM OR cat:cs.AI"


def fetch_arxiv(query: str, max_results: int, _email: str) -> list[dict[str, Any]]:
    """Search arXiv for medical ML preprints."""
    full_query = f"({query}) AND ({ARXIV_CATEGORIES})"
    log.info("[arXiv] Searching: %s (max %d)", query, max_results)
    papers: list[dict[str, Any]] = []
    batch_size = min(max_results, 100)
    start = 0

    while len(papers) < max_results:
        params = urllib.parse.urlencode({
            "search_query": full_query,
            "start": start,
            "max_results": batch_size,
        })
        url = f"{ARXIV_BASE}?{params}"
        try:
            raw = _get(url)
        except Exception as exc:
            log.warning("[arXiv] Request failed: %s", exc)
            break

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            break

        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            p = _empty_paper()
            p["source"] = "arxiv"
            p["journal"] = "arXiv preprint"
            p["is_open_access"] = True

            title_el = entry.find("atom:title", ns)
            p["title"] = " ".join((title_el.text or "").split()) if title_el is not None else ""

            summary_el = entry.find("atom:summary", ns)
            p["abstract"] = " ".join((summary_el.text or "").split()) if summary_el is not None else ""

            # Authors
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    p["authors"].append(name_el.text.strip())

            # Year from published date
            pub_el = entry.find("atom:published", ns)
            if pub_el is not None and pub_el.text:
                m = re.search(r"\d{4}", pub_el.text)
                p["year"] = int(m.group()) if m else None

            # IDs — extract arXiv ID and build URLs
            for link in entry.findall("atom:link", ns):
                if link.get("type") == "application/pdf":
                    p["pdf_url"] = link.get("href", "")
            id_el = entry.find("atom:id", ns)
            if id_el is not None and id_el.text:
                p["url"] = id_el.text.strip()

            # DOI if present
            doi_el = entry.find("arxiv:doi", ns)
            if doi_el is not None and doi_el.text:
                p["doi"] = doi_el.text.strip()

            papers.append(p)

        start += len(entries)
        if len(entries) < batch_size:
            break
        time.sleep(3.0)  # arXiv rate limit

    return papers[:max_results]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicates preferring records with more data."""
    seen_doi: dict[str, dict[str, Any]] = {}
    seen_title: dict[str, dict[str, Any]] = {}
    unique: list[dict[str, Any]] = []

    def _score(p: dict[str, Any]) -> int:
        return (
            bool(p["doi"]) * 3
            + bool(p["abstract"]) * 2
            + bool(p["pmid"])
            + len(p["authors"]) // 3
            + len(p["mesh_terms"])
        )

    for p in papers:
        doi_key = p["doi"].lower().strip() if p["doi"] else ""
        title_key = re.sub(r"\W+", " ", p["title"].lower()).strip()[:80]

        if doi_key and doi_key in seen_doi:
            existing = seen_doi[doi_key]
            if _score(p) > _score(existing):
                existing.update(p)
            continue

        if title_key and title_key in seen_title:
            existing = seen_title[title_key]
            if _score(p) > _score(existing):
                existing.update(p)
            continue

        unique.append(p)
        if doi_key:
            seen_doi[doi_key] = p
        if title_key:
            seen_title[title_key] = p

    return unique


# ---------------------------------------------------------------------------
# Folder ID generation
# ---------------------------------------------------------------------------
def _ascii_slug(text: str) -> str:
    """Convert text to ASCII slug."""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return slug


def make_folder_id(paper: dict[str, Any]) -> str:
    """Generate folder name: <first_author>_<year>_<title_slug_3_words>."""
    first_author = ""
    if paper["authors"]:
        # PubMed: "LastName FirstName"; others: "FirstName LastName"
        # Taking the first token gives the last name for PubMed-sourced records
        # and is a reasonable slug for all other sources.
        first_author = _ascii_slug(paper["authors"][0].split()[0])

    year = str(paper["year"]) if paper["year"] else "nodate"

    # Pick 3 meaningful words from title
    stop_words = {
        "a", "an", "the", "and", "or", "for", "of", "in", "on", "at", "to",
        "with", "using", "based", "from", "via", "by", "as", "is", "are",
        "machine", "learning", "deep", "neural", "prediction", "clinical",
        "study", "approach", "model", "models",
    }
    title_words = [
        w for w in re.sub(r"[^a-zA-Z0-9 ]", "", paper["title"]).lower().split()
        if w not in stop_words and len(w) > 2
    ][:3]
    title_slug = "_".join(title_words) if title_words else "paper"

    parts = [p for p in [first_author, year, title_slug] if p]
    return "_".join(parts)[:80]


# ---------------------------------------------------------------------------
# metadata.json builder
# ---------------------------------------------------------------------------
def build_metadata(paper: dict[str, Any], tier: str, domain: str) -> dict[str, Any]:
    """Build a pre-filled metadata.json dict from a paper record."""
    today = date.today().isoformat()
    return {
        "_generated_by": "fetch_papers.py",
        "_source": paper["source"],
        "_fetched_date": today,

        "bibliographic": {
            "title": paper["title"],
            "authors": paper["authors"],
            "journal": paper["journal"],
            "year": paper["year"],
            "doi": paper["doi"],
            "pmid": paper["pmid"],
            "pmcid": paper.get("pmcid", ""),
            "url": paper["url"],
            "pdf_filename": "paper.pdf",
        },

        "study_design": {
            "prediction_type": "binary_classification",
            "outcome": "",
            "prediction_unit": "patient",
            "prediction_horizon": "",
            "setting": "",
            "study_period_start": None,
            "study_period_end": None,
            "is_multicenter": None,
            "has_external_validation": None,
            "external_cohort_description": "",
        },

        "dataset": {
            "source_type": "",
            "_source_type_options": [
                "EHR_single_center", "EHR_multicenter", "public_dataset",
                "registry", "biobank", "claims_data", "mixed",
            ],
            "source_name": "",
            "n_patients_total": None,
            "n_events_positive": None,
            "n_events_negative": None,
            "prevalence_pct": None,
            "split_strategy": "",
            "_split_strategy_options": ["random", "temporal", "site_based", "not_reported"],
            "train_n": None,
            "valid_n": None,
            "test_n": None,
            "features_n": None,
            "has_missing_data": None,
            "missing_data_strategy": "",
        },

        "model": {
            "model_type": "",
            "_model_type_examples": [
                "logistic_regression", "random_forest", "xgboost", "lightgbm",
                "deep_learning", "ensemble", "other",
            ],
            "n_candidate_models": None,
            "selection_criterion": "",
            "hyperparameter_tuning": "",
            "tuning_set": "",
            "_tuning_set_options": [
                "validation_only", "train_validation", "test_used", "not_reported",
            ],
            "feature_selection_method": "",
            "preprocessing_pipeline": "",
        },

        "performance_metrics": {
            "primary_metric": "",
            "test_auroc": None,
            "test_auroc_ci_lower": None,
            "test_auroc_ci_upper": None,
            "test_auprc": None,
            "test_sensitivity": None,
            "test_specificity": None,
            "test_ppv": None,
            "test_npv": None,
            "test_f1": None,
            "test_brier_score": None,
            "calibration_method": "",
            "calibration_reported": None,
            "dca_reported": None,
            "bootstrap_ci_reported": None,
            "n_bootstrap_resamples": None,
            "external_auroc": None,
            "external_auroc_ci_lower": None,
            "external_auroc_ci_upper": None,
        },

        "reporting_standards": {
            "tripod_ai_claimed": None,
            "probast_ai_claimed": None,
            "stard_ai_claimed": None,
            "equator_guideline_cited": "",
            "limitation_section": None,
            "code_availability": "",
            "_code_availability_options": [
                "public_github", "on_request", "not_available", "not_mentioned",
            ],
            "data_availability": "",
            "_data_availability_options": [
                "public", "on_request", "restricted", "not_available", "not_mentioned",
            ],
        },

        "leakage_risk_assessment": {
            "_instructions": "Fill after reading the methods section carefully",
            "patient_level_split_confirmed": None,
            "temporal_split_confirmed": None,
            "preprocessing_fit_on_train_only": None,
            "tuning_used_test_data": None,
            "target_leakage_risk": "",
            "_leakage_risk_options": ["low", "medium", "high", "cannot_assess"],
            "post_index_feature_risk": "",
            "phenotype_definition_overlap_risk": "",
            "notes": "",
        },

        "mlgg_audit": {
            "_instructions": "Populated automatically by audit_external_project.py",
            "audit_run_date": None,
            "mlgg_total_score": None,
            "mlgg_grade": None,
            "gates_passed": None,
            "gates_failed": None,
            "gates_missing": None,
            "conformance_level": None,
            "top_gaps": [],
        },

        "auto_classification": {
            "journal_tier_folder": tier,
            "disease_domain_folder": domain,
            "is_open_access": paper["is_open_access"],
            "pdf_url": paper["pdf_url"],
            "mesh_terms": paper["mesh_terms"],
            "keywords": paper["keywords"],
            "abstract_snippet": paper["abstract"][:500] if paper["abstract"] else "",
        },

        "reviewer_notes": {
            "added_by": "fetch_papers.py (auto)",
            "added_date": today,
            "priority": "medium",
            "notes": "",
        },
    }


# ---------------------------------------------------------------------------
# PDF downloader (open-access only)
# ---------------------------------------------------------------------------
def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF from *url* to *dest*. Returns True on success."""
    if not url:
        return False
    try:
        data = _get(url, retries=2, timeout=30)
        if data[:4] == b"%PDF":
            dest.write_bytes(data)
            log.info("  Downloaded PDF → %s", dest)
            return True
        log.warning("  URL did not return a PDF (got %d bytes)", len(data))
        return False
    except Exception as exc:
        log.warning("  PDF download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------
def write_paper(
    paper: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
    download_pdf_flag: bool,
    manifest_all: list[dict[str, Any]],
) -> None:
    """Create folder + metadata.json for one paper."""
    tier = classify_journal(paper["journal"])
    domain = classify_disease(paper["title"], paper["abstract"], paper["mesh_terms"])
    folder_id = make_folder_id(paper)

    if not folder_id:
        log.warning("Skipping paper with no usable ID: %s", paper.get("title", "")[:60])
        return

    paper_dir = output_dir / tier / domain / folder_id

    if dry_run:
        log.info("[DRY-RUN] Would create: %s", paper_dir)
        return

    paper_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = paper_dir / "audit_output"
    audit_dir.mkdir(exist_ok=True)
    (audit_dir / ".gitkeep").touch()

    metadata_path = paper_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = build_metadata(paper, tier, domain)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        log.info("  Written: %s", metadata_path)
    else:
        log.info("  Skipped (exists): %s", metadata_path)

    if download_pdf_flag and paper.get("pdf_url") and paper.get("is_open_access"):
        pdf_path = paper_dir / "paper.pdf"
        if not pdf_path.exists():
            download_pdf(paper["pdf_url"], pdf_path)
            time.sleep(1.0)

    manifest_all.append({
        "id": folder_id,
        "path": str(paper_dir.relative_to(output_dir.parent)),
        "label": f"{paper['authors'][0].split()[-1] if paper['authors'] else 'Unknown'}"
                 f" et al. {paper['year'] or '?'} — {paper['title'][:60]}",
        "journal": paper["journal"],
        "doi": paper["doi"],
        "source": paper["source"],
        "is_open_access": paper["is_open_access"],
    })


# ---------------------------------------------------------------------------
# Manifest updater
# ---------------------------------------------------------------------------
def update_manifests(output_dir: Path, new_entries: list[dict[str, Any]]) -> None:
    """Append new entries to batch_manifest_all.json (dedup by id)."""
    manifest_path = output_dir / "manifests" / "batch_manifest_all.json"
    if not manifest_path.exists():
        log.warning("Manifest not found at %s; skipping update.", manifest_path)
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    existing_ids = {p["id"] for p in manifest.get("projects", [])}
    added = 0
    for entry in new_entries:
        if entry["id"] not in existing_ids:
            manifest["projects"].append(entry)
            existing_ids.add(entry["id"])
            added += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info("Updated manifest: +%d new entries (%d total)", added, len(manifest["projects"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SOURCE_MAP = {
    "pubmed": fetch_pubmed,
    "pmc": fetch_pmc,
    "semantic_scholar": fetch_semantic_scholar,
    "openalex": fetch_openalex,
    "arxiv": fetch_arxiv,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch medical ML papers from public APIs and build MLGG paper library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--query", required=True, help="Search query string")
    p.add_argument(
        "--sources",
        nargs="+",
        choices=list(SOURCE_MAP.keys()),
        default=["pubmed", "semantic_scholar"],
        metavar="SOURCE",
        help="Sources to query: pubmed pmc semantic_scholar openalex arxiv",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Max results per source (default: 50)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers"),
        help="Root papers directory (default: papers/)",
    )
    p.add_argument(
        "--download-pdf",
        action="store_true",
        help="Download open-access PDFs (PMC, arXiv) where available",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without creating files",
    )
    p.add_argument(
        "--email",
        default="",
        help="Email for NCBI/OpenAlex polite-pool rate limits (recommended)",
    )
    p.add_argument(
        "--no-manifest-update",
        action="store_true",
        help="Skip updating batch_manifest_all.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Collect papers from all sources
    all_papers: list[dict[str, Any]] = []
    for source in args.sources:
        fetcher = SOURCE_MAP[source]
        try:
            results = fetcher(args.query, args.max_results, args.email)
            log.info("[%s] Retrieved %d records", source.upper(), len(results))
            all_papers.extend(results)
        except Exception as exc:
            log.error("[%s] Fatal error: %s", source.upper(), exc)

    log.info("Total before dedup: %d", len(all_papers))
    unique = deduplicate(all_papers)
    log.info("Total after dedup:  %d", len(unique))

    if not unique:
        log.warning("No papers to write.")
        return 0

    manifest_entries: list[dict[str, Any]] = []
    skipped = 0

    for paper in unique:
        if not paper.get("title"):
            skipped += 1
            continue
        write_paper(paper, args.output_dir, args.dry_run, args.download_pdf, manifest_entries)

    if skipped:
        log.info("Skipped %d papers with no title", skipped)

    if not args.dry_run and not args.no_manifest_update and manifest_entries:
        update_manifests(args.output_dir, manifest_entries)

    log.info("Done. Wrote %d paper folders.", len(manifest_entries))

    # Summary table
    tier_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for entry in manifest_entries:
        path_parts = Path(entry["path"]).parts
        # path format: papers/<tier>/<domain>/<id>
        if len(path_parts) >= 3:
            tier_counts[path_parts[1]] = tier_counts.get(path_parts[1], 0) + 1
            domain_counts[path_parts[2]] = domain_counts.get(path_parts[2], 0) + 1

    if tier_counts:
        print("\n=== Journal Tier Distribution ===")
        for tier, count in sorted(tier_counts.items(), key=lambda x: -x[1]):
            print(f"  {tier:<30} {count:>4}")
        print("\n=== Disease Domain Distribution ===")
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            print(f"  {domain:<30} {count:>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
