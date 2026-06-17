from dataclasses import dataclass
from datetime import datetime
import os
from typing import Awaitable, Callable, Dict, List, Tuple

from sources import search_arxiv, search_ptable, search_pubmed


# Contributor guide for new sources:
# - Implement an async function with signature: search_<source>(query: str, max_results: int) -> List[Dict]
# - Return a list of dictionaries with exactly these keys so the API and UI can render them consistently:
#   {
#       "title": str,
#       "abstract": str,
#       "authors": List[str],
#       "journal": str,
#       "year": str,
#       "url": str,
#       "doi": str,
#       "source": str,
#   }
# - Then register the function once in SOURCE_DEFINITIONS below.


@dataclass(frozen=True)
class SourceDefinition:
    key: str
    label: str
    default_max_results: int
    search: Callable[[str, int], Awaitable[List[Dict]]]


SOURCE_DEFINITIONS = {
    "pubmed": SourceDefinition(
        key="pubmed",
        label="PubMed",
        default_max_results=4,
        search=search_pubmed,
    ),
    "arxiv": SourceDefinition(
        key="arxiv",
        label="arXiv",
        default_max_results=3,
        search=search_arxiv,
    ),
    "ptable": SourceDefinition(
        key="ptable",
        label="Ptable",
        default_max_results=2,
        search=search_ptable,
    ),
}

BASE_SOURCE_TRUST = {
    "pubmed": 0.78,
    "arxiv": 0.58,
    "ptable": 0.48,
}

PRIORITY_JOURNALS = (
    "nature",
    "science",
    "proc natl acad sci",
    "n engl j med",
    "lancet",
)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


_confidence_high = _float_env("ATLAS_CONFIDENCE_HIGH", 0.80)
_confidence_medium = _float_env("ATLAS_CONFIDENCE_MEDIUM", 0.60)
if _confidence_medium > _confidence_high:
    _confidence_medium = _confidence_high

CONFIDENCE_THRESHOLDS = {
    "high": _confidence_high,
    "medium": _confidence_medium,
}


def _source_key(source: Dict) -> str:
    return str(source.get("source", "")).strip().lower()


def _parse_year(value: str) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 4:
        return None
    year = int(digits[:4])
    now = datetime.now().year
    if 1800 <= year <= now + 1:
        return year
    return None


def _recency_component(source: Dict) -> float:
    year = _parse_year(source.get("year", ""))
    if year is None:
        return 0.0
    age = datetime.now().year - year
    if age <= 2:
        return 0.12
    if age <= 5:
        return 0.09
    if age <= 10:
        return 0.06
    if age <= 20:
        return 0.03
    return 0.01


def _metadata_component(source: Dict) -> float:
    score = 0.0
    abstract = str(source.get("abstract", "")).strip()
    authors = source.get("authors") or []
    doi = str(source.get("doi", "")).strip()
    url = str(source.get("url", "")).strip()
    journal = str(source.get("journal", "")).strip()

    if len(abstract) >= 120:
        score += 0.06
    elif len(abstract) >= 40:
        score += 0.03

    if isinstance(authors, list) and authors:
        score += 0.04

    if doi:
        score += 0.05

    if url.startswith("http"):
        score += 0.02

    if journal:
        score += 0.03

    return score


def _journal_component(source: Dict) -> float:
    journal = str(source.get("journal", "")).lower()
    if any(name in journal for name in PRIORITY_JOURNALS):
        return 0.08
    return 0.0


def _compute_confidence(source: Dict) -> float:
    source_key = _source_key(source)
    base = BASE_SOURCE_TRUST.get(source_key, 0.45)
    score = base + _recency_component(source) + _metadata_component(source) + _journal_component(source)
    return max(0.0, min(1.0, score))


def _confidence_level(score: float) -> str:
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def _rank_sources(sources: List[Dict]) -> List[Dict]:
    ranked = []
    for source in sources:
        source_copy = dict(source)
        score = round(_compute_confidence(source_copy), 3)
        source_copy["confidence_score"] = score
        source_copy["confidence_level"] = _confidence_level(score)
        ranked.append(source_copy)

    ranked.sort(
        key=lambda item: (
            item.get("confidence_score", 0.0),
            _parse_year(item.get("year", "")) or 0,
        ),
        reverse=True,
    )
    return ranked


def get_source_definitions() -> Dict[str, SourceDefinition]:
    return SOURCE_DEFINITIONS


def get_confidence_thresholds() -> Dict[str, float]:
    return CONFIDENCE_THRESHOLDS


def get_default_source_limits() -> Dict[str, int]:
    return {
        key: source.default_max_results
        for key, source in SOURCE_DEFINITIONS.items()
    }


def normalize_source_limits(overrides: Dict[str, int] | None = None) -> Dict[str, int]:
    limits = get_default_source_limits()
    if not overrides:
        return limits

    for key, value in overrides.items():
        if key not in SOURCE_DEFINITIONS:
            continue
        try:
            limit = int(value)
        except (TypeError, ValueError):
            continue
        if limit >= 0:
            limits[key] = limit

    return limits


async def search_all_sources(
    query: str,
    source_limits: Dict[str, int] | None = None,
) -> Tuple[List[Dict], List[str]]:
    limits = normalize_source_limits(source_limits)
    enabled_sources = [
        source
        for key, source in SOURCE_DEFINITIONS.items()
        if limits.get(key, 0) > 0
    ]

    if not enabled_sources:
        return [], []

    import asyncio

    results = await asyncio.gather(
        *(source.search(query, limits[source.key]) for source in enabled_sources),
        return_exceptions=True,
    )

    merged_sources = []
    source_errors = []
    for source, result in zip(enabled_sources, results):
        if isinstance(result, Exception):
            source_errors.append(f"{source.label} retrieval failed: {result}")
            continue
        merged_sources.extend(result)

    return _rank_sources(merged_sources), source_errors