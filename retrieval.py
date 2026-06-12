from dataclasses import dataclass
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


def get_source_definitions() -> Dict[str, SourceDefinition]:
    return SOURCE_DEFINITIONS


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

    return merged_sources, source_errors