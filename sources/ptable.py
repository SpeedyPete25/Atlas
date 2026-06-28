import re
from typing import List
from urllib.parse import quote

import httpx
from .types import SourceResults

"""Ptable-based chemistry source adapter.

This module resolves formulas from direct chemical notation and name-based
lookup (PubChem), then queries Ptable compounds and normalizes results into the
Atlas source contract.
"""

PTABLE_COMPOUNDS = "https://ptable.com/JSON/compounds/"
PUBCHEM_NAME_TO_FORMULA = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/MolecularFormula/JSON"
)
FORMULA_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
SYMBOL_PATTERN = re.compile(r"([A-Z][a-z]?)\d*")

# Alias fallback for colloquial names PubChem may not always resolve reliably.
COMPOUND_ALIASES = {
    "washing soda": "Na2CO3",
    "baking soda": "NaHCO3",
    "table salt": "NaCl",
}

NAME_RESOLUTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "can",
    "dangerous",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "its",
    "mass",
    "me",
    "molar",
    "molecular",
    "name",
    "of",
    "on",
    "properties",
    "the",
    "to",
    "tell",
    "weight",
    "what",
    "with",
}


def _extract_formulas(query: str) -> List[str]:
    """Extract formula-like tokens (for example Na2CO3) from free-text queries."""

    formulas = []
    seen = set()
    for match in FORMULA_PATTERN.findall(query):
        if match not in seen:
            seen.add(match)
            formulas.append(match)
    return formulas


def _normalize_query(text: str) -> str:
    """Normalize free text for phrase extraction and name-resolution heuristics."""

    normalized = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "


def _candidate_name_phrases(query: str) -> List[str]:
    """Generate likely compound name phrases from a question string."""

    normalized = _normalize_query(query).strip()
    tokens = normalized.split()
    if not tokens:
        return []

    filtered = [token for token in tokens if token not in NAME_RESOLUTION_STOPWORDS]
    if not filtered:
        filtered = tokens

    phrases = []
    seen = set()

    def add_phrase(value: str):
        phrase = value.strip()
        if not phrase or phrase in seen:
            return
        seen.add(phrase)
        phrases.append(phrase)

    add_phrase(" ".join(filtered))

    max_len = min(4, len(filtered))
    for size in range(max_len, 0, -1):
        for start in range(0, len(filtered) - size + 1):
            add_phrase(" ".join(filtered[start:start + size]))
            if len(phrases) >= 12:
                return phrases

    return phrases


async def _resolve_formula_from_name(name: str, client: httpx.AsyncClient) -> str:
    """Resolve a compound name to molecular formula using aliases then PubChem."""

    lowered = name.lower().strip()
    if lowered in COMPOUND_ALIASES:
        return COMPOUND_ALIASES[lowered]

    encoded_name = quote(name, safe="")
    url = PUBCHEM_NAME_TO_FORMULA.format(encoded_name)

    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return ""

    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return ""

    formula = properties[0].get("MolecularFormula", "")
    return formula if isinstance(formula, str) else ""


async def _extract_named_formulas(query: str, client: httpx.AsyncClient) -> List[str]:
    """Resolve up to three unique formulas from candidate name phrases."""

    formulas = []
    seen = set()

    for phrase in _candidate_name_phrases(query):
        formula = await _resolve_formula_from_name(phrase, client)
        if formula and formula not in seen:
            seen.add(formula)
            formulas.append(formula)
        if len(formulas) >= 3:
            break

    return formulas


def _symbols_signature(formula: str) -> str:
    """Build a symbol signature used by Ptable formula lookup endpoint."""

    symbols = sorted(set(SYMBOL_PATTERN.findall(formula)))
    return "".join(symbols)


async def search_ptable(query: str, max_results: int = 3) -> SourceResults:
    """Search Ptable compounds based on formulas extracted/resolved from query.

    The function returns Atlas-formatted source records and may include a linked
    Wikipedia reference when available from Ptable metadata.
    """

    results: SourceResults = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        formulas = []
        seen = set()
        named_formulas = await _extract_named_formulas(query, client)
        for formula in _extract_formulas(query) + named_formulas:
            if formula not in seen:
                seen.add(formula)
                formulas.append(formula)

        if not formulas:
            return []

        for formula in formulas[:3]:
            signature = _symbols_signature(formula)
            if not signature:
                continue

            try:
                resp = await client.get(f"{PTABLE_COMPOUNDS}formula={signature}")
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                continue

            matches = payload.get("matches", [])
            if not matches:
                continue

            exact = next(
                (match for match in matches if match.get("molecularformula", "") == formula),
                None,
            )
            best = exact or matches[0]

            name = ", ".join(best.get("allnames", [])[:3]) or "Unknown compound"
            molecular = best.get("molecularformula", formula)
            article = (best.get("articles") or [""])[0] or ""
            wikipedia_url = (
                f"https://en.wikipedia.org/wiki/{article.replace(' ', '_')}"
                if article
                else ""
            )

            abstract = (
                f"Ptable compound match for formula {formula}. "
                f"Closest cataloged compound: {molecular}. "
                f"Names: {name}."
            )

            results.append({
                "title": f"Ptable compound record: {molecular}",
                "abstract": abstract,
                "authors": ["Ptable"],
                "journal": "Ptable (periodic table/compound index)",
                "year": "",
                "url": f"https://ptable.com/#Compounds?formula={formula}",
                "doi": "",
                "source": "Ptable",
            })

            if wikipedia_url:
                results.append({
                    "title": f"Wikipedia article for {molecular} (linked by Ptable)",
                    "abstract": f"Reference page linked from Ptable for compound {molecular}.",
                    "authors": ["Wikipedia"],
                    "journal": "Wikipedia",
                    "year": "",
                    "url": wikipedia_url,
                    "doi": "",
                    "source": "Ptable",
                })

            if len(results) >= max_results:
                break

    return results[:max_results]