import re
from typing import Dict, List

import httpx

PTABLE_COMPOUNDS = "https://ptable.com/JSON/compounds/"
FORMULA_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
SYMBOL_PATTERN = re.compile(r"([A-Z][a-z]?)\d*")

COMPOUND_NAME_TO_FORMULA = {
    "sodium carbonate": "Na2CO3",
    "washing soda": "Na2CO3",
    "sodium bicarbonate": "NaHCO3",
    "baking soda": "NaHCO3",
    "sodium chloride": "NaCl",
    "table salt": "NaCl",
    "water": "H2O",
    "hydrogen peroxide": "H2O2",
    "carbon dioxide": "CO2",
    "carbon monoxide": "CO",
    "methane": "CH4",
    "ethanol": "C2H6O",
    "glucose": "C6H12O6",
    "sulfuric acid": "H2SO4",
    "sulphuric acid": "H2SO4",
    "hydrochloric acid": "HCl",
    "nitric acid": "HNO3",
    "acetic acid": "C2H4O2",
    "ammonia": "NH3",
}


def _extract_formulas(query: str) -> List[str]:
    formulas = []
    seen = set()
    for match in FORMULA_PATTERN.findall(query):
        if match not in seen:
            seen.add(match)
            formulas.append(match)
    return formulas


def _normalize_query(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "


def _extract_named_formulas(query: str) -> List[str]:
    normalized_query = _normalize_query(query)
    formulas = []
    seen = set()
    for name, formula in COMPOUND_NAME_TO_FORMULA.items():
        needle = f" {_normalize_query(name).strip()} "
        if needle in normalized_query and formula not in seen:
            seen.add(formula)
            formulas.append(formula)
    return formulas


def _symbols_signature(formula: str) -> str:
    symbols = sorted(set(SYMBOL_PATTERN.findall(formula)))
    return "".join(symbols)


async def search_ptable(query: str, max_results: int = 3) -> List[Dict]:
    """Search Ptable compounds endpoint for formulas found in user query."""
    formulas = []
    seen = set()
    for formula in _extract_formulas(query) + _extract_named_formulas(query):
        if formula not in seen:
            seen.add(formula)
            formulas.append(formula)

    if not formulas:
        return []

    results = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
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