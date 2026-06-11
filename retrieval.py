import httpx
import xml.etree.ElementTree as ET
import re
from typing import List, Dict

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_SEARCH = "https://export.arxiv.org/api/query"
PTABLE_COMPOUNDS = "https://ptable.com/JSON/compounds/"

# Priority journals requested by the user
PRIORITY_JOURNALS = [
    "Nature", "Science", "Proc Natl Acad Sci", "N Engl J Med", "Lancet"
]

FORMULA_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
SYMBOL_PATTERN = re.compile(r"([A-Z][a-z]?)\d*")

# Common chemistry names/aliases for name-only questions.
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


async def search_pubmed(query: str, max_results: int = 5) -> List[Dict]:
    """Search PubMed/NCBI and return a list of paper metadata dicts."""
    async with httpx.AsyncClient(timeout=20) as client:
        search_resp = await client.get(PUBMED_SEARCH, params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_resp = await client.get(PUBMED_FETCH, params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
            "rettype": "abstract",
        })
        fetch_resp.raise_for_status()
        return _parse_pubmed_xml(fetch_resp.text)


def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for article in root.findall(".//PubmedArticle"):
        try:
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown"

            # Collect all AbstractText sections (structured abstracts)
            abstract_parts = [
                "".join(el.itertext()).strip()
                for el in article.findall(".//AbstractText")
            ]
            abstract = " ".join(abstract_parts)[:1500]

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else "Unknown Journal"

            year_el = article.find(".//PubDate/Year")
            year = year_el.text if year_el is not None else ""

            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            doi_el = article.find(".//ArticleId[@IdType='doi']")
            doi = doi_el.text if doi_el is not None else ""

            authors = []
            for author in article.findall(".//Author")[:4]:
                last = author.find("LastName")
                first = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if first is not None:
                        name += f" {first.text}"
                    authors.append(name)

            if pmid:
                results.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": journal,
                    "year": year,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "doi": f"https://doi.org/{doi}" if doi else "",
                    "source": "PubMed",
                })
        except Exception:
            continue

    return results


async def search_arxiv(query: str, max_results: int = 4) -> List[Dict]:
    """Search arXiv and return a list of paper metadata dicts."""
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(ARXIV_SEARCH, params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        })
        resp.raise_for_status()
        return _parse_arxiv_xml(resp.text)


def _parse_arxiv_xml(xml_text: str) -> List[Dict]:
    results = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for entry in root.findall("atom:entry", ns):
        try:
            title_el = entry.find("atom:title", ns)
            title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Unknown"

            summary_el = entry.find("atom:summary", ns)
            abstract = summary_el.text.strip()[:1500] if summary_el is not None else ""

            id_el = entry.find("atom:id", ns)
            url = id_el.text.strip() if id_el is not None else ""

            published_el = entry.find("atom:published", ns)
            year = published_el.text[:4] if published_el is not None else ""

            authors = []
            for author in entry.findall("atom:author", ns)[:4]:
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text)

            # Extract arXiv categories for the "journal" field
            categories = [
                tag.get("term", "")
                for tag in entry.findall("{http://arxiv.org/schemas/atom}primary_category", ns)
            ]
            category = categories[0] if categories else "arXiv"

            if url:
                results.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": f"arXiv [{category}] (preprint)",
                    "year": year,
                    "url": url,
                    "doi": "",
                    "source": "arXiv",
                })
        except Exception:
            continue

    return results


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
    """Search ptable compounds endpoint for formulas found in user query."""
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
                (m for m in matches if m.get("molecularformula", "") == formula),
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
