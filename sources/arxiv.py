import xml.etree.ElementTree as ET
from typing import List

import httpx
from .types import SourceResults

"""arXiv source adapter.

This module queries the arXiv Atom API and normalizes entries into the common
Atlas source record shape.
"""

ARXIV_SEARCH = "https://export.arxiv.org/api/query"


async def search_arxiv(query: str, max_results: int = 4) -> SourceResults:
    """Search arXiv and return Atlas-formatted source records.

    Redirects are followed to handle API endpoint behavior robustly.
    """
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        resp = await client.get(ARXIV_SEARCH, params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        })
        resp.raise_for_status()
        return _parse_arxiv_xml(resp.text)


def _parse_arxiv_xml(xml_text: str) -> SourceResults:
    """Parse Atom XML from arXiv into Atlas source dictionaries.

    Invalid entries are skipped; malformed XML returns an empty list.
    """

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