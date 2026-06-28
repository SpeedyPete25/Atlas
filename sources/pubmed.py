import xml.etree.ElementTree as ET
from typing import List

import httpx
from .types import SourceResults

"""PubMed/NCBI source adapter.

This module queries NCBI E-utilities, then normalizes XML article metadata into
the common Atlas source record shape.
"""

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


async def search_pubmed(query: str, max_results: int = 5) -> SourceResults:
    """Search PubMed and return source records in Atlas contract format.

    Returns an empty list when no IDs are found or parsing fails.
    Network failures are surfaced to the caller.
    """
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


def _parse_pubmed_xml(xml_text: str) -> SourceResults:
    """Parse PubMed XML payload into Atlas source dictionaries.

    Best-effort parsing is used: malformed records are skipped and parsing
    errors return an empty list.
    """

    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for article in root.findall(".//PubmedArticle"):
        try:
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown"

            abstract_parts = [
                "".join(element.itertext()).strip()
                for element in article.findall(".//AbstractText")
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