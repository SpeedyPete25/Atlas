import httpx
import xml.etree.ElementTree as ET
from typing import List, Dict

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_SEARCH = "http://export.arxiv.org/api/query"

# Priority journals requested by the user
PRIORITY_JOURNALS = [
    "Nature", "Science", "Proc Natl Acad Sci", "N Engl J Med", "Lancet"
]


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
    async with httpx.AsyncClient(timeout=20) as client:
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
