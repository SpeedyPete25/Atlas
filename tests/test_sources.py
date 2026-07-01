"""Tests for source adapter contract compliance and error handling."""

import pytest
from unittest.mock import AsyncMock, patch
from sources import SourceRecord


# Required keys that every source adapter must return
REQUIRED_SOURCE_FIELDS = {
    "title",
    "abstract",
    "authors",
    "journal",
    "year",
    "url",
    "doi",
    "source",
}


def validate_source_record(record: dict) -> bool:
    """Check that a source record contains all required fields with correct types."""
    if not isinstance(record, dict):
        return False

    for field in REQUIRED_SOURCE_FIELDS:
        if field not in record:
            return False

    # Type checks for key fields
    if not isinstance(record.get("title"), str):
        return False
    if not isinstance(record.get("abstract"), str):
        return False
    if not isinstance(record.get("authors"), list):
        return False
    if not all(isinstance(a, str) for a in record.get("authors", [])):
        return False
    if not isinstance(record.get("journal"), str):
        return False
    if not isinstance(record.get("url"), str):
        return False

    return True


@pytest.mark.asyncio
async def test_pubmed_contract():
    """Test that PubMed adapter returns records matching the source contract."""
    from sources.pubmed import search_pubmed

    # Mock successful PubMed response
    mock_search_response = {
        "esearchresult": {"idlist": ["12345678"]}
    }
    mock_fetch_response = """
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>Test Article Title</ArticleTitle>
                    <Journal><Title>Test Journal</Title></Journal>
                    <PubDate><Year>2024</Year></PubDate>
                    <Abstract><AbstractText>Test abstract content.</AbstractText></Abstract>
                    <AuthorList>
                        <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
                    </AuthorList>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1234/test</ArticleId>
                    </ArticleIdList>
                </Article>
            </MedlineCitation>
            <PMID>12345678</PMID>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    with patch("sources.pubmed.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # First call is esearch, second is efetch
        mock_search = AsyncMock()
        mock_search.json.return_value = mock_search_response
        mock_search.raise_for_status = lambda: None

        mock_fetch = AsyncMock()
        mock_fetch.text = mock_fetch_response
        mock_fetch.raise_for_status = lambda: None

        mock_client.get.side_effect = [mock_search, mock_fetch]

        results = await search_pubmed("test query", max_results=1)

        assert isinstance(results, list)
        assert len(results) > 0
        for record in results:
            assert validate_source_record(record), f"Invalid record: {record}"
            assert record["source"] == "PubMed"


@pytest.mark.asyncio
async def test_arxiv_contract():
    """Test that arXiv adapter returns records matching the source contract."""
    from sources.arxiv import search_arxiv

    mock_arxiv_response = """
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
        <entry>
            <title>Test arXiv Paper</title>
            <summary>Test abstract for arXiv paper.</summary>
            <author><name>Test Author</name></author>
            <published>2024-01-15T00:00:00Z</published>
            <id>http://arxiv.org/abs/2401.12345v1</id>
            <arxiv:primary_category term="cs.AI"/>
        </entry>
    </feed>
    """

    with patch("sources.arxiv.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.text = mock_arxiv_response
        mock_resp.raise_for_status = lambda: None
        mock_client.get.return_value = mock_resp

        results = await search_arxiv("test query", max_results=1)

        assert isinstance(results, list)
        assert len(results) > 0
        for record in results:
            assert validate_source_record(record), f"Invalid record: {record}"
            assert record["source"] == "arXiv"


@pytest.mark.asyncio
async def test_ptable_contract():
    """Test that Ptable adapter returns records matching the source contract."""
    from sources.ptable import search_ptable

    mock_ptable_response = {
        "matches": [
            {
                "molecularformula": "NaCl",
                "allnames": ["Sodium chloride", "Table salt"],
                "articles": ["Sodium_chloride"],
            }
        ]
    }

    with patch("sources.ptable.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.json.return_value = mock_ptable_response
        mock_resp.raise_for_status = lambda: None
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp

        results = await search_ptable("NaCl", max_results=2)

        assert isinstance(results, list)
        for record in results:
            assert validate_source_record(record), f"Invalid record: {record}"
            assert record["source"] == "Ptable"


@pytest.mark.asyncio
async def test_pubmed_empty_results():
    """Test PubMed gracefully handles no results."""
    from sources.pubmed import search_pubmed

    with patch("sources.pubmed.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.json.return_value = {"esearchresult": {"idlist": []}}
        mock_resp.raise_for_status = lambda: None
        mock_client.get.return_value = mock_resp

        results = await search_pubmed("obscure xyz query", max_results=5)

        assert isinstance(results, list)
        assert len(results) == 0


@pytest.mark.asyncio
async def test_arxiv_network_failure():
    """Test arXiv adapter handles network errors gracefully."""
    from sources.arxiv import search_arxiv

    with patch("sources.arxiv.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_client.get.side_effect = Exception("Network timeout")

        with pytest.raises(Exception):
            await search_arxiv("test", max_results=5)
