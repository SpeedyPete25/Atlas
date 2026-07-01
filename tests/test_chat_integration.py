"""Integration tests for /chat endpoint: retrieval, gating, generation, and audit."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from main import app, _evaluate_evidence_gate, _extract_citation_indices, _chosen_references


client = TestClient(app)


# Mock source records for testing
MOCK_HIGH_CONFIDENCE_SOURCE = {
    "title": "High Quality Research",
    "abstract": "This is a well-sourced study showing benefits of X.",
    "authors": ["Smith, J.", "Doe, J."],
    "journal": "Nature",
    "year": "2024",
    "url": "https://example.com/1",
    "doi": "https://doi.org/10.1234/test",
    "source": "PubMed",
    "confidence_score": 0.85,
    "confidence_level": "high",
}

MOCK_MEDIUM_CONFIDENCE_SOURCE = {
    "title": "Medium Quality Study",
    "abstract": "Some evidence about Y with limited scope.",
    "authors": ["Brown, A."],
    "journal": "Journal of Tests",
    "year": "2022",
    "url": "https://example.com/2",
    "doi": "",
    "source": "arXiv",
    "confidence_score": 0.65,
    "confidence_level": "medium",
}

MOCK_LOW_CONFIDENCE_SOURCE = {
    "title": "Preliminary Report",
    "abstract": "Minimal evidence about Z.",
    "authors": [],
    "journal": "Unknown",
    "year": "2020",
    "url": "https://example.com/3",
    "doi": "",
    "source": "Ptable",
    "confidence_score": 0.45,
    "confidence_level": "low",
}


def test_health_endpoint():
    """Test that health endpoint responds."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sources_endpoint():
    """Test /sources endpoint returns policy configuration."""
    response = client.get("/sources")
    assert response.status_code == 200
    data = response.json()

    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "confidence_thresholds" in data
    assert "evidence_gate" in data
    assert "audit_logging" in data


class TestEvidenceGate:
    """Test evidence gating logic."""

    def test_gate_sufficient_evidence(self):
        """Test that gate passes with sufficient high-confidence sources."""
        sources = [MOCK_HIGH_CONFIDENCE_SOURCE, MOCK_MEDIUM_CONFIDENCE_SOURCE]
        result = _evaluate_evidence_gate(sources)
        assert result is None  # None means gate passed

    def test_gate_insufficient_total_sources(self):
        """Test that gate fails with too few sources."""
        sources = [MOCK_LOW_CONFIDENCE_SOURCE]
        result = _evaluate_evidence_gate(sources)
        assert result is not None
        assert "INSUFFICIENT_EVIDENCE" in result

    def test_gate_insufficient_medium_or_higher(self):
        """Test gate fails when medium/high sources are too few."""
        sources = [
            MOCK_LOW_CONFIDENCE_SOURCE,
            MOCK_LOW_CONFIDENCE_SOURCE,
            MOCK_LOW_CONFIDENCE_SOURCE,
        ]
        result = _evaluate_evidence_gate(sources)
        assert result is not None
        assert "INSUFFICIENT_EVIDENCE" in result

    def test_gate_response_structure(self):
        """Test that gated response has expected structure."""
        sources = [MOCK_LOW_CONFIDENCE_SOURCE]
        result = _evaluate_evidence_gate(sources)

        assert "INSUFFICIENT_EVIDENCE" in result
        assert "reason:" in result
        assert "summary:" in result
        assert "total_sources:" in result
        assert "next_steps:" in result


class TestCitationExtraction:
    """Test citation index extraction and verification."""

    def test_extract_single_citation(self):
        """Test extraction of single citation index."""
        answer = "This is true [1] according to research."
        indices = _extract_citation_indices(answer, max_index=3)
        assert 1 in indices

    def test_extract_multiple_citations(self):
        """Test extraction of multiple citations."""
        answer = "Source [1] and [2] agree, unlike [3]."
        indices = _extract_citation_indices(answer, max_index=5)
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices

    def test_extract_ranged_citations(self):
        """Test extraction of comma-separated citations."""
        answer = "As shown in [1, 2, 3], the evidence is clear."
        indices = _extract_citation_indices(answer, max_index=5)
        assert 1 in indices
        assert 2 in indices
        assert 3 in indices

    def test_filter_out_of_range_citations(self):
        """Test that out-of-range citations are excluded."""
        answer = "Research shows [1] but [99] is invalid."
        indices = _extract_citation_indices(answer, max_index=3)
        assert 1 in indices
        assert 99 not in indices

    def test_remove_duplicate_citations(self):
        """Test that duplicate citations are deduplicated."""
        answer = "As noted [1, 1, 1], the research is clear."
        indices = _extract_citation_indices(answer, max_index=5)
        assert indices.count(1) == 1


class TestChosenReferences:
    """Test resolution of citation indices to reference metadata."""

    def test_resolve_valid_citations(self):
        """Test resolving valid citation indices to references."""
        sources = [MOCK_HIGH_CONFIDENCE_SOURCE, MOCK_MEDIUM_CONFIDENCE_SOURCE]
        citations = [1, 2]
        references = _chosen_references(citations, sources)

        assert len(references) == 2
        assert references[0]["index"] == 1
        assert references[1]["index"] == 2
        assert references[0]["title"] == MOCK_HIGH_CONFIDENCE_SOURCE["title"]

    def test_resolve_single_citation(self):
        """Test resolving a single citation."""
        sources = [MOCK_HIGH_CONFIDENCE_SOURCE, MOCK_MEDIUM_CONFIDENCE_SOURCE]
        citations = [1]
        references = _chosen_references(citations, sources)

        assert len(references) == 1
        assert references[0]["source"] == "PubMed"
        assert "url" in references[0]

    def test_resolve_out_of_range_ignored(self):
        """Test that out-of-range indices are gracefully ignored."""
        sources = [MOCK_HIGH_CONFIDENCE_SOURCE]
        citations = [1, 99]
        references = _chosen_references(citations, sources)

        assert len(references) == 1
        assert references[0]["index"] == 1


@pytest.mark.asyncio
async def test_chat_no_sources_found():
    """Test /chat returns graceful message when no sources retrieved."""
    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = ([], [])

        with patch("main._audit_log"):
            response = client.post(
                "/chat",
                json={"question": "What is X?", "model": "llama3"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "No relevant scientific literature" in data["answer"]
            assert len(data["sources"]) == 0


@pytest.mark.asyncio
async def test_chat_gated_response():
    """Test /chat returns gated response when evidence is weak."""
    weak_sources = [MOCK_LOW_CONFIDENCE_SOURCE]

    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = (weak_sources, [])

        with patch("main._audit_log"):
            response = client.post(
                "/chat",
                json={"question": "What is X?", "model": "llama3"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "INSUFFICIENT_EVIDENCE" in data["answer"]
            assert len(data["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_generates_answer():
    """Test /chat generates answer and includes sources."""
    strong_sources = [
        MOCK_HIGH_CONFIDENCE_SOURCE,
        MOCK_MEDIUM_CONFIDENCE_SOURCE,
    ]

    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = (strong_sources, [])

        with patch("main.generate_answer") as mock_generate:
            mock_generate.return_value = "Based on [1] and [2], the answer is positive."

            with patch("main._audit_log"):
                response = client.post(
                    "/chat",
                    json={"question": "What is X?", "model": "llama3"},
                )

                assert response.status_code == 200
                data = response.json()
                assert "[1]" in data["answer"]
                assert len(data["sources"]) == 2


@pytest.mark.asyncio
async def test_chat_audit_logging():
    """Test that chat endpoint logs audit events."""
    sources = [MOCK_HIGH_CONFIDENCE_SOURCE]

    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = (sources, [])

        with patch("main.generate_answer") as mock_generate:
            mock_generate.return_value = "Test answer [1]."

            with patch("main._audit_log") as mock_audit:
                response = client.post(
                    "/chat",
                    json={"question": "Test?", "model": "llama3"},
                )

                assert response.status_code == 200
                # Verify audit_log was called multiple times for various events
                assert mock_audit.call_count > 0
                # Check for expected event types
                calls = [call[0][0] for call in mock_audit.call_args_list]
                assert "chat_request" in calls
                assert "retrieval_result" in calls
                assert "chat_answer" in calls


@pytest.mark.asyncio
async def test_chat_retrieval_error_returns_502():
    """Test /chat returns 502 when retrieval fails."""
    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = ([], ["PubMed retrieval failed: timeout"])

        with patch("main._audit_log"):
            response = client.post(
                "/chat",
                json={"question": "What is X?", "model": "llama3"},
            )

            assert response.status_code == 502


@pytest.mark.asyncio
async def test_chat_ollama_error_returns_503():
    """Test /chat returns 503 when Ollama generation fails."""
    sources = [MOCK_HIGH_CONFIDENCE_SOURCE]

    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = (sources, [])

        with patch("main.generate_answer") as mock_generate:
            mock_generate.side_effect = Exception("Ollama unreachable")

            with patch("main._audit_log"):
                response = client.post(
                    "/chat",
                    json={"question": "Test?", "model": "llama3"},
                )

                assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_source_limits_respected():
    """Test /chat respects source limit overrides."""
    with patch("main.search_all_sources") as mock_search:
        mock_search.return_value = ([], [])

        with patch("main._audit_log"):
            response = client.post(
                "/chat",
                json={
                    "question": "Test?",
                    "model": "llama3",
                    "source_limits": {"pubmed": 10, "arxiv": 0, "ptable": 0},
                },
            )

            assert response.status_code == 200
            # Verify search_all_sources was called with the correct limits
            mock_search.assert_called_once()
            call_limits = mock_search.call_args[0][1]
            assert call_limits["pubmed"] == 10
            assert call_limits["arxiv"] == 0
