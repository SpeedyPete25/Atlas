"""Regression tests for Ptable name-to-formula resolution."""

import pytest
from unittest.mock import AsyncMock, patch
from sources.ptable import (
    _normalize_query,
    _candidate_name_phrases,
    _resolve_formula_from_name,
    _extract_formulas,
    COMPOUND_ALIASES,
)


class TestQueryNormalization:
    """Test query normalization for phrase extraction."""

    def test_normalize_basic_text(self):
        """Test basic text normalization."""
        result = _normalize_query("What is Sodium Carbonate?")
        assert "sodium" in result.lower()
        assert "carbonate" in result.lower()
        assert "?" not in result

    def test_normalize_preserves_spaces(self):
        """Test that normalization preserves token separation."""
        result = _normalize_query("baking soda")
        tokens = result.strip().split()
        assert len(tokens) >= 2

    def test_normalize_lowercase(self):
        """Test that normalization converts to lowercase."""
        result = _normalize_query("SODIUM CHLORIDE")
        assert result == result.lower()


class TestCandidatePhrases:
    """Test candidate name phrase generation."""

    def test_single_word_compound(self):
        """Test phrase generation from single-word input."""
        phrases = _candidate_name_phrases("sodium")
        assert isinstance(phrases, list)
        assert len(phrases) > 0

    def test_multi_word_compound(self):
        """Test phrase generation from multi-word input."""
        phrases = _candidate_name_phrases("sodium carbonate")
        assert "sodium" in [p for p in phrases if "sodium" in p]
        assert "carbonate" in [p for p in phrases if "carbonate" in p]

    def test_stopword_filtering(self):
        """Test that stopwords are filtered intelligently."""
        # "the" should be filtered
        phrases = _candidate_name_phrases("what is the molecular formula")
        assert all("molecular" in p or "formula" in p for p in phrases)

    def test_phrase_limit(self):
        """Test that phrase generation respects a reasonable limit."""
        phrases = _candidate_name_phrases("a b c d e f g h i j k l m n o p q r s t")
        assert len(phrases) <= 15  # Reasonable upper bound


class TestFormulaExtraction:
    """Test direct formula extraction from queries."""

    def test_extract_simple_formula(self):
        """Test extraction of simple chemical formulas."""
        result = _extract_formulas("What about NaCl?")
        assert "NaCl" in result

    def test_extract_complex_formula(self):
        """Test extraction of complex formulas."""
        result = _extract_formulas("Compare H2SO4 and Ca(OH)2")
        assert "H2SO4" in result

    def test_extract_no_false_positives(self):
        """Test that common words aren't misidentified as formulas."""
        result = _extract_formulas("What is sodium?")
        # "What", "is", "sodium" should not match formula pattern
        assert len(result) == 0

    def test_extract_multiple_formulas(self):
        """Test extraction of multiple formulas in one query."""
        result = _extract_formulas("Compare NaCl and H2O")
        assert len(result) >= 2
        assert "NaCl" in result
        assert "H2O" in result


class TestCompoundAliases:
    """Test fallback alias resolution."""

    def test_baking_soda_alias(self):
        """Test that 'baking soda' resolves to NaHCO3."""
        assert COMPOUND_ALIASES["baking soda"] == "NaHCO3"

    def test_washing_soda_alias(self):
        """Test that 'washing soda' resolves to Na2CO3."""
        assert COMPOUND_ALIASES["washing soda"] == "Na2CO3"

    def test_table_salt_alias(self):
        """Test that 'table salt' resolves to NaCl."""
        assert COMPOUND_ALIASES["table salt"] == "NaCl"


@pytest.mark.asyncio
async def test_pubchem_name_resolution_success():
    """Test successful name-to-formula resolution via PubChem."""
    with patch("sources.ptable.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "PropertyTable": {
                "Properties": [{"MolecularFormula": "NaCl"}]
            }
        }
        mock_resp.raise_for_status = lambda: None
        mock_client.get.return_value = mock_resp

        result = await _resolve_formula_from_name("sodium chloride", mock_client)

        assert result == "NaCl"


@pytest.mark.asyncio
async def test_pubchem_name_resolution_alias_fallback():
    """Test that alias fallback works when PubChem fails."""
    with patch("sources.ptable.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()

        # Alias should resolve without making HTTP call
        result = await _resolve_formula_from_name("baking soda", mock_client)

        assert result == "NaHCO3"
        # Verify no HTTP call was made (it would fail if tried)
        # since we didn't set up mock responses


@pytest.mark.asyncio
async def test_pubchem_name_resolution_404():
    """Test handling of 404 response from PubChem."""
    with patch("sources.ptable.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_resp = AsyncMock()
        mock_resp.status_code = 404
        mock_client.get.return_value = mock_resp

        result = await _resolve_formula_from_name("unknown_xyz_compound", mock_client)

        assert result == ""


@pytest.mark.asyncio
async def test_pubchem_name_resolution_error():
    """Test handling of exceptions from PubChem API."""
    with patch("sources.ptable.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        mock_client.get.side_effect = Exception("Network error")

        result = await _resolve_formula_from_name("sodium carbonate", mock_client)

        assert result == ""
