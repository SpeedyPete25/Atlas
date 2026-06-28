"""Shared type contracts for Atlas retrieval source adapters.

Use these aliases in source modules to keep the output schema consistent across
all retrievers.
"""

from typing import List, Literal, NotRequired, TypedDict


class SourceRecord(TypedDict):
    """Normalized source record contract returned by retrievers."""

    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    url: str
    doi: str
    source: str
    confidence_score: NotRequired[float]
    confidence_level: NotRequired[Literal["low", "medium", "high"]]


SourceResults = List[SourceRecord]
