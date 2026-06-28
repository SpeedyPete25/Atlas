"""Public exports for Atlas retrieval source adapters.

Each exported search function should return a list of dictionaries matching the
shared source contract used by retrieval and API response models.
"""

from .arxiv import search_arxiv
from .ptable import search_ptable
from .pubmed import search_pubmed
from .types import SourceRecord, SourceResults

__all__ = [
	"search_pubmed",
	"search_arxiv",
	"search_ptable",
	"SourceRecord",
	"SourceResults",
]