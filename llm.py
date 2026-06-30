import httpx
import re
from typing import List

from sources.types import SourceResults

"""LLM integration and citation post-processing.

This module builds a constrained evidence prompt, sends it to Ollama, and
post-processes model output so citation indices only reference retrieved
sources.
"""

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

_CITATION_BLOCK_RE = re.compile(r"\[([^\[\]]*\d[^\[\]]*)\]")


def _build_system_prompt() -> str:
    """Return strict behavior instructions for evidence-grounded generation."""
    return (
        "You are a rigorous scientific assistant. Your role is to answer questions "
        "ONLY using the peer-reviewed and verified scientific sources provided to you. "
        "Rules you must follow:\n"
        "1. Cite every claim using the reference numbers provided, e.g. [1], [2].\n"
        "2. If the provided sources do not contain enough information to answer the question, "
        "state clearly: 'The available sources do not provide sufficient information on this topic.'\n"
        "3. Never introduce knowledge not found in the provided sources.\n"
        "4. Do not speculate beyond what the abstracts state.\n"
        "5. When sources contradict each other, acknowledge the disagreement and cite both.\n"
        "6. Keep your answer factual, concise, and well-structured."
    )


def _build_user_message(question: str, sources: SourceResults) -> str:
    """Build retrieval context with numbered references consumed by the model."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        authors_str = ", ".join(src.get("authors", [])) or "Unknown authors"
        context_parts.append(
            f"[{i}] **{src['title']}**\n"
            f"    Source: {src['source']} | Journal: {src['journal']} | Year: {src['year']}\n"
            f"    Authors: {authors_str}\n"
            f"    Abstract: {src['abstract']}\n"
            f"    URL: {src['url']}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"SCIENTIFIC SOURCES:\n\n{context}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer strictly based on the sources above, citing them by number."
    )


def _verify_and_fix_citations(answer: str, total_sources: int) -> str:
    """Keep only citations that point to existing source indices.

    Supported citation format: [1], [1, 2], [2,3,4].
    Invalid indices are dropped; empty citation blocks are removed.
    """
    if total_sources <= 0:
        return answer

    def replace_block(match: re.Match) -> str:
        raw_block = match.group(1)
        values = re.findall(r"\d+", raw_block)

        valid = []
        seen = set()
        for value in values:
            index = int(value)
            if 1 <= index <= total_sources and index not in seen:
                seen.add(index)
                valid.append(index)

        if not valid:
            return ""

        joined = ", ".join(str(index) for index in valid)
        return f"[{joined}]"

    fixed = _CITATION_BLOCK_RE.sub(replace_block, answer)
    fixed = re.sub(r"\s{2,}", " ", fixed)
    fixed = re.sub(r"\s+([,.;:!?])", r"\1", fixed)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip()


async def generate_answer(question: str, sources: SourceResults, model: str = "llama3") -> str:
    """Generate an answer from Ollama and sanitize citation references.

    Args:
        question: User prompt text.
        sources: Retrieved evidence records already scored/ranked upstream.
        model: Ollama model name.

    Returns:
        Model answer with citation blocks normalized to existing source indices.
    """
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_user_message(question, sources)},
    ]

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(OLLAMA_CHAT_URL, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for factual accuracy
            },
        })
        resp.raise_for_status()
        raw_answer = resp.json()["message"]["content"]
        # Prevent dangling citations such as [99] when only a few sources exist.
        return _verify_and_fix_citations(raw_answer, total_sources=len(sources))


async def list_models() -> List[str]:
    """Return a list of locally available Ollama model names."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        return []
