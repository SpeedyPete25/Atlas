import httpx
from typing import List, Dict

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _build_system_prompt() -> str:
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


def _build_user_message(question: str, sources: List[Dict]) -> str:
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


async def generate_answer(question: str, sources: List[Dict], model: str = "llama3") -> str:
    """Send context + question to Ollama and return the model's response."""
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
        return resp.json()["message"]["content"]


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
