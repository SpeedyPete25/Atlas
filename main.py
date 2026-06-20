from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List

from retrieval import (
    get_confidence_thresholds,
    get_default_source_limits,
    get_source_definitions,
    normalize_source_limits,
    search_all_sources,
)
from llm import generate_answer, list_models

app = FastAPI(title="Scientific Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

CONTRADICTION_PAIRS = [
    ("effective", "ineffective"),
    ("benefit", "no significant"),
    ("improves", "worsens"),
    ("safe", "toxic"),
    ("protective", "harmful"),
]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    model: str = Field(default="llama3")
    source_limits: Dict[str, int] = Field(default_factory=get_default_source_limits)
    pubmed_results: int | None = Field(default=None, ge=0, le=10)
    arxiv_results: int | None = Field(default=None, ge=0, le=10)
    ptable_results: int | None = Field(default=None, ge=0, le=10)


class Source(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    url: str
    doi: str
    source: str
    confidence_score: float | None = None
    confidence_level: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


def _score(source: Dict) -> float:
    value = source.get("confidence_score", 0.0)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _detect_contradiction(sources: List[Dict], min_score: float) -> bool:
    strong_texts = []
    for source in sources:
        if _score(source) < min_score:
            continue
        text = f"{source.get('title', '')} {source.get('abstract', '')}".lower()
        strong_texts.append(text)

    if len(strong_texts) < 2:
        return False

    for positive, negative in CONTRADICTION_PAIRS:
        has_positive = any(positive in text for text in strong_texts)
        has_negative = any(negative in text for text in strong_texts)
        if has_positive and has_negative:
            return True

    return False


def _build_insufficient_evidence_response(
    reason: str,
    total_sources: int,
    high_count: int,
    medium_or_higher_count: int,
) -> str:
    return (
        "INSUFFICIENT_EVIDENCE\n"
        f"reason: {reason}\n"
        "summary:\n"
        f"- total_sources: {total_sources}\n"
        f"- high_confidence_sources: {high_count}\n"
        f"- medium_or_higher_sources: {medium_or_higher_count}\n"
        "next_steps:\n"
        "- refine your question with more specific terms (population, intervention, outcome)\n"
        "- enable more sources or increase per-source limits\n"
        "- ask for a narrower claim that can be verified with current evidence"
    )


def _evaluate_evidence_gate(sources: List[Dict]) -> str | None:
    thresholds = get_confidence_thresholds()
    high_threshold = float(thresholds.get("high", 0.8))
    medium_threshold = float(thresholds.get("medium", 0.6))

    scored = [_score(source) for source in sources]
    total = len(scored)
    high_count = sum(1 for score in scored if score >= high_threshold)
    medium_or_higher_count = sum(1 for score in scored if score >= medium_threshold)
    average_score = (sum(scored) / total) if total else 0.0

    contradiction = _detect_contradiction(sources, min_score=medium_threshold)
    weak_evidence = (
        total < 2
        or medium_or_higher_count == 0
        or average_score < medium_threshold
    )

    if contradiction:
        return _build_insufficient_evidence_response(
            reason="retrieved sources contain contradictory claims",
            total_sources=total,
            high_count=high_count,
            medium_or_higher_count=medium_or_higher_count,
        )

    if weak_evidence:
        return _build_insufficient_evidence_response(
            reason="retrieved evidence quality is too weak for a reliable answer",
            total_sources=total,
            high_count=high_count,
            medium_or_higher_count=medium_or_higher_count,
        )

    return None


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    legacy_overrides = {
        key: value
        for key, value in {
            "pubmed": request.pubmed_results,
            "arxiv": request.arxiv_results,
            "ptable": request.ptable_results,
        }.items()
        if value is not None
    }
    source_limits = normalize_source_limits({
        **request.source_limits,
        **legacy_overrides,
    })
    sources, source_errors = await search_all_sources(question, source_limits)

    if not sources:
        if source_errors:
            raise HTTPException(
                status_code=502,
                detail="; ".join(source_errors),
            )
        return ChatResponse(
            answer=(
                "No relevant scientific literature was found for your question. "
                "Please try rephrasing or using different keywords."
            ),
            sources=[],
        )

    gated_response = _evaluate_evidence_gate(sources)
    if gated_response is not None:
        return ChatResponse(answer=gated_response, sources=sources)

    try:
        answer = await generate_answer(question, sources, model=request.model)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ollama. Is it running? ({e})",
        )

    return ChatResponse(answer=answer, sources=sources)


@app.get("/sources")
async def get_sources():
    definitions = get_source_definitions()
    return {
        "sources": [
            {
                "key": source.key,
                "label": source.label,
                "default_max_results": source.default_max_results,
            }
            for source in definitions.values()
        ],
        "confidence_thresholds": get_confidence_thresholds(),
    }


@app.get("/models")
async def get_models():
    """Return the list of locally available Ollama models."""
    models = await list_models()
    return {"models": models}


@app.get("/health")
async def health():
    return {"status": "ok"}
