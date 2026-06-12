from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List

from retrieval import (
    get_default_source_limits,
    get_source_definitions,
    normalize_source_limits,
    search_all_sources,
)
from llm import generate_answer, list_models

app = FastAPI(title="Scientific Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


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


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


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
        ]
    }


@app.get("/models")
async def get_models():
    """Return the list of locally available Ollama models."""
    models = await list_models()
    return {"models": models}


@app.get("/health")
async def health():
    return {"status": "ok"}
