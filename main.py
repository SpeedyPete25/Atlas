import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List

from retrieval import search_pubmed, search_arxiv
from llm import generate_answer, list_models

app = FastAPI(title="Scientific Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    model: str = Field(default="llama3")
    pubmed_results: int = Field(default=4, ge=1, le=10)
    arxiv_results: int = Field(default=3, ge=0, le=10)


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

    # Retrieve from both sources in parallel
    pubmed_task = search_pubmed(question, max_results=request.pubmed_results)
    arxiv_task = search_arxiv(question, max_results=request.arxiv_results)
    pubmed_results, arxiv_results = await asyncio.gather(pubmed_task, arxiv_task)

    sources = pubmed_results + arxiv_results

    if not sources:
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


@app.get("/models")
async def get_models():
    """Return the list of locally available Ollama models."""
    models = await list_models()
    return {"models": models}


@app.get("/health")
async def health():
    return {"status": "ok"}
