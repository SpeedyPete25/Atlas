# Atlas

Atlas is a scientific Q&A chatbot that answers using retrieved evidence from verified sources before sending context to a local LLM (Ollama).

Current sources include:
- PubMed
- arXiv
- Ptable

The retrieval pipeline now supports source confidence scoring and source toggles from the frontend.

## Features

- Evidence-first workflow: retrieves sources before generation.
- Local model support via Ollama.
- Dynamic source controls in the web UI (loaded from the backend).
- Modular source architecture under the sources package.
- Source confidence scoring and ranking before answer generation.

## Tech Stack

- Backend: FastAPI
- Frontend: Vanilla HTML/CSS/JS
- LLM runtime: Ollama
- HTTP client: httpx

## Project Structure

- main.py: FastAPI app and API routes
- llm.py: Ollama chat integration
- retrieval.py: source registry, limits, scoring, ranking
- sources/pubmed.py: PubMed retrieval
- sources/arxiv.py: arXiv retrieval
- sources/ptable.py: Ptable retrieval + name-to-formula support
- static/index.html: web UI

## Prerequisites

- Python 3.11+
- Ollama installed
- At least one Ollama model pulled (example: llama3)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Pull an Ollama model.

Example on Windows PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ollama pull llama3

## Run

Start the API server:

    python -m uvicorn main:app --reload

Open in browser:

    http://127.0.0.1:8000

## API Overview

- GET /: web app
- GET /health: health check
- GET /models: local Ollama models
- GET /sources: available retrieval sources and defaults
- POST /chat: retrieves evidence and generates an answer

## POST /chat Payload

Minimal payload:

    {
      "question": "What is the molecular weight of sodium carbonate?",
      "model": "llama3"
    }

With source controls:

    {
      "question": "What is the molecular weight of sodium carbonate?",
      "model": "llama3",
      "source_limits": {
        "pubmed": 4,
        "arxiv": 2,
        "ptable": 2
      }
    }

## Confidence Scoring

Retrieved items are ranked before generation using:
- Base trust by source type
- Recency from publication year
- Metadata completeness (abstract/authors/DOI/URL/journal)
- Priority journal bonus

This ranking improves evidence quality passed to the LLM.

### Confidence Threshold Configuration

You can configure confidence chip/ranking thresholds via environment variables:

- `ATLAS_CONFIDENCE_HIGH`: score threshold for `high` confidence (default: `0.80`)
- `ATLAS_CONFIDENCE_MEDIUM`: score threshold for `medium` confidence (default: `0.60`)

Notes:
- Values are clamped to the range `0.0` to `1.0`.
- If `ATLAS_CONFIDENCE_MEDIUM` is greater than `ATLAS_CONFIDENCE_HIGH`, medium is automatically set equal to high.

Example (Windows PowerShell):

    $env:ATLAS_CONFIDENCE_HIGH="0.85"
    $env:ATLAS_CONFIDENCE_MEDIUM="0.65"
    python -m uvicorn main:app --reload

### Evidence Gate Configuration

Evidence gating decides whether Atlas should answer normally or return an `INSUFFICIENT_EVIDENCE` response.

Configure gate strictness via environment variables:

- `ATLAS_GATE_MIN_TOTAL_SOURCES`: minimum number of retrieved sources required (default: `2`)
- `ATLAS_GATE_MIN_MEDIUM_OR_HIGHER_SOURCES`: minimum count of sources at or above medium confidence (default: `1`)
- `ATLAS_GATE_MIN_AVERAGE_CONFIDENCE`: minimum average confidence score across retrieved sources (default: current medium threshold)
- `ATLAS_GATE_CONTRADICTION_MIN_SCORE`: minimum confidence score for sources considered in contradiction checks (default: current medium threshold)

Notes:
- Confidence values are clamped to `0.0` to `1.0`.
- Integer values are clamped to `>= 0`.

Example (Windows PowerShell):

    $env:ATLAS_GATE_MIN_TOTAL_SOURCES="3"
    $env:ATLAS_GATE_MIN_MEDIUM_OR_HIGHER_SOURCES="2"
    $env:ATLAS_GATE_MIN_AVERAGE_CONFIDENCE="0.7"
    $env:ATLAS_GATE_CONTRADICTION_MIN_SCORE="0.65"
    python -m uvicorn main:app --reload

## Adding a New Verified Source

1. Create a new module in sources, for example sources/crossref.py.
2. Implement async search function:

    async def search_crossref(query: str, max_results: int) -> list[dict]:
        ...

3. Return list items with this exact shape:

    {
      "title": str,
      "abstract": str,
      "authors": list[str],
      "journal": str,
      "year": str,
      "url": str,
      "doi": str,
      "source": str
    }

4. Export it in sources/__init__.py.
5. Register it in retrieval.py SOURCE_DEFINITIONS with key, label, default_max_results, search.

Once registered, the source is automatically available to:
- GET /sources
- Frontend source toggles
- POST /chat retrieval pipeline

## Troubleshooting

If uvicorn is not recognized, run:

    python -m uvicorn main:app --reload

If model calls fail, verify Ollama is running and has a model:

    ollama list

If no answers are returned, check selected source limits in the UI and retry with a clearer query.
