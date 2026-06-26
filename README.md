# Atlas

Atlas is an evidence-first scientific Q&A chatbot. It retrieves sources first, ranks evidence, applies evidence gating, and only then calls a local Ollama model.

## What Atlas Does

- Retrieves from verified/scientific sources (currently PubMed, arXiv, Ptable).
- Ranks retrieved records by confidence before generation.
- Applies evidence gating: if evidence is too weak or contradictory, returns `INSUFFICIENT_EVIDENCE` instead of a speculative answer.
- Verifies/fixes citation indices in model output so references map to real retrieved sources.
- Emits JSON audit logs for debugging and traceability.

## Quick Start (Windows PowerShell)

1. Create and activate virtual environment.
2. Install dependencies.
3. Pull an Ollama model.
4. Run the app.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3
python -m uvicorn main:app --reload
```

Open: http://127.0.0.1:8000

## Project Layout

- `main.py`: API routes, evidence gate, audit logging
- `retrieval.py`: source registry, source limits, confidence ranking
- `llm.py`: Ollama call + citation verification post-processing
- `sources/pubmed.py`: PubMed retriever
- `sources/arxiv.py`: arXiv retriever
- `sources/ptable.py`: Ptable retriever + name-to-formula behavior
- `static/index.html`: frontend UI

## Request Lifecycle

1. `/chat` receives question and source configuration.
2. Retrieval runs for enabled sources.
3. Sources are confidence-scored and sorted.
4. Evidence gate checks quality/contradiction.
5. If gate passes, LLM generates answer.
6. Citation indices are validated against retrieved sources.

## API Endpoints

- `GET /` UI
- `GET /health` health check
- `GET /models` available local Ollama models
- `GET /sources` source metadata + active confidence/gate/audit policy
- `POST /chat` retrieve + gate + generate

## Documentation Map

- `README.md` (this file): setup, runtime behavior, configuration
- `docs/architecture.md`: component architecture, data contracts, audit events

### `/chat` Example

```json
{
    "question": "What is the molecular weight of sodium carbonate?",
    "model": "llama3",
    "source_limits": {
        "pubmed": 4,
        "arxiv": 2,
        "ptable": 2
    }
}
```

### `/chat` Response Modes

Atlas can return one of three practical response modes:

1. Normal answer
   - Evidence gate passes.
   - `answer` contains generated text with citations.
   - `sources` contains retrieved records used for context.

2. Gated response
   - Evidence gate fails due to weak or contradictory evidence.
   - `answer` begins with `INSUFFICIENT_EVIDENCE` and includes a reason + next steps.
   - `sources` still contains retrieved records for transparency.

3. No-source response
   - No records retrieved and no upstream source error.
   - `answer` explains that no relevant literature was found.
   - `sources` is an empty list.

Error behavior:
- If all retrieval fails with source errors, `/chat` returns HTTP `502`.
- If Ollama generation fails after successful retrieval, `/chat` returns HTTP `503`.

### `/sources` Example

```json
{
    "sources": [
        {"key": "pubmed", "label": "PubMed", "default_max_results": 5},
        {"key": "arxiv", "label": "arXiv", "default_max_results": 3},
        {"key": "ptable", "label": "Ptable", "default_max_results": 2}
    ],
    "confidence_thresholds": {
        "high": 0.8,
        "medium": 0.6
    },
    "evidence_gate": {
        "min_total_sources": 2,
        "min_medium_or_higher_sources": 1,
        "min_average_confidence": 0.6,
        "contradiction_min_score": 0.6
    },
    "audit_logging": {
        "mode": "full"
    }
}
```

## Configuration (Environment Variables)

### Confidence Thresholds

- `ATLAS_CONFIDENCE_HIGH` (default: `0.80`)
- `ATLAS_CONFIDENCE_MEDIUM` (default: `0.60`)

Rules:
- Values are clamped to `0.0..1.0`.
- If medium > high, medium is set to high.

### Evidence Gate

- `ATLAS_GATE_MIN_TOTAL_SOURCES` (default: `2`)
- `ATLAS_GATE_MIN_MEDIUM_OR_HIGHER_SOURCES` (default: `1`)
- `ATLAS_GATE_MIN_AVERAGE_CONFIDENCE` (default: current medium threshold)
- `ATLAS_GATE_CONTRADICTION_MIN_SCORE` (default: current medium threshold)

Rules:
- Float values are clamped to `0.0..1.0`.
- Integer values are clamped to `>= 0`.

### Audit Logging

- `ATLAS_AUDIT_LOG_MODE` values:
    - `full` (default): full payloads including question text and chosen references
    - `basic`: reduced payload detail (question length, counts)
    - `off`: no audit events

### Example Config

```powershell
$env:ATLAS_CONFIDENCE_HIGH="0.85"
$env:ATLAS_CONFIDENCE_MEDIUM="0.65"
$env:ATLAS_GATE_MIN_TOTAL_SOURCES="3"
$env:ATLAS_GATE_MIN_MEDIUM_OR_HIGHER_SOURCES="2"
$env:ATLAS_GATE_MIN_AVERAGE_CONFIDENCE="0.70"
$env:ATLAS_GATE_CONTRADICTION_MIN_SCORE="0.65"
$env:ATLAS_AUDIT_LOG_MODE="basic"
python -m uvicorn main:app --reload
```

## Extending Sources

1. Add a module under `sources/` (for example `sources/crossref.py`).
2. Implement `async def search_crossref(query: str, max_results: int) -> list[dict]`.
3. Return source items with this shape:

```python
{
    "title": str,
    "abstract": str,
    "authors": list[str],
    "journal": str,
    "year": str,
    "url": str,
    "doi": str,
    "source": str,
}
```

4. Export from `sources/__init__.py`.
5. Register in `retrieval.py` `SOURCE_DEFINITIONS`.

Once registered, it appears automatically in `/sources`, the UI source toggles, and the retrieval pipeline.

## Troubleshooting

- If `uvicorn` is not recognized:

```powershell
python -m uvicorn main:app --reload
```

- If model requests fail:

```powershell
ollama list
```

- If responses are gated often, increase source limits or relax gate thresholds.
- If you need deeper operational detail, see `docs/architecture.md`.
