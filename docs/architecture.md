# Atlas Architecture and Behavior Reference

This document describes how Atlas processes a question from input to output, and captures the key contracts used across backend, retrieval, and UI.

## Components

- `main.py`: FastAPI routes, request validation, evidence gate, audit logging.
- `retrieval.py`: source registry, source-limit normalization, confidence scoring and ranking.
- `sources/*.py`: source-specific retrieval adapters.
- `llm.py`: constrained evidence prompt to Ollama plus citation cleanup.
- `static/index.html`: web UI and source/policy controls.

## End-to-End Flow

1. Client sends `POST /chat`.
2. Request is validated (`question`, `model`, and optional source limits).
3. Source limits are normalized and selected sources are resolved.
4. Retrieval runs across enabled sources.
5. Results are confidence-scored and ranked.
6. Evidence gate evaluates quality and contradiction heuristics.
7. If gate passes, LLM generation runs with numbered evidence context.
8. Citation indices are verified and out-of-range references are removed.
9. Response is returned with `answer` and retrieved `sources`.

## Source Record Contract

Each retriever should return records with this shape:

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

Optional fields added by ranking pipeline:

```python
{
    "confidence_score": float,
    "confidence_level": "low" | "medium" | "high",
}
```

## Evidence Gate Behavior

Gate checks use configurable thresholds and may return `INSUFFICIENT_EVIDENCE` when:

- Too few total sources are retrieved.
- Too few medium-or-higher confidence sources are present.
- Average confidence is below configured minimum.
- Contradiction heuristics detect conflicting high-confidence claims.

When gated, Atlas returns a structured text block with reason, summary counts, and actionable next steps.

## Citation Verification Behavior

After model generation, citation blocks are sanitized so references map to existing source indices.

Supported forms:

- `[1]`
- `[1, 2]`
- `[2,3,4]`

Invalid indices are dropped. Empty citation blocks are removed.

## Audit Logging

Atlas emits JSON audit logs through `atlas.audit` logger.

Controlled by `ATLAS_AUDIT_LOG_MODE`:

- `full`: complete payloads
- `basic`: reduced payload detail (counts/lengths)
- `off`: logging disabled

Typical event sequence:

1. `chat_request`
2. `retrieval_result`
3. `chat_gated` or `chat_answer`

Failure-related events:

- `chat_no_sources`
- `chat_no_sources_error`

## API Notes

- `GET /sources` exposes source definitions and active policy values (confidence thresholds, evidence gate settings, audit mode).
- `POST /chat` returns HTTP `502` when retrieval fails with source errors.
- `POST /chat` returns HTTP `503` when Ollama is unreachable/fails after retrieval.

## Extension Guide (Short)

To add a new source:

1. Create `sources/new_source.py` with `async search_new_source(query, max_results)`.
2. Return records matching the source record contract.
3. Export from `sources/__init__.py`.
4. Register in `retrieval.py` source registry.

After registration, the source appears in `/sources`, frontend toggles, and `/chat` retrieval.
