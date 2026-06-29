from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import json
import logging
import os
import re
from typing import Dict, List
from sources import SourceRecord, SourceResults

from retrieval import (
    get_confidence_thresholds,
    get_default_source_limits,
    get_source_definitions,
    normalize_source_limits,
    search_all_sources,
)
from llm import generate_answer, list_models

"""Atlas API orchestration layer.

This module coordinates retrieval, evidence gating, answer generation, and audit
logging for the scientific chatbot endpoints.
"""

app = FastAPI(title="Scientific Chatbot", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

AUDIT_LOGGER = logging.getLogger("atlas.audit")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_CITATION_BLOCK_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

CONTRADICTION_PAIRS = [
    ("effective", "ineffective"),
    ("benefit", "no significant"),
    ("improves", "worsens"),
    ("safe", "toxic"),
    ("protective", "harmful"),
]


class ChatRequest(BaseModel):
    """Input payload for /chat with optional per-source result overrides."""

    question: str = Field(..., min_length=3, max_length=1000)
    model: str = Field(default="llama3")
    source_limits: Dict[str, int] = Field(default_factory=get_default_source_limits)
    pubmed_results: int | None = Field(default=None, ge=0, le=10)
    arxiv_results: int | None = Field(default=None, ge=0, le=10)
    ptable_results: int | None = Field(default=None, ge=0, le=10)


class Source(BaseModel):
    """Normalized source item returned to clients in /chat responses."""

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
    """Response payload containing the generated answer and retrieved sources."""

    answer: str
    sources: List[Source]


def get_audit_log_mode() -> str:
    """Return audit verbosity mode from env, defaulting invalid values to full."""

    mode = os.getenv("ATLAS_AUDIT_LOG_MODE", "full").strip().lower()
    if mode in {"off", "basic", "full"}:
        return mode
    return "full"


def _to_basic_audit_payload(payload: Dict) -> Dict:
    """Downsample sensitive/large audit fields when mode is basic."""

    basic = {}
    for key, value in payload.items():
        if key == "question":
            basic["question_length"] = len(str(value))
        elif key == "chosen_references":
            basic["chosen_reference_count"] = len(value) if isinstance(value, list) else 0
        elif key == "source_errors":
            basic["source_error_count"] = len(value) if isinstance(value, list) else 0
        elif key == "citation_indices":
            basic["citation_count"] = len(value) if isinstance(value, list) else 0
        else:
            basic[key] = value
    return basic


def _audit_log(event: str, **fields) -> None:
    """Emit structured JSON audit events honoring configured verbosity mode."""

    mode = get_audit_log_mode()
    if mode == "off":
        return

    payload = {
        "event": event,
        "component": "chat",
        **fields,
    }

    if mode == "basic":
        payload = _to_basic_audit_payload(payload)

    AUDIT_LOGGER.info(json.dumps(payload, ensure_ascii=True, default=str))


def _selected_source_keys(source_limits: Dict[str, int]) -> List[str]:
    """Return sorted source keys enabled by positive max-result limits."""

    keys = []
    for key, value in source_limits.items():
        try:
            enabled = int(value) > 0
        except (TypeError, ValueError):
            enabled = False
        if enabled:
            keys.append(key)
    return sorted(keys)


def _source_counts(sources: SourceResults) -> Dict[str, int]:
    """Count retrieved records grouped by source name."""

    counts: Dict[str, int] = {}
    for source in sources:
        name = str(source.get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def _extract_citation_indices(answer: str, max_index: int) -> List[int]:
    """Extract unique, in-range citation indices from answer text."""

    indices = []
    seen = set()
    for block in _CITATION_BLOCK_RE.findall(answer):
        for token in re.findall(r"\d+", block):
            idx = int(token)
            if 1 <= idx <= max_index and idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def _chosen_references(citation_indices: List[int], sources: SourceResults) -> List[Dict]:
    """Resolve cited indices to lightweight source reference metadata."""

    references = []
    for idx in citation_indices:
        if not (1 <= idx <= len(sources)):
            continue
        source = sources[idx - 1]
        references.append({
            "index": idx,
            "source": source.get("source", ""),
            "title": source.get("title", ""),
            "url": source.get("url", ""),
        })
    return references


def _gate_reason(gated_response: str) -> str:
    """Parse reason text from a structured gated response payload."""

    for line in gated_response.splitlines():
        if line.lower().startswith("reason:"):
            return line.split(":", 1)[1].strip()
    return "insufficient evidence"


def _score(source: SourceRecord) -> float:
    """Read and clamp source confidence score into the [0.0, 1.0] interval."""

    value = source.get("confidence_score", 0.0)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _detect_contradiction(sources: SourceResults, min_score: float) -> bool:
    """Detect simple contradictory claim pairs across stronger sources."""

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
    """Create a deterministic insufficient-evidence message for clients."""

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


def _int_env(name: str, default: int) -> int:
    """Read non-negative integer env vars with safe fallback."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


def _float_env(name: str, default: float) -> float:
    """Read and clamp float env vars to [0.0, 1.0] with safe fallback."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


def get_evidence_gate_config() -> Dict[str, float | int]:
    """Return effective evidence-gate policy using env overrides and defaults."""

    confidence = get_confidence_thresholds()
    medium_default = float(confidence.get("medium", 0.6))

    min_total_sources = _int_env("ATLAS_GATE_MIN_TOTAL_SOURCES", 2)
    min_medium_or_higher_sources = _int_env("ATLAS_GATE_MIN_MEDIUM_OR_HIGHER_SOURCES", 1)
    min_average_confidence = _float_env("ATLAS_GATE_MIN_AVERAGE_CONFIDENCE", medium_default)
    contradiction_min_score = _float_env("ATLAS_GATE_CONTRADICTION_MIN_SCORE", medium_default)

    return {
        "min_total_sources": min_total_sources,
        "min_medium_or_higher_sources": min_medium_or_higher_sources,
        "min_average_confidence": min_average_confidence,
        "contradiction_min_score": contradiction_min_score,
    }


def _evaluate_evidence_gate(sources: SourceResults) -> str | None:
    """Return a gated response string when evidence is insufficient, else None."""

    gate_config = get_evidence_gate_config()
    thresholds = get_confidence_thresholds()
    high_threshold = float(thresholds.get("high", 0.8))
    medium_threshold = float(thresholds.get("medium", 0.6))

    scored = [_score(source) for source in sources]
    total = len(scored)
    high_count = sum(1 for score in scored if score >= high_threshold)
    medium_or_higher_count = sum(1 for score in scored if score >= medium_threshold)
    average_score = (sum(scored) / total) if total else 0.0

    contradiction = _detect_contradiction(
        sources,
        min_score=float(gate_config["contradiction_min_score"]),
    )
    weak_evidence = (
        total < int(gate_config["min_total_sources"])
        or medium_or_higher_count < int(gate_config["min_medium_or_higher_sources"])
        or average_score < float(gate_config["min_average_confidence"])
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
    """Serve the single-page frontend."""

    return FileResponse("static/index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle scientific Q&A: retrieve, gate, generate, and audit the request."""

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
    selected_sources = _selected_source_keys(source_limits)

    _audit_log(
        "chat_request",
        question=question,
        model=request.model,
        selected_sources=selected_sources,
        source_limits=source_limits,
    )

    sources, source_errors = await search_all_sources(question, source_limits)

    _audit_log(
        "retrieval_result",
        question=question,
        selected_sources=selected_sources,
        source_limits=source_limits,
        total_sources=len(sources),
        retrieved_source_counts=_source_counts(sources),
        source_errors=source_errors,
    )

    if not sources:
        if source_errors:
            _audit_log(
                "chat_no_sources_error",
                question=question,
                selected_sources=selected_sources,
                source_limits=source_limits,
                source_errors=source_errors,
            )
            raise HTTPException(
                status_code=502,
                detail="; ".join(source_errors),
            )
        _audit_log(
            "chat_no_sources",
            question=question,
            selected_sources=selected_sources,
            source_limits=source_limits,
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
        _audit_log(
            "chat_gated",
            question=question,
            selected_sources=selected_sources,
            source_limits=source_limits,
            gate_reason=_gate_reason(gated_response),
            chosen_references=[],
        )
        return ChatResponse(answer=gated_response, sources=sources)

    try:
        answer = await generate_answer(question, sources, model=request.model)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach Ollama. Is it running? ({e})",
        )

    citation_indices = _extract_citation_indices(answer, max_index=len(sources))
    chosen_references = _chosen_references(citation_indices, sources)
    _audit_log(
        "chat_answer",
        question=question,
        selected_sources=selected_sources,
        source_limits=source_limits,
        citation_indices=citation_indices,
        chosen_references=chosen_references,
    )

    return ChatResponse(answer=answer, sources=sources)


@app.get("/sources")
async def get_sources():
    """Expose source catalog and active policy configuration for the frontend."""

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
        "evidence_gate": get_evidence_gate_config(),
        "audit_logging": {
            "mode": get_audit_log_mode(),
        },
    }


@app.get("/models")
async def get_models():
    """Return the list of locally available Ollama models."""
    models = await list_models()
    return {"models": models}


@app.get("/health")
async def health():
    """Simple health endpoint for readiness/liveness checks."""

    return {"status": "ok"}
