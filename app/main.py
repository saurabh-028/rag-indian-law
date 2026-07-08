"""
main.py — FastAPI application for the Indian Law RAG system.

Endpoints:
    POST /query       — Retrieve + GPT-4 answer
    POST /retrieve    — Raw retrieval (no LLM)
    GET  /sectors     — List indexed sectors
    GET  /health      — System health check

Environment variables:
    OPENAI_API_KEY   — Required for /query
    INDEX_DIR        — Path to unified.index + metadata.json (default: ./index)
    EMBED_MODEL      — SentenceTransformer model name (default: BAAI/bge-base-en)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env from project root (two levels up from app/main.py)
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

from app import retriever  # noqa: E402
from app.generator import Generator  # noqa: E402
from app.index_loader import download_index_from_s3  # noqa: E402
from app.language import detect_language, translator, SUPPORTED_LANGS  # noqa: E402

_generator = Generator()
_STATIC = Path(__file__).parent / "static"
_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and embedding model at startup."""
    global _startup_time
    _startup_time = time.time()
    index_dir = os.getenv("INDEX_DIR", "./index")
    model_name = os.getenv("EMBED_MODEL", "BAAI/bge-base-en")
    try:
        download_index_from_s3(index_dir)
        retriever.load(index_dir=index_dir, model_name=model_name)
    except (FileNotFoundError, ValueError) as exc:
        # Allow the server to start without an index so /health can respond
        logger.warning("Index not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title="Indian Law RAG API",
    description=(
        "Retrieval-Augmented Generation system for Indian legal documents. "
        "Covers traffic law, criminal law (BNS), and rental law."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Legal question to answer.")
    top_k: int = Field(10, ge=1, le=20, description="Number of chunks to retrieve.")
    sector_filter: Optional[str] = Field(
        None,
        description="Filter results to a specific sector (e.g. 'traffic', 'criminal_law', 'rental_law').",
    )
    lang: Optional[str] = Field(
        None,
        description="Override detected language (e.g. 'hi', 'en'). Auto-detected if not provided.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    sector_used: Optional[str]
    detected_lang: str = Field("en", description="Detected or overridden language of the query.")
    usage: dict


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(10, ge=1, le=20)
    sector_filter: Optional[str] = None
    doc_type_filter: Optional[str] = Field(
        None,
        description="Filter to a specific doc_type (e.g. 'fine_schedule', 'legislation', 'actionable_procedure').",
    )


class RetrieveResponse(BaseModel):
    chunks: list[dict]


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    model: str
    index_built_at: str
    index_source: str
    uptime_seconds: float
    chunks_by_sector: dict


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(_STATIC / "index.html")


@app.get("/chat", include_in_schema=False)
async def serve_chat():
    return FileResponse(_STATIC / "chat.html")


@app.post("/query", response_model=QueryResponse, summary="Ask a legal question (RAG)")
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Retrieve relevant legal chunks and generate an answer via GPT-4."""
    if not retriever.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run pipeline/build_index.py and restart the server.",
        )

    detected_lang = req.lang or detect_language(req.question)
    logger.info("query | lang=%s sector=%s q=%r", detected_lang, req.sector_filter or "auto", req.question[:120])

    # Translate to English for retrieval — the index is English-only.
    # GPT-4o reads the original question directly, so no back-translation is needed.
    retrieval_question = req.question
    if detected_lang in SUPPORTED_LANGS:
        try:
            retrieval_question = translator.to_english(req.question, src_lang=detected_lang)
        except Exception as exc:
            logger.warning("Translation failed (%s->en): %s — using original question.", detected_lang, exc)

    chunks = retriever.retrieve(
        question=retrieval_question,
        top_k=req.top_k,
        sector_filter=req.sector_filter,
    )

    if not chunks:
        no_info_msgs = {
            "hi": "मुझे लोड किए गए दस्तावेज़ों में पर्याप्त जानकारी नहीं मिली।",
            "mr": "लोड केलेल्या दस्तावेजांमध्ये पुरेशी माहिती आढळली नाही.",
        }
        no_info_msg = no_info_msgs.get(detected_lang, "I don't have enough information in the loaded documents.")
        return QueryResponse(
            answer=no_info_msg,
            sources=[],
            chunks_used=0,
            sector_used=None,
            detected_lang=detected_lang,
            usage={},
        )

    # Low-confidence guard: if even the top chunk has a very low RRF score,
    # the query is likely out-of-domain — return a fallback rather than hallucinating.
    if chunks[0].get("score", 1.0) < 0.008:
        fallback_msgs = {
            "hi": "मुझे लोड किए गए कानूनी दस्तावेज़ों में इस विषय पर प्रासंगिक जानकारी नहीं मिली। कृपया किसी योग्य वकील से परामर्श लें।",
            "mr": "लोड केलेल्या कायदेशीर दस्तावेजांमध्ये या विषयावर संबंधित माहिती आढळली नाही. कृपया एखाद्या पात्र वकिलाचा सल्ला घ्या.",
        }
        fallback = fallback_msgs.get(
            detected_lang,
            "I could not find relevant information on this topic in the loaded legal documents. Please consult a qualified lawyer.",
        )
        return QueryResponse(
            answer=fallback,
            sources=[],
            chunks_used=0,
            sector_used=None,
            detected_lang=detected_lang,
            usage={},
        )

    # Auto-detect sector via majority vote across retrieved chunks
    if req.sector_filter:
        sector = req.sector_filter
    else:
        sector_votes: dict = {}
        for c in chunks:
            s = c.get("sector", "")
            if s:
                sector_votes[s] = sector_votes.get(s, 0) + 1
        sector = max(sector_votes, key=sector_votes.get) if sector_votes else chunks[0].get("sector")

    # Pass the original question so GPT responds in the user's language
    try:
        result = _generator.generate(
            question=req.question,
            context_chunks=chunks,
            sector=sector,
            response_lang=detected_lang,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    answer = result["answer"]
    logger.info("answer | sector=%s chunks=%d tokens=%d", sector, len(chunks), result["usage"].get("total_tokens", 0))

    sources = list(
        dict.fromkeys(
            f"{c.get('source', 'unknown')} § {c.get('section_number', '?')}"
            for c in chunks
        )
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(chunks),
        sector_used=sector,
        detected_lang=detected_lang,
        usage=result["usage"],
    )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve raw legal chunks (no LLM)",
)
async def retrieve_endpoint(req: RetrieveRequest) -> RetrieveResponse:
    """Return the top-k most relevant legal chunks without calling an LLM."""
    if not retriever.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run pipeline/build_index.py and restart the server.",
        )

    chunks = retriever.retrieve(
        question=req.question,
        top_k=req.top_k,
        sector_filter=req.sector_filter,
        doc_type_filter=req.doc_type_filter,
    )
    return RetrieveResponse(chunks=chunks)


@app.get("/sectors", summary="List available sectors")
async def sectors_endpoint() -> dict:
    """Return the list of unique sectors present in the loaded index."""
    if not retriever.is_loaded():
        raise HTTPException(status_code=503, detail="Index not loaded.")
    return {"sectors": retriever.unique_sectors()}


@app.get("/health", response_model=HealthResponse, summary="System health check")
async def health_endpoint() -> HealthResponse:
    """Returns deployment info: index version, source, uptime, and chunk breakdown."""
    info = retriever.index_info()
    return HealthResponse(
        status="ok" if retriever.is_loaded() else "no_index",
        vector_count=retriever.vector_count(),
        model=retriever.current_model(),
        index_built_at=info.get("built_at", "unknown"),
        index_source=info.get("index_source", "unknown"),
        uptime_seconds=round(time.time() - _startup_time, 1),
        chunks_by_sector=info.get("chunks_by_sector", {}),
    )
