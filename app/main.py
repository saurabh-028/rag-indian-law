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

import os
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

from app import retriever  # noqa: E402
from app.generator import Generator  # noqa: E402
from app.index_loader import download_index_from_s3  # noqa: E402
from app.language import detect_language, translator, SUPPORTED_LANGS  # noqa: E402

_generator = Generator()
_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and embedding model at startup."""
    index_dir = os.getenv("INDEX_DIR", "./index")
    model_name = os.getenv("EMBED_MODEL", "BAAI/bge-base-en")
    try:
        download_index_from_s3(index_dir)
        retriever.load(index_dir=index_dir, model_name=model_name)
    except (FileNotFoundError, ValueError) as exc:
        # Allow the server to start without an index so /health can respond
        print(f"[startup] WARNING: {exc}")
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
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve.")
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
    top_k: int = Field(3, ge=1, le=20)
    sector_filter: Optional[str] = None


class RetrieveResponse(BaseModel):
    chunks: list[dict]


class HealthResponse(BaseModel):
    status: str
    vector_count: int
    model: str


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

    # Translate to English for retrieval — the index is English-only.
    # GPT-4o reads the original question directly, so no back-translation is needed.
    retrieval_question = req.question
    if detected_lang in SUPPORTED_LANGS:
        try:
            retrieval_question = translator.to_english(req.question, src_lang=detected_lang)
        except Exception as exc:
            print(f"[query] Translation failed ({detected_lang}→en): {exc} — using original question for retrieval.")

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
    """Returns basic health information including vector count and model name."""
    return HealthResponse(
        status="ok",
        vector_count=retriever.vector_count(),
        model=retriever.current_model(),
    )
