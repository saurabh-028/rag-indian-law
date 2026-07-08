"""
Load the unified FAISS index and retrieve top-K chunks for a query.

Retrieval strategy: Hybrid BM25 + dense vector search with Reciprocal Rank Fusion (RRF).
- Dense (InLegalBERT): captures legal terminology and semantic similarity.
- BM25 (keyword):      catches exact matches — "helmet" -> "protective headgear (Helmet)",
                        "theft" -> BNS §303, "murder" -> §101, etc.
- RRF merges both rankings without requiring score calibration.
"""

import json
import logging
import re
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Module-level singleton used by main.py
_retriever: "Retriever | None" = None

_RRF_K = 60  # standard RRF constant; higher = less weight on top-ranked docs


def load(index_dir: str, model_name: str = "BAAI/bge-base-en") -> None:
    """Initialise the global Retriever instance."""
    global _retriever
    _retriever = Retriever(index_dir=index_dir, model_name=model_name)


def is_loaded() -> bool:
    return _retriever is not None


def retrieve(question: str, top_k: int = 5, sector_filter: str = None, doc_type_filter: str = None) -> list:
    return _retriever.search(question, top_k=top_k, sector_filter=sector_filter, doc_type_filter=doc_type_filter)


def format_context(chunks: list) -> str:
    return _retriever.format_context(chunks)


def unique_sectors() -> list:
    sectors = {chunk.get("sector", "") for chunk in _retriever.metadata}
    return sorted(s for s in sectors if s)


def vector_count() -> int:
    if _retriever is None:
        return 0
    return _retriever.index.ntotal


def current_model() -> str:
    if _retriever is None:
        return ""
    return _retriever.model_name


def index_info() -> dict:
    """Return deployment-relevant metadata about the loaded index."""
    if _retriever is None:
        return {}
    return {
        "built_at":        _retriever.built_at,
        "index_source":    _retriever.index_source,
        "vector_count":    _retriever.index.ntotal,
        "chunks_by_sector": _retriever.chunks_by_sector,
    }


# Normalization map: British→American spellings + key stem collapses.
# This ensures "licence"=="license", "driving"=="drive" etc. across
# query and corpus so BM25 keyword matching works correctly.
_NORM: dict[str, str] = {
    # British → American
    "licence": "license", "licences": "licenses", "licenced": "licensed",
    "offence": "offense",  "offences": "offenses",
    "defence": "defense",
    # Verb → root (keeps BM25 from missing stemmed forms)
    "driving": "drive",  "drives": "drive",  "drove": "drive",
    "riding":  "ride",   "rides":  "ride",
    "wearing": "wear",   "wears":  "wear",
    "parking": "park",   "parked": "park",
    "drunk":   "drink",  "drank":  "drink",  # "drunk and drive"="drink and drive"
    "jumping": "jump",   "jumped": "jump",
    "speeding":"speed",
    "punished":"punish", "punishment":"punish", "punishable":"punish",
    "imprisoned":"imprison", "imprisonment":"imprison",
    # Traffic domain synonyms
    "challan":     "challan",   # keep — now appears in fine_schedule content too
    "alcohol":     "drink",     # "alcohol" → same stem as "drunk and drive"
    "intoxicated": "drink",
    "seatbelt":    "belt",      # "seatbelt" → "belt" matches "safety belt"
}


def _tokenize(text: str) -> list[str]:
    """Tokenize + normalize for BM25: lowercase, British→American, key stems."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [_NORM.get(t, t) for t in tokens]


class Retriever:
    def __init__(self, index_dir: str, model_name: str = "BAAI/bge-base-en"):
        index_dir = Path(index_dir)

        # Validate model matches the one used to build the index
        import os
        config_path = index_dir / "index_config.json"
        self.built_at = "unknown"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            built_with = config.get("embed_model", "")
            self.built_at = config.get("built_at", "unknown")
            if built_with and built_with != model_name:
                raise ValueError(
                    f"Model mismatch: index was built with '{built_with}' "
                    f"but retriever was asked to load '{model_name}'. "
                    f"Rebuild the index with --model {model_name} or set "
                    f"EMBED_MODEL={built_with} in your environment."
                )

        # Record where the index came from
        s3_bucket = os.getenv("S3_INDEX_BUCKET")
        s3_prefix = os.getenv("S3_INDEX_PREFIX", "index")
        self.index_source = f"s3://{s3_bucket}/{s3_prefix}" if s3_bucket else "local"

        logger.info("Loading FAISS index from %s (built_at=%s, source=%s)", index_dir, self.built_at, self.index_source)
        self.index      = faiss.read_index(str(index_dir / "unified.index"))
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name

        with open(index_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.ntotal != len(self.metadata):
            logger.warning(
                "Index has %d vectors but metadata has %d entries — rebuild the index.",
                self.index.ntotal, len(self.metadata),
            )

        # Summarise chunk distribution by sector for health endpoint
        sector_counts: dict = {}
        for chunk in self.metadata:
            s = chunk.get("sector", "unknown")
            sector_counts[s] = sector_counts.get(s, 0) + 1
        self.chunks_by_sector = sector_counts

        # Build BM25 index.
        # section_title is repeated 4x so that exact offense-name matches
        # (e.g. "Drunk and Drive", "Riding without helmet") dominate BM25
        # scoring, preventing chunks with only a shared template prefix from
        # outranking the genuinely relevant one.
        logger.info("Building BM25 index over %d chunks ...", len(self.metadata))
        corpus_tokens = []
        for chunk in self.metadata:
            content = chunk.get("content", "")
            title   = chunk.get("section_title", "")
            # Title repeated 4x: boosts exact title matches strongly in TF-IDF
            combined = f"{title} {title} {title} {title} {content}" if title else content
            corpus_tokens.append(_tokenize(combined))
        self.bm25 = BM25Okapi(corpus_tokens)

        logger.info("Retriever ready -- %d vectors, %d chunks, sectors: %s", self.index.ntotal, len(self.metadata), list(self.chunks_by_sector.keys()))

    def search(self, query: str, top_k: int = 5, sector_filter: str = None, doc_type_filter: str = None) -> list:
        """
        Hybrid search: BM25 + dense vector, merged via Reciprocal Rank Fusion.
        Both searches fetch a large candidate pool; filters applied after merge.
        """
        needs_filter = sector_filter or doc_type_filter
        # Fetch a large candidate pool — must be enough to survive filtering
        candidate_k = min(len(self.metadata), max(top_k * 50, 200)) if needs_filter else min(len(self.metadata), top_k * 10)

        # --- Dense retrieval ---
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        dense_scores, dense_indices = self.index.search(q_emb, candidate_k)

        # --- BM25 retrieval ---
        q_tokens   = _tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        # top candidate_k BM25 hits (by index into metadata)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:candidate_k]

        # --- RRF merge ---
        # rrf_score[metadata_idx] = sum of 1/(k + rank) from each retriever
        rrf: dict[int, float] = {}

        # Dense weight 1.0, BM25 weight 2.0.
        # Dense (InLegalBERT) excels at semantic similarity for long legislation text.
        # BM25 excels at exact keyword matching (offense names, section numbers).
        # Giving BM25 more weight prevents dense-embedding collapse from burying
        # fine_schedule chunks that all share a similar template prefix.
        _DENSE_W = 1.0
        _BM25_W  = 2.0

        for rank, idx in enumerate(dense_indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            rrf[idx] = rrf.get(idx, 0.0) + _DENSE_W / (_RRF_K + rank + 1)

        for rank, idx in enumerate(bm25_top_idx):
            if idx < 0 or idx >= len(self.metadata):
                continue
            rrf[idx] = rrf.get(idx, 0.0) + _BM25_W / (_RRF_K + rank + 1)

        # Sort by descending RRF score
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

        # --- Apply filters and collect top_k ---
        results = []
        for idx, rrf_score in ranked:
            chunk = self.metadata[idx]

            if sector_filter and chunk.get("sector") != sector_filter:
                continue
            if doc_type_filter and chunk.get("doc_type") != doc_type_filter:
                continue

            results.append({
                "rank"           : len(results) + 1,
                "doc_id"         : chunk.get("doc_id", ""),
                "section_number" : chunk.get("section_number", ""),
                "section_title"  : chunk.get("section_title", ""),
                "chapter"        : chunk.get("chapter", ""),
                "content"        : chunk.get("content", ""),
                "source"         : chunk.get("source", ""),
                "sector"         : chunk.get("sector", ""),
                "doc_type"       : chunk.get("doc_type", ""),
                "score"          : round(rrf_score, 6),
            })

            if len(results) >= top_k:
                break

        return results

    def format_context(self, results: list) -> str:
        """Format retrieved chunks into a single context string for the LLM prompt."""
        parts = []
        for r in results:
            parts.append(
                f"[Source: {r['source']} | Section {r['section_number']} — {r['section_title']}]\n"
                f"{r['content']}"
            )
        return "\n\n---\n\n".join(parts)
