"""
Load the unified FAISS index and retrieve top-K chunks for a query.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Module-level singleton used by main.py
_retriever: "Retriever | None" = None


def load(index_dir: str, model_name: str = "BAAI/bge-base-en") -> None:
    """Initialise the global Retriever instance."""
    global _retriever
    _retriever = Retriever(index_dir=index_dir, model_name=model_name)


def is_loaded() -> bool:
    return _retriever is not None


def retrieve(question: str, top_k: int = 5, sector_filter: str = None) -> list:
    return _retriever.search(question, top_k=top_k, sector_filter=sector_filter)


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


class Retriever:
    def __init__(self, index_dir: str, model_name: str = "BAAI/bge-base-en"):
        index_dir = Path(index_dir)

        # Validate model matches the one used to build the index
        config_path = index_dir / "index_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            built_with = config.get("embed_model", "")
            if built_with and built_with != model_name:
                raise ValueError(
                    f"Model mismatch: index was built with '{built_with}' "
                    f"but retriever was asked to load '{model_name}'. "
                    f"Rebuild the index with --model {model_name} or set "
                    f"EMBED_MODEL={built_with} in your environment."
                )

        print(f"Loading FAISS index from {index_dir} ...")
        self.index      = faiss.read_index(str(index_dir / "unified.index"))
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name

        with open(index_dir / "metadata.json", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Retriever ready -- {self.index.ntotal} vectors, {len(self.metadata)} chunks")

    def search(self, query: str, top_k: int = 5, sector_filter: str = None) -> list:
        """Search the index for the top_k most relevant chunks."""
        q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)

        # Fetch more candidates when filtering so we still get top_k after sector filtering.
        # 50x multiplier ensures small sectors (e.g. matrimonial) aren't missed.
        fetch_k = min(self.index.ntotal, top_k * 50) if sector_filter else top_k
        scores, indices = self.index.search(q_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.metadata[idx]

            if sector_filter and chunk.get("sector") != sector_filter:
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
                "score"          : float(score),
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
