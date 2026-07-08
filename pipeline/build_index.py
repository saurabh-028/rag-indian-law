"""
build_index.py — CLI script to build a unified FAISS index from legal PDFs.

Usage:
    python -m pipeline.build_index \\
        --data_dir ./data \\
        --output_dir ./index \\
        --model BAAI/bge-base-en \\
        --batch_size 64

Steps:
    1. Walks data_dir recursively for *.pdf and *.json files.
    2. Looks up each file in REGISTRY (warns + skips unknown files).
    3. For table-based PDFs  -> calls cleaner.parse(pdf_path)
       For text-based PDFs   -> extracts text via pymupdf, then clean() + parse()
       For JSON files        -> calls cleaner.parse_from_json(path)
    4. Embeds the 'content' field of every chunk with sentence-transformers.
    5. Builds a single FAISS IndexFlatIP (cosine similarity via L2-normalised vectors).
    6. Saves unified.index, metadata.json, and index_config.json to output_dir.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# Suppress noisy deprecation warnings from sentence-transformers
warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.cleaner_registry import REGISTRY  # noqa: E402


def _extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pymupdf."""
    try:
        import fitz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pymupdf is required. Install with: pip install pymupdf") from exc

    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages_text)


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    print(f"[build_index] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def _embed_chunks(
    model,
    chunks: list[dict],
    batch_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed the 'content' field of each chunk. Returns L2-normalised float32 array."""
    texts = [chunk["content"] for chunk in chunks]
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    all_embeddings: list[np.ndarray] = []
    batches = [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    iterator = tqdm(batches, desc="Embedding batches", unit="batch") if show_progress else batches

    for batch in iterator:
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embeddings.astype(np.float32))

    matrix = np.vstack(all_embeddings)

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(matrix)
    return matrix


def build_index(
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths  = sorted(data_dir.rglob("*.pdf"))
    json_paths = sorted(data_dir.rglob("*.json"))
    all_paths  = pdf_paths + json_paths

    if not all_paths:
        print(f"[build_index] No source files found under {data_dir}. Exiting.")
        return

    print(f"[build_index] Found {len(pdf_paths)} PDF(s) and {len(json_paths)} JSON(s) under {data_dir}")

    all_chunks: list[dict] = []

    for file_path in all_paths:
        filename = file_path.name
        entry = REGISTRY.get(filename)

        if entry is None:
            print(f"[build_index] WARNING: '{filename}' not in registry — skipping.")
            continue

        cleaner        = entry["cleaner"]
        is_table_based = entry.get("table_based", False)
        is_json_based  = entry.get("json_based", False)

        print(f"[build_index] Processing: {filename} (table_based={is_table_based}, json_based={is_json_based})")

        try:
            if is_json_based:
                chunks = cleaner.parse_from_json(str(file_path))
            elif is_table_based:
                chunks = cleaner.parse(file_path)
            else:
                raw_text = _extract_text_pymupdf(file_path)
                clean_text = cleaner.clean(raw_text)
                chunks = cleaner.parse(clean_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[build_index] ERROR processing '{filename}': {exc}")
            continue

        print(f"[build_index]   -> {len(chunks)} chunks extracted")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[build_index] No chunks produced. Nothing to index.")
        return

    print(f"[build_index] Total chunks: {len(all_chunks)}")

    model = _load_model(model_name)

    print("[build_index] Embedding chunks …")
    embeddings = _embed_chunks(model, all_chunks, batch_size=batch_size)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on L2-normalised vectors = cosine similarity
    index.add(embeddings)
    print(f"[build_index] FAISS index built — {index.ntotal} vectors, dim={dim}")

    index_path = output_dir / "unified.index"
    faiss.write_index(index, str(index_path))
    print(f"[build_index] Saved FAISS index -> {index_path}")

    metadata: list[dict] = []
    for faiss_id, chunk in enumerate(all_chunks):
        record = {"faiss_id": faiss_id}
        record.update(chunk)
        metadata.append(record)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[build_index] Saved metadata   -> {metadata_path}")

    # Save config so the retriever can validate it's loading the right model
    config = {
        "embed_model": model_name,
        "dim": dim,
        "vector_count": index.ntotal,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    config_path = output_dir / "index_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"[build_index] Saved index config -> {config_path}")
    print("[build_index] Done.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified FAISS index from Indian legal PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="Root directory containing sector sub-folders with PDFs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./index"),
        help="Directory where unified.index and metadata.json will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-base-en",
        help="SentenceTransformer model name or path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(
        data_dir=args.data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        model_name=args.model,
        batch_size=args.batch_size,
    )
