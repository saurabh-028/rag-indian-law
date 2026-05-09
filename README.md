# Indian Law RAG System

A Retrieval-Augmented Generation (RAG) system for Indian legal documents, covering traffic law, criminal law (BNS), and rental law (Maharashtra).

## Project Structure

```
rag_indian_law/
├── data/
│   ├── traffic/              # Central MV Act, Maharashtra MV Act, fine schedule
│   ├── criminal_law/         # Bharatiya Nyaya Sanhita (BNS)
│   └── rental_law/           # Maharashtra Rent Control Act
├── pipeline/
│   ├── cleaners/
│   │   ├── mv_act_central.py
│   │   ├── mv_act_maharashtra.py
│   │   ├── traffic_fines.py          # table-based PDF cleaner
│   │   ├── bns.py
│   │   └── rent_control_maharashtra.py
│   ├── cleaner_registry.py   # maps PDF filename -> cleaner + metadata
│   └── build_index.py        # CLI: build FAISS index from all PDFs
├── index/                    # output: unified.index + metadata.json
├── app/
│   ├── main.py               # FastAPI application (4 endpoints)
│   └── retriever.py          # FAISS retriever + context formatter
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
pip install -r requirements.txt
```

## PDF Files (already in data/)

| File | Directory |
|------|-----------|
| `Central Motor Vehicle act(amended till 2019).pdf` | `data/traffic/` |
| `Maharashtra Motor Vehicle act 1989.pdf` | `data/traffic/` |
| `traffic violation fines.pdf` | `data/traffic/` |
| `Bhartiya_Nyay_Sanhita(BNS).pdf` | `data/criminal_law/` |
| `eng_maharashtra_rent_control_ac.pdf` | `data/rental_law/` |

## Building the Index

From the `rag_indian_law/` directory:

```bash
python -m pipeline.build_index \
    --data_dir ./data \
    --output_dir ./index \
    --model BAAI/bge-base-en \
    --batch_size 64
```

To use InLegalBERT instead:

```bash
python -m pipeline.build_index --model law-ai/InLegalBERT
```

Produces:
- `index/unified.index` — FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
- `index/metadata.json` — Array of chunk metadata records

## Running the API

```bash
export OPENAI_API_KEY=sk-...
export INDEX_DIR=./index          # optional, default: ./index
export EMBED_MODEL=BAAI/bge-base-en  # optional

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: http://localhost:8000/docs

## API Endpoints

### `POST /query`
Ask a legal question; returns a GPT-4 answer grounded in indexed documents.

```json
{
  "question": "What is the fine for driving without a helmet?",
  "top_k": 5,
  "sector_filter": "traffic"
}
```

Response:
```json
{
  "answer": "According to Section 129 of the Central Motor Vehicle Act...",
  "sources": ["Central Motor Vehicle act(amended till 2019).pdf § 129"],
  "chunks_used": 5
}
```

### `POST /retrieve`
Return raw retrieved chunks without calling an LLM.

```json
{
  "question": "eviction notice period",
  "top_k": 3,
  "sector_filter": "rental_law"
}
```

### `GET /sectors`
Returns `{"sectors": ["criminal_law", "rental_law", "traffic"]}`.

### `GET /health`
Returns `{"status": "ok", "vector_count": 1234, "model": "BAAI/bge-base-en"}`.

## Architecture

```
PDF files
   │
   ├── text-based  →  pymupdf (fitz)  →  cleaner.clean()  →  cleaner.parse()
   └── table-based →  pdfplumber      →  cleaner.parse(pdf_path)
                                             │
                                      list of chunk dicts
                                             │
                              SentenceTransformer (BAAI/bge-base-en)
                                             │
                                   L2-normalise embeddings
                                             │
                              FAISS IndexFlatIP  →  index/unified.index
                                                    index/metadata.json
                                             │
                                     FastAPI /query
                                             │
                              retriever.retrieve()  →  top-k chunks
                              retriever.format_context()
                                             │
                              OpenAI GPT-4  →  grounded answer
```

## Swapping the Embedding Model

1. Rebuild the index: `python -m pipeline.build_index --model law-ai/InLegalBERT`
2. Set `EMBED_MODEL=law-ai/InLegalBERT` before starting the server

Both index build and query time must use the same model.

## Adding New Documents

1. Drop the PDF into the appropriate `data/` subdirectory.
2. Create a cleaner in `pipeline/cleaners/` following the existing pattern.
3. Register it in `pipeline/cleaner_registry.py`.
4. Rebuild the index.
