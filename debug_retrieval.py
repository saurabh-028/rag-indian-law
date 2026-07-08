"""
debug_retrieval.py — quick retrieval debugger to inspect what chunks are returned
for sample queries and whether fine_schedule or legislation chunks rank higher.

Usage:
    .venv/Scripts/python debug_retrieval.py
    .venv/Scripts/python debug_retrieval.py --query "fine for not wearing helmet" --top_k 10
    .venv/Scripts/python debug_retrieval.py --sector traffic --top_k 15
"""

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

INDEX_DIR  = str(_ROOT / "index")
MODEL_NAME = os.getenv("EMBED_MODEL", "law-ai/InLegalBERT")

DEFAULT_QUERIES = [
    ("traffic",       "fine for not wearing helmet"),
    ("traffic",       "penalty for drunk driving"),
    ("traffic",       "fine for driving without licence"),
    ("traffic",       "challan for jumping red light"),
    ("criminal_law",  "punishment for theft"),
    ("criminal_law",  "punishment for murder"),
    ("hindu_marriage_laws", "grounds for divorce"),
    ("hindu_marriage_laws", "maintenance for wife"),
]


def run(query: str, sector: str = None, top_k: int = 10, doc_type_filter: str = None):
    from app.retriever import Retriever
    r = Retriever(index_dir=INDEX_DIR, model_name=MODEL_NAME)
    results = r.search(query, top_k=top_k, sector_filter=sector, doc_type_filter=doc_type_filter)

    print(f"\n{'='*70}")
    print(f"Query   : {query!r}")
    print(f"Sector  : {sector or 'all'}   doc_type_filter: {doc_type_filter or 'all'}")
    print(f"{'='*70}")
    for res in results:
        print(
            f"  #{res['rank']:>2}  score={res['score']:.4f}  "
            f"doc_type={res['doc_type']:<20}  "
            f"source={res['source'][:35]:<35}  "
            f"sec={res['section_number']}"
        )
        # Show first 120 chars of content
        snippet = res["content"].replace("\n", " ")[:120]
        print(f"       {snippet}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   default=None)
    parser.add_argument("--sector",  default=None)
    parser.add_argument("--top_k",   type=int, default=10)
    parser.add_argument("--doc_type", default=None)
    args = parser.parse_args()

    if args.query:
        run(args.query, sector=args.sector, top_k=args.top_k, doc_type_filter=args.doc_type)
    else:
        print(f"Running {len(DEFAULT_QUERIES)} default debug queries...")
        for sector, query in DEFAULT_QUERIES:
            run(query, sector=sector, top_k=args.top_k)

        # Extra: show traffic fine vs legislation breakdown for a fine query
        print("\n\n--- TRAFFIC: fine_schedule only ---")
        run("fine for not wearing helmet", sector="traffic", top_k=5, doc_type_filter="fine_schedule")
        print("\n--- TRAFFIC: legislation only ---")
        run("fine for not wearing helmet", sector="traffic", top_k=5, doc_type_filter="legislation")


if __name__ == "__main__":
    main()
