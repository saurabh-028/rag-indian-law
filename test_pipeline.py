"""
test_pipeline.py — End-to-end test for the RAG pipeline.
Run this before starting the API to verify everything works.

Usage:
    python test_pipeline.py --index_dir ./index
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.retriever import Retriever
from app.generator import Generator


TEST_QUERIES = [
    {
        "question"     : "What is the fine for not wearing a helmet?",
        "sector_filter": "traffic",
        "expect_source": "Motor_Vehicles",
    },
    {
        "question"     : "What is the punishment for murder under BNS?",
        "sector_filter": "criminal_law",
        "expect_source": "Bharatiya_Nyaya",
    },
    {
        "question"     : "What are the rights of a tenant under Maharashtra rent law?",
        "sector_filter": "rental_law",
        "expect_source": "Maharashtra_Rent",
    },
    {
        "question"     : "What is the penalty for driving without a valid licence?",
        "sector_filter": "traffic",
        "expect_source": "Motor_Vehicles",
    },
    {
        "question"     : "What is kidnapping?",
        "sector_filter": None,
        "expect_source": None,
    },
]

REQUIRED_CHUNK_FIELDS = [
    "doc_id", "section_number", "section_title",
    "chapter", "content", "source", "sector", "doc_type",
]


def check(condition: bool, msg: str):
    status = "OK" if condition else "FAIL"
    print(f"  [{status}]  {msg}")
    return condition


def run_tests(index_dir: str, test_gpt: bool = False):
    print("\n" + "="*60)
    print("  INDIAN LAW RAG — END-TO-END TEST")
    print("="*60)

    passed = 0
    failed = 0

    # Test 1: Load index
    print("\n[1] Loading index...")
    try:
        retriever = Retriever(index_dir=index_dir)
        ok = check(retriever.index.ntotal > 0, f"FAISS index loaded — {retriever.index.ntotal} vectors")
        ok = check(len(retriever.metadata) > 0, f"Metadata loaded — {len(retriever.metadata)} chunks")
        ok = check(
            retriever.index.ntotal == len(retriever.metadata),
            f"Vector count matches metadata count"
        )
        passed += 3
    except Exception as e:
        print(f"  [FAIL]  Index failed to load: {e}")
        failed += 1
        return

    # Test 2: Sector distribution
    print("\n[2] Checking sector distribution...")
    sector_counts: dict = {}
    for chunk in retriever.metadata:
        s = chunk.get("sector", "MISSING")
        sector_counts[s] = sector_counts.get(s, 0) + 1

    for sector, count in sorted(sector_counts.items()):
        ok = check(count > 0, f"Sector '{sector}': {count} chunks")
        passed += 1

    # Test 3: Chunk field validation
    print("\n[3] Validating chunk fields (sampling 20 chunks)...")
    sample = retriever.metadata[:20]
    missing_fields = set()
    for chunk in sample:
        for field in REQUIRED_CHUNK_FIELDS:
            if field not in chunk:
                missing_fields.add(field)

    ok = check(len(missing_fields) == 0, f"All required fields present in chunks")
    if missing_fields:
        print(f"       Missing fields: {missing_fields}")
        failed += 1
    else:
        passed += 1

    # Test 4: Retrieval quality
    print("\n[4] Retrieval tests...")
    for q in TEST_QUERIES:
        results = retriever.search(
            q["question"],
            top_k=3,
            sector_filter=q["sector_filter"],
        )

        got_results = check(len(results) > 0, f"Query returned results: \"{q['question'][:55]}\"")
        if not got_results:
            failed += 1
            continue
        passed += 1

        top_score = results[0]["score"]
        ok = check(top_score > 0.2, f"  Top score acceptable: {top_score:.4f}")
        passed += 1 if ok else 0
        failed += 0 if ok else 1

        if q["sector_filter"]:
            sectors_returned = set(r["sector"] for r in results)
            ok = check(
                sectors_returned == {q["sector_filter"]},
                f"  Sector filter respected: {sectors_returned}"
            )
            passed += 1 if ok else 0
            failed += 0 if ok else 1

        if q["expect_source"]:
            sources = [r["source"] for r in results]
            source_hit = any(q["expect_source"] in s for s in sources)
            ok = check(source_hit, f"  Expected source found: {q['expect_source']}")
            passed += 1 if ok else 0
            failed += 0 if ok else 1

        top = results[0]
        print(f"       -> [{top['doc_id']}] {top['section_title'][:50]} (score={top_score:.3f})")

    # Test 5: GPT-4 generation (optional)
    if test_gpt:
        print("\n[5] GPT-4 generation test...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("  OPENAI_API_KEY not set — skipping generation test")
        else:
            try:
                gen = Generator(api_key=api_key)
                q = "What is the fine for jumping a red signal?"
                results = retriever.search(q, top_k=3, sector_filter="traffic")
                context = retriever.format_context(results)
                result  = gen.generate(question=q, context=context, sector="traffic")

                ok = check(len(result["answer"]) > 50, f"GPT-4 returned a non-empty answer")
                ok = check("usage" in result, f"Token usage returned")
                passed += 2 if ok else 0
                print(f"\n  Sample answer:\n  {result['answer'][:300]}...\n")
                print(f"  Tokens used: {result['usage']['total_tokens']}")
            except Exception as e:
                print(f"  [FAIL]  GPT-4 test failed: {e}")
                failed += 1

    # Summary
    print("\n" + "="*60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed   |   {failed} failed")
    print("="*60 + "\n")

    if failed == 0:
        print("All tests passed — pipeline is ready!\n")
    else:
        print("Some tests failed — check output above before running the API.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end test for the RAG pipeline")
    parser.add_argument("--index_dir", default="./index", help="Path to the index folder")
    parser.add_argument("--test_gpt",  action="store_true", help="Also test GPT-4 generation")
    args = parser.parse_args()

    run_tests(args.index_dir, test_gpt=args.test_gpt)
