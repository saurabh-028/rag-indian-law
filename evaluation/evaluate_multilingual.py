"""
evaluation/evaluate_multilingual.py — Runs ShastraShaw's actual production
pipeline (translate-for-retrieval -> hybrid retrieve -> sector-prompted
generate -> citation verify/retry) against the Hindi/Marathi cross-lingual
sub-study dataset built by build_multilingual_subset.py.

This mirrors app/main.py's /query endpoint logic directly (rather than
spinning up FastAPI), with two deliberate simplifications appropriate for a
fixed-sector eval set rather than live traffic:
  - `sector_filter` is taken from the dataset row (as evaluate.py already
    does for the English gold set) instead of running app/router.py's
    cross-sector disambiguation.
  - `lang` is taken from the dataset's `language` field instead of running
    app/language.py's detect_language() — langdetect is documented in
    CLAUDE.md as unreliable on short queries, and we already know the true
    language here, so there's no reason to risk a wrong auto-detect
    confounding the Hindi/Marathi results with an English-prompt bug.
Everything else (translation-for-retrieval, original-language generation
prompt, citation verification + one retry) matches production exactly.

Usage:
    python evaluation/evaluate_multilingual.py \\
        --index_dir ./index \\
        --dataset   ./evaluation/gold_dataset_multilingual.json \\
        --out_dir   ./evaluation/results/shastrashaw_multilingual \\
        [--top_k 10] [--openai_model gpt-4o]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", default="./index")
    p.add_argument("--dataset", default="./evaluation/gold_dataset_multilingual.json")
    p.add_argument("--out_dir", default="./evaluation/results/shastrashaw_multilingual")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--embed_model", default="law-ai/InLegalBERT")
    p.add_argument("--openai_model", default="gpt-4o")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[Dataset] Loaded {len(dataset)} multilingual Q&A rows from {args.dataset}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not set.")
        sys.exit(1)

    from app.retriever import Retriever
    from app.generator import Generator
    from app import verifier
    from app.language import translator, SUPPORTED_LANGS

    print(f"[Retriever] Loading FAISS index from {args.index_dir} ...")
    retriever = Retriever(index_dir=args.index_dir, model_name=args.embed_model)
    print(f"  Ready — {retriever.index.ntotal} vectors")

    generator = Generator(api_key=api_key, model=args.openai_model)
    print(f"[Generator] Using {args.openai_model}\n")

    out_rows = []
    n_retried = 0
    for item in dataset:
        lang = item.get("language", "en")
        question = item["question"]

        # Translate to English for retrieval only — index is English-only.
        # Mirrors app/main.py's /query handling exactly.
        retrieval_question = question
        if lang in SUPPORTED_LANGS:
            try:
                retrieval_question = translator.to_english(question, src_lang=lang)
            except Exception as exc:
                print(f"  [{item['id']}] translation failed ({lang}->en): {exc} — using original text for retrieval")

        chunks = retriever.search(
            retrieval_question,
            top_k=args.top_k,
            sector_filter=item["sector"],
        )

        try:
            gen_result = generator.generate(
                question=question,          # original-language question, same as production
                context_chunks=chunks,
                sector=item["sector"],
                response_lang=lang,
            )
            answer = gen_result["answer"]

            check = verifier.verify_citations(answer, chunks)
            retried = False
            if not check["verified"]:
                if check["available_sections"]:
                    correction = (
                        f"Your previous answer cited Section(s) {', '.join(check['unverified_sections'])}, "
                        "which do not appear anywhere in the context provided above. The section numbers "
                        f"actually present in the context are: {', '.join(check['available_sections'])}. "
                        "Use only these when citing a specific section — do not repeat the incorrect one(s) or guess another."
                    )
                else:
                    correction = (
                        f"Your previous answer cited Section(s) {', '.join(check['unverified_sections'])}, "
                        "which do not appear anywhere in the context provided above, and the context contains no "
                        "numbered sections at all. Rely only on what's stated in the context without citing a section number."
                    )
                retry_result = generator.generate(
                    question=question,
                    context_chunks=chunks,
                    sector=item["sector"],
                    response_lang=lang,
                    correction=correction,
                )
                answer = retry_result["answer"]
                retried = True
                n_retried += 1
        except Exception as e:
            print(f"  [{item['id']}] generation FAILED: {e}")
            answer = ""
            retried = False

        out_rows.append({
            "id": item["id"],
            "original_id": item.get("original_id", item["id"]),
            "sector": item["sector"],
            "language": lang,
            "question": question,
            "generated": answer,
            "citation_retried": retried,
            "chunks_retrieved": len(chunks),
            "top_score": chunks[0]["score"] if chunks else 0.0,
        })
        status = "retry" if retried else "ok"
        print(f"  [{item['id']}] ({lang}, {status}) {question[:45]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated_answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    n_empty = sum(1 for r in out_rows if not r["generated"])
    print(f"\n  Generated: {len(out_rows) - n_empty}/{len(out_rows)}  (failed: {n_empty})")
    print(f"  Citation retries triggered: {n_retried}/{len(out_rows)}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
