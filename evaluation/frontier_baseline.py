"""
evaluation/frontier_baseline.py — Control condition for the HECA research study.

Simulates what a general-purpose LLM assistant (no retrieval, no legal-domain
prompt engineering) gives a citizen who asks it an Indian legal question
directly — the thing ShastraShaw is actually competing with in practice,
since a plain chatbot is most users' first port of call.

Each question is sent to the model completely bare:
  - No retrieved legal context (unlike app/generator.py's build_user_prompt).
  - No sector-specific system prompt (unlike SECTOR_PROMPTS in app/generator.py).
  - No citation verification / retry (there is no retrieved context to verify
    citations against — a bare LLM either knows the right section or it doesn't).
  - The question is passed exactly as it appears in the dataset, in whatever
    language it's written in (English, Hindi, or Marathi), so translation
    quality is not a confound in this condition.

Output shape matches evaluate.py's generated_answers.json (id-keyed list with
a "generated" field) so evaluation/citation_accuracy.py, evaluation/llm_judge.py,
and evaluation/outdated_law_check.py can consume it unmodified.

Usage:
    python evaluation/frontier_baseline.py \\
        --dataset ./evaluation/gold_dataset.json \\
        --out_dir ./evaluation/results/frontier_baseline \\
        [--model gpt-4o]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.retry_utils import call_with_retry

GENERIC_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable general-purpose AI assistant. A user has asked you "
    "a legal question about Indian law. Answer it as accurately and specifically as you "
    "can using your own knowledge — you do not have access to any external documents or "
    "search tools, only what you already know. Respond in the same language the question "
    "is written in. Be direct and specific, and cite section numbers or Act names if you "
    "know them."
)


def generate_one(client: OpenAI, model: str, question: str) -> tuple[str, dict]:
    response = call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
        max_tokens=1500,
    ))
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return response.choices[0].message.content.strip(), usage


def main():
    ap = argparse.ArgumentParser(description="Frontier (no-retrieval) baseline generator")
    ap.add_argument("--dataset", default="./evaluation/gold_dataset.json")
    ap.add_argument("--out_dir", default="./evaluation/results/frontier_baseline")
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY not set.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    print(f"[Frontier Baseline] {args.model}, NO retrieval context, NO sector prompt")
    print(f"  Dataset: {args.dataset} ({len(dataset)} questions)\n")

    out_rows = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for item in dataset:
        try:
            answer, usage = generate_one(client, args.model, item["question"])
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
        except Exception as e:
            print(f"  [{item['id']}] generation FAILED: {e}")
            answer = ""

        out_rows.append({
            "id": item["id"],
            "sector": item["sector"],
            "question": item["question"],
            "language": item.get("language", "en"),
            "generated": answer,
            "citation_retried": False,
            "condition": "frontier_baseline",
            "model": args.model,
        })
        print(f"  [{item['id']}] ({item.get('language', 'en')}) {item['question'][:55]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated_answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2, ensure_ascii=False)

    n_failed = sum(1 for r in out_rows if not r["generated"])
    print(f"\n  Generated: {len(out_rows) - n_failed}/{len(out_rows)}  (failed: {n_failed})")
    print(f"  Total tokens used: {total_usage['total_tokens']}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
