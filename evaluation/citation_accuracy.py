"""
evaluation/citation_accuracy.py — Deterministic citation-correctness check.

Replaces the subjective "Legal Correctness" Likert score with a hard,
reproducible comparison: does the generated answer cite gold_dataset.json's
expected_section, exactly? No LLM judgment involved, so unlike the original
Legal_AI_Expert_Evaluation_Report.xlsx (which scored a documented completeness
gap as 5/5 in several cases) or the rebuilt llm_judge.py rubric, this number
cannot be internally inconsistent or drift between runs based on judge mood.

Uses the same section-citation regex as app/verifier.py (the production
citation-verification code path), so "correct" here means the same thing
"verified" means at answer time.

Only questions with a non-empty expected_section are scored — procedural
questions (e.g. "how do I file an FIR") don't have a single right answer key
and are intentionally excluded, same as the original report's blank-section
rows.

Usage:
    python evaluation/citation_accuracy.py \\
        --generated ./evaluation/results/generated_answers.json \\
        --dataset   ./evaluation/gold_dataset.json
"""

import argparse
import json
import re
from collections import defaultdict

_SECTION_RE = re.compile(r"\bSection\s+(\d{1,3}[A-Z]{0,2})\b", re.IGNORECASE)
# Same shape verifier.py uses to distinguish a real statute section number from
# an internal actionable-procedure doc ID (e.g. "MAT_GRV_005") — those aren't
# something a model would ever literally cite as "Section MAT_GRV_005", so
# scoring them against this regex-based checker would always be a false
# negative. They're excluded from this check entirely (procedural questions
# need a different kind of correctness check than a section-number match).
_SECTION_NUMBER_SHAPE = re.compile(r"^\d{1,3}[A-Z]{0,2}$")

SECTOR_LABELS = {
    "criminal_law": "Criminal Law",
    "traffic": "Traffic Law",
    "rental_law": "Rental Law",
    "matrimonial": "Matrimonial & Family",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="./evaluation/results/generated_answers.json")
    ap.add_argument("--dataset", default="./evaluation/gold_dataset.json")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(args.generated, encoding="utf-8") as f:
        generated_by_id = {g["id"]: g for g in json.load(f)}

    scored = [
        item for item in dataset
        if item.get("expected_section")
        and _SECTION_NUMBER_SHAPE.match(item["expected_section"].strip())
    ]
    print(f"[Citation Accuracy] {len(scored)}/{len(dataset)} questions have a scoreable "
          f"numeric expected_section (procedural questions and actionable-doc-ID "
          f"references excluded)\n")

    correct, wrong, missing = [], [], []
    for item in scored:
        gen_entry = generated_by_id.get(item["id"], {})
        answer = gen_entry.get("generated", "")
        cited = {m.group(1).strip().upper() for m in _SECTION_RE.finditer(answer)}
        expected = item["expected_section"].strip().upper()

        if not answer:
            missing.append(item)
        elif expected in cited:
            correct.append(item)
        else:
            wrong.append((item, sorted(cited)))

    print(f"  Correct : {len(correct)}/{len(scored)}")
    print(f"  Wrong   : {len(wrong)}/{len(scored)}")
    print(f"  Missing : {len(missing)}/{len(scored)} (no generated answer)")

    if wrong:
        print("\n  Wrong citations:")
        for item, cited in wrong:
            print(f"    [{item['id']}] expected {item['expected_section']!r}, "
                  f"cited {cited or '(none)'} — {item['question'][:60]}")

    print(f"\n{'='*65}")
    print("  SECTOR-WISE CITATION ACCURACY")
    print(f"{'='*65}")
    by_sector = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in correct:
        by_sector[item["sector"]]["correct"] += 1
        by_sector[item["sector"]]["total"] += 1
    for item, _ in wrong:
        by_sector[item["sector"]]["total"] += 1
    for item in missing:
        by_sector[item["sector"]]["total"] += 1

    for sector in sorted(by_sector):
        s = by_sector[sector]
        label = SECTOR_LABELS.get(sector, sector)
        pct = 100 * s["correct"] / s["total"] if s["total"] else 0
        print(f"  {label:<24} {s['correct']}/{s['total']} ({pct:.0f}%)")

    overall_pct = 100 * len(correct) / len(scored) if scored else 0
    print(f"\n  OVERALL: {len(correct)}/{len(scored)} ({overall_pct:.1f}%)")


if __name__ == "__main__":
    main()
