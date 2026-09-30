"""
evaluation/outdated_law_check.py — Deterministic check: does a generated answer
cite the CURRENT law (BNS/BNSS 2023) or default to the REPEALED IPC/CrPC that
BNS/BNSS replaced on 1 July 2024?

This is the headline research question for the HECA abstract, so it is
deliberately kept to a pure regex scan — no LLM judgment involved — in the
same spirit as citation_accuracy.py, so the number is reproducible and cannot
drift between runs.

Two metrics are computed, for the criminal_law sector and the dowry-related
matrimonial questions (the two places this gold dataset asks about an offence
that BNS 2023 recodified):

  1. law_regime_primacy — of the two code families (new: BNS/BNSS/BSA vs.
     old: IPC/CrPC/Indian Evidence Act), which is named FIRST in the answer?
     "new_only" / "old_only" / "new_primary" (both mentioned, new first) /
     "old_primary" (both mentioned, old first) / "neither" (no code name at
     all — rare, e.g. a refusal to answer).

  2. cites_repealed_section_number — for the subset of questions with a
     well-established, unambiguous 1:1 old-section <-> new-section mapping
     (LEGACY_SECTION_MAP below, sourced from the published BNS 2023 / BNSS
     2023 concordance tables), did the answer cite the OLD section number
     as the section governing this offence, and NOT the correct new one?
     This is the more surgical, "hard number" version of (1): it asks not
     just "did it mention IPC" (which ShastraShaw's own prompts also do,
     deliberately, in brackets for reader familiarity — see
     app/generator.py's criminal_law prompt) but "did it cite the OLD
     number as if it were the live one, without the correct new number."

Only questions with a mapping entry are scored for (2); (1) is computed for
every criminal_law / dowry-matrimonial question regardless.

Usage:
    python evaluation/outdated_law_check.py \\
        --generated ./evaluation/results/frontier_baseline/generated_answers.json \\
        --dataset   ./evaluation/gold_dataset.json \\
        --label     "Frontier baseline (no retrieval)" \\
        --out       ./evaluation/results/frontier_baseline/outdated_law_check.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NEW_LAW_RE = re.compile(
    r"\b(?:BNSS|BNS|BSA)\b"
    r"|\bBharatiya\s+Nyaya\s+Sanhita\b"
    r"|\bBharatiya\s+Nagarik\s+Suraksha\s+Sanhita\b"
    r"|\bBharatiya\s+Sakshya\s+Adhiniyam\b",
    re.IGNORECASE,
)
OLD_LAW_RE = re.compile(
    r"\bIPC\b|\bCrPC\b|\bCr\.P\.C\b"
    r"|\bIndian\s+Penal\s+Code\b"
    r"|\bCode\s+of\s+Criminal\s+Procedure\b"
    r"|\bIndian\s+Evidence\s+Act\b",
    re.IGNORECASE,
)

# Reuse the same "Section N" shape as app/verifier.py / citation_accuracy.py.
_SECTION_RE = re.compile(r"\bSection\s+(\d{1,3}[A-Z]{0,2})\b", re.IGNORECASE)

# Well-established, unambiguous BNS/BNSS <-> IPC/CrPC correspondences for
# offences this gold dataset asks about, per the Ministry of Home Affairs'
# published BNS 2023 / BNSS 2023 concordance tables. Deliberately excludes
# CL_002 (arrest rights), CL_005 (bailable/non-bailable classification) and
# CL_007 (kidnapping, a 359-361 IPC range mapping to a 137-140 BNS range) —
# those involve either a many-to-many reorganisation across BNSS or a range
# rather than a single section, so a strict 1:1 "old section X" framing would
# be a contestable simplification. The 9 entries below are all clean 1:1
# swaps that appear in virtually every BNS explainer published in 2023-24.
LEGACY_SECTION_MAP = {
    "CL_001": {"new": "103",  "old": "302",  "old_act": "IPC",  "offence": "murder"},
    "CL_003": {"new": "303",  "old": "379",  "old_act": "IPC",  "offence": "theft"},
    "CL_004": {"new": "173",  "old": "154",  "old_act": "CrPC", "offence": "FIR registration"},
    "CL_006": {"new": "64",   "old": "376",  "old_act": "IPC",  "offence": "rape"},
    "CL_009": {"new": "117",  "old": "325",  "old_act": "IPC",  "offence": "grievous hurt"},
    "CL_011": {"new": "318",  "old": "420",  "old_act": "IPC",  "offence": "cheating"},
    "CL_012": {"new": "482",  "old": "438",  "old_act": "CrPC", "offence": "anticipatory bail"},
    "HM_003": {"new": "85",   "old": "498A", "old_act": "IPC",  "offence": "dowry cruelty"},
    "HM_007": {"new": "85",   "old": "498A", "old_act": "IPC",  "offence": "dowry cruelty (defence)"},
}

# Sectors this check applies to (BNS/BNSS is the relevant current law there).
RELEVANT_SECTORS = {"criminal_law", "matrimonial"}


def _first_pos(regex, text):
    m = regex.search(text or "")
    return m.start() if m else None


def law_regime(answer: str) -> str:
    new_pos = _first_pos(NEW_LAW_RE, answer)
    old_pos = _first_pos(OLD_LAW_RE, answer)
    if new_pos is None and old_pos is None:
        return "neither"
    if new_pos is None:
        return "old_only"
    if old_pos is None:
        return "new_only"
    return "old_primary" if old_pos < new_pos else "new_primary"


def section_verdict(answer: str, mapping: dict) -> str:
    """Returns 'current', 'repealed', 'both', or 'neither' for the section-number check."""
    cited = {m.group(1).strip().upper() for m in _SECTION_RE.finditer(answer or "")}
    has_new = mapping["new"].upper() in cited
    has_old = mapping["old"].upper() in cited
    if has_new and has_old:
        return "both"
    if has_new:
        return "current"
    if has_old:
        return "repealed"
    return "neither"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--dataset", default="./evaluation/gold_dataset.json")
    ap.add_argument("--label", default=None, help="Human-readable label for this run in the printout")
    ap.add_argument("--out", default=None, help="Optional path to also save a JSON summary")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)
    with open(args.generated, encoding="utf-8") as f:
        generated_by_id = {g["id"]: g for g in json.load(f)}

    label = args.label or args.generated
    print(f"{'='*70}")
    print(f"  OUTDATED-LAW CITATION CHECK — {label}")
    print(f"{'='*70}\n")

    # ---- Metric 1: law-regime primacy, over all criminal_law + matrimonial items ----
    relevant_items = [it for it in dataset if it["sector"] in RELEVANT_SECTORS]
    regime_counts = defaultdict(int)
    regime_rows = []
    for item in relevant_items:
        gen = generated_by_id.get(item["id"], {})
        answer = gen.get("generated", "")
        verdict = law_regime(answer) if answer else "no_answer"
        regime_counts[verdict] += 1
        regime_rows.append({"id": item["id"], "sector": item["sector"], "verdict": verdict})

    n_relevant = len(relevant_items)
    print(f"[1] Law-regime primacy — which code family is named FIRST "
          f"(criminal_law + matrimonial, n={n_relevant})")
    for verdict in ["new_only", "new_primary", "old_primary", "old_only", "neither", "no_answer"]:
        c = regime_counts.get(verdict, 0)
        if c:
            pct = 100 * c / n_relevant
            print(f"    {verdict:<14} {c}/{n_relevant} ({pct:.0f}%)")

    old_defaults = regime_counts.get("old_primary", 0) + regime_counts.get("old_only", 0)
    old_default_pct = 100 * old_defaults / n_relevant if n_relevant else 0
    print(f"\n    --> Defaults to repealed law (IPC/CrPC primary or exclusive): "
          f"{old_defaults}/{n_relevant} ({old_default_pct:.0f}%)")

    # ---- Metric 2: repealed section number cited for a clean 1:1 mapped offence ----
    mapped_items = [it for it in dataset if it["id"] in LEGACY_SECTION_MAP]
    section_counts = defaultdict(int)
    section_rows = []
    for item in mapped_items:
        mapping = LEGACY_SECTION_MAP[item["id"]]
        gen = generated_by_id.get(item["id"], {})
        answer = gen.get("generated", "")
        verdict = section_verdict(answer, mapping) if answer else "no_answer"
        section_counts[verdict] += 1
        section_rows.append({
            "id": item["id"], "offence": mapping["offence"],
            "expected_new": mapping["new"], "old_equivalent": f"{mapping['old']} {mapping['old_act']}",
            "verdict": verdict,
        })

    n_mapped = len(mapped_items)
    print(f"\n[2] Repealed section-number citation — clean 1:1 mapped offences (n={n_mapped})")
    for row in section_rows:
        print(f"    [{row['id']:<7}] {row['offence']:<26} "
              f"expected {row['expected_new']!r} (old: {row['old_equivalent']}) -> {row['verdict']}")

    for verdict in ["current", "both", "repealed", "neither", "no_answer"]:
        c = section_counts.get(verdict, 0)
        if c:
            pct = 100 * c / n_mapped if n_mapped else 0
            print(f"    TOTAL {verdict:<10} {c}/{n_mapped} ({pct:.0f}%)")

    repealed_only_pct = 100 * section_counts.get("repealed", 0) / n_mapped if n_mapped else 0
    print(f"\n    --> Cited ONLY the repealed section number (no current one at all): "
          f"{section_counts.get('repealed', 0)}/{n_mapped} ({repealed_only_pct:.0f}%)")

    summary = {
        "label": label,
        "generated_file": args.generated,
        "law_regime_primacy": {
            "n": n_relevant,
            "counts": dict(regime_counts),
            "old_default_count": old_defaults,
            "old_default_pct": round(old_default_pct, 1),
            "rows": regime_rows,
        },
        "repealed_section_citation": {
            "n": n_mapped,
            "counts": dict(section_counts),
            "repealed_only_pct": round(repealed_only_pct, 1),
            "rows": section_rows,
        },
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

    return summary


if __name__ == "__main__":
    main()
