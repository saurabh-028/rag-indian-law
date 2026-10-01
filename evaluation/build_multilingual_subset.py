"""
evaluation/build_multilingual_subset.py — Builds the Hindi + Marathi cross-lingual
sub-study dataset for the HECA research study.

gold_dataset.json is 100% English with no language field, so it cannot test
whether ShastraShaw (or a frontier baseline) holds up in Hindi/Marathi. This
script selects a fixed 16-question subset (4 per sector, chosen to span both
legislation-lookup and procedural questions, and to overlap with
outdated_law_check.py's LEGACY_SECTION_MAP where possible) and pairs each
with a Hindi and a Marathi translation, producing 32 additional rows.

The translations themselves (TRANSLATIONS dict below) were written by hand
(not machine-translated) aiming for the same natural, everyday spoken
register the production system's own language instructions ask the LLM to
use (see LANGUAGE_INSTRUCTIONS in app/generator.py) — e.g. "चालान"/"फाइन"
rather than stiff Sanskritised legal prose — with Act names and section
numbers kept in their original Latin-script/numeral form, exactly as the
production prompts also require, so a downstream regex check (e.g.
outdated_law_check.py) still finds "BNS"/"IPC" etc. even inside a
Hindi/Marathi sentence.

This is a smaller, ADDED sub-study on top of the 44-question English gold
set, not a replacement for it — see RESEARCH_FINDINGS.md for the caveat this
implies (16 questions x 2 languages = 32 datapoints per condition, far too
small to generalise beyond "a first signal").

Usage:
    python evaluation/build_multilingual_subset.py \\
        --dataset ./evaluation/gold_dataset.json \\
        --out ./evaluation/gold_dataset_multilingual.json
"""

import argparse
import json

# 4 questions per sector; ids chosen to overlap with outdated_law_check.py's
# LEGACY_SECTION_MAP (CL_001, CL_003, CL_006, CL_011, HM_003) wherever possible,
# and otherwise to mix legislation-lookup and procedural question types.
SELECTED_IDS = [
    "CL_001", "CL_003", "CL_006", "CL_011",
    "TR_001", "TR_002", "TR_003", "TR_004",
    "RL_001", "RL_002", "RL_003", "RL_004",
    "HM_001", "HM_003", "HM_004", "HM_006",
]

# Hand-written Hindi / Marathi translations (see module docstring for style notes).
TRANSLATIONS = {
    "CL_001": {
        "hi": "BNS 2023 के तहत हत्या की सज़ा क्या है?",
        "mr": "BNS 2023 अंतर्गत खुनाची शिक्षा काय आहे?",
    },
    "CL_003": {
        "hi": "BNS 2023 के तहत चोरी की सज़ा क्या है?",
        "mr": "BNS 2023 अंतर्गत चोरीची शिक्षा काय आहे?",
    },
    "CL_006": {
        "hi": "BNS 2023 के तहत बलात्कार की सज़ा क्या है?",
        "mr": "BNS 2023 अंतर्गत बलात्काराची शिक्षा काय आहे?",
    },
    "CL_011": {
        "hi": "BNS 2023 के तहत धोखाधड़ी (cheating) की सज़ा क्या है?",
        "mr": "BNS 2023 अंतर्गत फसवणुकीची (cheating) शिक्षा काय आहे?",
    },
    "TR_001": {
        "hi": "महाराष्ट्र में हेलमेट न पहनने पर कितना चालान (फाइन) कटता है?",
        "mr": "महाराष्ट्रात हेल्मेट न घातल्यास किती दंड (फाइन) आकारला जातो?",
    },
    "TR_002": {
        "hi": "Motor Vehicles Act के तहत शराब पीकर गाड़ी चलाने (drunk driving) पर क्या सज़ा है?",
        "mr": "Motor Vehicles Act अंतर्गत दारू पिऊन गाडी चालवल्यास (drunk driving) काय शिक्षा आहे?",
    },
    "TR_003": {
        "hi": "लाल सिग्नल जंप करने पर कितना फाइन लगता है?",
        "mr": "लाल सिग्नल तोडल्यास किती फाइन आकारला जातो?",
    },
    "TR_004": {
        "hi": "बिना वैध ड्राइविंग लाइसेंस के गाड़ी चलाने पर क्या पेनल्टी है?",
        "mr": "वैध ड्रायव्हिंग लायसन्सशिवाय गाडी चालवल्यास काय दंड आहे?",
    },
    "RL_001": {
        "hi": "महाराष्ट्र रेंट कंट्रोल एक्ट 1999 के तहत किराएदार (tenant) के क्या अधिकार हैं?",
        "mr": "महाराष्ट्र भाडे नियंत्रण कायदा 1999 अंतर्गत भाडेकरूचे (tenant) कोणते अधिकार आहेत?",
    },
    "RL_002": {
        "hi": "क्या महाराष्ट्र में मकान मालिक हर साल 4% से ज़्यादा किराया बढ़ा सकता है?",
        "mr": "महाराष्ट्रात घरमालक दरवर्षी 4% पेक्षा जास्त भाडेवाढ करू शकतो का?",
    },
    "RL_003": {
        "hi": "किराएदार को बेदखल (eviction) करने के वैध कानूनी आधार क्या हैं?",
        "mr": "भाडेकरूला बेदखल (eviction) करण्यासाठी कायदेशीर कारणे कोणती आहेत?",
    },
    "RL_004": {
        "hi": "अगर मकान मालिक ने मुझे घर खाली कराने के लिए पानी की सप्लाई बंद कर दी है तो मैं क्या कर सकता हूँ?",
        "mr": "घरमालकाने घर रिकामे करण्यासाठी पाणीपुरवठा बंद केला असेल तर मी काय करू शकतो?",
    },
    "HM_001": {
        "hi": "हिंदू मैरिज एक्ट 1955 के तहत तलाक (divorce) के आधार क्या हैं?",
        "mr": "हिंदू विवाह कायदा 1955 अंतर्गत घटस्फोटाची (divorce) कारणे कोणती आहेत?",
    },
    "HM_003": {
        "hi": "BNS के तहत दहेज उत्पीड़न (dowry harassment) की सज़ा क्या है?",
        "mr": "BNS अंतर्गत हुंडाबळी छळाची (dowry harassment) शिक्षा काय आहे?",
    },
    "HM_004": {
        "hi": "एक वैध हिंदू विवाह (Hindu marriage) की शर्तें क्या हैं?",
        "mr": "वैध हिंदू विवाहासाठी (Hindu marriage) अटी काय आहेत?",
    },
    "HM_006": {
        "hi": "घरेलू हिंसा से महिलाओं के संरक्षण अधिनियम (Domestic Violence Act) क्या है और शिकायत कौन कर सकता है?",
        "mr": "घरगुती हिंसाचारापासून महिलांचे संरक्षण कायदा (Domestic Violence Act) म्हणजे काय आणि तक्रार कोण करू शकते?",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="./evaluation/gold_dataset.json")
    ap.add_argument("--out", default="./evaluation/gold_dataset_multilingual.json")
    args = ap.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = {item["id"]: item for item in json.load(f)}

    missing = [i for i in SELECTED_IDS if i not in dataset]
    if missing:
        raise SystemExit(f"IDs not found in {args.dataset}: {missing}")
    missing_tr = [i for i in SELECTED_IDS if i not in TRANSLATIONS]
    if missing_tr:
        raise SystemExit(f"No translation entry for: {missing_tr}")

    rows = []
    for oid in SELECTED_IDS:
        item = dataset[oid]
        for lang in ("hi", "mr"):
            rows.append({
                "id": f"{oid}_{lang}",
                "original_id": oid,
                "sector": item["sector"],
                "language": lang,
                "question": TRANSLATIONS[oid][lang],
                "question_en": item["question"],
                "ground_truth": item["ground_truth"],
                "expected_sources": item["expected_sources"],
                "expected_section": item["expected_section"],
            })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Built {len(rows)} rows ({len(SELECTED_IDS)} questions x 2 languages) -> {args.out}")
    by_sector = {}
    for r in rows:
        by_sector[r["sector"]] = by_sector.get(r["sector"], 0) + 1
    for sector, n in sorted(by_sector.items()):
        print(f"  {sector:<14} {n}")


if __name__ == "__main__":
    main()
