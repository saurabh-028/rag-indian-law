"""
router.py — Sector routing and ambiguity detection.

The old logic in main.py just took the majority-vote sector across
retrieved chunks and ran with it, even when votes were closely split
across two unrelated legal domains (e.g. "my husband hit me while driving
drunk" pulls both matrimonial and traffic/criminal chunks). Blending an
answer across domains like that reads as confused rather than helpful.

route() turns the vote count into an explicit decision: a clear winner
proceeds as before; a genuine split returns which sectors are contending
so main.py can ask a clarifying question instead of guessing.
"""

SECTOR_LABELS = {
    "en": {
        "traffic": "Traffic & Motor Vehicles Law",
        "criminal_law": "Criminal Law",
        "matrimonial": "Matrimonial & Family Law",
        "rental_law": "Rental & Tenancy Law",
    },
    "hi": {
        "traffic": "यातायात एवं मोटर वाहन कानून",
        "criminal_law": "आपराधिक कानून",
        "matrimonial": "वैवाहिक एवं पारिवारिक कानून",
        "rental_law": "किराया एवं किरायेदारी कानून",
    },
    "mr": {
        "traffic": "वाहतूक व मोटार वाहन कायदा",
        "criminal_law": "फौजदारी कायदा",
        "matrimonial": "वैवाहिक व कौटुंबिक कायदा",
        "rental_law": "भाडे व भाडेकरू कायदा",
    },
}

# A sector must hold at least this share of retrieved chunks to count as a
# clear routing decision. Below it — with a real runner-up — treat the
# question as ambiguous rather than silently picking the top vote-getter.
_DOMINANCE_THRESHOLD = 0.6

# Minimum vote count for a sector to count as a real contender, not noise
# from a single stray cross-sector chunk (expected given InLegalBERT/BM25
# vocabulary overlap between sectors, e.g. "complaint" matching both
# traffic and matrimonial actionable-procedure chunks).
_MIN_CONTENDER_VOTES = 2


def route(chunks: list) -> dict:
    """Decide the sector for a set of retrieved chunks, and whether the
    question is ambiguous enough to warrant a clarifying question instead
    of an answer blended across unrelated legal domains."""
    votes: dict[str, int] = {}
    for c in chunks:
        s = c.get("sector", "")
        if s:
            votes[s] = votes.get(s, 0) + 1

    if not votes:
        return {"sector": None, "ambiguous": False, "candidates": []}

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_sector, top_votes = ranked[0]
    total = sum(votes.values())

    contenders = [s for s, v in ranked if v >= _MIN_CONTENDER_VOTES]
    ambiguous = len(contenders) >= 2 and (top_votes / total) < _DOMINANCE_THRESHOLD

    return {
        "sector": top_sector,
        "ambiguous": ambiguous,
        "candidates": contenders if ambiguous else [top_sector],
    }


_CONNECTORS = {"en": "or", "hi": "या", "mr": "किंवा"}


def _join_labels(labels: list, lang: str) -> str:
    connector = _CONNECTORS.get(lang, "or")
    if len(labels) <= 2:
        return f" {connector} ".join(labels)
    return ", ".join(labels[:-1]) + f", {connector} {labels[-1]}"


def clarifying_question(candidates: list, lang: str = "en") -> str:
    """Build a clarifying question naming the candidate sectors, fully in the user's language
    (not just a translated template with English labels stuck in the middle)."""
    labels_by_sector = SECTOR_LABELS.get(lang, SECTOR_LABELS["en"])
    labels = [labels_by_sector.get(c, SECTOR_LABELS["en"].get(c, c)) for c in candidates]
    listed = _join_labels(labels, lang)

    templates = {
        "hi": f"आपका सवाल एक से ज़्यादा कानूनी क्षेत्रों से जुड़ा लग रहा है — {listed}। कृपया बताएं आप किस बारे में पूछ रहे हैं?",
        "mr": f"तुमचा प्रश्न एकापेक्षा जास्त कायदेशीर क्षेत्रांशी संबंधित दिसतोय — {listed}. कृपया स्पष्ट करा तुम्ही कशाबद्दल विचारत आहात?",
    }
    return templates.get(
        lang,
        f"Your question seems to touch more than one area of law — {listed}. "
        "Could you clarify which one you're asking about, or add a bit more detail?",
    )
