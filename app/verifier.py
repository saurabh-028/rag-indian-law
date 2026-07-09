"""
verifier.py — Post-generation citation check.

Extracts every "Section N" reference from a drafted answer and checks it
against the section_number metadata of the chunks that were actually
retrieved for the query. Catches fabricated section numbers — a known
hallucination failure mode (the pre-reindex eval found the model citing
BNS sections that were never in the index at all) — before the answer
reaches the user.

Deliberately out of scope: named case citations (e.g. "Arnesh Kumar
(2014)"). Those are static legal knowledge baked directly into the
sector system prompts, not retrieval-grounded, so checking them against
retrieved context would flag every answer that correctly cites them.

Also a known false-positive source: historical references like "Section
125 CrPC" mentioned alongside a current BNSS citation (the criminal_law
and matrimonial prompts explicitly ask for this). The old number won't
be in the retrieved chunks, so it reads as unverified even though it's
intentional. Accepted for v1 — worst case is one extra retry.
"""

import re

_SECTION_RE = re.compile(r"\bSection\s+(\d{1,3}[A-Z]{0,2})\b", re.IGNORECASE)
# Matches the same shape _SECTION_RE captures — used to filter chunk section_number
# values down to real legislation citations, excluding actionable-procedure IDs
# like "traffic_challan_COURTS" which also live in the section_number field.
_SECTION_NUMBER_SHAPE = re.compile(r"^\d{1,3}[A-Z]{0,2}$")


def verify_citations(answer: str, context_chunks: list) -> dict:
    """Check every 'Section N' citation in the answer against retrieved chunk metadata."""
    retrieved = {
        c.get("section_number", "").strip().upper()
        for c in context_chunks
        if c.get("section_number")
    }
    legislation_sections = {s for s in retrieved if _SECTION_NUMBER_SHAPE.match(s)}
    cited = {m.group(1).strip().upper() for m in _SECTION_RE.finditer(answer)}
    unverified = sorted(cited - legislation_sections)
    return {
        "verified": not unverified,
        "cited_sections": sorted(cited),
        "unverified_sections": unverified,
        # Legit section numbers the model could have cited instead — lets a
        # correction retry point at the right answer rather than just saying "wrong".
        "available_sections": sorted(legislation_sections),
    }
