"""
Cleaner for: Dowry_Prohibition_Act_1961.pdf
Sector: matrimonial

Unlike hindu_marriage_act.py's source, this bare-act PDF (indiacode.nic.in,
Act No. 28 of 1961, modified 2018) has extraction quirks that break the
plain HMA-style SECTION_RE if applied directly:

1. Every section inserted/substituted by a later amending Act is wrapped in
   a footnote-marker bracket at the START of its line, e.g.
       "1[4. Penalty for demanding dowry.--..."
       "4[7. Cognizance of offences.--..."
   The leading "N[" hides the real section number from SECTION_RE's
   line-start anchor, so sections 4, 4A, 7, 8, 8A and 10 are invisible
   until this prefix is stripped.

2. Each page's numbered footnote citations (bottom of page, e.g.
   "4. The Explanation I omitted by s. 2, ibid. (w.e.f. 2-10-1985).")
   themselves match the "N. Title...-" shape and, left in place, corrupt
   body-boundary detection for the section immediately before them — text
   that continues on the next physical page gets silently dropped. These
   lines are structurally deleted before section matching, not just
   filtered from the output the way hindu_marriage_act.py's _EDITORIAL_RE
   alone would do.

3. Section 8's title ("Offences to be cognizable for certain purposes and
   to be bailable and non-compoundable") is extracted by pymupdf as one
   word per line (a justified-text rendering artifact unique to this PDF),
   and the section number sits alone on its own line above the title.
   Both are collapsed back into a single line before matching.

Verified directly against the real extracted PDF text (pymupdf): this
pipeline produces exactly the 13 real sections (1, 2, 3, 4, 4A, 5, 6, 7, 8,
8A, 8B, 9, 10) with correct titles and correct body text.
"""

import re

SECTOR   = "matrimonial"
SOURCE   = "Dowry_Prohibition_Act_1961"
DOC_TYPE = "legislation"

# The Act isn't divided into chapters in the bare-act text, unlike HMA/PWDVA.
_CHAPTER_LABEL = "Dowry Prohibition Act, 1961"

SECTION_RE = re.compile(
    r"(?m)^(\d{1,2}[A-Z]?)\. ([^\n]{3,120}?)(?:\.[-—–]+|\.(?=\s*\())"
)

# Broader than HMA's _EDITORIAL_RE: also catches this Act's specific footnote
# phrasing. Kept as a defensive post-filter IN ADDITION to the line-level
# _FOOTNOTE_LINE_RE deletion below (belt and suspenders).
_EDITORIAL_RE = re.compile(
    r"^(?:Ins\b|Subs\b|Sub\b|Omitted\b|Rep\b|Clauses?\b|The\s+words?\b"
    r"|Now\s+see\b|See\s+now\b|Added\b|Substituted\b|\d+\s+of\s+\d+"
    r"|The\s+Explanation\b|Section\s+\d+\s+renumbered\b|The\s+proviso\b"
    r"|Sub-section\s+\(\d+\)\s+renumbered\b)",
    re.IGNORECASE,
)

# Deletes whole footnote-citation lines (bottom-of-page amendment history)
# BEFORE section matching. Left in place, these fake "N. Title.--" lines
# both appear as bogus extra sections AND truncate the real preceding
# section's body at the point they occur (see module docstring point 2).
_FOOTNOTE_LINE_RE = re.compile(
    r"(?m)^\d{1,2}\.\s+(?:Subs\.|Ins\.|Omitted\b|Rep\.|The\s+Explanation\b|The\s+proviso\b"
    r"|Section\s+\d+\s+renumbered\b|Sub-section\s+\(\d+\)\s+renumbered\b"
    r"|\d{1,2}(?:st|nd|rd|th)\s+\w+,?\s+\d{4}).*$\n?"
)

# Strips the leading footnote-reference digit that amending Acts wrap real
# section headers in, e.g. "1[4. Penalty for demanding dowry.--" -> "4. Penalty
# for demanding dowry.--". The lookahead requires what follows to look like an
# actual section-number token, so it does NOT touch inline sub-section
# markers elsewhere in the body text.
_BRACKET_PREFIX_RE = re.compile(r"(?m)^\d{1,2}\[(?=\d{1,2}[A-Z]?\.\s)")

# Collapses runs of 2+ consecutive single-bare-word lines — the justified-
# text rendering artifact affecting Section 8's long title (see docstring
# point 3). Deliberately narrow (letters only, no digits/punctuation) so it
# can't accidentally swallow real numbered sub-clauses like "(a)\n(b)".
_WRAPPED_WORD_RUN_RE = re.compile(r"(?:^[A-Za-z]+ ?\n){2,}", re.MULTILINE)

# A section-number line with nothing else on it before its title starts on
# the next line (also part of Section 8's artifact) — join it forward.
_LONE_SECTION_NUM_RE = re.compile(r"(?m)^(\d{1,2}[A-Z]?)\.\s*\n(?=\S)")

_TITLE_SYNONYMS: dict[str, str] = {
    "3":  "dowry given or taken, family gave dowry",
    "4":  "in-laws demanding dowry, husband asking for money or gold, dowry harassment",
    "4A": "dowry advertisement",
    "8A": "burden of proof in dowry case, who has to prove dowry was not demanded",
}


def _enrich_title(sec_num: str, title: str) -> str:
    extra = _TITLE_SYNONYMS.get(sec_num)
    return f"{title} ({extra})" if extra else title


def clean(raw: str) -> str:
    raw = raw.replace("’", "'").replace("‘", "'")
    raw = raw.replace("“", '"').replace("”", '"')
    raw = raw.replace("–", "-").replace("—", "-")
    raw = raw.replace("ﬁ", "fi").replace("ﬂ", "fl")
    raw = raw.replace("�", "")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = raw.strip()

    # Order matters: delete footnote lines first (so they can't feed the
    # word-run collapse or the bracket-prefix strip), then un-hide bracketed
    # section headers, then fix the Section-8-only line-wrap artifact.
    raw = _FOOTNOTE_LINE_RE.sub("", raw)
    raw = _BRACKET_PREFIX_RE.sub("", raw)
    raw = _WRAPPED_WORD_RUN_RE.sub(lambda m: " ".join(m.group(0).split()) + " ", raw)
    raw = _LONE_SECTION_NUM_RE.sub(r"\1. ", raw)

    return raw.strip()


def _sub_chunk(text: str, max_chars: int = 2000) -> list:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?=\n\(\d+\)|\n\([a-z]\))", text)
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def parse(text: str) -> list:
    matches = list(SECTION_RE.finditer(text))
    print(f"    [Dowry] Section headers found: {len(matches)}")

    sections = []
    for i, m in enumerate(matches):
        sec_num   = m.group(1).strip()
        sec_title = re.sub(r"\s+", " ", m.group(2).strip())

        if _EDITORIAL_RE.search(sec_title):
            continue

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        chunks = _sub_chunk(body, max_chars=2000)
        display_section_title = _enrich_title(sec_num, sec_title)

        for chunk_idx, chunk_text in enumerate(chunks):
            full_text = f"{sec_num}. {sec_title}.\n{chunk_text}"

            record = {
                "doc_id"        : f"DOWRY_{sec_num}" + (f"_part{chunk_idx+1}" if len(chunks) > 1 else ""),
                "section_number": sec_num,
                "section_title" : display_section_title,
                "chapter"       : _CHAPTER_LABEL,
                "content"       : full_text,
                "source"        : SOURCE,
                "sector"        : SECTOR,
                "doc_type"      : DOC_TYPE,
            }
            if len(chunks) > 1:
                record["part"]        = chunk_idx + 1
                record["total_parts"] = len(chunks)

            sections.append(record)

    return sections
