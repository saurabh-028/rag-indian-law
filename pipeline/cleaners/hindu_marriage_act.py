"""
Cleaner for: Hindu_Marriage_Act.pdf
Sector: hindu_marriage_laws

Structure:
  - Pages 1-2: Table of contents (skipped)
  - Page 3+:   Content
  - Sections:  "5. Conditions for a Hindu marriage.--Body text..."

Notes:
  - Some sections are long (e.g. Section 13 with all divorce grounds) and are sub-chunked.
  - Separator after clean() is ".-" (em-dashes normalised) or period + "(" for sub-sectioned bodies.
"""

import re

SECTOR   = "matrimonial"
SOURCE   = "Hindu_Marriage_Act_1955"
DOC_TYPE = "legislation"

# Chapter groupings by section number ranges for metadata enrichment
_CHAPTER_MAP = {
    range(1, 4):   "Part I — Preliminary",
    range(4, 9):   "Part II — Conditions and Ceremonies of Marriage",
    range(9, 14):  "Part III — Restitution and Judicial Separation",
    range(13, 24): "Part IV — Divorce",
    range(24, 27): "Part V — Maintenance, Custody and Legitimacy",
    range(27, 31): "Part VI — Jurisdiction and Procedure",
}

SECTION_RE = re.compile(
    r"(?m)^(\d{1,2}[A-Z]?)\. ([^\n]{3,100}?)(?:\.[-\u2014\u2013]+|\.(?=\s*\())"
)

# Detects divorce/matrimonial ground provisions for sub-chunk annotation
_GROUNDS_RE = re.compile(
    r"\bgrounds?\s+for\s+divorce\b|adultery|cruelty|desertion|unsound\s+mind"
    r"|venereal\s+disease|leprosy|renounced\s+the\s+world|not\s+heard\s+of",
    re.IGNORECASE,
)

# Editorial footnotes in the HMA PDF are numbered (1., 2., 3. …) and look
# like section headers when extracted by pymupdf.  Skip any match whose
# captured title starts with these tell-tale patterns.
_EDITORIAL_RE = re.compile(
    r"^(?:Ins\b|Subs\b|Sub\b|Omitted\b|Rep\b|Clauses?\b|The\s+words?\b"
    r"|Now\s+see\b|See\s+now\b|Added\b|Substituted\b|\d+\s+of\s+\d+)",
    re.IGNORECASE,
)


def clean(raw: str) -> str:
    raw = raw.replace("\u2019", "'").replace("\u2018", "'")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')  # curly double-quotes
    raw = raw.replace("\u2013", "-").replace("\u2014", "-")
    raw = raw.replace("\ufb01", "fi").replace("\ufb02", "fl")
    raw = raw.replace("\ufffd", "")  # remove replacement characters from OCR

    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


def _get_chapter(sec_num_str: str) -> str:
    """Return chapter label for a section number string like '13' or '13A'."""
    digits = re.match(r"\d+", sec_num_str)
    if not digits:
        return "Unknown"
    n = int(digits.group())
    for rng, label in _CHAPTER_MAP.items():
        if n in rng:
            return label
    return "Unknown"


def _sub_chunk(text: str, max_chars: int = 2000) -> list:
    if len(text) <= max_chars:
        return [text]

    # Split at numbered sub-clauses: (1), (2)... or (a), (b)...
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
    print(f"    [HMA] Section headers found: {len(matches)}")

    sections = []
    for i, m in enumerate(matches):
        sec_num   = m.group(1).strip()
        sec_title = re.sub(r"\s+", " ", m.group(2).strip())

        # Skip editorial footnotes that pymupdf extracts as numbered paragraphs.
        # These look like section headers but are amendment annotations, e.g.:
        #   "3. Ins. by s. 16, ibid.--" or "6. Clauses (viii)…omitted by s. 2"
        if _EDITORIAL_RE.search(sec_title):
            continue

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        chapter = _get_chapter(sec_num)
        chunks  = _sub_chunk(body, max_chars=2000)

        for chunk_idx, chunk_text in enumerate(chunks):
            # Annotate sub-chunks that contain divorce/matrimonial grounds
            if len(chunks) > 1 and _GROUNDS_RE.search(chunk_text):
                display_title = f"{sec_title} — Grounds"
            else:
                display_title = sec_title

            full_text = f"{sec_num}. {display_title}.\n{chunk_text}"

            record = {
                "doc_id"        : f"HMA_{sec_num}" + (f"_part{chunk_idx+1}" if len(chunks) > 1 else ""),
                "section_number": sec_num,
                "section_title" : sec_title,
                "chapter"       : chapter,
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
