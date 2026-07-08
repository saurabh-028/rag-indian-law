"""
Cleaner for: Bhartiya_Nyay_Sanhita(BNS).pdf
Sector: criminal_law

Structure:
  - Pages 1-15: Title/ToC (skipped)
  - Page 16+:   Content
  - Chapters:   "CHAPTER I", "CHAPTER XIV" etc (all caps, title on next line)
  - Sections:   "123. Section Title.--Body text..."

Notes on PDF quirks:
  - Some section titles wrap to a second line (e.g. Section 62, 102, 119).
  - Some separators are ". —" (period + space + em-dash) rather than ".—".
  - Both are normalised after clean() runs (em-dashes → hyphens).
  - SECTION_RE handles these via an optional continuation line and r"\.[ ]?-" separator.
"""

import re

SECTOR   = "criminal_law"
SOURCE   = "Bharatiya_Nyaya_Sanhita_2023"
DOC_TYPE = "legislation"

CHAPTER_RE = re.compile(
    r"(?m)^(CHAPTER\s+[IVXLC]+)\s*\n([A-Z][A-Z\s,'-]{3,80})"
)

# Matches BNS section headers after clean() has run (em-dashes → hyphens).
#
# Two quirks handled:
#   1. Multi-line titles: some long titles wrap to a second line before the separator.
#      The optional  (?:\n[^\n]{1,100}?)?  captures that continuation line (lazy).
#   2. Space before separator: ". —" becomes ". -" after clean(), so the separator
#      pattern allows an optional space: \.[ ]?-{1,2}
#
# Both title parts are LAZY so the regex always finds the SHORTEST valid title
# before the separator, preventing one section's title from bleeding into the next.
SECTION_RE = re.compile(
    r"(?m)"
    r"^(\d{1,3}[A-Z]?)\.\s+"                  # section number  e.g. "63. "
    r"([^\n]{1,120}?(?:\n[^\n]{1,100}?)?)"     # title: lazy first line + optional lazy continuation
    r"(?:\.[ ]?-{1,2}|\u2014|\u2013|\u2012|-{2})",  # separator
)

# Detects punishment provisions inside a sub-chunk so we can annotate the title.
_PUNISHMENT_RE = re.compile(
    r"(?:shall|liable)\s+be\s+punished\b|punishable\s+with\b",
    re.IGNORECASE,
)


def clean(raw: str) -> str:
    # Normalise smart quotes
    raw = raw.replace("\u2019", "'").replace("\u2018", "'")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')  # curly double-quotes

    # Convert em-dash / en-dash / figure-dash to a plain hyphen so the
    # post-clean separator pattern (\.[ ]?-{1,2}) can match them.
    raw = raw.replace("\u2013", "-").replace("\u2014", "-")
    raw = raw.replace("\u2012", "-")

    # Fix common OCR ligatures
    raw = raw.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Remove footnote markers (superscript digits directly after a word character,
    # followed by whitespace).
    raw = re.sub(r"(?<=\w)(\d)(?=\s)", "", raw)

    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


def _build_chapter_map(text: str) -> list:
    chapters = []
    for m in CHAPTER_RE.finditer(text):
        label = f"{m.group(1).strip()} — {m.group(2).strip()}"
        chapters.append((m.start(), label))
    return chapters


def _get_chapter(offset: int, chapter_map: list) -> str:
    chapter = "Unknown"
    for pos, label in chapter_map:
        if pos <= offset:
            chapter = label
        else:
            break
    return chapter


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
    chapter_map = _build_chapter_map(text)
    matches     = list(SECTION_RE.finditer(text))
    print(f"    [BNS] Section headers found: {len(matches)}")

    sections = []

    for i, m in enumerate(matches):
        sec_num = m.group(1).strip()
        # Normalise multi-line titles: collapse internal whitespace/newlines.
        sec_title = re.sub(r"\s+", " ", m.group(2).strip())

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        chapter = _get_chapter(m.start(), chapter_map)
        chunks  = _sub_chunk(body, max_chars=2000)

        for chunk_idx, chunk_text in enumerate(chunks):
            # For multi-part sections, annotate sub-chunks that contain penalty
            # provisions so that punishment-related queries retrieve them reliably.
            if len(chunks) > 1 and _PUNISHMENT_RE.search(chunk_text):
                display_title = f"{sec_title} — Penalty"
            else:
                display_title = sec_title

            full_text = f"{sec_num}. {display_title}.\n{chunk_text}"

            record = {
                "doc_id"        : f"BNS_{sec_num}" + (f"_part{chunk_idx+1}" if len(chunks) > 1 else ""),
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
