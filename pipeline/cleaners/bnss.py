"""
Cleaner for: the_bharatiya_nagarik_suraksha_sanhita,_2023(BNSS).pdf
Sector: criminal_law

Structure:
  - Pages 1-15:   Title/ToC (skipped)
  - Pages 16-172: Content sections
  - Pages 173+:   First Schedule (classification table, skipped)
  - Chapters:     "CHAPTER I", "CHAPTER XXXVII" etc (all caps, title on next line)
  - Sections:     "35. Title.- Body"
"""

import re

SECTOR   = "criminal_law"
SOURCE   = "Bharatiya_Nagarik_Suraksha_Sanhita_2023"
DOC_TYPE = "legislation"

CONTENT_START = 15   # page index 15 = page 16 (first content page)
CONTENT_END   = 172  # page index 171 = page 172 (last section page)

CHAPTER_RE = re.compile(
    r"(?m)^(CHAPTER\s+[IVXLC]+)\s*\n\s*([A-Z][A-Z0-9 ,.'\-]{3,100})"
)

SECTION_RE = re.compile(
    r"(?m)^(\d{1,3}[A-Z]?)\.\s+([^\n]{3,120}?)(?:\.\s*-{1,2}|\u2014|\u2013|-{1,2})"
)


def clean(raw: str) -> str:
    text = raw

    text = text.replace("\ufffd", "\u2014")  # PyMuPDF fallback for em-dash

    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2012", "-")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Remove lone page number lines
    text = re.sub(r"^\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Remove running page header
    text = re.sub(
        r"^THE BHARATIYA NAGARIK SURAKSHA SANHITA,?\s*2023\s*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    text = re.sub(r"^ACT NO\.\s*46 OF 2023\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\[25th December,\s*2023\.\]\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _build_chapter_map(text: str) -> list:
    chapters = []
    for m in CHAPTER_RE.finditer(text):
        label = f"{m.group(1).strip()} - {m.group(2).strip()}"
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

    parts = re.split(
        r"(?=\n\(\d+\)|\n\([a-z]\)|\n\([ivxl]+\)|\nProvided|\nExplanation)",
        text,
    )

    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def parse(text: str) -> list:
    chapter_map = _build_chapter_map(text)
    matches     = list(SECTION_RE.finditer(text))
    print(f"    [BNSS] Chapter headers found: {len(chapter_map)}")
    print(f"    [BNSS] Section headers found: {len(matches)}")

    records = []

    for i, m in enumerate(matches):
        sec_num   = m.group(1).strip()
        sec_title = m.group(2).strip()

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        chapter = _get_chapter(m.start(), chapter_map)
        chunks  = _sub_chunk(body, max_chars=2000)

        for chunk_idx, chunk_text in enumerate(chunks):
            content = f"{sec_num}. {sec_title}.\n{chunk_text}"

            record = {
                "doc_id"        : f"BNSS_{sec_num}" + (f"_part{chunk_idx+1}" if len(chunks) > 1 else ""),
                "section_number": sec_num,
                "section_title" : sec_title,
                "chapter"       : chapter,
                "content"       : content,
                "source"        : SOURCE,
                "sector"        : SECTOR,
                "doc_type"      : DOC_TYPE,
            }
            if len(chunks) > 1:
                record["part"]        = chunk_idx + 1
                record["total_parts"] = len(chunks)

            records.append(record)

    return records
