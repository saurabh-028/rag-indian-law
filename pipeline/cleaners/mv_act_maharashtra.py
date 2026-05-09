"""
Cleaner for: Maharashtra Motor Vehicle act 1989.pdf
Sector: traffic
"""

import re
from dataclasses import dataclass, field
from typing import Optional

SECTOR   = "traffic"
SOURCE   = "Maharashtra_Motor_Vehicles_Rules_1989"
DOC_TYPE = "legislation"

RULE_HEADER_RE = re.compile(
    r"^(\d{1,4}[A-Z]?(?:-\d+)?)\.\s+\S",
    re.MULTILINE,
)
FOOTNOTE_RE = re.compile(r"^\d+\.\s+(Sub\.|Rule|Added|Inserted)", re.IGNORECASE)
AMENDMENT_RE = re.compile(r"(\d+\.\s+(Sub|Added|Inserted)[^\n]*)")


def clean(raw: str) -> str:
    raw = raw.replace("\ufb01", "fi").replace("\ufb02", "fl")
    raw = raw.replace("\x0c", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw


def _is_footnote(text: str) -> bool:
    first_line = text.split("\n")[0]
    return bool(FOOTNOTE_RE.match(first_line))


def _extract_amendments(text: str):
    amendments = []

    def remove_fn(m):
        amendments.append(m.group(0))
        return ""

    cleaned = AMENDMENT_RE.sub(remove_fn, text)
    return cleaned.strip(), amendments


def _build_chapter_map(text: str) -> list:
    chapters = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^CHAPTER\s+([A-Za-z]{1,5})\s*$", line.strip())
        if m:
            roman = m.group(1)
            subtitle = ""
            for j in range(i + 1, min(i + 4, len(lines))):
                if lines[j].strip():
                    subtitle = lines[j].strip()
                    break
            label  = f"Chapter {roman} — {subtitle}"
            offset = sum(len(l) + 1 for l in lines[:i])
            chapters.append((offset, label))
    return chapters


def _get_chapter(offset: int, chapter_map: list) -> str:
    chapter = "Unknown"
    for pos, label in chapter_map:
        if pos <= offset:
            chapter = label
        else:
            break
    return chapter


def parse(text: str) -> list:
    chapter_map = _build_chapter_map(text)
    matches     = list(RULE_HEADER_RE.finditer(text))

    records = []

    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body  = text[start:end].strip()

        if _is_footnote(body):
            continue

        rule_num   = m.group(1)
        first_line = body.split("\n")[0]
        title_text = re.sub(r"^\d{1,4}[A-Z]?(?:-\d+)?\.\s+", "", first_line)
        title      = re.split(r"\s*[-—]\s*", title_text)[0].strip()[:120]
        chapter    = _get_chapter(start, chapter_map)
        clean_body, amendments = _extract_amendments(body)

        records.append({
            "doc_id"        : f"{SOURCE}_R{rule_num}",
            "section_number": rule_num,
            "section_title" : title,
            "chapter"       : chapter,
            "content"       : clean_body,
            "amendments"    : amendments,
            "source"        : SOURCE,
            "sector"        : SECTOR,
            "doc_type"      : DOC_TYPE,
        })

    return records
