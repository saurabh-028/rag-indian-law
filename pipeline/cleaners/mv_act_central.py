"""
Cleaner for: Central Motor Vehicle act(amended till 2019).pdf
Sector: traffic

Section header format: "Section 123 - Title of section"
"""

import re

PAGE_HEADER = "MOTOR VEHICLES ACT, 1988 (Amended Upto 2019)"
FOOTNOTE_RE = re.compile(r"\[\d+[A-Z]?\]")

SECTION_RE = re.compile(
    r"^(Section\s+(\d+[A-Z]?)\s*[-–]\s*(.+))$",
    re.IGNORECASE | re.MULTILINE,
)

SECTOR   = "traffic"
SOURCE   = "Central_Motor_Vehicles_Act_1988"
DOC_TYPE = "legislation"


def clean(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []

    for line in lines:
        if line.strip() == PAGE_HEADER:
            continue
        if line.strip().startswith("Disclaimer:"):
            break
        if re.fullmatch(r"\[\d+[A-Z]?\]", line.strip()):
            continue
        line = FOOTNOTE_RE.sub("", line)
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _sub_chunk(text: str, max_chars: int = 3000) -> list:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?=\n\(\d+\))", text)
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
    matches = list(SECTION_RE.finditer(text))
    print(f"    [MV Act Central] Section headers found: {len(matches)}")

    if not matches:
        raise ValueError("No section headers found in Central MV Act — check PDF text.")

    sections = []

    # Preamble before Section 1
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({
            "doc_id"        : f"{SOURCE}_Preamble",
            "section_number": "Preamble",
            "section_title" : "Preamble",
            "chapter"       : "Preamble",
            "content"       : preamble,
            "source"        : SOURCE,
            "sector"        : SECTOR,
            "doc_type"      : DOC_TYPE,
        })

    for idx, match in enumerate(matches):
        heading   = match.group(1).strip()
        sec_num   = match.group(2).strip()
        sec_title = match.group(3).strip()

        c_start = match.end()
        c_end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[c_start:c_end].strip()

        subs = _sub_chunk(content)

        for si, chunk in enumerate(subs):
            record = {
                "doc_id"        : f"{SOURCE}_S{sec_num}" + (f"_part{si+1}" if len(subs) > 1 else ""),
                "section_number": sec_num,
                "section_title" : sec_title,
                "chapter"       : heading,
                "content"       : f"{heading}\n{chunk}",
                "source"        : SOURCE,
                "sector"        : SECTOR,
                "doc_type"      : DOC_TYPE,
            }
            if len(subs) > 1:
                record["part"]        = si + 1
                record["total_parts"] = len(subs)
            sections.append(record)

    return sections
