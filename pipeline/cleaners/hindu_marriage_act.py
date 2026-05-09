"""
Cleaner for: Hindu_Marriage_Act.pdf
Sector: hindu_marriage_laws

Structure:
  - Pages 1-2: Table of contents (skipped)
  - Page 3+:   Content
  - Sections:  "5. Conditions for a Hindu marriage.--Body text..."
"""

import re

SECTOR   = "hindu_marriage_laws"
SOURCE   = "Hindu_Marriage_Act_1955"
DOC_TYPE = "legislation"

SECTION_RE = re.compile(
    r"(?m)^(\d{1,2}[A-Z]?)\. ([^\n]{3,100}?)(?:\.[-\u2014\u2013]+|\.(?=\s*\())"
)


def clean(raw: str) -> str:
    raw = raw.replace("\u2019", "'").replace("\u2018", "'")
    raw = raw.replace("\u2013", "-").replace("\u2014", "-")
    raw = raw.replace("\ufb01", "fi").replace("\ufb02", "fl")
    raw = raw.replace("\ufffd", "")  # remove replacement characters from OCR

    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


def parse(text: str) -> list:
    matches = list(SECTION_RE.finditer(text))
    print(f"    [HMA] Section headers found: {len(matches)}")

    sections = []
    for i, m in enumerate(matches):
        sec_num   = m.group(1).strip()
        sec_title = m.group(2).strip()

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        full_text = f"{sec_num}. {sec_title}.\n{body}"

        sections.append({
            "doc_id"        : f"HMA_{sec_num}",
            "section_number": sec_num,
            "section_title" : sec_title,
            "content"       : full_text,
            "source"        : SOURCE,
            "sector"        : SECTOR,
            "doc_type"      : DOC_TYPE,
        })

    return sections
