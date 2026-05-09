"""
Cleaner for: eng_maharashtra_rent_control_ac.pdf
Sector: rental_law
"""

import re

SECTOR   = "rental_law"
SOURCE   = "Maharashtra_Rent_Control_Act_1999"
DOC_TYPE = "legislation"


def clean(raw: str) -> str:
    raw = re.sub(r'2000\s*:\s*Mah\.\s*XVIII.*', '', raw)
    raw = re.sub(r'The Maharashtra Rent Control Act, 1999', '', raw)
    raw = re.sub(r'^\s*\d+\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\n+', '\n', raw)
    raw = re.sub(r' +', ' ', raw)
    return raw.strip()


def _remove_toc(text: str) -> str:
    start_match = re.search(r'MAHARASHTRA ACT No\.', text)
    if start_match:
        text = text[start_match.start():]
    return text


def _extract_chapters(text: str) -> list:
    chapter_pattern = r'(CHAPTER\s+[IVXLC]+[\s\S]*?)(?=CHAPTER\s+[IVXLC]+|$)'
    return re.findall(chapter_pattern, text)


def parse(text: str) -> list:
    text = _remove_toc(text)
    chapters = _extract_chapters(text)
    print(f"    [Rent Control] Chapters found: {len(chapters)}")

    structured_data = []

    for chapter in chapters:
        chapter_title_match = re.search(r'(CHAPTER\s+[IVXLC]+)', chapter)
        chapter_title = chapter_title_match.group(1) if chapter_title_match else "Unknown Chapter"

        sections = re.split(r'\n(?=\d+\.\s)', chapter)

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            match = re.match(r'(\d+)\.\s+(.*)', sec)
            if match:
                section_number = match.group(1)
                section_title  = match.group(2).split('—')[0][:150].strip()

                structured_data.append({
                    "doc_id"        : f"{SOURCE}_S{section_number}",
                    "section_number": section_number,
                    "section_title" : section_title,
                    "chapter"       : chapter_title,
                    "content"       : sec,
                    "source"        : SOURCE,
                    "sector"        : SECTOR,
                    "doc_type"      : DOC_TYPE,
                })

    return structured_data
