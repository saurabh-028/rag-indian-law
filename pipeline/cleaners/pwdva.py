"""
Cleaner for: PWDVA_2005.pdf (Protection of Women from Domestic Violence Act, 2005)
Sector: matrimonial

Structure mirrors hindu_marriage_act.py:
  - Pages 1-2: Table of contents (skipped — TOC titles end in a plain period,
    not the em-dash/paren pattern SECTION_RE requires, so they never match)
  - Page 3+:   Content, section headers like "18. Protection orders.--The Magistrate..."

Verified against the real bare-act PDF (indiacode.nic.in, Act No. 43 of 2005):
  HMA's unmodified SECTION_RE/_EDITORIAL_RE/clean()/_sub_chunk() pattern matches
  all 37 real sections cleanly — no footnote pollution, no duplicate section
  numbers, no body-boundary corruption. Unlike the Dowry Prohibition Act PDF
  (see dowry_prohibition_act.py), this source needed no extra preprocessing.
"""

import re

SECTOR   = "matrimonial"
SOURCE   = "Protection_of_Women_from_Domestic_Violence_Act_2005"
DOC_TYPE = "legislation"

_CHAPTER_MAP = {
    range(1, 3):   "Chapter I — Preliminary",
    range(3, 4):   "Chapter II — Domestic Violence",
    range(4, 12):  "Chapter III — Powers and Duties of Protection Officers, Service Providers, etc.",
    range(12, 30): "Chapter IV — Procedure for Obtaining Orders of Reliefs",
    range(30, 38): "Chapter V — Miscellaneous",
}

SECTION_RE = re.compile(
    r"(?m)^(\d{1,2}[A-Z]?)\. ([^\n]{3,120}?)(?:\.[-—–]+|\.(?=\s*\())"
)

_EDITORIAL_RE = re.compile(
    r"^(?:Ins\b|Subs\b|Sub\b|Omitted\b|Rep\b|Clauses?\b|The\s+words?\b"
    r"|Now\s+see\b|See\s+now\b|Added\b|Substituted\b|\d+\s+of\s+\d+)",
    re.IGNORECASE,
)

# Detects sub-chunks of Section 3 (which enumerates the forms of domestic
# violence covered) for chunk-title clarity when it's long enough to split.
_ABUSE_FORMS_RE = re.compile(
    r"\bphysical\s+abuse\b|\bsexual\s+abuse\b|\bverbal\s+(?:and\s+)?emotional\s+abuse\b"
    r"|\beconomic\s+abuse\b|\bharms?\s+or\s+injures\b",
    re.IGNORECASE,
)

# Vocabulary bridge: BM25 is weighted 2x over dense retrieval and section_title
# is repeated 4x in the BM25 corpus (see app/retriever.py), so appending
# natural-language phrasing a real victim would type — "he hit me", "beats
# me", "kicked me out" — closes the gap to this Act's formal statutory
# language ("domestic violence", "shared household", "protection order").
# Appended as a parenthetical suffix (not a raw keyword dump) so section_title
# still reads like a real legal heading when shown back to the user.
_TITLE_SYNONYMS: dict[str, str] = {
    "3":  "hit, beaten, slapped, punched, abused, threatened, verbal abuse, emotional abuse, economic abuse, physical assault by husband",
    "12": "how to file a domestic violence complaint, approach the magistrate",
    "17": "kicked out of the house, thrown out, right to stay in matrimonial home",
    "18": "restraining order, stop him from hitting me, no-contact order",
    "19": "cannot be evicted, right to stay in the house, alternate accommodation",
    "20": "compensation for medical expenses, financial support, maintenance for domestic violence",
    "21": "child custody in a domestic violence case",
    "22": "compensation for injury or abuse",
    "23": "emergency protection order, urgent order, ex-parte order",
    "31": "he violated the protection order, husband broke the restraining order",
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
    return raw.strip()


def _get_chapter(sec_num_str: str) -> str:
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
    print(f"    [PWDVA] Section headers found: {len(matches)}")

    sections = []
    for i, m in enumerate(matches):
        sec_num   = m.group(1).strip()
        sec_title = re.sub(r"\s+", " ", m.group(2).strip())

        if _EDITORIAL_RE.search(sec_title):
            continue

        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        chapter = _get_chapter(sec_num)
        chunks  = _sub_chunk(body, max_chars=2000)
        display_section_title = _enrich_title(sec_num, sec_title)

        for chunk_idx, chunk_text in enumerate(chunks):
            if len(chunks) > 1 and _ABUSE_FORMS_RE.search(chunk_text):
                display_title = f"{sec_title} — Forms of Abuse"
            else:
                display_title = sec_title

            full_text = f"{sec_num}. {display_title}.\n{chunk_text}"

            record = {
                "doc_id"        : f"PWDVA_{sec_num}" + (f"_part{chunk_idx+1}" if len(chunks) > 1 else ""),
                "section_number": sec_num,
                "section_title" : display_section_title,
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
