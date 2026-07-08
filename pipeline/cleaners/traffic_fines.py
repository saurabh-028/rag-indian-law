"""
Cleaner for: traffic violation fines.pdf (table-based PDF)
Sector: traffic

PDF column layout (verified):
  0: Sr.No
  1: Offense Section  (e.g. "Sec 129/194(D) MVA")
  2: Offense Name English
  3: Offense Name Marathi  — skipped
  4: Fine
  5: Repetitive Fine

build_index.py calls parse(pdf_path) directly because table_based=True.
"""

import re
from pathlib import Path
from typing import Union

import pdfplumber

DOC_ID = "TRAFFIC_FINES"
SOURCE = "traffic violation fines.pdf"
SECTOR = "traffic"
DOC_TYPE = "fine_schedule"

# Vocabulary bridge: maps substrings in the PDF's offense name to additional
# synonym keywords appended to section_title.  This closes query-corpus gaps
# like "red light" (user query) vs "signal" (PDF terminology), "drunk driving"
# vs "Drunk and Drive", "no helmet" vs "without helmet" etc.
_TITLE_SYNONYMS: list[tuple[str, str]] = [
    ("signal",       "red light traffic signal jumping"),
    ("helmet",       "no helmet protective headgear"),
    ("drunk",        "drunk driving drink drive DUI alcohol"),
    ("license",      "licence driving license without license"),
    ("without valid license", "no license invalid license"),
    ("seat belt",    "seatbelt safety belt"),
    ("speed",        "speeding over speed"),
    ("mobile",       "phone mobile phone using phone"),
    ("parking",      "no parking illegal parking"),
    ("number plate", "no number plate no plate"),
    ("overload",     "overloaded vehicle"),
    ("triple",       "triple riding three on bike"),
    ("pillion",      "pillion rider helmet"),
    ("insurance",    "no insurance without insurance"),
]


def _enrich_title(offense: str, sr_no: int) -> str:
    """Append synonym keywords to section_title for better BM25 recall."""
    if not offense:
        return f"Row {sr_no}"
    base = f"{offense} challan fine"
    extras = []
    offense_lower = offense.lower()
    for keyword, synonyms in _TITLE_SYNONYMS:
        if keyword in offense_lower:
            extras.append(synonyms)
    return f"{base} {' '.join(extras)}" if extras else base

_COL_SR_NO   = 0
_COL_SECTION = 1
_COL_OFFENSE = 2
# _COL_MARATHI = 3  (skipped)
_COL_FINE    = 4
_COL_REP     = 5


def clean(raw_text: str = "") -> str:
    """No-op — table PDFs are parsed directly by parse()."""
    return raw_text.strip()


def parse(pdf_path: Union[str, Path]) -> list[dict]:
    """Extract all table rows from the fine schedule PDF."""
    pdf_path = Path(pdf_path)
    chunks: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
                for row in table:
                    if not row or not row[_COL_SR_NO]:
                        continue
                    sr_str = str(row[_COL_SR_NO]).strip()
                    if not sr_str.isdigit():
                        continue

                    sr_no   = int(sr_str)
                    section = _cell(row, _COL_SECTION)
                    offense = _cell(row, _COL_OFFENSE)
                    fine    = _cell(row, _COL_FINE)
                    rep     = _cell(row, _COL_REP)

                    # Strip PDF encoding artifacts
                    offense = re.sub(r"\(cid:\d+\)", "", offense)
                    fine    = re.sub(r"\(cid:\d+\)", "", fine)
                    rep     = re.sub(r"\(cid:\d+\)", "", rep)

                    offense = re.sub(r"\s+", " ", offense).strip()
                    fine    = re.sub(r"\s+", " ", fine).strip()
                    rep     = re.sub(r"\s+", " ", rep).strip()

                    if not offense and not fine:
                        continue

                    fine_str = f"Rs. {fine}" if fine and not fine.lower().startswith("rs") else fine
                    rep_str  = f"Rs. {rep}"  if rep  and not rep.lower().startswith("rs")  else rep
                    # Keep content minimal and offense-centric so InLegalBERT
                    # embeds each chunk distinctly (shared boilerplate previously
                    # caused all fine-schedule chunks to collapse to the same
                    # dense vector, making dense retrieval useless for these).
                    # "challan" = Indian English for traffic fine notice.
                    # Including it here ensures BM25 matches queries like
                    # "challan for drunk driving" to fine_schedule chunks,
                    # not only to actionable-procedure chunks about challenging challans.
                    content = f"{offense}. Traffic challan and fine under section {section}. Fine amount: {fine_str}."
                    if rep and rep.lower() not in ("na", ""):
                        content += f" Repeat offender fine: {rep_str}."

                    chunks.append({
                        "doc_id":           DOC_ID,
                        "section_number":   section or str(sr_no),
                        # Append domain-synonym keywords to section_title so BM25
                        # title-boosting bridges vocabulary gaps between user queries
                        # and the PDF's terminology (e.g. "red light" vs "signal").
                        "section_title":    _enrich_title(offense, sr_no),
                        "chapter":          f"Page {page_num}",
                        "content":          content,
                        "source":           SOURCE,
                        "sector":           SECTOR,
                        "doc_type":         DOC_TYPE,
                        "sr_no":            sr_no,
                        "violation_type":   offense,
                        "section_reference": section,
                        "fine_amount":      fine,
                        "repetitive_fine":  rep,
                        "penalty_points":   "",
                    })

    if not chunks:
        chunks = _fallback_text_parse(pdf_path)

    return chunks


def _cell(row: list, idx: int) -> str:
    """Safely get a cell value, returning empty string if missing."""
    if idx >= len(row) or row[idx] is None:
        return ""
    return str(row[idx]).strip()


def _fallback_text_parse(pdf_path: Path) -> list[dict]:
    """Fallback when no tables are detected — treats each text line as a chunk."""
    chunks = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for line_num, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                chunks.append({
                    "doc_id":           DOC_ID,
                    "section_number":   f"p{page_num}l{line_num}",
                    "section_title":    line[:80],
                    "chapter":          f"Page {page_num}",
                    "content":          line,
                    "source":           SOURCE,
                    "sector":           SECTOR,
                    "doc_type":         DOC_TYPE,
                    "sr_no":            None,
                    "violation_type":   None,
                    "section_reference": None,
                    "fine_amount":      None,
                    "repetitive_fine":  None,
                    "penalty_points":   None,
                })
    return chunks
