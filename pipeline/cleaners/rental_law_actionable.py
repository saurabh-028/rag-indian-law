"""
Cleaner for: rental_law_actionable_reference.json
Sector: rental_law

Converts the structured JSON into FAISS-compatible chunks.
Same pattern as traffic_actionable.py with sector-specific fields.
"""

import json
from typing import List, Dict

SECTOR   = "rental_law"
SOURCE   = "Maharashtra_Rental_Law_Actionable_Reference"
DOC_TYPE = "actionable_procedure"


def parse_from_json(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for mechanism in data.get("grievance_mechanisms", []):
        mech_id    = mechanism["id"]
        title      = mechanism["title"]
        category   = mechanism["category"]
        when       = mechanism.get("when_to_use", "")
        steps      = mechanism.get("steps", [])
        docs       = mechanism.get("required_documents", [])
        notes      = mechanism.get("important_notes", [])
        legal      = mechanism.get("legal_basis", "")
        source_ref = mechanism.get("source", SOURCE)
        timeline   = mechanism.get("resolution_timeline", "")

        # Main chunk
        content_parts = [f"## {title}", ""]
        if when:
            content_parts += [f"When to use: {when}", ""]
        if timeline:
            content_parts += [f"Resolution timeline: {timeline}", ""]
        if steps:
            content_parts.append("How to do it:")
            for step in steps:
                content_parts.append(f"  - {step}")
            content_parts.append("")
        if docs:
            content_parts.append("Required documents:")
            for doc in docs:
                content_parts.append(f"  - {doc}")
            content_parts.append("")
        if notes:
            content_parts.append("Important notes:")
            for note in notes:
                content_parts.append(f"  - {note}")
            content_parts.append("")
        if isinstance(legal, str) and legal:
            content_parts.append(f"Legal basis: {legal}")
        elif isinstance(legal, list):
            content_parts.append("Legal basis:")
            for l in legal:
                content_parts.append(f"  - {l}")

        records.append({
            "doc_id"         : f"{SOURCE}_{mech_id}",
            "section_number" : mech_id,
            "section_title"  : title,
            "chapter"        : category,
            "content"        : "\n".join(content_parts),
            "source"         : source_ref,
            "sector"         : SECTOR,
            "doc_type"       : DOC_TYPE,
        })

        # Court/forum info chunk
        court = mechanism.get("court_forum", {})
        if court:
            court_parts = [f"## Where to file: {title}", ""]
            for key, val in court.items():
                label = key.replace("_", " ").title()
                court_parts.append(f"  - {label}: {val}")
            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_courts",
                "section_number" : f"{mech_id}_COURTS",
                "section_title"  : f"Court/forum for — {title}",
                "chapter"        : "court_forum",
                "content"        : "\n".join(court_parts),
                "source"         : source_ref,
                "sector"         : SECTOR,
                "doc_type"       : "contact_reference",
            })

        # Appeal hierarchy chunk
        appeals = mechanism.get("appeal_hierarchy", [])
        if appeals:
            app_parts = [f"## Appeal process: {title}", ""]
            for a in appeals:
                app_parts.append(f"  - {a}")
            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_appeals",
                "section_number" : f"{mech_id}_APPEALS",
                "section_title"  : f"Appeal hierarchy — {title}",
                "chapter"        : "appeal_process",
                "content"        : "\n".join(app_parts),
                "source"         : source_ref,
                "sector"         : SECTOR,
                "doc_type"       : DOC_TYPE,
            })

        # Eviction grounds chunk
        grounds = mechanism.get("grounds_for_eviction", [])
        if grounds:
            g_parts = ["## Legal grounds for eviction under MRCA 1999", ""]
            for g in grounds:
                g_parts.append(f"  - {g}")
            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_grounds",
                "section_number" : f"{mech_id}_GROUNDS",
                "section_title"  : "Legal grounds for eviction — Maharashtra",
                "chapter"        : "landlord_rights",
                "content"        : "\n".join(g_parts),
                "source"         : source_ref,
                "sector"         : SECTOR,
                "doc_type"       : "citizen_rights",
            })

        # Helpline/contact chunk
        helpline = mechanism.get("helpline", {})
        if helpline:
            help_parts = [f"## Contact information: {title}", ""]
            for key, val in helpline.items():
                label = key.replace("_", " ").title()
                help_parts.append(f"  - {label}: {val}")
            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_contacts",
                "section_number" : f"{mech_id}_CONTACTS",
                "section_title"  : f"Contact info — {title}",
                "chapter"        : "contacts",
                "content"        : "\n".join(help_parts),
                "source"         : source_ref,
                "sector"         : SECTOR,
                "doc_type"       : "contact_reference",
            })

        # Escalation options chunk
        options = mechanism.get("options", [])
        if options:
            for opt in options:
                opt_parts = [f"## {opt['method']}", ""]
                if opt.get("when"):
                    opt_parts.append(f"When to use: {opt['when']}")
                    opt_parts.append("")
                if opt.get("portal_url"):
                    opt_parts.append(f"Portal: {opt['portal_url']}")
                if opt.get("url"):
                    opt_parts.append(f"Portal: {opt['url']}")
                for s in opt.get("steps", []):
                    opt_parts.append(f"  - {s}")

                records.append({
                    "doc_id"         : f"{SOURCE}_{mech_id}_{opt['method'].replace(' ', '_')[:30]}",
                    "section_number" : f"{mech_id}_{opt['method'][:20]}",
                    "section_title"  : f"Escalation — {opt['method']}",
                    "chapter"        : "escalation",
                    "content"        : "\n".join(opt_parts),
                    "source"         : source_ref,
                    "sector"         : SECTOR,
                    "doc_type"       : DOC_TYPE,
                })

        # RTI sample questions chunk
        rti_qs = mechanism.get("sample_questions", [])
        if rti_qs:
            rti_parts = ["## RTI sample questions for rent/housing disputes", ""]
            for q in rti_qs:
                rti_parts.append(f"  - {q}")
            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_rti",
                "section_number" : f"{mech_id}_RTI",
                "section_title"  : "RTI sample questions for rent/housing disputes",
                "chapter"        : "information_right",
                "content"        : "\n".join(rti_parts),
                "source"         : "RTI Act 2005 + Maharashtra RTI Portal",
                "sector"         : SECTOR,
                "doc_type"       : DOC_TYPE,
            })

    return records


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/rental_law/actionable/rental_law_actionable_reference.json"
    chunks = parse_from_json(path)
    print(f"\nParsed {len(chunks)} chunks from rental law actionable reference")
    for c in chunks:
        print(f"  [{c['doc_type']:>22}] {c['doc_id'][:55]:55} | {c['section_title'][:55]}")
