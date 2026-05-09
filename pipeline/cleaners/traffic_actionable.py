"""
Cleaner for: traffic_actionable_reference.json
Sector: traffic

Converts the structured JSON (helplines, step-by-step procedures, grievance mechanisms)
into chunks that match the same schema as legislation chunks so they sit together
in the unified FAISS index.
"""

import json
from typing import List, Dict

SECTOR   = "traffic"
SOURCE   = "Maharashtra_Traffic_Actionable_Reference"
DOC_TYPE = "actionable_procedure"


def parse_from_json(json_path: str) -> List[Dict]:
    """Read the actionable reference JSON and return a list of FAISS-compatible chunks."""
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

        # Main chunk: title + when to use + steps
        content_parts = []
        content_parts.append(f"## {title}")
        content_parts.append("")

        if when:
            content_parts.append(f"When to use: {when}")
            content_parts.append("")

        if timeline:
            content_parts.append(f"Resolution timeline: {timeline}")
            content_parts.append("")

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

        main_content = "\n".join(content_parts)

        records.append({
            "doc_id"         : f"{SOURCE}_{mech_id}",
            "section_number" : mech_id,
            "section_title"  : title,
            "chapter"        : category,
            "content"        : main_content,
            "source"         : source_ref,
            "sector"         : SECTOR,
            "doc_type"       : DOC_TYPE,
        })

        # Helpline/contact chunk
        helpline = mechanism.get("helpline", {})
        channels = mechanism.get("complaint_channels", [])

        if helpline or channels:
            help_parts = [f"## Contact information: {title}", ""]

            if isinstance(helpline, dict):
                for key, val in helpline.items():
                    label = key.replace("_", " ").title()
                    help_parts.append(f"  - {label}: {val}")
            elif isinstance(helpline, str):
                help_parts.append(f"  - Helpline: {helpline}")

            if channels:
                help_parts.append("")
                help_parts.append("Ways to file complaint:")
                for ch in channels:
                    help_parts.append(f"  - {ch['method']}: {ch['detail']}")

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

        # Police Complaints Authority chunk
        pca = mechanism.get("pca_levels", {})
        if pca:
            pca_parts = [f"## Police Complaints Authority details", ""]

            state = pca.get("state_pca", {})
            if state:
                pca_parts.append(f"State PCA: {state.get('jurisdiction', '')}")
                pca_parts.append(f"  Address: {state.get('address', '')}")
                for ph in state.get("phone", []):
                    pca_parts.append(f"  Phone: {ph}")
                pca_parts.append(f"  Email: {state.get('email', '')}")
                pca_parts.append("")

            divs = pca.get("divisional_pcas", {})
            if divs:
                pca_parts.append(f"Divisional PCAs: {divs.get('jurisdiction', '')}")
                for div in divs.get("divisions", []):
                    pca_parts.append(f"  - {div['name']}: {div.get('phone', 'N/A')} | {div.get('email', 'N/A')}")

            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_pca",
                "section_number" : f"{mech_id}_PCA",
                "section_title"  : "Police Complaints Authority — addresses and jurisdiction",
                "chapter"        : "contacts",
                "content"        : "\n".join(pca_parts),
                "source"         : "CHRI Maharashtra PCA User Guide 2019",
                "sector"         : SECTOR,
                "doc_type"       : "contact_reference",
            })

        # RTI sample questions chunk
        rti_qs = mechanism.get("sample_questions_for_traffic_challan", [])
        if rti_qs:
            rti_parts = [
                "## RTI sample questions for challenging a traffic challan",
                "",
                "You can file an RTI (Right to Information) application asking these questions:",
                ""
            ]
            for q in rti_qs:
                rti_parts.append(f"  - {q}")

            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_rti_samples",
                "section_number" : f"{mech_id}_RTI",
                "section_title"  : "RTI sample questions for traffic challan disputes",
                "chapter"        : "information_right",
                "content"        : "\n".join(rti_parts),
                "source"         : "RTI Act 2005 + Maharashtra RTI Portal",
                "sector"         : SECTOR,
                "doc_type"       : DOC_TYPE,
            })

        # Court/legal challenge options
        options = mechanism.get("options", [])
        if options:
            for opt in options:
                opt_parts = [f"## {opt['method']}", ""]
                if opt.get("when"):
                    opt_parts.append(f"When to use: {opt['when']}")
                    opt_parts.append("")
                for s in opt.get("steps", []):
                    opt_parts.append(f"  - {s}")
                if opt.get("note"):
                    opt_parts.append(f"\nNote: {opt['note']}")

                records.append({
                    "doc_id"         : f"{SOURCE}_{mech_id}_{opt['method'].replace(' ', '_')[:30]}",
                    "section_number" : f"{mech_id}_{opt['method'][:20]}",
                    "section_title"  : f"Legal challenge — {opt['method']}",
                    "chapter"        : "legal_challenge",
                    "content"        : "\n".join(opt_parts),
                    "source"         : source_ref,
                    "sector"         : SECTOR,
                    "doc_type"       : DOC_TYPE,
                })

        # Citizen rights chunk
        rights = mechanism.get("important_citizen_rights", [])
        if rights:
            rights_parts = ["## Your rights when dealing with traffic challans", ""]
            for r in rights:
                rights_parts.append(f"  - {r}")

            records.append({
                "doc_id"         : f"{SOURCE}_{mech_id}_rights",
                "section_number" : f"{mech_id}_RIGHTS",
                "section_title"  : "Citizen rights — traffic challans",
                "chapter"        : "citizen_rights",
                "content"        : "\n".join(rights_parts),
                "source"         : "Motor Vehicles Act 1988, Constitution of India",
                "sector"         : SECTOR,
                "doc_type"       : "citizen_rights",
            })

    return records


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/traffic/actionable/traffic_actionable_reference.json"
    chunks = parse_from_json(path)
    print(f"\nParsed {len(chunks)} chunks from actionable reference")
    for c in chunks:
        print(f"  [{c['doc_type']:>22}] {c['doc_id'][:50]:50} | {c['section_title'][:60]}")
