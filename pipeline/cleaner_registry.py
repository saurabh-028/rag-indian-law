"""
Maps each source file (PDF or JSON) to its cleaner module.

To add a new document:
  1. Create pipeline/cleaners/<name>.py with clean() and parse() functions
  2. Add an entry to REGISTRY below
"""

from pipeline.cleaners import (
    mv_act_central,
    mv_act_maharashtra,
    traffic_fines,
    traffic_actionable,
    bns,
    bnss,
    rent_control_maharashtra,
    hindu_marriage_act,
    matrimonial_actionable,
    criminal_law_actionable,
    rental_law_actionable,
)

REGISTRY = {
    # Traffic
    "Central Motor Vehicle act(amended till 2019).pdf": {
        "cleaner"    : mv_act_central,
        "sector"     : "traffic",
        "table_based": False,
    },
    "Maharashtra Motor Vehicle act 1989.pdf": {
        "cleaner"    : mv_act_maharashtra,
        "sector"     : "traffic",
        "table_based": False,
    },
    "traffic violation fines.pdf": {
        "cleaner"    : traffic_fines,
        "sector"     : "traffic",
        "table_based": True,
    },

    # Criminal Law
    "Bhartiya_Nyay_Sanhita(BNS).pdf": {
        "cleaner"    : bns,
        "sector"     : "criminal_law",
        "table_based": False,
    },
    "the_bharatiya_nagarik_suraksha_sanhita,_2023(BNSS).pdf": {
        "cleaner"    : bnss,
        "sector"     : "criminal_law",
        "table_based": False,
    },

    # Rental Law
    "eng_maharashtra_rent_control_ac.pdf": {
        "cleaner"    : rent_control_maharashtra,
        "sector"     : "rental_law",
        "table_based": False,
    },

    # Hindu Marriage Laws
    "Hindu_Marriage_Act.pdf": {
        "cleaner"    : hindu_marriage_act,
        "sector"     : "hindu_marriage_laws",
        "table_based": False,
    },

    # Actionable/grievance reference JSONs
    "traffic_actionable_reference.json": {
        "cleaner"    : traffic_actionable,
        "sector"     : "traffic",
        "table_based": False,
        "json_based" : True,
    },
    "matrimonial_actionable_reference.json": {
        "cleaner"    : matrimonial_actionable,
        "sector"     : "matrimonial",
        "table_based": False,
        "json_based" : True,
    },
    "criminal_law_actionable_reference.json": {
        "cleaner"    : criminal_law_actionable,
        "sector"     : "criminal_law",
        "table_based": False,
        "json_based" : True,
    },
    "rental_law_actionable_reference.json": {
        "cleaner"    : rental_law_actionable,
        "sector"     : "rental_law",
        "table_based": False,
        "json_based" : True,
    },
}


def get(filename: str) -> dict:
    """Returns the registry entry for a given filename, or raises ValueError."""
    if filename not in REGISTRY:
        registered = "\n  ".join(REGISTRY.keys())
        raise ValueError(
            f"\n[cleaner_registry] No entry found for: '{filename}'\n"
            f"Registered files:\n  {registered}\n"
            f"→ Add an entry to REGISTRY in cleaner_registry.py"
        )
    return REGISTRY[filename]


def list_all() -> None:
    """Print all registered files and their sectors."""
    print(f"{'Filename':<55} {'Sector':<20} {'Table?'}")
    print("-" * 85)
    for fname, cfg in REGISTRY.items():
        print(f"{fname:<55} {cfg['sector']:<20} {cfg['table_based']}")
