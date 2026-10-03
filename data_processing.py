"""
data_processing.py
==================
Step 1: Process questions_final_qmatrix.csv into three clean artefacts:
  - data/items_clean.csv        — parsed item bank for psychometric modelling
  - data/item_options.csv       — one row per answer option with misconception tags
  - data/concept_graph.csv      — directed prerequisite graph edges

Run:  py data_processing.py
"""

import os
import ast
import csv
import json
import math
import pandas as pd
import networkx as nx
from pathlib import Path

# ---------------------------------------------------------------------------
# Source data — use the calibrated Q-matrix CSV (post-rectification pipeline).
# Original: questions_final_qmatrix.csv → rectified by rectify_qmatrix.py
# ---------------------------------------------------------------------------
RAW_CSV   = Path(__file__).parent / "data" / "questions_calibrated_qmatrix.csv"
DATA_DIR  = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Number of MIRT ability dimensions (one per Q-matrix node / concept)
N_DIMS = 58

# ---------------------------------------------------------------------------
# Error class heuristics from reason text
# ---------------------------------------------------------------------------
ARITHMETIC_KEYWORDS  = ["arithmetic", "calculation", "numerical", "divide", "multiply",
                         "add", "subtract", "math", "compute", "value"]
CONCEPTUAL_KEYWORDS  = ["incorrect", "wrong concept", "confuses", "misunderstands",
                         "does not", "is not", "cannot", "not the"]

def infer_error_class(reason_text: str) -> str:
    """Heuristic: classify an option's error type from its rationale."""
    t = reason_text.lower()
    if any(k in t for k in ARITHMETIC_KEYWORDS):
        return "arithmetic_error"
    if any(k in t for k in CONCEPTUAL_KEYWORDS):
        return "conceptual_error"
    return "conceptual_error"          # safe default for wrong options

def infer_severity(error_class: str, z_vec: list[int]) -> str:
    """Severity based on how many skill dimensions a distractor activates."""
    active = sum(z_vec)
    if active >= 3:
        return "high"
    if active >= 1:
        return "medium"
    return "low"

def infer_trap_weight(z_vec: list[int]) -> float:
    """Normalised entrapment weight from a 4-dim z-vector."""
    return round(sum(z_vec) / 4.0, 4)

def infer_misconception_tag(z_vec: list[int]) -> str:
    """Return the ontology key for an option's active z dimension.

    z-vector positions are shared across the item bank, so item-local option
    labels must never be used as misconception identifiers.
    """
    active = [i for i, value in enumerate(z_vec) if int(value) == 1]
    return f"z_{active[0]:02d}" if active else "unknown_error"

def infer_semantic_dimension(a_vector: list[float]) -> str:
    """Map the primary active dimension of the 58-dim a_vector to a label.

    Uses a coarse grouping of the 58 Q-matrix dimensions into cognitive
    categories based on the tier structure of the concept DAG.
    """
    # Find primary active dimension
    if not a_vector:
        return "recall"
    primary_dim = max(range(len(a_vector)), key=lambda i: abs(a_vector[i]))

    # Coarse cognitive category mapping based on tier groupings
    if primary_dim <= 9:
        return "recall"           # foundational / Tier 0-2
    elif primary_dim <= 22:
        return "application"      # periodic trends / Tier 3-4
    elif primary_dim <= 39:
        return "analysis"         # bonding, d-block / Tier 5-6
    elif primary_dim <= 49:
        return "synthesis"        # coordination chemistry / Tier 7-8
    else:
        return "evaluation"       # isomerism, metallurgy / Tier 9-10


def initial_dynamic_c(semantic_entrapment: float) -> float:
    """Dynamic-c prior before empirical response data is available."""
    logit = 2.0 - 1.6 * float(semantic_entrapment)
    return round(0.25 / (1.0 + math.exp(-logit)), 4)

# ---------------------------------------------------------------------------
# 1. Load raw CSV
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df

def parse_vec(s: str) -> list:
    """Safely parse a string like '[1, 0, 0]' into a Python list."""
    try:
        return ast.literal_eval(str(s).strip())
    except Exception:
        return []

def compress_z_vec(z: list) -> list:
    """Compresses 15D z-vector down to 4D macro-categories."""
    if not z:
        return [0.0, 0.0, 0.0, 0.0]
    if len(z) < 15:
        z = z + [0.0] * (15 - len(z))
    return [
        float(max(z[0:4])),
        float(max(z[4:8])),
        float(max(z[8:12])),
        float(max(z[12:15]))
    ]

# ---------------------------------------------------------------------------
# 2. Build items_clean.csv
# ---------------------------------------------------------------------------
def build_items_clean(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        a_vec  = parse_vec(row["a_vector"])
        z1     = compress_z_vec(parse_vec(row["z1"]))
        z2     = compress_z_vec(parse_vec(row["z2"]))
        z3     = compress_z_vec(parse_vec(row["z3"]))
        z4     = compress_z_vec(parse_vec(row["z4"]))

        # Pad / truncate a_vector to exactly N_DIMS
        if len(a_vec) < N_DIMS:
            a_vec = a_vec + [0.0] * (N_DIMS - len(a_vec))
        a_vec = a_vec[:N_DIMS]

        correct_opt = int(row["correct_option"])

        # Entrapment index: average trap weight of wrong options
        wrong_zs   = [z for i, z in enumerate([z1, z2, z3, z4])
                      if (i + 1) != correct_opt]
        trap_weights = [infer_trap_weight(z) for z in wrong_zs]
        entrapment_index = float(row.get("E_semantic", round(sum(trap_weights) / len(trap_weights), 4) if trap_weights else 0.0))

        # Empirical and ambiguity terms are filled by DynamicC after attempts.
        c_j = initial_dynamic_c(entrapment_index)

        # Concept name — normalise unicode variants
        concept = str(row["concept"]).strip()

        # Use source_id as item_id (qmatrix uses 'source_id' not 'id')
        item_id = str(row.get("source_id", row.get("id", f"item_{_}")))

        records.append({
            "item_id":            item_id,
            "concept":            concept,
            "prereqs":            str(row.get("prereqs", "")),
            "correct_option":     correct_opt,
            "a_vector":           json.dumps(a_vec),
            "d_param":            float(row["d_param"]),
            "entrapment_index":   entrapment_index,
            "semantic_entrapment": entrapment_index,
            "empirical_entrapment": 0.0,
            "ambiguity_index":    float(row.get("E_ambiguity", 0.0)),
            "c_j":                c_j,
            # Provide sensible defaults for metadata absent from qmatrix
            "estimated_time_sec": int(row.get("estimated_time_sec", 90)),
            "item_exposure_limit":int(row.get("item_exposure_limit", 60)),
            "question_type":      str(row.get("question_type", "MCQ")),
            "difficulty_tier":    str(row.get("difficulty_tier", "medium")),
            "status":             str(row.get("status", "active")),
            "content_validated":  bool(row.get("content_validated", True)),
            "misconception_validated": bool(row.get("misconception_validated", True)),
            "version":            str(row.get("version", "1.0")),
        })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 3. Build item_options.csv
# ---------------------------------------------------------------------------
def build_item_options(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        item_id     = str(row.get("source_id", row.get("id", "")))
        concept     = str(row["concept"]).strip()
        correct_opt = int(row["correct_option"])
        a_vec       = parse_vec(row["a_vector"])

        for opt_no in range(1, 5):
            opt_text = str(row[f"opt{opt_no}"])
            reason   = str(row[f"reason{opt_no}"])
            z_vec    = compress_z_vec(parse_vec(row[f"z{opt_no}"]))
            is_correct = (opt_no == correct_opt)

            if is_correct:
                error_class       = "correct"
                severity          = "none"
                trap_weight       = 0.0
                misconception_tag = "none"
            else:
                error_class       = infer_error_class(reason)
                severity          = infer_severity(error_class, z_vec)
                trap_weight       = infer_trap_weight(z_vec)
                misconception_tag = infer_misconception_tag(z_vec)

            records.append({
                "item_id":           item_id,
                "option_no":         opt_no,
                "option_text":       opt_text,
                "is_correct":        is_correct,
                "rationale":         reason,
                "misconception_tag": misconception_tag,
                "error_class":       error_class,
                "severity":          severity,
                "trap_weight":       trap_weight,
                "semantic_dimension": infer_semantic_dimension(a_vec),
                "expert_confidence": "high",
                "review_status":     str(row.get("status", "active")),
                "z_vector":          str(z_vec),
            })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 4. Build concept_graph.csv
# ---------------------------------------------------------------------------
CONCEPT_PREREQ_MAP = {
    # Atomic Structure chain
    "Shielding Effect":               ["Effective Nuclear Charge"],
    "Orbital Penetration":            ["Effective Nuclear Charge", "Shielding Effect"],
    "Electron-Electron Repulsion":    ["Effective Nuclear Charge", "Shielding Effect"],
    "Energy Level Splitting in Atomic Orbitals": ["Orbital Penetration", "Shielding Effect"],
    "Exchange Energy and Half-Filled Shell Stability": ["Energy Level Splitting in Atomic Orbitals"],

    # Periodic Trends chain
    "Atomic Radius Trend":            ["Effective Nuclear Charge", "Shielding Effect"],
    "Ionization Enthalpy Trend":      ["Atomic Radius Trend", "Effective Nuclear Charge"],
    "Electron Gain Enthalpy Trend":   ["Atomic Radius Trend", "Electron-Electron Repulsion"],
    "Electronegativity Trend":        ["Ionization Enthalpy Trend", "Atomic Radius Trend"],
    "Charge Density (Z/r) and Ionic Potential": ["Atomic Radius Trend"],
    "Lattice Energy":                 ["Charge Density (Z/r) and Ionic Potential"],
    "Inert Pair Effect":              ["Ionization Enthalpy Trend"],
    "Diagonal Relationship":          ["Charge Density (Z/r) and Ionic Potential", "Electronegativity Trend"],
    "Thermal Stability from Lattice Energy and Polarization": ["Lattice Energy", "Polarization Effects (Fajan's Rule + Polarizing Power)"],
    "Polarization Effects (Fajan's Rule + Polarizing Power)": ["Charge Density (Z/r) and Ionic Potential"],
    "Redox Stability and Disproportionation Tendencies": ["Standard Electrode Potential Trends in d-block"],

    # Bonding chain
    "Sigma and Pi Bonding in Molecular Orbitals": ["Orbital Penetration"],
    "Hybridization and Orbital Mixing Principles": ["Sigma and Pi Bonding in Molecular Orbitals"],
    "Valence Shell Electron Pair Repulsion Theory": ["Hybridization and Orbital Mixing Principles"],
    "VSEPR Application to Hypervalent Molecules": ["Valence Shell Electron Pair Repulsion Theory"],
    "Bent's Rule":                    ["Hybridization and Orbital Mixing Principles"],
    "Dipole Moment and Molecular Polarity": ["Valence Shell Electron Pair Repulsion Theory", "Electronegativity Trend"],
    "Hydrogen Bonding":               ["Electronegativity Trend", "Dipole Moment and Molecular Polarity"],
    "Molecular Orbital Theory and Delocalization": ["Sigma and Pi Bonding in Molecular Orbitals"],
    "Back Bonding":                   ["Sigma and Pi Bonding in Molecular Orbitals", "Orbital Penetration"],
    "Electron-Deficient Bonding in Boranes": ["Sigma and Pi Bonding in Molecular Orbitals"],
    "Oxoacid Strength and Basicity from Structure": ["Electronegativity Trend", "Polarization Effects (Fajan's Rule + Polarizing Power)"],
    "Noble Gas Compound Stability":   ["Hybridization and Orbital Mixing Principles", "VSEPR Application to Hypervalent Molecules"],
    "Polymerization of Silicate Units": ["Sigma and Pi Bonding in Molecular Orbitals"],

    # Coordination Chemistry chain
    "Coordination Number and Geometry Relationships": ["Hybridization and Orbital Mixing Principles"],
    "Ligand Denticity and Polydentate Binding": ["Coordination Number and Geometry Relationships"],
    "Chelate Effect":                 ["Ligand Denticity and Polydentate Binding"],
    "Metal-Ligand Bonding (σ and π interactions in complexes)": ["Coordination Number and Geometry Relationships", "Back Bonding"],
    "Crystal Field Splitting in Octahedral Field": ["Coordination Number and Geometry Relationships"],
    "Crystal Field Splitting in Tetrahedral Field": ["Crystal Field Splitting in Octahedral Field"],
    "Crystal Field Stabilization Energy": ["Crystal Field Splitting in Octahedral Field"],
    "High-Spin vs Low-Spin Complexes": ["Crystal Field Splitting in Octahedral Field", "Spectrochemical Series"],
    "Spectrochemical Series":         ["Metal-Ligand Bonding (σ and π interactions in complexes)"],
    "Jahn-Teller Distortion":         ["Crystal Field Splitting in Octahedral Field"],
    "Ligand Field Theory":            ["Crystal Field Splitting in Octahedral Field", "Molecular Orbital Theory and Delocalization"],
    "Magnetic Properties from Unpaired d-electrons": ["Crystal Field Stabilization Energy", "High-Spin vs Low-Spin Complexes"],
    "Color Origin in Coordination Compounds": ["Crystal Field Splitting in Octahedral Field", "Spectrochemical Series"],
    "Stability Constants of Complexes": ["Metal-Ligand Bonding (σ and π interactions in complexes)", "Chelate Effect"],
    "Geometric Isomerism in Coordination Compounds": ["Coordination Number and Geometry Relationships"],
    "Optical Isomerism in Coordination Compounds": ["Geometric Isomerism in Coordination Compounds"],
    "Linkage Isomerism":              ["Coordination Number and Geometry Relationships"],
    "Hard and Soft Acids and Bases (HSAB) Principle": ["Electronegativity Trend", "Polarization Effects (Fajan's Rule + Polarizing Power)"],

    # d-block / f-block
    "Variable Oxidation State Stability in d-block": ["Ionization Enthalpy Trend", "Crystal Field Stabilization Energy"],
    "Standard Electrode Potential Trends in d-block": ["Variable Oxidation State Stability in d-block"],
    "Lanthanoid Contraction":         ["Atomic Radius Trend", "Effective Nuclear Charge"],
    "Actinoid Contraction":           ["Lanthanoid Contraction"],
    "4d and 5d Series Similarity Post-Lanthanoid Contraction": ["Lanthanoid Contraction"],

    # Metallurgy chain
    "Ellingham Diagram and Thermodynamic Feasibility": ["Lattice Energy"],
    "Electrochemical Reduction Principles in Metallurgy": ["Ellingham Diagram and Thermodynamic Feasibility",
                                                           "Standard Electrode Potential Trends in d-block"],

    # Solubility
    "Solubility Product and Precipitation Logic": ["Lattice Energy", "Charge Density (Z/r) and Ionic Potential"],
}

# Bidirectional DAG maps: invert the prerequisite map so evidence can also
# diffuse DOWNWARD (failed parent -> penalize children), not just upward.
CONCEPT_CHILDREN_MAP: dict[str, list[str]] = {}
for _child, _prereqs in CONCEPT_PREREQ_MAP.items():
    for _prereq in _prereqs:
        CONCEPT_CHILDREN_MAP.setdefault(_prereq, []).append(_child)

def build_concept_graph(all_concepts: list[str]) -> pd.DataFrame:
    records = []
    for target, sources in CONCEPT_PREREQ_MAP.items():
        for src in sources:
            records.append({
                "source_concept":  src,
                "target_concept":  target,
                "relation":        "prerequisite",
                "weight":          1.0,
                "expert_validated": True,
            })
    # Ensure all leaf concepts appear at least once
    covered = set(r["source_concept"] for r in records) | set(r["target_concept"] for r in records)
    for c in all_concepts:
        if c not in covered:
            records.append({
                "source_concept":  "ROOT",
                "target_concept":  c,
                "relation":        "foundation",
                "weight":          1.0,
                "expert_validated": True,
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading raw CSV ...")
    df = load_raw(RAW_CSV)
    print("  >> {} items, {} columns".format(len(df), len(df.columns)))

    print("Building items_clean.csv ...")
    items_clean = build_items_clean(df)
    items_clean.to_csv(DATA_DIR / "items_clean.csv", index=False)
    print("  >> {} rows saved".format(len(items_clean)))

    print("Building item_options.csv ...")
    item_opts = build_item_options(df)
    item_opts.to_csv(DATA_DIR / "item_options.csv", index=False)
    print("  >> {} rows saved (4 options x {} items)".format(len(item_opts), len(df)))

    print("Building concept_graph.csv ...")
    all_concepts = df["concept"].unique().tolist()
    cg = build_concept_graph(all_concepts)
    cg.to_csv(DATA_DIR / "concept_graph.csv", index=False)
    print("  >> {} directed edges saved".format(len(cg)))

    print("\nData processing complete. Files in codes/data/")

if __name__ == "__main__":
    main()
