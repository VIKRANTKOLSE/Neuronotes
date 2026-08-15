"""
data_processing.py
==================
Step 1: Process neuronotes_final_master_READY.csv into three clean artefacts:
  - data/items_clean.csv        — parsed item bank for psychometric modelling
  - data/item_options.csv       — one row per answer option with misconception tags
  - data/concept_graph.csv      — directed prerequisite graph edges

Run:  py data_processing.py
"""

import os
import ast
import csv
import json
import pandas as pd
import networkx as nx
from pathlib import Path

RAW_CSV   = Path(__file__).parent.parent / "neuronotes_final_master_READY.csv"
DATA_DIR  = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

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
    """Normalised entrapment weight from a 15-dim z-vector."""
    return round(sum(z_vec) / 15.0, 4)

def infer_misconception_tag(concept: str, option_no: int, reason: str) -> str:
    """Generate a deterministic misconception tag."""
    tag_base = concept.lower().replace(" ", "_").replace("(", "").replace(")", "")
    tag_base = "".join(c if c.isalnum() or c == "_" else "" for c in tag_base)
    return f"{tag_base}_opt{option_no}_error"

def infer_semantic_dimension(skill_vector: list[int]) -> str:
    """Map the first active dimension of the item's skill vector to a label."""
    dim_labels = [
        "recall", "application", "analysis", "synthesis",
        "evaluation", "calculation", "comparison", "classification",
        "inference", "explanation", "prediction", "generalisation",
        "transfer", "critique", "design"
    ]
    for i, v in enumerate(skill_vector):
        if v == 1:
            return dim_labels[i]
    return "recall"

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

# ---------------------------------------------------------------------------
# 2. Build items_clean.csv
# ---------------------------------------------------------------------------
def build_items_clean(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        a_vec  = parse_vec(row["a_vector"])
        sv     = parse_vec(row["skill_vector"])
        z1     = parse_vec(row["z1"])
        z2     = parse_vec(row["z2"])
        z3     = parse_vec(row["z3"])
        z4     = parse_vec(row["z4"])

        # Entrapment index: average trap weight of wrong options
        wrong_zs   = [z for i, z in enumerate([z1, z2, z3, z4])
                      if (i + 1) != int(row["correct_option"])]
        trap_weights = [infer_trap_weight(z) for z in wrong_zs]
        entrapment_index = round(sum(trap_weights) / len(trap_weights), 4) if trap_weights else 0.0

        # Dynamic c_j
        beta = 0.5
        c_j  = round(0.25 * (1 - beta * entrapment_index), 4)

        records.append({
            "item_id":            row["id"],
            "concept":            row["concept"],
            "concept_id":         row["concept_id"],
            "prereqs":            row["prereqs"],
            "prereq_ids":         row["prereq_ids"],
            "question_type":      row["question_type"],
            "difficulty_tier":    row["difficulty_tier"],
            "correct_option":     int(row["correct_option"]),
            "a1":                 a_vec[0] if len(a_vec) > 0 else 1.0,
            "a2":                 a_vec[1] if len(a_vec) > 1 else 0.5,
            "a3":                 a_vec[2] if len(a_vec) > 2 else 0.5,
            "d_param":            float(row["d_param"]),
            "skill_dim1":         sv[0] if len(sv) > 0 else 0,
            "skill_dim2":         sv[1] if len(sv) > 1 else 0,
            "skill_dim3":         sv[2] if len(sv) > 2 else 0,
            "entrapment_index":   entrapment_index,
            "c_j":                c_j,
            "estimated_time_sec": int(row["estimated_time_sec"]),
            "item_exposure_limit":int(row["item_exposure_limit"]),
            "status":             row["status"],
            "content_validated":  row["content_validated"],
            "misconception_validated": row["misconception_validated"],
            "version":            row["version"],
        })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 3. Build item_options.csv
# ---------------------------------------------------------------------------
def build_item_options(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        item_id     = row["id"]
        concept     = row["concept"]
        correct_opt = int(row["correct_option"])
        sv          = parse_vec(row["skill_vector"])

        for opt_no in range(1, 5):
            opt_text = str(row[f"opt{opt_no}"])
            reason   = str(row[f"reason{opt_no}"])
            z_vec    = parse_vec(row[f"z{opt_no}"])
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
                misconception_tag = infer_misconception_tag(concept, opt_no, reason)

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
                "semantic_dimension": infer_semantic_dimension(sv),
                "expert_confidence": "high",
                "review_status":     row["status"],
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
    "Third Ionization Enthalpy Anomalies": ["Ionization Enthalpy Trend"],
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
    "VSEPR Hypervalent Geometry":     ["VSEPR Application to Hypervalent Molecules"],
    "Bent's Rule":                    ["Hybridization and Orbital Mixing Principles"],
    "Dipole Moment and Molecular Polarity": ["Valence Shell Electron Pair Repulsion Theory", "Electronegativity Trend"],
    "Hydrogen Bonding":               ["Electronegativity Trend", "Dipole Moment and Molecular Polarity"],
    "Molecular Orbital Theory and Delocalization": ["Sigma and Pi Bonding in Molecular Orbitals"],
    "Back Bonding":                   ["Sigma and Pi Bonding in Molecular Orbitals", "Orbital Penetration"],
    "Electron-Deficient Bonding in Boranes": ["Sigma and Pi Bonding in Molecular Orbitals"],
    "Oxoacid Strength and Basicity from Structure": ["Electronegativity Trend", "Polarization Effects (Fajan's Rule + Polarizing Power)"],
    "Noble Gas Compound Stability":   ["Hybridization and Orbital Mixing Principles", "VSEPR Hypervalent Geometry"],
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
    "t2g Orbital Pi Bonding":         ["Metal-Ligand Bonding (σ and π interactions in complexes)", "Crystal Field Splitting in Octahedral Field"],
    "Magnetic Properties from Unpaired d-electrons": ["Crystal Field Stabilization Energy", "High-Spin vs Low-Spin Complexes"],
    "Color Origin in Coordination Compounds": ["Crystal Field Splitting in Octahedral Field", "Spectrochemical Series"],
    "Stability Constants of Complexes": ["Metal-Ligand Bonding (σ and π interactions in complexes)", "Chelate Effect"],
    "Geometric Isomerism in Coordination Compounds": ["Coordination Number and Geometry Relationships"],
    "Optical Isomerism in Coordination Compounds": ["Geometric Isomerism in Coordination Compounds"],
    "Linkage Isomerism":              ["Coordination Number and Geometry Relationships"],
    "HSAB Principle":                 ["Electronegativity Trend", "Polarization Effects (Fajan's Rule + Polarizing Power)"],
    "Hard and Soft Acids and Bases (HSAB) Principle": ["HSAB Principle"],

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
