"""
cc_mirt.py — Chemically Constrained Multidimensional Item Response Theory
==========================================================================
Module 1 of SMD-CC-MIRT-KL-CAT

Key design:
  - 3-dimensional ability vector θ = [θ1, θ2, θ3]
  - MIRT response probability: P(correct | θ, item) via compensatory model
      P = c_j + (1 - c_j) / (1 + exp(-a·θ - d))
  - Prerequisite soft constraint: if concept B requires concept A,
    cap θ_B ≤ θ_A + PREREQ_SLACK so implausible states are penalised.
  - Concept-to-dimension mapping: each concept maps to one primary skill
    dimension (θ1=foundational, θ2=periodic/bonding, θ3=coordination/advanced)
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional

DATA_DIR   = Path(__file__).parent.parent / "data"
PREREQ_SLACK = 0.5          # max advantage a child concept can have over its prereq
N_DIMS       = 3

# Concept → primary dimension mapping
CONCEPT_DIM_MAP = {
    # Dim 0: Foundational atomic structure
    "Effective Nuclear Charge":                     0,
    "Shielding Effect":                             0,
    "Orbital Penetration":                          0,
    "Electron-Electron Repulsion":                  0,
    "Energy Level Splitting in Atomic Orbitals":    0,
    "Exchange Energy and Half-Filled Shell Stability": 0,
    # Dim 1: Periodic trends & bonding
    "Atomic Radius Trend":                          1,
    "Ionization Enthalpy Trend":                    1,
    "Third Ionization Enthalpy Anomalies":          1,
    "Electron Gain Enthalpy Trend":                 1,
    "Electronegativity Trend":                      1,
    "Charge Density (Z/r) and Ionic Potential":     1,
    "Lattice Energy":                               1,
    "Polarization Effects (Fajan's Rule + Polarizing Power)": 1,
    "Inert Pair Effect":                            1,
    "Diagonal Relationship":                        1,
    "Sigma and Pi Bonding in Molecular Orbitals":   1,
    "Hybridization and Orbital Mixing Principles":  1,
    "Valence Shell Electron Pair Repulsion Theory": 1,
    "VSEPR Application to Hypervalent Molecules":   1,
    "VSEPR Hypervalent Geometry":                   1,
    "Bent's Rule":                                  1,
    "Dipole Moment and Molecular Polarity":         1,
    "Hydrogen Bonding":                             1,
    "Molecular Orbital Theory and Delocalization":  1,
    "Back Bonding":                                 1,
    "Electron-Deficient Bonding in Boranes":        1,
    "Oxoacid Strength and Basicity from Structure": 1,
    "Noble Gas Compound Stability":                 1,
    "Polymerization of Silicate Units":             1,
    "Thermal Stability from Lattice Energy and Polarization": 1,
    "Redox Stability and Disproportionation Tendencies": 1,
    "Solubility Product and Precipitation Logic":   1,
    "HSAB Principle":                               1,
    "Hard and Soft Acids and Bases (HSAB) Principle": 1,
    # Dim 2: Coordination / d-block / advanced
    "Coordination Number and Geometry Relationships": 2,
    "Ligand Denticity and Polydentate Binding":     2,
    "Chelate Effect":                               2,
    "Metal-Ligand Bonding (σ and π interactions in complexes)": 2,
    "Crystal Field Splitting in Octahedral Field":  2,
    "Crystal Field Splitting in Tetrahedral Field": 2,
    "Crystal Field Stabilization Energy":           2,
    "High-Spin vs Low-Spin Complexes":              2,
    "Spectrochemical Series":                       2,
    "Jahn-Teller Distortion":                       2,
    "Ligand Field Theory":                          2,
    "t2g Orbital Pi Bonding":                       2,
    "Magnetic Properties from Unpaired d-electrons": 2,
    "Color Origin in Coordination Compounds":       2,
    "Stability Constants of Complexes":             2,
    "Geometric Isomerism in Coordination Compounds": 2,
    "Optical Isomerism in Coordination Compounds":  2,
    "Linkage Isomerism":                            2,
    "Variable Oxidation State Stability in d-block": 2,
    "Standard Electrode Potential Trends in d-block": 2,
    "Lanthanoid Contraction":                       2,
    "Actinoid Contraction":                         2,
    "4d and 5d Series Similarity Post-Lanthanoid Contraction": 2,
    "Ellingham Diagram and Thermodynamic Feasibility": 2,
    "Electrochemical Reduction Principles in Metallurgy": 2,
}


class CCMIRT:
    """Chemically Constrained MIRT model.

    Parameters
    ----------
    concept_graph_path : Path or str, optional
        Path to concept_graph.csv. Defaults to data/concept_graph.csv.
    prereq_slack : float
        Maximum ability advantage a child concept can hold over a prerequisite.
    """

    def __init__(self,
                 concept_graph_path: Optional[Path] = None,
                 prereq_slack: float = PREREQ_SLACK):
        self.n_dims = N_DIMS
        self.prereq_slack = prereq_slack
        self._graph: nx.DiGraph = self._load_graph(concept_graph_path)

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------
    def _load_graph(self, path: Optional[Path]) -> nx.DiGraph:
        p = path or (DATA_DIR / "concept_graph.csv")
        G = nx.DiGraph()
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                G.add_edge(row["source_concept"], row["target_concept"],
                           weight=float(row["weight"]))
        return G

    # ------------------------------------------------------------------
    # MIRT probability
    # ------------------------------------------------------------------
    def prob(self,
             theta: np.ndarray,
             a_vec: np.ndarray,
             d: float,
             c_j: float = 0.25) -> float:
        """Compensatory MIRT probability of a correct response.

        P = c_j + (1 - c_j) * sigmoid(a·θ + d)
        """
        logit = float(np.dot(a_vec, theta)) + d
        pstar = 1.0 / (1.0 + np.exp(-logit))
        return c_j + (1.0 - c_j) * pstar

    # ------------------------------------------------------------------
    # Prerequisite constraint
    # ------------------------------------------------------------------
    def apply_prereq_constraint(self,
                                theta_by_concept: dict[str, float],
                                concept: str) -> float:
        """Return a soft-capped ability estimate for `concept`.

        If concept has prerequisites in the graph, its effective ability
        cannot exceed min(prereq_ability) + prereq_slack.
        """
        raw_theta = theta_by_concept.get(concept, 0.0)
        if concept not in self._graph:
            return raw_theta
        prereqs = list(self._graph.predecessors(concept))
        if not prereqs:
            return raw_theta
        prereq_abilities = [theta_by_concept.get(p, 0.0) for p in prereqs]
        cap = min(prereq_abilities) + self.prereq_slack
        return min(raw_theta, cap)

    # ------------------------------------------------------------------
    # Full constrained probability
    # ------------------------------------------------------------------
    def constrained_prob(self,
                         theta: np.ndarray,
                         concept: str,
                         a_vec: np.ndarray,
                         d: float,
                         c_j: float,
                         concept_thetas: dict[str, float]) -> float:
        """Apply prerequisite constraint then compute MIRT probability."""
        dim = CONCEPT_DIM_MAP.get(concept, 0)
        constrained_theta = theta.copy()
        cap_val = self.apply_prereq_constraint(concept_thetas, concept)
        constrained_theta[dim] = min(theta[dim], cap_val)
        return self.prob(constrained_theta, a_vec, d, c_j)

    # ------------------------------------------------------------------
    # Fisher information (for item selection)
    # ------------------------------------------------------------------
    def fisher_info(self,
                    theta: np.ndarray,
                    a_vec: np.ndarray,
                    d: float,
                    c_j: float = 0.25) -> np.ndarray:
        """3×3 Fisher information matrix for one item."""
        p = self.prob(theta, a_vec, d, c_j)
        q = 1.0 - p
        # Guard against numerical extremes
        pq = max(p * q, 1e-9)
        dp_dtheta = (p - c_j) * pq / max((1.0 - c_j) ** 2, 1e-9)
        # Information matrix I = (dP/dθ)² / (P·Q) · a·aᵀ  (scalar version per dim)
        scale = (dp_dtheta ** 2) / pq
        return scale * np.outer(a_vec, a_vec)
