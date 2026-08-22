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
                 prereq_slack: float = PREREQ_SLACK,
                 weight_calibration_interval: int = 100):
        self.n_dims = N_DIMS
        self.prereq_slack = prereq_slack
        self.weight_calibration_interval = max(1, int(weight_calibration_interval))
        self._prereq_attempt_log: list[dict] = []
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
    def prereq_penalty(self, theta_by_concept: dict[str, float]) -> float:
        """Graph loss: sum w_uv * max(0, theta_v-theta_u-epsilon)^2.

        Edges lacking an observed endpoint are excluded; unknown concepts must
        not be treated as zero mastery during an adaptive session.
        """
        loss = 0.0
        for source, target, data in self._graph.edges(data=True):
            if source not in theta_by_concept or target not in theta_by_concept:
                continue
            violation = theta_by_concept[target] - theta_by_concept[source] - self.prereq_slack
            if violation > 0:
                loss += float(data.get("weight", 1.0)) * violation ** 2
        return float(loss)

    def prereq_penalty_gradients(self, theta_by_concept: dict[str, float]) -> dict[str, float]:
        """Return the gradient of the soft prerequisite loss by concept."""
        gradients = {concept: 0.0 for concept in theta_by_concept}
        for source, target, data in self._graph.edges(data=True):
            if source not in theta_by_concept or target not in theta_by_concept:
                continue
            violation = theta_by_concept[target] - theta_by_concept[source] - self.prereq_slack
            if violation > 0:
                grad = 2.0 * float(data.get("weight", 1.0)) * violation
                gradients[target] += grad
                gradients[source] -= grad
        return gradients

    def apply_soft_prereq_penalty(self,
                                  theta: np.ndarray,
                                  theta_by_concept: dict[str, float],
                                  learning_rate: float = 0.05) -> np.ndarray:
        """Take one gradient step on the prerequisite loss without clipping."""
        gradients = self.prereq_penalty_gradients(theta_by_concept)
        for concept, gradient in gradients.items():
            if gradient == 0.0:
                continue
            dim = CONCEPT_DIM_MAP.get(concept, 0)
            theta[dim] -= learning_rate * gradient
            theta_by_concept[concept] -= learning_rate * gradient
        return np.clip(theta, -4.0, 4.0)

    def record_prereq_observation(self, concept: str, correct: bool) -> None:
        """Accumulate response evidence and periodically recalibrate edge weights."""
        self._prereq_attempt_log.append({"concept": concept, "correct": float(correct)})
        if len(self._prereq_attempt_log) % self.weight_calibration_interval == 0:
            self.calibrate_prereq_weights()

    def calibrate_prereq_weights(self, attempts: Optional[list[dict]] = None) -> None:
        """Fit edge strengths from observed prerequisite/target performance.

        An edge gains weight when target accuracy outpaces its prerequisite,
        which is evidence that the graph penalty needs to pull the estimates
        toward a more plausible ordering. A smoothed update prevents a small
        response batch from dominating expert-initialized weights.
        """
        logs = attempts if attempts is not None else self._prereq_attempt_log
        if not logs:
            return
        frame = pd.DataFrame(logs)
        accuracy = frame.groupby("concept")["correct"].mean().to_dict()
        for source, target, data in self._graph.edges(data=True):
            if source not in accuracy or target not in accuracy:
                continue
            target_weight = 1.0 + max(0.0, float(accuracy[target] - accuracy[source]))
            current = float(data.get("weight", 1.0))
            data["weight"] = float(np.clip(0.8 * current + 0.2 * target_weight, 0.1, 3.0))

    def apply_prereq_constraint(self,
                                theta_by_concept: dict[str, float],
                                concept: str) -> float:
        """Compatibility accessor; prerequisite handling is now loss-based."""
        return float(theta_by_concept.get(concept, 0.0))

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
        """Compute MIRT probability; prerequisite consistency is a soft loss.

        No ability estimate is hard-clipped at prediction time. Call
        :meth:`apply_soft_prereq_penalty` after a response update instead.
        """
        return self.prob(theta, a_vec, d, c_j)

    # ------------------------------------------------------------------
    # Fisher information (for item selection)
    # ------------------------------------------------------------------
    def fisher_info(self,
                    theta: np.ndarray,
                    a_vec: np.ndarray,
                    d: float,
                    c_j: float = 0.25) -> np.ndarray:
        """3×3 Fisher information matrix for one item (canonical 3PL MIRT formula).

        For the 3PL model:  P = c + (1-c) * P*
        where P* = sigmoid(a·θ + d).

        dP/dθ = (1-c) * P* * (1 - P*) * a
        I     = (dP/dθ)² / (P * Q)   [per dimension, then outer product]
        """
        p = self.prob(theta, a_vec, d, c_j)
        pq = max(p * (1.0 - p), 1e-9)
        # Guessing-free sigmoid probability
        p_star = (p - c_j) / max(1.0 - c_j, 1e-9)
        p_star = min(max(p_star, 1e-9), 1.0 - 1e-9)
        # Gradient of P w.r.t. θ (scalar × a_vec)
        dp_dtheta_scale = (1.0 - c_j) * p_star * (1.0 - p_star)
        # Information matrix: I = (dP/dθ outer dP/dθ) / (P·Q)
        scale = (dp_dtheta_scale ** 2) / pq
        return scale * np.outer(a_vec, a_vec)
