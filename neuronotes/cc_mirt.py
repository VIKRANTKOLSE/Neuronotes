"""
cc_mirt.py — Chemically Constrained Multidimensional Item Response Theory
==========================================================================
Module 1 of SMD-CC-MIRT-KL-CAT

Key design:
  - 58-dimensional ability vector θ (one dimension per Q-matrix concept node)
  - MIRT response probability: P(correct | θ, item) via compensatory model
      P = c_j + (1 - c_j) / (1 + exp(-a·θ - d))
  - Prerequisite soft constraint: if concept B requires concept A,
    cap θ_B ≤ θ_A + PREREQ_SLACK so implausible states are penalised.
  - CONCEPT_DIM_MAP: each concept maps to its primary Q-matrix dimension
"""

import math
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional

DATA_DIR     = Path(__file__).parent.parent / "data"
PREREQ_SLACK = 0.5          # max advantage a child concept can have over its prereq
N_DIMS       = 58           # one dimension per Q-matrix concept node

# Concept → primary Q-matrix dimension mapping
# Computed from the questions_final_qmatrix.csv a_vector loadings
CONCEPT_DIM_MAP = {
    "Effective Nuclear Charge":                     0,
    "Shielding Effect":                             1,
    "Orbital Penetration":                          2,
    "Electron-Electron Repulsion":                  3,
    "Energy Level Splitting in Atomic Orbitals":    4,
    "Charge Density (Z/r) and Ionic Potential":     5,
    "Lattice Energy":                               6,
    "Sigma and Pi Bonding in Molecular Orbitals":   9,
    "Atomic Radius Trend":                          10,
    "Ionization Enthalpy Trend":                    11,
    "Electron Gain Enthalpy Trend":                 12,
    "Electronegativity Trend":                      13,
    "Dipole Moment and Molecular Polarity":         14,
    "Valence Shell Electron Pair Repulsion Theory": 15,
    "Hybridization and Orbital Mixing Principles":  16,
    "Molecular Orbital Theory and Delocalization":  17,
    "Hydrogen Bonding":                             18,
    "Back Bonding":                                 19,
    "Bent's Rule":                                  20,
    "Polarization Effects (Fajan's Rule + Polarizing Power)": 21,
    "Solubility Product and Precipitation Logic":   22,
    "Diagonal Relationship":                        23,
    "Inert Pair Effect":                            24,
    "Polymerization of Silicate Units":             25,
    "Thermal Stability from Lattice Energy and Polarization": 26,
    "Oxoacid Strength and Basicity from Structure": 27,
    "Lanthanoid Contraction":                       28,
    "Actinoid Contraction":                         29,
    "4d and 5d Series Similarity Post-Lanthanoid Contraction": 30,
    "Variable Oxidation State Stability in d-block": 31,
    "Exchange Energy and Half-Filled Shell Stability": 32,
    "Standard Electrode Potential Trends in d-block": 33,
    "Magnetic Properties from Unpaired d-electrons": 34,
    "Hard and Soft Acids and Bases (HSAB) Principle": 35,
    "Electron-Deficient Bonding in Boranes":        36,
    "VSEPR Application to Hypervalent Molecules":   37,
    "Noble Gas Compound Stability":                 38,
    "Redox Stability and Disproportionation Tendencies": 39,
    "Spectrochemical Series":                       40,
    "Crystal Field Splitting in Octahedral Field":  41,
    "Crystal Field Splitting in Tetrahedral Field": 42,
    "Crystal Field Stabilization Energy":           43,
    "High-Spin vs Low-Spin Complexes":              44,
    "Jahn-Teller Distortion":                       45,
    "Ligand Field Theory":                          46,
    "Color Origin in Coordination Compounds":       47,
    "Chelate Effect":                               48,
    "Stability Constants of Complexes":             49,
    "Linkage Isomerism":                            50,
    "Geometric Isomerism in Coordination Compounds": 51,
    "Optical Isomerism in Coordination Compounds":  52,
    "Ellingham Diagram and Thermodynamic Feasibility": 53,
    "Electrochemical Reduction Principles in Metallurgy": 54,
    "Coordination Number and Geometry Relationships": 55,
    "Ligand Denticity and Polydentate Binding":     56,
    # Handle both unicode and lossy-encoded variants of σ/π
    "Metal-Ligand Bonding (σ and π interactions in complexes)": 57,
    "Metal-Ligand Bonding (? and ? interactions in complexes)": 57,
}


def parse_a_vector(value) -> np.ndarray:
    """Parse an a_vector from CSV (JSON string or list) into a numpy array of length N_DIMS."""
    if isinstance(value, np.ndarray):
        vec = value
    elif isinstance(value, (list, tuple)):
        vec = np.asarray(value, dtype=float)
    else:
        try:
            parsed = json.loads(str(value))
        except (ValueError, TypeError):
            try:
                import ast
                parsed = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                parsed = []
        vec = np.asarray(parsed, dtype=float)
    # Ensure exactly N_DIMS
    vec = vec.reshape(-1)[:N_DIMS]
    if len(vec) < N_DIMS:
        vec = np.pad(vec, (0, N_DIMS - len(vec)))
    return vec


def calculate_dag_magnitude(residual: float, t: int, gamma_0: float = 0.35, beta: float = 1.0) -> float:
    """Calculate decayed DAG evidence magnitude, vanishing as residual -> 0."""
    gamma_t = gamma_0 / math.sqrt(1.0 + beta * t)
    return min(gamma_t * abs(residual), 0.35)


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
        self._build_propagation_matrices()

    # ------------------------------------------------------------------
    # Graph loading & propagation matrix precomputation
    # ------------------------------------------------------------------
    def _load_graph(self, path: Optional[Path]) -> nx.DiGraph:
        p = path or (DATA_DIR / "concept_graph.csv")
        G = nx.DiGraph()
        if p.exists():
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                src = str(row["source_concept"]).strip()
                tgt = str(row["target_concept"]).strip()
                if src != "ROOT" and tgt != "ROOT":
                    G.add_edge(src, tgt, weight=float(row.get("weight", 1.0)))
        return G

    def _build_propagation_matrices(self,
                                    gamma: float = 0.55,
                                    alpha_up: float = 0.35,
                                    alpha_down: float = 0.35) -> None:
        """Precompute O(1) matrix propagation weights across the DAG."""
        self._M_up = np.zeros((self.n_dims, self.n_dims), dtype=float)
        self._M_down = np.zeros((self.n_dims, self.n_dims), dtype=float)

        try:
            path_lengths = dict(nx.all_pairs_shortest_path_length(self._graph))
        except Exception:
            path_lengths = {}

        for source, targets in path_lengths.items():
            if source not in CONCEPT_DIM_MAP:
                continue
            src_idx = CONCEPT_DIM_MAP[source]
            for target, dist in targets.items():
                if target not in CONCEPT_DIM_MAP or dist == 0:
                    continue
                tgt_idx = CONCEPT_DIM_MAP[target]
                # source is prerequisite (parent), target is child
                # If target is correct, propagate UP to prerequisite source:
                self._M_up[src_idx, tgt_idx] = max(
                    self._M_up[src_idx, tgt_idx],
                    alpha_up * (gamma ** (dist - 1))
                )
                # If source is incorrect, propagate DOWN to child target:
                self._M_down[tgt_idx, src_idx] = max(
                    self._M_down[tgt_idx, src_idx],
                    alpha_down * (gamma ** (dist - 1))
                )

    def propagate_dag_evidence(self,
                               theta: np.ndarray,
                               concept: str,
                               correct: bool,
                               residual: float,
                               t: int = 0) -> np.ndarray:
        """Propagate ability evidence through the prerequisite DAG.

        - On correct: propagate positive evidence upward to all prerequisite ancestors.
        - On incorrect: propagate negative evidence downward to all downstream descendants.
        """
        if concept not in CONCEPT_DIM_MAP:
            return theta

        c_idx = CONCEPT_DIM_MAP[concept]
        th = theta.copy()
        mag = calculate_dag_magnitude(residual, t)
        if mag <= 1e-9:
            return th

        if correct:
            weights = self._M_up[:, c_idx]
            mask = weights > 0
            # Pull prerequisites up towards mastery
            th[mask] = np.maximum(th[mask], np.clip(th[mask] + weights[mask] * mag, -3.0, 3.0))
        else:
            weights = self._M_down[:, c_idx]
            mask = weights > 0
            # Pull descendants down towards weakness
            th[mask] = np.minimum(th[mask], np.clip(th[mask] - weights[mask] * mag, -3.0, 3.0))

        return np.clip(th, -3.0, 3.0)

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
                                  learning_rate: float = 0.05,
                                  t: int = 0) -> np.ndarray:
        """Take one gradient step on the prerequisite loss with time-decayed learning rate."""
        eta_prereq = learning_rate / (1.0 + 0.02 * t)
        gradients = self.prereq_penalty_gradients(theta_by_concept)
        for concept, gradient in gradients.items():
            if gradient == 0.0:
                continue
            dim = CONCEPT_DIM_MAP.get(concept, 0)
            theta[dim] -= eta_prereq * gradient
            theta_by_concept[concept] -= eta_prereq * gradient
        return np.clip(theta, -4.0, 4.0)

    def record_prereq_observation(self, concept: str, correct: bool) -> None:
        """Accumulate response evidence and periodically recalibrate edge weights."""
        self._prereq_attempt_log.append({"concept": concept, "correct": float(correct)})
        if not hasattr(self, "_concept_correct_counts"):
            self._concept_correct_counts = {}
            self._concept_total_counts = {}
        self._concept_correct_counts[concept] = self._concept_correct_counts.get(concept, 0) + int(correct)
        self._concept_total_counts[concept] = self._concept_total_counts.get(concept, 0) + 1
        if len(self._prereq_attempt_log) % self.weight_calibration_interval == 0:
            self.calibrate_prereq_weights()

    def calibrate_prereq_weights(self, attempts: Optional[list[dict]] = None) -> None:
        """Fit edge strengths from observed prerequisite/target performance."""
        if attempts is not None:
            if not attempts:
                return
            frame = pd.DataFrame(attempts)
            accuracy = frame.groupby("concept")["correct"].mean().to_dict()
        else:
            if not hasattr(self, "_concept_total_counts") or not self._concept_total_counts:
                return
            accuracy = {c: self._concept_correct_counts[c] / self._concept_total_counts[c]
                        for c in self._concept_total_counts}
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
        """N_DIMS×N_DIMS Fisher information matrix for one item (canonical 3PL MIRT formula).

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
