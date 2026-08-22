"""
learner_generator.py — Synthetic Learner Population Generator
==============================================================
Generates N simulated learners with:
  - Known 3D ability vectors (θ_true)
  - Prerequisite-linked mastery patterns
  - Known misconception states per concept
  - Distractor-selection probabilities matching their misconceptions

Each learner is a dict with full ground-truth state for metric evaluation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from neuronotes.c_matrix import CMatrix

# Reproducible seeds
DEFAULT_SEED = 42
N_SEMANTIC_DIMS = 15

# Profile archetypes
PROFILES = {
    "strong_all":     {"theta_mean": [1.5,  1.5,  1.5],  "theta_std": [0.3, 0.3, 0.3]},
    "weak_all":       {"theta_mean": [-1.5, -1.5, -1.5], "theta_std": [0.3, 0.3, 0.3]},
    "strong_found":   {"theta_mean": [1.2,  0.0, -0.5],  "theta_std": [0.2, 0.4, 0.4]},
    "strong_bonding": {"theta_mean": [0.5,  1.2,  0.0],  "theta_std": [0.4, 0.2, 0.4]},
    "strong_coord":   {"theta_mean": [-0.2, 0.5,  1.5],  "theta_std": [0.4, 0.4, 0.2]},
    "mixed":          {"theta_mean": [0.0,  0.0,  0.0],  "theta_std": [0.8, 0.8, 0.8]},
}


@dataclass
class SyntheticLearner:
    """One simulated learner with ground-truth state."""
    learner_id:                 str
    profile:                    str
    theta_true:                 np.ndarray          # ground-truth 3D ability
    theta_init:                 np.ndarray          # starting estimate (θ₀ = [0,0,0])
    misconception_true:         dict                # tag → true strength [0,1]
    concept_mastery:            dict                # concept → True/False (prerequisite-aware)
    misconception_true_vector:  np.ndarray          # binary shared-ontology ground truth
    rng:                        np.random.Generator # private rng for response simulation


class LearnerGenerator:
    """Generates synthetic learner population with ground-truth psychometrics."""

    def __init__(self,
                 n_learners: int = 200,
                 seed: int = DEFAULT_SEED,
                 profile_dist: Optional[dict] = None):
        self.n_learners   = n_learners
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)
        self.profile_dist = profile_dist or {
            "strong_all":     0.15,
            "weak_all":       0.15,
            "strong_found":   0.20,
            "strong_bonding": 0.20,
            "strong_coord":   0.15,
            "mixed":          0.15,
        }
        # Instance-level CMatrix to prevent global/class-level state pollution
        self._c_matrix = CMatrix()

    def generate(self) -> list[SyntheticLearner]:
        learners = []
        profile_names = list(self.profile_dist.keys())
        profile_probs = np.array([self.profile_dist[k] for k in profile_names])
        profile_probs /= profile_probs.sum()

        for i in range(self.n_learners):
            profile = self.rng.choice(profile_names, p=profile_probs)
            spec    = PROFILES[profile]

            # Sample true ability
            theta_true = np.clip(
                self.rng.normal(spec["theta_mean"], spec["theta_std"]),
                -3.0, 3.0
            )

            # Sample misconceptions: learners with low foundational ability
            # are more likely to have strong misconceptions
            misc_state = {}
            misc_vector = np.zeros(N_SEMANTIC_DIMS, dtype=int)
            weakness = max(0.0, -float(np.mean(theta_true[:2]))) / 3.0
            # Each ontology dimension receives independent ground truth so
            # AUROC can be evaluated as a real multilabel ranking problem.
            # A single loop covers all 15 dims — the old MISCONCEPTION_ARCHETYPES
            # loop was removed because it overwrote z_00–z_04 with a separate
            # RNG draw, creating an inconsistency between misc_state and
            # misc_vector that corrupted AUROC computation.
            for dim in range(N_SEMANTIC_DIMS):
                prevalence = min(0.08 + 0.52 * weakness + 0.02 * (dim % 3), 0.80)
                if self.rng.random() < prevalence:
                    misc_vector[dim] = 1
                    misc_state[f"z_{dim:02d}"] = float(self.rng.uniform(0.3, 0.9))

            # Concept mastery: a concept is "mastered" if its dimension ability > 0
            concept_mastery = {
                "foundational": bool(theta_true[0] > 0),
                "periodic_bonding": bool(theta_true[1] > 0),
                "coordination_advanced": bool(theta_true[2] > 0),
            }

            learners.append(SyntheticLearner(
                learner_id=f"L{i:04d}",
                profile=profile,
                theta_true=theta_true,
                theta_init=np.zeros(3),
                misconception_true=misc_state,
                misconception_true_vector=misc_vector,
                concept_mastery=concept_mastery,
                rng=np.random.default_rng(int(self.rng.integers(0, 2**31))),
            ))

        return learners

    # ------------------------------------------------------------------
    def simulate_response(self,
                          learner: SyntheticLearner,
                          item_row,
                          c_j: float = 0.25) -> tuple[bool, int]:
        """Simulate a learner's response to an item.

        Returns (correct: bool, selected_option: int 1-indexed).
        """
        a_vec = np.array([float(item_row["a1"]),
                          float(item_row["a2"]),
                          float(item_row["a3"])])
        d     = float(item_row["d_param"])

        logit     = float(np.dot(a_vec, learner.theta_true)) + d
        p_star    = 1.0 / (1.0 + np.exp(-logit))
        p_correct = c_j + (1.0 - c_j) * p_star

        correct = learner.rng.random() < p_correct

        if correct:
            return True, int(item_row["correct_option"])

        # Choose a wrong option, weighted by misconception strength
        correct_opt = int(item_row["correct_option"])
        wrong_opts  = [o for o in [1, 2, 3, 4] if o != correct_opt]

        # Bias an option only when it activates a learner's shared ontology
        # slot; item-local option numbers are not a valid diagnostic identity.
        # Uses self._c_matrix (instance-level) to avoid class-state pollution.
        weights = np.ones(3)  # uniform across 3 wrong options
        for j, opt in enumerate(wrong_opts):
            for tag, strength in learner.misconception_true.items():
                option_tags = self._c_matrix.diagnose(str(item_row["item_id"]), opt).get("misconception_tags", [])
                if tag in option_tags:
                    weights[j] += strength * 2.0

        weights /= weights.sum()
        chosen_wrong = learner.rng.choice(wrong_opts, p=weights)
        return False, int(chosen_wrong)
