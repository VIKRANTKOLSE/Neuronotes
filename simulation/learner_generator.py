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
from dataclasses import dataclass, field
from typing import Optional

# Reproducible seeds
DEFAULT_SEED = 42

# Profile archetypes
PROFILES = {
    "strong_all":    {"theta_mean": [1.5,  1.5,  1.5],  "theta_std": [0.3, 0.3, 0.3]},
    "weak_all":      {"theta_mean": [-1.5, -1.5, -1.5], "theta_std": [0.3, 0.3, 0.3]},
    "strong_found":  {"theta_mean": [1.2,  0.0, -0.5],  "theta_std": [0.2, 0.4, 0.4]},
    "strong_bonding":{"theta_mean": [0.5,  1.2,  0.0],  "theta_std": [0.4, 0.2, 0.4]},
    "strong_coord":  {"theta_mean": [-0.2, 0.5,  1.5],  "theta_std": [0.4, 0.4, 0.2]},
    "mixed":         {"theta_mean": [0.0,  0.0,  0.0],  "theta_std": [0.8, 0.8, 0.8]},
}

# Misconception archetypes (tag suffix → strength range)
# Updated to match exact tags generated in data/item_options.csv
MISCONCEPTION_ARCHETYPES = {
    "effective_nuclear_charge_opt1_error":       (0.5, 0.9),
    "shielding_effect_opt2_error":               (0.4, 0.8),
    "orbital_penetration_opt3_error":            (0.3, 0.7),
    "electronelectron_repulsion_opt1_error":     (0.5, 0.9),
    "energy_level_splitting_in_atomic_orbitals_opt2_error": (0.3, 0.6),
}


@dataclass
class SyntheticLearner:
    """One simulated learner with ground-truth state."""
    learner_id:          str
    profile:             str
    theta_true:          np.ndarray          # ground-truth 3D ability
    theta_init:          np.ndarray          # starting estimate (θ₀ = [0,0,0])
    misconception_true:  dict                 # tag → true strength [0,1]
    concept_mastery:     dict                 # concept → True/False (prerequisite-aware)
    rng:                 np.random.Generator  # private rng for response simulation


class LearnerGenerator:
    """Generates reproducible synthetic learner populations.

    Parameters
    ----------
    n_learners : int
        Total number of learners to generate.
    seed : int
        Master random seed.
    profile_distribution : dict, optional
        Maps profile name → fraction. Must sum to 1.0.
    """

    def __init__(self,
                 n_learners: int = 200,
                 seed:       int = DEFAULT_SEED,
                 profile_distribution: Optional[dict] = None):
        self.n_learners = n_learners
        self.rng = np.random.default_rng(seed)

        # Default equal-weight distribution across all profiles
        default_dist = {k: 1.0 / len(PROFILES) for k in PROFILES}
        self.profile_dist = profile_distribution or default_dist

    # ------------------------------------------------------------------
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
            for tag, (lo, hi) in MISCONCEPTION_ARCHETYPES.items():
                # Probability of having this misconception ∝ weakness in dim 0 or 1
                weakness = max(0.0, -float(np.mean(theta_true[:2]))) / 3.0
                if self.rng.random() < (0.2 + 0.6 * weakness):
                    strength = float(self.rng.uniform(lo, hi))
                    misc_state[tag] = strength

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

        logit    = float(np.dot(a_vec, learner.theta_true)) + d
        p_star   = 1.0 / (1.0 + np.exp(-logit))
        p_correct = c_j + (1.0 - c_j) * p_star

        correct = learner.rng.random() < p_correct

        if correct:
            return True, int(item_row["correct_option"])

        # Choose a wrong option, weighted by misconception strength
        correct_opt = int(item_row["correct_option"])
        wrong_opts  = [o for o in [1, 2, 3, 4] if o != correct_opt]

        # If learner has a relevant misconception, bias toward that option
        weights = np.ones(3)  # uniform across 3 wrong options
        for j, opt in enumerate(wrong_opts):
            for tag, strength in learner.misconception_true.items():
                if str(opt) in tag or str(item_row.get("item_id", ""))[:5] in tag:
                    weights[j] += strength * 2.0

        weights /= weights.sum()
        chosen_wrong = learner.rng.choice(wrong_opts, p=weights)
        return False, int(chosen_wrong)
