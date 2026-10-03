"""
learner_generator.py — Synthetic Learner Population Generator
==============================================================
Generates N simulated learners with:
  - Known 58D ability vectors (θ_true)
  - Prerequisite-linked mastery patterns
  - Known misconception states per concept
  - Distractor-selection probabilities matching their misconceptions

Each learner is a dict with full ground-truth state for metric evaluation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from neuronotes.cc_mirt import N_DIMS, CONCEPT_DIM_MAP, parse_a_vector
from neuronotes.c_matrix import CMatrix

# Reproducible seeds
DEFAULT_SEED = 42
# The populated chemistry ontology currently defines four semantic families.
# Vectors are padded to the C-matrix's 15-slot schema below so inactive slots
# are excluded rather than treated as unobservable positive labels.
N_SEMANTIC_DIMS = 4

# Profile archetypes across 4 concept tiers:
#   Tier 1: 10 concepts (indices 0–9)
#   Tier 2: 13 concepts (indices 10–22)
#   Tier 3: 17 concepts (indices 23–39)
#   Tier 4: 18 concepts (indices 40–57)
# Total: 58 dimensions
def _make_profile_vectors(tier_means: tuple[float, float, float, float],
                          tier_stds: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)) -> dict:
    m = np.concatenate([
        np.full(10, tier_means[0]),
        np.full(13, tier_means[1]),
        np.full(17, tier_means[2]),
        np.full(18, tier_means[3]),
    ])
    s = np.concatenate([
        np.full(10, tier_stds[0]),
        np.full(13, tier_stds[1]),
        np.full(17, tier_stds[2]),
        np.full(18, tier_stds[3]),
    ])
    return {"theta_mean": m, "theta_std": s}


PROFILES = {
    "strong_all":     _make_profile_vectors((1.5, 1.5, 1.5, 1.5), (0.3, 0.3, 0.3, 0.3)),
    "weak_all":       _make_profile_vectors((-1.5, -1.5, -1.5, -1.5), (0.3, 0.3, 0.3, 0.3)),
    "strong_found":   _make_profile_vectors((1.2, 0.5, 0.0, -0.5), (0.2, 0.3, 0.4, 0.4)),
    "strong_bonding": _make_profile_vectors((0.5, 1.2, 1.0, 0.0), (0.4, 0.2, 0.3, 0.4)),
    "strong_coord":   _make_profile_vectors((0.0, 0.3, 0.8, 1.5), (0.4, 0.4, 0.3, 0.2)),
    "mixed":          _make_profile_vectors((0.0, 0.0, 0.0, 0.0), (0.8, 0.8, 0.8, 0.8)),
}


@dataclass
class SyntheticLearner:
    """One simulated learner with ground-truth state."""
    learner_id:                 str
    profile:                    str
    theta_true:                 np.ndarray          # ground-truth 58D ability
    theta_init:                 np.ndarray          # starting estimate (θ₀ = zeros(58))
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

            # Sample true ability (58 dimensions)
            theta_true = np.clip(
                self.rng.normal(spec["theta_mean"], spec["theta_std"]),
                -3.0, 3.0
            )

            # Sample misconceptions: learners with low foundational ability (Tier 1)
            # are more likely to have strong misconceptions
            misc_state = {}
            misc_vector = np.zeros(N_SEMANTIC_DIMS, dtype=int)
            weakness = max(0.0, -float(np.mean(theta_true[:10]))) / 3.0

            # Each ontology dimension receives independent ground truth so
            # AUROC can be evaluated as a real multilabel ranking problem.
            for dim in range(N_SEMANTIC_DIMS):
                prevalence = min(0.08 + 0.52 * weakness + 0.02 * (dim % 3), 0.80)
                if self.rng.random() < prevalence:
                    misc_vector[dim] = 1
                    misc_state[f"z_{dim:02d}"] = float(self.rng.uniform(0.3, 0.9))

            # Concept mastery: a concept is "mastered" if its primary dimension ability > 0
            concept_mastery = {
                concept_name: bool(theta_true[dim] > 0)
                for concept_name, dim in CONCEPT_DIM_MAP.items()
            }

            learners.append(SyntheticLearner(
                learner_id=f"S{self.seed}_L{i:04d}",
                profile=profile,
                theta_true=theta_true,
                theta_init=np.zeros(N_DIMS),
                misconception_true=misc_state,
                misconception_true_vector=np.pad(misc_vector, (0, 15 - N_SEMANTIC_DIMS)),
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
        if "a_vector" in item_row:
            a_vec = parse_a_vector(item_row["a_vector"])
        elif all(f"a{k}" in item_row for k in range(1, 4)):
            a_vec = np.zeros(N_DIMS)
            a_vec[0] = float(item_row["a1"])
            a_vec[1] = float(item_row["a2"])
            a_vec[2] = float(item_row["a3"])
        else:
            a_vec = np.zeros(N_DIMS)

        d = float(item_row["d_param"])

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
