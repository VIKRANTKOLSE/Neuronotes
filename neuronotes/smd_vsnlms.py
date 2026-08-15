"""
smd_vsnlms.py — Semantic Variable-Step Normalized LMS Updater with Momentum
============================================================================
Module 4 of SMD-CC-MIRT-KL-CAT

Updates the learner's 3D ability vector (θ) and per-misconception state
after each response, using a step size that varies with:
  - error_class  (conceptual > arithmetic > slip)
  - repetition   (momentum boosts confidence on repeated errors)

Update rule
-----------
    gradient  = (correct - P_pred) * a_vec      (NLMS step)
    μ_eff     = base_lr * class_scale * momentum_factor / norm(a_vec)
    θ_new     = θ + μ_eff * gradient

Misconception state update
--------------------------
    For each active misconception: state[tag] += δ (exponential decay on correct)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Step-size multipliers per error class
ERROR_CLASS_SCALE = {
    "correct":           0.0,   # no misconception update on correct answer
    "conceptual_error":  1.0,   # full update
    "arithmetic_error":  0.4,   # smaller – don't penalise ability heavily
    "slip":              0.1,   # very small – likely one-off
    "unknown_error":     0.6,
}

BASE_LR          = 0.15   # learning rate for ability update
MOMENTUM_MAX     = 2.0    # maximum momentum multiplier
MOMENTUM_DECAY   = 0.8    # momentum decays each step (on correct answer)
MISC_STRENGTH    = 0.15   # misconception state increment per wrong response
MISC_DECAY       = 0.05   # misconception state decay per correct response


@dataclass
class LearnerState:
    """Mutable state for one simulated learner during a test session."""
    theta:            np.ndarray = field(default_factory=lambda: np.zeros(3))
    # concept_theta: per-concept ability (for prerequisite checks)
    concept_theta:    dict       = field(default_factory=dict)
    # misconception strengths: tag → float in [0, 1]
    misconception:    dict       = field(default_factory=dict)
    # per-misconception repeat counter for momentum
    repeat_count:     dict       = field(default_factory=dict)
    # full response history
    response_history: list       = field(default_factory=list)


class SMDVSNLMSUpdater:
    """Semantic Variable-Step Normalized LMS updater.

    Parameters
    ----------
    base_lr : float
        Base learning rate μ₀.
    momentum_max : float
        Maximum momentum multiplier (caps repeated-error boost).
    """

    def __init__(self,
                 base_lr:      float = BASE_LR,
                 momentum_max: float = MOMENTUM_MAX):
        self.base_lr      = base_lr
        self.momentum_max = momentum_max

    # ------------------------------------------------------------------
    def update(self,
               state:      LearnerState,
               item_id:    str,
               concept:    str,
               a_vec:      np.ndarray,
               d_param:    float,
               p_correct:  float,
               correct:    bool,
               error_class: str,
               misconception_tag: str,
               severity:   str = "medium") -> LearnerState:
        """Apply one SMD-VSNLMS update step.

        Parameters
        ----------
        state          : current LearnerState (mutated in-place)
        item_id        : item identifier (for logging)
        concept        : chemistry concept name
        a_vec          : discrimination vector (3-dim)
        d_param        : difficulty
        p_correct      : predicted probability of correct response
        correct        : whether learner answered correctly
        error_class    : 'correct','conceptual_error','arithmetic_error','slip'
        misconception_tag : tag from C-Matrix
        severity       : 'high','medium','low'
        """
        response_val = 1 if correct else 0
        residual     = response_val - p_correct

        # Determine class-based step scale
        # Use full step for correct answers, otherwise scale by error type
        ec_scale = 1.0 if correct else ERROR_CLASS_SCALE.get(error_class, 0.5)

        # Compute momentum multiplier for repeated misconceptions
        momentum = 1.0
        if not correct and misconception_tag != "none":
            cnt = state.repeat_count.get(misconception_tag, 0) + 1
            state.repeat_count[misconception_tag] = cnt
            momentum = min(1.0 + 0.3 * (cnt - 1), self.momentum_max)
        elif correct and misconception_tag != "none":
            state.repeat_count[misconception_tag] = max(
                0, state.repeat_count.get(misconception_tag, 0) - 1
            )

        # Effective step size (normalised by ||a||)
        a_norm = max(np.linalg.norm(a_vec), 1e-6)
        mu_eff = self.base_lr * ec_scale * momentum / a_norm

        # Ability update: θ ← θ + μ * residual * a
        state.theta = state.theta + mu_eff * residual * a_vec
        state.theta = np.clip(state.theta, -4.0, 4.0)

        # Per-concept ability: update the primary dimension's concept estimate
        from .cc_mirt import CONCEPT_DIM_MAP
        dim = CONCEPT_DIM_MAP.get(concept, 0)
        state.concept_theta[concept] = float(state.theta[dim])

        # Misconception state update
        if not correct and misconception_tag and misconception_tag != "none":
            sev_scale = {"high": 1.2, "medium": 1.0, "low": 0.6}.get(severity, 1.0)
            state.misconception[misconception_tag] = min(
                1.0,
                state.misconception.get(misconception_tag, 0.0) + MISC_STRENGTH * sev_scale
            )
        elif correct and misconception_tag and misconception_tag != "none":
            # Correct answer provides weak evidence against the misconception
            state.misconception[misconception_tag] = max(
                0.0,
                state.misconception.get(misconception_tag, 0.0) - MISC_DECAY
            )

        # Log response
        state.response_history.append({
            "item_id":          item_id,
            "concept":          concept,
            "correct":          correct,
            "error_class":      error_class,
            "misconception_tag": misconception_tag,
            "theta_after":      state.theta.tolist(),
            "mu_eff":           round(mu_eff, 5),
            "momentum":         round(momentum, 3),
        })

        return state
