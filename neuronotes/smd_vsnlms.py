"""Semantic variable-step normalized LMS updates backed by the z ontology.

For a response to option k of item j, the update is:

``delta_theta = eta * (residual * a_j + alpha * B @ z_jk) + mu * delta_prev``

``B`` is a 3-by-15 learnable semantic mapping. It starts from a small,
deterministic prior and is updated online from response residuals.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ERROR_CLASS_SCALE = {
    "correct": 0.0,
    "conceptual_error": 1.0,
    "arithmetic_error": 0.4,
    "slip": 0.1,
    "unknown_error": 0.6,
}

BASE_LR = 0.15
MOMENTUM_MAX = 2.0
MOMENTUM_DECAY = 0.8
MISC_STRENGTH = 0.15
MISC_DECAY = 0.05
N_SEMANTIC_DIMS = 15


@dataclass
class LearnerState:
    theta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    concept_theta: dict = field(default_factory=dict)
    misconception: dict = field(default_factory=dict)
    # Continuous posterior-like evidence for every shared ontology slot.
    # This is the score used for AUROC; the dict above is retained for routing.
    misconception_probs: np.ndarray = field(
        default_factory=lambda: np.full(N_SEMANTIC_DIMS, 0.10, dtype=float)
    )
    repeat_count: dict = field(default_factory=dict)
    response_history: list = field(default_factory=list)
    previous_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))


class SMDVSNLMSUpdater:
    """Update learner state using both MIRT residual and z-vector evidence."""

    def __init__(self,
                 base_lr: float = BASE_LR,
                 momentum_max: float = MOMENTUM_MAX,
                 semantic_alpha: float = 0.35,
                 semantic_matrix_lr: float = 0.02,
                 momentum: float = 0.2,
                 semantic_matrix: Optional[np.ndarray] = None):
        self.base_lr = base_lr
        self.momentum_max = momentum_max
        self.semantic_alpha = semantic_alpha
        self.semantic_matrix_lr = semantic_matrix_lr
        self.momentum = momentum
        if semantic_matrix is None:
            # Weak prior: every shared ontology slot initially maps to one
            # ability axis; observations refine these values online.
            self.B = np.zeros((3, N_SEMANTIC_DIMS), dtype=float)
            self.B[np.arange(N_SEMANTIC_DIMS) % 3, np.arange(N_SEMANTIC_DIMS)] = -0.05
        else:
            matrix = np.asarray(semantic_matrix, dtype=float)
            if matrix.shape != (3, N_SEMANTIC_DIMS):
                raise ValueError("semantic_matrix must have shape (3, 15)")
            self.B = matrix.copy()

    @staticmethod
    def _z_vector(z_vector: Optional[np.ndarray]) -> np.ndarray:
        if z_vector is None:
            return np.zeros(N_SEMANTIC_DIMS, dtype=float)
        vector = np.asarray(z_vector, dtype=float).reshape(-1)
        return np.pad(vector[:N_SEMANTIC_DIMS], (0, max(0, N_SEMANTIC_DIMS - len(vector))))

    @staticmethod
    def _tags(misconception_tag: str, misconception_tags: Optional[list[str]], z_vector: np.ndarray) -> list[str]:
        if misconception_tags:
            return [tag for tag in misconception_tags if tag not in {"none", "unknown_error"}]
        if misconception_tag not in {"none", "unknown_error"}:
            return [misconception_tag]
        return [f"z_{index:02d}" for index, value in enumerate(z_vector) if value > 0]

    def update(self,
               state: LearnerState,
               item_id: str,
               concept: str,
               a_vec: np.ndarray,
               d_param: float,
               p_correct: float,
               correct: bool,
               error_class: str,
               misconception_tag: str,
               severity: str = "medium",
               z_vector: Optional[np.ndarray] = None,
               misconception_tags: Optional[list[str]] = None) -> LearnerState:
        """Apply one ontology-aware SMD-VSNLMS update in place."""
        del d_param  # retained in the public API for backwards compatibility
        z = self._z_vector(z_vector)
        tags = self._tags(misconception_tag, misconception_tags, z)
        residual = (1 if correct else 0) - p_correct
        class_scale = 1.0 if correct else ERROR_CLASS_SCALE.get(error_class, 0.5)

        repetition = 1.0
        if not correct:
            for tag in tags:
                count = state.repeat_count.get(tag, 0) + 1
                state.repeat_count[tag] = count
                repetition = max(repetition, min(1.0 + 0.3 * (count - 1), self.momentum_max))
        else:
            for tag in tags:
                state.repeat_count[tag] = max(0, state.repeat_count.get(tag, 0) - 1)

        eta = self.base_lr * class_scale * repetition / max(np.linalg.norm(a_vec), 1e-6)
        semantic_term = np.zeros(3) if correct else self.semantic_alpha * (self.B @ z)
        delta = eta * (residual * a_vec + semantic_term) + self.momentum * state.previous_delta
        state.theta = np.clip(state.theta + delta, -4.0, 4.0)
        state.previous_delta = delta

        # Update B only when a misconception vector is actually observed.
        if not correct and np.any(z):
            self.B += self.semantic_matrix_lr * np.outer(residual * a_vec, z)
            self.B = np.clip(self.B, -1.0, 1.0)

        from .cc_mirt import CONCEPT_DIM_MAP
        dim = CONCEPT_DIM_MAP.get(concept, 0)
        state.concept_theta[concept] = float(state.theta[dim])

        # Continuous misconception evidence. A wrong distractor is positive
        # evidence only for the ontology slots it activates; a correct answer
        # supplies weak negative evidence across the current posterior rather
        # than turning any diagnosis into a binary label.
        if correct:
            state.misconception_probs *= (1.0 - MISC_DECAY)
        elif np.any(z):
            severity_scale = {"high": 1.2, "medium": 1.0, "low": 0.6}.get(severity, 1.0)
            update_rate = MISC_STRENGTH * severity_scale
            state.misconception_probs += update_rate * z * (1.0 - state.misconception_probs)
        state.misconception_probs = np.clip(state.misconception_probs, 0.001, 0.999)

        if not correct:
            severity_scale = {"high": 1.2, "medium": 1.0, "low": 0.6}.get(severity, 1.0)
            for tag in tags:
                index = int(tag.split("_")[-1]) if tag.startswith("z_") else None
                value = state.misconception_probs[index] if index is not None else 0.0
                state.misconception[tag] = float(value)
        elif tags:
            for tag in tags:
                index = int(tag.split("_")[-1]) if tag.startswith("z_") else None
                value = state.misconception_probs[index] if index is not None else 0.0
                state.misconception[tag] = float(value)

        state.response_history.append({
            "item_id": item_id,
            "concept": concept,
            "correct": correct,
            "error_class": error_class,
            "misconception_tags": tags,
            "z_vector": z.tolist(),
            "misconception_probs": state.misconception_probs.tolist(),
            "theta_after": state.theta.tolist(),
            "delta_theta": delta.tolist(),
            "eta": round(eta, 5),
            "repetition": round(repetition, 3),
        })
        return state
