"""Semantic variable-step normalized LMS updates backed by the z ontology.

For a response to option k of item j, the update is:

``delta_theta = eta * (residual * a_j + alpha * B @ z_jk) + mu * delta_prev``

``B`` is a 58-by-15 learnable semantic mapping. It starts from a small,
deterministic prior and is updated online from response residuals.
"""

from dataclasses import dataclass, field
from typing import Optional, NamedTuple

import numpy as np

from .cc_mirt import N_DIMS

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

# Hyperparameters for residual-variance dynamic learning rate (eta_s)
BETA: float = 0.9      # Forgetting factor for the exponential moving average
ETA_MAX: float = 0.5   # Upper bound learning rate for consistent performance
ETA_MIN: float = 0.05  # Lower bound learning rate to guarantee non-zero learning
RHO: float = 10.0      # Scaling sensitivity factor


class DynamicLearningRateResult(NamedTuple):
    """Container for state transition output: (updated_variance, current_learning_rate)."""
    updated_variance: float
    current_learning_rate: float


def compute_dynamic_learning_rate(
    current_residual: float,
    previous_variance: float,
    beta: float = BETA,
    eta_max: float = ETA_MAX,
    eta_min: float = ETA_MIN,
    rho: float = RHO,
) -> DynamicLearningRateResult:
    """Compute adaptive step size eta_s based on residual variance EMA.

    Protects the student's 58-dimensional ability vector from erratic updates (e.g., wild guessing).

    Mathematical Equations:
    1. Variance EMA:
       v_t = beta * v_{t-1} + (1 - beta) * (r_jk ** 2)
    2. Adaptive Step Size:
       eta_s = max(eta_min, eta_max / (1 + rho * v_t))

    Parameters
    ----------
    current_residual : float
        The psychometric error r_jk from current prediction.
    previous_variance : float
        The running variance state v_{t-1}. Must be non-negative.
    beta : float
        Forgetting factor for EMA (default: 0.9).
    eta_max : float
        Maximum learning rate (default: 0.5).
    eta_min : float
        Minimum learning rate (default: 0.05).
    rho : float
        Sensitivity factor (default: 10.0).

    Returns
    -------
    DynamicLearningRateResult
        (updated_variance: float, current_learning_rate: float)
    """
    import math
    if not isinstance(current_residual, (int, float)) or not math.isfinite(current_residual):
        raise TypeError(f"current_residual must be a finite float, got {current_residual}")
    if not isinstance(previous_variance, (int, float)) or not math.isfinite(previous_variance) or previous_variance < 0:
        raise TypeError(f"previous_variance must be a non-negative finite float, got {previous_variance}")

    # 1. Variance EMA: v_t = beta * v_{t-1} + (1 - beta) * (r_jk ** 2)
    updated_variance = float(beta * previous_variance + (1.0 - beta) * (float(current_residual) ** 2))

    # 2. Adaptive Step Size: eta_s = max(eta_min, eta_max / (1 + rho * v_t))
    denominator = 1.0 + rho * updated_variance
    raw_eta = eta_max / denominator
    current_learning_rate = float(max(eta_min, min(eta_max, raw_eta)))

    return DynamicLearningRateResult(
        updated_variance=updated_variance,
        current_learning_rate=current_learning_rate,
    )


@dataclass
class LearnerState:
    theta: np.ndarray = field(default_factory=lambda: np.zeros(N_DIMS))
    concept_theta: dict = field(default_factory=dict)
    misconception: dict = field(default_factory=dict)
    # Continuous posterior-like evidence for every shared ontology slot.
    # This is the score used for AUROC; the dict above is retained for routing.
    misconception_probs: np.ndarray = field(
        default_factory=lambda: np.full(N_SEMANTIC_DIMS, 0.10, dtype=float)
    )
    repeat_count: dict = field(default_factory=dict)
    response_history: list = field(default_factory=list)
    previous_delta: np.ndarray = field(default_factory=lambda: np.zeros(N_DIMS))
    residual_variance: float = 0.0


class SMDVSNLMSUpdater:
    """Update learner state using both MIRT residual and z-vector evidence."""

    def __init__(self,
                 base_lr: float = BASE_LR,
                 momentum_max: float = MOMENTUM_MAX,
                 semantic_alpha: float = 0.35,
                 semantic_matrix_lr: float = 0.02,
                 momentum: float = 0.2,
                 semantic_matrix: Optional[np.ndarray] = None,
                 use_dynamic_lr: bool = True,
                 total_session_length: int = 30):
        self.base_lr = base_lr
        self.momentum_max = momentum_max
        self.semantic_alpha = semantic_alpha
        self.semantic_matrix_lr = semantic_matrix_lr
        self.momentum = momentum
        self.use_dynamic_lr = use_dynamic_lr
        self.total_session_length = total_session_length
        if semantic_matrix is None:
            # Weak prior: every shared ontology slot initially maps to one
            # ability axis; observations refine these values online.
            self.B = np.zeros((N_DIMS, N_SEMANTIC_DIMS), dtype=float)
            self.B[np.arange(N_SEMANTIC_DIMS) % N_DIMS, np.arange(N_SEMANTIC_DIMS)] = -0.05
        else:
            matrix = np.asarray(semantic_matrix, dtype=float)
            if matrix.shape != (N_DIMS, N_SEMANTIC_DIMS):
                raise ValueError(f"semantic_matrix must have shape ({N_DIMS}, {N_SEMANTIC_DIMS})")
            self.B = matrix.copy()
        self._initial_B = self.B.copy()

    def reset(self) -> None:
        """Reset mutable internal matrices to initial prior for an isolated learner session."""
        self.B = self._initial_B.copy()

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
               misconception_tags: Optional[list[str]] = None,
               z_mask: Optional[np.ndarray] = None,
               total_session_length: Optional[int] = None,
               t: Optional[int] = None) -> LearnerState:
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

        if self.use_dynamic_lr:
            dlr = compute_dynamic_learning_rate(
                current_residual=residual,
                previous_variance=getattr(state, "residual_variance", 0.0),
            )
            state.residual_variance = dlr.updated_variance
            step_base_lr = dlr.current_learning_rate
        else:
            step_base_lr = self.base_lr

        step_t = len(state.response_history) if t is None else int(t)
        import math
        alpha_t = self.semantic_alpha / math.sqrt(1.0 + 0.05 * step_t)

        eta = step_base_lr * class_scale * repetition / max(np.linalg.norm(a_vec), 1e-6)
        semantic_term = np.zeros(N_DIMS) if correct else alpha_t * (self.B @ z)
        delta = eta * (residual * a_vec + semantic_term) + self.momentum * state.previous_delta
        state.theta = np.clip(state.theta + delta, -4.0, 4.0)
        state.previous_delta = delta

        # Update B symmetrically with centered residual on both correct and incorrect:
        # On correct: positive reinforcement for avoiding the distractors present on the item (z_mask)
        # On incorrect: negative adjustment for selecting the specific distractor trap (z)
        if correct:
            if z_mask is not None and np.any(z_mask):
                mask_vec = self._z_vector(z_mask)
                self.B += self.semantic_matrix_lr * np.outer(residual * a_vec, mask_vec)
                self.B = np.clip(self.B, -1.0, 1.0)
        elif np.any(z):
            self.B += self.semantic_matrix_lr * np.outer(residual * a_vec, z)
            self.B = np.clip(self.B, -1.0, 1.0)

        from .cc_mirt import CONCEPT_DIM_MAP
        dim = CONCEPT_DIM_MAP.get(concept, 0)
        state.concept_theta[concept] = float(state.theta[dim])

        # Continuous misconception evidence. A wrong distractor is positive
        # evidence only for the ontology slots it activates; a correct answer
        # decays only the specific misconception traps present on that item.
        if correct:
            N = total_session_length if total_session_length is not None else getattr(self, "total_session_length", 30)
            lambda_decay = min(0.05, 1.5 / max(int(N), 1))
            if z_mask is not None:
                mask = np.asarray(z_mask, dtype=float).reshape(-1)
                mask = np.pad(mask[:N_SEMANTIC_DIMS], (0, max(0, N_SEMANTIC_DIMS - len(mask))))
            else:
                mask = np.ones(N_SEMANTIC_DIMS, dtype=float)
            state.misconception_probs *= (1.0 - lambda_decay * mask)
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
