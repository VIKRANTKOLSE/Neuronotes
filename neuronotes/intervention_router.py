"""
intervention_router.py — Deterministic Rule-Based Intervention Router
=====================================================================
Module 6 of SMD-CC-MIRT-KL-CAT

Maps (error_class, misconception_tag, repeat_count, theta_dim) →
intervention type + metadata.

Intervention types
------------------
  NORMAL_Q       — Continue with next adaptive question (no special intervention)
  HINT           — Provide a hint about the current concept
  EXPLANATION    — Show the expert-approved explanation for the selected option
  PREREQ_Q       — Route to a prerequisite concept question
  CONTRAST_Q     — Provide a contrastive example to correct the misconception
  REMEDIAL_SET   — Assign a short remedial practice set (3–5 items)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Thresholds
REPEAT_HINT_THRESH      = 1   # 1st wrong → explanation
REPEAT_CONTRAST_THRESH  = 2   # 2nd consecutive same-misconception → contrast
REPEAT_REMEDIAL_THRESH  = 3   # 3rd+ → remedial set
THETA_LOW_THRESH        = -1.0  # ability below which prereq question is triggered


@dataclass
class Intervention:
    """Represents a recommended intervention action."""
    intervention_type: str      # one of the types listed above
    message:           str      # human-readable pedagogical message
    target_concept:    Optional[str] = None   # for PREREQ_Q and REMEDIAL_SET
    misconception_tag: Optional[str] = None


class InterventionRouter:
    """Deterministic rule-based router.

    Usage
    -----
        router = InterventionRouter(concept_graph_path=...)
        iv = router.route(
            error_class='conceptual_error',
            misconception_tag='effective_nuclear_charge_opt1_error',
            repeat_count=2,
            concept='Shielding Effect',
            theta=np.array([-0.5, 0.2, 0.1]),
            rationale='Incorrect. Shielding reduces the nuclear pull...',
        )
    """

    def __init__(self, concept_graph_path=None):
        self._prereq_map: dict[str, list[str]] = {}
        if concept_graph_path is not None:
            self._load_prereqs(concept_graph_path)

    def _load_prereqs(self, path):
        import pandas as pd
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                tgt = str(row["target_concept"])
                src = str(row["source_concept"])
                self._prereq_map.setdefault(tgt, []).append(src)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def route(self,
              error_class:      str,
              misconception_tag: str,
              repeat_count:     int,
              concept:          str,
              theta:            "np.ndarray",
              rationale:        str = "",
              use_interventions: bool = True) -> Intervention:
        """Route to the appropriate intervention.

        Parameters
        ----------
        error_class       : from CMatrix (conceptual_error, arithmetic_error, slip, correct)
        misconception_tag : from CMatrix
        repeat_count      : how many consecutive times this misconception appeared
        concept           : current concept name
        theta             : 3D ability vector (numpy array)
        rationale         : expert rationale text for the wrong option
        use_interventions : if False, always returns NORMAL_Q (ablation mode)
        """
        import numpy as np

        if error_class == "correct":
            return Intervention(
                intervention_type="NORMAL_Q",
                message="Correct! Proceeding to next question.",
            )

        if not use_interventions:
            return Intervention(
                intervention_type="NORMAL_Q",
                message="[Interventions disabled] Proceeding.",
            )

        # ---- slip: very light touch ----
        if error_class == "slip":
            return Intervention(
                intervention_type="HINT",
                message=(
                    f"You may have made a small slip on '{concept}'. "
                    "Take a moment to review your working."
                ),
                misconception_tag=misconception_tag,
            )

        # ---- arithmetic error ----
        if error_class == "arithmetic_error":
            return Intervention(
                intervention_type="HINT",
                message=(
                    f"The calculation step in '{concept}' needs attention. "
                    f"Hint: {rationale[:200] if rationale else 'check your arithmetic.'}"
                ),
                misconception_tag=misconception_tag,
            )

        # ---- conceptual error: escalating response ----
        # Check if prerequisite ability is low → route to prereq
        from .cc_mirt import CONCEPT_DIM_MAP
        dim    = CONCEPT_DIM_MAP.get(concept, 0)
        theta_val = float(theta[dim]) if hasattr(theta, "__len__") else float(theta)
        prereqs = self._prereq_map.get(concept, [])

        if theta_val < THETA_LOW_THRESH and prereqs:
            # Find the immediate prerequisite with the lowest ability
            weakest_prereq = prereqs[0]
            lowest_theta = float('inf')
            for pr in prereqs:
                p_dim = CONCEPT_DIM_MAP.get(pr, 0)
                p_theta = float(theta[p_dim]) if hasattr(theta, "__len__") else float(theta)
                if p_theta < lowest_theta:
                    lowest_theta = p_theta
                    weakest_prereq = pr

            return Intervention(
                intervention_type="PREREQ_Q",
                message=(
                    f"Your foundation on '{weakest_prereq}' needs strengthening before "
                    f"tackling '{concept}'. Let's revisit the prerequisite."
                ),
                target_concept=weakest_prereq,
                misconception_tag=misconception_tag,
            )

        if repeat_count >= REPEAT_REMEDIAL_THRESH:
            return Intervention(
                intervention_type="REMEDIAL_SET",
                message=(
                    f"You've made this error {repeat_count} times on '{concept}'. "
                    "A short remedial practice set has been queued to consolidate this concept."
                ),
                target_concept=concept,
                misconception_tag=misconception_tag,
            )

        if repeat_count >= REPEAT_CONTRAST_THRESH:
            return Intervention(
                intervention_type="CONTRAST_Q",
                message=(
                    f"Let's compare the correct and incorrect reasoning for '{concept}': "
                    f"{rationale[:300] if rationale else 'See the explanation below.'}"
                ),
                misconception_tag=misconception_tag,
            )

        # First instance: show explanation
        return Intervention(
            intervention_type="EXPLANATION",
            message=(
                f"'{concept}' explanation: "
                f"{rationale[:400] if rationale else 'Review the concept explanation.'}"
            ),
            misconception_tag=misconception_tag,
        )
