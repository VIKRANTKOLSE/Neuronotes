"""
kl_cat.py — KL-Divergence MIRT Computerized Adaptive Testing Selector
======================================================================
Module 5 of SMD-CC-MIRT-KL-CAT

Item selection criterion (maximise expected utility U_j):

    U_j = w_kl  * E_KL(j)          # expected KL gain
         + w_pre * prereq_score(j)  # prerequisite relevance bonus
         + w_mis * diag_value(j)    # misconception diagnostic value bonus
         - w_rep * exposure_penalty(j)  # repetition / over-exposure penalty

Expected KL gain (over next response):
    E_KL = P_j * KL(post_1 || prior) + Q_j * KL(post_0 || prior)
where post_1 / post_0 are the approximate posterior theta distributions
after a correct / incorrect response.

Approximate posterior via a Gaussian update around current theta.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from scipy.stats import multivariate_normal

from .cc_mirt  import CCMIRT, CONCEPT_DIM_MAP
from .c_matrix import CMatrix

DATA_DIR = Path(__file__).parent.parent / "data"

# Default scoring weights
W_KL  = 0.50
W_PRE = 0.20
W_MIS = 0.20
W_REP = 0.10

THETA_SIGMA = 0.5   # assumed posterior spread for KL computation
MAX_EXPOSURE = 60   # hard cap on single-item exposure


class KLCAT:
    """KL-MIRT-CAT item selector.

    Parameters
    ----------
    items_clean_path : Path, optional
    concept_graph_path : Path, optional
    c_matrix : CMatrix, optional
    mirt : CCMIRT, optional
    weights : dict, optional  keys: kl, prereq, misc, rep
    use_prereq  : bool  — enable prerequisite relevance bonus
    use_misc    : bool  — enable misconception diagnostic bonus
    use_rep_pen : bool  — enable repetition penalty
    """

    def __init__(self,
                 items_clean_path: Optional[Path] = None,
                 concept_graph_path: Optional[Path] = None,
                 c_matrix:   Optional[CMatrix] = None,
                 mirt:       Optional[CCMIRT]  = None,
                 weights:    Optional[dict]    = None,
                 use_prereq:  bool = True,
                 use_misc:    bool = True,
                 use_rep_pen: bool = True):
        self.mirt    = mirt  or CCMIRT(concept_graph_path)
        self.c_mat   = c_matrix or CMatrix()
        self.use_prereq  = use_prereq
        self.use_misc    = use_misc
        self.use_rep_pen = use_rep_pen

        w = weights or {}
        self.w_kl  = w.get("kl",    W_KL)
        self.w_pre = w.get("prereq", W_PRE)
        self.w_mis = w.get("misc",  W_MIS)
        self.w_rep = w.get("rep",   W_REP)

        self._items: pd.DataFrame = pd.DataFrame()
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        if p.exists():
            self._items = pd.read_csv(p)
            # Pre-cache misconception diagnostic values (avoid recomputing per step)
            self._items = self._items.copy()
            self._items["_misc_diag"] = self._items.apply(
                lambda r: self.c_mat.misconception_diagnostic_value(
                    str(r["item_id"]), int(r["correct_option"])
                ), axis=1
            )

        # Exposure tracker: item_id -> count
        self._exposure: dict = {}

    # ------------------------------------------------------------------
    # KL divergence between two 3D Gaussians (diagonal cov)
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_diag_gaussian(mu1: np.ndarray, mu2: np.ndarray,
                           sigma: float = THETA_SIGMA) -> float:
        """KL(N(mu1,σI) || N(mu2,σI)) = ||mu1-mu2||² / (2σ²)"""
        return float(np.sum((mu1 - mu2) ** 2) / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------
    # Expected KL gain for a single item
    # ------------------------------------------------------------------
    def _expected_kl(self,
                     theta: np.ndarray,
                     a_vec: np.ndarray,
                     d: float,
                     c_j: float) -> float:
        p  = self.mirt.prob(theta, a_vec, d, c_j)
        q  = 1.0 - p
        lr = 0.3  # proxy update step (mirrors SMD-VSNLMS base_lr)

        # Approximate posterior means after correct / incorrect
        post_1 = np.clip(theta + lr * q * a_vec, -4, 4)
        post_0 = np.clip(theta - lr * p * a_vec, -4, 4)

        kl_1   = self._kl_diag_gaussian(post_1, theta)
        kl_0   = self._kl_diag_gaussian(post_0, theta)
        return float(p * kl_1 + q * kl_0)

    # ------------------------------------------------------------------
    # Prerequisite relevance: reward items whose concept is a direct prereq
    # of an unsatisfied target concept (theta near 0 in that dimension)
    # ------------------------------------------------------------------
    def _prereq_score(self, concept: str, theta: np.ndarray) -> float:
        dim    = CONCEPT_DIM_MAP.get(concept, 0)
        ability = float(theta[dim])
        # Higher bonus for items that probe weakly-mastered dimensions
        weakness = max(0.0, -ability)          # positive when theta < 0
        return min(weakness, 1.5)

    # ------------------------------------------------------------------
    # Exposure penalty
    # ------------------------------------------------------------------
    def _exposure_penalty(self, item_id: str, exposure_limit: int) -> float:
        count = self._exposure.get(item_id, 0)
        if count >= exposure_limit:
            return 1e6   # effectively infinite penalty → item excluded
        ratio = count / max(exposure_limit, 1)
        return ratio ** 2   # quadratic growth

    # ------------------------------------------------------------------
    # Select next item
    # ------------------------------------------------------------------
    def select(self,
               theta:           np.ndarray,
               seen_items:      set,
               concept_thetas:  dict,
               misconception_state: dict,
               active_misconception: Optional[str] = None) -> Optional[pd.Series]:
        """Return the best item row from items_clean or None if bank exhausted.

        Fully vectorised over the candidate pool for speed.
        """
        if self._items.empty:
            return None

        # Candidate pool: exclude seen items
        candidates = self._items[~self._items["item_id"].isin(seen_items)]
        if candidates.empty:
            return None

        # ---- Vectorised KL gain ----
        A = candidates[["a1", "a2", "a3"]].values          # (N, 3)
        D = candidates["d_param"].values                    # (N,)
        C = candidates["c_j"].values                        # (N,)

        logits  = A @ theta + D                             # (N,)
        p_star  = 1.0 / (1.0 + np.exp(-logits))
        P       = C + (1.0 - C) * p_star                   # (N,)
        Q       = 1.0 - P
        lr      = 0.3
        sigma2  = THETA_SIGMA ** 2

        # post_1 = theta + lr*Q*a,  post_0 = theta - lr*P*a  (broadcast)
        # KL = ||delta||^2 / (2*sigma^2)
        delta1  = lr * Q[:, None] * A                       # (N, 3)
        delta0  = -lr * P[:, None] * A
        kl_1    = np.sum(delta1 ** 2, axis=1) / (2 * sigma2)
        kl_0    = np.sum(delta0 ** 2, axis=1) / (2 * sigma2)
        kl_gain = P * kl_1 + Q * kl_0                      # (N,)

        # ---- Prereq bonus (vectorised via concept dim map) ----
        if self.use_prereq:
            dims    = candidates["concept"].map(
                lambda c: CONCEPT_DIM_MAP.get(c, 0)
            ).values                                        # (N,)
            ability = theta[dims]                           # (N,)
            pre     = np.clip(-ability, 0.0, 1.5)
        else:
            pre = np.zeros(len(candidates))

        # ---- Misconception diagnostic bonus (use pre-cached column) ----
        if self.use_misc and "_misc_diag" in candidates.columns:
            mis = candidates["_misc_diag"].values.astype(float)
            if active_misconception:
                # Only boost if we have an active target — skip expensive apply
                boost = np.zeros(len(candidates))
                for idx, (_, r) in enumerate(candidates.iterrows()):
                    wrongs = self.c_mat.all_wrong_options(
                        str(r["item_id"]), int(r["correct_option"])
                    )
                    if any(w["misconception_tag"] == active_misconception for w in wrongs):
                        boost[idx] = 0.5
                mis = mis + boost
        else:
            mis = np.zeros(len(candidates))

        # ---- Repetition penalty (vectorised) ----
        if self.use_rep_pen:
            counts  = candidates["item_id"].map(
                lambda iid: self._exposure.get(str(iid), 0)
            ).values.astype(float)
            limits  = candidates["item_exposure_limit"].values.astype(float)
            ratio   = counts / np.maximum(limits, 1)
            rep     = ratio ** 2
            # Hard-exclude over-exposed (set score to -inf)
            over    = counts >= limits
            rep[over] = 1e6
        else:
            rep = np.zeros(len(candidates))

        # ---- Combined utility ----
        scores   = (self.w_kl * kl_gain + self.w_pre * pre
                    + self.w_mis * mis - self.w_rep * rep)
        best_idx = int(np.argmax(scores))
        best_row = candidates.iloc[best_idx]

        # Increment exposure
        bid = str(best_row["item_id"])
        self._exposure[bid] = self._exposure.get(bid, 0) + 1
        return best_row

    # ------------------------------------------------------------------
    def reset_exposure(self):
        self._exposure.clear()
