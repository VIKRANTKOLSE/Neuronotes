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

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any

from .cc_mirt  import CCMIRT, CONCEPT_DIM_MAP, N_DIMS, parse_a_vector
from .c_matrix import CMatrix
from .t_matrix import ColdStartRouter
from .dynamic_c import DynamicC

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
    dynamic_c : DynamicC, optional — dynamic guessing floor provider
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
                 dynamic_c:  Optional[DynamicC] = None,
                 weights:    Optional[dict]    = None,
                 use_prereq:  bool = True,
                 use_misc:    bool = True,
                 use_rep_pen: bool = True,
                 use_cold_start: bool = True):
        self.mirt       = mirt or CCMIRT(concept_graph_path)
        self.c_mat      = c_matrix or CMatrix()
        self.dynamic_c  = dynamic_c
        self.use_prereq  = use_prereq
        self.use_misc    = use_misc
        self.use_rep_pen = use_rep_pen
        self.use_cold_start = use_cold_start
        self.cold_start = ColdStartRouter(concept_graph_path)

        w = weights or {}
        self.w_kl  = w.get("kl",    W_KL)
        self.w_pre = w.get("prereq", W_PRE)
        self.w_mis = w.get("misc",  W_MIS)
        self.w_rep = w.get("rep",   W_REP)

        self._items: pd.DataFrame = pd.DataFrame()
        self._A_cache: Optional[np.ndarray] = None  # pre-parsed (N, 58) matrix
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        if p.exists():
            self._items = pd.read_csv(p)
            self._item_ids = self._items["item_id"].values.astype(str)
            self._d_params = self._items["d_param"].values.astype(float)
            self._c_params = self._items["c_j"].values.astype(float)
            self._concept_dims = np.array([CONCEPT_DIM_MAP.get(c, 0) for c in self._items["concept"].values], dtype=int)
            self._exposure_limits = self._items["item_exposure_limit"].values.astype(float)
            # Pre-parse 58D a_vectors into numpy array cache
            self._A_cache = np.stack(
                self._items["a_vector"].apply(parse_a_vector).values
            )  # (N, N_DIMS)
            # Pre-cache misconception diagnostic values (avoid recomputing per step)
            self._items = self._items.copy()
            self._items["_misc_diag"] = self._items.apply(
                lambda r: self.c_mat.misconception_diagnostic_value(
                    str(r["item_id"]), int(r["correct_option"])
                ), axis=1
            )
            self._misc_diag_cache = self._items["_misc_diag"].values.astype(float)
            # Pre-compute mapping from misconception_tag -> set of item_ids for O(1) active misconception bonus
            self._items_with_misc: dict[str, set[str]] = {}
            for _, r in self._items.iterrows():
                iid = str(r["item_id"])
                opt = int(r["correct_option"])
                for w in self.c_mat.all_wrong_options(iid, opt):
                    tag = w.get("misconception_tag")
                    if tag and tag not in {"none", "unknown_error"}:
                        self._items_with_misc.setdefault(tag, set()).add(iid)

        # Exposure tracker: item_id -> count
        self._exposure: dict = {}

    # ------------------------------------------------------------------
    # KL divergence between two N_DIMS-D Gaussians (diagonal cov)
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

        if hasattr(self.mirt, "diffusion_matrix"):
            effective_a = self.mirt.diffusion_matrix @ a_vec
        else:
            effective_a = a_vec

        # Approximate posterior means after correct / incorrect
        post_1 = np.clip(theta + lr * q * effective_a, -4, 4)
        post_0 = np.clip(theta - lr * p * effective_a, -4, 4)

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
               active_misconception: Optional[str] = None,
               repeated_items: Optional[set] = None,
               repeat_discount: float = 0.5,
               tested_dims: Optional[Any] = None) -> Optional[pd.Series]:
        """Return the best item row from items_clean or None if bank exhausted.

        Fully vectorised over the candidate pool for speed.
        """
        if self._items.empty:
            return None

        if not seen_items:
            candidate_indices = np.arange(len(self._items))
        else:
            candidate_indices = np.flatnonzero(~np.isin(self._item_ids, list(seen_items)))

        if len(candidate_indices) == 0:
            return None

        # ---- Vectorised KL gain ----
        A = self._A_cache[candidate_indices]                 # (N, N_DIMS)
        D = self._d_params[candidate_indices]                # (N,)
        cand_ids = self._item_ids[candidate_indices]
        if self.dynamic_c is not None:
            C = np.array([self.dynamic_c.get(iid) for iid in cand_ids], dtype=float)
        else:
            C = self._c_params[candidate_indices]            # (N,)

        logits  = A @ theta + D                              # (N,)
        p_star  = 1.0 / (1.0 + np.exp(-logits))
        P       = C + (1.0 - C) * p_star                    # (N,)
        Q       = 1.0 - P
        lr      = 0.3
        sigma2  = THETA_SIGMA ** 2

        # post_1 = theta + lr*Q*a,  post_0 = theta - lr*P*a  (broadcast)
        # KL = ||delta||^2 / (2*sigma^2)
        delta1  = lr * Q[:, None] * A                        # (N, N_DIMS)
        delta0  = -lr * P[:, None] * A
        kl_1    = np.sum(delta1 ** 2, axis=1) / (2 * sigma2)
        kl_0    = np.sum(delta0 ** 2, axis=1) / (2 * sigma2)
        kl_gain = P * kl_1 + Q * kl_0                       # (N,)

        # ---- Prereq bonus (vectorised via pre-cached concept dim map) ----
        if self.use_prereq:
            dims    = self._concept_dims[candidate_indices]  # (N,)
            ability = theta[dims]                            # (N,)
            pre     = np.clip(-ability, 0.0, 1.5)
        else:
            pre = np.zeros(len(candidate_indices))

        # ---- Misconception diagnostic bonus (use pre-cached column) ----
        if self.use_misc:
            mis = self._misc_diag_cache[candidate_indices].copy()
            if active_misconception and hasattr(self, "_items_with_misc") and active_misconception in self._items_with_misc:
                target_items = self._items_with_misc[active_misconception]
                boost = np.isin(cand_ids, list(target_items)).astype(float) * 0.6
                mis = mis + boost
        else:
            mis = np.zeros(len(candidate_indices))

        # ---- Repetition penalty (vectorised) ----
        if self.use_rep_pen:
            counts  = np.array([self._exposure.get(iid, 0) for iid in cand_ids], dtype=float)
            limits  = self._exposure_limits[candidate_indices]
            ratio   = counts / np.maximum(limits, 1)
            rep     = ratio ** 2
            # Hard-exclude over-exposed (set score to -inf)
            over    = counts >= limits
            rep[over] = 1e6
        else:
            rep = np.zeros(len(candidate_indices))

        # ---- Cold-start routing (only before any response evidence) ----
        cold = np.zeros(len(candidate_indices))
        if self.use_cold_start and not seen_items:
            target_concept = self.cold_start.recommend_concept(set())
            if target_concept:
                cand_concepts = self._items["concept"].values[candidate_indices]
                cold = (cand_concepts == target_concept).astype(float)

        # ---- Combined utility ----
        scores   = (self.w_kl * kl_gain + self.w_pre * pre
                    + self.w_mis * mis - self.w_rep * rep + 0.5 * cold)

        # Coverage bonus: encourage testing dimensions that haven't been tested yet in this session
        if tested_dims is not None and len(candidate_indices) > 0:
            dims = self._concept_dims[candidate_indices]
            # If tested_dims is a boolean mask of shape (N_DIMS,)
            if isinstance(tested_dims, np.ndarray) and tested_dims.dtype == bool:
                untested_indicator = (~tested_dims[dims]).astype(float)
            elif isinstance(tested_dims, (set, list, tuple)):
                untested_indicator = np.array([1.0 if d not in tested_dims else 0.0 for d in dims], dtype=float)
            else:
                untested_indicator = np.zeros(len(candidate_indices))
            # Modest exploration bonus to explore new dimensions
            scores += 0.25 * untested_indicator

        # SpacedCAT: discount repeated items so unseen items rank higher
        if repeated_items:
            is_rep = np.isin(cand_ids, list(repeated_items))
            # Multiply positive utilities by discount, scale negative if needed
            scores = np.where(is_rep, scores * repeat_discount, scores)

        best_pos = int(np.argmax(scores))
        best_idx = candidate_indices[best_pos]
        best_row = self._items.iloc[best_idx]

        # Increment exposure
        bid = str(best_row["item_id"])
        self._exposure[bid] = self._exposure.get(bid, 0) + 1
        return best_row

    # ------------------------------------------------------------------
    def reset_exposure(self):
        self._exposure.clear()
