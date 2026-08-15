"""B5 — KL-MIRT-CAT without Misconception Modelling.

Same KL-divergence selection as the full Neuronotes system but:
  - No misconception diagnostic bonus (w_mis = 0)
  - No prerequisite relevance bonus (w_pre = 0)
  - Fixed c = 0.25 (no dynamic guessing)
  - No repetition penalty

Fully vectorised with numpy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
FIXED_C  = 0.25
LR       = 0.3
SIGMA2   = 0.5 ** 2


class KLMIRTNoMiscSelector:
    """KL-MIRT-CAT without misconception or prerequisite features. Vectorised."""

    name = "B5_KL_MIRT_NoMisc"

    def __init__(self, items_clean_path: Optional[Path] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        pool = self._items[~self._items["item_id"].isin(seen_items)]
        if pool.empty:
            return None
        A      = pool[["a1", "a2", "a3"]].values          # (N, 3)
        D      = pool["d_param"].values                    # (N,)
        logits = A @ theta + D
        p_star = 1.0 / (1.0 + np.exp(-logits))
        P      = FIXED_C + (1.0 - FIXED_C) * p_star       # (N,)
        Q      = 1.0 - P
        delta1 = LR * Q[:, None] * A                       # (N, 3)
        delta0 = -LR * P[:, None] * A
        kl_1   = np.sum(delta1 ** 2, axis=1) / (2 * SIGMA2)
        kl_0   = np.sum(delta0 ** 2, axis=1) / (2 * SIGMA2)
        scores = P * kl_1 + Q * kl_0
        return pool.iloc[int(np.argmax(scores))]

    def reset_exposure(self): pass

