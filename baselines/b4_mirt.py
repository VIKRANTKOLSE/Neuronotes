"""B4 — Standard MIRT baseline (Maximum Fisher Information, no prereq constraints).

Selects items by maximising the trace of the 3D MIRT Fisher information matrix.
No prerequisite constraints, no misconception modelling, fixed c = 0.25.
Fully vectorised with numpy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
FIXED_C  = 0.25


class MIRTSelector:
    """Standard MIRT: max trace of Fisher info, no constraints. Vectorised."""

    name = "B4_MIRT"

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
        P      = FIXED_C + (1.0 - FIXED_C) * p_star
        Q      = 1.0 - P
        PQ     = np.maximum(P * Q, 1e-9)
        dP     = (P - FIXED_C) * PQ / max((1.0 - FIXED_C) ** 2, 1e-9)
        scale  = (dP ** 2) / PQ
        # trace(I) = scale * ||a||^2
        info   = scale * np.sum(A ** 2, axis=1)
        return pool.iloc[int(np.argmax(info))]

    def reset_exposure(self): pass
