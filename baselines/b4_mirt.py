"""B4 — Standard MIRT baseline (Maximum Fisher Information, no prereq constraints).

Selects items by maximising the trace of the MIRT Fisher information matrix.
No prerequisite constraints, no misconception modelling, fixed c = 0.25.
Fully vectorised with numpy. Supports 58D.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from neuronotes.cc_mirt import parse_a_vector, N_DIMS

DATA_DIR = Path(__file__).parent.parent / "data"
FIXED_C  = 0.25


class MIRTSelector:
    """Standard MIRT: max trace of Fisher info, no constraints. Vectorised."""

    name = "B4_MIRT"

    def __init__(self, items_clean_path: Optional[Path] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if not self._items.empty:
            if "a_vector" in self._items.columns:
                self._A = np.vstack([parse_a_vector(v) for v in self._items["a_vector"]])
            elif all(c in self._items.columns for c in ["a1", "a2", "a3"]):
                self._A = self._items[["a1", "a2", "a3"]].values
            else:
                self._A = np.zeros((len(self._items), N_DIMS))
        else:
            self._A = np.zeros((0, N_DIMS))

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        if self._items.empty:
            return None
        mask = ~self._items["item_id"].isin(seen_items)
        pool = self._items[mask]
        if pool.empty:
            return None
        candidate_indices = np.flatnonzero(mask.values)
        A      = self._A[candidate_indices]                # (N, N_DIMS)
        D      = pool["d_param"].values.astype(float)      # (N,)
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
