"""B3 — Standard 3PL IRT baseline.

Selects items by maximising the scalar Fisher information under the standard
1-dimensional 3PL model. Uses d_param and a scalar discrimination (MDISC from a_vector).
Fully vectorised with numpy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from neuronotes.cc_mirt import parse_a_vector

DATA_DIR = Path(__file__).parent.parent / "data"
FIXED_C  = 0.25


class IRT3PLSelector:
    """Standard 3PL IRT item selector (1D). Vectorised."""

    name = "B3_3PL_IRT"

    def __init__(self, items_clean_path: Optional[Path] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()
        if not self._items.empty:
            if "a_vector" in self._items.columns:
                self._A_scalar = np.array([
                    float(np.linalg.norm(parse_a_vector(v))) for v in self._items["a_vector"]
                ])
            elif "a1" in self._items.columns:
                self._A_scalar = self._items["a1"].values.astype(float)
            else:
                self._A_scalar = np.ones(len(self._items))
        else:
            self._A_scalar = np.array([])

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        if self._items.empty:
            return None
        mask = ~self._items["item_id"].isin(seen_items)
        pool = self._items[mask]
        if pool.empty:
            return None
        candidate_indices = np.flatnonzero(mask.values)
        theta_s = float(np.mean(theta))
        A = self._A_scalar[candidate_indices]
        D = pool["d_param"].values.astype(float)
        logits = A * theta_s + D
        p_star = 1.0 / (1.0 + np.exp(-logits))
        P = FIXED_C + (1.0 - FIXED_C) * p_star
        Q = 1.0 - P
        PQ = np.maximum(P * Q, 1e-9)
        dP_dtheta = (1.0 - FIXED_C) * p_star * (1.0 - p_star)
        info = (dP_dtheta ** 2) / PQ
        return pool.iloc[int(np.argmax(info))]

    def reset_exposure(self): pass
