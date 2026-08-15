"""B3 — Standard 3PL IRT baseline.

Selects items by maximising the scalar Fisher information under the standard
1-dimensional 3PL model. Uses only d_param and a scalar discrimination (a1).
Fully vectorised with numpy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
FIXED_C  = 0.25


class IRT3PLSelector:
    """Standard 3PL IRT item selector (1D). Vectorised."""

    name = "B3_3PL_IRT"

    def __init__(self, items_clean_path: Optional[Path] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        pool = self._items[~self._items["item_id"].isin(seen_items)]
        if pool.empty:
            return None
        theta_s = float(np.mean(theta))
        A  = pool["a1"].values
        D  = pool["d_param"].values
        logits = A * theta_s + D
        p_star = 1.0 / (1.0 + np.exp(-logits))
        P  = FIXED_C + (1.0 - FIXED_C) * p_star
        Q  = 1.0 - P
        PQ = np.maximum(P * Q, 1e-9)
        dP = (P - FIXED_C) * PQ / max((1.0 - FIXED_C) ** 2, 1e-9)
        info = (dP ** 2) / PQ
        return pool.iloc[int(np.argmax(info))]

    def reset_exposure(self): pass

