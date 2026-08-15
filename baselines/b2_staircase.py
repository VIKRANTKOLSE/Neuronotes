"""B2 — Difficulty-Staircase Selection baseline.

Selects items with difficulty closest to the learner's current estimated ability
(scalar: mean of theta dims). Implements a staircase: correct → harder, wrong → easier.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"


class StaircaseSelector:
    """Difficulty-staircase adaptive selector."""

    name = "B2_Staircase"

    def __init__(self, items_clean_path: Optional[Path] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        pool = self._items[~self._items["item_id"].isin(seen_items)]
        if pool.empty:
            return None
        ability = float(np.mean(theta))
        diffs   = np.abs(pool["d_param"].values - ability)
        best    = int(np.argmin(diffs))
        return pool.iloc[best]

    def reset_exposure(self): pass
