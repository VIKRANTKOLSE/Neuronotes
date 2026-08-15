"""B1 — Random Question Selection baseline."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"


class RandomSelector:
    """Selects the next item uniformly at random from unseen items."""

    name = "B1_Random"

    def __init__(self, items_clean_path: Optional[Path] = None, rng: Optional[np.random.Generator] = None):
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        self._items = pd.read_csv(p) if p.exists() else pd.DataFrame()
        self.rng = rng or np.random.default_rng(42)

    def select(self, theta, seen_items, **kwargs) -> Optional[pd.Series]:
        pool = self._items[~self._items["item_id"].isin(seen_items)]
        if pool.empty:
            return None
        idx = self.rng.integers(0, len(pool))
        return pool.iloc[idx]

    def reset_exposure(self): pass
