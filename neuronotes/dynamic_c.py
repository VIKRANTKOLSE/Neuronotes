"""
dynamic_c.py — Dynamic Guessing Parameter
==========================================
Module 3 of SMD-CC-MIRT-KL-CAT

Replaces the fixed c = 0.25 with an item-specific value:

    c_j = 0.25 * (1 - β * E_j)

where E_j is the Entrapment Index of item j (pre-computed and stored in
items_clean.csv) and β ∈ [0, 1] controls how much attractive distractors
suppress the guessing floor.

Intuition:
  - E_j near 0  → options are not enticing → guessing floor stays at 0.25
  - E_j near 1  → all distractors are highly trapping → c_j drops toward 0
                  (a random guesser is unlikely to pick correctly because
                   distractors are very attractive)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_BETA = 0.5      # recommended β per blueprint


class DynamicC:
    """Computes item-specific guessing parameters c_j.

    Parameters
    ----------
    items_clean_path : Path, optional
        Path to items_clean.csv. Defaults to data/items_clean.csv.
    beta : float
        Sensitivity parameter β ∈ [0, 1].
    """

    def __init__(self,
                 items_clean_path: Optional[Path] = None,
                 beta: float = DEFAULT_BETA):
        self.beta = beta
        self._c_map: dict[str, float] = {}
        p = items_clean_path or (DATA_DIR / "items_clean.csv")
        if p.exists():
            self._load(p)

    def _load(self, path: Path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            # items_clean already stores pre-computed c_j
            self._c_map[str(row["item_id"])] = float(row["c_j"])

    # ------------------------------------------------------------------
    def get(self, item_id: str, entrapment_index: Optional[float] = None) -> float:
        """Return c_j for an item.

        Falls back to formula if item_id not in table.

        Parameters
        ----------
        item_id : str
        entrapment_index : float, optional
            Used only if item_id is not found in precomputed table.
        """
        if item_id in self._c_map:
            return self._c_map[item_id]
        if entrapment_index is not None:
            return self._compute(entrapment_index)
        return 0.25  # default fixed guessing

    def _compute(self, e_j: float) -> float:
        """c_j = 0.25 * (1 - β * E_j), clipped to [0.01, 0.25]."""
        c = 0.25 * (1.0 - self.beta * float(e_j))
        return float(np.clip(c, 0.01, 0.25))

    # ------------------------------------------------------------------
    def update_beta(self, beta: float):
        """Update β and recompute all c_j values."""
        self.beta = float(np.clip(beta, 0.0, 1.0))

    @staticmethod
    def fixed(value: float = 0.25) -> "DynamicC":
        """Return a DynamicC instance that always returns a fixed c."""
        obj = DynamicC.__new__(DynamicC)
        obj.beta     = 0.0
        obj._c_map   = {}
        obj._fixed   = float(value)
        obj.get      = lambda item_id, entrapment_index=None: obj._fixed  # type: ignore
        return obj
