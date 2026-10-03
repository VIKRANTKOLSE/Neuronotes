"""Adaptive, log-calibrated guessing parameters.

``c_j = 0.25 * sigmoid(beta0 + beta_sem*E_semantic +
                         beta_emp*E_empirical - beta_amb*E_ambiguity)``

The semantic component is available at import time from the z-ontology.
Empirical entrapment and wrong-option ambiguity are estimated periodically
from accumulated response logs.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_COEFFICIENTS = {
    "intercept": 2.0,
    "semantic": -1.6,
    "empirical": -1.2,
    "ambiguity": 0.8,
    "semantic_x_difficulty": -0.8,
}


def _difficulty_gate(d_param: float) -> float:
    """Bounded difficulty factor in [0, 1] for the E_sem x d interaction."""
    return 0.5 + 0.5 * float(np.tanh(float(d_param)))


class DynamicC:
    """Compute and periodically recalibrate item-specific guessing floors."""

    def __init__(self,
                 items_clean_path: Optional[Path] = None,
                 coefficients: Optional[dict] = None,
                 recalibration_interval: int = 100):
        self.coefficients = {**DEFAULT_COEFFICIENTS, **(coefficients or {})}
        self.recalibration_interval = max(1, int(recalibration_interval))
        self._c_map: dict[str, float] = {}
        self._item_features: dict[str, dict[str, float]] = {}
        self._attempt_log: list[dict] = []
        self._attempt_stats: dict[str, dict] = {}
        self._dirty_items: set[str] = set()
        path = items_clean_path or DATA_DIR / "items_clean.csv"
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            item_id = str(row["item_id"])
            semantic = float(row.get("semantic_entrapment", row.get("entrapment_index", 0.0)))
            difficulty = float(row.get("d_param", 0.0))
            features = {
                "semantic": semantic,
                "empirical": 0.0,
                "ambiguity": 0.0,
                "difficulty": difficulty,
            }
            self._item_features[item_id] = features
            self._c_map[item_id] = self._compute(**features)

    def get(self, item_id: str, entrapment_index: Optional[float] = None) -> float:
        if item_id in self._c_map:
            return self._c_map[item_id]
        if entrapment_index is not None:
            return self._compute(float(entrapment_index), 0.0, 0.0, 0.0)
        return 0.25

    def _compute(self, semantic: float, empirical: float, ambiguity: float,
                 difficulty: float = 0.0) -> float:
        b = self.coefficients
        logit = (b["intercept"] + b["semantic"] * float(semantic)
                 + b["empirical"] * float(empirical)
                 + b["ambiguity"] * float(ambiguity)
                 + b.get("semantic_x_difficulty", 0.0)
                 * float(semantic) * _difficulty_gate(difficulty))
        return float(np.clip(0.25 / (1.0 + np.exp(-logit)), 0.01, 0.25))

    def record_response(self, item_id: str, selected_option: int, correct: bool) -> None:
        """Record one attempt in O(1) and periodically refresh changed items."""
        item_id = str(item_id)
        record = {"item_id": item_id, "option": int(selected_option), "correct": bool(correct)}
        self._attempt_log.append(record)
        stats = self._attempt_stats.setdefault(item_id, {"total": 0, "incorrect": 0, "options": {}})
        stats["total"] += 1
        if not correct:
            stats["incorrect"] += 1
            stats["options"][int(selected_option)] = stats["options"].get(int(selected_option), 0) + 1
        self._dirty_items.add(item_id)
        if len(self._attempt_log) % self.recalibration_interval == 0:
            self.recalibrate()

    def recalibrate(self, attempts: Optional[list[dict]] = None) -> None:
        """Update empirical and ambiguity features from response logs.

        Empirical entrapment is the observed incorrect-response rate.
        Ambiguity is normalized entropy over incorrect option choices.
        """
        if attempts is not None:
            stats_by_item: dict[str, dict] = {}
            for attempt in attempts:
                item_id = str(attempt["item_id"])
                stats = stats_by_item.setdefault(item_id, {"total": 0, "incorrect": 0, "options": {}})
                stats["total"] += 1
                if not bool(attempt["correct"]):
                    stats["incorrect"] += 1
                    option = int(attempt["option"])
                    stats["options"][option] = stats["options"].get(option, 0) + 1
            items_to_refresh = stats_by_item.items()
        else:
            items_to_refresh = ((item_id, self._attempt_stats[item_id]) for item_id in self._dirty_items)

        for item_id, stats in items_to_refresh:
            if item_id not in self._item_features:
                continue
            empirical = float(stats["incorrect"] / max(stats["total"], 1))
            if stats["incorrect"] > 1:
                probabilities = np.asarray(list(stats["options"].values()), dtype=float) / stats["incorrect"]
                ambiguity = float(-(probabilities * np.log(probabilities)).sum() / np.log(3))
            else:
                ambiguity = 0.0
            features = self._item_features[item_id]
            features.update({"empirical": empirical, "ambiguity": ambiguity})
            self._c_map[item_id] = self._compute(**features)
        if attempts is None:
            self._dirty_items.clear()

    def update_coefficients(self, coefficients: dict) -> None:
        """Apply newly fitted coefficients and refresh all loaded items."""
        self.coefficients.update({key: float(value) for key, value in coefficients.items()})
        for item_id, features in self._item_features.items():
            self._c_map[item_id] = self._compute(**features)

    @staticmethod
    def fixed(value: float = 0.25) -> "DynamicC":
        """Return a minimal module that always serves the requested constant."""
        obj = DynamicC.__new__(DynamicC)
        obj.coefficients = DEFAULT_COEFFICIENTS.copy()
        obj._c_map = {}
        obj._item_features = {}
        obj._attempt_log = []
        obj._attempt_stats = {}
        obj._dirty_items = set()
        obj._fixed = float(value)
        obj.get = lambda item_id, entrapment_index=None: obj._fixed  # type: ignore
        return obj
