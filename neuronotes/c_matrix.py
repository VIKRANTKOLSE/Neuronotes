"""
c_matrix.py — Option-Level Confusion Matrix (C-Matrix)
=======================================================
Module 2 of SMD-CC-MIRT-KL-CAT

Maps each (item_id, option_no) pair → misconception tag, error class, severity.
Loaded from item_options.csv produced by data_processing.py.

Usage
-----
    cm = CMatrix()
    result = cm.diagnose("NCERT_EffNucCharge_01", selected_option=1)
    # {'misconception_tag': ..., 'error_class': ..., 'severity': ..., 'trap_weight': ...}
"""

import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"

# Fallback option data when CSV is absent
_FALLBACK_RECORD = {
    "misconception_tag": "unknown_error",
    "error_class":       "conceptual_error",
    "severity":          "medium",
    "trap_weight":       0.0,
    "semantic_dimension": "recall",
    "is_correct":        False,
    "rationale":         "",
}


class CMatrix:
    """Option-Level C-Matrix: maps (item, option) → misconception profile.

    Parameters
    ----------
    item_options_path : Path, optional
        Path to item_options.csv. Defaults to data/item_options.csv.
    """

    def __init__(self, item_options_path: Optional[Path] = None):
        p = item_options_path or (DATA_DIR / "item_options.csv")
        self._matrix: dict[tuple[str, int], dict] = {}
        if p.exists():
            self._load(p)

    # ------------------------------------------------------------------
    def _load(self, path: Path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (str(row["item_id"]), int(row["option_no"]))
            self._matrix[key] = {
                "misconception_tag": str(row["misconception_tag"]),
                "error_class":       str(row["error_class"]),
                "severity":          str(row["severity"]),
                "trap_weight":       float(row["trap_weight"]),
                "semantic_dimension": str(row["semantic_dimension"]),
                "is_correct":        bool(row["is_correct"]),
                "rationale":         str(row["rationale"]),
            }

    # ------------------------------------------------------------------
    def diagnose(self, item_id: str, selected_option: int) -> dict:
        """Return misconception profile for a learner's selected option.

        Parameters
        ----------
        item_id : str
        selected_option : int  (1-indexed)

        Returns
        -------
        dict with keys: misconception_tag, error_class, severity,
                        trap_weight, semantic_dimension, is_correct, rationale
        """
        key = (item_id, selected_option)
        return self._matrix.get(key, _FALLBACK_RECORD.copy())

    # ------------------------------------------------------------------
    def is_correct(self, item_id: str, selected_option: int) -> bool:
        return self.diagnose(item_id, selected_option)["is_correct"]

    def error_class(self, item_id: str, selected_option: int) -> str:
        return self.diagnose(item_id, selected_option)["error_class"]

    def misconception_tag(self, item_id: str, selected_option: int) -> str:
        return self.diagnose(item_id, selected_option)["misconception_tag"]

    def trap_weight(self, item_id: str, selected_option: int) -> float:
        return self.diagnose(item_id, selected_option)["trap_weight"]

    # ------------------------------------------------------------------
    def all_wrong_options(self, item_id: str, correct_option: int) -> list[dict]:
        """Return misconception profiles for all 3 distractors."""
        return [
            {"option_no": opt, **self.diagnose(item_id, opt)}
            for opt in range(1, 5)
            if opt != correct_option
        ]

    def misconception_diagnostic_value(self, item_id: str, correct_option: int) -> float:
        """Aggregate diagnostic value: mean trap_weight across distractors."""
        distractors = self.all_wrong_options(item_id, correct_option)
        if not distractors:
            return 0.0
        return sum(d["trap_weight"] for d in distractors) / len(distractors)
