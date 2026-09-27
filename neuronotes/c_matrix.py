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

import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"

# Domain-specific Chemistry Misconception Ontology Mapping (z_00 to z_14)
MISCONCEPTION_ONTOLOGY = {
    "z_00": "Shielding_vs_Nuclear_Charge_Confusion",
    "z_01": "Penetration_Distance_Quantum_Inversion",
    "z_02": "Electron_Repulsion_Pairing_Energy_Omission",
    "z_03": "Orbital_Energy_Level_Splitting_Reversal",
    "z_04": "Exchange_Energy_Subshell_Stability_Neglect",
    "z_05": "Periodic_Radius_Lattice_Trend_Inversion",
    "z_06": "Ionization_Gain_Enthalpy_Anomaly_Confusion",
    "z_07": "Electronegativity_Polarization_Fajan_Confusion",
    "z_08": "Hybridization_VSEPR_Geometry_Mismatch",
    "z_09": "Bent_Rule_Hypervalent_Bonding_Misconception",
    "z_10": "Pi_Back_Bonding_Electron_Density_Reversal",
    "z_11": "Crystal_Field_Splitting_Oct_Tet_Confusion",
    "z_12": "High_Low_Spin_Pairing_Condition_Inversion",
    "z_13": "Spectrochemical_Series_Ligand_Strength_Error",
    "z_14": "Chelate_Entropy_Thermodynamic_Misattribution",
}

# Fallback option data when CSV is absent
_FALLBACK_RECORD = {
    "misconception_tag": "unknown_error",
    "misconception_tags": ["unknown_error"],
    "z_vector": np.zeros(15, dtype=float),
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
            z_vector = self._parse_z_vector(row.get("z_vector", "[]"))
            tag = str(row["misconception_tag"])
            is_correct = bool(row["is_correct"])
            # Correct options may carry a source z annotation, but they are
            # not misconception evidence and must not update error state.
            tags = [] if is_correct else [f"z_{index:02d}" for index, value in enumerate(z_vector) if value > 0]
            if not tags and tag not in {"none", "nan"}:
                tags = [tag]
            self._matrix[key] = {
                "misconception_tag": tag,
                "misconception_tags": tags,
                "z_vector": z_vector,
                "error_class":       str(row["error_class"]),
                "severity":          str(row["severity"]),
                "trap_weight":       float(row["trap_weight"]),
                "semantic_dimension": str(row["semantic_dimension"]),
                "is_correct":        is_correct,
                "rationale":         str(row["rationale"]),
            }

    @staticmethod
    def _parse_z_vector(value) -> np.ndarray:
        """Read a persisted 15-dimensional ontology vector safely."""
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(str(value))
            except (ValueError, TypeError):
                parsed = []
        vector = np.asarray(parsed, dtype=float).reshape(-1)
        return np.pad(vector[:15], (0, max(0, 15 - len(vector))))

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

    def z_vector(self, item_id: str, selected_option: int) -> np.ndarray:
        """Return the option's shared 15-dimensional misconception vector."""
        return self.diagnose(item_id, selected_option)["z_vector"].copy()

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

    def item_distractor_mask(self, item_id: str, correct_option: Optional[int] = None) -> np.ndarray:
        """Return a binary 15D mask indicating which misconceptions are present in the item's distractors."""
        mask = np.zeros(15, dtype=float)
        for opt in range(1, 5):
            rec = self._matrix.get((str(item_id), opt))
            if rec is not None:
                if correct_option is not None:
                    if opt == int(correct_option) or rec.get("is_correct", False):
                        continue
                elif rec.get("is_correct", False):
                    continue
                z = rec.get("z_vector")
                if z is not None and len(z) > 0:
                    vec = np.asarray(z, dtype=float)[:15]
                    mask = np.maximum(mask, (vec > 0).astype(float))
        return mask
