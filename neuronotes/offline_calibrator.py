"""
offline_calibrator.py — Offline E-Weight Recalibration Engine
=============================================================
After a batch of N learners (default: 1000), recalibrates entrapment
weights (E_semantic, E_empirical, E_ambiguity) and DynamicC coefficients
using accumulated response evidence.

Strategy
--------
Phase 1 (Online): Every learner session logs (item_id, option, correct, theta_norm).
Phase 2 (Offline, every N learners):
  1. Compute per-item empirical error rate (E_emp_new)
  2. Compute distractor entropy (E_amb_new) — are students spread across wrong
     options (high ambiguity) or funnelled into one trap (low ambiguity)?
  3. Compute per-z-dimension trap contribution — which misconceptions actually
     trap students vs. which are theoretical?
  4. Blend E_semantic with empirical data using Bayesian shrinkage:
       E_sem_updated = α * E_sem_prior + (1 - α) * E_emp_new
     where α decays with calibration rounds (more data → more trust in empirical)
  5. Refit DynamicC logistic coefficients via maximum likelihood
  6. Persist updated item bank to disk

Usage
-----
    calibrator = OfflineCalibrator(items_clean_path, threshold=1000)
    # During each learner session:
    calibrator.log_response(item_id, selected_option, correct, theta_norm)
    # After session ends:
    if calibrator.should_recalibrate():
        report = calibrator.recalibrate()
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_COEFFICIENTS = {
    "intercept": 2.0,
    "semantic": -1.6,
    "empirical": -1.2,
    "ambiguity": 0.8,
    "semantic_x_difficulty": -0.8,
}


def _difficulty_gate(d_param: float) -> float:
    """Bounded difficulty factor in [0, 1] used for the E_sem x d interaction."""
    return 0.5 + 0.5 * math.tanh(float(d_param))


@dataclass
class CalibrationReport:
    """Summary of one offline recalibration pass."""
    calibration_round: int
    total_responses: int
    items_updated: int
    items_flagged: int               # items where E_sem badly mismatches reality
    alpha_used: float                # shrinkage weight for this round
    mean_abs_delta_e_sem: float      # average |E_sem_new - E_sem_old|
    coefficient_update: dict         # new DynamicC logistic coefficients
    z_dim_trap_rates: list[float]    # per z-dimension actual trap rate
    flagged_items: list[str] = field(default_factory=list)


class OfflineCalibrator:
    """Accumulates response logs and performs periodic offline E recalibration.

    Parameters
    ----------
    items_clean_path : Path to items_clean.csv
    threshold : Number of learners (not responses) before triggering recalibration
    min_attempts_per_item : Minimum attempts for an item to be recalibrated
                            (prevents noisy updates from low-N items)
    """

    def __init__(self,
                 items_clean_path: Optional[Path] = None,
                 threshold: int = 1000,
                 min_attempts_per_item: int = 10):
        self.threshold = max(1, threshold)
        self.min_attempts = max(1, min_attempts_per_item)
        self._items_path = items_clean_path or DATA_DIR / "items_clean.csv"

        # Accumulated response log: list of dicts
        self._response_log: list[dict] = []
        self._learner_count = 0
        self._calibration_round = 0

        # Load current item bank
        self._items_df: Optional[pd.DataFrame] = None
        if self._items_path.exists():
            self._items_df = pd.read_csv(self._items_path)

    # ------------------------------------------------------------------
    # Online phase: log responses
    # ------------------------------------------------------------------
    def log_response(self, item_id: str, selected_option: int,
                     correct: bool, theta_norm: float = 0.0,
                     z_vector: Optional[np.ndarray] = None) -> None:
        """Record one response. O(1) amortised cost."""
        self._response_log.append({
            "item_id": str(item_id),
            "option": int(selected_option),
            "correct": bool(correct),
            "theta_norm": float(theta_norm),
            "z_vector": z_vector.tolist() if z_vector is not None else None,
        })

    def end_learner_session(self) -> None:
        """Call after each learner finishes their 30-question session."""
        self._learner_count += 1

    def should_recalibrate(self) -> bool:
        """Check if the learner threshold has been crossed."""
        return self._learner_count >= self.threshold and len(self._response_log) > 0

    @property
    def learner_count(self) -> int:
        return self._learner_count

    # ------------------------------------------------------------------
    # Offline phase: recalibrate
    # ------------------------------------------------------------------
    def recalibrate(self) -> CalibrationReport:
        """Run the full offline recalibration pipeline.

        Returns a CalibrationReport with diagnostics.
        """
        self._calibration_round += 1

        if self._items_df is None:
            if self._items_path.exists():
                self._items_df = pd.read_csv(self._items_path)
            else:
                raise FileNotFoundError(f"Item bank not found: {self._items_path}")

        df_log = pd.DataFrame(self._response_log)
        total_responses = len(df_log)

        # --- Step 1: Per-item empirical statistics ---
        item_stats = self._compute_item_stats(df_log)

        # --- Step 2: Per z-dimension trap contribution ---
        z_dim_rates = self._compute_z_dim_trap_rates(df_log)

        # --- Step 3: Bayesian shrinkage blend ---
        # α decays with calibration rounds: starts at 0.8 (trust prior),
        # converges to 0.2 (trust empirical) after many rounds
        alpha = 0.2 + 0.6 * math.exp(-0.3 * (self._calibration_round - 1))
        alpha = max(0.2, min(0.9, alpha))

        items_updated = 0
        items_flagged = 0
        flagged_items = []
        delta_e_accum = []

        for _, row in self._items_df.iterrows():
            item_id = str(row["item_id"])
            if item_id not in item_stats:
                continue
            stats = item_stats[item_id]
            if stats["total"] < self.min_attempts:
                continue

            old_e_sem = float(row.get("semantic_entrapment", 0.0))
            e_emp_new = stats["error_rate"]
            e_amb_new = stats["ambiguity"]

            # Blend: updated semantic entrapment
            e_sem_new = alpha * old_e_sem + (1.0 - alpha) * e_emp_new

            # Update item bank
            idx = self._items_df.index[self._items_df["item_id"] == item_id]
            if len(idx) > 0:
                self._items_df.loc[idx, "semantic_entrapment"] = round(e_sem_new, 6)
                self._items_df.loc[idx, "empirical_entrapment"] = round(e_emp_new, 6)
                self._items_df.loc[idx, "ambiguity_index"] = round(e_amb_new, 6)
                items_updated += 1
                delta_e_accum.append(abs(e_sem_new - old_e_sem))

            # Flag items where prior E_sem badly mismatches reality
            # (> 0.3 absolute difference with enough data)
            if stats["total"] >= 30 and abs(old_e_sem - e_emp_new) > 0.3:
                items_flagged += 1
                flagged_items.append(item_id)

        # --- Step 4: Refit DynamicC logistic coefficients ---
        new_coefficients = self._refit_coefficients(df_log, item_stats)

        # --- Step 5: Recompute c_j for all items ---
        self._recompute_cj(new_coefficients)

        # --- Step 6: Persist ---
        self._items_df.to_csv(self._items_path, index=False)

        # Reset for next batch
        self._response_log.clear()
        self._learner_count = 0

        mean_delta = float(np.mean(delta_e_accum)) if delta_e_accum else 0.0

        return CalibrationReport(
            calibration_round=self._calibration_round,
            total_responses=total_responses,
            items_updated=items_updated,
            items_flagged=items_flagged,
            alpha_used=round(alpha, 4),
            mean_abs_delta_e_sem=round(mean_delta, 6),
            coefficient_update=new_coefficients,
            z_dim_trap_rates=[round(r, 4) for r in z_dim_rates],
            flagged_items=flagged_items,
        )

    # ------------------------------------------------------------------
    # Internal: compute per-item statistics
    # ------------------------------------------------------------------
    def _compute_item_stats(self, df: pd.DataFrame) -> dict[str, dict]:
        """Per-item error rate and option entropy."""
        stats: dict[str, dict] = {}
        for item_id, group in df.groupby("item_id"):
            total = len(group)
            incorrect = group[~group["correct"]]
            n_incorrect = len(incorrect)
            error_rate = n_incorrect / max(total, 1)

            # Ambiguity = normalised entropy of wrong-option distribution
            if n_incorrect > 1:
                option_counts = incorrect["option"].value_counts().values.astype(float)
                probs = option_counts / option_counts.sum()
                entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
                # Normalise by log(3) since there are at most 3 wrong options
                ambiguity = entropy / math.log(3)
            else:
                ambiguity = 0.0

            # Theta-conditional trap rate: how often do high-ability students
            # still get this wrong? (proxy for true entrapment vs. difficulty)
            high_theta_mask = group["theta_norm"] > 0.5
            if high_theta_mask.sum() >= 3:
                high_theta_error = float(
                    (~group.loc[high_theta_mask, "correct"]).mean()
                )
            else:
                high_theta_error = error_rate  # fallback

            stats[str(item_id)] = {
                "total": total,
                "n_incorrect": n_incorrect,
                "error_rate": error_rate,
                "ambiguity": ambiguity,
                "high_theta_error": high_theta_error,
            }
        return stats

    # ------------------------------------------------------------------
    # Internal: per z-dimension actual trap rates
    # ------------------------------------------------------------------
    def _compute_z_dim_trap_rates(self, df: pd.DataFrame) -> list[float]:
        """Per z-dimension actual trap rate.

        Rate = (times students fell for a trap with z_dim active) /
               (times students were exposed to items with z_dim on a distractor)

        Only counts incorrect responses where the z_vector is provided.
        Exposure denominator counts all responses (correct or not) where z_vector is present.
        """
        z_fell = np.zeros(15, dtype=float)     # numerator: incorrect + z active
        z_exposed = np.zeros(15, dtype=float)  # denominator: any response + z active

        for _, row in df.iterrows():
            z = row.get("z_vector")
            if z is None:
                continue
            if not isinstance(z, (list, np.ndarray)):
                continue
            z = np.asarray(z, dtype=float)
            if len(z) != 15:
                continue
            active = z > 0
            # Every response to an item with this z-dim present counts as exposure
            z_exposed[active] += 1
            # Only incorrect responses count as "fell for the trap"
            if not row["correct"]:
                z_fell[active] += 1

        rates = np.zeros(15)
        mask = z_exposed > 0
        rates[mask] = z_fell[mask] / z_exposed[mask]
        return rates.tolist()

    # ------------------------------------------------------------------
    # Internal: refit DynamicC logistic coefficients
    # ------------------------------------------------------------------
    def _refit_coefficients(self, df_log: pd.DataFrame,
                            item_stats: dict) -> dict:
        """Fit β₀ + β₁*E_sem + β₂*E_emp + β₃*E_amb → P(incorrect)
        using iteratively reweighted least squares (Newton-Raphson).

        Falls back to default coefficients if there's insufficient data.
        """
        if self._items_df is None or len(item_stats) < 20:
            return dict(DEFAULT_COEFFICIENTS)

        # Build design matrix from items that have enough responses
        X_rows = []
        y_rows = []
        for _, row in self._items_df.iterrows():
            item_id = str(row["item_id"])
            if item_id not in item_stats:
                continue
            stats = item_stats[item_id]
            if stats["total"] < self.min_attempts:
                continue

            e_sem = float(row.get("semantic_entrapment", 0.0))
            e_emp = stats["error_rate"]
            e_amb = stats["ambiguity"]
            d_param = float(row.get("d_param", 0.0))

            # Interaction column: E_sem x difficulty gate. Traps only catch
            # mid-level students (experts avoid, novices guess randomly), so
            # the plain E_emp regressor under-weights semantic entrapment on
            # hard items. Carrying E_sem through this channel keeps the
            # expert prior alive even when AdamW clamps the main coefficient.
            X_rows.append([1.0, e_sem, e_emp, e_amb,
                           e_sem * _difficulty_gate(d_param)])
            # Target: logit of observed c_j / 0.35
            # c_j = 0.35 * sigmoid(logit), so logit = log(c_j / (0.35 - c_j))
            # We use the empirical error rate as a proxy for the guessing floor
            c_target = max(0.02, min(0.34, e_emp * 0.35))
            logit_target = math.log(c_target / max(0.35 - c_target, 0.01))
            y_rows.append(logit_target)

        if len(X_rows) < 20:
            return dict(DEFAULT_COEFFICIENTS)

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Maximum A Posteriori (MAP) estimation using AdamW
        # Initialize with prior mean
        w = np.array([2.0, -1.6, -1.2, 0.8, -0.8])
        
        # AdamW hyperparameters
        lr = 0.05
        weight_decay = 0.01
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        epochs = 300
        
        m = np.zeros(5)
        v = np.zeros(5)
        
        for t in range(1, epochs + 1):
            y_pred = X @ w
            # Gradient of MSE loss
            grad = (2.0 / len(X)) * X.T @ (y_pred - y)
            
            # Decoupled weight decay
            w -= lr * weight_decay * w
            
            # Adam momentum updates
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Strict MAP Constraint: semantic coefficients must be <= 0
            w[1] = min(w[1], 0.0)
            w[4] = min(w[4], 0.0)
            
        beta = w

        return {
            "intercept": round(float(beta[0]), 4),
            "semantic":  round(float(beta[1]), 4),
            "empirical": round(float(beta[2]), 4),
            "ambiguity": round(float(beta[3]), 4),
            "semantic_x_difficulty": round(float(beta[4]), 4),
        }

    # ------------------------------------------------------------------
    # Internal: recompute c_j from updated features and coefficients
    # ------------------------------------------------------------------
    def _recompute_cj(self, coefficients: dict) -> None:
        """Update c_j column in the item bank using new coefficients."""
        if self._items_df is None:
            return
        b = coefficients
        for idx, row in self._items_df.iterrows():
            e_sem = float(row.get("semantic_entrapment", 0.0))
            e_emp = float(row.get("empirical_entrapment", 0.0))
            e_amb = float(row.get("ambiguity_index", 0.0))
            d_param = float(row.get("d_param", 0.0))
            logit = (b["intercept"] + b["semantic"] * e_sem
                     + b["empirical"] * e_emp + b["ambiguity"] * e_amb
                     + b.get("semantic_x_difficulty", 0.0) * e_sem * _difficulty_gate(d_param))
            c_j = float(np.clip(0.35 / (1.0 + np.exp(-logit)), 0.01, 0.35))
            self._items_df.at[idx, "c_j"] = round(c_j, 4)
