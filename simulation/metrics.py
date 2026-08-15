"""
metrics.py — All Evaluation Metrics for Neuronotes Experiments
==============================================================
Computes:
  - RMSE, MAE, ECE  (ability estimation)
  - Precision, Recall, F1, AUROC  (misconception detection)
  - Efficiency (test length to confidence threshold)
  - 95% CI and effect size (Cohen's d) for all comparisons
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
@dataclass
class MetricBundle:
    """Container for all metrics from one experiment run."""
    system_name:        str
    n_learners:         int
    mean_questions:     float    # mean test length
    rmse:               float
    mae:                float
    ece:                float
    precision:          float
    recall:             float
    f1:                 float
    auroc:              float
    efficiency_q:       float    # mean questions to reach confidence threshold
    rmse_trajectory:    list     # RMSE per question step (averaged over learners)

    # Stats
    rmse_std:           float = 0.0
    rmse_ci95_lo:       float = 0.0
    rmse_ci95_hi:       float = 0.0


# ---------------------------------------------------------------------------
# Ability estimation metrics
# ---------------------------------------------------------------------------

def compute_rmse_trajectory(theta_true_list: list[np.ndarray],
                             theta_est_trajectory: list[list[np.ndarray]]) -> list[float]:
    """Compute mean RMSE at each question step across all learners.

    Parameters
    ----------
    theta_true_list : list of (3,) arrays, length N_learners
    theta_est_trajectory : list of lists; theta_est_trajectory[i][t] = θ̂ at step t for learner i
    """
    if not theta_est_trajectory:
        return []
    max_steps = max(len(traj) for traj in theta_est_trajectory)
    rmses = []
    for t in range(max_steps):
        sq_errors = []
        for i, traj in enumerate(theta_est_trajectory):
            if t < len(traj):
                err = float(np.sqrt(np.mean((traj[t] - theta_true_list[i]) ** 2)))
                sq_errors.append(err)
        rmses.append(float(np.mean(sq_errors)) if sq_errors else np.nan)
    return rmses


def rmse_final(theta_true: np.ndarray, theta_est: np.ndarray) -> float:
    """RMSE across all learners × dimensions at end of test."""
    return float(np.sqrt(np.mean((theta_true - theta_est) ** 2)))


def mae_final(theta_true: np.ndarray, theta_est: np.ndarray) -> float:
    return float(np.mean(np.abs(theta_true - theta_est)))


def expected_calibration_error(p_correct_list: list[float],
                               correct_list: list[bool],
                               n_bins: int = 10) -> float:
    """ECE: measure of calibration between predicted P and empirical accuracy."""
    p   = np.array(p_correct_list)
    y   = np.array(correct_list, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n_total = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        conf    = float(np.mean(p[mask]))
        acc     = float(np.mean(y[mask]))
        ece_val += (mask.sum() / n_total) * abs(conf - acc)
    return float(ece_val)


# ---------------------------------------------------------------------------
# Misconception detection metrics
# ---------------------------------------------------------------------------

def misconception_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_score: Optional[np.ndarray] = None) -> dict:
    """Compute precision, recall, F1, and AUROC for misconception detection.

    Parameters
    ----------
    y_true  : binary ground-truth labels (1 = misconception present)
    y_pred  : binary predicted labels
    y_score : continuous scores for AUROC (optional)
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auroc = 0.5
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = 0.5
    return {
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1),
        "auroc":     auroc,
    }


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def confidence_interval_95(data: list[float]) -> tuple[float, float]:
    """Return 95% CI via t-distribution."""
    n = len(data)
    if n < 2:
        return (float(data[0]) if data else 0.0, float(data[0]) if data else 0.0)
    arr = np.array(data)
    se  = stats.sem(arr)
    h   = se * stats.t.ppf(0.975, df=n - 1)
    mu  = float(np.mean(arr))
    return (mu - h, mu + h)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Effect size: Cohen's d between two groups."""
    a, b = np.array(group_a), np.array(group_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled_std < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Efficiency metric
# ---------------------------------------------------------------------------

def questions_to_threshold(theta_trajectory: list[np.ndarray],
                            theta_true: np.ndarray,
                            rmse_threshold: float = 0.3) -> int:
    """Return the first step at which RMSE drops below the threshold.
    Returns max_steps if threshold is never reached.
    """
    for t, theta_est in enumerate(theta_trajectory):
        err = float(np.sqrt(np.mean((theta_est - theta_true) ** 2)))
        if err <= rmse_threshold:
            return t + 1
    return len(theta_trajectory)


# ---------------------------------------------------------------------------
# Bundle builder
# ---------------------------------------------------------------------------

def build_metric_bundle(system_name: str,
                         theta_true_list:        list,
                         theta_final_list:        list,
                         theta_est_trajectories:  list,
                         p_correct_all:           list,
                         correct_all:             list,
                         misc_y_true:             list,
                         misc_y_pred:             list,
                         misc_y_score:            Optional[list] = None) -> MetricBundle:
    """Build a complete MetricBundle from experiment results."""
    N = len(theta_true_list)
    theta_true_arr  = np.array(theta_true_list)
    theta_final_arr = np.array(theta_final_list)

    # Per-learner RMSE
    per_rmse = [float(np.sqrt(np.mean((theta_final_arr[i] - theta_true_arr[i]) ** 2)))
                for i in range(N)]

    rmse_val    = float(np.mean(per_rmse))
    rmse_std    = float(np.std(per_rmse, ddof=1))
    ci_lo, ci_hi = confidence_interval_95(per_rmse)

    mae_val     = mae_final(theta_true_arr, theta_final_arr)
    ece_val     = expected_calibration_error(p_correct_all, correct_all)

    # Misconception detection
    misc_y_true_arr = np.array(misc_y_true, dtype=int)
    misc_y_pred_arr = np.array(misc_y_pred, dtype=int)
    misc_score_arr  = np.array(misc_y_score) if misc_y_score else None
    misc_m          = misconception_metrics(misc_y_true_arr, misc_y_pred_arr, misc_score_arr)

    # Trajectory
    traj = compute_rmse_trajectory(theta_true_list, theta_est_trajectories)

    # Efficiency
    eff_list = [
        questions_to_threshold(theta_est_trajectories[i], theta_true_list[i])
        for i in range(N)
    ]
    efficiency_q = float(np.mean(eff_list))
    mean_q       = float(np.mean([len(t) for t in theta_est_trajectories]))

    return MetricBundle(
        system_name=system_name,
        n_learners=N,
        mean_questions=mean_q,
        rmse=rmse_val,
        rmse_std=rmse_std,
        rmse_ci95_lo=ci_lo,
        rmse_ci95_hi=ci_hi,
        mae=mae_val,
        ece=ece_val,
        precision=misc_m["precision"],
        recall=misc_m["recall"],
        f1=misc_m["f1"],
        auroc=misc_m["auroc"],
        efficiency_q=efficiency_q,
        rmse_trajectory=traj,
    )
