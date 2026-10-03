"""Evaluation metrics for Neuronotes simulations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass
class MetricBundle:
    system_name: str
    n_learners: int
    mean_questions: float
    rmse: float
    mae: float
    ece: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auroc_micro: float
    valid_misconception_dims: int
    efficiency_q: float
    rmse_trajectory: list
    rmse_std: float = 0.0
    rmse_ci95_lo: float = 0.0
    rmse_ci95_hi: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    auprc: float = 0.0
    target_attainment_rate: float = 0.0
    per_dimension_rmse: Optional[np.ndarray] = None
    per_dimension_coverage: Optional[np.ndarray] = None


def compute_rmse_trajectory(theta_true_list: list[np.ndarray],
                             theta_est_trajectory: list[list[np.ndarray]]) -> list[float]:
    if not theta_est_trajectory:
        return []
    rmses = []
    for step in range(max(len(trajectory) for trajectory in theta_est_trajectory)):
        errors = [float(np.sqrt(np.mean((trajectory[step] - theta_true_list[i]) ** 2)))
                  for i, trajectory in enumerate(theta_est_trajectory) if step < len(trajectory)]
        rmses.append(float(np.mean(errors)) if errors else np.nan)
    return rmses


def mae_final(theta_true: np.ndarray, theta_est: np.ndarray) -> float:
    return float(mean_absolute_error(theta_true.reshape(-1), theta_est.reshape(-1)))


def expected_calibration_error(p_correct_list: list[float], correct_list: list[bool], n_bins: int = 10) -> float:
    probabilities = np.asarray(p_correct_list, dtype=float)
    labels = np.asarray(correct_list, dtype=float)
    if not len(probabilities):
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.any():
            ece += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return float(ece)


def select_f1_thresholds(y_true: np.ndarray, y_score: np.ndarray,
                          default: float = 0.5) -> np.ndarray:
    """Choose calibrated per-dimension thresholds from held-out labels.

    The thresholds use the empirical validation distribution rather than a
    fixed 0.5 cutoff, adapting to each misconception family's prevalence.
    """
    y_true, y_score = np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)
    if y_true.size == 0 or y_true.ndim < 2 or y_true.shape[1] == 0:
        return np.full(15, default, dtype=float)
    thresholds = np.full(y_true.shape[1], default, dtype=float)
    for dimension in range(y_true.shape[1]):
        labels, scores = y_true[:, dimension], y_score[:, dimension]
        if np.unique(labels).size < 2:
            continue
        from sklearn.metrics import roc_curve
        fpr, tpr, candidates = roc_curve(labels, scores)
        if len(candidates):
            # Youden's J statistic
            j_stat = tpr - fpr
            thresholds[dimension] = float(candidates[int(np.argmax(j_stat))])
    return thresholds


def multilabel_misconception_metrics(y_true: np.ndarray, y_score: np.ndarray,
                                     thresholds: Optional[np.ndarray] = None,
                                     active_dims: Optional[np.ndarray] = None) -> dict:
    """Compute macro/micro AUROC from continuous misconception scores.

    Parameters
    ----------
    active_dims : array of int, optional
        Indices of ontology dimensions that have real ground-truth labels.
        When provided, only these columns are evaluated — dormant dimensions
        (always 0 in ground truth) are excluded to prevent false positives
        from inflating precision/recall penalties.

    Dimensions with only one true class are excluded from AUROC and macro F1;
    this prevents rare ontology slots from silently forcing a fake 0.5 score.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("misconception labels and scores must have equal shape (learners, ontology_dims)")

    # Restrict to active dimensions if specified
    if active_dims is not None:
        active_dims = np.asarray(active_dims, dtype=int)
        labels = labels[:, active_dims]
        scores = scores[:, active_dims]
        if thresholds is not None:
            thresholds = np.asarray(thresholds, dtype=float)[active_dims]

    valid = np.array([np.unique(labels[:, dim]).size > 1 for dim in range(labels.shape[1])])
    if not valid.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": np.nan,
                "auroc_micro": np.nan, "valid_dims": 0}
    threshold_vector = np.full(labels.shape[1], 0.5) if thresholds is None else np.asarray(thresholds, dtype=float)
    predicted = (scores >= threshold_vector[None, :]).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels[:, valid], predicted[:, valid], average="macro", zero_division=0
    )
    per_dimension_auc = [roc_auc_score(labels[:, dim], scores[:, dim])
                         for dim in np.flatnonzero(valid)]
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(np.mean(per_dimension_auc)),
        "auroc_micro": float(roc_auc_score(labels[:, valid].ravel(), scores[:, valid].ravel())),
        "valid_dims": int(valid.sum()),
    }


def confidence_interval_95(data: list[float]) -> tuple[float, float]:
    if len(data) < 2:
        value = float(data[0]) if data else 0.0
        return value, value
    array = np.asarray(data, dtype=float)
    half_width = stats.sem(array) * stats.t.ppf(0.975, df=len(array) - 1)
    mean = float(array.mean())
    return mean - half_width, mean + half_width


def bootstrap_ci(data: list[float],
                 n_bootstrap: int = 1000,
                 confidence: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """Bootstrap confidence interval for mean.
    
    Valid for single-seed within-cohort uncertainty.
    """
    if len(data) < 2:
        value = float(data[0]) if data else 0.0
        return value, value
    
    rng = np.random.default_rng(seed)
    array = np.asarray(data, dtype=float)
    
    bootstrap_means = [
        np.mean(rng.choice(array, size=len(array), replace=True))
        for _ in range(n_bootstrap)
    ]
    
    lo = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    hi = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return float(lo), float(hi)


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    a, b = np.asarray(group_a), np.asarray(group_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return 0.0 if pooled_std < 1e-9 else float((np.mean(a) - np.mean(b)) / pooled_std)


@dataclass
class EfficiencyResult:
    """Efficiency metrics with proper right-censoring handling."""
    target_attainment_rate: float
    mean_questions_to_target: float
    censored_fraction: float
    median_questions: float


def questions_to_threshold(theta_trajectory: list[np.ndarray], theta_true: np.ndarray,
                           rmse_threshold: float = 0.35) -> tuple[int, bool]:
    """Count questions until RMSE drops below threshold.
    
    Returns (questions, attained) where attained=True if threshold reached.
    """
    for step, theta_estimate in enumerate(theta_trajectory):
        if float(np.sqrt(np.mean((theta_estimate - theta_true) ** 2))) <= rmse_threshold:
            return step, True
    return max(len(theta_trajectory) - 1, 0), False


def compute_efficiency(theta_trajectories: list[list[np.ndarray]],
                       theta_true_list: list[np.ndarray],
                       rmse_threshold: float = 0.35,
                       max_questions: int = 30) -> EfficiencyResult:
    """Compute efficiency with proper right-censoring.
    
    CRITICAL: Returns target-attainment rate, not fake efficiency.
    """
    attained = []
    questions_to_target = []
    
    for traj, true in zip(theta_trajectories, theta_true_list):
        q, reached = questions_to_threshold(traj, true, rmse_threshold)
        attained.append(reached)
        if reached:
            questions_to_target.append(q)
    
    n_attained = sum(attained)
    n_total = len(attained)
    censored = n_total - n_attained
    
    return EfficiencyResult(
        target_attainment_rate=n_attained / n_total if n_total > 0 else 0.0,
        mean_questions_to_target=np.mean(questions_to_target) if questions_to_target else np.nan,
        censored_fraction=censored / n_total if n_total > 0 else 0.0,
        median_questions=np.median(questions_to_target) if questions_to_target else np.nan,
    )


def compute_comprehensive_metrics(p_correct_all: list[float],
                                   correct_all: list[bool],
                                   misc_y_true: list[np.ndarray],
                                   misc_y_score: list[np.ndarray],
                                   active_dims: Optional[np.ndarray] = None) -> dict:
    """Compute Brier score, log loss, AUPRC, per-dimension coverage."""
    probs = np.asarray(p_correct_all, dtype=float)
    labels = np.asarray(correct_all, dtype=float)
    
    # Brier score: mean((prob - label)^2)
    brier = float(np.mean((probs - labels) ** 2))
    
    # Log loss
    eps = 1e-10
    probs_clipped = np.clip(probs, eps, 1 - eps)
    log_loss = float(-np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped)))
    
    # AUPRC (macro average across misconception dimensions)
    y_true = np.asarray(misc_y_true)
    y_score = np.asarray(misc_y_score)
    
    if active_dims is not None:
        y_true = y_true[:, active_dims]
        y_score = y_score[:, active_dims]
    
    valid_dims = [dim for dim in range(y_true.shape[1]) if len(np.unique(y_true[:, dim])) > 1]
    if valid_dims:
        auprc_values = [average_precision_score(y_true[:, dim], y_score[:, dim]) 
                       for dim in valid_dims]
        auprc = float(np.mean(auprc_values))
    else:
        auprc = 0.0
    
    # Per-dimension RMSE and coverage
    per_dim_rmse = None
    per_dim_coverage = None
    
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "auprc": auprc,
        "per_dim_rmse": per_dim_rmse,
        "per_dim_coverage": per_dim_coverage,
    }


def build_metric_bundle(system_name: str,
                         theta_true_list: list[np.ndarray],
                         theta_final_list: list[np.ndarray],
                         theta_est_trajectories: list[list[np.ndarray]],
                         p_correct_all: list[float],
                         correct_all: list[bool],
                         misc_y_true: list[np.ndarray],
                         misc_y_score: list[np.ndarray],
                         misc_thresholds: Optional[np.ndarray] = None,
                         rmse_threshold: float = 0.35,
                         active_dims: Optional[np.ndarray] = None,
                         ci_method: str = "bootstrap") -> MetricBundle:
    theta_true, theta_final = np.asarray(theta_true_list), np.asarray(theta_final_list)
    per_learner_rmse = [float(np.sqrt(np.mean((theta_final[i] - theta_true[i]) ** 2)))
                        for i in range(len(theta_true))]
    
    if ci_method == "bootstrap":
        ci_lo, ci_hi = bootstrap_ci(per_learner_rmse)
    else:
        ci_lo, ci_hi = confidence_interval_95(per_learner_rmse)
    
    diagnostic = multilabel_misconception_metrics(
        np.asarray(misc_y_true), np.asarray(misc_y_score), misc_thresholds,
        active_dims=active_dims,
    )
    
    # Compute comprehensive metrics
    comp_metrics = compute_comprehensive_metrics(p_correct_all, correct_all, misc_y_true, misc_y_score, active_dims)
    
    # Compute per-dimension RMSE for observed dimensions
    n_dims = theta_true.shape[1] if theta_true.ndim > 1 else 1
    if theta_true.ndim > 1:
        dim_errors = np.sqrt(np.mean((theta_final - theta_true) ** 2, axis=0))
        per_dim_rmse = dim_errors
        # Coverage: proportion of learners where dimension was tested
        per_dim_coverage = np.ones(n_dims)
    else:
        per_dim_rmse = np.array([float(np.sqrt(np.mean((theta_final - theta_true) ** 2)))])
        per_dim_coverage = np.array([1.0])
    
    # Compute efficiency with proper censoring
    efficiency_result = compute_efficiency(
        theta_est_trajectories, theta_true_list, rmse_threshold
    )
    
    return MetricBundle(
        system_name=system_name,
        n_learners=len(theta_true_list),
        mean_questions=float(np.mean([max(len(trajectory) - 1, 0) for trajectory in theta_est_trajectories])),
        rmse=float(np.mean(per_learner_rmse)),
        rmse_std=float(np.std(per_learner_rmse, ddof=1)),
        rmse_ci95_lo=ci_lo,
        rmse_ci95_hi=ci_hi,
        mae=mae_final(theta_true, theta_final),
        ece=expected_calibration_error(p_correct_all, correct_all),
        precision=diagnostic["precision"], recall=diagnostic["recall"], f1=diagnostic["f1"],
        auroc=diagnostic["auroc"], auroc_micro=diagnostic["auroc_micro"],
        valid_misconception_dims=diagnostic["valid_dims"],
        efficiency_q=efficiency_result.median_questions,
        rmse_trajectory=compute_rmse_trajectory(theta_true_list, theta_est_trajectories),
        brier_score=comp_metrics["brier_score"],
        log_loss=comp_metrics["log_loss"],
        auprc=comp_metrics["auprc"],
        target_attainment_rate=efficiency_result.target_attainment_rate,
        per_dimension_rmse=per_dim_rmse,
        per_dimension_coverage=per_dim_coverage,
    )
