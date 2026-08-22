"""Evaluation metrics for Neuronotes simulations."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
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
    auroc: float                 # macro AUROC across valid ontology dimensions
    auroc_micro: float
    valid_misconception_dims: int
    efficiency_q: float
    rmse_trajectory: list
    rmse_std: float = 0.0
    rmse_ci95_lo: float = 0.0
    rmse_ci95_hi: float = 0.0


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
    """Choose one F1-maximising threshold per ontology dimension on validation data."""
    y_true, y_score = np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)
    thresholds = np.full(y_true.shape[1], default, dtype=float)
    for dimension in range(y_true.shape[1]):
        labels, scores = y_true[:, dimension], y_score[:, dimension]
        if np.unique(labels).size < 2:
            continue
        precision, recall, candidates = precision_recall_curve(labels, scores)
        if len(candidates):
            f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
            thresholds[dimension] = float(candidates[int(np.argmax(f1))])
    return thresholds


def multilabel_misconception_metrics(y_true: np.ndarray, y_score: np.ndarray,
                                     thresholds: Optional[np.ndarray] = None) -> dict:
    """Compute macro/micro AUROC from continuous 15-dimensional scores.

    Dimensions with only one true class are excluded from AUROC and macro F1;
    this prevents rare ontology slots from silently forcing a fake 0.5 score.
    """
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("misconception labels and scores must have equal shape (learners, ontology_dims)")
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


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    a, b = np.asarray(group_a), np.asarray(group_b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return 0.0 if pooled_std < 1e-9 else float((np.mean(a) - np.mean(b)) / pooled_std)


def questions_to_threshold(theta_trajectory: list[np.ndarray], theta_true: np.ndarray,
                           rmse_threshold: float = 0.35) -> int:
    """Count questions until RMSE first drops below rmse_threshold."""
    for step, theta_estimate in enumerate(theta_trajectory):
        if float(np.sqrt(np.mean((theta_estimate - theta_true) ** 2))) <= rmse_threshold:
            return step  # index 0 is the pre-question initial estimate
    return max(len(theta_trajectory) - 1, 0)


def build_metric_bundle(system_name: str,
                         theta_true_list: list[np.ndarray],
                         theta_final_list: list[np.ndarray],
                         theta_est_trajectories: list[list[np.ndarray]],
                         p_correct_all: list[float],
                         correct_all: list[bool],
                         misc_y_true: list[np.ndarray],
                         misc_y_score: list[np.ndarray],
                         misc_thresholds: Optional[np.ndarray] = None,
                         rmse_threshold: float = 0.35) -> MetricBundle:
    theta_true, theta_final = np.asarray(theta_true_list), np.asarray(theta_final_list)
    per_learner_rmse = [float(np.sqrt(np.mean((theta_final[i] - theta_true[i]) ** 2)))
                        for i in range(len(theta_true))]
    ci_lo, ci_hi = confidence_interval_95(per_learner_rmse)
    diagnostic = multilabel_misconception_metrics(
        np.asarray(misc_y_true), np.asarray(misc_y_score), misc_thresholds
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
        efficiency_q=float(np.mean([questions_to_threshold(theta_est_trajectories[i], theta_true[i],
                                                           rmse_threshold)
                                    for i in range(len(theta_true))])),
        rmse_trajectory=compute_rmse_trajectory(theta_true_list, theta_est_trajectories),
    )
