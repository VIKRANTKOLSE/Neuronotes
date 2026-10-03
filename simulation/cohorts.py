"""
cohorts.py — Three-Cohort Evaluation Protocol
=============================================
Implements strict separation between development, calibration, and test cohorts
to prevent data leakage and ensure valid statistical inference.

Cohort purposes:
  - Development: Hyperparameter tuning, F1 threshold selection, component design
  - Calibration: Fit shared item parameters (DynamicC, offline calibrator)
  - Test: Locked evaluation; freeze all shared state; no feedback to model

Critical principle:
  - Test cohort is NEVER used for any tuning or calibration
  - Shared state is frozen before test evaluation
  - Per-learner state is reset between test learners
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .ground_truth import GroundTruth, LearnerGroundTruth


@dataclass
class CohortConfig:
    """Configuration for three-cohort split."""
    development_ids: list[str]
    calibration_ids: list[str]
    test_ids: list[str]
    
    def get_cohort(self, cohort_name: str) -> list[str]:
        if cohort_name == "development":
            return self.development_ids
        elif cohort_name == "calibration":
            return self.calibration_ids
        elif cohort_name == "test":
            return self.test_ids
        else:
            raise ValueError(f"Unknown cohort: {cohort_name}")
    
    def __len__(self) -> int:
        return len(self.development_ids) + len(self.calibration_ids) + len(self.test_ids)


@dataclass
class CohortGroundTruth:
    """Ground truth filtered to a specific cohort."""
    cohort_name: str
    learners: list[LearnerGroundTruth]
    
    def __len__(self) -> int:
        return len(self.learners)


def split_cohorts(ground_truth: GroundTruth,
                  dev_frac: float = 0.20,
                  cal_frac: float = 0.20,
                  shuffle: bool = True,
                  seed: Optional[int] = None) -> CohortConfig:
    """
    Split ground-truth population into three disjoint cohorts.
    
    Parameters
    ----------
    ground_truth : GroundTruth
        Complete ground truth for all learners
    dev_frac : float
        Fraction of population for development (default: 0.20)
    cal_frac : float
        Fraction of population for calibration (default: 0.20)
    shuffle : bool
        Whether to shuffle before splitting (default: True)
    seed : int, optional
        Random seed for shuffling (default: use ground_truth.seed)
    
    Returns
    -------
    CohortConfig
        learner_ids split into three cohorts
    
    Notes
    -----
    Test cohort gets the remaining learners (1 - dev_frac - cal_frac).
    For default 200 learners: 40 dev, 40 cal, 120 test.
    """
    n_total = len(ground_truth.learners)
    n_dev = int(n_total * dev_frac)
    n_cal = int(n_total * cal_frac)
    n_test = n_total - n_dev - n_cal
    
    # Get all learner IDs
    all_ids = [l.learner_id for l in ground_truth.learners]
    
    # Shuffle if requested
    if shuffle:
        rng = np.random.default_rng(seed if seed is not None else ground_truth.seed)
        indices = rng.permutation(n_total)
        all_ids = [all_ids[i] for i in indices]
    
    # Split into cohorts
    dev_ids = all_ids[:n_dev]
    cal_ids = all_ids[n_dev:n_dev + n_cal]
    test_ids = all_ids[n_dev + n_cal:]
    
    return CohortConfig(
        development_ids=dev_ids,
        calibration_ids=cal_ids,
        test_ids=test_ids,
    )


def filter_ground_truth_to_cohort(ground_truth: GroundTruth,
                                   cohort_ids: list[str]) -> CohortGroundTruth:
    """
    Filter ground truth to only include learners in the specified cohort.
    
    Parameters
    ----------
    ground_truth : GroundTruth
        Complete ground truth for all learners
    cohort_ids : list[str]
        Learner IDs to include in the cohort
    
    Returns
    -------
    CohortGroundTruth
        Filtered ground truth for the cohort
    """
    cohort_learners = [l for l in ground_truth.learners if l.learner_id in cohort_ids]
    
    # Sort by learner_id to maintain order
    cohort_learners.sort(key=lambda l: l.learner_id)
    
    # Determine cohort name from size (heuristic)
    n_total = len(ground_truth.learners)
    n_cohort = len(cohort_learners)
    
    if n_cohort == 0:
        cohort_name = "empty"
    elif n_cohort < 0.25 * n_total:
        cohort_name = "development"
    elif n_cohort < 0.50 * n_total:
        cohort_name = "calibration"
    else:
        cohort_name = "test"
    
    return CohortGroundTruth(
        cohort_name=cohort_name,
        learners=cohort_learners,
    )


class CohortManager:
    """
    Manages three-cohort evaluation with proper isolation.
    
    Usage:
        manager = CohortManager(ground_truth)
        
        # Development phase
        dev_results = manager.run_cohort("development", system_factory)
        thresholds = calibrate_thresholds(dev_results)
        
        # Calibration phase
        cal_results = manager.run_cohort("calibration", system_factory)
        # Update shared parameters...
        
        # Test phase (LOCKED)
        manager.freeze_shared_state()
        test_results = manager.run_cohort("test", system_factory, locked=True)
    """
    
    def __init__(self, 
                 ground_truth: GroundTruth,
                 dev_frac: float = 0.20,
                 cal_frac: float = 0.20):
        self.ground_truth = ground_truth
        self.cohort_config = split_cohorts(ground_truth, dev_frac, cal_frac)
        
        self._frozen = False
        self._shared_state = {}
    
    def get_cohort_ground_truth(self, cohort_name: str) -> CohortGroundTruth:
        """Get ground truth for a specific cohort."""
        cohort_ids = self.cohort_config.get_cohort(cohort_name)
        return filter_ground_truth_to_cohort(self.ground_truth, cohort_ids)
    
    def freeze_shared_state(self):
        """
        Freeze all shared state before test evaluation.
        
        This prevents test learners from influencing:
          - DynamicC response history and c estimates
          - CCMIRT prerequisite graph weights
          - SMD-VSNLMS semantic matrix B
          - Item exposure counts
        """
        self._frozen = True
        # Actual state freezing happens in run_experiment.py
    
    def is_frozen(self) -> bool:
        return self._frozen
    
    def summary(self) -> dict:
        """Return summary of cohort sizes."""
        return {
            "total": len(self.cohort_config),
            "development": len(self.cohort_config.development_ids),
            "calibration": len(self.cohort_config.calibration_ids),
            "test": len(self.cohort_config.test_ids),
            "frozen": self._frozen,
        }
    
    def __repr__(self) -> str:
        s = self.summary()
        return (f"CohortManager(total={s['total']}, "
                f"dev={s['development']}, cal={s['calibration']}, test={s['test']}, "
                f"frozen={s['frozen']})")
