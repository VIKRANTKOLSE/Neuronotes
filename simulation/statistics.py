"""
statistics.py — Statistical Testing and Uncertainty Quantification
===================================================================
Provides paired bootstrap CIs, significance tests, and effect size CIs.
"""

import numpy as np
from typing import Optional
from scipy import stats


def paired_bootstrap_ci(metric_a: list[float],
                        metric_b: list[float],
                        n_bootstrap: int = 10000,
                        confidence: float = 0.95,
                        seed: int = 42) -> tuple[float, float, float]:
    """Compute paired bootstrap CI for difference A - B.
    
    Parameters
    ----------
    metric_a : list[float]
        Metric values for system A (same learners as B)
    metric_b : list[float]
        Metric values for system B (same learners as A)
    n_bootstrap : int
        Number of bootstrap resamples
    confidence : float
        Confidence level (0-1)
    seed : int
        Random seed
    
    Returns
    -------
    (ci_lo, ci_hi, diff_mean)
        Lower/upper CI bounds and mean difference
    """
    if len(metric_a) != len(metric_b):
        raise ValueError("Metrics must have same number of samples")
    
    rng = np.random.default_rng(seed)
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    diffs = a - b
    
    bootstrap_diffs = [
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_bootstrap)
    ]
    
    lo = np.percentile(bootstrap_diffs, (1 - confidence) / 2 * 100)
    hi = np.percentile(bootstrap_diffs, (1 + confidence) / 2 * 100)
    mean_diff = float(np.mean(diffs))
    
    return float(lo), float(hi), mean_diff


def cohens_d_ci(group_a: list[float],
                group_b: list[float],
                confidence: float = 0.95,
                n_bootstrap: int = 10000,
                seed: int = 42) -> tuple[float, float, float]:
    """Compute Cohen's d with bootstrap CI.
    
    Parameters
    ----------
    group_a, group_b : list[float]
        Two independent groups
    confidence : float
        Confidence level
    n_bootstrap : int
        Bootstrap samples
    seed : int
        Random seed
    
    Returns
    -------
    (ci_lo, ci_hi, d_mean)
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std
    
    bootstrap_ds = []
    for _ in range(n_bootstrap):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        
        pooled_boot = np.sqrt((np.var(a_boot, ddof=1) + np.var(b_boot, ddof=1)) / 2)
        if pooled_boot > 1e-9:
            bootstrap_ds.append((np.mean(a_boot) - np.mean(b_boot)) / pooled_boot)
    
    lo = np.percentile(bootstrap_ds, (1 - confidence) / 2 * 100)
    hi = np.percentile(bootstrap_ds, (1 + confidence) / 2 * 100)
    
    return float(lo), float(hi), float(d)


def permutation_test(metric_a: list[float],
                     metric_b: list[float],
                     n_permutations: int = 10000,
                     seed: int = 42,
                     alternative: str = "two-sided") -> tuple[float, float]:
    """Permutation test for difference in means.
    
    Parameters
    ----------
    metric_a, metric_b : list[float]
    n_permutations : int
    alternative : str
        "two-sided", "greater", "less"
    
    Returns
    -------
    (observed_diff, p_value)
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    
    observed_diff = np.mean(a) - np.mean(b)
    
    combined = np.concatenate([a, b])
    n_a = len(a)
    
    perm_diffs = []
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diffs.append(np.mean(perm_a) - np.mean(perm_b))
    
    perm_diffs = np.array(perm_diffs)
    
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == "greater":
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == "less":
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    
    return float(observed_diff), float(p_value)


def multiplicity_correction(p_values: list[float],
                            method: str = "bonferroni") -> list[float]:
    """Correct p-values for multiple comparisons.
    
    Parameters
    ----------
    p_values : list[float]
    method : str
        "bonferroni", "holm", "bh" (Benjamini-Hochberg)
    
    Returns
    -------
    corrected p-values
    """
    p = np.asarray(p_values)
    n = len(p)
    
    if method == "bonferroni":
        return list(np.minimum(p * n, 1.0))
    
    elif method == "holm":
        sorted_idx = np.argsort(p)
        corrected = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            corrected[idx] = min(1.0, p[idx] * (n - i))
        return list(corrected)
    
    elif method == "bh":
        sorted_idx = np.argsort(p)
        corrected = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            rank = n - i
            corrected[idx] = min(1.0, p[idx] * n / rank)
        
        # Ensure monotonicity (corrected p-values must be non-decreasing in sorted order)
        for i in range(n - 2, -1, -1):
            corrected[sorted_idx[i]] = min(corrected[sorted_idx[i]], corrected[sorted_idx[i + 1]])
        
        return list(corrected)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_se(data: list[float],
                 n_bootstrap: int = 1000,
                 seed: int = 42) -> float:
    """Bootstrap standard error.
    
    Parameters
    ----------
    data : list[float]
    n_bootstrap : int
    
    Returns
    -------
    Bootstrap SE
    """
    rng = np.random.default_rng(seed)
    array = np.asarray(data, dtype=float)
    
    bootstrap_means = [
        np.mean(rng.choice(array, size=len(array), replace=True))
        for _ in range(n_bootstrap)
    ]
    
    return float(np.std(bootstrap_means, ddof=1))
