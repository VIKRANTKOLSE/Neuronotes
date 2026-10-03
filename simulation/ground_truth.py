"""
ground_truth.py — Immutable Ground-Truth Parameters for Valid Evaluation
=========================================================================
Generates FIXED item and learner parameters BEFORE any model runs.
Models may ESTIMATE parameters, but the simulator uses ONLY these immutable values.

Critical principle: 
  - c_true is generated once and NEVER modified
  - Models can estimate c_j, but simulator uses only c_true
  - This prevents the data leakage where P7 vs C3 faced different response distributions
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path

from neuronotes.cc_mirt import N_DIMS, CONCEPT_DIM_MAP, parse_a_vector
from neuronotes.c_matrix import CMatrix


N_SEMANTIC_DIMS = 4  # Active misconception dimensions (padded to 15 in evaluation)


@dataclass
class LearnerGroundTruth:
    """Immutable ground-truth state for ONE synthetic learner."""
    learner_id: str
    theta_true: np.ndarray              # (58,) true ability vector
    theta_init: np.ndarray              # (58,) initial estimate (zeros)
    misconception_true: np.ndarray      # (4,) binary misconception labels
    misconception_strength: np.ndarray  # (4,) continuous strength [0,1]
    profile: str                        # archetype name
    rng: np.random.Generator            # private RNG for response simulation


@dataclass
class ItemGroundTruth:
    """Immutable TRUE item parameters (never model-estimated)."""
    item_id: str
    a_true: np.ndarray      # (58,) discrimination vector
    d_true: float           # difficulty
    c_true: float           # TRUE guessing floor (immutable)
    correct_option: int
    concept: str
    
    # Distractor-level misconception associations (for option selection model)
    # Shape: (4 options, 4 misconception dims) — probability each option activates each misconception
    option_misc_probs: np.ndarray


@dataclass
class GroundTruth:
    """Complete immutable ground truth for an entire experiment."""
    learners: list[LearnerGroundTruth]
    items: dict[str, ItemGroundTruth]  # item_id -> ItemGroundTruth
    seed: int
    
    def get_item(self, item_id: str) -> ItemGroundTruth:
        return self.items[item_id]
    
    def get_learner(self, learner_id: str) -> LearnerGroundTruth:
        for l in self.learners:
            if l.learner_id == learner_id:
                return l
        raise KeyError(f"Learner {learner_id} not found")


class GroundTruthGenerator:
    """
    Generate FIXED ground truth BEFORE any model runs.
    
    This class replaces the on-the-fly parameter generation in run_experiment.py
    to ensure all systems face IDENTICAL response distributions.
    
    Supports multiple generator families for stress-testing:
      - "tier_correlated": Current 4-tier correlated profiles (default)
      - "independent": 58 independent dimensions N(0,1)
      - "noncompensatory": DAG-consistent prerequisite mastery
      - "mixed_population": Mix of different profile types
    """
    
    def __init__(self,
                 n_learners: int = 200,
                 seed: int = 42,
                 items_path: Optional[Path] = None,
                 profile_dist: Optional[dict] = None,
                 generator_family: str = "tier_correlated"):
        self.n_learners = n_learners
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.items_path = items_path or (Path(__file__).parent.parent / "data" / "items_clean.csv")
        self.generator_family = generator_family
        
        self.profile_dist = profile_dist or {
            "strong_all": 0.15,
            "weak_all": 0.15,
            "strong_found": 0.20,
            "strong_bonding": 0.20,
            "strong_coord": 0.15,
            "mixed": 0.15,
        }
        
        # Instance-level CMatrix for option-misconception mapping
        self._c_matrix = CMatrix()
        
        # Profile archetypes (from learner_generator.py)
        self._profiles = self._define_profiles()
    
    def _define_profiles(self) -> dict:
        """Define tier-correlated ability profiles."""
        def _make_profile(tier_means: tuple, tier_stds: tuple = (0.3, 0.3, 0.3, 0.3)) -> dict:
            m = np.concatenate([
                np.full(10, tier_means[0]),
                np.full(13, tier_means[1]),
                np.full(17, tier_means[2]),
                np.full(18, tier_means[3]),
            ])
            s = np.concatenate([
                np.full(10, tier_stds[0]),
                np.full(13, tier_stds[1]),
                np.full(17, tier_stds[2]),
                np.full(18, tier_stds[3]),
            ])
            return {"theta_mean": m, "theta_std": s}
        
        return {
            "strong_all": _make_profile((1.5, 1.5, 1.5, 1.5), (0.3, 0.3, 0.3, 0.3)),
            "weak_all": _make_profile((-1.5, -1.5, -1.5, -1.5), (0.3, 0.3, 0.3, 0.3)),
            "strong_found": _make_profile((1.2, 0.5, 0.0, -0.5), (0.2, 0.3, 0.4, 0.4)),
            "strong_bonding": _make_profile((0.5, 1.2, 1.0, 0.0), (0.4, 0.2, 0.3, 0.4)),
            "strong_coord": _make_profile((0.0, 0.3, 0.8, 1.5), (0.4, 0.4, 0.3, 0.2)),
            "mixed": _make_profile((0.0, 0.0, 0.0, 0.0), (0.8, 0.8, 0.8, 0.8)),
        }
    
    def generate(self) -> GroundTruth:
        """Generate complete immutable ground truth for all learners and items."""
        
        # Step 1: Generate FIXED item parameters
        items = self._generate_item_ground_truth()
        
        # Step 2: Generate FIXED learner parameters
        learners = self._generate_learner_ground_truths()
        
        return GroundTruth(
            learners=learners,
            items=items,
            seed=self.seed,
        )
    
    def _generate_item_ground_truth(self) -> dict[str, ItemGroundTruth]:
        """
        Generate FIXED item parameters from items_clean.csv.
        
        KEY PRINCIPLE: c_true is read from file or generated ONCE and NEVER modified.
        Models may ESTIMATE c_j, but this c_true is what the simulator uses.
        """
        if not self.items_path.exists():
            raise FileNotFoundError(f"Items file not found: {self.items_path}")
        
        items_df = pd.read_csv(self.items_path)
        items = {}
        
        for _, row in items_df.iterrows():
            item_id = str(row["item_id"])
            
            # Parse a_vector
            a_vec = parse_a_vector(row["a_vector"])
            
            # CRITICAL: Use FIXED c_j from file as c_true
            # If c_j column doesn't exist, use default 0.25
            c_true = float(row.get("c_j", 0.25))
            
            # Generate option-misconception probability matrix
            # This defines which distractor options activate which misconception dimensions
            option_misc_probs = self._compute_option_misc_probs(item_id, int(row["correct_option"]))
            
            items[item_id] = ItemGroundTruth(
                item_id=item_id,
                a_true=a_vec,
                d_true=float(row["d_param"]),
                c_true=c_true,  # IMMUTABLE
                correct_option=int(row["correct_option"]),
                concept=str(row["concept"]),
                option_misc_probs=option_misc_probs,
            )
        
        return items
    
    def _compute_option_misc_probs(self, item_id: str, correct_option: int) -> np.ndarray:
        """
        Compute probability that each option activates each misconception dimension.
        
        Returns: (4, 4) array where [option_idx, misc_dim] = probability
        """
        probs = np.zeros((4, N_SEMANTIC_DIMS), dtype=float)
        
        for opt in range(1, 5):
            if opt == correct_option:
                continue  # Correct option doesn't activate misconceptions
            
            # Get misconception tags for this option
            diag = self._c_matrix.diagnose(item_id, opt)
            tags = diag.get("misconception_tags", [])
            
            for tag in tags:
                if tag.startswith("z_"):
                    try:
                        dim = int(tag.split("_")[1])
                        if 0 <= dim < N_SEMANTIC_DIMS:
                            probs[opt - 1, dim] = 1.0
                    except (ValueError, IndexError):
                        pass
        
        return probs
    
    def _generate_learner_ground_truths(self) -> list[LearnerGroundTruth]:
        """Generate FIXED learner parameters (theta, misconceptions)."""
        
        learners = []
        
        for i in range(self.n_learners):
            if self.generator_family == "tier_correlated":
                theta_true, profile = self._generate_tier_correlated()
            elif self.generator_family == "independent":
                theta_true, profile = self._generate_independent()
            elif self.generator_family == "noncompensatory":
                theta_true, profile = self._generate_noncompensatory()
            elif self.generator_family == "mixed_population":
                theta_true, profile = self._generate_mixed_population()
            else:
                theta_true, profile = self._generate_tier_correlated()
            
            misconception_true, misconception_strength = self._generate_misconceptions(theta_true)
            
            learners.append(LearnerGroundTruth(
                learner_id=f"L{i:04d}",
                theta_true=theta_true,
                theta_init=np.zeros(N_DIMS),
                misconception_true=misconception_true,
                misconception_strength=misconception_strength,
                profile=profile,
                rng=np.random.default_rng(int(self.rng.integers(0, 2**31))),
            ))
        
        return learners
    
    def _generate_tier_correlated(self) -> tuple[np.ndarray, str]:
        """Current generator: 4-tier correlated profiles."""
        profile_names = list(self.profile_dist.keys())
        profile_probs = np.array([self.profile_dist[k] for k in profile_names])
        profile_probs /= profile_probs.sum()
        
        profile = self.rng.choice(profile_names, p=profile_probs)
        spec = self._profiles[profile]
        
        theta_true = np.clip(
            self.rng.normal(spec["theta_mean"], spec["theta_std"]),
            -3.0, 3.0
        )
        return theta_true, profile
    
    def _generate_independent(self) -> tuple[np.ndarray, str]:
        """Independent dimensions: each of 58 dims ~ N(0, 1)."""
        theta_true = np.clip(self.rng.normal(0, 1, N_DIMS), -3.0, 3.0)
        return theta_true, "independent"
    
    def _generate_noncompensatory(self) -> tuple[np.ndarray, str]:
        """DAG-consistent prerequisite mastery.
        
        If a child concept is mastered, all prerequisites must be mastered.
        This creates strong prerequisite graph consistency.
        """
        theta_true = self.rng.normal(0, 0.8, N_DIMS)
        
        # Load prerequisite graph for consistency constraints
        graph_path = self.items_path.parent / "concept_graph.csv"
        if graph_path.exists():
            import pandas as pd
            graph_df = pd.read_csv(graph_path)
            
            # Enforce: if child mastered (theta > 0), parent must be mastered
            for _, row in graph_df.iterrows():
                child = row.get("child")
                parent = row.get("parent")
                if child and parent:
                    from neuronotes.cc_mirt import CONCEPT_DIM_MAP
                    child_dim = CONCEPT_DIM_MAP.get(str(child), None)
                    parent_dim = CONCEPT_DIM_MAP.get(str(parent), None)
                    
                    if child_dim is not None and parent_dim is not None:
                        if theta_true[child_dim] > 0:
                            theta_true[parent_dim] = max(theta_true[parent_dim], 0.3)
        
        theta_true = np.clip(theta_true, -3.0, 3.0)
        return theta_true, "noncompensatory"
    
    def _generate_mixed_population(self) -> tuple[np.ndarray, str]:
        """Mix of different ability patterns for diversity testing."""
        pattern = self.rng.choice(["uniform", "bimodal", "sparse", "tier_correlated"])
        
        if pattern == "uniform":
            theta_true = self.rng.uniform(-2, 2, N_DIMS)
        elif pattern == "bimodal":
            theta_true = self.rng.choice([
                self.rng.normal(1.5, 0.5, N_DIMS),
                self.rng.normal(-1.5, 0.5, N_DIMS)
            ])
        elif pattern == "sparse":
            theta_true = self.rng.normal(0, 0.5, N_DIMS)
            sparse_mask = self.rng.random(N_DIMS) < 0.7
            theta_true[sparse_mask] = 0.0
        else:
            return self._generate_tier_correlated()
        
        theta_true = np.clip(theta_true, -3.0, 3.0)
        return theta_true, f"mixed_{pattern}"
    
    def _generate_misconceptions(self, theta_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate misconception labels and strengths."""
        misconception_true = np.zeros(N_SEMANTIC_DIMS, dtype=int)
        misconception_strength = np.zeros(N_SEMANTIC_DIMS, dtype=float)
        
        weakness = max(0.0, -float(np.mean(theta_true[:10]))) / 3.0
        
        for dim in range(N_SEMANTIC_DIMS):
            prevalence = min(0.08 + 0.52 * weakness + 0.02 * (dim % 3), 0.80)
            if self.rng.random() < prevalence:
                misconception_true[dim] = 1
                misconception_strength[dim] = float(self.rng.uniform(0.3, 0.9))
        
        return misconception_true, misconception_strength


def simulate_response_from_ground_truth(
    learner_gt: LearnerGroundTruth,
    item_gt: ItemGroundTruth,
) -> tuple[bool, int]:
    """
    Simulate a response using ONLY immutable ground-truth parameters.
    
    CRITICAL: This function does NOT accept any model-estimated parameters.
    It uses ONLY the fixed c_true from ground truth.
    
    Returns: (correct: bool, selected_option: int)
    """
    # Compute probability of correct response using TRUE parameters
    logit = float(np.dot(item_gt.a_true, learner_gt.theta_true)) + item_gt.d_true
    p_star = 1.0 / (1.0 + np.exp(-logit))
    
    # Use FIXED c_true (never model-estimated)
    p_correct = item_gt.c_true + (1.0 - item_gt.c_true) * p_star
    
    correct = learner_gt.rng.random() < p_correct
    
    if correct:
        return True, item_gt.correct_option
    
    # Select wrong option
    correct_opt = item_gt.correct_option
    wrong_opts = [o for o in [1, 2, 3, 4] if o != correct_opt]
    
    # Bias wrong option selection by learner's misconceptions
    weights = np.ones(3, dtype=float)
    
    for j, opt in enumerate(wrong_opts):
        # Option's misconception activation probabilities
        opt_misc = item_gt.option_misc_probs[opt - 1]  # (4,)
        
        # Weight by learner's misconception strength
        for dim in range(N_SEMANTIC_DIMS):
            if learner_gt.misconception_true[dim]:
                weights[j] += learner_gt.misconception_strength[dim] * opt_misc[dim] * 2.0
    
    weights /= weights.sum()
    chosen_wrong = learner_gt.rng.choice(wrong_opts, p=weights)
    
    return False, int(chosen_wrong)
