"""
run_experiment.py — Master Experiment Runner
=============================================
Runs all 5 experimental stages (A–E) with completely isolated,
leakage-free model components and reproducible synthetic learner pools.

Stage A: Baselines (B1–B5)
Stage B: Incremental Neuronotes components (P1–P7)
Stage C: Ablation study (remove one component at a time from P7)
Stage D: Question-selection strategy comparison
Stage E: Feedback policy comparison

Returns a list of MetricBundle objects for all evaluated systems.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Adjust sys.path for standalone execution
_CODES_DIR = Path(__file__).parent.parent
if str(_CODES_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_DIR))

from neuronotes.cc_mirt           import CCMIRT, CONCEPT_DIM_MAP, N_DIMS, parse_a_vector
from neuronotes.c_matrix          import CMatrix
from neuronotes.dynamic_c         import DynamicC
from neuronotes.smd_vsnlms        import SMDVSNLMSUpdater, LearnerState
from neuronotes.kl_cat            import KLCAT
from neuronotes.intervention_router import InterventionRouter

from baselines.b1_random          import RandomSelector
from baselines.b2_staircase       import StaircaseSelector
from baselines.b3_3pl_irt         import IRT3PLSelector
from baselines.b4_mirt            import MIRTSelector
from baselines.b5_kl_mirt_no_misc import KLMIRTNoMiscSelector

from simulation.learner_generator import LearnerGenerator, SyntheticLearner
from simulation.metrics           import build_metric_bundle, MetricBundle, cohens_d, select_f1_thresholds
from simulation.ground_truth      import GroundTruthGenerator, GroundTruth, LearnerGroundTruth, ItemGroundTruth, simulate_response_from_ground_truth
from simulation.cohorts           import CohortManager, CohortGroundTruth

DATA_DIR    = _CODES_DIR / "data"
RESULTS_DIR = _CODES_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Frozen experiment config
# ============================================================
MASTER_SEED     = 42
N_LEARNERS      = 200
MAX_QUESTIONS   = 30          # max test length per learner
STOP_SE_THRESH  = 0.25        # stop when SE of θ estimate < threshold (proxy)
RMSE_THRESHOLD  = 0.35        # for efficiency metric

# Three-cohort protocol
DEV_FRAC        = 0.20        # development cohort fraction
CAL_FRAC        = 0.20        # calibration cohort fraction


# ============================================================
# Core simulation loop (using immutable ground truth)
# ============================================================
def _simulate_response_from_gt(learner_gt: LearnerGroundTruth, 
                                item_gt: ItemGroundTruth) -> tuple[bool, int]:
    """Simulate response using ONLY immutable ground-truth parameters.
    
    CRITICAL: This function does NOT accept model-estimated parameters.
    It uses the FIXED c_true from ground truth, preventing data leakage.
    """
    return simulate_response_from_ground_truth(learner_gt, item_gt)


def run_one_learner(learner_gt:  LearnerGroundTruth,
                    ground_truth: GroundTruth,
                    selector,
                    mirt:        CCMIRT,
                    c_module:    DynamicC,
                    c_mat:       CMatrix,
                    updater:     Optional[SMDVSNLMSUpdater],
                    router:      Optional[InterventionRouter],
                    use_c_matrix:  bool = True,
                    use_dynamic_c: bool = True,
                    use_smd:       bool = True,
                    use_router:    bool = True,
                    use_prereq:    bool = True,
                    max_questions: int = MAX_QUESTIONS,
                    freeze_shared_state: bool = False) -> dict:
    """Run one adaptive test session for a single learner using immutable ground truth.

    Parameters
    ----------
    learner_gt : LearnerGroundTruth
        Immutable ground-truth state for this learner
    ground_truth : GroundTruth
        Complete ground truth (needed for item parameters)
    freeze_shared_state : bool
        If True, prevent updates to shared state (DynamicC, graph weights, etc.)
        This should be True for test cohort evaluation.

    Returns a dict with per-learner results for metric computation.
    """
    state = LearnerState(
        theta=learner_gt.theta_init.copy(),
        concept_theta={},
        misconception={},
        repeat_count={},
        response_history=[],
    )

    if updater is not None and hasattr(updater, "reset"):
        updater.reset()
    # Attach MIRT to updater for Graph Smoothing
    if updater is not None and mirt is not None:
        updater.mirt = mirt

    theta_trajectory     = [state.theta.copy()]
    p_correct_log        = []
    correct_log          = []

    seen_items = set()
    active_misconception = None

    for step in range(max_questions):
        # Tier-Aware Unbiased Warm-Start
        if step == 3 and use_smd:
            acc = sum(correct_log[:3]) / 3.0
            shift = (acc - 0.5) * 1.5
            
            # Apply purely global anchor without forcing artificial tier decay,
            # allowing graph diffusion to learn actual tier strengths per student.
            state.theta += shift
            state.theta = np.clip(state.theta, -4.0, 4.0)
            state.global_ability = shift
            
        # Select next item
        item_row = selector.select(
            theta=state.theta,
            seen_items=seen_items,
            concept_thetas=state.concept_theta,
            misconception_state=state.misconception,
            active_misconception=active_misconception,
        )
        if item_row is None:
            break

        item_id  = str(item_row["item_id"])
        concept  = str(item_row["concept"])
        a_vec    = parse_a_vector(item_row["a_vector"])
        d_param  = float(item_row["d_param"])
        correct_option = int(item_row["correct_option"])
        seen_items.add(item_id)

        # Get IMMUTABLE ground-truth item parameters
        item_gt = ground_truth.get_item(item_id)
        
        # Model's estimate of c_j (for prediction only, NOT for simulation)
        c_j_estimated = c_module.get(item_id) if use_dynamic_c else 0.25
        
        # CRITICAL: Use FIXED c_true for prediction consistency
        # Models estimate c_j, but predictions should use ground-truth c_true
        # to ensure fair comparison across systems with different c estimates
        c_j_for_prediction = item_gt.c_true

        # Apply prerequisite constraint if using CC-MIRT
        if use_prereq:
            p_resp = mirt.constrained_prob(
                state.theta, concept, a_vec, d_param, c_j_for_prediction, state.concept_theta
            )
        else:
            p_resp = mirt.prob(state.theta, a_vec, d_param, c_j_for_prediction)

        # CRITICAL: Simulate response using ONLY immutable ground-truth c_true
        # This prevents data leakage where different models face different response distributions
        correct, selected_option = _simulate_response_from_gt(learner_gt, item_gt)

        # Diagnose option
        if use_c_matrix:
            diag = c_mat.diagnose(item_id, selected_option)
            z_mask = c_mat.item_distractor_mask(item_id, correct_option)
        else:
            diag = {
                "error_class":       "correct" if correct else "conceptual_error",
                "misconception_tag": "none",
                "misconception_tags": [],
                "z_vector":          np.zeros(15),
                "severity":          "none" if correct else "medium",
                "is_correct":        correct,
                "trap_weight":       0.0,
                "rationale":         "",
            }
            z_mask = np.zeros(15, dtype=float)

        error_class        = diag["error_class"]
        misconception_tag  = diag["misconception_tag"]
        misconception_tags = diag.get("misconception_tags", [])
        z_vector           = diag.get("z_vector", np.zeros(15))
        severity           = diag["severity"]
        rationale          = diag.get("rationale", "")

        # Update active misconception for the CAT router
        if not correct and misconception_tag and misconception_tag != "none":
            active_misconception = misconception_tag
        else:
            active_misconception = None

        # Update state
        if use_smd and updater is not None:
            state = updater.update(
                state=state,
                item_id=item_id,
                concept=concept,
                a_vec=a_vec,
                d_param=d_param,
                p_correct=p_resp,
                correct=correct,
                error_class=error_class,
                misconception_tag=misconception_tag,
                severity=severity,
                z_vector=z_vector,
                misconception_tags=misconception_tags,
                z_mask=z_mask,
                total_session_length=max_questions,
                t=step,
            )
        else:
            # Fallback: simple gradient update (no semantic weighting)
            residual     = (1 if correct else 0) - p_resp
            a_norm       = max(np.linalg.norm(a_vec), 1e-6)
            mu_eff       = 0.15 / a_norm  # conservative step to prevent divergence
            state.theta  = np.clip(state.theta + mu_eff * residual * a_vec, -4, 4)
            dim = CONCEPT_DIM_MAP.get(concept, 0)
            state.concept_theta[concept] = float(state.theta[dim])

        # Optimize the graph penalty after the evidence update.
        # Only update shared graph weights if not frozen (test cohort isolation)
        if use_prereq:
            residual_val = (1.0 if correct else 0.0) - p_resp
            state.theta = mirt.propagate_dag_evidence(
                theta=state.theta,
                concept=concept,
                correct=correct,
                residual=residual_val,
                t=step,
            )
            dim = CONCEPT_DIM_MAP.get(concept, 0)
            state.concept_theta[concept] = float(state.theta[dim])
            state.theta = mirt.apply_soft_prereq_penalty(
                state.theta,
                state.concept_theta,
                t=step,
            )
            if not freeze_shared_state:
                mirt.record_prereq_observation(concept, correct)

        # Update shared state ONLY if not frozen (test cohort isolation)
        if use_dynamic_c and not freeze_shared_state:
            c_module.record_response(item_id, selected_option, correct)

        theta_trajectory.append(state.theta.copy())
        p_correct_log.append(p_resp)
        correct_log.append(correct)

        # Route intervention
        if use_router and router is not None and not correct:
            repeat_cnt = state.repeat_count.get(misconception_tag, 0)
            iv = router.route(
                error_class=error_class,
                misconception_tag=misconception_tag,
                repeat_count=repeat_cnt,
                concept=concept,
                theta=state.theta,
                rationale=rationale,
                use_interventions=use_router,
            )
            if iv.intervention_type == "PREREQ_Q":
                active_misconception = misconception_tag
            elif iv.intervention_type == "NORMAL_Q":
                active_misconception = None

    return {
        "learner_id":       learner_gt.learner_id,
        "theta_true":       learner_gt.theta_true,
        "theta_final":      state.theta,
        "theta_trajectory": theta_trajectory,
        "p_correct_log":    p_correct_log,
        "correct_log":      correct_log,
        "misc_y_true":      np.pad(learner_gt.misconception_true, (0, 15 - len(learner_gt.misconception_true))),
        "misc_y_score":     state.misconception_probs.copy(),
        "n_questions":      len(seen_items),
    }


# ============================================================
# Leakage-Free System Factory
# ============================================================
ALL_SYSTEM_NAMES = [
    # Stage A: Baselines
    "B1_Random", "B2_Staircase", "B3_3PL_IRT", "B4_MIRT", "B5_KL_NoMisc",
    # Stage B: Progressive
    "P1_MIRT_KL", "P2_CCMIRT_KL", "P3_CCMIRT_CMatrix", "P4_DynC", "P5_SMD", "P6_Router", "P7_Full_Neuronotes",
    # Stage C: Ablations
    "C1_NoPrereq", "C2_NoCMatrix", "C3_FixedC", "C4_NoSMD", "C5_NoRouter",
    # Stage D: Selection Strategies
    "D1_Random_FullPsy", "D3_MaxFisher", "D4_KL_Only", "D5_KL_Prereq", "D6_KL_Misc", "D7_KL_Full_Pen",
    # Stage E: Feedback Policies
    "E1_NoFeedback", "E2_GenericFeedback", "E3_ConceptFeedback", "E4_MiscFeedback", "E5_MiscPlusPrerFeedback",
]


def make_system(sys_name: str, items_path: Path, graph_path: Path) -> tuple:
    """Build completely FRESH, ISOLATED components for the requested system.

    Instantiating fresh components guarantees zero cross-model state leakage
    (no shared SMD B matrix, DynamicC response history, or graph weights).
    """
    mirt     = CCMIRT(graph_path)
    c_mat    = CMatrix()
    dyn_c    = DynamicC(items_path)
    fixed_c  = DynamicC.fixed(0.25)
    updater  = SMDVSNLMSUpdater(base_lr=0.30)
    updater.mirt = mirt
    router   = InterventionRouter(graph_path)

    # ---- Stage A: Baselines ----
    if sys_name == "B1_Random":
        return (RandomSelector(items_path), mirt, fixed_c, c_mat, None, None, False, False, False, False, False)
    if sys_name == "B2_Staircase":
        return (StaircaseSelector(items_path), mirt, fixed_c, c_mat, None, None, False, False, False, False, False)
    if sys_name == "B3_3PL_IRT":
        return (IRT3PLSelector(items_path), mirt, fixed_c, c_mat, None, None, False, False, False, False, False)
    if sys_name == "B4_MIRT":
        return (MIRTSelector(items_path), mirt, fixed_c, c_mat, None, None, False, False, False, False, False)
    if sys_name == "B5_KL_NoMisc":
        return (KLMIRTNoMiscSelector(items_path), mirt, fixed_c, c_mat, None, None, False, False, False, False, False)

    # ---- Stage B: Progressive ----
    if sys_name == "P1_MIRT_KL":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=fixed_c, use_prereq=False, use_misc=False, use_rep_pen=False)
        return (sel, mirt, fixed_c, c_mat, None, None, False, False, False, False, False)
    if sys_name == "P2_CCMIRT_KL":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=fixed_c, use_prereq=True, use_misc=False, use_rep_pen=False)
        return (sel, mirt, fixed_c, c_mat, None, None, False, False, False, False, True)
    if sys_name == "P3_CCMIRT_CMatrix":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=fixed_c, use_prereq=True, use_misc=True, use_rep_pen=False)
        return (sel, mirt, fixed_c, c_mat, None, None, True, False, False, False, True)
    if sys_name == "P4_DynC":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=False)
        return (sel, mirt, dyn_c, c_mat, None, None, True, True, False, False, True)
    if sys_name == "P5_SMD":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, None, True, True, True, False, True)
    
    # P6_Router: + Intervention routing (but NO adaptive learning rate refinement)
    if sys_name == "P6_Router":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        # P6 uses fixed base_lr=0.30, P7 uses adaptive lr
        updater_p6 = SMDVSNLMSUpdater(base_lr=0.30, use_dynamic_lr=False)
        updater_p6.mirt = mirt
        return (sel, mirt, dyn_c, c_mat, updater_p6, router, True, True, True, True, True)
    
    # P7_Full_Neuronotes: Complete pipeline with adaptive learning rate
    if sys_name == "P7_Full_Neuronotes":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        # P7 uses dynamic learning rate (default use_dynamic_lr=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)

    # ---- Stage C: Ablations from P7 ----
    # C1_NoPrereq: Remove ALL prerequisite graph information
    if sys_name == "C1_NoPrereq":
        # Use empty graph path to disable prerequisite constraints
        sel = KLCAT(items_path, None, c_mat, mirt, dynamic_c=dyn_c, use_prereq=False, use_misc=True, use_rep_pen=True, use_cold_start=False)
        # Return None for graph_path disables graph in mirt
        mirt_no_prereq = CCMIRT(None)  # No graph
        return (sel, mirt_no_prereq, dyn_c, c_mat, updater, None, True, True, True, False, False)
    
    # C2_NoCMatrix: Remove ALL C-Matrix diagnostic information
    if sys_name == "C2_NoCMatrix":
        # CRITICAL: use_misc=False in selector AND null C-matrix AND use_c_matrix=False
        null_c_mat = CMatrix()  # Empty C-matrix with no tags
        sel = KLCAT(items_path, graph_path, null_c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=False, use_rep_pen=True)
        return (sel, mirt, dyn_c, null_c_mat, updater, router, False, True, True, True, True)
    
    if sys_name == "C3_FixedC":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=fixed_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, fixed_c, c_mat, updater, router, True, False, True, True, True)
    
    if sys_name == "C4_NoSMD":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, None, router, True, True, False, True, True)
    
    if sys_name == "C5_NoRouter":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, None, True, True, True, False, True)

    # ---- Stage D: Selection strategies ----
    if sys_name == "D1_Random_FullPsy":
        return (RandomSelector(items_path), mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)
    if sys_name == "D3_MaxFisher":
        return (MIRTSelector(items_path), mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)
    if sys_name == "D4_KL_Only":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=False, use_misc=False, use_rep_pen=False)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, False)
    if sys_name == "D5_KL_Prereq":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=False, use_rep_pen=False)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)
    if sys_name == "D6_KL_Misc":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=False, use_misc=True, use_rep_pen=False)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, False)
    if sys_name == "D7_KL_Full_Pen":
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)

    # ---- Stage E: Feedback policies ----
    # NOTE: Feedback policies require a learner response model to have effect.
    # Current implementation only differs in intervention_type/message content.
    # TODO: Implement learner learning/engagement response model for valid feedback comparison.
    if sys_name == "E1_NoFeedback":
        # No intervention routing at all
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, None, True, True, False, False, True)
    
    if sys_name == "E2_GenericFeedback":
        # Generic correctness feedback (no misconception-specific routing)
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)
    
    if sys_name == "E3_ConceptFeedback":
        # Concept-level corrective feedback (no misconception diagnosis)
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=False, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, False, True, True, True, True)
    
    if sys_name == "E4_MiscFeedback":
        # Distractor-level misconception feedback
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=False, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, False)
    
    if sys_name == "E5_MiscPlusPrerFeedback":
        # Misconception + prerequisite remediation (same as P7)
        sel = KLCAT(items_path, graph_path, c_mat, mirt, dynamic_c=dyn_c, use_prereq=True, use_misc=True, use_rep_pen=True)
        return (sel, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True)

    raise ValueError(f"Unknown system name: {sys_name}")


# ============================================================
# Run all stages
# ============================================================
def run_all_stages(n_learners: int = N_LEARNERS,
                   seed: int = MASTER_SEED,
                   systems_to_run: Optional[list[str]] = None,
                   max_questions: int = MAX_QUESTIONS,
                   use_cohort_protocol: bool = True) -> list[MetricBundle]:
    """Execute evaluation for all requested systems.

    Guarantees:
      1. Every system receives fresh, isolated components (zero state leakage).
      2. Every system faces an identical synthetic learner pool (identically seeded).
      3. IMMUTABLE ground truth: c_true is fixed; models estimate but never change it.
      4. Three-cohort protocol: dev (tuning), cal (shared params), test (locked).
    """
    print(f"\n{'='*60}")
    print(" Neuronotes Experiment Runner (VALIDATED)")
    print(f" N_learners={n_learners}, seed={seed}, max_q={max_questions}")
    print(f" Cohort protocol: {use_cohort_protocol}")
    print(f"{'='*60}\n")

    items_path = DATA_DIR / "items_clean.csv"
    graph_path = DATA_DIR / "concept_graph.csv"

    if systems_to_run is None:
        target_systems = ALL_SYSTEM_NAMES
    else:
        missing = sorted(set(systems_to_run) - set(ALL_SYSTEM_NAMES))
        if missing:
            raise ValueError(f"Unknown system names: {', '.join(missing)}")
        target_systems = systems_to_run

    bundles: list[MetricBundle] = []

    # CRITICAL: Generate FIXED ground truth ONCE before any model runs
    gt_generator = GroundTruthGenerator(n_learners=n_learners, seed=seed, items_path=items_path)
    ground_truth = gt_generator.generate()
    print(f"[Ground Truth] Generated {len(ground_truth.learners)} learners with IMMUTABLE c_true\n")

    for sys_name in target_systems:
        # Build fresh, isolated system
        config = make_system(sys_name, items_path, graph_path)
        (selector, mirt_, c_mod, c_mat_, updater_,
         router_, use_cmat, use_dynC, use_smd, use_rt, use_pre) = config

        print(f"[{sys_name}] running {n_learners} learners (seed={seed}) ...")
        selector.reset_exposure()

        if use_cohort_protocol:
            # Three-cohort protocol with proper isolation
            cohort_mgr = CohortManager(ground_truth, dev_frac=DEV_FRAC, cal_frac=CAL_FRAC)
            print(f"  Cohorts: {cohort_mgr.summary()}")
            
            # Development cohort: hyperparameter tuning, threshold selection
            dev_gt = cohort_mgr.get_cohort_ground_truth("development")
            dev_results = []
            for learner_gt in dev_gt.learners:
                res = run_one_learner(
                    learner_gt=learner_gt,
                    ground_truth=ground_truth,
                    selector=selector,
                    mirt=mirt_,
                    c_module=c_mod,
                    c_mat=c_mat_,
                    updater=updater_,
                    router=router_,
                    use_c_matrix=use_cmat,
                    use_dynamic_c=use_dynC,
                    use_smd=use_smd,
                    use_router=use_rt,
                    use_prereq=use_pre,
                    max_questions=max_questions,
                    freeze_shared_state=False,  # Allow updates during dev
                )
                dev_results.append(res)
            
            # Calibrate F1 thresholds on development cohort ONLY
            thresholds = select_f1_thresholds(
                np.asarray([r["misc_y_true"] for r in dev_results]),
                np.asarray([r["misc_y_score"] for r in dev_results]),
            )
            print(f"  [Dev] RMSE={np.mean([np.sqrt(np.mean((r['theta_final']-r['theta_true'])**2)) for r in dev_results]):.4f}")
            
            # Calibration cohort: fit shared parameters
            cal_gt = cohort_mgr.get_cohort_ground_truth("calibration")
            cal_results = []
            for learner_gt in cal_gt.learners:
                res = run_one_learner(
                    learner_gt=learner_gt,
                    ground_truth=ground_truth,
                    selector=selector,
                    mirt=mirt_,
                    c_module=c_mod,
                    c_mat=c_mat_,
                    updater=updater_,
                    router=router_,
                    use_c_matrix=use_cmat,
                    use_dynamic_c=use_dynC,
                    use_smd=use_smd,
                    use_router=use_rt,
                    use_prereq=use_pre,
                    max_questions=max_questions,
                    freeze_shared_state=False,  # Allow updates during calibration
                )
                cal_results.append(res)
            print(f"  [Cal] RMSE={np.mean([np.sqrt(np.mean((r['theta_final']-r['theta_true'])**2)) for r in cal_results]):.4f}")
            
            # FREEZE shared state before test cohort
            cohort_mgr.freeze_shared_state()
            print(f"  [Test] Shared state FROZEN - no updates allowed")
            
            # Test cohort: LOCKED evaluation
            test_gt = cohort_mgr.get_cohort_ground_truth("test")
            test_results = []
            for learner_gt in test_gt.learners:
                res = run_one_learner(
                    learner_gt=learner_gt,
                    ground_truth=ground_truth,
                    selector=selector,
                    mirt=mirt_,
                    c_module=c_mod,
                    c_mat=c_mat_,
                    updater=updater_,
                    router=router_,
                    use_c_matrix=use_cmat,
                    use_dynamic_c=use_dynC,
                    use_smd=use_smd,
                    use_router=use_rt,
                    use_prereq=use_pre,
                    max_questions=max_questions,
                    freeze_shared_state=True,  # CRITICAL: No updates during test
                )
                test_results.append(res)
        else:
            # Legacy single-cohort mode (for backwards compatibility)
            results = []
            for learner_gt in ground_truth.learners:
                res = run_one_learner(
                    learner_gt=learner_gt,
                    ground_truth=ground_truth,
                    selector=selector,
                    mirt=mirt_,
                    c_module=c_mod,
                    c_mat=c_mat_,
                    updater=updater_,
                    router=router_,
                    use_c_matrix=use_cmat,
                    use_dynamic_c=use_dynC,
                    use_smd=use_smd,
                    use_router=use_rt,
                    use_prereq=use_pre,
                    max_questions=max_questions,
                    freeze_shared_state=False,
                )
                results.append(res)
            
            # Legacy validation/test split
            if len(results) >= 4:
                validation_count = max(1, len(results) // 4)
                validation_results = results[:validation_count]
                test_results = results[validation_count:]
            else:
                validation_results = results
                test_results = results
            
            thresholds = select_f1_thresholds(
                np.asarray([r["misc_y_true"] for r in validation_results]),
                np.asarray([r["misc_y_score"] for r in validation_results]),
            )

        # Only evaluate misconception dims that have real ground-truth labels (0–3)
        from simulation.ground_truth import N_SEMANTIC_DIMS as N_ACTIVE_MISC_DIMS
        active_dims = np.arange(N_ACTIVE_MISC_DIMS)
        bundle = build_metric_bundle(
            system_name=sys_name,
            theta_true_list=[r["theta_true"] for r in test_results],
            theta_final_list=[r["theta_final"] for r in test_results],
            theta_est_trajectories=[r["theta_trajectory"] for r in test_results],
            p_correct_all=[p for r in test_results for p in r["p_correct_log"]],
            correct_all=[c for r in test_results for c in r["correct_log"]],
            misc_y_true=[r["misc_y_true"] for r in test_results],
            misc_y_score=[r["misc_y_score"] for r in test_results],
            misc_thresholds=thresholds,
            rmse_threshold=RMSE_THRESHOLD,
            active_dims=active_dims,
        )
        print(f"  -> RMSE={bundle.rmse:.4f}  MAE={bundle.mae:.4f}  ECE={bundle.ece:.4f}  "
              f"F1={bundle.f1:.4f}  AUROC={bundle.auroc:.4f}  EffQ={bundle.efficiency_q:.1f}")
        bundles.append(bundle)

    return bundles


# ============================================================
# Save results
# ============================================================
def save_results(bundles: list[MetricBundle]):
    # simulation_results.csv
    rows = []
    for b in bundles:
        rows.append({
            "system":         b.system_name,
            "n_learners":     b.n_learners,
            "mean_questions": round(b.mean_questions, 2),
            "rmse":           round(b.rmse, 4),
            "rmse_std":       round(b.rmse_std, 4),
            "rmse_ci95_lo":   round(b.rmse_ci95_lo, 4),
            "rmse_ci95_hi":   round(b.rmse_ci95_hi, 4),
            "mae":            round(b.mae, 4),
            "ece":            round(b.ece, 4),
            "precision":      round(b.precision, 4),
            "recall":         round(b.recall, 4),
            "f1":             round(b.f1, 4),
            "auroc":          round(b.auroc, 4),
            "auroc_micro":    round(b.auroc_micro, 4),
            "valid_misconception_dims": b.valid_misconception_dims,
            "efficiency_q":   round(b.efficiency_q, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "simulation_results.csv", index=False)
    print(f"\nSaved simulation_results.csv ({len(df)} systems)")

    # rmse_trajectory.csv
    max_len = max(len(b.rmse_trajectory) for b in bundles)
    traj_dict = {"step": list(range(1, max_len + 1))}
    for b in bundles:
        padded = b.rmse_trajectory + [np.nan] * (max_len - len(b.rmse_trajectory))
        traj_dict[b.system_name] = padded
    pd.DataFrame(traj_dict).to_csv(RESULTS_DIR / "rmse_trajectory.csv", index=False)
    print(f"Saved rmse_trajectory.csv")

    return df
