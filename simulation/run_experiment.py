"""
run_experiment.py — Master Experiment Runner
=============================================
Runs all 5 experimental stages (A–E) from the blueprint with
frozen experimental conditions (seed, item bank, learner pool).

Stage A: Baselines (B1–B5)
Stage B: Incremental Neuronotes components (P1–P7)
Stage C: Ablation study (remove one component at a time from P7)
Stage D: Question-selection strategy comparison
Stage E: Feedback policy comparison

Returns a list of MetricBundle objects for all systems.
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

from neuronotes.cc_mirt           import CCMIRT, CONCEPT_DIM_MAP
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
from simulation.metrics           import build_metric_bundle, MetricBundle, cohens_d

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


# ============================================================
# Core simulation loop
# ============================================================
def _simulate_response(learner: SyntheticLearner, item_row, c_j: float = 0.25):
    """Call the learner generator's simulate_response using the learner's own RNG."""
    # We can just instantiate a dummy LearnerGenerator since it only uses learner.rng
    return LearnerGenerator(seed=0).simulate_response(learner, item_row, c_j)


def run_one_learner(learner:     SyntheticLearner,
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
                    use_prereq:    bool = True) -> dict:
    """Run one adaptive test session for a single learner.

    Returns a dict with per-learner results for metric computation.
    """
    state = LearnerState(
        theta=learner.theta_init.copy(),
        concept_theta={},
        misconception={},
        repeat_count={},
        response_history=[],
    )

    theta_trajectory     = [state.theta.copy()]
    p_correct_log        = []
    correct_log          = []
    misc_y_true_log      = []
    misc_y_pred_log      = []
    misc_y_score_log     = []

    seen_items = set()
    active_misconception = None

    for step in range(MAX_QUESTIONS):
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
        a_vec    = np.array([float(item_row["a1"]),
                             float(item_row["a2"]),
                             float(item_row["a3"])])
        d_param  = float(item_row["d_param"])
        correct_option = int(item_row["correct_option"])
        seen_items.add(item_id)

        # Get dynamic c_j
        c_j = c_module.get(item_id) if use_dynamic_c else 0.25

        # Apply prerequisite constraint if using CC-MIRT
        if use_prereq:
            p_resp = mirt.constrained_prob(
                state.theta, concept, a_vec, d_param, c_j, state.concept_theta
            )
        else:
            p_resp = mirt.prob(state.theta, a_vec, d_param, c_j)

        # Simulate response using learner's own deterministic RNG
        correct, selected_option = _simulate_response(learner, item_row, c_j)

        # Diagnose option
        if use_c_matrix:
            diag = c_mat.diagnose(item_id, selected_option)
        else:
            diag = {
                "error_class":       "correct" if correct else "conceptual_error",
                "misconception_tag": "none",
                "severity":          "none" if correct else "medium",
                "is_correct":        correct,
                "trap_weight":       0.0,
                "rationale":         "",
            }

        error_class       = diag["error_class"]
        misconception_tag = diag["misconception_tag"]
        severity          = diag["severity"]
        rationale         = diag.get("rationale", "")

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
            )
        else:
            # Fallback: simple gradient update (no semantic weighting)
            residual     = (1 if correct else 0) - p_resp
            a_norm       = max(np.linalg.norm(a_vec), 1e-6)
            mu_eff       = 0.15 / a_norm  # conservative step to prevent divergence
            state.theta  = np.clip(state.theta + mu_eff * residual * a_vec, -4, 4)
            dim = CONCEPT_DIM_MAP.get(concept, 0)
            state.concept_theta[concept] = float(state.theta[dim])

        theta_trajectory.append(state.theta.copy())
        p_correct_log.append(p_resp)
        correct_log.append(correct)

        # Misconception ground-truth detection
        true_misc = learner.misconception_true
        for tag in true_misc:
            is_true_misc  = int(true_misc[tag] > 0.3)
            is_pred_misc  = int(state.misconception.get(tag, 0.0) > 0.14)
            score_misc    = float(state.misconception.get(tag, 0.0))
            misc_y_true_log.append(is_true_misc)
            misc_y_pred_log.append(is_pred_misc)
            misc_y_score_log.append(score_misc)

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
        "learner_id":       learner.learner_id,
        "theta_true":       learner.theta_true,
        "theta_final":      state.theta,
        "theta_trajectory": theta_trajectory,
        "p_correct_log":    p_correct_log,
        "correct_log":      correct_log,
        "misc_y_true":      misc_y_true_log,
        "misc_y_pred":      misc_y_pred_log,
        "misc_y_score":     misc_y_score_log,
        "n_questions":      len(seen_items),
    }


# ============================================================
# Build system configurations
# ============================================================
def make_systems(items_path: Path, graph_path: Path) -> dict:
    """Build all selector / component configs keyed by system name."""
    mirt     = CCMIRT(graph_path)
    c_mat    = CMatrix()
    dyn_c    = DynamicC(items_path)
    fixed_c  = DynamicC.fixed(0.25)
    updater  = SMDVSNLMSUpdater()
    router   = InterventionRouter(graph_path)

    klcat_full = KLCAT(items_path, graph_path, c_mat, mirt,
                        use_prereq=True, use_misc=True, use_rep_pen=True)

    systems = {
        # ---- Stage A: Baselines ----
        "B1_Random":          (RandomSelector(items_path),       mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),
        "B2_Staircase":       (StaircaseSelector(items_path),    mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),
        "B3_3PL_IRT":         (IRT3PLSelector(items_path),       mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),
        "B4_MIRT":            (MIRTSelector(items_path),         mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),
        "B5_KL_NoMisc":       (KLMIRTNoMiscSelector(items_path), mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),

        # ---- Stage B: Progressive ----
        "P1_MIRT_KL":         (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=False, use_misc=False, use_rep_pen=False),
                               mirt, fixed_c, c_mat, None,    None,   False, False, False, False, False),
        "P2_CCMIRT_KL":       (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=False, use_rep_pen=False),
                               mirt, fixed_c, c_mat, None,    None,   False, False, False, False, True),
        "P3_CCMIRT_CMatrix":  (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=False),
                               mirt, fixed_c, c_mat, None,    None,   True,  False, False, False, True),
        "P4_DynC":            (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=False),
                               mirt, dyn_c,   c_mat, None,    None,   True,  True,  False, False, True),
        "P5_SMD":             (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=True),
                               mirt, dyn_c,   c_mat, updater, None,   True,  True,  True,  False, True),
        "P6_Router":          (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=True),
                               mirt, dyn_c,   c_mat, updater, router, True,  True,  True,  True,  True),
        "P7_Full_Neuronotes": (klcat_full,
                               mirt, dyn_c,   c_mat, updater, router, True,  True,  True,  True,  True),

        # ---- Stage C: Ablations from P7 ----
        "C1_NoPrereq":        (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=False, use_misc=True,  use_rep_pen=True),
                               mirt, dyn_c,   c_mat, updater, router, True,  True,  True,  True,  False),
        "C2_NoCMatrix":       (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=False, use_rep_pen=True),
                               mirt, dyn_c,   c_mat, updater, router, False, True,  True,  True,  True),
        "C3_FixedC":          (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=True),
                               mirt, fixed_c, c_mat, updater, router, True,  False, True,  True,  True),
        "C4_NoSMD":           (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=True),
                               mirt, dyn_c,   c_mat, None,    router, True,  True,  False, True,  True),
        "C5_NoRouter":        (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True,  use_misc=True,  use_rep_pen=True),
                               mirt, dyn_c,   c_mat, updater, None,   True,  True,  True,  False, True),

        # ---- Stage D: Selection strategies (share P7 psychometrics) ----
        "D1_Random_FullPsy":  (RandomSelector(items_path),       mirt, dyn_c, c_mat, updater, router, True, True, True, True, True),
        "D3_MaxFisher":       (MIRTSelector(items_path),          mirt, dyn_c, c_mat, updater, router, True, True, True, True, True),
        "D4_KL_Only":         (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=False, use_misc=False, use_rep_pen=False),
                               mirt, dyn_c, c_mat, updater, router, True, True, True, True, False),
        "D5_KL_Prereq":       (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=True, use_misc=False, use_rep_pen=False),
                               mirt, dyn_c, c_mat, updater, router, True, True, True, True, True),
        "D6_KL_Misc":         (KLCAT(items_path, graph_path, c_mat, mirt, use_prereq=False, use_misc=True, use_rep_pen=False),
                               mirt, dyn_c, c_mat, updater, router, True, True, True, True, False),
        "D7_KL_Full_Pen":     (klcat_full, mirt, dyn_c, c_mat, updater, router, True, True, True, True, True),

        # ---- Stage E: Feedback policies (share P7 selector + psychometrics) ----
        "E1_NoFeedback":      (klcat_full, mirt, dyn_c, c_mat, updater, None,   True, True, True, False, True),
        "E2_GenericFeedback": (klcat_full, mirt, dyn_c, c_mat, updater, router, True, True, True, True,  True),
    }
    # E2–E5 all use the same router but vary intervention depth via router config
    # (For simplicity in simulation, we treat E2–E5 as router=True with full P7)
    systems["E3_ConceptFeedback"]   = systems["E2_GenericFeedback"]
    systems["E4_MiscFeedback"]      = systems["E2_GenericFeedback"]
    systems["E5_MiscPlusPrerFeedback"] = systems["P7_Full_Neuronotes"]

    return systems


# ============================================================
# Run all stages
# ============================================================
def run_all_stages(n_learners: int = N_LEARNERS,
                   seed: int = MASTER_SEED) -> list[MetricBundle]:
    print(f"\n{'='*60}")
    print(" Neuronotes Experiment Runner")
    print(f" N_learners={n_learners}, seed={seed}, max_q={MAX_QUESTIONS}")
    print(f"{'='*60}\n")
    print("============================================================")

    items_path = DATA_DIR / "items_clean.csv"
    graph_path = DATA_DIR / "concept_graph.csv"

    # Generate frozen learner pool
    print("Generating learner pool ...")
    gen     = LearnerGenerator(n_learners=n_learners, seed=seed)
    learners = gen.generate()
    print("  >> {} learners generated".format(len(learners)))

    # Build all systems
    systems = make_systems(items_path, graph_path)
    bundles: list[MetricBundle] = []

    for sys_name, config in systems.items():
        (selector, mirt_, c_mod, c_mat_, updater_,
         router_, use_cmat, use_dynC, use_smd, use_rt, use_pre) = config

        print(f"\n[{sys_name}] running {n_learners} learners ...")
        selector.reset_exposure()

        results = []
        for learner in learners:
            res = run_one_learner(
                learner=learner,
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
            )
            results.append(res)

        bundle = build_metric_bundle(
            system_name=sys_name,
            theta_true_list=[r["theta_true"] for r in results],
            theta_final_list=[r["theta_final"] for r in results],
            theta_est_trajectories=[r["theta_trajectory"] for r in results],
            p_correct_all=[p for r in results for p in r["p_correct_log"]],
            correct_all=[c for r in results for c in r["correct_log"]],
            misc_y_true=[y for r in results for y in r["misc_y_true"]],
            misc_y_pred=[y for r in results for y in r["misc_y_pred"]],
            misc_y_score=[y for r in results for y in r["misc_y_score"]],
        )
        print(f"  RMSE={bundle.rmse:.4f}  F1={bundle.f1:.4f}  "
              f"EffQ={bundle.efficiency_q:.1f}")
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
