"""Integration test: multi-session learner state persistence and warm-start."""

import shutil
import tempfile
from pathlib import Path
import numpy as np

from simulation.ground_truth import GroundTruthGenerator, simulate_response_from_ground_truth
from simulation.run_experiment import make_system, run_one_learner, DATA_DIR
from neuronotes.learner_store import LearnerStore


def test_multi_session_warm_start():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_sessions.db"

    try:
        items_path = DATA_DIR / "items_clean.csv"
        graph_path = DATA_DIR / "concept_graph.csv"

        # Generate ground truth
        gt_gen = GroundTruthGenerator(n_learners=5, seed=123, items_path=items_path)
        gt = gt_gen.generate()
        learner_gt = gt.learners[0]

        # Session 1: Run learner with persistence enabled
        config1 = make_system("P7_Full_Neuronotes", items_path, graph_path)
        (sel1, mirt1, dyn_c1, c_mat1, updater1, router1,
         use_cmat, use_dync, use_smd, use_rt, use_pre) = config1

        store = LearnerStore(db_path)

        res_sess1 = run_one_learner(
            learner_gt=learner_gt,
            ground_truth=gt,
            selector=sel1,
            mirt=mirt1,
            c_module=dyn_c1,
            c_mat=c_mat1,
            updater=updater1,
            router=router1,
            use_c_matrix=use_cmat,
            use_dynamic_c=use_dync,
            use_smd=use_smd,
            use_router=use_rt,
            use_prereq=use_pre,
            max_questions=15,
            learner_store=store,
            persist_sessions=True,
            session_id="session_1",
        )

        # Verify session 1 recorded
        rec = store.get_learner(learner_gt.learner_id)
        assert rec is not None
        assert rec.session_count == 1
        assert rec.total_questions_answered == 15
        assert np.allclose(rec.theta, res_sess1["theta_final"])

        seen_sess1 = store.get_seen_items(learner_gt.learner_id)
        assert len(seen_sess1) == 15

        # Session 2: Run same learner with warm-start
        config2 = make_system("P7_Full_Neuronotes", items_path, graph_path)
        (sel2, mirt2, dyn_c2, c_mat2, updater2, router2,
         _, _, _, _, _) = config2

        res_sess2 = run_one_learner(
            learner_gt=learner_gt,
            ground_truth=gt,
            selector=sel2,
            mirt=mirt2,
            c_module=dyn_c2,
            c_mat=c_mat2,
            updater=updater2,
            router=router2,
            use_c_matrix=use_cmat,
            use_dynamic_c=use_dync,
            use_smd=use_smd,
            use_router=use_rt,
            use_prereq=use_pre,
            max_questions=15,
            learner_store=store,
            persist_sessions=True,
            session_id="session_2",
        )

        # Check session count updated to 2
        rec2 = store.get_learner(learner_gt.learner_id)
        assert rec2.session_count == 2
        assert rec2.total_questions_answered == 30

        # Check warm-start trajectory start vs cold-start
        theta_start_sess2 = res_sess2["theta_trajectory"][0]
        # In session 2, initial theta should match session 1 final theta (decay ~ 0 over 0 seconds)
        assert np.allclose(theta_start_sess2, res_sess1["theta_final"], atol=1e-3)

        # Verify questions asked in session 2 did not repeat the 15 questions from session 1
        seen_sess2 = store.get_seen_items(learner_gt.learner_id)
        assert len(seen_sess2) == 30  # 15 distinct from s1 + 15 distinct from s2

        store.close()
        print("PASS: test_multi_session_warm_start verified successfully.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_stateless_default():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "should_not_exist.db"

    try:
        items_path = DATA_DIR / "items_clean.csv"
        graph_path = DATA_DIR / "concept_graph.csv"

        gt_gen = GroundTruthGenerator(n_learners=2, seed=42, items_path=items_path)
        gt = gt_gen.generate()
        learner_gt = gt.learners[0]

        config = make_system("P7_Full_Neuronotes", items_path, graph_path)
        (sel, mirt, dyn_c, c_mat, updater, router,
         use_cmat, use_dync, use_smd, use_rt, use_pre) = config

        # Run without persist_sessions (default)
        res = run_one_learner(
            learner_gt=learner_gt,
            ground_truth=gt,
            selector=sel,
            mirt=mirt,
            c_module=dyn_c,
            c_mat=c_mat,
            updater=updater,
            router=router,
            use_c_matrix=use_cmat,
            use_dynamic_c=use_dync,
            use_smd=use_smd,
            use_router=use_rt,
            use_prereq=use_pre,
            max_questions=5,
            persist_sessions=False,
        )

        assert not db_path.exists()
        assert np.allclose(res["theta_trajectory"][0], learner_gt.theta_init)
        print("PASS: test_stateless_default verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_multi_session_warm_start()
    test_stateless_default()
    print("All integration tests passed!")
