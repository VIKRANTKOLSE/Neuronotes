"""Unit tests for LearnerStore SQLite persistence and warm-start decay."""

import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pytest

from neuronotes.learner_store import LearnerStore


def test_learner_store_save_and_load():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_learners.db"
    try:
        with LearnerStore(db_path) as store:
            theta = np.array([0.5, -0.2, 1.1, 0.0])
            concept_theta = {"calc_limits": 0.5, "derivative": -0.2}
            misc_probs = np.array([0.1, 0.8, 0.05])
            questions = [
                {"item_id": "q1", "concept": "calc_limits", "correct": True},
                {"item_id": "q2", "concept": "derivative", "correct": False},
            ]
            t_start = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
            t_end = datetime(2026, 10, 1, 10, 20, tzinfo=timezone.utc)

            store.save_learner_session(
                learner_id="student_42",
                session_id="sess_001",
                theta=theta,
                concept_theta=concept_theta,
                misconception_probs=misc_probs,
                questions_record=questions,
                session_start=t_start,
                session_end=t_end,
            )

            # Retrieve
            record = store.get_learner("student_42")
            assert record is not None
            assert record.learner_id == "student_42"
            assert np.allclose(record.theta, theta)
            assert record.concept_theta == concept_theta
            assert np.allclose(record.misconception_probs, misc_probs)
            assert record.session_count == 1
            assert record.total_questions_answered == 2
            assert record.dim == 4

            # Seen items
            seen = store.get_seen_items("student_42")
            assert seen == {"q1", "q2"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_learner_store_time_decay_warm_start():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_decay.db"
    try:
        with LearnerStore(db_path) as store:
            theta_learned = np.array([2.0, 2.0, 2.0])
            theta_prior = np.array([0.0, 0.0, 0.0])
            t_end = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

            store.save_learner_session(
                learner_id="student_decay",
                session_id="s1",
                theta=theta_learned,
                concept_theta={},
                misconception_probs=np.array([]),
                questions_record=[],
                session_start=t_end,
                session_end=t_end,
            )

            # Case 1: delta_t = 0 (same day immediately)
            warm_immediate = store.get_warm_start_theta(
                "student_decay", theta_prior, lambda_decay=0.1, current_time=t_end
            )
            assert np.allclose(warm_immediate, theta_learned)

            # Case 2: delta_t = 10 days
            t_future = t_end + timedelta(days=10)
            decay_factor = np.exp(-0.1 * 10) # e^-1 ~= 0.367879
            expected = theta_prior + (theta_learned - theta_prior) * decay_factor
            warm_10d = store.get_warm_start_theta(
                "student_decay", theta_prior, lambda_decay=0.1, current_time=t_future
            )
            assert np.allclose(warm_10d, expected, atol=1e-5)

            # Case 3: Dimension mismatch fallback
            theta_wrong_dim = np.zeros(5)
            warm_wrong = store.get_warm_start_theta(
                "student_decay", theta_wrong_dim, lambda_decay=0.1, current_time=t_future
            )
            assert warm_wrong is None

            # Case 4: Non-existent student fallback
            assert store.get_warm_start_theta("ghost", theta_prior) is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_learner_store_deletion():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_delete.db"
    try:
        with LearnerStore(db_path) as store:
            store.save_learner_session(
                learner_id="student_del1",
                session_id="s1",
                theta=np.ones(4),
                concept_theta={},
                misconception_probs=np.zeros(3),
                questions_record=[{"item_id": "q1", "correct": True}],
                session_start=datetime.now(timezone.utc),
                session_end=datetime.now(timezone.utc),
            )
            store.save_learner_session(
                learner_id="student_del2",
                session_id="s2",
                theta=np.ones(4) * 2,
                concept_theta={},
                misconception_probs=np.zeros(3),
                questions_record=[{"item_id": "q2", "correct": False}],
                session_start=datetime.now(timezone.utc),
                session_end=datetime.now(timezone.utc),
            )

            assert len(store.list_learners()) == 2

            # Delete del1
            assert store.delete_learner("student_del1") is True
            assert store.get_learner("student_del1") is None
            assert len(store.get_seen_items("student_del1")) == 0
            assert len(store.list_learners()) == 1

            # Delete non-existent
            assert store.delete_learner("fake_student") is False

            # Clear all
            count = store.clear_all()
            assert count == 1
            assert len(store.list_learners()) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_learner_store_save_and_load()
    test_learner_store_time_decay_warm_start()
    test_learner_store_deletion()
    print("All LearnerStore tests passed!")
