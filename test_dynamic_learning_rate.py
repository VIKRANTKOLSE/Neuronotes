"""Unit tests for compute_dynamic_learning_rate in SMD-VSNLMS.

Compatible with Python's built-in unittest framework and pytest.
"""

import math
import unittest
from neuronotes.smd_vsnlms import (
    compute_dynamic_learning_rate,
    BETA,
    ETA_MAX,
    ETA_MIN,
    RHO,
    DynamicLearningRateResult,
)


class TestDynamicLearningRate(unittest.TestCase):
    """Test suite demonstrating adaptive step size behavior."""

    def test_low_residual_yields_high_learning_rate(self):
        """Small prediction error and low variance should maintain high learning rate."""
        result = compute_dynamic_learning_rate(current_residual=0.05, previous_variance=0.01)

        self.assertIsInstance(result, DynamicLearningRateResult)
        self.assertLess(result.updated_variance, 0.02)
        # Denominator = 1 + 10 * 0.00925 = 1.0925 -> eta_s ≈ 0.4577
        self.assertGreater(result.current_learning_rate, 0.40)
        self.assertLessEqual(result.current_learning_rate, ETA_MAX)

    def test_high_residual_yields_low_learning_rate(self):
        """Large prediction error and high variance (wild guessing) should throttle learning rate."""
        result = compute_dynamic_learning_rate(current_residual=0.95, previous_variance=0.80)

        self.assertGreater(result.updated_variance, 0.80)
        # Denominator = 1 + 10 * 0.81025 = 9.1025 -> eta_s ≈ 0.0549
        self.assertLess(result.current_learning_rate, 0.10)
        self.assertGreaterEqual(result.current_learning_rate, ETA_MIN)

    def test_exact_analytical_values(self):
        """Verify exact formula values: v_t = 0.9*0.1 + 0.1*(0.4^2) = 0.106."""
        result = compute_dynamic_learning_rate(current_residual=0.4, previous_variance=0.1)

        expected_v = 0.9 * 0.1 + 0.1 * (0.4 ** 2)  # 0.106
        expected_eta = max(ETA_MIN, ETA_MAX / (1.0 + RHO * expected_v))  # 0.5 / 2.06 ≈ 0.242718
        self.assertTrue(math.isclose(result.updated_variance, expected_v, abs_tol=1e-6))
        self.assertTrue(math.isclose(result.current_learning_rate, expected_eta, abs_tol=1e-6))

    def test_clamping_to_eta_min(self):
        """Extreme variance should clamp strictly at ETA_MIN without underflowing."""
        result = compute_dynamic_learning_rate(current_residual=3.0, previous_variance=5.0)
        self.assertEqual(result.current_learning_rate, ETA_MIN)

    def test_pure_function_no_side_effects(self):
        """Function should be deterministic and produce identical results for identical inputs."""
        res1 = compute_dynamic_learning_rate(current_residual=0.3, previous_variance=0.2)
        res2 = compute_dynamic_learning_rate(current_residual=0.3, previous_variance=0.2)
        self.assertEqual(res1, res2)

    def test_invalid_inputs_raise_type_error(self):
        """Invalid floats, NaNs, and negative variances should raise TypeError."""
        with self.assertRaises(TypeError):
            compute_dynamic_learning_rate(current_residual=float("nan"), previous_variance=0.1)

        with self.assertRaises(TypeError):
            compute_dynamic_learning_rate(current_residual=0.1, previous_variance=-0.01)

        with self.assertRaises(TypeError):
            compute_dynamic_learning_rate(current_residual=float("inf"), previous_variance=0.1)


class TestTargetedMisconceptionDecay(unittest.TestCase):
    """Test suite for targeted misconception decay and adaptive decay rate."""

    def setUp(self):
        import numpy as np
        from neuronotes.smd_vsnlms import SMDVSNLMSUpdater, LearnerState
        self.np = np
        self.updater = SMDVSNLMSUpdater()
        self.state = LearnerState()
        self.state.misconception_probs = np.full(15, 0.40, dtype=float)

    def test_targeted_decay_only_decays_distractor_misconceptions(self):
        """Only dimensions marked active in z_mask should decay; others remain intact."""
        z_mask = self.np.zeros(15, dtype=float)
        z_mask[3] = 1.0  # Only misconception z_03 was in the distractors
        z_mask[7] = 1.0  # and z_07

        initial_probs = self.state.misconception_probs.copy()
        a_vec = self.np.zeros(58)
        a_vec[0] = 1.0

        updated_state = self.updater.update(
            state=self.state,
            item_id="item_01",
            concept="Effective Nuclear Charge",
            a_vec=a_vec,
            d_param=0.0,
            p_correct=0.8,
            correct=True,
            error_class="correct",
            misconception_tag="none",
            z_mask=z_mask,
            total_session_length=30,  # lambda_decay = min(0.05, 1.5/30) = 0.05
        )

        probs = updated_state.misconception_probs
        # Dimensions 3 and 7 should decay by 5%
        self.assertAlmostEqual(probs[3], 0.40 * (1.0 - 0.05), places=5)
        self.assertAlmostEqual(probs[7], 0.40 * (1.0 - 0.05), places=5)

        # All other 13 dimensions must remain completely untouched at 0.40
        for dim in range(15):
            if dim not in (3, 7):
                self.assertAlmostEqual(probs[dim], 0.40, places=5)

    def test_adaptive_test_length_decay_rate(self):
        """Decay rate should scale down for long sessions (min(0.05, 1.5/N))."""
        z_mask = self.np.ones(15, dtype=float)
        a_vec = self.np.zeros(58)
        a_vec[0] = 1.0

        # N = 200 -> lambda_decay = min(0.05, 1.5 / 200) = 0.0075
        self.state.misconception_probs = self.np.full(15, 0.50, dtype=float)
        updated_state = self.updater.update(
            state=self.state,
            item_id="item_02",
            concept="Effective Nuclear Charge",
            a_vec=a_vec,
            d_param=0.0,
            p_correct=0.8,
            correct=True,
            error_class="correct",
            misconception_tag="none",
            z_mask=z_mask,
            total_session_length=200,
        )

        expected_prob = 0.50 * (1.0 - 0.0075)
        self.assertTrue(self.np.allclose(updated_state.misconception_probs, expected_prob, atol=1e-5))

    def test_array_broadcasting_and_shape_safety(self):
        """Ensure 15D vector broadcasting works seamlessly with list, slice, or 1D array."""
        a_vec = self.np.zeros(58)
        a_vec[0] = 1.0

        # Pass a Python list instead of numpy array
        updated_state = self.updater.update(
            state=self.state,
            item_id="item_03",
            concept="Effective Nuclear Charge",
            a_vec=a_vec,
            d_param=0.0,
            p_correct=0.8,
            correct=True,
            error_class="correct",
            misconception_tag="none",
            z_mask=[1.0, 0.0] * 7 + [1.0],  # 15 elements
        )
        self.assertEqual(updated_state.misconception_probs.shape, (15,))


class TestSymmetricBMatrixAndReset(unittest.TestCase):
    """Test suite for updater reset, symmetric B matrix updates, and alpha decay."""

    def setUp(self):
        import numpy as np
        from neuronotes.smd_vsnlms import SMDVSNLMSUpdater, LearnerState
        self.np = np
        self.updater = SMDVSNLMSUpdater()
        self.state = LearnerState()

    def test_updater_reset_restores_initial_B(self):
        """updater.reset() should restore B to its initial prior."""
        initial_B = self.updater.B.copy()
        self.updater.B += 0.5  # mutate
        self.assertFalse(self.np.allclose(self.updater.B, initial_B))
        self.updater.reset()
        self.assertTrue(self.np.allclose(self.updater.B, initial_B))

    def test_symmetric_B_update_positive_on_correct(self):
        """On correct response, avoided distractors (z_mask) should update B with positive residual."""
        z_mask = self.np.zeros(15, dtype=float)
        z_mask[2] = 1.0
        a_vec = self.np.zeros(58)
        a_vec[0] = 1.0

        b_before = self.updater.B[0, 2]
        self.updater.update(
            state=self.state,
            item_id="item_sym_1",
            concept="Effective Nuclear Charge",
            a_vec=a_vec,
            d_param=0.0,
            p_correct=0.4,  # residual = 1 - 0.4 = +0.6 > 0
            correct=True,
            error_class="correct",
            misconception_tag="none",
            z_mask=z_mask,
        )
        b_after = self.updater.B[0, 2]
        # B[0, 2] should increase because residual * a_vec[0] * z_mask[2] > 0
        self.assertGreater(b_after, b_before)

    def test_symmetric_B_update_negative_on_incorrect(self):
        """On incorrect response, chosen distractor (z) should update B with negative residual."""
        z = self.np.zeros(15, dtype=float)
        z[4] = 1.0
        a_vec = self.np.zeros(58)
        a_vec[0] = 1.0

        b_before = self.updater.B[0, 4]
        self.updater.update(
            state=self.state,
            item_id="item_sym_2",
            concept="Effective Nuclear Charge",
            a_vec=a_vec,
            d_param=0.0,
            p_correct=0.7,  # residual = 0 - 0.7 = -0.7 < 0
            correct=False,
            error_class="conceptual_error",
            misconception_tag="z_04",
            z_vector=z,
        )
        b_after = self.updater.B[0, 4]
        # B[0, 4] should decrease because residual * a_vec[0] * z[4] < 0
        self.assertLess(b_after, b_before)


if __name__ == "__main__":
    unittest.main()
