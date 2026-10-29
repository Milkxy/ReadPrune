"""Model-free tests against direct determinant calculations."""

import unittest

import numpy as np

from readprune import select_tokens
from readprune import _math


def direct_objective(B, q, support):
    if not support:
        return 0.0
    sign, logdet = np.linalg.slogdet(B[np.ix_(support, support)])
    if sign <= 0:
        raise AssertionError("The reference kernel must be positive definite")
    return 4.0 * np.log(q[support]).sum() + logdet


def direct_greedy(B, q, k):
    order = []
    for _ in range(k):
        remaining = [i for i in range(len(B)) if i not in order]
        values = [(direct_objective(B, q, order + [i]), i) for i in remaining]
        best = max(value for value, _ in values)
        chosen = min(i for value, i in values if best - value <= _math.TIE_ATOL)
        order.append(chosen)
    return order


class CoreTests(unittest.TestCase):
    def test_quality_deadzone_sign_and_flat_inputs(self):
        flat, info = _math.quality(np.array([0, 1e-6, -1e-6, 0.2e-6]))
        np.testing.assert_array_equal(flat, np.ones(4))
        self.assertTrue(info["flat"])
        delta = np.array([0, 1e-6, 3e-6, -5e-6])
        s = np.maximum(np.abs(delta) - 1e-6, 0)
        expected = 1 + s / (s + s.mean())
        q, _ = _math.quality(delta)
        np.testing.assert_array_equal(q, _math.quality(-delta)[0])
        np.testing.assert_allclose(q, expected, atol=0, rtol=0)

    def test_greedy_matches_direct_q_squared_objective(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            features = rng.normal(size=(11, 17)).astype(np.float32)
            delta = rng.normal(size=11)
            result = select_tokens(features, delta, 6)
            B = _math.geometry(features)
            expected_order = direct_greedy(B, result["quality"], 6)
            self.assertEqual(result["diagnostics"]["order"], expected_order)
            self.assertEqual(result["indices"].tolist(), sorted(expected_order))
            expected = direct_objective(B, result["quality"], sorted(expected_order))
            self.assertAlmostEqual(result["diagnostics"]["objective"], expected, places=11)

    def test_preserves_original_vectors_dtype_and_inputs(self):
        rng = np.random.default_rng(78)
        features = rng.normal(size=(20, 8)).astype(np.float16)
        delta = rng.normal(size=20)
        before, delta_before = features.copy(), delta.copy()
        result = select_tokens(features, delta, 7)
        np.testing.assert_array_equal(result["tokens"], features[result["indices"]])
        np.testing.assert_array_equal(features, before)
        np.testing.assert_array_equal(delta, delta_before)
        self.assertEqual(result["tokens"].dtype, features.dtype)
        self.assertFalse(result["diagnostics"]["repair_applied"])

    def test_geometry_uses_signed_cosine_and_fixed_ridge(self):
        features = np.array([[1., 0.], [-1., 0.], [0., 2.]])
        expected = np.array([[1., -1., 0.], [-1., 1., 0.], [0., 0., 1.]])
        expected += 1e-6 * np.eye(3)
        np.testing.assert_array_equal(_math.geometry(features), expected)

    def test_budget_ties_and_full_budget(self):
        features = np.eye(8, dtype=np.float32)
        result = select_tokens(features, np.zeros(8), 3)
        self.assertEqual(result["indices"].tolist(), [0, 1, 2])
        full = select_tokens(features, np.zeros(8), 8)
        np.testing.assert_array_equal(full["tokens"], features)

    def test_invalid_inputs_fail_without_silent_repair(self):
        features = np.eye(5)
        for k in (0, -1, 6, 2.5, True):
            with self.assertRaises(ValueError):
                select_tokens(features, np.ones(5), k)
        for delta in (np.ones(4), np.array([1., 1., np.nan, 1., 1.])):
            with self.assertRaises(ValueError):
                select_tokens(features, delta, 2)
        with self.assertRaises(ValueError):
            select_tokens(np.zeros((5, 3)), np.ones(5), 2)


if __name__ == "__main__":
    unittest.main()
