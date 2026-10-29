"""CPU checks for auxiliary response transfer; no model or dataset required."""

import math
import unittest

import numpy as np

from readprune.spatial import map_letterbox_features, retain_budget


def scalar_reference(values, original_wh, target_hw, source_hw):
    """Independent four-neighbor interpolation in original-image coordinates."""
    ow, oh = original_wh
    side = max(ow, oh)
    left, top = (side - ow) // 2, (side - oh) // 2
    sh, sw = source_hw
    th, tw = target_hw
    field = np.asarray(values, dtype=np.float64).reshape(sh, sw, -1)
    result = []
    for row in range(th):
        for column in range(tw):
            image_x = (column + 0.5) * ow / tw
            image_y = (row + 0.5) * oh / th
            x = min(max((left + image_x) / side * sw - 0.5, 0), sw - 1)
            y = min(max((top + image_y) / side * sh - 0.5, 0), sh - 1)
            ix, iy = math.floor(x), math.floor(y)
            jx, jy = min(ix + 1, sw - 1), min(iy + 1, sh - 1)
            wx, wy = x - ix, y - iy
            result.append((1 - wy) * ((1 - wx) * field[iy, ix] + wx * field[iy, jx])
                          + wy * ((1 - wx) * field[jy, ix] + wx * field[jy, jx]))
    return np.asarray(result)


class SpatialTests(unittest.TestCase):
    def test_identity_and_original_inputs_unchanged(self):
        field = np.random.default_rng(41).normal(size=(576, 8))
        original = field.copy()
        actual, metadata = map_letterbox_features(field, (336, 336), (24, 24))
        np.testing.assert_array_equal(actual, field)
        np.testing.assert_array_equal(field, original)
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(metadata["letterbox_offset_xy"], [0, 0])

    def test_random_fields_match_independent_scalar_reference(self):
        rng = np.random.default_rng(719)
        for original_wh, target_hw, source_hw in (
            ((137, 81), (4, 17), (24, 24)),
            ((81, 137), (17, 4), (24, 24)),
            ((101, 80), (4, 5), (24, 24)),
            ((80, 101), (5, 4), (24, 24)),
            ((51, 51), (12, 7), (5, 9)),
            ((1, 100), (1, 8), (2, 3)),
        ):
            field = rng.normal(size=(math.prod(source_hw), 3))
            actual, _ = map_letterbox_features(field, original_wh, target_hw, source_hw)
            expected = scalar_reference(field, original_wh, target_hw, source_hw)
            np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)

    def test_rectangular_coordinates_and_integer_padding(self):
        yy, xx = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
        field = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        actual, metadata = map_letterbox_features(field, (240, 120), (6, 12))
        expected_x = (np.arange(12) + 0.5) * 2 - 0.5
        expected_y = 6 + (np.arange(6) + 0.5) * 2 - 0.5
        np.testing.assert_allclose(actual[:, 0].reshape(6, 12), np.broadcast_to(expected_x, (6, 12)))
        np.testing.assert_allclose(actual[:, 1].reshape(6, 12), np.broadcast_to(expected_y[:, None], (6, 12)))
        self.assertEqual(metadata["letterbox_offset_xy"], [0, 60])
        _, odd = map_letterbox_features(field, (101, 80), (4, 5))
        self.assertEqual(odd["letterbox_offset_xy"], [0, 10])

    def test_constant_signed_field_and_replicated_border(self):
        values = np.tile([-4., 0., 7.], (576, 1))
        for wh, target in (((137, 81), (4, 17)), ((81, 137), (17, 4))):
            actual, _ = map_letterbox_features(values, wh, target)
            np.testing.assert_allclose(actual, np.tile([-4., 0., 7.], (math.prod(target), 1)), atol=1e-15)
        field = np.array([[1.], [2.], [3.], [4.]])
        actual, _ = map_letterbox_features(field, (100, 100), (4, 4), (2, 2))
        np.testing.assert_array_equal(actual.reshape(4, 4)[np.ix_([0, 3], [0, 3])], [[1., 2.], [3., 4.]])

    def test_interpolation_precedes_absolute_sensitivity(self):
        field = np.array([[-2.], [2.], [2.], [-2.]])
        signed, _ = map_letterbox_features(field, (100, 100), (1, 1), (2, 2))
        unsigned, _ = map_letterbox_features(np.abs(field), (100, 100), (1, 1), (2, 2))
        np.testing.assert_array_equal(signed, [[0.]])
        np.testing.assert_array_equal(unsigned, [[2.]])

    def test_invalid_inputs_are_rejected(self):
        valid = np.ones((576, 2))
        for grid in ((0, 3), (3.5, 2), (True, 3), (2, 3, 4)):
            with self.assertRaises(ValueError):
                map_letterbox_features(valid, (336, 336), grid)
        for invalid in (np.ones(576), np.ones((575, 2)), np.full((576, 2), np.nan)):
            with self.assertRaises(ValueError):
                map_letterbox_features(invalid, (336, 336), (4, 4))

    def test_budget_uses_floor_and_one_token_minimum(self):
        for n, ratio, expected in ((576, .2, 115), (576, .1, 57), (7, .1, 1), (20, 1., 20)):
            self.assertEqual(retain_budget(n, ratio), expected)
        self.assertEqual(retain_budget(np.int64(17), .2), 3)
        for invalid in (0, -1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                retain_budget(100, invalid)
        for invalid in (0, -2, True, 10.0):
            with self.assertRaises(ValueError):
                retain_budget(invalid, .1)


if __name__ == "__main__":
    unittest.main()
