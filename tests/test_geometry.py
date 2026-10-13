"""
Automated unit tests for geometric calculations in FitnessVision.
"""

import unittest
import math
from posture_analysis.geometry import (
    calculate_angle_3points,
    calculate_euclidean_distance,
    calculate_normalized_distance,
    calculate_displacement,
    calculate_torso_inclination,
    calculate_symmetry,
    calculate_midpoint,
)


class TestGeometry(unittest.TestCase):
    def test_right_angle_90_degrees(self):
        pt_a = {"x": 0.0, "y": 1.0}
        pt_b = {"x": 0.0, "y": 0.0}
        pt_c = {"x": 1.0, "y": 0.0}
        angle = calculate_angle_3points(pt_a, pt_b, pt_c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_straight_line_180_degrees(self):
        pt_a = {"x": -1.0, "y": 0.0}
        pt_b = {"x": 0.0, "y": 0.0}
        pt_c = {"x": 1.0, "y": 0.0}
        angle = calculate_angle_3points(pt_a, pt_b, pt_c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_folded_line_0_degrees(self):
        pt_a = {"x": 1.0, "y": 0.0}
        pt_b = {"x": 0.0, "y": 0.0}
        pt_c = {"x": 1.0, "y": 0.0}
        angle = calculate_angle_3points(pt_a, pt_b, pt_c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 0.0, places=1)

    def test_diagonal_45_degrees(self):
        pt_a = {"x": 0.0, "y": 1.0}
        pt_b = {"x": 0.0, "y": 0.0}
        pt_c = {"x": 1.0, "y": 1.0}
        angle = calculate_angle_3points(pt_a, pt_b, pt_c)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_zero_length_vector_returns_none(self):
        # A and B are identical
        pt_a = {"x": 0.5, "y": 0.5}
        pt_b = {"x": 0.5, "y": 0.5}
        pt_c = {"x": 1.0, "y": 0.5}
        self.assertIsNone(calculate_angle_3points(pt_a, pt_b, pt_c))

    def test_invalid_and_missing_coordinates(self):
        # Missing point
        self.assertIsNone(calculate_angle_3points(None, {"x": 0, "y": 0}, {"x": 1, "y": 0}))
        # Incomplete dict
        self.assertIsNone(calculate_angle_3points({"x": 1}, {"x": 0, "y": 0}, {"x": 1, "y": 0}))
        # NaN or Inf values
        self.assertIsNone(calculate_angle_3points({"x": float("nan"), "y": 1}, {"x": 0, "y": 0}, {"x": 1, "y": 0}))
        self.assertIsNone(calculate_angle_3points({"x": float("inf"), "y": 1}, {"x": 0, "y": 0}, {"x": 1, "y": 0}))

    def test_tuple_and_list_input_support(self):
        angle = calculate_angle_3points((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_euclidean_distance(self):
        p1 = {"x": 0.0, "y": 0.0}
        p2 = {"x": 3.0, "y": 4.0}
        dist = calculate_euclidean_distance(p1, p2)
        self.assertEqual(dist, 5.0)

    def test_normalized_distance(self):
        p1 = {"x": 0.0, "y": 0.0}
        p2 = {"x": 2.0, "y": 0.0}
        norm_dist = calculate_normalized_distance(p1, p2, reference_dist=4.0)
        self.assertEqual(norm_dist, 0.5)
        # Invalid reference length
        self.assertIsNone(calculate_normalized_distance(p1, p2, reference_dist=0.0))

    def test_torso_inclination(self):
        # Perfectly vertical torso (shoulder directly above hip)
        sh = {"x": 0.5, "y": 0.2}
        hp = {"x": 0.5, "y": 0.6}
        inc = calculate_torso_inclination(sh, hp)
        self.assertEqual(inc, 0.0)

        # 45 degree inclined torso
        sh_45 = {"x": 0.2, "y": 0.2}
        hp_45 = {"x": 0.6, "y": 0.6}
        inc_45 = calculate_torso_inclination(sh_45, hp_45)
        self.assertAlmostEqual(inc_45, 45.0, places=1)

    def test_symmetry_calculation(self):
        symm = calculate_symmetry(90.0, 100.0)
        self.assertIsNotNone(symm)
        self.assertEqual(symm["absolute_diff"], 10.0)

        # None input
        self.assertIsNone(calculate_symmetry(None, 100.0))

    def test_midpoint_calculation(self):
        p1 = {"x": 0.0, "y": 0.0}
        p2 = {"x": 1.0, "y": 2.0}
        mid = calculate_midpoint(p1, p2)
        self.assertEqual(mid["x"], 0.5)
        self.assertEqual(mid["y"], 1.0)


if __name__ == "__main__":
    unittest.main()
