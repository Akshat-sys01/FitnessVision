"""
Automated unit tests for exercise posture analysis and error detection in FitnessVision.
Uses deterministic synthetic keypoint sets.
"""

import unittest
from posture_analysis.squat_analysis import SquatAnalyzer
from posture_analysis.curl_analysis import BicepCurlAnalyzer
from posture_analysis.feedback import generate_feedback
from posture_analysis.error_detection import get_error


def make_synthetic_squat_pose(knee_angle_type="standing", forward_lean=False, knee_valgus=False):
    """Generates synthetic 2D keypoints for a human body performing a squat."""
    sh_x_offset = 0.35 if forward_lean else 0.0

    return {
        "left_shoulder": {"x": 0.45 + sh_x_offset, "y": 0.20, "visibility": 0.95},
        "right_shoulder": {"x": 0.55 + sh_x_offset, "y": 0.20, "visibility": 0.95},
        "left_hip": {"x": 0.45, "y": 0.50, "visibility": 0.95},
        "right_hip": {"x": 0.55, "y": 0.50, "visibility": 0.95},
        # Knees
        "left_knee": {"x": 0.49 if knee_valgus else 0.44, "y": 0.70, "visibility": 0.95},
        "right_knee": {"x": 0.51 if knee_valgus else 0.56, "y": 0.70, "visibility": 0.95},
        # Ankles
        "left_ankle": {"x": 0.40, "y": 0.90, "visibility": 0.95},
        "right_ankle": {"x": 0.60, "y": 0.90, "visibility": 0.95},
    }


def make_synthetic_curl_pose(elbow_flare=False, torso_sway=False):
    """Generates synthetic 2D keypoints for bicep curls."""
    sh_x_offset = 0.20 if torso_sway else 0.0
    el_x_offset = 0.20 if elbow_flare else 0.0

    return {
        "left_shoulder": {"x": 0.45 + sh_x_offset, "y": 0.25, "visibility": 0.95},
        "right_shoulder": {"x": 0.55 + sh_x_offset, "y": 0.25, "visibility": 0.95},
        "left_elbow": {"x": 0.45 + el_x_offset, "y": 0.45, "visibility": 0.95},
        "right_elbow": {"x": 0.55 + el_x_offset, "y": 0.45, "visibility": 0.95},
        "left_wrist": {"x": 0.45, "y": 0.35, "visibility": 0.95},
        "right_wrist": {"x": 0.55, "y": 0.35, "visibility": 0.95},
        "left_hip": {"x": 0.45, "y": 0.65, "visibility": 0.95},
        "right_hip": {"x": 0.55, "y": 0.65, "visibility": 0.95},
    }


class TestPostureAnalysis(unittest.TestCase):
    def test_squat_good_form(self):
        analyzer = SquatAnalyzer()
        pose = make_synthetic_squat_pose(forward_lean=False, knee_valgus=False)
        result = analyzer.analyze_frame(pose)

        self.assertTrue(result["tracking_valid"])
        self.assertEqual(len(result["missing_joints"]), 0)
        self.assertIsNotNone(result["primary_knee_angle"])
        # No forward lean or valgus errors
        error_codes = [e.code for e in result["detected_errors"]]
        self.assertNotIn("EXCESSIVE_FORWARD_LEAN", error_codes)
        self.assertNotIn("KNEE_VALGUS", error_codes)

    def test_squat_excessive_forward_lean(self):
        analyzer = SquatAnalyzer(max_torso_inclination=40.0)
        pose = make_synthetic_squat_pose(forward_lean=True)
        result = analyzer.analyze_frame(pose)

        self.assertTrue(result["tracking_valid"])
        error_codes = [e.code for e in result["detected_errors"]]
        self.assertIn("EXCESSIVE_FORWARD_LEAN", error_codes)

    def test_squat_missing_keypoints_rejects_as_invalid(self):
        analyzer = SquatAnalyzer()
        pose = make_synthetic_squat_pose()
        # Remove knees
        del pose["left_knee"]
        del pose["right_knee"]

        result = analyzer.analyze_frame(pose)
        self.assertFalse(result["tracking_valid"])
        self.assertIn("left_knee", result["missing_joints"])
        self.assertIn("right_knee", result["missing_joints"])

    def test_bicep_curl_elbow_drift_detection(self):
        analyzer = BicepCurlAnalyzer()
        pose = make_synthetic_curl_pose(elbow_flare=True)
        result = analyzer.analyze_frame(pose)

        self.assertTrue(result["tracking_valid"])
        error_codes = [e.code for e in result["detected_errors"]]
        self.assertIn("EXCESSIVE_ELBOW_DRIFT", error_codes)

    def test_feedback_generation_priority(self):
        # 1. Invalid tracking takes highest priority
        fb_invalid = generate_feedback(
            tracking_valid=False,
            missing_joints=["left_knee", "right_knee"],
            detected_errors=[],
            phase="standing",
        )
        self.assertEqual(fb_invalid["status"], "warning")
        self.assertIn("Adjust camera position", fb_invalid["feedback"])

        # 2. Form error takes precedence over normal coaching cue
        err = get_error("EXCESSIVE_FORWARD_LEAN")
        fb_error = generate_feedback(
            tracking_valid=True,
            missing_joints=[],
            detected_errors=[err],
            phase="descending",
        )
        self.assertEqual(fb_error["status"], "incorrect")
        self.assertEqual(fb_error["feedback"], err.message)

        # 3. Clean form yields encouraging status
        fb_clean = generate_feedback(
            tracking_valid=True,
            missing_joints=[],
            detected_errors=[],
            phase="descending",
        )
        self.assertEqual(fb_clean["status"], "correct")
        self.assertIn("Control the descent", fb_clean["feedback"])


if __name__ == "__main__":
    unittest.main()
