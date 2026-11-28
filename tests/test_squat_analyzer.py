"""
Comprehensive unit and integration tests for the Research-Based Squat Posture Analyzer.

Covers:
1. Mathematical functions (angles, dot products, distances, aspect scaling, foot angles)
2. Landmark reliability validation (visibility, presence, NaN/Inf, missing ankle, jump spikes)
3. Movement & depth analysis (standing, descent, half squat, deep squat, ascent, micro-bounce)
4. Repetition counting state machine (valid rep, incomplete rep, bottom dwell, noise rejection)
5. Independent posture rules & boundary thresholds (Kwon & Kim 2022, Ahn & Kim 2023, FitnessVision)
6. Camera-view separation (side vs front) and feedback engine stabilization
7. Django API integration and OpenCV debug visualization
"""

import json
import math
import unittest
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse

from posture_analysis.geometry import (
    calculate_angle_3points,
    calculate_vector_dot_product,
    calculate_vector_angle,
    calculate_euclidean_distance,
    calculate_vertical_difference,
    calculate_horizontal_difference,
    calculate_foot_orientation_angle,
    scale_point_for_aspect,
)
from posture_analysis.squat import (
    SquatConfig,
    LandmarkValidator,
    SquatFeatureExtractor,
    SquatDepthAnalyzer,
    StanceAnalyzer,
    FootAnalyzer,
    RuleStatus,
    OverallSquatStatus,
)
from posture_analysis.squat_analysis import SquatAnalyzer
from exercises.repetition_counter import SquatRepetitionCounter
from pose_estimation.visualization import draw_squat_debug_overlay


def build_pose_from_angles(
    knee_angle_deg: float,
    hip_angle_deg: float = 170.0,
    side: str = "both",
    hip_y: float = 0.50,
    thigh_len: float = 0.20,
    shin_len: float = 0.20,
    torso_len: float = 0.25,
    foot_dx: float = 0.04,
    visibility: float = 0.95,
    stance_half_width: float = 0.10,
    shoulder_half_width: float = 0.10,
    knee_half_width: float = 0.10,
):
    """
    Constructs mathematically exact normalized 2D coordinates for a squat pose
    with controllable knee angle, hip angle, hip-knee vertical offset, and knee-foot alignment.
    """
    # Place knee at fixed horizontal reference; place ankle vertically below knee (shin along +y)
    # Then place hip at angle `knee_angle_deg` from ankle-to-knee vector.
    # For a side-view coordinate builder:
    # Ankle = (0.50, hip_y + thigh_len*cos?): simpler: fix knee at (0.52, 0.70), ankle at (0.52, 0.90)
    # Wait: vector knee->ankle = (0, +shin_len).
    # Vector knee->hip rotated by `knee_angle_deg` toward -x (hips back):
    # theta_k = radians(knee_angle_deg)
    # unit(knee->ankle) = (0, 1). Rotating by +theta_k clockwise in image coords (x left, y up):
    # hip_x = knee_x - thigh_len * sin(theta_k), hip_y_calc = knee_y + thigh_len * cos(theta_k)
    # Check: when theta_k = 180 deg, cos(180)=-1, sin(180)=0 -> hip = (knee_x, knee_y - thigh_len) directly above knee!
    # When theta_k = 90 deg, cos(90)=0, sin(90)=1 -> hip = (knee_x - thigh_len, knee_y) horizontal thigh!
    theta_k = math.radians(knee_angle_deg)
    knee_x = 0.52
    knee_y = 0.70
    ankle_x = knee_x
    ankle_y = knee_y + shin_len

    hip_x = knee_x - thigh_len * math.sin(theta_k)
    calc_hip_y = knee_y + thigh_len * math.cos(theta_k)
    if hip_y is not None:
        # Allow explicit override only when testing vertical difference separately
        pass

    # Vector hip->knee = (knee_x - hip_x, knee_y - calc_hip_y)
    hk_dx = knee_x - hip_x
    hk_dy = knee_y - calc_hip_y
    hk_angle = math.atan2(hk_dy, hk_dx)
    # Rotate hip->knee counter-clockwise by hip_angle_deg to place shoulder above hip
    theta_h = math.radians(hip_angle_deg)
    hs_angle = hk_angle - theta_h
    sh_x = hip_x + torso_len * math.cos(hs_angle)
    sh_y = calc_hip_y + torso_len * math.sin(hs_angle)

    pose = {
        "left_shoulder": {"x": sh_x - shoulder_half_width if side == "front" else sh_x, "y": sh_y, "visibility": visibility},
        "right_shoulder": {"x": sh_x + shoulder_half_width if side == "front" else sh_x, "y": sh_y, "visibility": visibility},
        "left_hip": {"x": hip_x - 0.08 if side == "front" else hip_x, "y": calc_hip_y, "visibility": visibility},
        "right_hip": {"x": hip_x + 0.08 if side == "front" else hip_x, "y": calc_hip_y, "visibility": visibility},
        "left_knee": {"x": knee_x - knee_half_width if side == "front" else knee_x, "y": knee_y, "visibility": visibility},
        "right_knee": {"x": knee_x + knee_half_width if side == "front" else knee_x, "y": knee_y, "visibility": visibility},
        "left_ankle": {"x": ankle_x - stance_half_width if side == "front" else ankle_x, "y": ankle_y, "visibility": visibility},
        "right_ankle": {"x": ankle_x + stance_half_width if side == "front" else ankle_x, "y": ankle_y, "visibility": visibility},
        "left_heel": {"x": (ankle_x - stance_half_width) if side == "front" else ankle_x - 0.01, "y": ankle_y + 0.02, "visibility": visibility},
        "right_heel": {"x": (ankle_x + stance_half_width) if side == "front" else ankle_x - 0.01, "y": ankle_y + 0.02, "visibility": visibility},
        "left_foot_index": {"x": (ankle_x - stance_half_width - 0.02) if side == "front" else ankle_x + foot_dx, "y": ankle_y + 0.05, "visibility": visibility},
        "right_foot_index": {"x": (ankle_x + stance_half_width + 0.02) if side == "front" else ankle_x + foot_dx, "y": ankle_y + 0.05, "visibility": visibility},
    }
    return pose


class TestSquatMath(unittest.TestCase):
    def test_vector_dot_product_and_angle(self):
        self.assertEqual(calculate_vector_dot_product((3.0, 4.0), (2.0, -1.0)), 2.0)
        self.assertAlmostEqual(calculate_vector_angle((0.0, 1.0), (1.0, 0.0)), 90.0, places=2)
        self.assertAlmostEqual(calculate_vector_angle((1.0, 0.0), (-1.0, 0.0)), 180.0, places=2)
        self.assertIsNone(calculate_vector_angle((0.0, 0.0), (1.0, 0.0)))

    def test_aspect_ratio_correction(self):
        pt = {"x": 0.5, "y": 0.5}
        scaled = scale_point_for_aspect(pt, image_width=1280, image_height=720)
        self.assertAlmostEqual(scaled[0], 0.5 * (1280.0 / 720.0), places=4)
        self.assertAlmostEqual(scaled[1], 0.5, places=4)

        # Aspect-aware distance on 16:9 image differs from raw normalized x distance
        d_raw = calculate_euclidean_distance({"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0})
        d_aspect = calculate_euclidean_distance(
            {"x": 0.0, "y": 0.0}, {"x": 0.9, "y": 0.0}, image_width=1280, image_height=720
        )
        self.assertAlmostEqual(d_raw, 0.9, places=4)
        self.assertAlmostEqual(d_aspect, 1.6, places=4)

    def test_vertical_and_horizontal_differences(self):
        self.assertAlmostEqual(
            calculate_vertical_difference({"x": 0.3, "y": 0.55}, {"x": 0.5, "y": 0.70}),
            0.15,
            places=4,
        )
        self.assertAlmostEqual(
            calculate_horizontal_difference({"x": 0.42, "y": 0.70}, {"x": 0.56, "y": 0.90}),
            0.14,
            places=4,
        )

    def test_foot_orientation_angle_and_foreshortening(self):
        # 30 deg outward toe angle in 2D plane: dx = tan(30)*dy
        dy = 0.05
        dx = math.tan(math.radians(30.0)) * dy
        ang = calculate_foot_orientation_angle(
            {"x": 0.40, "y": 0.90},
            {"x": 0.40 - dx, "y": 0.90 + dy},
        )
        self.assertAlmostEqual(ang, 30.0, places=1)

        # Foreshortened foot vector (< min_vector_length) rejected as None
        self.assertIsNone(
            calculate_foot_orientation_angle(
                {"x": 0.40, "y": 0.90},
                {"x": 0.402, "y": 0.902},
                min_vector_length=0.015,
            )
        )

    def test_built_pose_angles_exact(self):
        pose = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=85.0)
        k_ang = calculate_angle_3points(pose["left_hip"], pose["left_knee"], pose["left_ankle"])
        h_ang = calculate_angle_3points(pose["left_shoulder"], pose["left_hip"], pose["left_knee"])
        self.assertAlmostEqual(k_ang, 95.0, places=1)
        self.assertAlmostEqual(h_ang, 85.0, places=1)


class TestLandmarkReliability(unittest.TestCase):
    def test_unreliable_low_visibility_returns_insufficient_data(self):
        analyzer = SquatAnalyzer()
        pose = build_pose_from_angles(95.0, 90.0, visibility=0.2)
        res = analyzer.analyze(pose)
        self.assertEqual(res["overall_status"], OverallSquatStatus.INSUFFICIENT_DATA)
        self.assertEqual(res["landmark_reliability"]["status"], "INSUFFICIENT_VISIBILITY")
        self.assertEqual(res["rep_count"], 0)
        self.assertFalse(res["tracking_valid"])

    def test_presence_checked_when_present(self):
        cfg = SquatConfig(min_visibility=0.5, min_presence=0.6)
        validator = LandmarkValidator(cfg)
        pose = build_pose_from_angles(170.0, 170.0)
        # Set low presence on both knees
        pose["left_knee"]["presence"] = 0.2
        pose["right_knee"]["presence"] = 0.2
        rel = validator.validate(pose)
        self.assertFalse(rel.is_tracking_valid)
        self.assertEqual(rel.status, "INSUFFICIENT_VISIBILITY")

    def test_missing_left_ankle_disables_left_knee_without_averaging(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0)
        analyzer = SquatAnalyzer(config=cfg)
        pose = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=90.0)
        del pose["left_ankle"]

        res = analyzer.analyze_frame(pose)
        self.assertTrue(res["tracking_valid"])
        self.assertIsNone(res["measurements"]["left_knee_angle"])
        self.assertIsNotNone(res["measurements"]["right_knee_angle"])
        # Must NOT average when one side is unreliable
        self.assertIsNone(res["measurements"]["average_knee_angle"])
        self.assertAlmostEqual(
            res["primary_knee_angle"],
            res["measurements"]["right_knee_angle"],
            places=2,
        )

    def test_missing_foot_disables_knee_alignment_only(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0)
        analyzer = SquatAnalyzer(config=cfg)
        pose = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=90.0)
        del pose["left_foot_index"]
        del pose["right_foot_index"]

        res = analyzer.analyze_frame(pose)
        self.assertTrue(res["tracking_valid"])
        self.assertEqual(res["rules"]["knee_alignment"]["status"], RuleStatus.UNAVAILABLE)
        self.assertEqual(res["rules"]["hip_angle"]["status"], RuleStatus.PASS)
        self.assertEqual(res["overall_status"], OverallSquatStatus.CORRECT)

    def test_sudden_coordinate_jump_rejected(self):
        cfg = SquatConfig(max_coordinate_jump=0.25, jump_recovery_frames=3, smoothing_alpha=1.0)
        analyzer = SquatAnalyzer(config=cfg)

        pose1 = build_pose_from_angles(170.0, 170.0)
        res1 = analyzer.analyze(pose1)
        self.assertTrue(res1["tracking_valid"])

        # Single-frame implausible spike on hips & knees
        pose_spike = build_pose_from_angles(170.0, 170.0)
        pose_spike["left_hip"]["x"] += 0.45
        pose_spike["right_hip"]["x"] += 0.45
        res_spike = analyzer.analyze(pose_spike)
        self.assertFalse(res_spike["tracking_valid"])
        self.assertEqual(res_spike["landmark_reliability"]["status"], "COORD_JUMP_DETECTED")
        self.assertEqual(res_spike["overall_status"], OverallSquatStatus.INSUFFICIENT_DATA)


class TestSquatDepthAndRules(unittest.TestCase):
    def test_depth_categories_half_and_deep_modes(self):
        cfg_half = SquatConfig(target_depth_mode="half")
        depth_half = SquatDepthAnalyzer(cfg_half)
        res_half = depth_half.analyze(knee_angle=112.0, movement_state="BOTTOM")
        self.assertEqual(res_half.classification, "HALF_SQUAT")
        self.assertTrue(res_half.target_reached)

        cfg_deep = SquatConfig(target_depth_mode="deep")
        depth_deep = SquatDepthAnalyzer(cfg_deep)
        res_shallow_in_deep = depth_deep.analyze(knee_angle=112.0, movement_state="DESCENDING")
        self.assertEqual(res_shallow_in_deep.classification, "HALF_SQUAT")
        self.assertFalse(res_shallow_in_deep.target_reached)

        res_deep = depth_deep.analyze(knee_angle=92.0, movement_state="BOTTOM")
        self.assertEqual(res_deep.classification, "DEEP_SQUAT")
        self.assertTrue(res_deep.target_reached)

    def test_hip_angle_rule_boundary_and_warning_only(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0, hip_angle_min=60.0, hip_angle_max=120.0)
        analyzer = SquatAnalyzer(config=cfg)

        # Bottom with hip angle 50 deg (< 60) -> WARNING (NEEDS_ADJUSTMENT), NOT INCORRECT
        pose_closed_hip = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=50.0, foot_dx=0.02)
        res = analyzer.analyze_frame(pose_closed_hip)
        self.assertEqual(res["rules"]["hip_angle"]["status"], RuleStatus.WARNING)
        self.assertEqual(res["overall_status"], OverallSquatStatus.NEEDS_ADJUSTMENT)

        # Bottom with hip angle 90 deg (within 60-120) -> PASS
        pose_ok = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=90.0, foot_dx=0.02)
        res_ok = analyzer.analyze_frame(pose_ok)
        self.assertEqual(res_ok["rules"]["hip_angle"]["status"], RuleStatus.PASS)

    def test_thigh_position_rule_side_view_vs_front_view(self):
        cfg_side = SquatConfig(camera_view="side", smoothing_alpha=1.0, max_hip_knee_vertical_diff=0.20)
        analyzer_side = SquatAnalyzer(config=cfg_side)
        # Knee angle 115 (half squat) -> hip_y is ~0.085 above knee_y -> passes <= 0.20
        # Build deep-ish pose where |y_hip - y_knee| > 0.20 by lengthening thigh
        pose_high_hip = build_pose_from_angles(knee_angle_deg=118.0, hip_angle_deg=90.0, thigh_len=0.45, foot_dx=0.02)
        res_warn = analyzer_side.analyze_frame(pose_high_hip)
        self.assertGreater(res_warn["measurements"]["hip_knee_vertical_diff"], 0.20)
        self.assertEqual(res_warn["rules"]["thigh_position"]["status"], RuleStatus.WARNING)

        # In front view, thigh_position must be UNAVAILABLE
        cfg_front = SquatConfig(camera_view="front", smoothing_alpha=1.0)
        analyzer_front = SquatAnalyzer(config=cfg_front)
        res_front = analyzer_front.analyze_frame(pose_high_hip)
        self.assertEqual(res_front["rules"]["thigh_position"]["status"], RuleStatus.UNAVAILABLE)

    def test_knee_alignment_toggle_and_view_separation(self):
        # Side view with |x_knee - x_foot_index| = 0.16 > 0.10 -> WARNING
        pose_forward_knee = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=90.0, foot_dx=-0.16)
        cfg_on = SquatConfig(camera_view="side", smoothing_alpha=1.0, enable_knee_alignment_rule=True)
        res_on = SquatAnalyzer(config=cfg_on).analyze_frame(pose_forward_knee)
        self.assertEqual(res_on["rules"]["knee_alignment"]["status"], RuleStatus.WARNING)
        self.assertEqual(res_on["rules"]["knee_alignment"]["message"], "Check your knee and foot alignment.")

        # Disabled independently -> UNAVAILABLE
        cfg_off = SquatConfig(camera_view="side", smoothing_alpha=1.0, enable_knee_alignment_rule=False)
        res_off = SquatAnalyzer(config=cfg_off).analyze_frame(pose_forward_knee)
        self.assertEqual(res_off["rules"]["knee_alignment"]["status"], RuleStatus.UNAVAILABLE)

        # Front view must NOT apply side-view 0.1 threshold
        cfg_front = SquatConfig(camera_view="front", smoothing_alpha=1.0, enable_knee_alignment_rule=True)
        res_front = SquatAnalyzer(config=cfg_front).analyze_frame(pose_forward_knee)
        self.assertEqual(res_front["rules"]["knee_alignment"]["status"], RuleStatus.UNAVAILABLE)

    def test_stance_width_classification_front_and_side(self):
        # Front view: standard stance (ratio = 1.0)
        pose_std = build_pose_from_angles(
            170.0, 170.0, side="front", stance_half_width=0.10, shoulder_half_width=0.10
        )
        cfg_front = SquatConfig(camera_view="front", smoothing_alpha=1.0)
        res_std = SquatAnalyzer(config=cfg_front).analyze_frame(pose_std)
        self.assertEqual(res_std["stance_classification"], "STANDARD")
        self.assertAlmostEqual(res_std["measurements"]["stance_ratio"], 1.0, places=2)

        # Front view: wide stance (ratio = 1.40, within Ahn & Kim 1.30-1.50)
        pose_wide = build_pose_from_angles(
            170.0, 170.0, side="front", stance_half_width=0.14, shoulder_half_width=0.10
        )
        res_wide = SquatAnalyzer(config=cfg_front).analyze_frame(pose_wide)
        self.assertEqual(res_wide["stance_classification"], "WIDE")
        self.assertAlmostEqual(res_wide["measurements"]["stance_ratio"], 1.4, places=2)

        # Front view: ambiguous gap (ratio = 1.27) -> UNCLASSIFIED
        pose_ambig = build_pose_from_angles(
            170.0, 170.0, side="front", stance_half_width=0.127, shoulder_half_width=0.10
        )
        res_ambig = SquatAnalyzer(config=cfg_front).analyze_frame(pose_ambig)
        self.assertEqual(res_ambig["stance_classification"], "UNCLASSIFIED")

        # Side view -> UNAVAILABLE / UNCLASSIFIED
        cfg_side = SquatConfig(camera_view="side", smoothing_alpha=1.0)
        res_side = SquatAnalyzer(config=cfg_side).analyze_frame(pose_wide)
        self.assertEqual(res_side["rules"]["stance_width"]["status"], RuleStatus.UNAVAILABLE)
        self.assertIsNone(res_side["measurements"]["stance_ratio"])

    def test_trunk_inclination_experimental_no_feedback_by_default(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0)
        analyzer = SquatAnalyzer(config=cfg)
        # Forward-leaning hip angle 40 deg produces large trunk inclination
        pose = build_pose_from_angles(knee_angle_deg=95.0, hip_angle_deg=40.0, foot_dx=0.02)
        res = analyzer.analyze_frame(pose)
        self.assertIsNotNone(res["measurements"]["trunk_inclination"])
        # Trunk inclination rule stays PASS (no arbitrary corrective feedback)
        self.assertEqual(res["rules"]["trunk_inclination"]["status"], RuleStatus.PASS)


class TestSquatStateMachineAndSequence(unittest.TestCase):
    def _run_sequence(self, analyzer: SquatAnalyzer, knee_angles, hip_angles=None, foot_dx=0.02):
        results = []
        for idx, k_ang in enumerate(knee_angles):
            h_ang = hip_angles[idx] if hip_angles else max(75.0, min(175.0, k_ang - 5.0))
            pose = build_pose_from_angles(knee_angle_deg=k_ang, hip_angle_deg=h_ang, foot_dx=foot_dx)
            results.append(analyzer.analyze(pose))
        return results

    def test_full_valid_squat_counts_once(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0, min_bottom_frames=2)
        analyzer = SquatAnalyzer(config=cfg)

        angles = (
            [170.0] * 4
            + [150.0, 138.0, 122.0, 108.0]
            + [92.0] * 5  # dwell at bottom should NOT double-count
            + [112.0, 132.0, 148.0]
            + [165.0] * 4
        )
        results = self._run_sequence(analyzer, angles)
        self.assertEqual(results[-1]["rep_count"], 1)
        self.assertEqual(results[-1]["movement_state"], "STANDING")

    def test_incomplete_squat_and_micro_bounce_not_counted(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0, min_bottom_frames=2)
        analyzer = SquatAnalyzer(config=cfg)

        # Small knee bounces around standing + shallow descent to 122 deg (target deep=100)
        angles = (
            [170.0, 162.0, 158.0, 168.0]
            + [138.0, 125.0, 122.0, 122.0]
            + [142.0, 165.0, 170.0, 170.0]
        )
        results = self._run_sequence(analyzer, angles)
        self.assertEqual(results[-1]["rep_count"], 0)
        self.assertGreaterEqual(analyzer.counter.incomplete_reps, 1)

    def test_single_frame_bottom_spike_rejected_by_min_bottom_frames(self):
        counter = SquatRepetitionCounter(smoothing_window=1, min_bottom_frames=2)
        for a in [170.0, 130.0, 95.0, 130.0, 170.0]:
            res = counter.update(a)
        # Only 1 frame at 95 -> never confirmed BOTTOM -> rep_count stays 0
        self.assertEqual(res["rep_count"], 0)

    def test_posture_warning_does_not_block_rep_count(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0, min_bottom_frames=2)
        analyzer = SquatAnalyzer(config=cfg)
        # Complete movement cycle with excessive forward knee alignment warning (foot_dx = -0.18)
        angles = [170.0] * 3 + [135.0, 110.0] + [92.0] * 4 + [125.0, 165.0, 168.0, 170.0]
        results = self._run_sequence(analyzer, angles, foot_dx=-0.18)
        self.assertEqual(results[-1]["rep_count"], 1)
        # Confirm warning was emitted during bottom
        bottom_statuses = [r["overall_status"] for r in results if r["movement_state"] == "BOTTOM"]
        self.assertIn(OverallSquatStatus.NEEDS_ADJUSTMENT, bottom_statuses)


class TestSquatDjangoIntegrationAndDebugOverlay(TestCase):
    def setUp(self):
        self.client = Client()

    def test_api_analyze_pose_unified_squat_schema(self):
        pose = build_pose_from_angles(95.0, 90.0, foot_dx=0.02)
        payload = {
            "exercise": "squat",
            "session_token": "squat_unified_test",
            "camera_view": "side",
            "target_depth": "deep",
            "image_width": 640,
            "image_height": 480,
            "keypoints": pose,
        }
        resp = self.client.post(
            reverse("api_analyze_pose"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["camera_view"], "side")
        self.assertIn("overall_status", data)
        self.assertIn("depth_classification", data)
        self.assertIn("rules", data)
        self.assertIn("landmark_reliability", data)
        self.assertIn("average_knee_angle", data["measurements"])
        self.assertIn("average_hip_angle", data["measurements"])

    def test_opencv_debug_overlay_renders_cleanly(self):
        analyzer = SquatAnalyzer(config=SquatConfig(camera_view="side", debug_mode=True))
        pose = build_pose_from_angles(95.0, 90.0, foot_dx=0.02)
        res = analyzer.analyze(pose, image_width=640, image_height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_squat_debug_overlay(frame, pose, res)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertGreater(int(annotated.sum()), 0)

    def test_foot_orientation_rule_when_enabled_in_front_view(self):
        cfg = SquatConfig(
            camera_view="front",
            smoothing_alpha=1.0,
            enable_foot_orientation_rule=True,
            foot_angle_min_deg=14.0,
            foot_angle_max_deg=45.0,
        )
        analyzer = SquatAnalyzer(config=cfg)
        pose = build_pose_from_angles(170.0, 170.0, side="front")
        # Override heel/foot_index for ~25 deg outward toe angle
        dy = 0.05
        dx = math.tan(math.radians(25.0)) * dy
        pose["left_heel"] = {"x": 0.42, "y": 0.90, "visibility": 0.95}
        pose["left_foot_index"] = {"x": 0.42 - dx, "y": 0.90 + dy, "visibility": 0.95}
        pose["right_heel"] = {"x": 0.62, "y": 0.90, "visibility": 0.95}
        pose["right_foot_index"] = {"x": 0.62 + dx, "y": 0.90 + dy, "visibility": 0.95}

        res = analyzer.analyze_frame(pose)
        self.assertEqual(res["rules"]["foot_orientation"]["status"], RuleStatus.PASS)
        self.assertAlmostEqual(res["measurements"]["left_foot_angle"], 25.0, places=1)

        # Excessive 60 deg turnout triggers WARNING
        dx_wide = math.tan(math.radians(60.0)) * dy
        pose["right_foot_index"] = {"x": 0.62 + dx_wide, "y": 0.90 + dy, "visibility": 0.95}
        res_warn = analyzer.analyze_frame(pose)
        self.assertEqual(res_warn["rules"]["foot_orientation"]["status"], RuleStatus.WARNING)

    def test_feedback_stabilization_prevents_rapid_flicker(self):
        cfg = SquatConfig(camera_view="side", smoothing_alpha=1.0, min_feedback_hold_frames=3)
        analyzer = SquatAnalyzer(config=cfg)
        bad_pose = build_pose_from_angles(95.0, 90.0, foot_dx=-0.18)
        good_pose = build_pose_from_angles(95.0, 90.0, foot_dx=0.02)

        r1 = analyzer.analyze_frame(bad_pose)
        self.assertEqual(r1["feedback"], ["Check your knee and foot alignment."])
        # Immediate recovery on next frame holds message briefly to prevent UI flicker
        r2 = analyzer.analyze_frame(good_pose)
        self.assertEqual(r2["feedback"], ["Check your knee and foot alignment."])
        # After hold period expires, clears cleanly
        r3 = analyzer.analyze_frame(good_pose)
        r4 = analyzer.analyze_frame(good_pose)
        self.assertEqual(r4["feedback"], [])
