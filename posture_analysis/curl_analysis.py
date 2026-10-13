"""
Bicep curl posture analyzer for FitnessVision.
Performs geometric feature extraction, phase tracking, and form error detection for curls.
"""

from typing import Dict, Any, List, Optional
from .geometry import (
    calculate_angle_3points,
    calculate_torso_inclination,
    calculate_displacement,
    calculate_symmetry,
)
from .error_detection import get_error, FormError
from pose_estimation.keypoints import check_keypoint_visibility, EXERCISE_REQUIRED_KEYPOINTS


class BicepCurlAnalyzer:
    """
    Analyzes body keypoints during bicep curls.
    Evaluates elbow flexion, upper-arm stability, and torso momentum sway.
    """

    def __init__(
        self,
        min_visibility: float = 0.5,
        extended_threshold: float = 145.0,
        flexed_threshold: float = 60.0,
        max_elbow_drift_ratio: float = 0.20,
        max_torso_sway: float = 22.0,
    ):
        self.min_visibility = min_visibility
        self.extended_threshold = extended_threshold
        self.flexed_threshold = flexed_threshold
        self.max_elbow_drift_ratio = max_elbow_drift_ratio
        self.max_torso_sway = max_torso_sway

    def analyze_frame(self, keypoints: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyzes a single frame's keypoints for bicep curl form.

        Returns structured dict with:
            - tracking_valid: bool
            - missing_joints: List[str]
            - measurements: Dict[str, Optional[float]]
            - detected_errors: List[FormError]
            - primary_elbow_angle: Optional[float]
        """
        required = EXERCISE_REQUIRED_KEYPOINTS["bicep_curl"]
        is_valid, missing = check_keypoint_visibility(
            keypoints, required, min_visibility=self.min_visibility
        )

        measurements: Dict[str, Optional[float]] = {
            "left_elbow_angle": None,
            "right_elbow_angle": None,
            "primary_elbow_angle": None,
            "left_shoulder_angle": None,
            "right_shoulder_angle": None,
            "torso_inclination": None,
            "elbow_symmetry_diff": None,
        }
        detected_errors: List[FormError] = []

        if not is_valid:
            return {
                "tracking_valid": False,
                "missing_joints": missing,
                "measurements": measurements,
                "detected_errors": [get_error("KEYPOINTS_LOW_CONFIDENCE")],
                "primary_elbow_angle": None,
            }

        # 1. Calculate joint angles
        # Left elbow angle: shoulder - elbow - wrist
        left_elbow = calculate_angle_3points(
            keypoints.get("left_shoulder"),
            keypoints.get("left_elbow"),
            keypoints.get("left_wrist"),
        )
        # Right elbow angle: shoulder - elbow - wrist
        right_elbow = calculate_angle_3points(
            keypoints.get("right_shoulder"),
            keypoints.get("right_elbow"),
            keypoints.get("right_wrist"),
        )

        # Upper arm to torso angle: hip - shoulder - elbow
        left_sh_angle = calculate_angle_3points(
            keypoints.get("left_hip"),
            keypoints.get("left_shoulder"),
            keypoints.get("left_elbow"),
        )
        right_sh_angle = calculate_angle_3points(
            keypoints.get("right_hip"),
            keypoints.get("right_shoulder"),
            keypoints.get("right_elbow"),
        )

        angles = [a for a in (left_elbow, right_elbow) if a is not None]
        primary_elbow = round(sum(angles) / len(angles), 2) if angles else None

        # Torso inclination
        l_sh = keypoints.get("left_shoulder")
        r_sh = keypoints.get("right_shoulder")
        l_hp = keypoints.get("left_hip")
        r_hp = keypoints.get("right_hip")

        mid_sh = {"x": (l_sh["x"] + r_sh["x"]) / 2.0, "y": (l_sh["y"] + r_sh["y"]) / 2.0}
        mid_hp = {"x": (l_hp["x"] + r_hp["x"]) / 2.0, "y": (l_hp["y"] + r_hp["y"]) / 2.0}
        torso_inc = calculate_torso_inclination(mid_sh, mid_hp)

        measurements["left_elbow_angle"] = left_elbow
        measurements["right_elbow_angle"] = right_elbow
        measurements["primary_elbow_angle"] = primary_elbow
        measurements["left_shoulder_angle"] = left_sh_angle
        measurements["right_shoulder_angle"] = right_sh_angle
        measurements["torso_inclination"] = torso_inc

        # 2. Form Error Detection
        # Error A: Upper arm flare / excessive elbow swing
        # If upper arm angle > 40 degrees while curling, elbows are flaring or swinging forward/backward
        sh_angles = [a for a in (left_sh_angle, right_sh_angle) if a is not None]
        if sh_angles:
            max_sh = max(sh_angles)
            if max_sh > 40.0:
                detected_errors.append(get_error("EXCESSIVE_ELBOW_DRIFT"))

        # Error B: Torso sway / leaning back for momentum
        if torso_inc is not None and torso_inc > self.max_torso_sway:
            detected_errors.append(get_error("TORSO_SWAY"))

        # Error C: Asymmetry between arms during curl
        if left_elbow is not None and right_elbow is not None:
            symm = calculate_symmetry(left_elbow, right_elbow)
            if symm:
                measurements["elbow_symmetry_diff"] = symm["absolute_diff"]

        return {
            "tracking_valid": True,
            "missing_joints": [],
            "measurements": measurements,
            "detected_errors": detected_errors,
            "primary_elbow_angle": primary_elbow,
        }
