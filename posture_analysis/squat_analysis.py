"""
Squat posture analyzer for FitnessVision.
Performs geometric feature extraction, phase tracking, and form error detection for squats.
"""

from typing import Dict, Any, List, Optional
from .geometry import (
    calculate_angle_3points,
    calculate_torso_inclination,
    calculate_euclidean_distance,
    calculate_symmetry,
)
from .error_detection import get_error, FormError
from pose_estimation.keypoints import check_keypoint_visibility, EXERCISE_REQUIRED_KEYPOINTS


class SquatAnalyzer:
    """
    Analyzes body keypoints during a squat exercise.
    Configurable thresholds allow tuning for individual body mechanics.
    """

    def __init__(
        self,
        min_visibility: float = 0.5,
        standing_threshold: float = 160.0,
        depth_threshold: float = 100.0,
        shallow_threshold: float = 120.0,
        max_torso_inclination: float = 48.0,
        max_knee_asymmetry: float = 25.0,
    ):
        self.min_visibility = min_visibility
        self.standing_threshold = standing_threshold
        self.depth_threshold = depth_threshold
        self.shallow_threshold = shallow_threshold
        self.max_torso_inclination = max_torso_inclination
        self.max_knee_asymmetry = max_knee_asymmetry

    def analyze_frame(self, keypoints: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Analyzes a single frame's keypoints for squat mechanics.

        Returns structured dict with:
            - tracking_valid: bool
            - missing_joints: List[str]
            - measurements: Dict[str, Optional[float]]
            - detected_errors: List[FormError]
            - primary_knee_angle: Optional[float]
        """
        required = EXERCISE_REQUIRED_KEYPOINTS["squat"]
        is_valid, missing = check_keypoint_visibility(
            keypoints, required, min_visibility=self.min_visibility
        )

        measurements: Dict[str, Optional[float]] = {
            "left_knee_angle": None,
            "right_knee_angle": None,
            "primary_knee_angle": None,
            "left_hip_angle": None,
            "right_hip_angle": None,
            "torso_inclination": None,
            "knee_symmetry_diff": None,
        }
        detected_errors: List[FormError] = []

        if not is_valid:
            return {
                "tracking_valid": False,
                "missing_joints": missing,
                "measurements": measurements,
                "detected_errors": [get_error("KEYPOINTS_LOW_CONFIDENCE")],
                "primary_knee_angle": None,
            }

        # 1. Joint angles
        # Left knee angle: hip - knee - ankle
        left_knee_angle = calculate_angle_3points(
            keypoints.get("left_hip"),
            keypoints.get("left_knee"),
            keypoints.get("left_ankle"),
        )
        # Right knee angle: hip - knee - ankle
        right_knee_angle = calculate_angle_3points(
            keypoints.get("right_hip"),
            keypoints.get("right_knee"),
            keypoints.get("right_ankle"),
        )

        # Left hip angle: shoulder - hip - knee
        left_hip_angle = calculate_angle_3points(
            keypoints.get("left_shoulder"),
            keypoints.get("left_hip"),
            keypoints.get("left_knee"),
        )
        # Right hip angle: shoulder - hip - knee
        right_hip_angle = calculate_angle_3points(
            keypoints.get("right_shoulder"),
            keypoints.get("right_hip"),
            keypoints.get("right_knee"),
        )

        # Determine primary knee angle (average of both if visible, else the available one)
        angles = [a for a in (left_knee_angle, right_knee_angle) if a is not None]
        primary_knee = round(sum(angles) / len(angles), 2) if angles else None

        # Torso inclination relative to vertical
        # Average shoulder and average hip midpoint
        l_sh = keypoints.get("left_shoulder")
        r_sh = keypoints.get("right_shoulder")
        l_hp = keypoints.get("left_hip")
        r_hp = keypoints.get("right_hip")

        mid_sh = {"x": (l_sh["x"] + r_sh["x"]) / 2.0, "y": (l_sh["y"] + r_sh["y"]) / 2.0}
        mid_hp = {"x": (l_hp["x"] + r_hp["x"]) / 2.0, "y": (l_hp["y"] + r_hp["y"]) / 2.0}
        torso_inc = calculate_torso_inclination(mid_sh, mid_hp)

        measurements["left_knee_angle"] = left_knee_angle
        measurements["right_knee_angle"] = right_knee_angle
        measurements["primary_knee_angle"] = primary_knee
        measurements["left_hip_angle"] = left_hip_angle
        measurements["right_hip_angle"] = right_hip_angle
        measurements["torso_inclination"] = torso_inc

        # 2. Rule evaluation & Error detection
        # Error A: Excessive forward lean (torso inclination > threshold)
        if torso_inc is not None and torso_inc > self.max_torso_inclination:
            detected_errors.append(get_error("EXCESSIVE_FORWARD_LEAN"))

        # Error B: Knee asymmetry
        if left_knee_angle is not None and right_knee_angle is not None:
            symm = calculate_symmetry(left_knee_angle, right_knee_angle)
            if symm:
                measurements["knee_symmetry_diff"] = symm["absolute_diff"]
                if symm["absolute_diff"] > self.max_knee_asymmetry:
                    detected_errors.append(get_error("ASYMMETRIC_SQUAT"))

        # Error C: Knee valgus (knees caving inward compared to ankles)
        l_kn = keypoints.get("left_knee")
        r_kn = keypoints.get("right_knee")
        l_an = keypoints.get("left_ankle")
        r_an = keypoints.get("right_ankle")
        if l_kn and r_kn and l_an and r_an:
            knee_dist = calculate_euclidean_distance(l_kn, r_kn)
            ankle_dist = calculate_euclidean_distance(l_an, r_an)
            # Only evaluate valgus if squatting (knee angle < 140) and ankle distance is valid
            if primary_knee is not None and primary_knee < 140.0:
                if knee_dist is not None and ankle_dist is not None and ankle_dist > 0.05:
                    if knee_dist < (ankle_dist * 0.65):
                        detected_errors.append(get_error("KNEE_VALGUS"))

        return {
            "tracking_valid": True,
            "missing_joints": [],
            "measurements": measurements,
            "detected_errors": detected_errors,
            "primary_knee_angle": primary_knee,
        }
