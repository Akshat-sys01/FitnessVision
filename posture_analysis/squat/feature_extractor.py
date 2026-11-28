"""
Geometric feature extraction module for squat posture analysis.
Extracts smoothed landmarks and computes knee angles, hip angles,
coordinate displacements, trunk inclination, and symmetry metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from posture_analysis.geometry import (
    calculate_angle_3points,
    calculate_torso_inclination,
    calculate_vertical_difference,
    calculate_horizontal_difference,
    calculate_symmetry,
    calculate_midpoint,
)
from .config import SquatConfig
from .landmark_validator import LandmarkReliabilityResult


@dataclass
class SquatFeatures:
    """All extracted geometric features for a single frame."""

    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    average_knee_angle: Optional[float] = None
    primary_knee_angle: Optional[float] = None

    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    average_hip_angle: Optional[float] = None
    primary_hip_angle: Optional[float] = None

    left_hip_knee_y_diff: Optional[float] = None
    right_hip_knee_y_diff: Optional[float] = None
    hip_knee_vertical_diff: Optional[float] = None

    left_knee_foot_x_diff: Optional[float] = None
    right_knee_foot_x_diff: Optional[float] = None
    max_knee_foot_x_diff: Optional[float] = None

    trunk_inclination: Optional[float] = None
    knee_symmetry_diff: Optional[float] = None

    # Stance & foot metrics populated by dedicated analyzers
    stance_ratio: Optional[float] = None
    stance_classification: Optional[str] = None
    left_foot_angle: Optional[float] = None
    right_foot_angle: Optional[float] = None

    smoothed_landmarks: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_measurements_dict(self) -> Dict[str, Optional[float]]:
        """Serializes numeric measurements cleanly for API responses and logging."""
        return {
            "left_knee_angle": self.left_knee_angle,
            "right_knee_angle": self.right_knee_angle,
            "average_knee_angle": self.average_knee_angle,
            "primary_knee_angle": self.primary_knee_angle,
            "left_hip_angle": self.left_hip_angle,
            "right_hip_angle": self.right_hip_angle,
            "average_hip_angle": self.average_hip_angle,
            "hip_knee_vertical_diff": self.hip_knee_vertical_diff,
            "left_knee_foot_x_diff": self.left_knee_foot_x_diff,
            "right_knee_foot_x_diff": self.right_knee_foot_x_diff,
            "max_knee_foot_x_diff": self.max_knee_foot_x_diff,
            "stance_ratio": self.stance_ratio,
            "left_foot_angle": self.left_foot_angle,
            "right_foot_angle": self.right_foot_angle,
            "trunk_inclination": self.trunk_inclination,
            "torso_inclination": self.trunk_inclination,  # HUD/API backward compatibility
            "knee_symmetry_diff": self.knee_symmetry_diff,
        }


class SquatFeatureExtractor:
    """
    Computes geometric squat measurements from validated landmarks with
    optional temporal EMA smoothing and image aspect-ratio correction.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()
        self._ema_landmarks: Dict[str, Tuple[float, float]] = {}

    def reset(self):
        """Resets temporal EMA smoothing buffers."""
        self._ema_landmarks.clear()

    def _smooth_landmarks(
        self, reliable_landmarks: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        alpha = self.config.smoothing_alpha
        smoothed: Dict[str, Dict[str, float]] = {}

        for name, kp in reliable_landmarks.items():
            cx = float(kp["x"])
            cy = float(kp["y"])
            if alpha < 1.0 and name in self._ema_landmarks:
                px, py = self._ema_landmarks[name]
                sx = alpha * cx + (1.0 - alpha) * px
                sy = alpha * cy + (1.0 - alpha) * py
            else:
                sx, sy = cx, cy

            self._ema_landmarks[name] = (sx, sy)
            copy_kp = dict(kp)
            copy_kp["x"] = sx
            copy_kp["y"] = sy
            smoothed[name] = copy_kp

        return smoothed

    def extract(
        self,
        reliability: LandmarkReliabilityResult,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> SquatFeatures:
        """
        Extracts all geometric squat features from validated landmarks.
        Never averages two measurements when one side is unreliable.
        """
        if not reliability.is_tracking_valid:
            return SquatFeatures()

        lms = self._smooth_landmarks(reliability.reliable_landmarks)
        avail = reliability.rule_availability
        features = SquatFeatures(smoothed_landmarks=lms)

        # 1. Knee Angles (Hip -> Knee -> Ankle)
        if avail.get("left_knee_angle"):
            features.left_knee_angle = calculate_angle_3points(
                lms["left_hip"],
                lms["left_knee"],
                lms["left_ankle"],
                image_width=image_width,
                image_height=image_height,
            )
        if avail.get("right_knee_angle"):
            features.right_knee_angle = calculate_angle_3points(
                lms["right_hip"],
                lms["right_knee"],
                lms["right_ankle"],
                image_width=image_width,
                image_height=image_height,
            )

        if features.left_knee_angle is not None and features.right_knee_angle is not None:
            features.average_knee_angle = round(
                (features.left_knee_angle + features.right_knee_angle) / 2.0, 2
            )
            features.primary_knee_angle = features.average_knee_angle
        else:
            # Side-view fallback: use the single reliable side without averaging
            features.average_knee_angle = None
            features.primary_knee_angle = (
                features.left_knee_angle
                if features.left_knee_angle is not None
                else features.right_knee_angle
            )

        # 2. Hip Angles (Shoulder -> Hip -> Knee)
        if avail.get("left_hip_angle"):
            features.left_hip_angle = calculate_angle_3points(
                lms["left_shoulder"],
                lms["left_hip"],
                lms["left_knee"],
                image_width=image_width,
                image_height=image_height,
            )
        if avail.get("right_hip_angle"):
            features.right_hip_angle = calculate_angle_3points(
                lms["right_shoulder"],
                lms["right_hip"],
                lms["right_knee"],
                image_width=image_width,
                image_height=image_height,
            )

        if features.left_hip_angle is not None and features.right_hip_angle is not None:
            features.average_hip_angle = round(
                (features.left_hip_angle + features.right_hip_angle) / 2.0, 2
            )
            features.primary_hip_angle = features.average_hip_angle
        else:
            features.average_hip_angle = None
            features.primary_hip_angle = (
                features.left_hip_angle
                if features.left_hip_angle is not None
                else features.right_hip_angle
            )

        # 3. Hip-to-Knee Vertical Difference |y_hip - y_knee|
        if avail.get("left_thigh_position"):
            features.left_hip_knee_y_diff = calculate_vertical_difference(
                lms["left_hip"], lms["left_knee"]
            )
        if avail.get("right_thigh_position"):
            features.right_hip_knee_y_diff = calculate_vertical_difference(
                lms["right_hip"], lms["right_knee"]
            )

        if features.left_hip_knee_y_diff is not None and features.right_hip_knee_y_diff is not None:
            features.hip_knee_vertical_diff = round(
                (features.left_hip_knee_y_diff + features.right_hip_knee_y_diff) / 2.0, 4
            )
        else:
            features.hip_knee_vertical_diff = (
                features.left_hip_knee_y_diff
                if features.left_hip_knee_y_diff is not None
                else features.right_hip_knee_y_diff
            )

        # 4. Knee-to-Foot Horizontal Difference |x_knee - x_foot_index|
        if avail.get("left_knee_alignment"):
            features.left_knee_foot_x_diff = calculate_horizontal_difference(
                lms["left_knee"],
                lms["left_foot_index"],
                image_width=image_width,
                image_height=image_height,
            )
        if avail.get("right_knee_alignment"):
            features.right_knee_foot_x_diff = calculate_horizontal_difference(
                lms["right_knee"],
                lms["right_foot_index"],
                image_width=image_width,
                image_height=image_height,
            )

        x_diffs = [
            d for d in (features.left_knee_foot_x_diff, features.right_knee_foot_x_diff)
            if d is not None
        ]
        features.max_knee_foot_x_diff = max(x_diffs) if x_diffs else None

        # 5. Trunk Inclination (Experimental measurement)
        if self.config.enable_trunk_inclination_measurement:
            both_sh_hp = all(
                k in lms for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
            )
            if both_sh_hp:
                mid_sh = calculate_midpoint(lms["left_shoulder"], lms["right_shoulder"])
                mid_hp = calculate_midpoint(lms["left_hip"], lms["right_hip"])
                features.trunk_inclination = calculate_torso_inclination(
                    mid_sh, mid_hp, image_width=image_width, image_height=image_height
                )
            elif "left_shoulder" in lms and "left_hip" in lms:
                features.trunk_inclination = calculate_torso_inclination(
                    lms["left_shoulder"],
                    lms["left_hip"],
                    image_width=image_width,
                    image_height=image_height,
                )
            elif "right_shoulder" in lms and "right_hip" in lms:
                features.trunk_inclination = calculate_torso_inclination(
                    lms["right_shoulder"],
                    lms["right_hip"],
                    image_width=image_width,
                    image_height=image_height,
                )

        # 6. Bilateral Knee Symmetry
        if features.left_knee_angle is not None and features.right_knee_angle is not None:
            symm = calculate_symmetry(features.left_knee_angle, features.right_knee_angle)
            if symm:
                features.knee_symmetry_diff = symm["absolute_diff"]

        return features
