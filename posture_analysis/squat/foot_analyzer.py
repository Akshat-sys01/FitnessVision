"""
Foot-orientation estimation module for squat posture analysis.

Research foundation:
- Ahn & Kim (2023): outward foot angle range ~14-45 degrees.
- Kwon & Kim (2022): heel and foot-index landmarks.

A single 2D webcam cannot reliably measure 3D foot turn-out when feet are occluded,
viewed from the side, or foreshortened pointing directly toward the camera.
Unavailable measurements are reported as UNAVAILABLE rather than forced errors.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from posture_analysis.geometry import calculate_foot_orientation_angle
from .config import SquatConfig
from .landmark_validator import LandmarkReliabilityResult


@dataclass
class FootOrientationResult:
    left_foot_angle: Optional[float]
    right_foot_angle: Optional[float]
    available: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "left_foot_angle": self.left_foot_angle,
            "right_foot_angle": self.right_foot_angle,
            "available": self.available,
            "reason": self.reason,
        }


class FootAnalyzer:
    """
    Estimates 2D heel-to-foot-index orientation angles in frontal view when
    feet landmarks are clearly visible and not overly foreshortened.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()

    def analyze(
        self,
        reliability: LandmarkReliabilityResult,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> FootOrientationResult:
        if self.config.camera_view != "front":
            return FootOrientationResult(
                left_foot_angle=None,
                right_foot_angle=None,
                available=False,
                reason="2D foot orientation estimation requires front camera view.",
            )

        if not reliability.is_tracking_valid:
            return FootOrientationResult(
                left_foot_angle=None,
                right_foot_angle=None,
                available=False,
                reason="Tracking landmarks are unreliable.",
            )

        lms = reliability.reliable_landmarks
        avail = reliability.rule_availability

        left_angle: Optional[float] = None
        right_angle: Optional[float] = None

        if avail.get("left_foot_orientation"):
            left_angle = calculate_foot_orientation_angle(
                lms["left_heel"],
                lms["left_foot_index"],
                image_width=image_width,
                image_height=image_height,
                min_vector_length=self.config.min_foot_vector_length,
            )

        if avail.get("right_foot_orientation"):
            right_angle = calculate_foot_orientation_angle(
                lms["right_heel"],
                lms["right_foot_index"],
                image_width=image_width,
                image_height=image_height,
                min_vector_length=self.config.min_foot_vector_length,
            )

        if left_angle is None and right_angle is None:
            return FootOrientationResult(
                left_foot_angle=None,
                right_foot_angle=None,
                available=False,
                reason="Foot landmarks are occluded or foreshortened in 2D view.",
            )

        return FootOrientationResult(
            left_foot_angle=left_angle,
            right_foot_angle=right_angle,
            available=True,
            reason=None,
        )
