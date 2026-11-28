"""
Stance-width classification module for squat posture analysis.

Research foundation: Ahn & Kim (2023) compare ankle width to shoulder width,
classifying wide squat stance around 130%-150% of shoulder width.
Applicable only in frontal camera view.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from posture_analysis.geometry import calculate_euclidean_distance
from .config import SquatConfig
from .landmark_validator import LandmarkReliabilityResult


class StanceCategory:
    STANDARD = "STANDARD"
    WIDE = "WIDE"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class StanceResult:
    classification: str
    stance_ratio: Optional[float]
    ankle_width: Optional[float]
    shoulder_width: Optional[float]
    available: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "classification": self.classification,
            "stance_ratio": self.stance_ratio,
            "ankle_width": self.ankle_width,
            "shoulder_width": self.shoulder_width,
            "available": self.available,
            "reason": self.reason,
        }


class StanceAnalyzer:
    """
    Classifies squat stance into STANDARD, WIDE, or UNCLASSIFIED based on
    ankle_width / shoulder_width ratio in frontal camera view.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()

    def analyze(
        self,
        reliability: LandmarkReliabilityResult,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> StanceResult:
        if not self.config.enable_stance_analysis:
            return StanceResult(
                classification=StanceCategory.UNCLASSIFIED,
                stance_ratio=None,
                ankle_width=None,
                shoulder_width=None,
                available=False,
                reason="Stance analysis is disabled in configuration.",
            )

        if self.config.camera_view != "front":
            return StanceResult(
                classification=StanceCategory.UNCLASSIFIED,
                stance_ratio=None,
                ankle_width=None,
                shoulder_width=None,
                available=False,
                reason="Stance width requires front camera view.",
            )

        if not reliability.is_tracking_valid or not reliability.rule_availability.get("stance_width", False):
            return StanceResult(
                classification=StanceCategory.UNCLASSIFIED,
                stance_ratio=None,
                ankle_width=None,
                shoulder_width=None,
                available=False,
                reason="Required shoulder or ankle landmarks are unavailable.",
            )

        lms = reliability.reliable_landmarks
        ankle_width = calculate_euclidean_distance(
            lms["left_ankle"],
            lms["right_ankle"],
            image_width=image_width,
            image_height=image_height,
        )
        shoulder_width = calculate_euclidean_distance(
            lms["left_shoulder"],
            lms["right_shoulder"],
            image_width=image_width,
            image_height=image_height,
        )

        if (
            ankle_width is None
            or shoulder_width is None
            or shoulder_width < self.config.min_shoulder_width
        ):
            return StanceResult(
                classification=StanceCategory.UNCLASSIFIED,
                stance_ratio=None,
                ankle_width=ankle_width,
                shoulder_width=shoulder_width,
                available=False,
                reason="Shoulder width denominator is too small or unreliable.",
            )

        ratio = round(ankle_width / shoulder_width, 4)

        if self.config.standard_stance_min_ratio <= ratio <= self.config.standard_stance_max_ratio:
            classification = StanceCategory.STANDARD
        elif self.config.wide_stance_min_ratio <= ratio <= self.config.wide_stance_max_ratio:
            classification = StanceCategory.WIDE
        else:
            classification = StanceCategory.UNCLASSIFIED

        return StanceResult(
            classification=classification,
            stance_ratio=ratio,
            ankle_width=ankle_width,
            shoulder_width=shoulder_width,
            available=True,
            reason=None,
        )
