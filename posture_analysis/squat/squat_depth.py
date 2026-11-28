"""
Squat depth analysis module.

Classifies squat phase/depth into Standing, Descending, Half squat, Deep squat,
and Ascending using knee-angle measurements and relative vertical hip-knee position.
Operates independently of general posture-error classification.

Note on Ahn & Kim (2023):
The 2023 paper distinguishes half squats (15-20 deg) and deep squats (21-30 deg)
using a reference-vector angle-difference method. Because that measurement is
mathematically distinct from raw knee-joint flexion angle, FitnessVision does NOT
substitute raw knee angle for those 15-30 deg thresholds. Instead, depth is
classified using calibrated knee-angle thresholds and vertical hip-knee offset.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .config import SquatConfig


class DepthCategory:
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    HALF_SQUAT = "HALF_SQUAT"
    DEEP_SQUAT = "DEEP_SQUAT"
    ASCENDING = "ASCENDING"
    UNKNOWN = "UNKNOWN"


@dataclass
class SquatDepthResult:
    classification: str
    knee_angle: Optional[float]
    hip_knee_vertical_diff: Optional[float]
    target_depth_mode: str
    target_reached: bool
    is_at_bottom: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "classification": self.classification,
            "knee_angle": self.knee_angle,
            "hip_knee_vertical_diff": self.hip_knee_vertical_diff,
            "target_depth_mode": self.target_depth_mode,
            "target_reached": self.target_reached,
            "is_at_bottom": self.is_at_bottom,
        }


class SquatDepthAnalyzer:
    """
    Determines current squat depth category and whether the configured target
    depth mode ('half' or 'deep') has been satisfied.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()

    def analyze(
        self,
        knee_angle: Optional[float],
        movement_state: Optional[str] = None,
        hip_knee_vertical_diff: Optional[float] = None,
    ) -> SquatDepthResult:
        """
        Classifies current squat depth from knee angle and movement state.
        Does not mark a half squat as incorrect when target_depth_mode == 'half'.
        """
        mode = self.config.target_depth_mode

        if knee_angle is None:
            return SquatDepthResult(
                classification=DepthCategory.UNKNOWN,
                knee_angle=None,
                hip_knee_vertical_diff=hip_knee_vertical_diff,
                target_depth_mode=mode,
                target_reached=False,
                is_at_bottom=False,
            )

        state_upper = movement_state.upper() if movement_state else ""

        if knee_angle <= self.config.deep_squat_knee_angle:
            classification = DepthCategory.DEEP_SQUAT
            target_reached = True
            is_at_bottom = True
        elif knee_angle <= self.config.half_squat_knee_angle:
            classification = DepthCategory.HALF_SQUAT
            target_reached = mode == "half"
            is_at_bottom = state_upper == "BOTTOM" or mode == "half"
        elif state_upper == "ASCENDING":
            classification = DepthCategory.ASCENDING
            target_reached = False
            is_at_bottom = False
        elif knee_angle >= self.config.standing_knee_angle:
            classification = DepthCategory.STANDING
            target_reached = False
            is_at_bottom = False
        elif state_upper == "DESCENDING" or knee_angle < self.config.descent_knee_threshold:
            classification = DepthCategory.DESCENDING
            target_reached = False
            is_at_bottom = False
        else:
            classification = DepthCategory.STANDING
            target_reached = False
            is_at_bottom = False

        return SquatDepthResult(
            classification=classification,
            knee_angle=knee_angle,
            hip_knee_vertical_diff=hip_knee_vertical_diff,
            target_depth_mode=mode,
            target_reached=target_reached,
            is_at_bottom=is_at_bottom,
        )
