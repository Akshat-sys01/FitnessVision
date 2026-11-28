"""
Landmark reliability and validation module for squat analysis.

Verifies visibility/presence, coordinate finiteness, normalized bounds,
per-rule landmark availability, and sudden implausible coordinate jumps.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from .config import SquatConfig
from .landmark_config import (
    SQUAT_LEFT_CHAIN,
    SQUAT_RIGHT_CHAIN,
    SQUAT_LANDMARK_INDICES,
    RULE_REQUIRED_LANDMARKS,
)


@dataclass
class LandmarkReliabilityResult:
    """Structured landmark reliability summary for a single frame."""

    status: str  # "VALID", "INSUFFICIENT_VISIBILITY", "COORD_JUMP_DETECTED", "INVALID"
    is_tracking_valid: bool
    reliable_landmarks: Dict[str, Dict[str, float]]
    unreliable_landmarks: List[str] = field(default_factory=list)
    available_sides: List[str] = field(default_factory=list)
    rule_availability: Dict[str, bool] = field(default_factory=dict)
    jump_detected: bool = False
    jumped_landmarks: List[str] = field(default_factory=list)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "is_tracking_valid": self.is_tracking_valid,
            "unreliable_landmarks": list(self.unreliable_landmarks),
            "available_sides": list(self.available_sides),
            "jump_detected": self.jump_detected,
            "jumped_landmarks": list(self.jumped_landmarks),
            "message": self.message,
        }


class LandmarkValidator:
    """
    Validates MediaPipe landmarks for squat analysis without rejecting optional
    sub-features when only peripheral landmarks (e.g., toes) are occluded.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()
        self._previous_landmarks: Dict[str, Tuple[float, float]] = {}
        self._consecutive_jumps: int = 0

    def reset(self):
        """Resets temporal jump-detection memory."""
        self._previous_landmarks.clear()
        self._consecutive_jumps = 0

    def _is_landmark_valid(self, kp: Optional[Dict[str, float]]) -> bool:
        if kp is None or not isinstance(kp, dict):
            return False

        x = kp.get("x")
        y = kp.get("y")
        z = kp.get("z", 0.0)
        visibility = kp.get("visibility")

        if x is None or y is None or visibility is None:
            return False

        try:
            fx = float(x)
            fy = float(y)
            fz = float(0.0 if z is None else z)
            fvis = float(visibility)
        except (ValueError, TypeError):
            return False

        if not (math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz) and math.isfinite(fvis)):
            return False

        if self.config.enforce_normalized_bounds:
            if not (self.config.coord_min_bound <= fx <= self.config.coord_max_bound):
                return False
            if not (self.config.coord_min_bound <= fy <= self.config.coord_max_bound):
                return False

        if fvis < self.config.min_visibility:
            return False

        # Check presence only when populated by the MediaPipe version
        presence = kp.get("presence")
        if presence is not None and self.config.min_presence is not None:
            try:
                fpres = float(presence)
            except (ValueError, TypeError):
                return False
            if not math.isfinite(fpres) or fpres < self.config.min_presence:
                return False

        return True

    def validate(self, keypoints: Optional[Dict[str, Dict[str, float]]]) -> LandmarkReliabilityResult:
        """
        Validates incoming sanitized keypoints and checks per-rule availability.
        """
        if not keypoints or not isinstance(keypoints, dict):
            return LandmarkReliabilityResult(
                status="INSUFFICIENT_VISIBILITY",
                is_tracking_valid=False,
                reliable_landmarks={},
                unreliable_landmarks=list(SQUAT_LANDMARK_INDICES.keys()),
                available_sides=[],
                rule_availability={k: False for k in RULE_REQUIRED_LANDMARKS},
                message="Please ensure your full body is visible to the camera.",
            )

        reliable: Dict[str, Dict[str, float]] = {}
        unreliable: List[str] = []

        for name in SQUAT_LANDMARK_INDICES:
            kp = keypoints.get(name)
            if self._is_landmark_valid(kp):
                reliable[name] = kp
            else:
                unreliable.append(name)

        # Check core side chains
        left_core_ok = all(name in reliable for name in SQUAT_LEFT_CHAIN)
        right_core_ok = all(name in reliable for name in SQUAT_RIGHT_CHAIN)

        available_sides: List[str] = []
        if left_core_ok:
            available_sides.append("left")
        if right_core_ok:
            available_sides.append("right")

        # Require at least one complete leg-torso chain for squat movement tracking
        if not available_sides:
            missing_core = [
                n for n in (SQUAT_LEFT_CHAIN + SQUAT_RIGHT_CHAIN) if n not in reliable
            ]
            return LandmarkReliabilityResult(
                status="INSUFFICIENT_VISIBILITY",
                is_tracking_valid=False,
                reliable_landmarks=reliable,
                unreliable_landmarks=missing_core or unreliable,
                available_sides=[],
                rule_availability={k: False for k in RULE_REQUIRED_LANDMARKS},
                message="Please ensure your full body is visible to the camera.",
            )

        # Sudden coordinate jump detection across core tracked landmarks
        core_to_check = []
        if left_core_ok:
            core_to_check.extend(SQUAT_LEFT_CHAIN)
        if right_core_ok:
            core_to_check.extend(SQUAT_RIGHT_CHAIN)

        jumped: List[str] = []
        if self._previous_landmarks:
            for name in core_to_check:
                if name in self._previous_landmarks and name in reliable:
                    px, py = self._previous_landmarks[name]
                    cx, cy = float(reliable[name]["x"]), float(reliable[name]["y"])
                    dist = math.hypot(cx - px, cy - py)
                    if dist > self.config.max_coordinate_jump:
                        jumped.append(name)

        if jumped:
            self._consecutive_jumps += 1
            if self._consecutive_jumps < self.config.jump_recovery_frames:
                return LandmarkReliabilityResult(
                    status="COORD_JUMP_DETECTED",
                    is_tracking_valid=False,
                    reliable_landmarks=reliable,
                    unreliable_landmarks=unreliable,
                    available_sides=available_sides,
                    rule_availability={k: False for k in RULE_REQUIRED_LANDMARKS},
                    jump_detected=True,
                    jumped_landmarks=jumped,
                    message="Tracking stabilized momentarily; hold steady in camera view.",
                )
            # Re-anchor after sustained position change
            self._consecutive_jumps = 0
        else:
            self._consecutive_jumps = 0

        # Update temporal landmark anchor with reliable coordinates
        for name, kp in reliable.items():
            self._previous_landmarks[name] = (float(kp["x"]), float(kp["y"]))

        # Granular per-rule landmark availability
        rule_availability: Dict[str, bool] = {}
        for rule_key, req_names in RULE_REQUIRED_LANDMARKS.items():
            rule_availability[rule_key] = all(name in reliable for name in req_names)

        return LandmarkReliabilityResult(
            status="VALID",
            is_tracking_valid=True,
            reliable_landmarks=reliable,
            unreliable_landmarks=unreliable,
            available_sides=available_sides,
            rule_availability=rule_availability,
            jump_detected=False,
            jumped_landmarks=[],
            message=None,
        )
