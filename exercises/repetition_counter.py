"""
Temporal repetition counter state machines for FitnessVision.
Implements robust phase-transition logic, anti-jitter smoothing,
and rejection of incomplete or fluctuating movement cycles.
"""

from typing import Dict, Any, List, Optional
from collections import deque


class BaseRepetitionCounter:
    """Base class for exercise repetition state machines."""

    def __init__(self, smoothing_window: int = 4):
        self.smoothing_window = smoothing_window
        self.angle_history: deque = deque(maxlen=smoothing_window)
        self.rep_count: int = 0
        self.incomplete_reps: int = 0
        self.current_phase: str = "start"
        self.consecutive_lost_frames: int = 0
        self.min_cycle_angle: Optional[float] = None
        self.max_cycle_angle: Optional[float] = None

    def smooth_angle(self, angle: Optional[float]) -> Optional[float]:
        """Applies moving average smoothing over recent valid angles to suppress jitter."""
        if angle is None:
            return None
        self.angle_history.append(angle)
        return float(round(sum(self.angle_history) / len(self.angle_history), 2))

    def reset(self):
        """Resets counter session state."""
        self.angle_history.clear()
        self.rep_count = 0
        self.incomplete_reps = 0
        self.current_phase = "start"
        self.consecutive_lost_frames = 0
        self.min_cycle_angle = None
        self.max_cycle_angle = None


class SquatRepetitionCounter(BaseRepetitionCounter):
    """
    Temporal state machine for squat repetitions:
    START -> STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING (Rep +1)
    """

    def __init__(
        self,
        standing_angle: float = 150.0,
        descend_trigger: float = 140.0,
        bottom_depth: float = 100.0,
        shallow_cutoff: float = 115.0,
        smoothing_window: int = 4,
    ):
        super().__init__(smoothing_window=smoothing_window)
        self.standing_angle = standing_angle
        self.descend_trigger = descend_trigger
        self.bottom_depth = bottom_depth
        self.shallow_cutoff = shallow_cutoff

    def update(
        self,
        knee_angle: Optional[float],
        tracking_valid: bool = True,
    ) -> Dict[str, Any]:
        """
        Updates the repetition state machine with the latest knee angle.

        Returns:
            dict:
                - rep_count: int
                - phase: str ("start", "standing", "descending", "bottom", "ascending")
                - new_rep_completed: bool
                - incomplete_detected: bool
        """
        new_rep_completed = False
        incomplete_detected = False

        if not tracking_valid or knee_angle is None:
            self.consecutive_lost_frames += 1
            if self.consecutive_lost_frames > 25:
                # Reset to start state if user leaves view for extended period
                self.current_phase = "start"
                self.min_cycle_angle = None
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase,
                "new_rep_completed": False,
                "incomplete_detected": False,
            }

        self.consecutive_lost_frames = 0
        angle = self.smooth_angle(knee_angle)
        if angle is None:
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase,
                "new_rep_completed": False,
                "incomplete_detected": False,
            }

        # Track minimum angle reached during descent/bottom
        if self.min_cycle_angle is None or angle < self.min_cycle_angle:
            self.min_cycle_angle = angle

        # State Transitions
        if self.current_phase == "start":
            if angle >= self.standing_angle:
                self.current_phase = "standing"
                self.min_cycle_angle = angle

        elif self.current_phase == "standing":
            if angle < self.descend_trigger:
                self.current_phase = "descending"
                self.min_cycle_angle = angle

        elif self.current_phase == "descending":
            if angle <= self.bottom_depth:
                self.current_phase = "bottom"
            elif angle > (self.min_cycle_angle + 12.0) and angle > self.shallow_cutoff:
                # User reversed upward before hitting acceptable squat depth
                self.current_phase = "standing"
                self.incomplete_reps += 1
                incomplete_detected = True
                self.min_cycle_angle = angle

        elif self.current_phase == "bottom":
            # Rising upward out of the bottom position
            if angle > (self.bottom_depth + 15.0):
                self.current_phase = "ascending"

        elif self.current_phase == "ascending":
            if angle >= self.standing_angle:
                # Successfully completed a full rep cycle!
                self.rep_count += 1
                new_rep_completed = True
                self.current_phase = "standing"
                self.min_cycle_angle = angle

        return {
            "rep_count": self.rep_count,
            "phase": self.current_phase,
            "new_rep_completed": new_rep_completed,
            "incomplete_detected": incomplete_detected,
        }


class BicepCurlRepetitionCounter(BaseRepetitionCounter):
    """
    Temporal state machine for bicep curl repetitions:
    START -> EXTENDED -> CURLING -> TOP -> LOWERING -> EXTENDED (Rep +1)
    """

    def __init__(
        self,
        extended_angle: float = 145.0,
        curl_trigger: float = 125.0,
        top_flexion: float = 65.0,
        shallow_cutoff: float = 85.0,
        smoothing_window: int = 4,
    ):
        super().__init__(smoothing_window=smoothing_window)
        self.extended_angle = extended_angle
        self.curl_trigger = curl_trigger
        self.top_flexion = top_flexion
        self.shallow_cutoff = shallow_cutoff

    def update(
        self,
        elbow_angle: Optional[float],
        tracking_valid: bool = True,
    ) -> Dict[str, Any]:
        """
        Updates the repetition state machine with the latest elbow angle.
        """
        new_rep_completed = False
        incomplete_detected = False

        if not tracking_valid or elbow_angle is None:
            self.consecutive_lost_frames += 1
            if self.consecutive_lost_frames > 25:
                self.current_phase = "start"
                self.min_cycle_angle = None
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase,
                "new_rep_completed": False,
                "incomplete_detected": False,
            }

        self.consecutive_lost_frames = 0
        angle = self.smooth_angle(elbow_angle)
        if angle is None:
            return {
                "rep_count": self.rep_count,
                "phase": self.current_phase,
                "new_rep_completed": False,
                "incomplete_detected": False,
            }

        if self.min_cycle_angle is None or angle < self.min_cycle_angle:
            self.min_cycle_angle = angle

        # State Transitions
        if self.current_phase == "start":
            if angle >= self.extended_angle:
                self.current_phase = "extended"
                self.min_cycle_angle = angle

        elif self.current_phase == "extended":
            if angle < self.curl_trigger:
                self.current_phase = "curling"
                self.min_cycle_angle = angle

        elif self.current_phase == "curling":
            if angle <= self.top_flexion:
                self.current_phase = "top"
            elif angle > (self.min_cycle_angle + 15.0) and angle > self.shallow_cutoff:
                # User aborted curl before reaching top flexion
                self.current_phase = "extended"
                self.incomplete_reps += 1
                incomplete_detected = True
                self.min_cycle_angle = angle

        elif self.current_phase == "top":
            if angle > (self.top_flexion + 20.0):
                self.current_phase = "lowering"

        elif self.current_phase == "lowering":
            if angle >= self.extended_angle:
                # Full range of motion completed!
                self.rep_count += 1
                new_rep_completed = True
                self.current_phase = "extended"
                self.min_cycle_angle = angle

        return {
            "rep_count": self.rep_count,
            "phase": self.current_phase,
            "new_rep_completed": new_rep_completed,
            "incomplete_detected": incomplete_detected,
        }


def get_counter_for_exercise(exercise_slug: str) -> BaseRepetitionCounter:
    """Factory creating an appropriate repetition counter for the exercise."""
    slug = exercise_slug.lower().strip()
    if slug == "squat":
        return SquatRepetitionCounter()
    elif slug in ("bicep_curl", "curl"):
        return BicepCurlRepetitionCounter()
    else:
        # Default fallback
        return SquatRepetitionCounter()
