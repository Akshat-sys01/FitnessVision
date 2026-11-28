"""
Priority-based feedback engine and overall squat evaluation layer.

Converts independent rule results into:
- Overall squat classification (CORRECT, NEEDS_ADJUSTMENT, INCORRECT, INSUFFICIENT_DATA)
- Stabilized, prioritized corrective feedback messages (preventing rapid flicker)
- Legacy-compatible FormError list and status strings for FitnessVision APIs
"""

from typing import Dict, List, Optional, Tuple
from posture_analysis.error_detection import FormError, get_error
from .config import SquatConfig
from .landmark_validator import LandmarkReliabilityResult
from .posture_rules import RuleResult, RuleStatus


class OverallSquatStatus:
    CORRECT = "CORRECT"
    NEEDS_ADJUSTMENT = "NEEDS_ADJUSTMENT"
    INCORRECT = "INCORRECT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# Lower number = higher display priority when multiple rules fail simultaneously
RULE_PRIORITY: Dict[str, int] = {
    "knee_valgus": 1,
    "trunk_inclination": 2,
    "squat_depth": 3,
    "knee_alignment": 4,
    "thigh_position": 5,
    "hip_angle": 6,
    "knee_symmetry": 7,
    "foot_orientation": 8,
    "stance_width": 9,
}

RULE_TO_ERROR_CODE: Dict[str, str] = {
    "squat_depth": "INSUFFICIENT_SQUAT_DEPTH",
    "hip_angle": "HIP_ANGLE_DEVIATION",
    "thigh_position": "THIGH_POSITION_DEVIATION",
    "knee_alignment": "KNEE_FOOT_ALIGNMENT",
    "knee_valgus": "KNEE_VALGUS",
    "knee_symmetry": "ASYMMETRIC_SQUAT",
    "foot_orientation": "FOOT_ORIENTATION_DEVIATION",
    "trunk_inclination": "EXCESSIVE_FORWARD_LEAN",
}


class SquatFeedbackEngine:
    """
    Synthesizes overall squat evaluation and anti-flicker prioritized feedback.
    """

    def __init__(self, config: Optional[SquatConfig] = None):
        self.config = config or SquatConfig()
        self._active_feedback: List[str] = []
        self._active_severity: int = 0
        self._hold_frames_remaining: int = 0
        self._last_emitted_message: Optional[str] = None

    def reset(self):
        """Resets feedback hold memory."""
        self._active_feedback = []
        self._active_severity = 0
        self._hold_frames_remaining = 0
        self._last_emitted_message = None

    def evaluate(
        self,
        reliability: LandmarkReliabilityResult,
        rules: Dict[str, RuleResult],
        movement_state: str,
    ) -> Dict[str, object]:
        """
        Produces overall squat status, prioritized feedback list, and legacy HUD fields.
        Missing/UNAVAILABLE optional rules never trigger posture failure.
        """
        if not reliability.is_tracking_valid:
            vis_msg = reliability.message or "Please ensure your full body is visible to the camera."
            self._active_feedback = [vis_msg]
            self._active_severity = 3
            self._hold_frames_remaining = 0
            return {
                "overall_status": OverallSquatStatus.INSUFFICIENT_DATA,
                "legacy_status": "warning",
                "feedback": [vis_msg],
                "primary_feedback": vis_msg,
                "detected_errors": [get_error("KEYPOINTS_LOW_CONFIDENCE", custom_message=vis_msg)],
                "passed_rules": [],
                "warning_rules": [],
                "error_rules": [],
                "unavailable_rules": list(rules.keys()),
            }

        passed_rules: List[str] = []
        warning_rules: List[RuleResult] = []
        error_rules: List[RuleResult] = []
        unavailable_rules: List[str] = []

        for rule_id, res in rules.items():
            if res.status == RuleStatus.PASS:
                passed_rules.append(rule_id)
            elif res.status == RuleStatus.WARNING:
                warning_rules.append(res)
            elif res.status == RuleStatus.ERROR:
                error_rules.append(res)
            else:
                unavailable_rules.append(rule_id)

        # Sort failures by severity then biomechanical priority
        error_rules.sort(key=lambda r: RULE_PRIORITY.get(r.rule_id, 99))
        warning_rules.sort(key=lambda r: RULE_PRIORITY.get(r.rule_id, 99))

        if error_rules:
            overall_status = OverallSquatStatus.INCORRECT
            legacy_status = "incorrect"
            current_severity = 2
        elif warning_rules:
            overall_status = OverallSquatStatus.NEEDS_ADJUSTMENT
            legacy_status = "warning"
            current_severity = 1
        else:
            overall_status = OverallSquatStatus.CORRECT
            legacy_status = "correct"
            current_severity = 0

        # Build prioritized corrective feedback messages
        raw_messages: List[str] = []
        detected_errors: List[FormError] = []

        for r in error_rules + warning_rules:
            if r.message and r.message not in raw_messages:
                raw_messages.append(r.message)
            err_code = RULE_TO_ERROR_CODE.get(r.rule_id, r.rule_id.upper())
            err = get_error(err_code, custom_message=r.message)
            if r.status == RuleStatus.WARNING and err.severity == "high":
                err = FormError(
                    code=err.code,
                    message=err.message,
                    severity="medium",
                    category=err.category,
                )
            detected_errors.append(err)

        capped_messages = raw_messages[: self.config.max_feedback_messages]

        # Anti-flicker stabilization: hold corrective cues for configured frames
        # unless a higher-severity issue arises.
        if capped_messages:
            if (
                self._hold_frames_remaining > 0
                and self._active_feedback
                and current_severity <= self._active_severity
            ):
                stabilized_messages = list(self._active_feedback)
                self._hold_frames_remaining -= 1
            else:
                stabilized_messages = capped_messages
                self._active_feedback = list(capped_messages)
                self._active_severity = current_severity
                self._hold_frames_remaining = max(0, self.config.min_feedback_hold_frames - 1)
        else:
            if self._hold_frames_remaining > 0 and self._active_feedback:
                stabilized_messages = list(self._active_feedback)
                self._hold_frames_remaining -= 1
            else:
                stabilized_messages = []
                self._active_feedback = []
                self._active_severity = 0

        primary_feedback = (
            stabilized_messages[0]
            if stabilized_messages
            else self._phase_coaching_cue(movement_state)
        )
        is_repeated = primary_feedback == self._last_emitted_message
        self._last_emitted_message = primary_feedback

        return {
            "overall_status": overall_status,
            "legacy_status": legacy_status,
            "feedback": stabilized_messages,
            "primary_feedback": primary_feedback,
            "is_repeated_feedback": is_repeated,
            "detected_errors": detected_errors,
            "passed_rules": passed_rules,
            "warning_rules": [r.rule_id for r in warning_rules],
            "error_rules": [r.rule_id for r in error_rules],
            "unavailable_rules": unavailable_rules,
        }

    @staticmethod
    def _phase_coaching_cue(movement_state: str) -> str:
        state_lower = (movement_state or "").lower()
        if state_lower == "descending":
            return "Control the descent smoothly."
        if state_lower == "bottom":
            return "Good depth and control. Drive back up!"
        if state_lower == "ascending":
            return "Power up with steady alignment."
        if state_lower in ("standing", "start"):
            return "Ready. Begin your next repetition when ready."
        return "Good posture! Keep going."
