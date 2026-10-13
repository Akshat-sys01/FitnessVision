"""
Corrective feedback generator for FitnessVision.
Maps detected errors and movement states into prioritized, user-friendly instructions.
"""

from typing import List, Dict, Any, Optional
from .error_detection import FormError


SEVERITY_WEIGHT = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


def generate_feedback(
    tracking_valid: bool,
    missing_joints: List[str],
    detected_errors: List[FormError],
    phase: str,
    exercise_name: str = "exercise",
) -> Dict[str, Any]:
    """
    Synthesizes overall posture status, primary feedback message, and status category.

    Categories:
        - "correct": Form is good, no errors detected.
        - "warning": Pose visibility low or movement phase transitioning.
        - "incorrect": One or more form errors detected.

    Returns:
        dict:
            - status: "correct" | "warning" | "incorrect"
            - feedback: str (primary corrective instruction)
            - error_list: List[Dict[str, str]]
    """
    # 1. Handle tracking / visibility failure first
    if not tracking_valid:
        if missing_joints:
            readable_joints = ", ".join(j.replace("_", " ") for j in missing_joints[:3])
            return {
                "status": "warning",
                "feedback": f"Adjust camera position: {readable_joints} not clearly visible.",
                "error_list": [
                    {
                        "code": "KEYPOINTS_LOW_CONFIDENCE",
                        "message": f"Joints not visible: {readable_joints}",
                    }
                ],
            }
        return {
            "status": "warning",
            "feedback": "Step back and ensure your full body is visible to the camera.",
            "error_list": [],
        }

    # 2. If form errors are present, prioritize by severity
    if detected_errors:
        sorted_errors = sorted(
            detected_errors,
            key=lambda e: SEVERITY_WEIGHT.get(e.severity, 1),
            reverse=True,
        )
        primary_error = sorted_errors[0]
        return {
            "status": "incorrect",
            "feedback": primary_error.message,
            "error_list": [e.to_dict() for e in sorted_errors],
        }

    # 3. Posture is correct: Provide contextual coaching cues based on movement phase
    phase_lower = phase.lower()
    if phase_lower in ("descending", "lowering"):
        feedback_msg = "Control the descent smoothly."
    elif phase_lower in ("bottom", "top"):
        feedback_msg = "Good depth and control. Drive back up!"
    elif phase_lower in ("ascending", "curling"):
        feedback_msg = "Power up with steady alignment."
    elif phase_lower in ("standing", "extended", "start"):
        feedback_msg = "Ready. Begin your next repetition when ready."
    else:
        feedback_msg = "Good posture! Keep going."

    return {
        "status": "correct",
        "feedback": feedback_msg,
        "error_list": [],
    }
