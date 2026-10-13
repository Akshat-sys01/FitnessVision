"""
Form error detection definitions and evaluation for FitnessVision.
Provides structured error representations and rule checkers for exercise form.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FormError:
    code: str
    message: str
    severity: str  # "high", "medium", "low"
    category: str  # "alignment", "range_of_motion", "stability", "visibility"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# Catalog of defined error codes
ERROR_DEFINITIONS = {
    # Visibility
    "KEYPOINTS_LOW_CONFIDENCE": FormError(
        code="KEYPOINTS_LOW_CONFIDENCE",
        message="Keypoints are partially obscured or low confidence.",
        severity="high",
        category="visibility",
    ),
    "JOINTS_OUT_OF_FRAME": FormError(
        code="JOINTS_OUT_OF_FRAME",
        message="Required body joints are outside the camera frame.",
        severity="high",
        category="visibility",
    ),
    # Squat errors
    "EXCESSIVE_FORWARD_LEAN": FormError(
        code="EXCESSIVE_FORWARD_LEAN",
        message="Torso is leaning too far forward. Keep your chest elevated.",
        severity="high",
        category="alignment",
    ),
    "INSUFFICIENT_SQUAT_DEPTH": FormError(
        code="INSUFFICIENT_SQUAT_DEPTH",
        message="Squat depth was shallow. Lower your hips until thighs are near parallel.",
        severity="medium",
        category="range_of_motion",
    ),
    "KNEE_VALGUS": FormError(
        code="KNEE_VALGUS",
        message="Knees are caving inward. Push your knees outward over your toes.",
        severity="high",
        category="alignment",
    ),
    "ASYMMETRIC_SQUAT": FormError(
        code="ASYMMETRIC_SQUAT",
        message="Uneven weight distribution between left and right legs.",
        severity="medium",
        category="alignment",
    ),
    "INCOMPLETE_SQUAT_STAND": FormError(
        code="INCOMPLETE_SQUAT_STAND",
        message="Stand all the way up and extend your hips to complete the repetition.",
        severity="low",
        category="range_of_motion",
    ),
    # Bicep curl errors
    "EXCESSIVE_ELBOW_DRIFT": FormError(
        code="EXCESSIVE_ELBOW_DRIFT",
        message="Elbows are drifting or swinging. Keep your upper arms stationary at your sides.",
        severity="high",
        category="stability",
    ),
    "TORSO_SWAY": FormError(
        code="TORSO_SWAY",
        message="Torso is swinging backward to generate momentum. Stand still and brace your core.",
        severity="high",
        category="stability",
    ),
    "INCOMPLETE_CURL_LIFT": FormError(
        code="INCOMPLETE_CURL_LIFT",
        message="Curl the weight all the way up through full contraction.",
        severity="medium",
        category="range_of_motion",
    ),
    "INCOMPLETE_CURL_EXTENSION": FormError(
        code="INCOMPLETE_CURL_EXTENSION",
        message="Lower the weight completely to full arm extension before curling again.",
        severity="medium",
        category="range_of_motion",
    ),
}


def get_error(code: str, custom_message: Optional[str] = None) -> FormError:
    """Returns a FormError for the given code, optionally overriding the message."""
    if code in ERROR_DEFINITIONS:
        base = ERROR_DEFINITIONS[code]
        if custom_message:
            return FormError(
                code=base.code,
                message=custom_message,
                severity=base.severity,
                category=base.category,
            )
        return base
    return FormError(
        code=code,
        message=custom_message or "Exercise form deviation detected.",
        severity="medium",
        category="general",
    )
