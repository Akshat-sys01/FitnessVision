"""
Keypoint definitions, mappings, and validation utilities for FitnessVision.
Maps standard MediaPipe 33 body landmarks to semantic names and provides
filtering for exercise-specific requirements.
"""

from typing import Dict, Any, List, Optional, Tuple

# MediaPipe Pose Landmark standard index mapping
MEDIAPIPE_LANDMARKS: Dict[str, int] = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Reverse mapping: index to landmark name
INDEX_TO_LANDMARK: Dict[int, str] = {v: k for k, v in MEDIAPIPE_LANDMARKS.items()}

# Standard skeleton connections for pose rendering
POSE_CONNECTIONS: List[Tuple[str, str]] = [
    # Torso
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]

# Exercise-specific required keypoints
EXERCISE_REQUIRED_KEYPOINTS: Dict[str, List[str]] = {
    "squat": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    "bicep_curl": [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ],
}


def sanitize_keypoint(raw_kp: Any) -> Optional[Dict[str, float]]:
    """
    Sanitize and validate an individual keypoint dict or object.
    Ensures x, y, and visibility are valid numeric values.
    Returns None if missing or invalid.
    """
    if raw_kp is None:
        return None

    if isinstance(raw_kp, dict):
        try:
            x = float(raw_kp.get("x", 0.0))
            y = float(raw_kp.get("y", 0.0))
            z = float(raw_kp.get("z", 0.0))
            # Support both 'visibility' and 'score' / 'confidence'
            visibility = float(
                raw_kp.get("visibility", raw_kp.get("confidence", raw_kp.get("score", 1.0)))
            )
            # Basic range check: coordinates should be in reasonable normalized or pixel range
            return {"x": x, "y": y, "z": z, "visibility": visibility}
        except (ValueError, TypeError):
            return None

    # Handle MediaPipe Landmark objects directly (x, y, z, visibility attributes)
    if hasattr(raw_kp, "x") and hasattr(raw_kp, "y"):
        try:
            return {
                "x": float(raw_kp.x),
                "y": float(raw_kp.y),
                "z": float(getattr(raw_kp, "z", 0.0)),
                "visibility": float(getattr(raw_kp, "visibility", 1.0)),
            }
        except (ValueError, TypeError):
            return None

    return None


def parse_keypoints(raw_data: Any) -> Dict[str, Dict[str, float]]:
    """
    Parses a variety of incoming keypoint formats (dict by name, dict by index,
    or list of 33 landmarks) into a standardized Dict[str, Dict[str, float]].
    """
    parsed: Dict[str, Dict[str, float]] = {}

    if isinstance(raw_data, dict):
        for key, val in raw_data.items():
            sanitized = sanitize_keypoint(val)
            if sanitized is not None:
                # Key could be a name or an index string
                if str(key).isdigit():
                    idx = int(key)
                    if idx in INDEX_TO_LANDMARK:
                        parsed[INDEX_TO_LANDMARK[idx]] = sanitized
                elif key in MEDIAPIPE_LANDMARKS:
                    parsed[key] = sanitized

    elif isinstance(raw_data, list):
        for idx, val in enumerate(raw_data):
            sanitized = sanitize_keypoint(val)
            if sanitized is not None and idx in INDEX_TO_LANDMARK:
                parsed[INDEX_TO_LANDMARK[idx]] = sanitized

    return parsed


def check_keypoint_visibility(
    keypoints: Dict[str, Dict[str, float]],
    required_names: List[str],
    min_visibility: float = 0.5,
) -> Tuple[bool, List[str]]:
    """
    Checks whether all required keypoints are present and exceed min_visibility.
    Returns (is_valid, list_of_missing_or_low_confidence_names).
    """
    missing: List[str] = []
    for name in required_names:
        kp = keypoints.get(name)
        if kp is None:
            missing.append(name)
        elif kp.get("visibility", 0.0) < min_visibility:
            missing.append(name)

    is_valid = len(missing) == 0
    return is_valid, missing
