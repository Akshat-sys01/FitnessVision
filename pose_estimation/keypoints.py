"""
Keypoint definitions, mappings, and validation utilities for FitnessVision.
Maps standard MediaPipe 33 body landmarks to semantic names and provides
filtering for exercise-specific requirements.
"""

from typing import Dict, Any, List, Optional, Tuple
import math

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

# Squat-specific landmark groups (derived centrally from MEDIAPIPE_LANDMARKS)
SQUAT_LANDMARK_GROUPS: Dict[str, Tuple[str, str]] = {
    "shoulder": ("left_shoulder", "right_shoulder"),
    "hip": ("left_hip", "right_hip"),
    "knee": ("left_knee", "right_knee"),
    "ankle": ("left_ankle", "right_ankle"),
    "heel": ("left_heel", "right_heel"),
    "foot_index": ("left_foot_index", "right_foot_index"),
}

SQUAT_LEFT_CHAIN: List[str] = ["left_shoulder", "left_hip", "left_knee", "left_ankle"]
SQUAT_RIGHT_CHAIN: List[str] = ["right_shoulder", "right_hip", "right_knee", "right_ankle"]
SQUAT_FOOT_LANDMARKS: List[str] = ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]

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


def _extract_optional_presence(raw_kp: Any) -> Optional[float]:
    """Extracts presence confidence if explicitly supported and populated."""
    if isinstance(raw_kp, dict):
        val = raw_kp.get("presence")
        if val is None:
            return None
        try:
            p = float(val)
            return p if math.isfinite(p) else None
        except (ValueError, TypeError):
            return None

    # Protobuf landmark (MediaPipe Pose): check HasField before reading default 0.0
    if hasattr(raw_kp, "HasField"):
        try:
            if raw_kp.HasField("presence"):
                p = float(raw_kp.presence)
                return p if math.isfinite(p) else None
        except (ValueError, TypeError):
            return None
        return None

    val = getattr(raw_kp, "presence", None)
    if val is not None:
        try:
            p = float(val)
            return p if math.isfinite(p) else None
        except (ValueError, TypeError):
            return None
    return None


def sanitize_keypoint(raw_kp: Any) -> Optional[Dict[str, float]]:
    """
    Sanitize and validate an individual keypoint dict or object.
    Ensures x, y, z, and visibility are finite numeric values.
    Includes presence when supported and populated.
    Returns None if missing or invalid.
    """
    if raw_kp is None:
        return None

    if isinstance(raw_kp, dict):
        if "x" not in raw_kp or "y" not in raw_kp:
            return None
        try:
            x = float(raw_kp["x"])
            y = float(raw_kp["y"])
            raw_z = raw_kp.get("z", 0.0)
            z = float(0.0 if raw_z is None else raw_z)
            # Support both 'visibility' and 'score' / 'confidence'
            raw_vis = raw_kp.get("visibility", raw_kp.get("confidence", raw_kp.get("score", 1.0)))
            visibility = float(1.0 if raw_vis is None else raw_vis)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(visibility)):
                return None
            result: Dict[str, float] = {"x": x, "y": y, "z": z, "visibility": visibility}
            presence = _extract_optional_presence(raw_kp)
            if presence is not None:
                result["presence"] = presence
            return result
        except (ValueError, TypeError):
            return None

    # Handle MediaPipe Landmark objects directly (x, y, z, visibility attributes)
    if hasattr(raw_kp, "x") and hasattr(raw_kp, "y"):
        try:
            x = float(raw_kp.x)
            y = float(raw_kp.y)
            z = float(getattr(raw_kp, "z", 0.0))
            visibility = float(getattr(raw_kp, "visibility", 1.0))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(visibility)):
                return None
            result = {"x": x, "y": y, "z": z, "visibility": visibility}
            presence = _extract_optional_presence(raw_kp)
            if presence is not None:
                result["presence"] = presence
            return result
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
