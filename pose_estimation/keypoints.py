"""
Keypoint definitions, mappings, and validation utilities for FitnessVision.
Maps standard MediaPipe 33 body landmarks to semantic names and provides
filtering for exercise-specific requirements.
"""

from typing import Dict, Any, List, Optional, Tuple

# FitnessVision canonical 16-joint order
FITNESSVISION_JOINTS = [
    "right_ankle",       # 0
    "right_knee",        # 1
    "right_hip",         # 2
    "left_hip",          # 3
    "left_knee",         # 4
    "left_ankle",        # 5
    "pelvis",            # 6
    "thorax",            # 7
    "upper_neck",        # 8
    "head_top",          # 9
    "right_wrist",       # 10
    "right_elbow",       # 11
    "right_shoulder",    # 12
    "left_shoulder",     # 13
    "left_elbow",        # 14
    "left_wrist",        # 15
]

# Only the 12 directly mapped MediaPipe landmarks
MEDIAPIPE_LANDMARKS = {
    "right_ankle": 28,
    "right_knee": 26,
    "right_hip": 24,
    "left_hip": 23,
    "left_knee": 25,
    "left_ankle": 27,
    "right_wrist": 16,
    "right_elbow": 14,
    "right_shoulder": 12,
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
}

INDEX_TO_LANDMARK = {
    index: name
    for index, name in enumerate(FITNESSVISION_JOINTS)
}

# Only connections between the selected 16 joints
POSE_CONNECTIONS = [
    # Head and neck
    ("head_top", "upper_neck"),
    ("upper_neck", "thorax"),

    # Torso
    ("thorax", "left_shoulder"),
    ("thorax", "right_shoulder"),
    ("thorax", "pelvis"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),

    # Left arm
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    # Right arm
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
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

def parse_keypoints(raw_data):
    """
    Returns only FitnessVision's canonical 16 joints.
    """

    if isinstance(raw_data, (list, tuple)):
        return build_fitnessvision_keypoints(raw_data)

    if isinstance(raw_data, dict):
        return build_fitnessvision_keypoints(raw_data)

    return {}


def build_fitnessvision_keypoints(raw_landmarks):
    """
    Converts MediaPipe's 33 landmarks into FitnessVision's
    canonical 16-joint representation.

    Only 16 joints are returned.
    Facial landmarks are used internally for derivation,
    never returned as visualization landmarks.
    """

    raw = {}

    # Read MediaPipe landmark list
    if isinstance(raw_landmarks, (list, tuple)):
        for name, index in MEDIAPIPE_LANDMARKS.items():
            if index < len(raw_landmarks):
                kp = sanitize_keypoint(raw_landmarks[index])
                if kp:
                    raw[name] = kp

        # Nose is used internally for head/neck estimation
        if len(raw_landmarks) > 0:
            nose = sanitize_keypoint(raw_landmarks[0])
            if nose:
                raw["_nose"] = nose

    elif isinstance(raw_landmarks, dict):
        for name, index in MEDIAPIPE_LANDMARKS.items():
            kp = sanitize_keypoint(
                raw_landmarks.get(index,
                    raw_landmarks.get(str(index),
                       raw_landmarks.get(name)))
                )
            
            

            if kp:
                raw[name] = kp

        nose = sanitize_keypoint(
            raw_landmarks.get("0", raw_landmarks.get("nose"))
        )
        if nose:
            raw["_nose"] = nose

    result = dict(raw)

    def midpoint(a, b):
        if a not in result or b not in result:
            return None

        p1, p2 = result[a], result[b]

        return {
            "x": (p1["x"] + p2["x"]) / 2,
            "y": (p1["y"] + p2["y"]) / 2,
            "z": (p1["z"] + p2["z"]) / 2,
            "visibility": min(
                p1["visibility"], p2["visibility"]
            )
        }

    # Joint 6: Pelvis
    pelvis = midpoint("left_hip", "right_hip")
    if pelvis:
        result["pelvis"] = pelvis

    # Joint 7: Thorax
    thorax = midpoint("left_shoulder", "right_shoulder")
    if thorax:
        result["thorax"] = thorax

    # Joint 8: Upper Neck
    # Midpoint between shoulder center and nose.
    upper_neck = midpoint("thorax", "_nose")
    if upper_neck:
        result["upper_neck"] = upper_neck

    # Joint 9: Head Top
    # Approximate extrapolation above the nose.
    if "_nose" in result and "upper_neck" in result:
        nose = result["_nose"]
        neck = result["upper_neck"]

        result["head_top"] = {
            "x": nose["x"] + 0.35 * (nose["x"] - neck["x"]),
            "y": nose["y"] + 0.35 * (nose["y"] - neck["y"]),
            "z": nose["z"],
            "visibility": min(
                nose["visibility"],
                neck["visibility"]
            )
        }

    # Return ONLY canonical 16 joints.
    return {
        name: result[name]
        for name in FITNESSVISION_JOINTS
        if name in result
    }

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
