"""
Centralized MediaPipe Pose landmark configuration for squat analysis.
Reuses the single-source-of-truth definitions from pose_estimation.keypoints.
"""

from typing import Dict, List
from pose_estimation.keypoints import (
    MEDIAPIPE_LANDMARKS,
    INDEX_TO_LANDMARK,
    SQUAT_LANDMARK_GROUPS,
    SQUAT_LEFT_CHAIN,
    SQUAT_RIGHT_CHAIN,
    SQUAT_FOOT_LANDMARKS,
)

# Subset of MediaPipe landmark indices used for squat analysis
SQUAT_LANDMARK_INDICES: Dict[str, int] = {
    name: MEDIAPIPE_LANDMARKS[name]
    for pair in SQUAT_LANDMARK_GROUPS.values()
    for name in pair
}

# Granular landmark requirements per independent measurement / rule
RULE_REQUIRED_LANDMARKS: Dict[str, List[str]] = {
    "left_knee_angle": ["left_hip", "left_knee", "left_ankle"],
    "right_knee_angle": ["right_hip", "right_knee", "right_ankle"],
    "left_hip_angle": ["left_shoulder", "left_hip", "left_knee"],
    "right_hip_angle": ["right_shoulder", "right_hip", "right_knee"],
    "left_thigh_position": ["left_hip", "left_knee"],
    "right_thigh_position": ["right_hip", "right_knee"],
    "left_knee_alignment": ["left_knee", "left_foot_index"],
    "right_knee_alignment": ["right_knee", "right_foot_index"],
    "stance_width": ["left_shoulder", "right_shoulder", "left_ankle", "right_ankle"],
    "left_foot_orientation": ["left_heel", "left_foot_index"],
    "right_foot_orientation": ["right_heel", "right_foot_index"],
    "knee_valgus": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
    "knee_symmetry": [
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
    ],
}

__all__ = [
    "MEDIAPIPE_LANDMARKS",
    "INDEX_TO_LANDMARK",
    "SQUAT_LANDMARK_GROUPS",
    "SQUAT_LEFT_CHAIN",
    "SQUAT_RIGHT_CHAIN",
    "SQUAT_FOOT_LANDMARKS",
    "SQUAT_LANDMARK_INDICES",
    "RULE_REQUIRED_LANDMARKS",
]
