"""
Pose and skeleton visualization utilities using OpenCV.
Can annotate frames with keypoints, connections, joint angles, and workout feedback.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .keypoints import POSE_CONNECTIONS


def draw_skeleton(
    image: np.ndarray,
    keypoints: Dict[str, Dict[str, float]],
    connections: List[Tuple[str, str]] = POSE_CONNECTIONS,
    point_color: Tuple[int, int, int] = (0, 255, 127),      # Bright Lime Green in BGR
    line_color: Tuple[int, int, int] = (255, 200, 0),       # Cyan/Blue-gold in BGR
    min_visibility: float = 0.5,
) -> np.ndarray:
    """
    Renders skeleton keypoints and limb lines on an image.
    """
    if cv2 is None or image is None or not keypoints:
        return image

    h, w = image.shape[:2]

    # Draw connection lines
    for pt1_name, pt2_name in connections:
        kp1 = keypoints.get(pt1_name)
        kp2 = keypoints.get(pt2_name)
        if kp1 and kp2:
            if kp1.get("visibility", 0.0) >= min_visibility and kp2.get("visibility", 0.0) >= min_visibility:
                x1, y1 = int(kp1["x"] * w), int(kp1["y"] * h)
                x2, y2 = int(kp2["x"] * w), int(kp2["y"] * h)
                cv2.line(image, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)

    # Draw keypoint circles
    for name, kp in keypoints.items():
        if kp.get("visibility", 0.0) >= min_visibility:
            cx, cy = int(kp["x"] * w), int(kp["y"] * h)
            cv2.circle(image, (cx, cy), 5, point_color, -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def draw_feedback_banner(
    image: np.ndarray,
    status: str,
    feedback: str,
    rep_count: int,
    phase: str,
) -> np.ndarray:
    """
    Renders a modern status overlay on the image.
    """
    if cv2 is None or image is None:
        return image

    h, w = image.shape[:2]

    # Status color mapping
    status_lower = status.lower()
    if status_lower == "correct":
        banner_bg = (40, 167, 69)     # Green
    elif status_lower == "incorrect":
        banner_bg = (50, 50, 220)     # Red
    elif status_lower == "warning":
        banner_bg = (0, 165, 255)     # Orange/Yellow
    else:
        banner_bg = (80, 80, 80)      # Gray

    # Draw semi-transparent header bar
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), banner_bg, -1)
    alpha = 0.75
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Text overlay
    header_text = f"REPS: {rep_count} | PHASE: {phase.upper()} | {status.upper()}"
    cv2.putText(
        image,
        header_text,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if feedback:
        cv2.putText(
            image,
            feedback,
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return image
