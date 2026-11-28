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


RULE_STATUS_BGR_COLORS: Dict[str, Tuple[int, int, int]] = {
    "PASS": (90, 210, 90),         # Green
    "WARNING": (0, 175, 255),      # Amber / Orange
    "ERROR": (60, 60, 235),        # Red
    "UNAVAILABLE": (140, 140, 140), # Gray
}


def draw_squat_debug_overlay(
    image: np.ndarray,
    keypoints: Dict[str, Dict[str, float]],
    analysis_result: Dict[str, Any],
    min_visibility: float = 0.5,
) -> np.ndarray:
    """
    Optional OpenCV debug visualization for the Squat Posture Analyzer.
    Draws MediaPipe skeleton, squat landmarks, knee/hip angles, squat state,
    depth classification, rule statuses, visibility status, and feedback.
    """
    if cv2 is None or image is None:
        return image

    annotated = image.copy()
    h, w = annotated.shape[:2]

    draw_skeleton(annotated, keypoints, min_visibility=min_visibility)

    measurements = analysis_result.get("measurements", {})
    rules = analysis_result.get("rules", {})
    overall_status = analysis_result.get("overall_status", "INSUFFICIENT_DATA")
    rep_count = analysis_result.get("rep_count", 0)
    movement_state = analysis_result.get("movement_state", "START")
    depth_cls = analysis_result.get("depth_classification", "UNKNOWN")
    camera_view = analysis_result.get("camera_view", "side")
    reliability = analysis_result.get("landmark_reliability", {})
    feedback_text = analysis_result.get("primary_feedback", "")

    # Annotate left/right knee and hip angles directly near joints
    angle_annotations = [
        ("left_knee", measurements.get("left_knee_angle"), "LK"),
        ("right_knee", measurements.get("right_knee_angle"), "RK"),
        ("left_hip", measurements.get("left_hip_angle"), "LH"),
        ("right_hip", measurements.get("right_hip_angle"), "RH"),
    ]
    for joint_name, angle_val, prefix in angle_annotations:
        kp = keypoints.get(joint_name)
        if kp and angle_val is not None and kp.get("visibility", 0.0) >= min_visibility:
            px, py = int(kp["x"] * w) + 8, int(kp["y"] * h) - 6
            cv2.putText(
                annotated,
                f"{prefix}:{angle_val:.1f}",
                (px, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Top banner
    legacy_status = analysis_result.get("legacy_posture_status", "warning")
    draw_feedback_banner(
        annotated,
        status=legacy_status,
        feedback=feedback_text,
        rep_count=rep_count,
        phase=movement_state,
    )

    # Side telemetry panel for rule statuses and reliability
    panel_x, panel_y = 10, 65
    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"OVERALL: {overall_status} | VIEW: {camera_view.upper()}", (255, 255, 255)),
        (f"DEPTH: {depth_cls} | REL: {reliability.get('status', 'UNKNOWN')}", (220, 240, 255)),
    ]
    trunk = measurements.get("trunk_inclination")
    if trunk is not None:
        lines.append((f"TRUNK INCL: {trunk:.1f} deg", (200, 220, 255)))

    for rule_id, rule_info in rules.items():
        r_status = rule_info.get("status", "UNAVAILABLE")
        r_color = RULE_STATUS_BGR_COLORS.get(r_status, (140, 140, 140))
        r_meas = rule_info.get("measurement")
        lines.append((f"{rule_id}: {r_status} ({r_meas})", r_color))

    panel_h = 18 * len(lines) + 14
    overlay = annotated.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (min(w - 10, 360), panel_y + panel_h), (15, 20, 28), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    for idx, (text, color) in enumerate(lines):
        ty = panel_y + 18 + idx * 18
        cv2.putText(annotated, text, (panel_x + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    return annotated
