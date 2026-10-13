"""
Geometric calculations for pose analysis.
Includes 3-point joint angles, Euclidean distance, body alignment,
inclination, and symmetry metrics with robust numerical edge-case handling.
"""

import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


PointType = Union[Dict[str, Any], Tuple[float, float], Tuple[float, float, float], list, np.ndarray]


def _extract_xy(pt: Optional[PointType]) -> Optional[Tuple[float, float]]:
    """
    Safely extracts (x, y) coordinates from a dict, tuple, list, or ndarray.
    Returns None if missing, non-numeric, or containing inf/nan.
    """
    if pt is None:
        return None

    try:
        if isinstance(pt, dict):
            if "x" not in pt or "y" not in pt:
                return None
            x = float(pt["x"])
            y = float(pt["y"])
        elif isinstance(pt, (tuple, list, np.ndarray)):
            if len(pt) < 2:
                return None
            x = float(pt[0])
            y = float(pt[1])
        else:
            return None

        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None

        return (x, y)
    except (ValueError, TypeError):
        return None


def calculate_angle_3points(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    pt_c: Optional[PointType],
) -> Optional[float]:
    """
    Calculates the 2D planar joint angle ABC at joint vertex B in degrees.

    Points:
        A: Endpoint 1
        B: Joint vertex
        C: Endpoint 2

    Returns:
        float angle in degrees [0.0, 180.0], or None if points are invalid
        or either limb vector has near-zero length.
    """
    a = _extract_xy(pt_a)
    b = _extract_xy(pt_b)
    c = _extract_xy(pt_c)

    if a is None or b is None or c is None:
        return None

    # Vectors from vertex B
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])

    # Check for degenerate zero-length vectors
    if norm_ba < 1e-7 or norm_bc < 1e-7:
        return None

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    cosine = dot_product / (norm_ba * norm_bc)

    # Clamp cosine to [-1.0, 1.0] to eliminate floating point rounding errors
    cosine = max(-1.0, min(1.0, cosine))

    angle_rad = math.acos(cosine)
    angle_deg = math.degrees(angle_rad)

    return float(round(angle_deg, 2))


def calculate_euclidean_distance(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
) -> Optional[float]:
    """
    Calculates 2D Euclidean distance between two points.
    Returns None if either point is invalid.
    """
    a = _extract_xy(pt_a)
    b = _extract_xy(pt_b)

    if a is None or b is None:
        return None

    dist = math.hypot(a[0] - b[0], a[1] - b[1])
    return float(round(dist, 4))


def calculate_normalized_distance(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    reference_dist: Optional[float],
) -> Optional[float]:
    """
    Calculates Euclidean distance normalized by a reference length
    (e.g., shoulder width or torso height).
    Returns None if points or reference distance are invalid/zero.
    """
    if reference_dist is None or reference_dist < 1e-6:
        return None

    raw_dist = calculate_euclidean_distance(pt_a, pt_b)
    if raw_dist is None:
        return None

    return float(round(raw_dist / reference_dist, 4))


def calculate_displacement(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
) -> Optional[Dict[str, float]]:
    """
    Calculates horizontal (dx) and vertical (dy) displacements from A to B.
    """
    a = _extract_xy(pt_a)
    b = _extract_xy(pt_b)

    if a is None or b is None:
        return None

    dx = b[0] - a[0]
    dy = b[1] - a[1]

    return {
        "dx": float(round(dx, 4)),
        "dy": float(round(dy, 4)),
        "abs_dx": float(round(abs(dx), 4)),
        "abs_dy": float(round(abs(dy), 4)),
    }


def calculate_torso_inclination(
    shoulder: Optional[PointType],
    hip: Optional[PointType],
) -> Optional[float]:
    """
    Calculates the angle (in degrees) of the torso line (shoulder -> hip)
    relative to the vertical axis.
    An inclination of 0 means perfectly vertical torso.
    Returns angle in [0, 90] degrees, or None if invalid.
    """
    s = _extract_xy(shoulder)
    h = _extract_xy(hip)

    if s is None or h is None:
        return None

    dx = abs(s[0] - h[0])
    dy = abs(s[1] - h[1])

    if math.hypot(dx, dy) < 1e-6:
        return None

    # Angle relative to vertical
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    return float(round(angle_deg, 2))


def calculate_symmetry(
    left_metric: Optional[float],
    right_metric: Optional[float],
) -> Optional[Dict[str, float]]:
    """
    Computes absolute difference and symmetry index between left and right sides.
    """
    if left_metric is None or right_metric is None:
        return None

    diff = abs(left_metric - right_metric)
    avg = (abs(left_metric) + abs(right_metric)) / 2.0

    pct_diff = (diff / avg * 100.0) if avg > 1e-6 else 0.0

    return {
        "absolute_diff": float(round(diff, 2)),
        "percentage_diff": float(round(pct_diff, 2)),
    }


def calculate_midpoint(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
) -> Optional[Dict[str, float]]:
    """
    Calculates the 2D midpoint between two points.
    """
    a = _extract_xy(pt_a)
    b = _extract_xy(pt_b)

    if a is None or b is None:
        return None

    return {
        "x": float(round((a[0] + b[0]) / 2.0, 4)),
        "y": float(round((a[1] + b[1]) / 2.0, 4)),
    }
