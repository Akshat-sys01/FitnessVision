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


def scale_point_for_aspect(
    pt: Optional[PointType],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """
    Extracts (x, y) and, when valid positive image dimensions are provided,
    scales normalized x by (image_width / image_height) so horizontal and vertical
    distances share uniform spatial units without altering normalized y scale.
    """
    xy = _extract_xy(pt)
    if xy is None:
        return None

    if (
        image_width is not None
        and image_height is not None
        and image_width > 0
        and image_height > 0
        and math.isfinite(image_width)
        and math.isfinite(image_height)
    ):
        aspect = float(image_width) / float(image_height)
        return (xy[0] * aspect, xy[1])

    return xy


def calculate_vector_dot_product(
    vec_u: Tuple[float, float],
    vec_v: Tuple[float, float],
) -> Optional[float]:
    """Calculates 2D vector dot product u · v, rejecting non-finite inputs."""
    if vec_u is None or vec_v is None or len(vec_u) < 2 or len(vec_v) < 2:
        return None
    try:
        ux, uy = float(vec_u[0]), float(vec_u[1])
        vx, vy = float(vec_v[0]), float(vec_v[1])
    except (ValueError, TypeError):
        return None
    if not (math.isfinite(ux) and math.isfinite(uy) and math.isfinite(vx) and math.isfinite(vy)):
        return None
    return ux * vx + uy * vy


def calculate_vector_angle(
    vec_u: Tuple[float, float],
    vec_v: Tuple[float, float],
    min_norm: float = 1e-7,
) -> Optional[float]:
    """
    Calculates the angle in degrees [0.0, 180.0] between two 2D vectors using
    the clamped dot-product formula. Returns None on zero-length or invalid vectors.
    """
    dot = calculate_vector_dot_product(vec_u, vec_v)
    if dot is None:
        return None

    norm_u = math.hypot(float(vec_u[0]), float(vec_u[1]))
    norm_v = math.hypot(float(vec_v[0]), float(vec_v[1]))
    if norm_u < min_norm or norm_v < min_norm:
        return None

    cosine = dot / (norm_u * norm_v)
    cosine = max(-1.0, min(1.0, cosine))
    angle_deg = math.degrees(math.acos(cosine))
    if not math.isfinite(angle_deg):
        return None
    return float(round(angle_deg, 2))


def calculate_angle_3points(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    pt_c: Optional[PointType],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates the 2D planar joint angle ABC at joint vertex B in degrees.

    Points:
        A: Endpoint 1
        B: Joint vertex
        C: Endpoint 2

    When image_width and image_height are supplied, normalized x coordinates are
    aspect-ratio corrected before computing the angle.

    Returns:
        float angle in degrees [0.0, 180.0], or None if points are invalid
        or either limb vector has near-zero length.
    """
    a = scale_point_for_aspect(pt_a, image_width, image_height)
    b = scale_point_for_aspect(pt_b, image_width, image_height)
    c = scale_point_for_aspect(pt_c, image_width, image_height)

    if a is None or b is None or c is None:
        return None

    # Vectors from vertex B
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    return calculate_vector_angle(ba, bc)


def calculate_euclidean_distance(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates 2D Euclidean distance between two points.
    Applies aspect-ratio correction if image_width and image_height are supplied.
    Returns None if either point is invalid.
    """
    a = scale_point_for_aspect(pt_a, image_width, image_height)
    b = scale_point_for_aspect(pt_b, image_width, image_height)

    if a is None or b is None:
        return None

    dist = math.hypot(a[0] - b[0], a[1] - b[1])
    return float(round(dist, 4))


def calculate_normalized_distance(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    reference_dist: Optional[float],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates Euclidean distance normalized by a reference length
    (e.g., shoulder width or torso height).
    Returns None if points or reference distance are invalid/zero.
    """
    if reference_dist is None or not math.isfinite(reference_dist) or reference_dist < 1e-6:
        return None

    raw_dist = calculate_euclidean_distance(pt_a, pt_b, image_width=image_width, image_height=image_height)
    if raw_dist is None:
        return None

    return float(round(raw_dist / reference_dist, 4))


def calculate_vertical_difference(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
) -> Optional[float]:
    """Calculates absolute normalized vertical coordinate difference |y_a - y_b|."""
    a = _extract_xy(pt_a)
    b = _extract_xy(pt_b)
    if a is None or b is None:
        return None
    return float(round(abs(a[1] - b[1]), 4))


def calculate_horizontal_difference(
    pt_a: Optional[PointType],
    pt_b: Optional[PointType],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates absolute horizontal coordinate difference |x_a - x_b|,
    aspect-ratio corrected when image dimensions are supplied.
    """
    a = scale_point_for_aspect(pt_a, image_width, image_height)
    b = scale_point_for_aspect(pt_b, image_width, image_height)
    if a is None or b is None:
        return None
    return float(round(abs(a[0] - b[0]), 4))


def calculate_foot_orientation_angle(
    heel: Optional[PointType],
    foot_index: Optional[PointType],
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    min_vector_length: float = 0.015,
) -> Optional[float]:
    """
    Estimates the 2D image-plane angle (in degrees [0, 90]) of the heel-to-foot-index
    vector relative to the vertical (sagittal/forward) axis.
    Returns None if points are invalid or 2D projection length is too short (foreshortened).
    """
    h = scale_point_for_aspect(heel, image_width, image_height)
    f = scale_point_for_aspect(foot_index, image_width, image_height)
    if h is None or f is None:
        return None

    dx = f[0] - h[0]
    dy = f[1] - h[1]
    vec_len = math.hypot(dx, dy)
    if vec_len < min_vector_length:
        return None

    angle_deg = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-7)))
    return float(round(angle_deg, 2))


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
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates the angle (in degrees) of the torso line (shoulder -> hip)
    relative to the vertical axis.
    An inclination of 0 means perfectly vertical torso.
    Returns angle in [0, 90] degrees, or None if invalid.
    """
    s = scale_point_for_aspect(shoulder, image_width, image_height)
    h = scale_point_for_aspect(hip, image_width, image_height)

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
