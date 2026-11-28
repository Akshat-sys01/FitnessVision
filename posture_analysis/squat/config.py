"""
Configuration dataclass for the FitnessVision Squat Posture Analyzer.

Explicitly distinguishes:
1. Research-derived reference thresholds (Kwon & Kim 2022; Ahn & Kim 2023).
2. FitnessVision calibrated defaults (requiring empirical calibration per setup).
3. Experimental extensions (e.g., trunk inclination, optional 2D foot orientation).
"""

from dataclasses import dataclass
from typing import Optional


VALID_CAMERA_VIEWS = ("side", "front")
VALID_DEPTH_MODES = ("half", "deep")


@dataclass
class SquatConfig:
    """
    Configurable thresholds, camera-view settings, and rule toggles for squat analysis.
    """

    # --- Camera View Configuration ---
    camera_view: str = "side"  # "side" or "front"

    # --- Landmark Reliability & Validation (FitnessVision calibrated) ---
    min_visibility: float = 0.50
    min_presence: Optional[float] = 0.50  # Checked only when MediaPipe provides presence
    max_coordinate_jump: float = 0.35     # Max normalized jump per frame before flagging spike
    jump_recovery_frames: int = 3         # Consecutive frames before re-anchoring after jump
    min_vector_length: float = 1e-4
    enforce_normalized_bounds: bool = True
    coord_min_bound: float = -0.25
    coord_max_bound: float = 1.25

    # --- Temporal Smoothing (FitnessVision calibrated) ---
    smoothing_alpha: float = 0.65         # EMA factor in (0, 1]; 1.0 disables landmark EMA
    angle_smoothing_window: int = 4       # Rolling window size for repetition FSM

    # --- Knee-Angle Movement & Depth Thresholds (FitnessVision calibrated) ---
    # Note: Not universal biomechanical standards; calibrate per user mobility/camera.
    standing_knee_angle: float = 155.0
    descent_knee_threshold: float = 145.0
    target_depth_mode: str = "deep"       # "half" or "deep"
    half_squat_knee_angle: float = 120.0
    deep_squat_knee_angle: float = 100.0
    shallow_reversal_cutoff: float = 115.0
    return_to_standing_knee_angle: float = 150.0
    bottom_hysteresis_deg: float = 15.0
    reversal_hysteresis_deg: float = 12.0

    # --- Repetition State Machine Durations (FitnessVision calibrated) ---
    min_standing_frames: int = 1
    min_descending_frames: int = 1
    min_bottom_frames: int = 2
    min_ascending_frames: int = 1
    max_lost_frames: int = 25

    # --- Research Reference: Kwon & Kim (2022) ---
    # Hip angle range [60, 120] degrees during bottom phase (secondary feature)
    enable_hip_angle_rule: bool = True
    hip_angle_min: float = 60.0
    hip_angle_max: float = 120.0

    # Vertical coordinate difference |y_hip - y_knee| <= 0.2 during bottom (side view)
    enable_thigh_position_rule: bool = True
    max_hip_knee_vertical_diff: float = 0.20

    # Horizontal coordinate difference |x_knee - x_foot_index| <= 0.1 (side view)
    enable_knee_alignment_rule: bool = True
    side_max_knee_foot_x_diff: float = 0.10

    # --- Research Reference: Ahn & Kim (2023) ---
    # Stance width ratio: ankle_width / shoulder_width (wide ~ 1.30 - 1.50)
    enable_stance_analysis: bool = True
    standard_stance_min_ratio: float = 0.75
    standard_stance_max_ratio: float = 1.25
    wide_stance_min_ratio: float = 1.30
    wide_stance_max_ratio: float = 1.50
    min_shoulder_width: float = 0.05

    # Foot outward orientation angle [14, 45] degrees (front view, 2D approximation)
    # Disabled for posture-error flagging by default due to 2D webcam foreshortening limits
    enable_foot_orientation_rule: bool = False
    foot_angle_min_deg: float = 14.0
    foot_angle_max_deg: float = 45.0
    min_foot_vector_length: float = 0.015

    # --- FitnessVision Frontal Rules ---
    enable_knee_symmetry_rule: bool = True
    max_knee_asymmetry_deg: float = 25.0
    enable_knee_valgus_rule: bool = True
    min_knee_ankle_valgus_ratio: float = 0.65
    valgus_active_below_knee_angle: float = 140.0

    # --- Experimental Extension: Trunk Inclination ---
    # Angle is measured & stored in debug output; corrective feedback disabled by default
    # until validated against experimental ground truth.
    enable_trunk_inclination_measurement: bool = True
    enable_trunk_inclination_feedback: bool = False
    max_torso_inclination: float = 48.0

    # --- Feedback Engine Stabilization ---
    max_feedback_messages: int = 2
    min_feedback_hold_frames: int = 4

    # --- Debug Visualization ---
    debug_mode: bool = False

    def __post_init__(self):
        self.camera_view = self.camera_view.lower().strip()
        if self.camera_view not in VALID_CAMERA_VIEWS:
            raise ValueError(f"Unsupported camera_view '{self.camera_view}'. Must be one of {VALID_CAMERA_VIEWS}.")

        self.target_depth_mode = self.target_depth_mode.lower().strip()
        if self.target_depth_mode not in VALID_DEPTH_MODES:
            raise ValueError(f"Unsupported target_depth_mode '{self.target_depth_mode}'. Must be one of {VALID_DEPTH_MODES}.")

    @property
    def active_bottom_depth_threshold(self) -> float:
        """Returns the target knee-angle threshold for the selected depth mode."""
        if self.target_depth_mode == "half":
            return self.half_squat_knee_angle
        return self.deep_squat_knee_angle
