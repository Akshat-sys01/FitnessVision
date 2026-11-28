"""
Independent rule-based posture evaluators for squat analysis.

Separates:
- Squat depth rule
- Hip angle rule (Kwon & Kim 2022 reference)
- Thigh position / hip-knee vertical offset rule (Kwon & Kim 2022 reference)
- Knee-to-foot alignment rule (Kwon & Kim 2022 side-view reference)
- Frontal knee valgus & bilateral symmetry rules (FitnessVision calibrated)
- Stance width & foot orientation rules (Ahn & Kim 2023 reference)
- Trunk inclination experimental measurement
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from posture_analysis.geometry import calculate_euclidean_distance
from .config import SquatConfig
from .feature_extractor import SquatFeatures
from .foot_analyzer import FootOrientationResult
from .landmark_validator import LandmarkReliabilityResult
from .squat_depth import DepthCategory, SquatDepthResult
from .stance_analyzer import StanceCategory, StanceResult


class RuleStatus:
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    measurement: Optional[Any]
    threshold: Optional[Any]
    status: str
    message: Optional[str]
    reliable: bool
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "measurement": self.measurement,
            "threshold": self.threshold,
            "status": self.status,
            "message": self.message,
            "reliable": self.reliable,
            "source": self.source,
        }


def _is_bottom_evaluation_phase(
    movement_state: Optional[str], depth_result: SquatDepthResult
) -> bool:
    if movement_state and movement_state.upper() == "BOTTOM":
        return True
    if movement_state is None and depth_result.classification in (
        DepthCategory.HALF_SQUAT,
        DepthCategory.DEEP_SQUAT,
    ):
        return True
    return False


def evaluate_squat_rules(
    config: SquatConfig,
    reliability: LandmarkReliabilityResult,
    features: SquatFeatures,
    depth_result: SquatDepthResult,
    stance_result: StanceResult,
    foot_result: FootOrientationResult,
    movement_state: Optional[str] = None,
    incomplete_rep_detected: bool = False,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Dict[str, RuleResult]:
    """
    Evaluates all enabled squat posture rules independently, respecting
    camera-view applicability and landmark reliability.
    """
    rules: Dict[str, RuleResult] = {}
    state_upper = movement_state.upper() if movement_state else ""
    in_bottom = _is_bottom_evaluation_phase(movement_state, depth_result)
    is_squatting_active = (
        state_upper in ("DESCENDING", "BOTTOM", "ASCENDING")
        or (
            features.primary_knee_angle is not None
            and features.primary_knee_angle < config.standing_knee_angle
        )
    )

    # 1. Squat Depth Rule
    target_thresh = config.active_bottom_depth_threshold
    if not reliability.is_tracking_valid or features.primary_knee_angle is None:
        rules["squat_depth"] = RuleResult(
            rule_id="squat_depth",
            rule_name="Squat Depth",
            measurement=None,
            threshold={"mode": config.target_depth_mode, "max_knee_angle": target_thresh},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="fitnessvision_calibrated",
        )
    elif incomplete_rep_detected:
        rules["squat_depth"] = RuleResult(
            rule_id="squat_depth",
            rule_name="Squat Depth",
            measurement=features.primary_knee_angle,
            threshold={"mode": config.target_depth_mode, "max_knee_angle": target_thresh},
            status=RuleStatus.WARNING,
            message="Try reaching the selected squat depth.",
            reliable=True,
            source="fitnessvision_calibrated",
        )
    else:
        # In half-squat mode, reaching HALF_SQUAT or DEEP_SQUAT passes; in deep mode,
        # ascending from a shallow half-squat without reaching deep passes or warns only on shallow reversal.
        rules["squat_depth"] = RuleResult(
            rule_id="squat_depth",
            rule_name="Squat Depth",
            measurement=features.primary_knee_angle,
            threshold={"mode": config.target_depth_mode, "max_knee_angle": target_thresh},
            status=RuleStatus.PASS,
            message=None,
            reliable=True,
            source="fitnessvision_calibrated",
        )

    # 2. Hip Angle Rule (Kwon & Kim 2022 reference: 60-120 deg; secondary feature -> WARNING only)
    if not config.enable_hip_angle_rule or not reliability.is_tracking_valid or features.primary_hip_angle is None:
        rules["hip_angle"] = RuleResult(
            rule_id="hip_angle",
            rule_name="Hip Angle",
            measurement=None,
            threshold={"min": config.hip_angle_min, "max": config.hip_angle_max},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="research_kwon_2022",
        )
    else:
        hip_val = features.primary_hip_angle
        if in_bottom and not (config.hip_angle_min <= hip_val <= config.hip_angle_max):
            rules["hip_angle"] = RuleResult(
                rule_id="hip_angle",
                rule_name="Hip Angle",
                measurement=hip_val,
                threshold={"min": config.hip_angle_min, "max": config.hip_angle_max},
                status=RuleStatus.WARNING,
                message="Adjust your hip position.",
                reliable=True,
                source="research_kwon_2022",
            )
        else:
            rules["hip_angle"] = RuleResult(
                rule_id="hip_angle",
                rule_name="Hip Angle",
                measurement=hip_val,
                threshold={"min": config.hip_angle_min, "max": config.hip_angle_max},
                status=RuleStatus.PASS,
                message=None,
                reliable=True,
                source="research_kwon_2022",
            )

    # 3. Thigh Position / Hip-Knee Vertical Offset (Kwon & Kim 2022 reference: <= 0.2 in side view)
    if (
        not config.enable_thigh_position_rule
        or config.camera_view != "side"
        or not reliability.is_tracking_valid
        or features.hip_knee_vertical_diff is None
    ):
        rules["thigh_position"] = RuleResult(
            rule_id="thigh_position",
            rule_name="Thigh Horizontal Position",
            measurement=None,
            threshold={"max_vertical_diff": config.max_hip_knee_vertical_diff},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="research_kwon_2022",
        )
    else:
        y_diff = features.hip_knee_vertical_diff
        if in_bottom and y_diff > config.max_hip_knee_vertical_diff:
            rules["thigh_position"] = RuleResult(
                rule_id="thigh_position",
                rule_name="Thigh Horizontal Position",
                measurement=y_diff,
                threshold={"max_vertical_diff": config.max_hip_knee_vertical_diff},
                status=RuleStatus.WARNING,
                message="Check your thigh position.",
                reliable=True,
                source="research_kwon_2022",
            )
        else:
            rules["thigh_position"] = RuleResult(
                rule_id="thigh_position",
                rule_name="Thigh Horizontal Position",
                measurement=y_diff,
                threshold={"max_vertical_diff": config.max_hip_knee_vertical_diff},
                status=RuleStatus.PASS,
                message=None,
                reliable=True,
                source="research_kwon_2022",
            )

    # 4. Knee-to-Foot Alignment (Kwon & Kim 2022 side-view horizontal threshold; independently toggleable)
    if (
        not config.enable_knee_alignment_rule
        or config.camera_view != "side"
        or not reliability.is_tracking_valid
        or features.max_knee_foot_x_diff is None
    ):
        rules["knee_alignment"] = RuleResult(
            rule_id="knee_alignment",
            rule_name="Knee-to-Foot Alignment",
            measurement=None,
            threshold={"max_x_diff": config.side_max_knee_foot_x_diff},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="research_kwon_2022",
        )
    else:
        x_diff = features.max_knee_foot_x_diff
        if is_squatting_active and x_diff > config.side_max_knee_foot_x_diff:
            rules["knee_alignment"] = RuleResult(
                rule_id="knee_alignment",
                rule_name="Knee-to-Foot Alignment",
                measurement=x_diff,
                threshold={"max_x_diff": config.side_max_knee_foot_x_diff},
                status=RuleStatus.WARNING,
                message="Check your knee and foot alignment.",
                reliable=True,
                source="research_kwon_2022",
            )
        else:
            rules["knee_alignment"] = RuleResult(
                rule_id="knee_alignment",
                rule_name="Knee-to-Foot Alignment",
                measurement=x_diff,
                threshold={"max_x_diff": config.side_max_knee_foot_x_diff},
                status=RuleStatus.PASS,
                message=None,
                reliable=True,
                source="research_kwon_2022",
            )

    # 5. Frontal Knee Valgus Check (Front view only)
    if (
        not config.enable_knee_valgus_rule
        or config.camera_view != "front"
        or not reliability.is_tracking_valid
        or not reliability.rule_availability.get("knee_valgus", False)
    ):
        rules["knee_valgus"] = RuleResult(
            rule_id="knee_valgus",
            rule_name="Frontal Knee Tracking",
            measurement=None,
            threshold={"min_knee_ankle_ratio": config.min_knee_ankle_valgus_ratio},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="fitnessvision_calibrated",
        )
    else:
        lms = features.smoothed_landmarks
        knee_dist = calculate_euclidean_distance(
            lms["left_knee"], lms["right_knee"], image_width=image_width, image_height=image_height
        )
        ankle_dist = calculate_euclidean_distance(
            lms["left_ankle"], lms["right_ankle"], image_width=image_width, image_height=image_height
        )
        if knee_dist is not None and ankle_dist is not None and ankle_dist > config.min_shoulder_width:
            ratio = round(knee_dist / ankle_dist, 4)
            active_flexion = (
                features.primary_knee_angle is not None
                and features.primary_knee_angle < config.valgus_active_below_knee_angle
            )
            if active_flexion and ratio < config.min_knee_ankle_valgus_ratio:
                rules["knee_valgus"] = RuleResult(
                    rule_id="knee_valgus",
                    rule_name="Frontal Knee Tracking",
                    measurement=ratio,
                    threshold={"min_knee_ankle_ratio": config.min_knee_ankle_valgus_ratio},
                    status=RuleStatus.ERROR,
                    message="Knees are caving inward. Push your knees outward over your toes.",
                    reliable=True,
                    source="fitnessvision_calibrated",
                )
            else:
                rules["knee_valgus"] = RuleResult(
                    rule_id="knee_valgus",
                    rule_name="Frontal Knee Tracking",
                    measurement=ratio,
                    threshold={"min_knee_ankle_ratio": config.min_knee_ankle_valgus_ratio},
                    status=RuleStatus.PASS,
                    message=None,
                    reliable=True,
                    source="fitnessvision_calibrated",
                )
        else:
            rules["knee_valgus"] = RuleResult(
                rule_id="knee_valgus",
                rule_name="Frontal Knee Tracking",
                measurement=None,
                threshold={"min_knee_ankle_ratio": config.min_knee_ankle_valgus_ratio},
                status=RuleStatus.UNAVAILABLE,
                message=None,
                reliable=False,
                source="fitnessvision_calibrated",
            )

    # 6. Bilateral Knee Symmetry (Front view only)
    if (
        not config.enable_knee_symmetry_rule
        or config.camera_view != "front"
        or not reliability.is_tracking_valid
        or features.knee_symmetry_diff is None
    ):
        rules["knee_symmetry"] = RuleResult(
            rule_id="knee_symmetry",
            rule_name="Bilateral Knee Symmetry",
            measurement=None,
            threshold={"max_asymmetry_deg": config.max_knee_asymmetry_deg},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="fitnessvision_calibrated",
        )
    else:
        diff = features.knee_symmetry_diff
        if diff > config.max_knee_asymmetry_deg:
            rules["knee_symmetry"] = RuleResult(
                rule_id="knee_symmetry",
                rule_name="Bilateral Knee Symmetry",
                measurement=diff,
                threshold={"max_asymmetry_deg": config.max_knee_asymmetry_deg},
                status=RuleStatus.WARNING,
                message="Uneven weight distribution between left and right legs.",
                reliable=True,
                source="fitnessvision_calibrated",
            )
        else:
            rules["knee_symmetry"] = RuleResult(
                rule_id="knee_symmetry",
                rule_name="Bilateral Knee Symmetry",
                measurement=diff,
                threshold={"max_asymmetry_deg": config.max_knee_asymmetry_deg},
                status=RuleStatus.PASS,
                message=None,
                reliable=True,
                source="fitnessvision_calibrated",
            )

    # 7. Stance Width (Ahn & Kim 2023; Front view optional classification)
    if not stance_result.available or stance_result.stance_ratio is None:
        rules["stance_width"] = RuleResult(
            rule_id="stance_width",
            rule_name="Stance Width",
            measurement=None,
            threshold={
                "standard": [config.standard_stance_min_ratio, config.standard_stance_max_ratio],
                "wide": [config.wide_stance_min_ratio, config.wide_stance_max_ratio],
            },
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="research_ahn_2023",
        )
    else:
        rules["stance_width"] = RuleResult(
            rule_id="stance_width",
            rule_name="Stance Width",
            measurement={
                "ratio": stance_result.stance_ratio,
                "classification": stance_result.classification,
            },
            threshold={
                "standard": [config.standard_stance_min_ratio, config.standard_stance_max_ratio],
                "wide": [config.wide_stance_min_ratio, config.wide_stance_max_ratio],
            },
            status=RuleStatus.PASS,
            message=None,
            reliable=True,
            source="research_ahn_2023",
        )

    # 8. Foot Orientation (Ahn & Kim 2023; Front view optional, conservative)
    if not config.enable_foot_orientation_rule or not foot_result.available:
        rules["foot_orientation"] = RuleResult(
            rule_id="foot_orientation",
            rule_name="Foot Orientation",
            measurement=(
                {"left": foot_result.left_foot_angle, "right": foot_result.right_foot_angle}
                if foot_result.available
                else None
            ),
            threshold={"min_deg": config.foot_angle_min_deg, "max_deg": config.foot_angle_max_deg},
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=foot_result.available,
            source="research_ahn_2023",
        )
    else:
        angles = [
            a for a in (foot_result.left_foot_angle, foot_result.right_foot_angle) if a is not None
        ]
        out_of_range = any(
            not (config.foot_angle_min_deg <= a <= config.foot_angle_max_deg) for a in angles
        )
        rules["foot_orientation"] = RuleResult(
            rule_id="foot_orientation",
            rule_name="Foot Orientation",
            measurement={"left": foot_result.left_foot_angle, "right": foot_result.right_foot_angle},
            threshold={"min_deg": config.foot_angle_min_deg, "max_deg": config.foot_angle_max_deg},
            status=RuleStatus.WARNING if out_of_range else RuleStatus.PASS,
            message="Check your foot positioning." if out_of_range else None,
            reliable=True,
            source="research_ahn_2023",
        )

    # 9. Trunk Inclination (Experimental feature: stored/displayed, feedback disabled by default)
    if not reliability.is_tracking_valid or features.trunk_inclination is None:
        rules["trunk_inclination"] = RuleResult(
            rule_id="trunk_inclination",
            rule_name="Trunk Inclination (Experimental)",
            measurement=None,
            threshold={"max_deg": config.max_torso_inclination}
            if config.enable_trunk_inclination_feedback
            else None,
            status=RuleStatus.UNAVAILABLE,
            message=None,
            reliable=False,
            source="fitnessvision_experimental",
        )
    elif (
        config.enable_trunk_inclination_feedback
        and features.trunk_inclination > config.max_torso_inclination
    ):
        rules["trunk_inclination"] = RuleResult(
            rule_id="trunk_inclination",
            rule_name="Trunk Inclination",
            measurement=features.trunk_inclination,
            threshold={"max_deg": config.max_torso_inclination},
            status=RuleStatus.ERROR,
            message="Torso is leaning too far forward. Keep your chest elevated.",
            reliable=True,
            source="fitnessvision_experimental",
        )
    else:
        rules["trunk_inclination"] = RuleResult(
            rule_id="trunk_inclination",
            rule_name="Trunk Inclination (Experimental)",
            measurement=features.trunk_inclination,
            threshold={"max_deg": config.max_torso_inclination}
            if config.enable_trunk_inclination_feedback
            else None,
            status=RuleStatus.PASS,
            message=None,
            reliable=True,
            source="fitnessvision_experimental",
        )

    return rules
