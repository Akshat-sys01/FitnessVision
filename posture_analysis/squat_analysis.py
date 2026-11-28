"""
Squat posture analyzer coordinator for FitnessVision.

Coordinates:
- Landmark reliability validation (LandmarkValidator)
- Geometric feature extraction (SquatFeatureExtractor)
- Squat depth analysis (SquatDepthAnalyzer)
- Stance-width & foot-orientation analysis (StanceAnalyzer, FootAnalyzer)
- Independent posture rule evaluation (evaluate_squat_rules)
- Repetition counting state machine (SquatRepetitionCounter)
- Priority-based feedback & overall evaluation (SquatFeedbackEngine)
"""

from typing import Any, Dict, Optional

from .squat.config import SquatConfig
from .squat.landmark_validator import LandmarkValidator
from .squat.feature_extractor import SquatFeatureExtractor
from .squat.squat_depth import SquatDepthAnalyzer
from .squat.stance_analyzer import StanceAnalyzer
from .squat.foot_analyzer import FootAnalyzer
from .squat.posture_rules import evaluate_squat_rules
from .squat.feedback_engine import SquatFeedbackEngine
from .squat.repetition_counter import SquatRepetitionCounter, create_squat_counter


class SquatAnalyzer:
    """
    Modular, real-time, rule-based squat posture analyzer.
    Supports both SquatConfig and legacy constructor parameters.
    """

    def __init__(
        self,
        config: Optional[SquatConfig] = None,
        min_visibility: Optional[float] = None,
        standing_threshold: Optional[float] = None,
        depth_threshold: Optional[float] = None,
        shallow_threshold: Optional[float] = None,
        max_torso_inclination: Optional[float] = None,
        max_knee_asymmetry: Optional[float] = None,
        camera_view: Optional[str] = None,
        target_depth_mode: Optional[str] = None,
        debug_mode: Optional[bool] = None,
    ):
        cfg = config if config is not None else SquatConfig()

        if min_visibility is not None:
            cfg.min_visibility = min_visibility
        if standing_threshold is not None:
            cfg.standing_knee_angle = standing_threshold
        if depth_threshold is not None:
            cfg.deep_squat_knee_angle = depth_threshold
        if shallow_threshold is not None:
            cfg.half_squat_knee_angle = shallow_threshold
        if max_torso_inclination is not None:
            cfg.max_torso_inclination = max_torso_inclination
            cfg.enable_trunk_inclination_feedback = True
        if max_knee_asymmetry is not None:
            cfg.max_knee_asymmetry_deg = max_knee_asymmetry
        if camera_view is not None:
            cfg.camera_view = camera_view.lower().strip()
        if target_depth_mode is not None:
            cfg.target_depth_mode = target_depth_mode.lower().strip()
        if debug_mode is not None:
            cfg.debug_mode = debug_mode

        cfg.__post_init__()
        self.config = cfg

        self.validator = LandmarkValidator(self.config)
        self.feature_extractor = SquatFeatureExtractor(self.config)
        self.depth_analyzer = SquatDepthAnalyzer(self.config)
        self.stance_analyzer = StanceAnalyzer(self.config)
        self.foot_analyzer = FootAnalyzer(self.config)
        self.feedback_engine = SquatFeedbackEngine(self.config)
        self.counter: SquatRepetitionCounter = create_squat_counter(self.config)

    def update_settings(
        self,
        camera_view: Optional[str] = None,
        target_depth_mode: Optional[str] = None,
        debug_mode: Optional[bool] = None,
    ):
        """Updates runtime session settings without resetting repetition counts."""
        if camera_view is not None:
            view = camera_view.lower().strip()
            if view in ("side", "front"):
                self.config.camera_view = view
        if target_depth_mode is not None:
            mode = target_depth_mode.lower().strip()
            if mode in ("half", "deep"):
                self.config.target_depth_mode = mode
                self.counter.bottom_depth = self.config.active_bottom_depth_threshold
        if debug_mode is not None:
            self.config.debug_mode = bool(debug_mode)

    def reset(self):
        """Resets all stateful buffers (validator, smoothing, counter, feedback)."""
        self.validator.reset()
        self.feature_extractor.reset()
        self.feedback_engine.reset()
        self.counter.reset()

    def analyze(
        self,
        keypoints: Dict[str, Dict[str, float]],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        advance_counter: bool = True,
        movement_state_override: Optional[str] = None,
        rep_count_override: Optional[int] = None,
        incomplete_rep_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the complete research-based squat analysis pipeline on one frame.

        Returns the structured analysis result (including both the unified
        research schema and legacy keys for full backward compatibility).
        """
        # 1. Landmark Reliability Validation
        reliability = self.validator.validate(keypoints)

        # 2. Geometric Feature Extraction
        features = self.feature_extractor.extract(
            reliability,
            image_width=image_width,
            image_height=image_height,
        )

        # 3. Repetition State Machine Update
        if advance_counter:
            counter_res = self.counter.update(
                features.primary_knee_angle,
                tracking_valid=reliability.is_tracking_valid,
            )
            movement_state = counter_res["phase"].upper()
            rep_count = counter_res["rep_count"]
            incomplete_detected = counter_res["incomplete_detected"]
        else:
            movement_state = (movement_state_override or self.counter.current_phase).upper()
            rep_count = rep_count_override if rep_count_override is not None else self.counter.rep_count
            incomplete_detected = incomplete_rep_override

        # 4. Squat Depth Analysis
        depth_res = self.depth_analyzer.analyze(
            knee_angle=features.primary_knee_angle,
            movement_state=movement_state if advance_counter or movement_state_override else None,
            hip_knee_vertical_diff=features.hip_knee_vertical_diff,
        )

        # 5. Optional Stance & Foot Orientation Analysis (front view)
        stance_res = self.stance_analyzer.analyze(
            reliability, image_width=image_width, image_height=image_height
        )
        foot_res = self.foot_analyzer.analyze(
            reliability, image_width=image_width, image_height=image_height
        )

        features.stance_ratio = stance_res.stance_ratio
        features.stance_classification = stance_res.classification
        features.left_foot_angle = foot_res.left_foot_angle
        features.right_foot_angle = foot_res.right_foot_angle

        # 6. Independent Posture Rule Evaluation
        rules = evaluate_squat_rules(
            config=self.config,
            reliability=reliability,
            features=features,
            depth_result=depth_res,
            stance_result=stance_res,
            foot_result=foot_res,
            movement_state=movement_state if advance_counter or movement_state_override else None,
            incomplete_rep_detected=incomplete_detected,
            image_width=image_width,
            image_height=image_height,
        )

        # 7. Overall Evaluation & Priority Feedback
        eval_res = self.feedback_engine.evaluate(
            reliability=reliability,
            rules=rules,
            movement_state=movement_state,
        )

        measurements = features.to_measurements_dict()
        rules_serialized = {
            rule_id: {
                "status": res.status,
                "message": res.message,
                "measurement": res.measurement,
                "threshold": res.threshold,
                "reliable": res.reliable,
                "source": res.source,
            }
            for rule_id, res in rules.items()
        }

        return {
            "exercise": "squat",
            "overall_status": eval_res["overall_status"],
            "rep_count": rep_count,
            "movement_state": movement_state,
            "depth_classification": depth_res.classification,
            "stance_classification": stance_res.classification,
            "camera_view": self.config.camera_view,
            "target_depth_mode": self.config.target_depth_mode,
            "measurements": measurements,
            "rules": rules_serialized,
            "feedback": eval_res["feedback"],
            "primary_feedback": eval_res["primary_feedback"],
            "landmark_reliability": reliability.to_dict(),
            # Legacy compatibility fields for existing views/tests
            "tracking_valid": reliability.is_tracking_valid,
            "missing_joints": [] if reliability.is_tracking_valid else reliability.unreliable_landmarks,
            "detected_errors": eval_res["detected_errors"],
            "primary_knee_angle": features.primary_knee_angle,
            "legacy_posture_status": eval_res["legacy_status"],
        }

    def analyze_frame(
        self,
        keypoints: Dict[str, Dict[str, float]],
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Backward-compatible single-frame analysis entry point.
        Does not advance the internal repetition counter unless analyze() is used.
        """
        return self.analyze(
            keypoints,
            image_width=image_width,
            image_height=image_height,
            advance_counter=False,
        )
