"""
Modular research-based Squat Posture Analyzer package for FitnessVision.
"""

from .config import SquatConfig
from .landmark_config import SQUAT_LANDMARK_INDICES, RULE_REQUIRED_LANDMARKS
from .landmark_validator import LandmarkValidator, LandmarkReliabilityResult
from .feature_extractor import SquatFeatureExtractor, SquatFeatures
from .squat_depth import SquatDepthAnalyzer, SquatDepthResult
from .stance_analyzer import StanceAnalyzer, StanceResult
from .foot_analyzer import FootAnalyzer, FootOrientationResult
from .posture_rules import RuleResult, RuleStatus, evaluate_squat_rules
from .feedback_engine import SquatFeedbackEngine, OverallSquatStatus

__all__ = [
    "SquatConfig",
    "SQUAT_LANDMARK_INDICES",
    "RULE_REQUIRED_LANDMARKS",
    "LandmarkValidator",
    "LandmarkReliabilityResult",
    "SquatFeatureExtractor",
    "SquatFeatures",
    "SquatDepthAnalyzer",
    "SquatDepthResult",
    "StanceAnalyzer",
    "StanceResult",
    "FootAnalyzer",
    "FootOrientationResult",
    "RuleResult",
    "RuleStatus",
    "evaluate_squat_rules",
    "SquatFeedbackEngine",
    "OverallSquatStatus",
]
