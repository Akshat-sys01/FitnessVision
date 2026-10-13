"""
MediaPipe Pose detector wrapper for FitnessVision.
Extracts 33 body landmarks from RGB/BGR images using pretrained BlazePose.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    cv2 = None

from .keypoints import INDEX_TO_LANDMARK


class PoseDetector:
    """
    Wrapper around MediaPipe Pose for extracting normalized keypoints.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None

        if MEDIAPIPE_AVAILABLE and mp is not None:
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def process_frame(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Processes a BGR image frame (OpenCV format) and extracts 33 keypoints.

        Returns:
            dict with keys:
                - detected: bool
                - keypoints: Dict[str, Dict[str, float]]
                - raw_landmarks: original MediaPipe landmarks object if available
        """
        if not MEDIAPIPE_AVAILABLE or self._pose is None or image_bgr is None:
            return {"detected": False, "keypoints": {}, "raw_landmarks": None}

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # MediaPipe requires writeable=False for performance
        image_rgb.flags.writeable = False
        results = self._pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return {"detected": False, "keypoints": {}, "raw_landmarks": None}

        keypoints: Dict[str, Dict[str, float]] = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in INDEX_TO_LANDMARK:
                name = INDEX_TO_LANDMARK[idx]
                keypoints[name] = {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 1.0)),
                }

        return {
            "detected": True,
            "keypoints": keypoints,
            "raw_landmarks": results.pose_landmarks,
        }

    def close(self):
        """Releases MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None


# Module-level singleton instance for shared backend frame processing
_global_detector: Optional[PoseDetector] = None


def get_pose_detector() -> PoseDetector:
    """Returns a shared PoseDetector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _global_detector
