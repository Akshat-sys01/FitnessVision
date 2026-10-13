"""
Session and state management for active workouts in FitnessVision.
Maintains isolated repetition counters, error tallies, and analyzers per session token.
"""

from typing import Dict, Any, Optional
import time
from posture_analysis.squat_analysis import SquatAnalyzer
from posture_analysis.curl_analysis import BicepCurlAnalyzer
from exercises.repetition_counter import (
    BaseRepetitionCounter,
    SquatRepetitionCounter,
    BicepCurlRepetitionCounter,
)


class ActiveWorkoutState:
    """Encapsulates the live state of an in-progress workout session."""

    def __init__(self, session_token: str, exercise_slug: str):
        self.session_token = session_token
        self.exercise_slug = exercise_slug.lower().strip()
        self.created_at = time.time()
        self.last_active = time.time()

        # Instantiate exercise-specific analyzer
        if self.exercise_slug == "squat":
            self.analyzer = SquatAnalyzer()
            self.counter: BaseRepetitionCounter = SquatRepetitionCounter()
        elif self.exercise_slug in ("bicep_curl", "curl"):
            self.analyzer = BicepCurlAnalyzer()
            self.counter = BicepCurlRepetitionCounter()
        else:
            self.analyzer = SquatAnalyzer()
            self.counter = SquatRepetitionCounter()

        # Cumulative statistics
        self.total_frames = 0
        self.error_counts: Dict[str, int] = {}
        self.error_details: Dict[str, str] = {}
        self.rep_count = 0

    def touch(self):
        self.last_active = time.time()

    def record_errors(self, detected_errors: list):
        for err in detected_errors:
            code = err.code if hasattr(err, "code") else err.get("code")
            msg = err.message if hasattr(err, "message") else err.get("message", "")
            self.error_counts[code] = self.error_counts.get(code, 0) + 1
            if code not in self.error_details:
                self.error_details[code] = msg

    def reset(self):
        self.counter.reset()
        self.total_frames = 0
        self.error_counts.clear()
        self.error_details.clear()
        self.rep_count = 0
        self.created_at = time.time()
        self.touch()


class WorkoutSessionManager:
    """Manages workout states in-memory with session isolation."""

    def __init__(self):
        self._sessions: Dict[str, ActiveWorkoutState] = {}

    def get_or_create(self, session_token: str, exercise_slug: str) -> ActiveWorkoutState:
        # Check if session exists and matches exercise
        if session_token in self._sessions:
            state = self._sessions[session_token]
            if state.exercise_slug == exercise_slug.lower().strip():
                state.touch()
                return state

        # Create new or replace
        state = ActiveWorkoutState(session_token, exercise_slug)
        self._sessions[session_token] = state
        return state

    def get_session(self, session_token: str) -> Optional[ActiveWorkoutState]:
        return self._sessions.get(session_token)

    def reset(self, session_token: str):
        if session_token in self._sessions:
            self._sessions[session_token].reset()

    def remove(self, session_token: str) -> Optional[ActiveWorkoutState]:
        return self._sessions.pop(session_token, None)


# Singleton session manager
session_manager = WorkoutSessionManager()
