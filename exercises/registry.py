"""
Exercise registry for FitnessVision.
Maintains modular metadata and class bindings for all supported exercises.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExerciseInfo:
    slug: str
    name: str
    tagline: str
    description: str
    target_muscles: List[str]
    setup_tips: List[str]
    execution_cues: List[str]
    icon_name: str
    difficulty: str


EXERCISE_CATALOG: Dict[str, ExerciseInfo] = {
    "squat": ExerciseInfo(
        slug="squat",
        name="Bodyweight Squat",
        tagline="Fundamental lower-body compound movement.",
        description=(
            "The bodyweight squat develops leg strength, hip mobility, and core stability. "
            "Proper form ensures your knees track properly and torso stays supported."
        ),
        target_muscles=["Quadriceps", "Glutes", "Hamstrings", "Core Stabilizers"],
        setup_tips=[
            "Stand facing the camera with your whole body from head to feet visible.",
            "Place feet shoulder-width apart, toes turned slightly outward.",
            "Keep your arms out in front or at your chest for balance.",
        ],
        execution_cues=[
            "Hinge at your hips and bend your knees as if sitting into a chair.",
            "Descend until your thighs are roughly parallel to the floor (around 90-100° knee angle).",
            "Keep your chest proud and prevent knees from collapsing inward.",
            "Drive through mid-foot and stand completely upright to complete the repetition.",
        ],
        icon_name="squat",
        difficulty="Beginner to Intermediate",
    ),
    "bicep_curl": ExerciseInfo(
        slug="bicep_curl",
        name="Bicep Curl",
        tagline="Targeted upper-arm isolation exercise.",
        description=(
            "The bicep curl isolates and strengthens the elbow flexor muscles. "
            "Clean form prevents torso momentum and ensures upper arms remain locked in place."
        ),
        target_muscles=["Biceps Brachii", "Brachialis", "Forearm Flexors"],
        setup_tips=[
            "Stand upright with shoulders back and feet comfortably hip-width apart.",
            "Ensure both arms from shoulders down to wrists are visible to the camera.",
            "Hold weights or maintain closed fists facing forward.",
        ],
        execution_cues=[
            "Pin your elbows to the sides of your torso and avoid swinging them forward.",
            "Flex your elbows to lift your hands toward your shoulders smoothly.",
            "Squeeze at the top of the curl without swaying your upper body.",
            "Lower the hands fully under control until arms are extended straight.",
        ],
        icon_name="bicep_curl",
        difficulty="Beginner",
    ),
}


def get_all_exercises() -> List[ExerciseInfo]:
    """Returns a list of all registered exercises."""
    return list(EXERCISE_CATALOG.values())


def get_exercise(slug: str) -> Optional[ExerciseInfo]:
    """Returns exercise information by slug, or None if not found."""
    return EXERCISE_CATALOG.get(slug.lower().strip())
