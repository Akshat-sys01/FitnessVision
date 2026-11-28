"""
Squat repetition state-machine adapter wired to SquatConfig.
Reuses SquatRepetitionCounter from exercises.repetition_counter without duplication.
"""

from exercises.repetition_counter import SquatRepetitionCounter
from .config import SquatConfig


def create_squat_counter(config: SquatConfig) -> SquatRepetitionCounter:
    """Builds a SquatRepetitionCounter configured from SquatConfig."""
    return SquatRepetitionCounter(
        standing_angle=config.standing_knee_angle,
        descend_trigger=config.descent_knee_threshold,
        bottom_depth=config.active_bottom_depth_threshold,
        shallow_cutoff=config.shallow_reversal_cutoff,
        smoothing_window=config.angle_smoothing_window,
        return_to_standing_angle=config.return_to_standing_knee_angle,
        bottom_hysteresis=config.bottom_hysteresis_deg,
        reversal_hysteresis=config.reversal_hysteresis_deg,
        min_standing_frames=config.min_standing_frames,
        min_descending_frames=config.min_descending_frames,
        min_bottom_frames=config.min_bottom_frames,
        min_ascending_frames=config.min_ascending_frames,
        max_lost_frames=config.max_lost_frames,
    )


__all__ = ["SquatRepetitionCounter", "create_squat_counter"]
