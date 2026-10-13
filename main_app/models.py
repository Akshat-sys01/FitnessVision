"""
Database models for FitnessVision.
Persists workout sessions, repetition counts, and posture error logs.
"""

from django.db import models
import uuid


class WorkoutSession(models.Model):
    """Represents an active or completed workout session."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_token = models.CharField(max_length=64, db_index=True)
    exercise_slug = models.CharField(max_length=50)
    exercise_name = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(default=0)
    repetition_count = models.IntegerField(default=0)
    incomplete_repetitions = models.IntegerField(default=0)
    total_frames_analyzed = models.IntegerField(default=0)
    errors_detected_count = models.IntegerField(default=0)
    error_breakdown = models.JSONField(default=dict, blank=True)
    completed = models.BooleanField(default=False)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"{self.exercise_name} Session ({self.repetition_count} reps) - {self.start_time.strftime('%Y-%m-%d %H:%M')}"
