"""
Django admin configuration for FitnessVision.
"""

from django.contrib import admin
from .models import WorkoutSession


@admin.register(WorkoutSession)
class WorkoutSessionAdmin(admin.ModelAdmin):
    list_display = (
        "exercise_name",
        "repetition_count",
        "errors_detected_count",
        "duration_seconds",
        "start_time",
        "completed",
    )
    list_filter = ("exercise_slug", "completed", "start_time")
    search_fields = ("session_token", "exercise_name")
