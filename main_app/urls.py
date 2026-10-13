"""
URL routes for main_app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_view, name="home"),
    path("exercises/", views.exercise_selection_view, name="exercise_selection"),
    path("workout/<slug:exercise_slug>/", views.workout_view, name="workout"),
    path("results/", views.results_view, name="results"),
    # API endpoints
    path("api/analyze-pose/", views.api_analyze_pose, name="api_analyze_pose"),
    path("api/analyze-frame/", views.api_analyze_frame, name="api_analyze_frame"),
    path("api/reset-workout/", views.api_reset_workout, name="api_reset_workout"),
    path("api/finish-workout/", views.api_finish_workout, name="api_finish_workout"),
]
