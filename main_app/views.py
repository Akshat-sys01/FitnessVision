"""
Django views and API endpoints for FitnessVision.
Handles page rendering, pose analysis API, frame processing, and session results.
"""

import json
import base64
import time
import uuid
from typing import Dict, Any

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone

import numpy as np

from exercises.registry import get_exercise, get_all_exercises
from pose_estimation.keypoints import parse_keypoints
from pose_estimation.detector import get_pose_detector, MEDIAPIPE_AVAILABLE
from posture_analysis.feedback import generate_feedback
from .session_manager import session_manager
from .models import WorkoutSession

try:
    import cv2
except ImportError:
    cv2 = None


def home_view(request):
    """Landing page view."""
    return render(request, "home.html", {
        "exercises": get_all_exercises(),
    })


def exercise_selection_view(request):
    """Exercise catalog and selection view."""
    return render(request, "exercise_selection.html", {
        "exercises": get_all_exercises(),
    })


def workout_view(request, exercise_slug: str):
    """Live workout view for the selected exercise."""
    exercise = get_exercise(exercise_slug)
    if not exercise:
        return redirect("exercise_selection")

    # Generate or reuse session token in Django session
    if "workout_session_token" not in request.session:
        request.session["workout_session_token"] = str(uuid.uuid4())
    session_token = request.session["workout_session_token"]

    # Pre-initialize session state
    session_manager.get_or_create(session_token, exercise_slug)

    return render(request, "workout.html", {
        "exercise": exercise,
        "session_token": session_token,
        "mediapipe_backend_available": MEDIAPIPE_AVAILABLE,
    })


def results_view(request):
    """Workout summary and results page."""
    session_id = request.GET.get("session_id")
    session_record = None

    if session_id:
        try:
            session_record = WorkoutSession.objects.filter(id=session_id).first()
        except Exception:
            session_record = None

    if not session_record:
        # Fallback to latest session in session token
        token = request.session.get("workout_session_token")
        if token:
            session_record = WorkoutSession.objects.filter(session_token=token).first()

    return render(request, "results.html", {
        "session": session_record,
    })


@csrf_exempt
def api_analyze_pose(request):
    """
    Core API endpoint receiving structured keypoints from client.
    Performs server-side geometric feature extraction, error detection,
    and temporal repetition counting.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed. Use POST."}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"success": False, "error": "Invalid JSON payload."}, status=400)

    exercise_slug = data.get("exercise", "").lower().strip()
    if not exercise_slug or not get_exercise(exercise_slug):
        return JsonResponse({"success": False, "error": f"Unsupported or missing exercise: '{exercise_slug}'"}, status=400)

    session_token = data.get("session_token") or request.session.get("workout_session_token") or "default_session"
    raw_keypoints = data.get("keypoints", {})

    # Parse and sanitize incoming keypoints
    keypoints = parse_keypoints(raw_keypoints)

    # Fetch isolated session state
    state = session_manager.get_or_create(session_token, exercise_slug)
    state.total_frames += 1

    # 1. Exercise analysis & geometric features
    analysis_res = state.analyzer.analyze_frame(keypoints)
    tracking_valid = analysis_res.get("tracking_valid", False)
    missing_joints = analysis_res.get("missing_joints", [])
    measurements = analysis_res.get("measurements", {})
    detected_errors = analysis_res.get("detected_errors", [])

    # 2. Temporal Repetition Counter Update
    if exercise_slug == "squat":
        primary_angle = analysis_res.get("primary_knee_angle")
    else:
        primary_angle = analysis_res.get("primary_elbow_angle")

    counter_res = state.counter.update(primary_angle, tracking_valid=tracking_valid)
    rep_count = counter_res.get("rep_count", 0)
    current_phase = counter_res.get("phase", "start")
    state.rep_count = rep_count

    if detected_errors:
        state.record_errors(detected_errors)

    # 3. Corrective Feedback Generation
    feedback_data = generate_feedback(
        tracking_valid=tracking_valid,
        missing_joints=missing_joints,
        detected_errors=detected_errors,
        phase=current_phase,
        exercise_name=exercise_slug,
    )

    return JsonResponse({
        "success": True,
        "exercise": exercise_slug,
        "repetition_count": rep_count,
        "movement_phase": current_phase,
        "posture_status": feedback_data["status"],
        "detected_errors": feedback_data["error_list"],
        "feedback": feedback_data["feedback"],
        "tracking_status": "valid" if tracking_valid else "invalid",
        "measurements": measurements,
    })


@csrf_exempt
def api_analyze_frame(request):
    """
    Server-side frame analysis endpoint.
    Accepts base64 encoded image frame, runs MediaPipe Pose detector,
    and returns detected keypoints along with geometric posture analysis.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "Method not allowed. Use POST."}, status=405)

    if not MEDIAPIPE_AVAILABLE or cv2 is None:
        return JsonResponse({"success": False, "error": "Server-side MediaPipe/OpenCV not available."}, status=503)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"success": False, "error": "Invalid JSON payload."}, status=400)

    image_b64 = data.get("image")
    if not image_b64:
        return JsonResponse({"success": False, "error": "Missing image data."}, status=400)

    # Strip data URL prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return JsonResponse({"success": False, "error": "Unable to decode image frame."}, status=400)
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Image processing failed: {str(e)}"}, status=400)

    detector = get_pose_detector()
    det_res = detector.process_frame(frame_bgr)

    # Route through keypoint analysis pipeline
    data["keypoints"] = det_res["keypoints"]
    exercise_slug = data.get("exercise", "squat")
    session_token = data.get("session_token", "default_session")

    state = session_manager.get_or_create(session_token, exercise_slug)
    state.total_frames += 1

    analysis_res = state.analyzer.analyze_frame(det_res["keypoints"])
    tracking_valid = analysis_res.get("tracking_valid", False)
    missing_joints = analysis_res.get("missing_joints", [])
    measurements = analysis_res.get("measurements", {})
    detected_errors = analysis_res.get("detected_errors", [])

    primary_angle = (
        analysis_res.get("primary_knee_angle")
        if exercise_slug == "squat"
        else analysis_res.get("primary_elbow_angle")
    )
    counter_res = state.counter.update(primary_angle, tracking_valid=tracking_valid)
    rep_count = counter_res.get("rep_count", 0)
    current_phase = counter_res.get("phase", "start")
    state.rep_count = rep_count

    if detected_errors:
        state.record_errors(detected_errors)

    feedback_data = generate_feedback(
        tracking_valid=tracking_valid,
        missing_joints=missing_joints,
        detected_errors=detected_errors,
        phase=current_phase,
        exercise_name=exercise_slug,
    )

    return JsonResponse({
        "success": True,
        "pose_detected": det_res["detected"],
        "keypoints": det_res["keypoints"],
        "exercise": exercise_slug,
        "repetition_count": rep_count,
        "movement_phase": current_phase,
        "posture_status": feedback_data["status"],
        "detected_errors": feedback_data["error_list"],
        "feedback": feedback_data["feedback"],
        "tracking_status": "valid" if tracking_valid else "invalid",
        "measurements": measurements,
    })


@csrf_exempt
def api_reset_workout(request):
    """Resets the repetition counter and frame history for the current session."""
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required."}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        data = {}

    session_token = data.get("session_token") or request.session.get("workout_session_token")
    if session_token:
        session_manager.reset(session_token)

    return JsonResponse({"success": True, "message": "Workout counter reset."})


@csrf_exempt
def api_finish_workout(request):
    """Finalizes workout session, writes summary record to database, and returns results URL."""
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required."}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        data = {}

    session_token = data.get("session_token") or request.session.get("workout_session_token") or "default_session"
    exercise_slug = data.get("exercise_slug", "squat")
    client_rep_count = data.get("rep_count")
    duration_seconds = int(data.get("duration_seconds", 0))

    state = session_manager.get_session(session_token)
    rep_count = client_rep_count if client_rep_count is not None else (state.rep_count if state else 0)
    exercise_info = get_exercise(exercise_slug)
    exercise_name = exercise_info.name if exercise_info else exercise_slug.title()

    total_frames = state.total_frames if state else 0
    error_breakdown = state.error_counts if state else {}
    total_errors = sum(error_breakdown.values()) if error_breakdown else 0
    incomplete_reps = state.counter.incomplete_reps if state else 0

    session_record = WorkoutSession.objects.create(
        session_token=session_token,
        exercise_slug=exercise_slug,
        exercise_name=exercise_name,
        end_time=timezone.now(),
        duration_seconds=duration_seconds,
        repetition_count=rep_count,
        incomplete_repetitions=incomplete_reps,
        total_frames_analyzed=total_frames,
        errors_detected_count=total_errors,
        error_breakdown=error_breakdown,
        completed=True,
    )

    # Clean up in-memory state
    session_manager.remove(session_token)

    return JsonResponse({
        "success": True,
        "session_id": str(session_record.id),
        "results_url": f"/results/?session_id={session_record.id}",
    })
