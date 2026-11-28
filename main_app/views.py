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


def _run_exercise_analysis(
    state,
    exercise_slug: str,
    keypoints: Dict[str, Dict[str, float]],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Shared analysis pipeline for both /api/analyze-pose/ and /api/analyze-frame/.
    """
    image_width = payload.get("image_width")
    image_height = payload.get("image_height")

    if exercise_slug == "squat":
        state.analyzer.update_settings(
            camera_view=payload.get("camera_view"),
            target_depth_mode=payload.get("target_depth") or payload.get("target_depth_mode"),
            debug_mode=payload.get("debug"),
        )
        analysis_res = state.analyzer.analyze(
            keypoints,
            image_width=image_width,
            image_height=image_height,
        )
        tracking_valid = analysis_res.get("tracking_valid", False)
        detected_errors = analysis_res.get("detected_errors", [])
        rep_count = analysis_res.get("rep_count", 0)
        movement_state = analysis_res.get("movement_state", "START")
        current_phase = movement_state.lower()
        state.rep_count = rep_count

        if detected_errors:
            state.record_errors(detected_errors)

        return {
            "exercise": exercise_slug,
            "repetition_count": rep_count,
            "movement_phase": current_phase,
            "movement_state": movement_state,
            "overall_status": analysis_res.get("overall_status"),
            "depth_classification": analysis_res.get("depth_classification"),
            "stance_classification": analysis_res.get("stance_classification"),
            "camera_view": analysis_res.get("camera_view"),
            "target_depth_mode": analysis_res.get("target_depth_mode"),
            "posture_status": analysis_res.get("legacy_posture_status", "warning"),
            "detected_errors": [e.to_dict() for e in detected_errors],
            "feedback": analysis_res.get("primary_feedback", ""),
            "feedback_messages": analysis_res.get("feedback", []),
            "rules": analysis_res.get("rules", {}),
            "landmark_reliability": analysis_res.get("landmark_reliability", {}),
            "tracking_status": "valid" if tracking_valid else "invalid",
            "measurements": analysis_res.get("measurements", {}),
        }

    # Default path for other exercises (e.g., bicep_curl)
    analysis_res = state.analyzer.analyze_frame(keypoints)
    tracking_valid = analysis_res.get("tracking_valid", False)
    missing_joints = analysis_res.get("missing_joints", [])
    measurements = analysis_res.get("measurements", {})
    detected_errors = analysis_res.get("detected_errors", [])

    primary_angle = analysis_res.get("primary_elbow_angle")
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

    return {
        "exercise": exercise_slug,
        "repetition_count": rep_count,
        "movement_phase": current_phase,
        "posture_status": feedback_data["status"],
        "detected_errors": feedback_data["error_list"],
        "feedback": feedback_data["feedback"],
        "tracking_status": "valid" if tracking_valid else "invalid",
        "measurements": measurements,
    }


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

    response_payload = _run_exercise_analysis(state, exercise_slug, keypoints, data)
    response_payload["success"] = True
    return JsonResponse(response_payload)


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

    exercise_slug = data.get("exercise", "squat").lower().strip()
    if not get_exercise(exercise_slug):
        return JsonResponse({"success": False, "error": f"Unsupported or missing exercise: '{exercise_slug}'"}, status=400)

    session_token = data.get("session_token") or request.session.get("workout_session_token") or "default_session"
    data.setdefault("image_width", det_res.get("image_width", frame_bgr.shape[1]))
    data.setdefault("image_height", det_res.get("image_height", frame_bgr.shape[0]))

    state = session_manager.get_or_create(session_token, exercise_slug)
    state.total_frames += 1

    response_payload = _run_exercise_analysis(state, exercise_slug, det_res["keypoints"], data)
    response_payload["success"] = True
    response_payload["pose_detected"] = det_res["detected"]
    response_payload["keypoints"] = det_res["keypoints"]
    return JsonResponse(response_payload)


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
