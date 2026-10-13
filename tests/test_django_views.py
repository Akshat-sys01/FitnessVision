"""
Automated tests for Django views and API routes in FitnessVision.
"""

import json
from django.test import TestCase, Client
from django.urls import reverse
from main_app.models import WorkoutSession
from tests.test_posture_analysis import make_synthetic_squat_pose


class TestDjangoViews(TestCase):
    def setUp(self):
        self.client = Client()

    def test_home_page_loads(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "FitnessVision")
        self.assertContains(response, "Train Smarter")

    def test_exercise_selection_page_loads(self):
        response = self.client.get(reverse("exercise_selection"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Bodyweight Squat")
        self.assertContains(response, "Bicep Curl")

    def test_workout_page_valid_exercise(self):
        response = self.client.get(reverse("workout", kwargs={"exercise_slug": "squat"}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Bodyweight Squat")

    def test_workout_page_invalid_exercise_redirects(self):
        response = self.client.get(reverse("workout", kwargs={"exercise_slug": "non_existent_exercise"}))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("exercise_selection"))

    def test_results_page_loads(self):
        response = self.client.get(reverse("results"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Workout Summary")

    def test_api_analyze_pose_get_rejected(self):
        response = self.client.get(reverse("api_analyze_pose"))
        self.assertEqual(response.status_code, 405)

    def test_api_analyze_pose_invalid_json(self):
        response = self.client.post(
            reverse("api_analyze_pose"),
            data="invalid-raw-string",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_api_analyze_pose_unsupported_exercise(self):
        payload = {
            "exercise": "flying_kick",
            "session_token": "test_token_123",
            "keypoints": {},
        }
        response = self.client.post(
            reverse("api_analyze_pose"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_api_analyze_pose_valid_squat_payload(self):
        pose = make_synthetic_squat_pose()
        payload = {
            "exercise": "squat",
            "session_token": "test_token_456",
            "keypoints": pose,
        }
        response = self.client.post(
            reverse("api_analyze_pose"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["exercise"], "squat")
        self.assertIn("repetition_count", data)
        self.assertIn("movement_phase", data)
        self.assertIn("posture_status", data)
        self.assertIn("feedback", data)
        self.assertIn("measurements", data)

    def test_api_finish_workout_persists_session(self):
        payload = {
            "session_token": "finish_session_test",
            "exercise_slug": "squat",
            "rep_count": 8,
            "duration_seconds": 45,
        }
        response = self.client.post(
            reverse("api_finish_workout"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("session_id", data)
        self.assertIn("results_url", data)

        # Verify database record
        saved = WorkoutSession.objects.filter(session_token="finish_session_test").first()
        self.assertIsNotNone(saved)
        self.assertEqual(saved.repetition_count, 8)
        self.assertEqual(saved.duration_seconds, 45)
        self.assertTrue(saved.completed)
