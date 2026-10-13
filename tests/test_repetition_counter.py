"""
Automated unit tests for exercise repetition state machines in FitnessVision.
"""

import unittest
from exercises.repetition_counter import (
    SquatRepetitionCounter,
    BicepCurlRepetitionCounter,
)


class TestRepetitionCounter(unittest.TestCase):
    def test_squat_complete_repetition(self):
        counter = SquatRepetitionCounter()

        # Phase sequence: Standing (170) -> Descending (130) -> Bottom (88) -> Ascending (135) -> Standing (168)
        # 1. Start / Standing
        for _ in range(3):
            res = counter.update(170.0)
        self.assertEqual(res["phase"], "standing")
        self.assertEqual(res["rep_count"], 0)

        # 2. Descending
        for _ in range(4):
            res = counter.update(130.0)
        self.assertEqual(res["phase"], "descending")
        self.assertEqual(res["rep_count"], 0)

        # 3. Bottom position
        for _ in range(4):
            res = counter.update(88.0)
        self.assertEqual(res["phase"], "bottom")
        self.assertEqual(res["rep_count"], 0)

        # 4. Ascending
        for _ in range(4):
            res = counter.update(135.0)
        self.assertEqual(res["phase"], "ascending")
        self.assertEqual(res["rep_count"], 0)

        # 5. Standing (Rep completed)
        for _ in range(4):
            res = counter.update(168.0)
        self.assertEqual(res["rep_count"], 1)
        self.assertEqual(res["phase"], "standing")

    def test_squat_incomplete_repetition_rejected(self):
        counter = SquatRepetitionCounter()

        # Standing
        for _ in range(3):
            counter.update(170.0)

        # Descending slightly but aborting before depth (reversing at 128)
        for _ in range(3):
            counter.update(128.0)
        self.assertEqual(counter.current_phase, "descending")

        # Reverse upward without reaching bottom
        for _ in range(4):
            res = counter.update(165.0)

        # Rep count should remain 0, incomplete detected
        self.assertEqual(res["rep_count"], 0)
        self.assertTrue(counter.incomplete_reps >= 1)

    def test_squat_jitter_does_not_double_count(self):
        counter = SquatRepetitionCounter()

        # User is standing with micro-jitter (168, 172, 169, 171)
        for angle in [168.0, 172.0, 169.0, 171.0, 167.0, 170.0]:
            res = counter.update(angle)
            self.assertEqual(res["rep_count"], 0)

    def test_squat_temporary_tracking_loss_tolerance(self):
        counter = SquatRepetitionCounter()

        # Get into descending phase
        for _ in range(4):
            counter.update(170.0)
        for _ in range(4):
            counter.update(130.0)
        self.assertEqual(counter.current_phase, "descending")

        # Tracking briefly drops for 2 frames
        counter.update(None, tracking_valid=False)
        counter.update(None, tracking_valid=False)

        # Phase should still be preserved
        self.assertEqual(counter.current_phase, "descending")

        # Resume tracking and reach bottom
        for _ in range(4):
            counter.update(85.0)
        self.assertEqual(counter.current_phase, "bottom")

    def test_counter_reset_clears_state(self):
        counter = SquatRepetitionCounter()
        counter.rep_count = 5
        counter.current_phase = "bottom"
        counter.reset()

        self.assertEqual(counter.rep_count, 0)
        self.assertEqual(counter.current_phase, "start")

    def test_bicep_curl_complete_repetition(self):
        counter = BicepCurlRepetitionCounter()

        # Extended (160) -> Curling (110) -> Top (55) -> Lowering (100) -> Extended (155)
        # 1. Extended
        for _ in range(3):
            res = counter.update(160.0)
        self.assertEqual(res["phase"], "extended")
        self.assertEqual(res["rep_count"], 0)

        # 2. Curling
        for _ in range(4):
            res = counter.update(110.0)
        self.assertEqual(res["phase"], "curling")

        # 3. Top flexion
        for _ in range(4):
            res = counter.update(55.0)
        self.assertEqual(res["phase"], "top")

        # 4. Lowering
        for _ in range(4):
            res = counter.update(100.0)
        self.assertEqual(res["phase"], "lowering")

        # 5. Return to full extension
        for _ in range(4):
            res = counter.update(155.0)
        self.assertEqual(res["rep_count"], 1)
        self.assertEqual(res["phase"], "extended")


if __name__ == "__main__":
    unittest.main()
