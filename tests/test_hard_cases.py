import unittest

from tracking.hard_cases import classify_hard_case


class HardCaseTests(unittest.TestCase):
    def test_predicted_is_hard_case(self):
        row = {"status": "predicted", "confidence": "", "x": "10", "y": "10"}
        self.assertEqual(
            classify_hard_case(row, None, 1000.0, 0.35, 0.08),
            "predicted",
        )

    def test_low_confidence_detection(self):
        row = {"status": "detected", "confidence": "0.20", "x": "10", "y": "10"}
        self.assertEqual(
            classify_hard_case(row, None, 1000.0, 0.35, 0.08),
            "low_confidence",
        )

    def test_large_jump_detection(self):
        row = {"status": "detected", "confidence": "0.90", "x": "200", "y": "0"}
        self.assertEqual(
            classify_hard_case(row, (0.0, 0.0), 1000.0, 0.35, 0.08),
            "large_jump",
        )

    def test_normal_detection_is_not_hard_case(self):
        row = {"status": "detected", "confidence": "0.90", "x": "20", "y": "20"}
        self.assertIsNone(
            classify_hard_case(row, (0.0, 0.0), 1000.0, 0.35, 0.08)
        )


if __name__ == "__main__":
    unittest.main()
