import csv
import tempfile
import unittest
from pathlib import Path

from tracking.metrics import summarize_trajectory


class TrackingMetricsTests(unittest.TestCase):
    def test_summary_counts_and_streaks(self):
        rows = [
            [1, 0.0, "detected", 10, 10, 4, 4, 0.9, 0],
            [2, 0.1, "predicted", 11, 10, 4, 4, "", 1],
            [3, 0.2, "predicted", 12, 10, 4, 4, "", 2],
            [4, 0.3, "detected", 13, 10, 4, 4, 0.7, 0],
            [5, 0.4, "lost", "", "", "", "", "", ""],
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "frame",
                        "time_seconds",
                        "status",
                        "x",
                        "y",
                        "width",
                        "height",
                        "confidence",
                        "missed_frames",
                    ]
                )
                writer.writerows(rows)

            summary = summarize_trajectory(path)

        self.assertEqual(summary["total_frames"], 5)
        self.assertEqual(summary["detected_frames"], 2)
        self.assertEqual(summary["predicted_frames"], 2)
        self.assertEqual(summary["lost_frames"], 1)
        self.assertEqual(summary["longest_predicted_streak"], 2)
        self.assertEqual(summary["longest_continuous_track"], 4)
        self.assertAlmostEqual(summary["mean_detection_confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
