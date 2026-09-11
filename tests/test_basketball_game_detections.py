import tempfile
import unittest
from pathlib import Path

from training.basketball_game_detections import (
    YoloObject,
    _resolve_split_path,
    ball_size_bucket,
    box_iou,
    build_class_map,
    infer_source_group,
    parse_and_remap_labels,
)


class BasketballGameDetectionsTests(unittest.TestCase):
    def test_class_mapping_and_ignored_events(self):
        mapping, ignored = build_class_map(
            ["Ball", "Player", "Ref", "Hoop", "FG Attempt", "FG Made"]
        )
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 4})
        self.assertEqual(ignored, {4, 5})

    def test_hoop_is_canonical_rim(self):
        mapping, ignored = build_class_map(
            {0: "Player", 1: "Ball", 2: "Hoop", 3: "Ref"}
        )
        self.assertEqual(mapping[2], 4)
        self.assertFalse(ignored)

    def test_parse_labels_ignores_event_classes(self):
        mapping, ignored = build_class_map(
            ["Ball", "Player", "Ref", "Hoop", "FG Attempt", "FG Made"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text(
                "0 0.5 0.5 0.02 0.02\n"
                "1 0.5 0.5 0.20 0.50\n"
                "3 0.8 0.4 0.10 0.06\n"
                "4 0.5 0.5 0.20 0.20\n",
                encoding="utf-8",
            )
            objects = parse_and_remap_labels(path, mapping, ignored)
        self.assertEqual([obj.class_id for obj in objects], [0, 1, 4])

    def test_ball_size_buckets(self):
        self.assertEqual(ball_size_bucket([]), "none")
        self.assertEqual(ball_size_bucket([YoloObject(0, 0.5, 0.5, 0.01, 0.01)]), "tiny")
        self.assertEqual(ball_size_bucket([YoloObject(0, 0.5, 0.5, 0.03, 0.03)]), "small")
        self.assertEqual(ball_size_bucket([YoloObject(0, 0.5, 0.5, 0.06, 0.06)]), "medium")
        self.assertEqual(ball_size_bucket([YoloObject(0, 0.5, 0.5, 0.10, 0.10)]), "large")

    def test_iou(self):
        self.assertAlmostEqual(box_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)), 1.0)
        self.assertEqual(box_iou((0.0, 0.0, 0.2, 0.2), (0.8, 0.8, 1.0, 1.0)), 0.0)

    def test_source_group_requires_deliberate_regex(self):
        self.assertIsNone(infer_source_group("game_001_frame_002"))
        self.assertEqual(
            infer_source_group(
                "game_001_frame_002", r"^(?P<source>game_\d+)_frame_\d+$"
            ),
            "game_001",
        )

    def test_resolve_roboflow_parent_relative_split_inside_export_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Basketball Game Detections.v9i.yolo26"
            images = root / "train" / "images"
            images.mkdir(parents=True)
            yaml_path = root / "data.yaml"
            yaml_path.write_text("train: ../train/images\n", encoding="utf-8")

            resolved = _resolve_split_path(
                yaml_path,
                {"train": "../train/images"},
                "../train/images",
            )

            self.assertEqual(resolved, images.resolve())


if __name__ == "__main__":
    unittest.main()
