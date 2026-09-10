import unittest

from training.ebard import build_class_id_map, infer_game_id, remap_yolo_line


class EBARDTests(unittest.TestCase):
    def test_class_mapping_uses_hoop_as_rim(self):
        mapping = build_class_id_map(["basketball", "hoop", "player", "referee"])
        self.assertEqual(mapping[0], 0)
        self.assertEqual(mapping[1], 4)
        self.assertEqual(mapping[2], 1)
        self.assertEqual(mapping[3], 2)

    def test_plural_class_names_are_supported(self):
        mapping = build_class_id_map(
            {0: "Players", 1: "Referees", 2: "Hoops", 3: "Basketballs"}
        )
        self.assertEqual(mapping, {0: 1, 1: 2, 2: 4, 3: 0})

    def test_yolo_line_is_remapped_without_changing_box(self):
        line = remap_yolo_line("1 0.5 0.4 0.2 0.1", {1: 4})
        self.assertEqual(line, "4 0.5 0.4 0.2 0.1")

    def test_named_classes_are_supported(self):
        self.assertEqual(
            remap_yolo_line("player 0.5 0.4 0.2 0.1", {}),
            "1 0.5 0.4 0.2 0.1",
        )
        self.assertEqual(
            remap_yolo_line("basketball 0.5 0.4 0.2 0.1", {}),
            "0 0.5 0.4 0.2 0.1",
        )
        self.assertEqual(
            remap_yolo_line("referee 0.5 0.4 0.2 0.1", {}),
            "2 0.5 0.4 0.2 0.1",
        )
        self.assertEqual(
            remap_yolo_line("hoop 0.5 0.4 0.2 0.1", {}),
            "4 0.5 0.4 0.2 0.1",
        )

    def test_auto_game_id_strips_frame_suffix(self):
        self.assertEqual(infer_game_id("game_ABC_frame_0012"), "game_ABC")
        self.assertEqual(infer_game_id("game_ABC-12"), "game_ABC")

    def test_source_regex_can_override_auto_detection(self):
        value = infer_game_id(
            "nba_2025_BOS_LAL_clip17_frame09",
            r"^(?P<game>nba_2025_BOS_LAL)_",
        )
        self.assertEqual(value, "nba_2025_BOS_LAL")


if __name__ == "__main__":
    unittest.main()
