import unittest

from training.data import (
    assert_dataset_budget,
    assert_source_allowed,
    assign_sources,
    validate_collection_config,
    validate_split_config,
)
from evaluation.compare_models import promotion_check


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "split": {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42},
            "collection": {
                "max_dataset_images": 4000,
                "default_sample_fps": 1.0,
                "default_frames_per_source": 120,
                "max_frames_per_source": 200,
            },
            "sources": [
                {
                    "id": "gold",
                    "role": "gold_test",
                    "allow_training": False,
                    "allow_hard_case_training": False,
                }
            ],
        }

    def test_ratios_are_70_15_15(self):
        validate_split_config(self.registry["split"])

    def test_collection_policy_caps_dataset_at_4000(self):
        validate_collection_config(self.registry["collection"])
        assert_dataset_budget(4000, self.registry)
        with self.assertRaises(ValueError):
            assert_dataset_budget(4001, self.registry)

        invalid = dict(self.registry["collection"])
        invalid["max_dataset_images"] = 4001
        with self.assertRaises(ValueError):
            validate_collection_config(invalid)

    def test_gold_source_is_protected_and_forced_to_test(self):
        with self.assertRaises(ValueError):
            assert_source_allowed("gold", "training", self.registry)
        assignments = assign_sources(
            {"gold": 15, "a": 50, "b": 20, "c": 10, "d": 5}, self.registry
        )
        self.assertEqual(assignments["gold"], "test")

    def test_promotion_priorities(self):
        baseline = {
            "per_class": {
                "basketball": {"precision": 0.80, "recall": 0.70},
                "player": {"precision": 0.90, "recall": 0.90},
                "rim": {"precision": 0.85, "recall": 0.80},
            }
        }
        candidate = {
            "per_class": {
                "basketball": {"precision": 0.85, "recall": 0.68},
                "player": {"precision": 0.89, "recall": 0.91},
                "rim": {"precision": 0.84, "recall": 0.80},
            }
        }
        result = promotion_check(baseline, candidate)
        self.assertTrue(result["eligible_for_promotion_on_detection_metrics"])


if __name__ == "__main__":
    unittest.main()
