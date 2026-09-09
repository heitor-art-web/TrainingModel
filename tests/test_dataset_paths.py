import tempfile
import unittest
from pathlib import Path

import yaml

from training.common import materialize_dataset_yaml


class DatasetPathTests(unittest.TestCase):
    def test_materialized_dataset_path_is_absolute(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_config = project_root / "tests" / "_temp_dataset.yaml"
        temp_config.write_text(
            "path: dataset\ntrain: images/train\nval: images/val\nnames:\n  0: basketball\n",
            encoding="utf-8",
        )

        runtime_path = None
        try:
            runtime_path = materialize_dataset_yaml(temp_config)
            with runtime_path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
            self.assertTrue(Path(config["path"]).is_absolute())
            self.assertEqual(Path(config["path"]), (project_root / "dataset").resolve())
        finally:
            temp_config.unlink(missing_ok=True)
            if runtime_path is not None:
                runtime_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
