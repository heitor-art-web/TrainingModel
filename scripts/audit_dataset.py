from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a YOLO detection dataset before training."
    )
    parser.add_argument(
        "--dataset",
        default="dataset.yaml",
        help="Dataset YAML relative to project root or an absolute path.",
    )
    return parser.parse_args()


def resolve(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_dataset_config(path: Path) -> tuple[dict, Path, dict[int, str]]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    root = Path(config.get("path", "dataset")).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        names = {idx: name for idx, name in enumerate(raw_names)}
    else:
        names = {int(idx): str(name) for idx, name in raw_names.items()}

    return config, root.resolve(), names


def inspect_label_file(label_path: Path, valid_classes: set[int]) -> tuple[Counter, list[str], bool]:
    counts: Counter[int] = Counter()
    errors: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return counts, errors, True

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}: line {line_number} must contain 5 values")
            continue

        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_path}: line {line_number} contains non-numeric values")
            continue

        if class_id not in valid_classes:
            errors.append(f"{label_path}: line {line_number} invalid class {class_id}")
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{label_path}: line {line_number} has values outside YOLO range")
            continue

        xmin = xc - width / 2
        xmax = xc + width / 2
        ymin = yc - height / 2
        ymax = yc + height / 2
        if xmin < 0 or ymin < 0 or xmax > 1 or ymax > 1:
            errors.append(f"{label_path}: line {line_number} bounding box crosses image bounds")
            continue

        counts[class_id] += 1

    return counts, errors, False


def audit_split(dataset_root: Path, split: str, names: dict[int, str]) -> dict:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    report = {
        "images": 0,
        "labels": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "empty_labels": 0,
        "invalid_label_entries": [],
        "objects_by_class": {name: 0 for name in names.values()},
    }

    if not images_dir.exists():
        report["status"] = f"missing images directory: {images_dir}"
        return report
    if not labels_dir.exists():
        report["status"] = f"missing labels directory: {labels_dir}"
        return report

    images = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {path.stem: path for path in labels_dir.glob("*.txt") if path.is_file()}

    report["images"] = len(images)
    report["labels"] = len(labels)
    report["missing_labels"] = sorted(str(images[key]) for key in images.keys() - labels.keys())
    report["orphan_labels"] = sorted(str(labels[key]) for key in labels.keys() - images.keys())

    for image_path in images.values():
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            report["corrupt_images"].append(f"{image_path}: {exc}")

    object_counts: Counter[int] = Counter()
    valid_classes = set(names)
    for label_path in labels.values():
        counts, errors, empty = inspect_label_file(label_path, valid_classes)
        object_counts.update(counts)
        report["invalid_label_entries"].extend(errors)
        if empty:
            report["empty_labels"] += 1

    report["objects_by_class"] = {
        names[class_id]: object_counts[class_id] for class_id in sorted(names)
    }
    report["status"] = "ok" if not (
        report["missing_labels"]
        or report["orphan_labels"]
        or report["corrupt_images"]
        or report["invalid_label_entries"]
    ) else "issues_found"

    return report


def main() -> None:
    args = parse_args()
    dataset_config = resolve(args.dataset)
    if not dataset_config.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")

    config, dataset_root, names = read_dataset_config(dataset_config)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    splits = [split for split in ("train", "val", "test") if split in config]
    report = {
        "dataset_config": str(dataset_config),
        "dataset_root": str(dataset_root),
        "classes": names,
        "splits": {split: audit_split(dataset_root, split, names) for split in splits},
    }

    output_dir = PROJECT_ROOT / "runs" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Audit report: {output_path}")

    has_issues = any(data.get("status") != "ok" for data in report["splits"].values())
    if has_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
