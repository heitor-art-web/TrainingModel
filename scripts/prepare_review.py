from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import assert_source_allowed, load_data_registry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_FIELDS = [
    "image",
    "source_id",
    "frame",
    "time_seconds",
    "reason",
    "original_confidence",
    "original_detections_json",
    "review_status",
    "reviewed_at",
    "label_original",
    "label_reviewed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a non-destructive local bounding-box review batch."
    )
    parser.add_argument("--batch", required=True, help="Batch name under runs/review/.")
    parser.add_argument("--source-id", required=True, help="Stable video/image-source ID.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hard-cases",
        help="Hard-case run directory containing frames/ and manifest.csv.",
    )
    source.add_argument("--images", help="Image directory to review.")
    parser.add_argument("--labels", help="Optional YOLO label directory for --images.")
    parser.add_argument(
        "--purpose",
        choices=("training", "evaluation"),
        default="training",
        help="Protected gold-test sources are allowed only for evaluation review.",
    )
    parser.add_argument("--registry", default="configs/data_sources.toml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_autolabel_metadata(root: Path) -> dict[str, list[dict]]:
    path = root / "autolabel_predictions.jsonl"
    if not path.exists():
        return {}
    result: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[Path(row["image"]).name] = row.get("detections", [])
    return result


def main() -> None:
    args = parse_args()
    registry = load_data_registry(args.registry)
    if args.purpose == "training":
        assert_source_allowed(args.source_id, "training", registry)

    batch_dir = PROJECT_ROOT / "runs" / "review" / args.batch
    if batch_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Review batch already exists: {batch_dir}. Use --force to replace it."
            )
        shutil.rmtree(batch_dir)

    out_images = batch_dir / "images"
    out_original = batch_dir / "labels_original"
    out_reviewed = batch_dir / "labels_reviewed"
    for directory in (out_images, out_original, out_reviewed):
        directory.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict] = []
    labels_dir: Path | None = None
    autolabel_meta: dict[str, list[dict]] = {}

    if args.hard_cases:
        hard_cases = resolve(args.hard_cases)
        frames_dir = hard_cases / "frames"
        manifest_path = hard_cases / "manifest.csv"
        if not frames_dir.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Expected frames/ and manifest.csv under {hard_cases}."
            )
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                image_value = row.get("image", "")
                image_path = hard_cases / image_value
                source_rows.append(
                    {
                        "image_path": image_path,
                        "frame": row.get("frame", ""),
                        "time_seconds": row.get("time_seconds", ""),
                        "reason": row.get("reason", ""),
                        "confidence": row.get("confidence", ""),
                    }
                )
        possible_labels = hard_cases / "labels"
        if possible_labels.exists():
            labels_dir = possible_labels
    else:
        images_dir = resolve(args.images)
        labels_dir = resolve(args.labels) if args.labels else None
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if labels_dir:
            autolabel_meta = load_autolabel_metadata(labels_dir.parent)
        for image_path in sorted(images_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                source_rows.append(
                    {
                        "image_path": image_path,
                        "frame": "",
                        "time_seconds": "",
                        "reason": "manual_sample",
                        "confidence": "",
                    }
                )

    manifest_rows: list[dict] = []
    for item in source_rows:
        image_path = Path(item["image_path"])
        if not image_path.exists():
            continue
        target_image = out_images / image_path.name
        shutil.copy2(image_path, target_image)

        original_label = out_original / f"{image_path.stem}.txt"
        reviewed_label = out_reviewed / f"{image_path.stem}.txt"
        source_label = labels_dir / f"{image_path.stem}.txt" if labels_dir else None
        if source_label and source_label.exists():
            shutil.copy2(source_label, original_label)
        else:
            original_label.write_text("", encoding="utf-8")
        shutil.copy2(original_label, reviewed_label)

        detections = autolabel_meta.get(image_path.name, [])
        confidence = item["confidence"]
        if not confidence and len(detections) == 1:
            confidence = str(detections[0].get("confidence", ""))

        manifest_rows.append(
            {
                "image": image_path.name,
                "source_id": args.source_id,
                "frame": item["frame"],
                "time_seconds": item["time_seconds"],
                "reason": item["reason"],
                "original_confidence": confidence,
                "original_detections_json": json.dumps(
                    detections, separators=(",", ":")
                ),
                "review_status": "",
                "reviewed_at": "",
                "label_original": f"labels_original/{original_label.name}",
                "label_reviewed": f"labels_reviewed/{reviewed_label.name}",
            }
        )

    with (batch_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Prepared {len(manifest_rows)} frame(s) for review.")
    print(f"Batch: {batch_dir}")
    print("Original labels are preserved; edits go only to labels_reviewed/.")


if __name__ == "__main__":
    main()
