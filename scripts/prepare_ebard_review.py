from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = [
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
        description="Prepare a distributed manual QA sample for E-BARD backboard enrichment."
    )
    parser.add_argument("--root", default="local_data/ebard_import/sources")
    parser.add_argument("--batch", default="ebard_backboard_qa")
    parser.add_argument(
        "--per-source",
        type=int,
        default=4,
        help="Frames sampled per original game/source. Default 4 (~240 for 60 games).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def has_backboard(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if parts and int(parts[0]) == 3:
            return True
    return False


def choose_samples(source_dir: Path, per_source: int, rng: random.Random) -> list[Path]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    images = sorted(path for path in images_dir.iterdir() if path.is_file())
    predicted = [image for image in images if has_backboard(labels_dir / f"{image.stem}.txt")]
    negative = [image for image in images if image not in set(predicted)]
    rng.shuffle(predicted)
    rng.shuffle(negative)

    # Prefer enriched positives, but keep roughly one quarter no-prediction frames
    # to catch Grounding DINO backboard misses.
    negative_quota = 1 if per_source >= 4 and negative else 0
    positive_quota = max(0, per_source - negative_quota)
    selected = predicted[:positive_quota]
    selected.extend(negative[:negative_quota])

    if len(selected) < per_source:
        selected_set = set(selected)
        remainder = [image for image in images if image not in selected_set]
        rng.shuffle(remainder)
        selected.extend(remainder[: per_source - len(selected)])
    return selected[:per_source]


def main() -> None:
    args = parse_args()
    if args.per_source <= 0:
        raise ValueError("--per-source must be greater than zero.")

    root = resolve(args.root)
    if not root.exists():
        raise FileNotFoundError(f"E-BARD staging root not found: {root}")
    batch_dir = PROJECT_ROOT / "runs" / "review" / args.batch
    if batch_dir.exists():
        if not args.force:
            raise FileExistsError(f"Review batch already exists: {batch_dir}. Use --force.")
        shutil.rmtree(batch_dir)

    images_out = batch_dir / "images"
    labels_original = batch_dir / "labels_original"
    labels_reviewed = batch_dir / "labels_reviewed"
    for directory in (images_out, labels_original, labels_reviewed):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows: list[dict] = []
    source_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for source_dir in source_dirs:
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        for image_path in choose_samples(source_dir, args.per_source, rng):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            # Prefix batch files with source ID so filenames cannot collide across games.
            batch_name = f"{source_dir.name}__{image_path.name}"
            batch_stem = Path(batch_name).stem
            shutil.copy2(image_path, images_out / batch_name)
            shutil.copy2(label_path, labels_original / f"{batch_stem}.txt")
            shutil.copy2(label_path, labels_reviewed / f"{batch_stem}.txt")
            rows.append(
                {
                    "image": batch_name,
                    "source_id": source_dir.name,
                    "frame": "",
                    "time_seconds": "",
                    "reason": "ebard_backboard_qa",
                    "original_confidence": "",
                    "original_detections_json": "",
                    "review_status": "",
                    "reviewed_at": "",
                    "label_original": f"labels_original/{batch_stem}.txt",
                    "label_reviewed": f"labels_reviewed/{batch_stem}.txt",
                }
            )

    if not rows:
        raise ValueError("No E-BARD frames were available for QA sampling.")
    with (batch_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("E-BARD QA sample prepared.")
    print(f"Sources sampled: {len({row['source_id'] for row in rows})}")
    print(f"Frames sampled: {len(rows)}")
    print(f"Batch: {batch_dir}")
    print(f"Next: python3 scripts/review_labels.py {args.batch}")


if __name__ == "__main__":
    main()
