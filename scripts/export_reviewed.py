from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED = {"corrected", "already_correct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export approved review rows into curated source directories."
    )
    parser.add_argument("batch", help="Batch name or runs/review/<batch> path.")
    parser.add_argument(
        "--destination-root",
        default="local_data/sources",
        help="Curated source root. A <source_id>/ directory is created underneath.",
    )
    return parser.parse_args()


def resolve_batch(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        direct = PROJECT_ROOT / path
        path = direct if direct.exists() else PROJECT_ROOT / "runs" / "review" / value
    return path.resolve()


def main() -> None:
    args = parse_args()
    batch_dir = resolve_batch(args.batch)
    manifest_path = batch_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Review manifest not found: {manifest_path}")

    destination_root = Path(args.destination_root).expanduser()
    if not destination_root.is_absolute():
        destination_root = PROJECT_ROOT / destination_root
    destination_root = destination_root.resolve()

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    exported = 0
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("review_status") not in APPROVED:
            continue
        by_source.setdefault(row["source_id"], []).append(row)

    for source_id, source_rows in by_source.items():
        target = destination_root / source_id
        images_dir = target / "images"
        labels_dir = target / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        existing_manifest = target / "manifest.csv"
        merged: dict[str, dict] = {}
        if existing_manifest.exists():
            with existing_manifest.open("r", encoding="utf-8", newline="") as file:
                for old in csv.DictReader(file):
                    merged[old["image"]] = old

        for row in source_rows:
            source_image = batch_dir / "images" / row["image"]
            source_label = batch_dir / row["label_reviewed"]
            if not source_image.exists() or not source_label.exists():
                continue
            shutil.copy2(source_image, images_dir / source_image.name)
            shutil.copy2(source_label, labels_dir / f"{source_image.stem}.txt")
            merged[source_image.name] = {
                "image": source_image.name,
                "source_id": source_id,
                "reason": row.get("reason", ""),
                "review_status": row.get("review_status", ""),
                "reviewed_at": row.get("reviewed_at", ""),
                "frame": row.get("frame", ""),
                "time_seconds": row.get("time_seconds", ""),
            }
            exported += 1

        with existing_manifest.open("w", encoding="utf-8", newline="") as file:
            fieldnames = [
                "image",
                "source_id",
                "reason",
                "review_status",
                "reviewed_at",
                "frame",
                "time_seconds",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged[name] for name in sorted(merged))

    print(f"Exported {exported} approved frame(s) into {destination_root}.")
    print("skip/uncertain rows were intentionally excluded.")


if __name__ == "__main__":
    main()
