from __future__ import annotations

import argparse
from collections import defaultdict
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
        description=(
            "Prepare a stratified QA sample from staged Basketball Game Detections data."
        )
    )
    parser.add_argument("--root", default="local_data/bgd_import/sources")
    parser.add_argument("--batch", default="bgd_qa")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=320,
        help="Total QA frames across all staged sources. Default 320.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_candidates(root: Path) -> list[dict]:
    candidates: list[dict] = []
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = source_dir / "manifest.csv"
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                image_path = source_dir / "images" / row["image"]
                label_path = source_dir / row.get("label", f"labels/{Path(row['image']).stem}.txt")
                if image_path.exists() and label_path.exists():
                    candidates.append(
                        {
                            "source_id": source_dir.name,
                            "image_path": image_path,
                            "label_path": label_path,
                            "bucket": row.get("selection_bucket", "other") or "other",
                            "score": int(float(row.get("selection_score", 0) or 0)),
                        }
                    )
    return candidates


def choose_samples(candidates: list[dict], sample_size: int, seed: int) -> list[dict]:
    sample_size = min(sample_size, len(candidates))
    rng = random.Random(seed)
    fractions = {
        "model_fn": 0.25,
        "model_fp": 0.15,
        "tiny_ball": 0.20,
        "small_ball": 0.15,
        "no_ball_hard_negative": 0.10,
        "ball_and_rim": 0.10,
        "other": 0.05,
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        grouped[item["bucket"]].append(item)
    for items in grouped.values():
        rng.shuffle(items)
        items.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict] = []
    selected_keys: set[tuple[str, str]] = set()
    for bucket, fraction in fractions.items():
        quota = int(round(sample_size * fraction))
        for item in grouped.get(bucket, [])[:quota]:
            key = (item["source_id"], item["image_path"].name)
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)

    if len(selected) < sample_size:
        remainder = [
            item
            for item in candidates
            if (item["source_id"], item["image_path"].name) not in selected_keys
        ]
        rng.shuffle(remainder)
        remainder.sort(key=lambda item: item["score"], reverse=True)
        selected.extend(remainder[: sample_size - len(selected)])
    return selected[:sample_size]


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero.")

    root = resolve(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Basketball Game Detections staging root not found: {root}")
    candidates = load_candidates(root)
    if not candidates:
        raise ValueError("No staged Basketball Game Detections frames were available for QA.")

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

    rows: list[dict] = []
    for item in choose_samples(candidates, args.sample_size, args.seed):
        image_path = item["image_path"]
        label_path = item["label_path"]
        batch_name = f"{item['source_id']}__{image_path.name}"
        batch_stem = Path(batch_name).stem
        shutil.copy2(image_path, images_out / batch_name)
        shutil.copy2(label_path, labels_original / f"{batch_stem}.txt")
        shutil.copy2(label_path, labels_reviewed / f"{batch_stem}.txt")
        rows.append(
            {
                "image": batch_name,
                "source_id": item["source_id"],
                "frame": "",
                "time_seconds": "",
                "reason": f"bgd_{item['bucket']}_qa",
                "original_confidence": "",
                "original_detections_json": "",
                "review_status": "",
                "reviewed_at": "",
                "label_original": f"labels_original/{batch_stem}.txt",
                "label_reviewed": f"labels_reviewed/{batch_stem}.txt",
            }
        )

    with (batch_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    reasons = defaultdict(int)
    for row in rows:
        reasons[row["reason"]] += 1
    print("Basketball Game Detections QA sample prepared.")
    print(f"Frames sampled: {len(rows)}")
    print(f"Buckets: {dict(reasons)}")
    print(f"Batch: {batch_dir}")
    print(f"Next: python3 scripts/review_labels.py {args.batch}")


if __name__ == "__main__":
    main()
