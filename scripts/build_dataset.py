from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import (
    VALID_SPLITS,
    assign_sources,
    get_source_policy,
    load_data_registry,
    sanitize_source_id,
    split_summary,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
APPROVED_REVIEW_STATUSES = {"corrected", "already_correct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a YOLO dataset using source-level train/val/test splitting "
            "and duplicate protection."
        )
    )
    parser.add_argument("--sources-root", default="local_data/sources")
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--registry", default="configs/data_sources.toml")
    parser.add_argument(
        "--near-duplicate-hamming",
        type=int,
        default=None,
        help="Override dHash Hamming threshold. 0 disables near-duplicate removal.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output dataset."
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def read_priority_manifest(source_dir: Path) -> dict[str, dict]:
    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    result: dict[str, dict] = {}
    for row in rows:
        image = row.get("image") or row.get("dataset_image") or row.get("source_image")
        if image:
            result[Path(image).name] = row
    return result


def collect_samples(sources_root: Path, registry: dict) -> list[dict]:
    samples: list[dict] = []
    if not sources_root.exists():
        raise FileNotFoundError(
            f"Sources root not found: {sources_root}. "
            "Export reviewed sources to local_data/sources/<source_id>/ first."
        )

    for source_dir in sorted(path for path in sources_root.iterdir() if path.is_dir()):
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        source_id = source_dir.name
        policy = get_source_policy(source_id, registry)
        metadata = read_priority_manifest(source_dir)

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            row = metadata.get(image_path.name, {})
            reason = (row.get("reason") or "").strip()
            review_status = (row.get("review_status") or "").strip()
            if policy.role == "gold_test" and review_status not in APPROVED_REVIEW_STATUSES:
                raise ValueError(
                    f"Gold-test image {image_path.name} in source {source_id!r} "
                    "has not been fully reviewed. Mark it corrected/already_correct first."
                )
            priority = bool(reason) or review_status in APPROVED_REVIEW_STATUSES
            samples.append(
                {
                    "source_id": source_id,
                    "policy_role": policy.role,
                    "image": image_path,
                    "label": label_path,
                    "reason": reason,
                    "review_status": review_status,
                    "priority": priority,
                }
            )
    if not samples:
        raise ValueError(f"No image/label pairs found under {sources_root}.")
    return samples


def exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, hash_size: int = 8) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((hash_size + 1, hash_size)).getdata())
    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            value |= pixels[start + col] > pixels[start + col + 1]
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def deduplicate_samples(samples: list[dict], threshold: int) -> tuple[list[dict], list[dict]]:
    """Deduplicate globally, prioritising protected test and hard-case samples."""
    role_rank = {"gold_test": 0, "test": 1, "val": 2, "train": 3, "pool": 4}
    ordered = sorted(
        samples,
        key=lambda item: (
            role_rank.get(item["policy_role"], 5),
            0 if item["priority"] else 1,
            item["source_id"],
            item["image"].name,
        ),
    )

    exact_seen: dict[str, dict] = {}
    hash_records: list[tuple[int, dict]] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[dict] = []
    removed: list[dict] = []

    for sample in ordered:
        sha = exact_hash(sample["image"])
        if sha in exact_seen:
            removed.append(
                {
                    **sample,
                    "duplicate_reason": "exact",
                    "duplicate_of": str(exact_seen[sha]["image"]),
                }
            )
            continue

        phash = difference_hash(sample["image"])
        duplicate = None
        if threshold > 0:
            candidate_indices: set[int] = set()
            for band in range(8):
                byte = (phash >> (band * 8)) & 0xFF
                candidate_indices.update(buckets.get((band, byte), []))
            for idx in candidate_indices:
                other_hash, other_sample = hash_records[idx]
                if hamming_distance(phash, other_hash) <= threshold:
                    duplicate = other_sample
                    break

        if duplicate is not None:
            removed.append(
                {
                    **sample,
                    "duplicate_reason": "near",
                    "duplicate_of": str(duplicate["image"]),
                }
            )
            continue

        exact_seen[sha] = sample
        index = len(hash_records)
        hash_records.append((phash, sample))
        for band in range(8):
            byte = (phash >> (band * 8)) & 0xFF
            buckets[(band, byte)].append(index)
        kept.append(sample)

    return kept, removed


def main() -> None:
    args = parse_args()
    registry = load_data_registry(args.registry)
    sources_root = resolve(args.sources_root)
    output = resolve(args.output)
    threshold = (
        int(args.near_duplicate_hamming)
        if args.near_duplicate_hamming is not None
        else int(registry["split"].get("near_duplicate_hamming", 4))
    )

    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"Dataset output already exists: {output}. Use --force to replace it."
            )
        shutil.rmtree(output)

    samples = collect_samples(sources_root, registry)
    samples, duplicates = deduplicate_samples(samples, threshold)

    source_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        source_counts[sample["source_id"]] += 1
    assignments = assign_sources(dict(source_counts), registry)

    for source_id, split in assignments.items():
        policy = get_source_policy(source_id, registry)
        if split in {"train", "val"} and not policy.allow_training:
            raise ValueError(
                f"Protected source {source_id!r} was assigned to {split}; refusing dataset build."
            )

    for split in VALID_SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    collision_guard: set[str] = set()
    for sample in sorted(samples, key=lambda item: (item["source_id"], item["image"].name)):
        split = assignments[sample["source_id"]]
        prefix = sanitize_source_id(sample["source_id"])
        target_name = f"{prefix}__{sample['image'].name}"
        if target_name in collision_guard:
            raise ValueError(f"Dataset filename collision: {target_name}")
        collision_guard.add(target_name)

        target_image = output / "images" / split / target_name
        target_label = output / "labels" / split / f"{Path(target_name).stem}.txt"
        shutil.copy2(sample["image"], target_image)
        shutil.copy2(sample["label"], target_label)
        manifest_rows.append(
            {
                "split": split,
                "source_id": sample["source_id"],
                "source_image": str(sample["image"]),
                "dataset_image": str(target_image.relative_to(output)),
                "reason": sample["reason"],
                "review_status": sample["review_status"],
            }
        )

    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "source_id",
                "source_image",
                "dataset_image",
                "reason",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "configured_ratios": {
            name: float(registry["split"][name]) for name in VALID_SPLITS
        },
        "source_assignments": assignments,
        "splits": split_summary(assignments, dict(source_counts)),
        "near_duplicate_hamming": threshold,
        "duplicates_removed": len(duplicates),
        "duplicates": [
            {
                "source_id": item["source_id"],
                "image": str(item["image"]),
                "reason": item["duplicate_reason"],
                "duplicate_of": item["duplicate_of"],
            }
            for item in duplicates
        ],
    }
    (output / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("Dataset build complete.")
    for split in VALID_SPLITS:
        data = report["splits"][split]
        print(
            f"{split}: {data['frames']} frames "
            f"({data['ratio']:.1%}) from {len(data['sources'])} source(s)"
        )
    print(f"Duplicates removed: {len(duplicates)}")
    print(f"Manifest: {output / 'manifest.csv'}")
    print(f"Report:   {output / 'split_report.json'}")


if __name__ == "__main__":
    main()
