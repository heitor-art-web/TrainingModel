from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.basketball_game_detections import (
    build_class_map,
    find_dataset_yaml,
    parse_and_remap_labels,
    yolo_line,
)

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
            "Prepare a pre-import diagnostic QA batch directly from an existing "
            "Basketball Game Detections analysis manifest. No model inference is rerun."
        )
    )
    parser.add_argument("--input", required=True, help="Original unpacked YOLO export root.")
    parser.add_argument(
        "--analysis",
        default="runs/dataset_analysis/basketball_game_detections/analysis_manifest.csv",
    )
    parser.add_argument("--batch", default="bgd_model_fn_diagnostic")
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def truthy(value: str | bool | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def safe_join(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {relative!r}") from exc
    return candidate


def diagnostic_bucket(row: dict) -> str | None:
    if truthy(row.get("duplicate")):
        return None
    has_ball = truthy(row.get("has_ball"))
    if truthy(row.get("model_ball_miss")):
        return "model_fn"
    if has_ball and truthy(row.get("model_ball_hit")):
        return "model_hit_control"
    if truthy(row.get("model_ball_false_positive")):
        return "model_fp"
    if not has_ball and not truthy(row.get("model_ball_false_positive")):
        return "clean_no_ball_control"
    return None


def choose_samples(rows: list[dict], sample_size: int, seed: int) -> list[tuple[str, dict]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero.")
    sample_size = min(sample_size, sum(diagnostic_bucket(row) is not None for row in rows))
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {
        "model_fn": [],
        "model_hit_control": [],
        "model_fp": [],
        "clean_no_ball_control": [],
    }
    for row in rows:
        bucket = diagnostic_bucket(row)
        if bucket:
            grouped[bucket].append(row)
    for items in grouped.values():
        rng.shuffle(items)

    # Most of the batch validates whether the surprising v1.1.0 FN population has
    # trustworthy ground-truth labels. Controls make systematic label issues visible.
    fractions = {
        "model_fn": 0.625,
        "model_hit_control": 0.125,
        "model_fp": 0.125,
        "clean_no_ball_control": 0.125,
    }
    selected: list[tuple[str, dict]] = []
    used: set[str] = set()
    for bucket, fraction in fractions.items():
        quota = int(round(sample_size * fraction))
        for row in grouped[bucket][:quota]:
            image = row.get("image", "")
            if image and image not in used:
                selected.append((bucket, row))
                used.add(image)

    if len(selected) < sample_size:
        remainder: list[tuple[str, dict]] = []
        for bucket, items in grouped.items():
            for row in items:
                if row.get("image", "") not in used:
                    remainder.append((bucket, row))
        rng.shuffle(remainder)
        selected.extend(remainder[: sample_size - len(selected)])
    return selected[:sample_size]


def reviewer_metadata(bucket: str, row: dict) -> list[dict]:
    """Encode diagnostic signals in the list-of-detections shape expected by the reviewer."""
    entries = [
        {"class_name": f"diagnostic:{bucket}"},
        {"class_name": f"origin_split:{row.get('original_split', '') or 'unknown'}"},
        {"class_name": f"ball_size:{row.get('ball_size', '') or 'unknown'}"},
        {"class_name": f"model_ball_count:{row.get('model_ball_count', '') or '0'}"},
        {"class_name": f"model_ball_hit:{row.get('model_ball_hit', '') or 'false'}"},
        {"class_name": f"model_ball_miss:{row.get('model_ball_miss', '') or 'false'}"},
        {
            "class_name": (
                "model_ball_false_positive:"
                f"{row.get('model_ball_false_positive', '') or 'false'}"
            )
        },
    ]
    confidence = row.get("model_ball_max_conf", "")
    if confidence not in {"", None}:
        try:
            entries[0]["confidence"] = float(confidence)
        except (TypeError, ValueError):
            pass
    return entries


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input)
    analysis_path = resolve(args.analysis)
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset input not found: {input_root}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis manifest not found: {analysis_path}")

    yaml_path, config = find_dataset_yaml(input_root)
    class_map, ignored_ids = build_class_map(config["names"])
    with analysis_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    selected = choose_samples(rows, args.sample_size, args.seed)
    if not selected:
        raise ValueError("No diagnostic QA candidates were available in the analysis manifest.")

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

    review_rows: list[dict] = []
    bucket_counts: dict[str, int] = {}
    for index, (bucket, row) in enumerate(selected, 1):
        image_path = safe_join(input_root, row["image"])
        label_path = safe_join(input_root, row["label"])
        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(
                f"Diagnostic pair missing: image={image_path} label={label_path}"
            )
        objects = parse_and_remap_labels(label_path, class_map, ignored_ids)
        # Prefix with a stable index because Roboflow exports may reuse stems across splits.
        batch_name = f"{index:04d}__{image_path.name}"
        batch_stem = Path(batch_name).stem
        shutil.copy2(image_path, images_out / batch_name)
        label_text = "\n".join(yolo_line(obj) for obj in objects)
        original_path = labels_original / f"{batch_stem}.txt"
        reviewed_path = labels_reviewed / f"{batch_stem}.txt"
        original_path.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        shutil.copy2(original_path, reviewed_path)

        review_rows.append(
            {
                "image": batch_name,
                "source_id": "bgd_analysis_pool",
                "frame": "",
                "time_seconds": "",
                "reason": f"bgd_{bucket}_diagnostic",
                "original_confidence": row.get("model_ball_max_conf", ""),
                "original_detections_json": json.dumps(
                    reviewer_metadata(bucket, row), separators=(",", ":")
                ),
                "review_status": "",
                "reviewed_at": "",
                "label_original": f"labels_original/{batch_stem}.txt",
                "label_reviewed": f"labels_reviewed/{batch_stem}.txt",
            }
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    with (batch_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)

    print("Basketball Game Detections pre-import diagnostic QA prepared.")
    print(f"Frames sampled: {len(review_rows)}")
    print(f"Buckets: {bucket_counts}")
    print(f"Batch: {batch_dir}")
    print(f"Next: python3 scripts/review_labels.py {args.batch}")
    print(
        "This batch is diagnostic only. Do not export it into local_data/sources; "
        "use it to validate source labels and the model_fn population before import."
    )


if __name__ == "__main__":
    main()
