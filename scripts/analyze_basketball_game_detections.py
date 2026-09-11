from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.basketball_game_detections import (
    ball_size_bucket,
    box_iou,
    build_class_map,
    collect_pairs_from_yaml,
    find_dataset_yaml,
    infer_source_group,
    parse_and_remap_labels,
)
from training.common import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Basketball Game Detections before import. Produces class/ball-size "
            "statistics, duplicate flags, v1.1.0 basketball errors and a curated selection."
        )
    )
    parser.add_argument(
        "--input",
        default="local_data/external/basketball_game_detections",
        help="Unpacked YOLO export of Basketball Game Detections.",
    )
    parser.add_argument(
        "--output",
        default="runs/dataset_analysis/basketball_game_detections",
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/model_v1.1.0_promoted/weights/best.pt",
        help="TrainingModel checkpoint used to find current-model hard cases.",
    )
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-select", type=int, default=3500)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    parser.add_argument(
        "--source-regex",
        help=(
            "Optional regex that proves original source/game identity. Use a named "
            "(?P<source>...) group or the first capture group. Without it, source "
            "identity is deliberately left unknown."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def exact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((hash_size + 1, hash_size)).getdata())
    value = 0
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            value |= pixels[start + col] > pixels[start + col + 1]
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def mark_duplicates(rows: list[dict], threshold: int) -> None:
    exact_seen: dict[str, dict] = {}
    hash_records: list[tuple[int, dict]] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    for row in rows:
        image_path = Path(row["image_path"])
        sha = exact_hash(image_path)
        if sha in exact_seen:
            row["duplicate"] = True
            row["duplicate_reason"] = "exact"
            row["duplicate_of"] = exact_seen[sha]["image_path"]
            continue

        dhash = difference_hash(image_path)
        duplicate = None
        if threshold > 0:
            candidate_indices: set[int] = set()
            for band in range(8):
                byte = (dhash >> (band * 8)) & 0xFF
                candidate_indices.update(buckets.get((band, byte), []))
            for index in candidate_indices:
                other_hash, other = hash_records[index]
                if hamming(dhash, other_hash) <= threshold:
                    duplicate = other
                    break
        if duplicate is not None:
            row["duplicate"] = True
            row["duplicate_reason"] = "near"
            row["duplicate_of"] = duplicate["image_path"]
            continue

        row["duplicate"] = False
        row["duplicate_reason"] = ""
        row["duplicate_of"] = ""
        exact_seen[sha] = row
        index = len(hash_records)
        hash_records.append((dhash, row))
        for band in range(8):
            byte = (dhash >> (band * 8)) & 0xFF
            buckets[(band, byte)].append(index)


def normalized_prediction_boxes(result, ball_conf: float) -> list[tuple[float, tuple[float, float, float, float]]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    height, width = result.orig_shape
    predictions = []
    for xyxy, confidence, class_id in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        if int(class_id) != 0 or float(confidence) < ball_conf:
            continue
        x1, y1, x2, y2 = xyxy
        predictions.append(
            (
                float(confidence),
                (x1 / width, y1 / height, x2 / width, y2 / height),
            )
        )
    return predictions


def add_model_signals(rows: list[dict], args: argparse.Namespace) -> None:
    if args.skip_model:
        for row in rows:
            row.update(
                {
                    "model_checked": False,
                    "model_ball_count": 0,
                    "model_ball_max_conf": 0.0,
                    "model_ball_hit": False,
                    "model_ball_miss": False,
                    "model_ball_false_positive": False,
                }
            )
        return

    weights = resolve(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Analysis checkpoint not found: {weights}. Pass --weights or --skip-model."
        )

    from ultralytics import YOLO

    device = select_device(args.device)
    print(f"Hard-case model: {weights}")
    print(f"Device: {device}")
    print(f"Basketball operating threshold: {args.ball_conf}")

    model = YOLO(str(weights))
    usable = [row for row in rows if not row["duplicate"]]
    batch_size = 32
    for offset in range(0, len(usable), batch_size):
        batch = usable[offset : offset + batch_size]
        results = model.predict(
            source=[row["image_path"] for row in batch],
            conf=max(0.001, min(args.ball_conf, 1.0)),
            imgsz=args.imgsz,
            device=device,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            predictions = normalized_prediction_boxes(result, args.ball_conf)
            gt_balls = row["gt_ball_corners"]
            hit = any(
                box_iou(gt_box, pred_box) >= args.match_iou
                for gt_box in gt_balls
                for _, pred_box in predictions
            )
            row.update(
                {
                    "model_checked": True,
                    "model_ball_count": len(predictions),
                    "model_ball_max_conf": max((conf for conf, _ in predictions), default=0.0),
                    "model_ball_hit": hit,
                    "model_ball_miss": bool(gt_balls) and not hit,
                    "model_ball_false_positive": not gt_balls and bool(predictions),
                }
            )
        print(f"Model analysis: {min(offset + batch_size, len(usable))}/{len(usable)}")

    for row in rows:
        if row["duplicate"]:
            row.update(
                {
                    "model_checked": False,
                    "model_ball_count": 0,
                    "model_ball_max_conf": 0.0,
                    "model_ball_hit": False,
                    "model_ball_miss": False,
                    "model_ball_false_positive": False,
                }
            )


def classify_selection(row: dict) -> tuple[str, int]:
    if row["duplicate"]:
        return "duplicate", -1000
    score = 0
    if row["model_ball_miss"]:
        return "model_fn", 120
    if row["model_ball_false_positive"]:
        return "model_fp", 110
    if row["ball_size"] == "tiny":
        return "tiny_ball", 90
    if row["ball_size"] == "small":
        return "small_ball", 70
    if not row["has_ball"] and row["player_count"] > 0:
        return "no_ball_hard_negative", 60
    if row["has_ball"] and row["has_rim"]:
        return "ball_and_rim", 50
    if row["has_referee"]:
        score += 15
    if row["has_ball"]:
        score += 20
    score += min(row["player_count"], 10)
    return "other", score


def select_rows(rows: list[dict], max_select: int, seed: int) -> None:
    max_select = min(max_select, sum(not row["duplicate"] for row in rows))
    if max_select <= 0:
        raise ValueError("--max-select must be positive and the dataset must contain usable images.")

    fractions = {
        "model_fn": 0.30,
        "model_fp": 0.10,
        "tiny_ball": 0.20,
        "small_ball": 0.15,
        "no_ball_hard_negative": 0.10,
        "ball_and_rim": 0.10,
        "other": 0.05,
    }
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        bucket, score = classify_selection(row)
        row["selection_bucket"] = bucket
        row["selection_score"] = score
        row["selected"] = False
        if bucket != "duplicate":
            grouped[bucket].append(row)

    for bucket_rows in grouped.values():
        rng.shuffle(bucket_rows)
        bucket_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    selected: list[dict] = []
    selected_ids: set[int] = set()
    for bucket, fraction in fractions.items():
        quota = int(round(max_select * fraction))
        for row in grouped.get(bucket, [])[:quota]:
            identity = id(row)
            if identity not in selected_ids:
                selected.append(row)
                selected_ids.add(identity)

    if len(selected) < max_select:
        remaining = [
            row
            for row in rows
            if not row["duplicate"] and id(row) not in selected_ids
        ]
        rng.shuffle(remaining)
        remaining.sort(key=lambda row: row["selection_score"], reverse=True)
        selected.extend(remaining[: max_select - len(selected)])

    for row in selected[:max_select]:
        row["selected"] = True


def csv_row(row: dict, input_root: Path) -> dict:
    return {
        "selected": str(bool(row["selected"])).lower(),
        "selection_bucket": row["selection_bucket"],
        "selection_score": row["selection_score"],
        "original_split": row["original_split"],
        "source_group": row["source_group"] or "",
        "image": str(Path(row["image_path"]).relative_to(input_root)),
        "label": str(Path(row["label_path"]).relative_to(input_root)),
        "has_ball": str(row["has_ball"]).lower(),
        "ball_count": row["ball_count"],
        "ball_size": row["ball_size"],
        "player_count": row["player_count"],
        "has_referee": str(row["has_referee"]).lower(),
        "has_rim": str(row["has_rim"]).lower(),
        "model_checked": str(row["model_checked"]).lower(),
        "model_ball_count": row["model_ball_count"],
        "model_ball_max_conf": f"{row['model_ball_max_conf']:.6f}",
        "model_ball_hit": str(row["model_ball_hit"]).lower(),
        "model_ball_miss": str(row["model_ball_miss"]).lower(),
        "model_ball_false_positive": str(row["model_ball_false_positive"]).lower(),
        "duplicate": str(row["duplicate"]).lower(),
        "duplicate_reason": row["duplicate_reason"],
        "duplicate_of": row["duplicate_of"],
    }


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input)
    output = resolve(args.output)
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset input not found: {input_root}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"Analysis output exists: {output}. Use --force to replace reports.")
    output.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in output.glob("*.csv"):
            path.unlink()
        for path in output.glob("*.json"):
            path.unlink()

    yaml_path, config = find_dataset_yaml(input_root)
    class_map, ignored_ids = build_class_map(config["names"])
    pairs = collect_pairs_from_yaml(yaml_path, config)

    rows: list[dict] = []
    class_counts: Counter[int] = Counter()
    for image_path, label_path, original_split in pairs:
        objects = parse_and_remap_labels(label_path, class_map, ignored_ids)
        counts = Counter(obj.class_id for obj in objects)
        class_counts.update(counts)
        balls = [obj for obj in objects if obj.class_id == 0]
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "original_split": original_split,
                "source_group": infer_source_group(image_path.stem, args.source_regex),
                "objects": objects,
                "gt_ball_corners": [obj.corners for obj in balls],
                "has_ball": bool(balls),
                "ball_count": len(balls),
                "ball_size": ball_size_bucket(objects),
                "player_count": counts[1],
                "has_referee": counts[2] > 0,
                "has_rim": counts[4] > 0,
            }
        )

    mark_duplicates(rows, args.near_duplicate_hamming)
    add_model_signals(rows, args)
    select_rows(rows, args.max_select, args.seed)

    fieldnames = list(csv_row(rows[0], input_root).keys())
    manifest_path = output / "analysis_manifest.csv"
    selected_path = output / "selected_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_row(row, input_root) for row in rows)
    with selected_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_row(row, input_root) for row in rows if row["selected"])

    bucket_counts = Counter(row["selection_bucket"] for row in rows if row["selected"])
    ball_sizes = Counter(row["ball_size"] for row in rows if not row["duplicate"])
    source_groups = {row["source_group"] for row in rows if row["source_group"]}
    summary = {
        "dataset": "Basketball Game Detections",
        "source_yaml": str(yaml_path.relative_to(input_root)),
        "images_total": len(rows),
        "images_usable_after_dedup": sum(not row["duplicate"] for row in rows),
        "duplicates": sum(row["duplicate"] for row in rows),
        "selected": sum(row["selected"] for row in rows),
        "class_instances": {
            "basketball": class_counts[0],
            "player": class_counts[1],
            "referee": class_counts[2],
            "rim": class_counts[4],
        },
        "ball_size_distribution": dict(ball_sizes),
        "selection_buckets": dict(bucket_counts),
        "model_ball_misses": sum(row["model_ball_miss"] for row in rows),
        "model_ball_false_positives": sum(row["model_ball_false_positive"] for row in rows),
        "source_groups_proven": len(source_groups),
        "source_grouping_mode": "regex" if args.source_regex else "unknown_training_pool",
        "ball_conf": args.ball_conf,
        "match_iou": args.match_iou,
        "near_duplicate_hamming": args.near_duplicate_hamming,
        "note": (
            "If source_groups_proven is zero, import selected frames as a training-only "
            "pool so this external dataset cannot contaminate validation/test."
        ),
    }
    summary_path = output / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Basketball Game Detections analysis complete.")
    print(f"Images scanned: {summary['images_total']}")
    print(f"Usable after dedup: {summary['images_usable_after_dedup']}")
    print(f"Selected: {summary['selected']}")
    print(f"Ball-size distribution: {summary['ball_size_distribution']}")
    print(f"Selection buckets: {summary['selection_buckets']}")
    print(f"v1.1.0 ball misses: {summary['model_ball_misses']}")
    print(f"v1.1.0 ball false positives: {summary['model_ball_false_positives']}")
    print(f"Source groups proven: {summary['source_groups_proven']}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected: {selected_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
