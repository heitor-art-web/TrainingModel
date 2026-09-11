from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["basketball", "player", "referee", "backboard", "rim"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare labels_original and labels_reviewed for a review batch and "
            "summarize what annotators changed by class and edit type."
        )
    )
    parser.add_argument("batch", help="Batch name or path under runs/review/.")
    parser.add_argument(
        "--same-class-iou",
        type=float,
        default=0.50,
        help="Minimum IoU for pairing original/reviewed boxes of the same class.",
    )
    parser.add_argument(
        "--unchanged-iou",
        type=float,
        default=0.95,
        help="Matched boxes below this IoU are counted as geometry edits.",
    )
    parser.add_argument(
        "--reclass-iou",
        type=float,
        default=0.70,
        help="Minimum IoU for treating unmatched cross-class boxes as reclassified.",
    )
    return parser.parse_args()


def resolve_batch(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        direct = PROJECT_ROOT / path
        path = direct if direct.exists() else PROJECT_ROOT / "runs" / "review" / value
    path = path.resolve()
    if not (path / "manifest.csv").exists():
        raise FileNotFoundError(f"Review manifest not found under {path}.")
    return path


def parse_boxes(path: Path) -> list[dict]:
    boxes: list[dict] = []
    if not path.exists():
        return boxes
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}: line {line_number} must contain at least 5 values")
        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}: line {line_number} has unknown class id {class_id}")
        xc, yc, width, height = map(float, parts[1:5])
        boxes.append(
            {
                "class_id": class_id,
                "corners": (
                    xc - width / 2.0,
                    yc - height / 2.0,
                    xc + width / 2.0,
                    yc + height / 2.0,
                ),
            }
        )
    return boxes


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_pairs(
    original: list[dict],
    reviewed: list[dict],
    minimum_iou: float,
    same_class: bool,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for oi, left in enumerate(original):
        for ri, right in enumerate(reviewed):
            if same_class and left["class_id"] != right["class_id"]:
                continue
            if not same_class and left["class_id"] == right["class_id"]:
                continue
            value = iou(left["corners"], right["corners"])
            if value >= minimum_iou:
                candidates.append((value, oi, ri))
    candidates.sort(reverse=True)
    used_original: set[int] = set()
    used_reviewed: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for value, oi, ri in candidates:
        if oi in used_original or ri in used_reviewed:
            continue
        used_original.add(oi)
        used_reviewed.add(ri)
        pairs.append((oi, ri, value))
    return pairs


def main() -> None:
    args = parse_args()
    batch_dir = resolve_batch(args.batch)
    with (batch_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Review batch contains no rows.")

    totals = Counter()
    class_stats: dict[int, Counter] = defaultdict(Counter)
    reclassifications = Counter()
    reason_stats: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        status = (row.get("review_status") or "unreviewed").strip() or "unreviewed"
        reason = (row.get("reason") or "n/a").strip() or "n/a"
        totals["frames"] += 1
        totals[f"status:{status}"] += 1
        reason_stats[reason]["frames"] += 1
        reason_stats[reason][f"status:{status}"] += 1

        original_path = batch_dir / row.get("label_original", "")
        reviewed_path = batch_dir / row.get("label_reviewed", "")
        original = parse_boxes(original_path)
        reviewed = parse_boxes(reviewed_path)

        for box in original:
            class_stats[box["class_id"]]["original_boxes"] += 1
        for box in reviewed:
            class_stats[box["class_id"]]["reviewed_boxes"] += 1

        same_pairs = greedy_pairs(original, reviewed, args.same_class_iou, same_class=True)
        matched_original = {oi for oi, _, _ in same_pairs}
        matched_reviewed = {ri for _, ri, _ in same_pairs}
        frame_changed_classes: set[int] = set()

        for oi, ri, value in same_pairs:
            class_id = original[oi]["class_id"]
            if value < args.unchanged_iou:
                class_stats[class_id]["geometry_changed"] += 1
                frame_changed_classes.add(class_id)

        unmatched_original = [box for index, box in enumerate(original) if index not in matched_original]
        unmatched_reviewed = [box for index, box in enumerate(reviewed) if index not in matched_reviewed]
        reclass_pairs = greedy_pairs(
            unmatched_original,
            unmatched_reviewed,
            args.reclass_iou,
            same_class=False,
        )
        reclass_original = {oi for oi, _, _ in reclass_pairs}
        reclass_reviewed = {ri for _, ri, _ in reclass_pairs}

        for oi, ri, _ in reclass_pairs:
            before = unmatched_original[oi]["class_id"]
            after = unmatched_reviewed[ri]["class_id"]
            class_stats[before]["reclassified_out"] += 1
            class_stats[after]["reclassified_in"] += 1
            reclassifications[(before, after)] += 1
            frame_changed_classes.update({before, after})

        for index, box in enumerate(unmatched_original):
            if index in reclass_original:
                continue
            class_stats[box["class_id"]]["deleted"] += 1
            frame_changed_classes.add(box["class_id"])

        for index, box in enumerate(unmatched_reviewed):
            if index in reclass_reviewed:
                continue
            class_stats[box["class_id"]]["added"] += 1
            frame_changed_classes.add(box["class_id"])

        if frame_changed_classes:
            totals["frames_with_box_changes"] += 1
            reason_stats[reason]["frames_with_box_changes"] += 1
            for class_id in frame_changed_classes:
                class_stats[class_id]["frames_changed"] += 1

    print(f"Batch: {batch_dir}")
    print(f"Frames: {totals['frames']}")
    print(f"Frames with box changes: {totals['frames_with_box_changes']}")
    print("\nBy class:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        stats = class_stats[class_id]
        print(f"  {class_name}:")
        print(
            "    original={original_boxes}, reviewed={reviewed_boxes}, added={added}, "
            "deleted={deleted}, geometry_changed={geometry_changed}, "
            "reclassified_in={reclassified_in}, reclassified_out={reclassified_out}, "
            "frames_changed={frames_changed}".format(**{
                key: stats.get(key, 0)
                for key in (
                    "original_boxes",
                    "reviewed_boxes",
                    "added",
                    "deleted",
                    "geometry_changed",
                    "reclassified_in",
                    "reclassified_out",
                    "frames_changed",
                )
            })
        )

    print("\nReclassifications:")
    if not reclassifications:
        print("  none")
    else:
        for (before, after), count in reclassifications.most_common():
            print(f"  {CLASS_NAMES[before]} -> {CLASS_NAMES[after]}: {count}")

    print("\nBy reason:")
    for reason in sorted(reason_stats):
        stats = reason_stats[reason]
        print(
            f"  {reason}: frames={stats['frames']}, "
            f"box_changes={stats['frames_with_box_changes']}, "
            f"already_correct={stats['status:already_correct']}, "
            f"corrected={stats['status:corrected']}, "
            f"skip={stats['status:skip']}, uncertain={stats['status:uncertain']}"
        )


if __name__ == "__main__":
    main()
