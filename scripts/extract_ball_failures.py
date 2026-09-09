from __future__ import annotations

import argparse
import csv
from collections import Counter
from math import hypot
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.hard_cases import classify_hard_case
from training.common import load_toml


def parse_args() -> argparse.Namespace:
    config = load_toml(PROJECT_ROOT / "configs" / "ball_tracking.toml")
    parser = argparse.ArgumentParser(
        description="Extract representative ball-tracking hard cases from a tracked video run."
    )
    parser.add_argument("source", help="Original input video used for tracking.")
    parser.add_argument(
        "--trajectory",
        help="Trajectory CSV. Defaults to runs/ball_tracking/<video>/trajectory.csv.",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=float(config.get("hard_case_confidence", 0.35)),
    )
    parser.add_argument(
        "--jump-ratio",
        type=float,
        default=float(config.get("hard_case_jump_ratio", 0.08)),
        help="Flag large center jumps above this fraction of frame diagonal.",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=int(config.get("hard_case_min_gap_frames", 15)),
        help="Minimum frames between extracted examples of the same reason.",
    )
    parser.add_argument(
        "--max-per-reason",
        type=int,
        default=int(config.get("hard_case_max_per_reason", 100)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    trajectory = (
        Path(args.trajectory).expanduser().resolve()
        if args.trajectory
        else PROJECT_ROOT / "runs" / "ball_tracking" / source.stem / "trajectory.csv"
    )
    if not trajectory.exists():
        raise FileNotFoundError(
            f"Trajectory not found: {trajectory}. Run scripts/ball_tracker.py first."
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diagonal = hypot(width, height)

    with trajectory.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    output_dir = PROJECT_ROOT / "runs" / "hard_cases" / source.stem
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    last_saved: dict[str, int] = {}
    saved_counts: Counter[str] = Counter()
    previous_xy: tuple[float, float] | None = None

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["frame", "time_seconds", "reason", "status", "confidence", "image"])

        for row in rows:
            frame_number = int(row["frame"])
            reason = classify_hard_case(
                row=row,
                previous_xy=previous_xy,
                frame_diagonal=frame_diagonal,
                low_confidence_threshold=args.low_confidence,
                jump_ratio_threshold=args.jump_ratio,
            )

            x_text = row.get("x", "").strip()
            y_text = row.get("y", "").strip()
            if x_text and y_text:
                previous_xy = (float(x_text), float(y_text))

            if reason is None:
                continue
            if saved_counts[reason] >= args.max_per_reason:
                continue
            if frame_number - last_saved.get(reason, -10**9) < args.min_gap:
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ok, frame = capture.read()
            if not ok:
                continue

            filename = f"frame_{frame_number:07d}_{reason}.jpg"
            image_path = frames_dir / filename
            if not cv2.imwrite(str(image_path), frame):
                continue

            writer.writerow(
                [
                    frame_number,
                    row.get("time_seconds", ""),
                    reason,
                    row.get("status", ""),
                    row.get("confidence", ""),
                    str(image_path.relative_to(output_dir)),
                ]
            )
            saved_counts[reason] += 1
            last_saved[reason] = frame_number

    capture.release()

    print("Hard-case extraction complete.")
    for reason in ("predicted", "lost", "low_confidence", "large_jump"):
        print(f"{reason}: {saved_counts[reason]}")
    print(f"Frames:   {frames_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
