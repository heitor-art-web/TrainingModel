from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data import (
    assert_source_allowed,
    load_data_registry,
    sanitize_source_id,
    source_config,
)

SOURCE_KINDS = ("broadcast", "fixed_camera", "stands", "smartphone", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a compact, evenly spaced frame sample from a local video while "
            "preserving its source identity."
        )
    )
    parser.add_argument("video", help="Local video file. This script does not download URLs.")
    parser.add_argument("--source-id", required=True, help="Stable ID for this video/source.")
    parser.add_argument("--kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument(
        "--purpose",
        choices=("training", "evaluation"),
        default="training",
        help="Protected test sources require evaluation purpose.",
    )
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds.")
    parser.add_argument("--fps", type=float, default=None, help="Sampling rate in frames/second.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum normal frames from this source; capped by project policy.",
    )
    parser.add_argument("--registry", default="configs/data_sources.toml")
    parser.add_argument("--output-root", default="local_data/collected")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def evenly_spaced(values: list[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def main() -> None:
    args = parse_args()
    registry = load_data_registry(args.registry)
    source_id = args.source_id.strip()
    if not source_id:
        raise ValueError("--source-id cannot be empty.")
    if args.purpose == "training":
        assert_source_allowed(source_id, "training", registry)

    collection = registry.get("collection", {})
    sample_fps = float(
        args.fps if args.fps is not None else collection.get("default_sample_fps", 1.0)
    )
    requested_max = int(
        args.max_frames
        if args.max_frames is not None
        else collection.get("default_frames_per_source", 120)
    )
    absolute_max = int(collection.get("max_frames_per_source", 200))
    if sample_fps <= 0:
        raise ValueError("Sampling FPS must be positive.")
    if requested_max <= 0:
        raise ValueError("--max-frames must be positive.")
    if requested_max > absolute_max:
        raise ValueError(
            f"Requested {requested_max} frames, above the per-source collection cap of "
            f"{absolute_max}. Collect another independent source instead."
        )

    video = resolve(args.video)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    registered = source_config(source_id, registry)
    start_seconds = (
        float(args.start)
        if args.start is not None
        else float(registered.get("start_seconds", 0.0))
    )
    registered_end = registered.get("end_seconds")
    end_seconds = (
        float(args.end)
        if args.end is not None
        else (float(registered_end) if registered_end is not None else None)
    )
    if start_seconds < 0:
        raise ValueError("Start time cannot be negative.")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time.")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata does not contain a valid FPS/frame count.")

    duration = frame_count / source_fps
    effective_end = min(duration, end_seconds) if end_seconds is not None else duration
    if start_seconds >= effective_end:
        capture.release()
        raise ValueError(
            f"Requested segment starts at {start_seconds:.2f}s but video/segment ends at "
            f"{effective_end:.2f}s."
        )

    start_frame = max(0, int(round(start_seconds * source_fps)))
    end_frame = min(frame_count - 1, int(round(effective_end * source_fps)) - 1)
    interval = max(1, int(round(source_fps / sample_fps)))
    candidates = list(range(start_frame, end_frame + 1, interval))
    selected = evenly_spaced(candidates, requested_max)

    safe_id = sanitize_source_id(source_id)
    output_dir = resolve(args.output_root) / safe_id
    images_dir = output_dir / "images"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            capture.release()
            raise FileExistsError(
                f"Collection already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for frame_number in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        filename = f"{safe_id}__frame_{frame_number:07d}.jpg"
        destination = images_dir / filename
        if not cv2.imwrite(str(destination), frame):
            continue
        rows.append(
            {
                "source_id": source_id,
                "kind": args.kind,
                "purpose": args.purpose,
                "frame": frame_number,
                "time_seconds": f"{frame_number / source_fps:.3f}",
                "image": f"images/{filename}",
            }
        )
    capture.release()

    if not rows:
        raise RuntimeError("No frames were written from the requested video segment.")

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["source_id", "kind", "purpose", "frame", "time_seconds", "image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    source_metadata = {
        "source_id": source_id,
        "kind": args.kind,
        "purpose": args.purpose,
        "video": str(video),
        "registered_reference": registered.get("reference"),
        "start_seconds": start_seconds,
        "end_seconds": effective_end,
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "frames_written": len(rows),
        "normal_frame_cap": requested_max,
    }
    (output_dir / "source.json").write_text(
        json.dumps(source_metadata, indent=2), encoding="utf-8"
    )

    print(f"Collected {len(rows)} frames from {source_id}.")
    print(f"Source directory: {output_dir}")
    print("Next: auto-label this source, then send selected frames through the reviewer.")


if __name__ == "__main__":
    main()
