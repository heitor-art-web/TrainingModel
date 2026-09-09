from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from math import hypot
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import load_toml, resolve_weights, select_device


@dataclass
class BallState:
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    missed: int = 0


class BallTracker:
    def __init__(
        self,
        max_gap_frames: int,
        max_distance_ratio: float,
        position_alpha: float,
        velocity_beta: float,
        size_alpha: float,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.max_distance_ratio = max_distance_ratio
        self.position_alpha = position_alpha
        self.velocity_beta = velocity_beta
        self.size_alpha = size_alpha
        self.state: BallState | None = None

    def reset(self) -> None:
        self.state = None

    def predict_position(self) -> tuple[float, float] | None:
        if self.state is None:
            return None
        return self.state.x + self.state.vx, self.state.y + self.state.vy

    def update(
        self,
        detections: list[tuple[float, float, float, float, float]],
        frame_width: int,
        frame_height: int,
    ) -> tuple[str, BallState | None, float | None]:
        """Update tracker from basketball detections.

        Detection tuples are (center_x, center_y, width, height, confidence).
        Returns (status, state, confidence) where status is detected, predicted or lost.
        """
        if self.state is None:
            if not detections:
                return "lost", None, None
            best = max(detections, key=lambda item: item[4])
            x, y, width, height, confidence = best
            self.state = BallState(x, y, 0.0, 0.0, width, height)
            return "detected", self.state, confidence

        predicted = self.predict_position()
        assert predicted is not None
        px, py = predicted
        frame_diagonal = hypot(frame_width, frame_height)
        gate = self.max_distance_ratio * frame_diagonal

        candidate = None
        candidate_distance = None
        for detection in detections:
            x, y, width, height, confidence = detection
            distance = hypot(x - px, y - py)
            if distance <= gate and (candidate_distance is None or distance < candidate_distance):
                candidate = detection
                candidate_distance = distance

        if candidate is not None:
            x, y, width, height, confidence = candidate
            residual_x = x - px
            residual_y = y - py

            self.state.x = px + self.position_alpha * residual_x
            self.state.y = py + self.position_alpha * residual_y
            self.state.vx = self.state.vx + self.velocity_beta * residual_x
            self.state.vy = self.state.vy + self.velocity_beta * residual_y
            self.state.width = (
                self.size_alpha * width + (1.0 - self.size_alpha) * self.state.width
            )
            self.state.height = (
                self.size_alpha * height + (1.0 - self.size_alpha) * self.state.height
            )
            self.state.missed = 0
            return "detected", self.state, confidence

        self.state.x = px
        self.state.y = py
        self.state.missed += 1

        if self.state.missed <= self.max_gap_frames:
            return "predicted", self.state, None

        self.reset()
        return "lost", None, None


def parse_args() -> argparse.Namespace:
    config = load_toml(PROJECT_ROOT / "configs" / "ball_tracking.toml")

    parser = argparse.ArgumentParser(
        description="Hybrid basketball detection + temporal trajectory tracking for video."
    )
    parser.add_argument("source", help="Input video path.")
    parser.add_argument("--weights", help="Optional YOLO checkpoint override.")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--conf", type=float, default=float(config["confidence"]))
    parser.add_argument("--iou", type=float, default=float(config["iou"]))
    parser.add_argument("--imgsz", type=int, default=int(config["imgsz"]))
    parser.add_argument("--max-gap", type=int, default=int(config["max_gap_frames"]))
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=float(config["max_distance_ratio"]),
        help="Maximum association distance as a fraction of frame diagonal.",
    )
    parser.add_argument("--no-video", action="store_true", help="Only write the CSV trajectory.")
    return parser.parse_args()


def find_ball_class_id(names: dict[int, str]) -> int:
    for class_id, class_name in names.items():
        if class_name.lower().strip() == "basketball":
            return int(class_id)
    raise ValueError("The loaded model does not contain a 'basketball' class.")


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    weights = resolve_weights(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Pass --weights /path/to/model.pt if needed."
        )

    config = load_toml(PROJECT_ROOT / "configs" / "ball_tracking.toml")
    device = select_device(args.device)
    model = YOLO(str(weights))
    ball_class_id = find_ball_class_id(model.names)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = PROJECT_ROOT / "runs" / "ball_tracking" / source.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory.csv"
    video_path = output_dir / "tracked.mp4"

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {video_path}")

    tracker = BallTracker(
        max_gap_frames=args.max_gap,
        max_distance_ratio=args.max_distance_ratio,
        position_alpha=float(config["position_alpha"]),
        velocity_beta=float(config["velocity_beta"]),
        size_alpha=float(config["size_alpha"]),
    )
    trail = deque(maxlen=int(config["trail_length"]))

    detected_frames = 0
    predicted_frames = 0
    lost_frames = 0
    frame_index = 0

    print(f"Weights: {weights}")
    print(f"Device: {device}")
    print(f"Video: {source}")
    print(f"Frames: {total_frames if total_frames > 0 else 'unknown'}")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "frame",
                "time_seconds",
                "status",
                "x",
                "y",
                "width",
                "height",
                "confidence",
                "missed_frames",
            ]
        )

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                classes=[ball_class_id],
                verbose=False,
            )[0]

            detections: list[tuple[float, float, float, float, float]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    detections.append(
                        (
                            (x1 + x2) / 2.0,
                            (y1 + y2) / 2.0,
                            x2 - x1,
                            y2 - y1,
                            confidence,
                        )
                    )

            status, state, confidence = tracker.update(detections, width, height)

            if status == "detected":
                detected_frames += 1
            elif status == "predicted":
                predicted_frames += 1
            else:
                lost_frames += 1

            if state is not None:
                trail.append((int(round(state.x)), int(round(state.y)), status))
                csv_writer.writerow(
                    [
                        frame_index,
                        frame_index / fps,
                        status,
                        round(state.x, 3),
                        round(state.y, 3),
                        round(state.width, 3),
                        round(state.height, 3),
                        "" if confidence is None else round(confidence, 5),
                        state.missed,
                    ]
                )
            else:
                trail.clear()
                csv_writer.writerow(
                    [frame_index, frame_index / fps, status, "", "", "", "", "", ""]
                )

            if writer is not None:
                for index in range(1, len(trail)):
                    cv2.line(frame, trail[index - 1][:2], trail[index][:2], (255, 255, 255), 2)

                if state is not None:
                    x = int(round(state.x))
                    y = int(round(state.y))
                    half_w = max(3, int(round(state.width / 2)))
                    half_h = max(3, int(round(state.height / 2)))
                    box_color = (0, 255, 0) if status == "detected" else (0, 165, 255)
                    cv2.rectangle(
                        frame,
                        (x - half_w, y - half_h),
                        (x + half_w, y + half_h),
                        box_color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"ball: {status}",
                        (max(0, x - half_w), max(20, y - half_h - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                writer.write(frame)

            if frame_index % 250 == 0:
                print(f"Processed {frame_index} frames...")

    capture.release()
    if writer is not None:
        writer.release()

    print("\nTracking complete.")
    print(f"Detected frames:  {detected_frames}")
    print(f"Predicted frames: {predicted_frames}")
    print(f"Lost frames:      {lost_frames}")
    print(f"Trajectory CSV:   {csv_path}")
    if writer is not None:
        print(f"Tracked video:    {video_path}")


if __name__ == "__main__":
    main()
