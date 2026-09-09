from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import load_model_config, resolve_weights, select_device


def parse_args() -> argparse.Namespace:
    model_config = load_model_config()
    inference_config = model_config.get("inference", {})

    parser = argparse.ArgumentParser(
        description="Run the current basketball detector on an image, video, directory or URL."
    )
    parser.add_argument("source", help="Image/video path, directory, URL or supported camera source.")
    parser.add_argument("--weights", help="Optional checkpoint override.")
    parser.add_argument(
        "--conf",
        type=float,
        default=float(inference_config.get("confidence", 0.25)),
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=float(inference_config.get("iou", 0.70)),
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(inference_config.get("imgsz", 768)),
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images/video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    version = model_config["version"]
    weights = resolve_weights(args.weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Copy the checkpoint to this machine "
            "or pass --weights /path/to/model.pt."
        )

    device = select_device(args.device)
    output_project = PROJECT_ROOT / "runs" / "inference"
    run_name = f"model_v{version}"

    print(f"Model version: {version}")
    print(f"Weights: {weights}")
    print(f"Device: {device}")

    model = YOLO(str(weights))
    counts: Counter[str] = Counter()
    processed = 0

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=not args.no_save,
        project=str(output_project),
        name=run_name,
        exist_ok=True,
        stream=True,
    )

    for result in results:
        processed += 1
        if result.boxes is None:
            continue
        for class_id in result.boxes.cls.tolist():
            counts[result.names[int(class_id)]] += 1

    print(f"Processed frames/images: {processed}")
    if counts:
        print("Detections:")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")
    else:
        print("No detections above the selected confidence threshold.")

    if not args.no_save:
        print(f"Annotated output: {output_project / run_name}")


if __name__ == "__main__":
    main()
