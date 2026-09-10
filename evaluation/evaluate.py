from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import (
    load_model_config,
    materialize_dataset_yaml,
    resolve_weights,
    select_device,
)


def parse_args() -> argparse.Namespace:
    model_config = load_model_config()
    inference_config = model_config.get("inference", {})

    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on a YOLO dataset split."
    )
    parser.add_argument("--weights", help="Optional checkpoint override.")
    parser.add_argument(
        "--dataset",
        default="dataset.yaml",
        help="Dataset YAML path relative to project root or absolute path.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(inference_config.get("imgsz", 768)),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def per_class_metrics(metrics) -> dict[str, dict[str, float]]:
    names = metrics.names
    class_indices = list(getattr(metrics.box, "ap_class_index", range(len(metrics.box.p))))
    precisions = list(metrics.box.p)
    recalls = list(metrics.box.r)
    map50s = list(metrics.box.ap50)
    maps = list(metrics.box.maps)

    result: dict[str, dict[str, float]] = {}
    for offset, class_id in enumerate(class_indices):
        class_id = int(class_id)
        name = str(names[class_id]) if isinstance(names, dict) else str(names[class_id])
        result[name] = {
            "precision": float(precisions[offset]),
            "recall": float(recalls[offset]),
            "map50": float(map50s[offset]),
            "map50_95": float(maps[offset]),
        }
    return result


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    configured_version = model_config["version"]
    weights = resolve_weights(args.weights)

    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. Pass --weights if evaluating another checkpoint."
        )

    runtime_dataset = materialize_dataset_yaml(args.dataset)
    device = select_device(args.device)
    output_project = PROJECT_ROOT / "runs" / "evaluation"
    run_name = f"{weights.stem}_{args.split}"

    model = YOLO(str(weights))

    try:
        metrics = model.val(
            data=str(runtime_dataset),
            split=args.split,
            imgsz=args.imgsz,
            device=device,
            plots=True,
            project=str(output_project),
            name=run_name,
            exist_ok=True,
        )
    finally:
        runtime_dataset.unlink(missing_ok=True)

    summary = {
        "configured_promoted_model_version": configured_version,
        "checkpoint": weights.name,
        "weights": str(weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": device,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class": per_class_metrics(metrics),
    }

    output_dir = output_project / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
