from __future__ import annotations

import argparse
from pathlib import Path
import re

from ultralytics import YOLO

from common import (
    PROJECT_ROOT,
    load_model_config,
    load_toml,
    materialize_dataset_yaml,
    resolve_weights,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the current basketball detector into a new version."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Target semantic version, e.g. 1.1.0. This prevents accidental retraining.",
    )
    parser.add_argument(
        "--weights",
        help="Optional baseline checkpoint. Defaults to configs/model.toml.",
    )
    parser.add_argument(
        "--dataset",
        default="dataset.yaml",
        help="Dataset YAML path relative to the project root, or an absolute path.",
    )
    parser.add_argument(
        "--config",
        default="configs/train.toml",
        help="Training TOML path relative to the project root.",
    )
    return parser.parse_args()


def validate_version(version: str, current_version: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Version must use semantic form MAJOR.MINOR.PATCH, e.g. 1.1.0")
    if version == current_version:
        raise ValueError(
            f"Target version {version} matches the current model. Choose a new version."
        )


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    validate_version(args.version, model_config["version"])

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    train_config = load_toml(config_path)

    weights = resolve_weights(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Could not find baseline weights at {weights}. "
            "Training requires an existing baseline checkpoint."
        )

    runtime_dataset = materialize_dataset_yaml(args.dataset)
    dataset_root = Path(__import__("yaml").safe_load(runtime_dataset.read_text())["path"])
    if not dataset_root.exists():
        runtime_dataset.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. "
            "Copy or prepare the dataset before fine-tuning."
        )

    device = select_device(str(train_config.get("device", "auto")))
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    run_name = f"model_v{args.version}"

    print(f"Baseline model version: {model_config['version']}")
    print(f"Loading baseline weights: {weights}")
    print(f"Target model version: {args.version}")
    print(f"Training device: {device}")

    model = YOLO(str(weights))

    try:
        model.train(
            data=str(runtime_dataset),
            epochs=int(train_config["epochs"]),
            imgsz=int(train_config["imgsz"]),
            batch=int(train_config["batch"]),
            lr0=float(train_config["lr0"]),
            device=device,
            workers=int(train_config["workers"]),
            cache=train_config.get("cache", "disk"),
            plots=bool(train_config.get("plots", True)),
            save=bool(train_config.get("save", True)),
            patience=int(train_config["patience"]),
            project=str(runs_dir),
            name=run_name,
            exist_ok=False,
        )
    finally:
        runtime_dataset.unlink(missing_ok=True)

    output_weights = runs_dir / run_name / "weights" / "best.pt"
    print("Fine-tuning complete.")
    print(f"Candidate weights: {output_weights}")
    print(
        "Evaluate this checkpoint before promoting it and updating configs/model.toml."
    )


if __name__ == "__main__":
    main()
