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
    resolve_project_path,
    resolve_weights,
    select_device,
)
from data import assert_dataset_budget, dataset_image_limit, load_data_registry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a controlled candidate for the basketball detector."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Target semantic version, e.g. 1.1.0.",
    )
    parser.add_argument(
        "--init",
        choices=("promoted", "pretrained"),
        default="promoted",
        help="Start from the promoted model or the generic pretrained YOLO base.",
    )
    parser.add_argument(
        "--weights",
        help=(
            "Optional custom local checkpoint. When supplied it overrides --init and "
            "the run is labelled custom."
        ),
    )
    parser.add_argument(
        "--pretrained-model",
        help="Optional pretrained model name/path overriding configs/train.toml.",
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


def resolve_initialization(
    args: argparse.Namespace, train_config: dict
) -> tuple[str, str | Path]:
    if args.weights:
        custom = resolve_project_path(args.weights)
        if not custom.exists():
            raise FileNotFoundError(f"Custom checkpoint not found: {custom}")
        return "custom", custom

    if args.init == "promoted":
        promoted = resolve_weights()
        if not promoted.exists():
            raise FileNotFoundError(
                f"Promoted baseline checkpoint not found: {promoted}."
            )
        return "promoted", promoted

    pretrained = args.pretrained_model or train_config.get("pretrained_model", "yolo26b.pt")
    return "pretrained", str(pretrained)


def count_dataset_images(dataset_root: Path) -> int:
    images_root = dataset_root / "images"
    if not images_root.exists():
        return 0
    return sum(
        1
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    validate_version(args.version, model_config["version"])

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    train_config = load_toml(config_path)

    init_name, initial_weights = resolve_initialization(args, train_config)

    runtime_dataset = materialize_dataset_yaml(args.dataset)
    dataset_root = Path(__import__("yaml").safe_load(runtime_dataset.read_text())["path"])
    if not dataset_root.exists():
        runtime_dataset.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}. "
            "Build and audit the dataset before training."
        )

    registry = load_data_registry()
    dataset_images = count_dataset_images(dataset_root)
    if dataset_images <= 0:
        runtime_dataset.unlink(missing_ok=True)
        raise ValueError(f"No dataset images found under {dataset_root / 'images'}.")
    try:
        assert_dataset_budget(dataset_images, registry)
    except Exception:
        runtime_dataset.unlink(missing_ok=True)
        raise

    device = select_device(str(train_config.get("device", "auto")))
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    run_name = f"model_v{args.version}_{init_name}"

    print(f"Promoted model version: {model_config['version']}")
    print(f"Initialization: {init_name}")
    print(f"Initial weights/model: {initial_weights}")
    print(f"Target model version: {args.version}")
    print(f"Dataset images: {dataset_images}/{dataset_image_limit(registry)}")
    print(f"Training device: {device}")

    model = YOLO(str(initial_weights))

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
    print("Training complete.")
    print(f"Candidate weights: {output_weights}")
    print("Evaluate this checkpoint before promotion.")


if __name__ == "__main__":
    main()
