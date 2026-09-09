from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tomllib

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy frame auto-label helper. Unlike autolabel.py, this moves frames "
            "with no retained detections into discarded_frames."
        )
    )
    parser.add_argument("--root", default="modelv0.2")
    parser.add_argument("--config", default="configs/autolabel.toml")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def choose_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    config_path = resolve(args.config)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    frame_dir = root / "frames"
    label_dir = root / "labels"
    discarded_dir = root / "discarded_frames"
    label_dir.mkdir(parents=True, exist_ok=True)
    discarded_dir.mkdir(parents=True, exist_ok=True)

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    class_mapping = {str(name): int(class_id) for name, class_id in config["class_mapping"].items()}
    text_prompt = " . ".join(class_mapping.keys()) + " ."
    device = choose_device(args.device)

    processor = AutoProcessor.from_pretrained(config["model"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config["model"]).to(device).eval()

    images = sorted(
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )

    moved = 0
    labelled = 0
    skipped = 0

    for image_path in tqdm(images, desc="CVAT frame auto-label"):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_grounded_object_detection(
            outputs,
            threshold=float(config["box_threshold"]),
            text_threshold=float(config["text_threshold"]),
            target_sizes=[(height, width)],
        )[0]

        lines: list[str] = []
        for box, label in zip(result["boxes"], result["labels"]):
            detected = str(label).lower().strip()
            if detected not in class_mapping:
                continue

            xmin, ymin, xmax, ymax = box.tolist()
            xmin = max(0.0, min(float(width), xmin))
            xmax = max(0.0, min(float(width), xmax))
            ymin = max(0.0, min(float(height), ymin))
            ymax = max(0.0, min(float(height), ymax))
            if xmax <= xmin or ymax <= ymin:
                continue

            xc = ((xmin + xmax) / 2) / width
            yc = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            lines.append(
                f"{class_mapping[detected]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            labelled += 1
        else:
            shutil.move(str(image_path), str(discarded_dir / image_path.name))
            moved += 1

    print(f"Labelled: {labelled}")
    print(f"Skipped existing labels: {skipped}")
    print(f"Moved to discarded_frames: {moved}")


if __name__ == "__main__":
    main()
