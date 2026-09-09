from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tomllib

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label basketball images with Grounding DINO in YOLO format."
    )
    parser.add_argument("--root", default="modelv0.2", help="Working dataset directory.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source folder inside --root. May be supplied multiple times. Default: images_input.",
    )
    parser.add_argument("--config", default="configs/autolabel.toml")
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, cuda, etc.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label images that already have a label file.",
    )
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Do not merge source folders; label images already present in <root>/images.",
    )
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


def load_config(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def merge_sources(root: Path, source_folders: list[str], images_folder: Path) -> int:
    manifest_path = root / "autolabel_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    numeric_stems = [
        int(path.stem)
        for path in images_folder.iterdir()
        if path.is_file() and path.stem.isdigit()
    ]
    counter = max(numeric_stems, default=0) + 1
    copied = 0

    for folder_name in source_folders:
        source = root / folder_name
        if not source.exists():
            print(f"Missing source folder: {source}")
            continue

        for file in sorted(source.iterdir()):
            if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            key = f"{folder_name}/{file.name}"
            known_name = manifest.get(key)
            if known_name and (images_folder / known_name).exists():
                continue

            new_name = f"{counter:06d}{file.suffix.lower()}"
            shutil.copy2(file, images_folder / new_name)
            manifest[key] = new_name
            counter += 1
            copied += 1

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return copied


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    config = load_config(resolve(args.config))
    source_folders = args.source or ["images_input"]

    images_folder = root / "images"
    labels_folder = root / "labels"
    images_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)

    if not args.skip_copy:
        copied = merge_sources(root, source_folders, images_folder)
        print(f"Copied {copied} new images into {images_folder}.")

    class_mapping = {str(name): int(class_id) for name, class_id in config["class_mapping"].items()}
    text_prompt = " . ".join(class_mapping.keys()) + " ."
    device = choose_device(args.device)

    print(f"Grounding DINO model: {config['model']}")
    print(f"Device: {device}")
    print(
        f"Thresholds: box={config['box_threshold']} text={config['text_threshold']}"
    )

    processor = AutoProcessor.from_pretrained(config["model"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config["model"]).to(device).eval()

    images = sorted(
        path
        for path in images_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    labelled = 0
    skipped = 0
    empty = 0
    detections = 0

    for image_path in tqdm(images, desc="Auto-labeling"):
        label_path = labels_folder / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
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

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        labelled += 1
        detections += len(lines)
        if not lines:
            empty += 1

    print(f"Images available: {len(images)}")
    print(f"Labelled this run: {labelled}")
    print(f"Skipped existing labels: {skipped}")
    print(f"Images with no retained detections: {empty}")
    print(f"Detections written: {detections}")


if __name__ == "__main__":
    main()
