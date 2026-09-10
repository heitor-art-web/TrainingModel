from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tomllib

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKBOARD_CLASS_ID = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add Grounding DINO backboard pre-annotations to staged E-BARD labels "
            "without altering the preserved labels_ebard copies."
        )
    )
    parser.add_argument(
        "--root",
        default="local_data/ebard_import/sources",
        help="Root created by scripts/import_e_bard.py.",
    )
    parser.add_argument("--config", default="configs/autolabel.toml")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite-backboard",
        action="store_true",
        help="Replace existing class-3 boxes in working labels before adding new predictions.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
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


def without_backboard(lines: list[str]) -> list[str]:
    result = []
    for line in lines:
        parts = line.strip().split()
        if parts and int(parts[0]) == BACKBOARD_CLASS_ID:
            continue
        if parts:
            result.append(line.strip())
    return result


def to_yolo(box: list[float], width: int, height: int) -> str | None:
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(float(width), float(xmin)))
    xmax = max(0.0, min(float(width), float(xmax)))
    ymin = max(0.0, min(float(height), float(ymin)))
    ymax = max(0.0, min(float(height), float(ymax)))
    if xmax <= xmin or ymax <= ymin:
        return None
    xc = ((xmin + xmax) / 2.0) / width
    yc = ((ymin + ymax) / 2.0) / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    return f"{BACKBOARD_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def update_manifest(source_dir: Path, enriched_images: set[str]) -> None:
    path = source_dir / "manifest.csv"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return
    for row in rows:
        if row.get("image") in enriched_images:
            row["backboard_enriched"] = "true"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"E-BARD staging root not found: {root}. Run scripts/import_e_bard.py first."
        )

    config = load_config(resolve(args.config))
    device = choose_device(args.device)
    model_name = str(config["model"])
    box_threshold = float(config["box_threshold"])
    text_threshold = float(config["text_threshold"])

    print(f"Grounding DINO model: {model_name}")
    print(f"Device: {device}")
    print(f"Backboard thresholds: box={box_threshold} text={text_threshold}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()

    source_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    total_images = 0
    total_boxes = 0

    for source_dir in source_dirs:
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue
        enriched_images: set[str] = set()

        image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
        for image_path in tqdm(image_paths, desc=source_dir.name, leave=False):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            existing = label_path.read_text(encoding="utf-8").splitlines()
            existing_backboards = [
                line for line in existing if line.strip() and int(line.split()[0]) == BACKBOARD_CLASS_ID
            ]
            if existing_backboards and not args.overwrite_backboard:
                enriched_images.add(image_path.name)
                continue
            base_lines = without_backboard(existing) if args.overwrite_backboard else [line.strip() for line in existing if line.strip()]

            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            inputs = processor(images=image, text="backboard .", return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)],
            )[0]

            new_lines: list[str] = []
            for box in result["boxes"]:
                line = to_yolo(box.tolist(), width, height)
                if line is not None:
                    new_lines.append(line)

            final_lines = [*base_lines, *new_lines]
            label_path.write_text(
                "\n".join(final_lines) + ("\n" if final_lines else ""),
                encoding="utf-8",
            )
            enriched_images.add(image_path.name)
            total_images += 1
            total_boxes += len(new_lines)

        update_manifest(source_dir, enriched_images)

    print("E-BARD backboard enrichment complete.")
    print(f"Images processed: {total_images}")
    print(f"Backboard boxes added: {total_boxes}")
    print("Original E-BARD labels remain unchanged under labels_ebard/.")
    print("Next step: prepare/review the enriched labels before exporting to local_data/sources/.")


if __name__ == "__main__":
    main()
