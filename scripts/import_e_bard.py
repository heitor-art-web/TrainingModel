from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.ebard import (
    build_class_id_map,
    infer_game_id,
    remap_yolo_line,
    source_label_path,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_KEYS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the YAML-defined YOLO E-BARD detection splits into TrainingModel "
            "while preserving game/source grouping and mapping hoop to rim."
        )
    )
    parser.add_argument(
        "--input",
        default="local_data/external/e_bard",
        help="Unpacked E-BARD detection directory.",
    )
    parser.add_argument(
        "--output",
        default="local_data/ebard_import/sources",
        help="Staging directory for source-grouped E-BARD data.",
    )
    parser.add_argument(
        "--source-regex",
        help=(
            "Optional regex for extracting the original game ID from each image stem. "
            "Use a named (?P<game>...) group or the first capture group."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing E-BARD staging directory.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_dataset_yaml(root: Path) -> tuple[Path, dict]:
    candidates: list[tuple[Path, dict]] = []
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = data.get("names")
        if names is None or not any(data.get(key) for key in SPLIT_KEYS):
            continue
        try:
            build_class_id_map(names)
        except (ValueError, TypeError):
            continue
        candidates.append((path, data))
    if not candidates:
        raise FileNotFoundError(
            "Could not find an E-BARD dataset YAML containing basketball, player, "
            "referee and hoop/rim classes plus train/val/test paths."
        )

    # Prefer the authors' YOLO configuration when several compatible YAML files exist.
    candidates.sort(
        key=lambda item: (
            0 if "yolo" in {part.lower() for part in item[0].parts} else 1,
            len(item[0].parts),
            str(item[0]),
        )
    )
    return candidates[0]


def _candidate_dataset_roots(yaml_path: Path, config: dict) -> list[Path]:
    roots: list[Path] = []
    declared = config.get("path")
    if declared:
        declared_path = Path(str(declared)).expanduser()
        if declared_path.is_absolute():
            if declared_path.exists():
                roots.append(declared_path.resolve())
        else:
            roots.append((yaml_path.parent / declared_path).resolve())
    roots.append(yaml_path.parent.resolve())

    # Some exported YAML files retain a cluster path even though the archive itself
    # is self-contained. Walking a few parents gives us a safe local fallback.
    parent = yaml_path.parent
    for _ in range(4):
        parent = parent.parent
        roots.append(parent.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _resolve_split_path(yaml_path: Path, config: dict, value: str) -> Path:
    split_path = Path(str(value)).expanduser()
    if split_path.is_absolute() and split_path.exists():
        return split_path.resolve()

    for root in _candidate_dataset_roots(yaml_path, config):
        candidate = (root / split_path).resolve() if not split_path.is_absolute() else root / split_path.name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve E-BARD split path {value!r} from YAML {yaml_path}."
    )


def collect_pairs_from_yaml(yaml_path: Path, config: dict) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    seen_images: set[Path] = set()

    for split in SPLIT_KEYS:
        value = config.get(split)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        for split_value in values:
            images_root = _resolve_split_path(yaml_path, config, str(split_value))
            image_paths = (
                [images_root]
                if images_root.is_file()
                else sorted(path for path in images_root.rglob("*") if path.is_file())
            )
            for image_path in image_paths:
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                resolved_image = image_path.resolve()
                if resolved_image in seen_images:
                    continue
                label_path = source_label_path(image_path)
                if label_path is None:
                    raise FileNotFoundError(
                        f"Missing YOLO label for E-BARD image referenced by {split}: {image_path}"
                    )
                seen_images.add(resolved_image)
                pairs.append((image_path, label_path, split))

    if not pairs:
        raise ValueError(f"No E-BARD image/label pairs found via {yaml_path}.")
    return pairs


def unique_name(source_dir: Path, image_path: Path) -> str:
    candidate = image_path.name
    if not (source_dir / "images" / candidate).exists():
        return candidate
    token = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{token}{image_path.suffix.lower()}"


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input)
    output_root = resolve(args.output)
    if not input_root.exists():
        raise FileNotFoundError(
            f"E-BARD input not found: {input_root}. Download and unzip all.zip first."
        )

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --force to rebuild it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    yaml_path, dataset_config = find_dataset_yaml(input_root)
    class_map = build_class_id_map(dataset_config["names"])
    pairs = collect_pairs_from_yaml(yaml_path, dataset_config)

    rows_by_source: dict[str, list[dict]] = {}
    imported = 0
    class_counts = {0: 0, 1: 0, 2: 0, 4: 0}
    normalized_lines = 0
    pixel_lines = 0
    skipped_boxes = 0
    split_counts = {key: 0 for key in SPLIT_KEYS}

    for image_path, label_path, original_split in pairs:
        game_id = infer_game_id(image_path.stem, args.source_regex)
        source_id = f"ebard_{game_id}"
        source_dir = output_root / source_id
        images_dir = source_dir / "images"
        labels_original_dir = source_dir / "labels_ebard"
        labels_dir = source_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_original_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        target_name = unique_name(source_dir, image_path)
        target_image = images_dir / target_name
        target_label_original = labels_original_dir / f"{Path(target_name).stem}.txt"
        target_label = labels_dir / f"{Path(target_name).stem}.txt"

        with Image.open(image_path) as image:
            image_size = image.size

        converted: list[str] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if not parts:
                continue
            try:
                raw_values = [float(value) for value in parts[1:5]]
            except (ValueError, IndexError):
                raw_values = []
            if raw_values and any(value > 1.0 for value in raw_values):
                pixel_lines += 1
            elif raw_values:
                normalized_lines += 1

            mapped = remap_yolo_line(raw, class_map, image_size=image_size)
            if mapped is None:
                skipped_boxes += 1
                continue
            converted.append(mapped)
            class_counts[int(mapped.split()[0])] += 1

        shutil.copy2(image_path, target_image)
        text = "\n".join(converted) + ("\n" if converted else "")
        target_label_original.write_text(text, encoding="utf-8")
        target_label.write_text(text, encoding="utf-8")

        row = {
            "source_id": source_id,
            "image": target_name,
            "label": f"labels/{Path(target_name).stem}.txt",
            "label_ebard": f"labels_ebard/{Path(target_name).stem}.txt",
            "origin_image": str(image_path.relative_to(input_root)),
            "origin_label": str(label_path.relative_to(input_root)),
            "origin_split": original_split,
            "dataset": "GabrieleGiudici/E-BARD-detection",
            "license": "CC BY 4.0",
            "backboard_enriched": "false",
            "review_status": "",
        }
        rows_by_source.setdefault(source_id, []).append(row)
        imported += 1
        split_counts[original_split] += 1

    for source_id, rows in rows_by_source.items():
        source_dir = output_root / source_id
        with (source_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (source_dir / "source.json").write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "dataset": "GabrieleGiudici/E-BARD-detection",
                    "license": "CC BY 4.0",
                    "source_yaml": str(yaml_path.relative_to(input_root)),
                    "class_mapping": {
                        "basketball": 0,
                        "player": 1,
                        "referee": 2,
                        "hoop": 4,
                    },
                    "notes": (
                        "Only image paths explicitly referenced by the E-BARD YOLO YAML "
                        "are imported. E-BARD hoop is intentionally treated as "
                        "TrainingModel rim."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("E-BARD import complete.")
    print(f"Source YAML: {yaml_path.relative_to(input_root)}")
    print(f"Frames imported: {imported}")
    print(f"Original split frames: {split_counts}")
    print(f"Original games/sources detected: {len(rows_by_source)}")
    print(f"Class instances: {class_counts}")
    print(
        "Annotation lines: "
        f"normalized={normalized_lines}, absolute_pixels={pixel_lines}, skipped={skipped_boxes}"
    )
    print(f"Staging output: {output_root}")
    if imported < 1700 or imported > 1900:
        print(
            "WARNING: E-BARD officially reports 1,800 detection frames; inspect the "
            "selected YAML/split paths before enrichment."
        )
    if len(rows_by_source) < 20 or len(rows_by_source) > 100:
        print(
            "WARNING: detected source count is unexpected for a 60-game dataset. "
            "Inspect the inferred source IDs and re-run with --source-regex if needed."
        )


if __name__ == "__main__":
    main()
