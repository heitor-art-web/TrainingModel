from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import an unpacked E-BARD detection dataset into TrainingModel while "
            "preserving game/source grouping and mapping hoop to rim."
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
        if names is None:
            continue
        try:
            build_class_id_map(names)
        except (ValueError, TypeError):
            continue
        candidates.append((path, data))
    if not candidates:
        raise FileNotFoundError(
            "Could not find an E-BARD YOLO dataset YAML containing basketball, "
            "player, referee and hoop/rim classes."
        )
    return candidates[0]


def collect_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = source_label_path(image_path)
        if label_path is not None:
            pairs.append((image_path, label_path))
    if not pairs:
        raise ValueError(f"No YOLO image/label pairs found under {root}.")
    return pairs


def unique_name(source_dir: Path, image_path: Path) -> str:
    candidate = image_path.name
    if not (source_dir / "images" / candidate).exists():
        return candidate
    token = abs(hash(str(image_path))) % 10_000_000
    return f"{image_path.stem}_{token:07d}{image_path.suffix.lower()}"


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
    pairs = collect_pairs(input_root)

    rows_by_source: dict[str, list[dict]] = {}
    imported = 0
    class_counts = {0: 0, 1: 0, 2: 0, 4: 0}

    for image_path, label_path in pairs:
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

        converted: list[str] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            mapped = remap_yolo_line(raw, class_map)
            if mapped is None:
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
            "dataset": "GabrieleGiudici/E-BARD-detection",
            "license": "CC BY 4.0",
            "backboard_enriched": "false",
            "review_status": "",
        }
        rows_by_source.setdefault(source_id, []).append(row)
        imported += 1

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
                    "notes": "E-BARD hoop is intentionally treated as TrainingModel rim.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("E-BARD import complete.")
    print(f"Frames imported: {imported}")
    print(f"Original games/sources detected: {len(rows_by_source)}")
    print(f"Class instances: {class_counts}")
    print(f"Staging output: {output_root}")
    if len(rows_by_source) < 20 or len(rows_by_source) > 100:
        print(
            "WARNING: detected source count is unexpected for a 60-game dataset. "
            "Inspect the inferred source IDs and re-run with --source-regex if needed."
        )


if __name__ == "__main__":
    main()
