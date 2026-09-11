from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.basketball_game_detections import (
    build_class_map,
    find_dataset_yaml,
    parse_and_remap_labels,
    yolo_line,
)
from training.data import sanitize_source_id

DATASET_NAME = "Basketball Game Detections"
DATASET_URL = "https://universe.roboflow.com/firsttime/basketball-game-detections"
DATASET_LICENSE = "CC BY 4.0"
DATASET_ATTRIBUTION = "FirstTime / Roboflow Universe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import only the images selected by analyze_basketball_game_detections.py "
            "into TrainingModel curated sources."
        )
    )
    parser.add_argument(
        "--input",
        default="local_data/external/basketball_game_detections",
        help="Unpacked YOLO export used by the analyzer.",
    )
    parser.add_argument(
        "--selection",
        default="runs/dataset_analysis/basketball_game_detections/selected_manifest.csv",
    )
    parser.add_argument("--output", default="local_data/sources")
    parser.add_argument(
        "--source-id",
        default="bgd_training_pool",
        help=(
            "Source ID used when original game/source identity is unknown. This source "
            "is registered as training-only in configs/data_sources.toml."
        ),
    )
    parser.add_argument(
        "--use-proven-source-groups",
        action="store_true",
        help=(
            "Create one source per source_group from the analysis manifest. Only use "
            "this when --source-regex was based on reliable original-game metadata."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def safe_join(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Selection path escapes dataset root: {relative!r}") from exc
    return candidate


def unique_name(images_dir: Path, image_path: Path) -> str:
    candidate = image_path.name
    if not (images_dir / candidate).exists():
        return candidate
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem)
    index = 2
    while (images_dir / f"{stem}_{index}{image_path.suffix.lower()}").exists():
        index += 1
    return f"{stem}_{index}{image_path.suffix.lower()}"


def source_for_row(row: dict, args: argparse.Namespace) -> str:
    if not args.use_proven_source_groups:
        return sanitize_source_id(args.source_id)
    group = (row.get("source_group") or "").strip()
    if not group:
        raise ValueError(
            "--use-proven-source-groups requires every selected row to contain a "
            "source_group. Re-run analysis with a reliable --source-regex or import "
            "as the default training-only pool."
        )
    return sanitize_source_id(f"bgd_{group}")


def main() -> None:
    args = parse_args()
    input_root = resolve(args.input)
    selection_path = resolve(args.selection)
    output_root = resolve(args.output)
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset input not found: {input_root}")
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Selection manifest not found: {selection_path}. Run the analyzer first."
        )

    yaml_path, config = find_dataset_yaml(input_root)
    class_map, ignored_ids = build_class_map(config["names"])
    with selection_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if row.get("selected", "true").lower() == "true"]
    if not rows:
        raise ValueError(f"No selected images found in {selection_path}.")

    rows_by_source: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_source.setdefault(source_for_row(row, args), []).append(row)

    existing = [output_root / source_id for source_id in rows_by_source if (output_root / source_id).exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Curated source already exists: "
            + ", ".join(str(path) for path in existing)
            + ". Use --force to replace only these Basketball Game Detections sources."
        )
    for path in existing:
        shutil.rmtree(path)

    total_images = 0
    class_counts = {0: 0, 1: 0, 2: 0, 4: 0}
    for source_id, source_rows in rows_by_source.items():
        source_dir = output_root / source_id
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict] = []

        for row in source_rows:
            image_path = safe_join(input_root, row["image"])
            label_path = safe_join(input_root, row["label"])
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Selected pair missing: image={image_path} label={label_path}"
                )
            objects = parse_and_remap_labels(label_path, class_map, ignored_ids)
            for obj in objects:
                class_counts[obj.class_id] += 1

            target_name = unique_name(images_dir, image_path)
            shutil.copy2(image_path, images_dir / target_name)
            target_label = labels_dir / f"{Path(target_name).stem}.txt"
            text = "\n".join(yolo_line(obj) for obj in objects)
            target_label.write_text(text + ("\n" if text else ""), encoding="utf-8")

            manifest_rows.append(
                {
                    "source_id": source_id,
                    "image": target_name,
                    "label": f"labels/{target_label.name}",
                    "origin_image": row["image"],
                    "origin_label": row["label"],
                    "origin_split": row.get("original_split", ""),
                    "selection_bucket": row.get("selection_bucket", ""),
                    "selection_score": row.get("selection_score", ""),
                    "source_group": row.get("source_group", ""),
                    "dataset": DATASET_NAME,
                    "dataset_url": DATASET_URL,
                    "attribution": DATASET_ATTRIBUTION,
                    "license": DATASET_LICENSE,
                    "review_status": "",
                }
            )
            total_images += 1

        manifest_path = source_dir / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        source_json = {
            "source_id": source_id,
            "dataset": DATASET_NAME,
            "source": DATASET_URL,
            "attribution": DATASET_ATTRIBUTION,
            "license": DATASET_LICENSE,
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_yaml": str(yaml_path.relative_to(input_root)),
            "selection_manifest": str(selection_path),
            "import_mode": (
                "proven_source_groups" if args.use_proven_source_groups else "training_only_pool"
            ),
            "class_mapping": {
                "Ball": 0,
                "Player": 1,
                "Ref": 2,
                "Hoop": 4,
                "FG Attempt": "ignored",
                "FG Made": "ignored",
            },
            "notes": (
                "TrainingModel canonicalizes Hoop as rim. Backboard is not natively "
                "annotated by this dataset. Selected data was transformed/remapped "
                "from the original dataset; original upstream content is not redistributed "
                "by this Git repository."
            ),
        }
        (source_dir / "source.json").write_text(
            json.dumps(source_json, indent=2), encoding="utf-8"
        )

    print("Basketball Game Detections import complete.")
    print(f"Selected images imported: {total_images}")
    print(f"Curated sources created: {len(rows_by_source)}")
    print(f"Class instances: {class_counts}")
    if not args.use_proven_source_groups:
        print(
            "Import mode: training-only pool. Keep bgd_training_pool registered with "
            "role=train to prevent external-source leakage into validation/test."
        )
    print(f"Destination: {output_root}")


if __name__ == "__main__":
    main()
