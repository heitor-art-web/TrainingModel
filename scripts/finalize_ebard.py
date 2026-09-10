from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED = {"corrected", "already_correct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote staged E-BARD sources into local_data/sources after the "
            "backboard QA sample has been fully reviewed."
        )
    )
    parser.add_argument("--root", default="local_data/ebard_import/sources")
    parser.add_argument("--batch", default="ebard_backboard_qa")
    parser.add_argument("--destination-root", default="local_data/sources")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    destination_root = resolve(args.destination_root)
    batch_dir = PROJECT_ROOT / "runs" / "review" / args.batch
    manifest_path = batch_dir / "manifest.csv"
    if not root.exists():
        raise FileNotFoundError(f"E-BARD staging root not found: {root}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"E-BARD QA batch not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        review_rows = list(csv.DictReader(file))
    if not review_rows:
        raise ValueError("E-BARD QA batch is empty.")
    pending = [row for row in review_rows if row.get("review_status") not in APPROVED]
    if pending:
        states = sorted({row.get("review_status", "") or "unreviewed" for row in pending})
        raise ValueError(
            "E-BARD cannot be finalized until every QA sample is corrected or already_correct. "
            f"Outstanding states: {states}"
        )

    reviewed_labels: dict[tuple[str, str], Path] = {}
    for row in review_rows:
        source_id = row["source_id"]
        batch_name = row["image"]
        prefix = f"{source_id}__"
        if not batch_name.startswith(prefix):
            raise ValueError(f"Unexpected E-BARD QA filename: {batch_name}")
        original_name = batch_name[len(prefix):]
        reviewed_labels[(source_id, Path(original_name).stem)] = batch_dir / row["label_reviewed"]

    destination_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    corrected = 0
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target = destination_root / source_dir.name
        if target.exists():
            if not args.force:
                raise FileExistsError(
                    f"Curated source already exists: {target}. Use --force to replace E-BARD sources."
                )
            shutil.rmtree(target)
        images_out = target / "images"
        labels_out = target / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict] = []
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
            source_label = labels_dir / f"{image_path.stem}.txt"
            review_label = reviewed_labels.get((source_dir.name, image_path.stem))
            label_to_copy = review_label if review_label is not None else source_label
            if not label_to_copy.exists():
                continue
            shutil.copy2(image_path, images_out / image_path.name)
            shutil.copy2(label_to_copy, labels_out / f"{image_path.stem}.txt")
            status = "corrected" if review_label is not None else "qa_sampled_source"
            corrected += int(review_label is not None)
            manifest_rows.append(
                {
                    "image": image_path.name,
                    "source_id": source_dir.name,
                    "reason": "ebard_import",
                    "review_status": status,
                    "reviewed_at": "",
                    "frame": "",
                    "time_seconds": "",
                }
            )
            exported += 1

        if manifest_rows:
            with (target / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

    print("E-BARD finalized into curated sources.")
    print(f"Frames exported: {exported}")
    print(f"QA-reviewed labels applied: {corrected}")
    print(f"Destination: {destination_root}")
    print("E-BARD sources are now eligible for scripts/build_dataset.py.")


if __name__ == "__main__":
    main()
