from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPROVED = {"corrected", "already_correct"}
NATIVE_CLASS_IDS = {0, 1, 2, 4}  # basketball, player, referee, rim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote staged Basketball Game Detections sources into local_data/sources. "
            "By default the final QA sample must be fully reviewed; --skip-qa explicitly "
            "promotes the staged labels without applying a final review batch."
        )
    )
    parser.add_argument("--root", default="local_data/bgd_import/sources")
    parser.add_argument("--batch", default="bgd_qa")
    parser.add_argument("--destination-root", default="local_data/sources")
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help=(
            "Explicitly skip the final BGD QA gate and promote staged labels as-is. "
            "No reviewed labels are applied."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def native_label_text(path: Path) -> str:
    """Return only BGD-native TrainingModel classes.

    Basketball Game Detections does not provide backboard annotations. A reviewer may
    add class 3 while inspecting a QA sample, but applying those labels to only the
    sampled frames would create inconsistent supervision because unsampled frames
    with visible backboards would remain unlabeled. Backboard therefore requires a
    separate, comprehensive enrichment step and is deliberately excluded here.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO class id in {path}: {parts[0]!r}") from exc
        if class_id in NATIVE_CLASS_IDS:
            lines.append(raw)
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    destination_root = resolve(args.destination_root)
    batch_dir = PROJECT_ROOT / "runs" / "review" / args.batch
    manifest_path = batch_dir / "manifest.csv"

    if not root.exists():
        raise FileNotFoundError(f"Basketball Game Detections staging root not found: {root}")

    review_rows: list[dict] = []
    reviewed_labels: dict[tuple[str, str], Path] = {}
    review_statuses: dict[tuple[str, str], str] = {}

    if not args.skip_qa:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Basketball Game Detections QA batch not found: {manifest_path}. "
                "Run the QA workflow or explicitly pass --skip-qa."
            )

        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            review_rows = list(csv.DictReader(file))
        if not review_rows:
            raise ValueError("Basketball Game Detections QA batch is empty.")

        pending = [row for row in review_rows if row.get("review_status") not in APPROVED]
        if pending:
            states = sorted({row.get("review_status", "") or "unreviewed" for row in pending})
            raise ValueError(
                "Basketball Game Detections cannot be finalized until every QA sample is "
                "corrected or already_correct. "
                f"Outstanding states: {states}. "
                "If you intentionally want to promote staged labels without final QA, "
                "rerun with --skip-qa."
            )

        for row in review_rows:
            source_id = row["source_id"]
            batch_name = row["image"]
            prefix = f"{source_id}__"
            if not batch_name.startswith(prefix):
                raise ValueError(f"Unexpected Basketball Game Detections QA filename: {batch_name}")
            original_name = batch_name[len(prefix) :]
            key = (source_id, Path(original_name).stem)
            reviewed_labels[key] = batch_dir / row["label_reviewed"]
            review_statuses[key] = row["review_status"]

    destination_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    reviewed_applied = 0
    removed_backboard_boxes = 0

    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target = destination_root / source_dir.name
        if target.exists():
            if not args.force:
                raise FileExistsError(
                    f"Curated source already exists: {target}. Use --force to replace it."
                )
            shutil.rmtree(target)

        images_out = target / "images"
        labels_out = target / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        source_metadata = source_dir / "source.json"
        if source_metadata.exists():
            shutil.copy2(source_metadata, target / "source.json")

        staged_manifest = source_dir / "manifest.csv"
        staged_rows: dict[str, dict] = {}
        if staged_manifest.exists():
            with staged_manifest.open("r", encoding="utf-8", newline="") as file:
                staged_rows = {row["image"]: row for row in csv.DictReader(file)}

        manifest_rows: list[dict] = []
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file()):
            source_label = labels_dir / f"{image_path.stem}.txt"
            key = (source_dir.name, image_path.stem)
            review_label = reviewed_labels.get(key)
            label_to_copy = review_label if review_label is not None else source_label
            if not label_to_copy.exists():
                continue

            shutil.copy2(image_path, images_out / image_path.name)
            output_label = labels_out / f"{image_path.stem}.txt"
            if review_label is not None:
                raw_lines = [
                    line
                    for line in review_label.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                native_text = native_label_text(review_label)
                native_lines = [line for line in native_text.splitlines() if line.strip()]
                removed_backboard_boxes += len(raw_lines) - len(native_lines)
                output_label.write_text(native_text, encoding="utf-8")
            else:
                # Staged BGD labels already contain only the mapped native classes.
                # In --skip-qa mode they are intentionally promoted unchanged.
                shutil.copy2(source_label, output_label)
            reviewed_applied += int(review_label is not None)

            source_row = staged_rows.get(image_path.name, {})
            if args.skip_qa:
                final_review_status = "qa_skipped_source"
            elif review_label is not None:
                final_review_status = review_statuses[key]
            else:
                final_review_status = "qa_sampled_source"

            manifest_rows.append(
                {
                    **source_row,
                    "image": image_path.name,
                    "source_id": source_dir.name,
                    "label": f"labels/{image_path.stem}.txt",
                    "review_status": final_review_status,
                }
            )
            exported += 1

        if manifest_rows:
            fieldnames: list[str] = []
            for row in manifest_rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with (target / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)

    print("Basketball Game Detections finalized into curated sources.")
    print(f"Frames exported: {exported}")
    print(f"Final QA skipped: {'yes' if args.skip_qa else 'no'}")
    print(f"QA-reviewed labels applied: {reviewed_applied}")
    print(f"Non-native backboard boxes excluded from QA labels: {removed_backboard_boxes}")
    print(f"Destination: {destination_root}")
    print("Sources are now eligible for scripts/build_dataset.py.")


if __name__ == "__main__":
    main()
