from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common import load_model_config

REVIEW_STATUSES = ("corrected", "already_correct", "skip", "uncertain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the local bounding-box reviewer for a prepared batch."
    )
    parser.add_argument("batch", help="Batch name or path under runs/review/.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def resolve_batch(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        direct = PROJECT_ROOT / path
        path = direct if direct.exists() else PROJECT_ROOT / "runs" / "review" / value
    path = path.resolve()
    if not (path / "manifest.csv").exists():
        raise FileNotFoundError(f"Review manifest not found under {path}.")
    return path


def read_manifest(batch_dir: Path) -> list[dict]:
    with (batch_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_manifest(batch_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with (batch_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def yolo_to_boxes(label_path: Path, image_path: Path, class_names: list[str]) -> list[dict]:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[dict] = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if class_id < 0 or class_id >= len(class_names):
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        xmin = max(0.0, (xc - bw / 2) * width)
        ymin = max(0.0, (yc - bh / 2) * height)
        xmax = min(float(width), (xc + bw / 2) * width)
        ymax = min(float(height), (yc + bh / 2) * height)
        boxes.append(
            {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "label": class_names[class_id],
            }
        )
    return boxes


def boxes_to_yolo(annotation: dict | None, image_path: Path, class_names: list[str]) -> str:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size

    lines: list[str] = []
    boxes = (annotation or {}).get("boxes") or []
    for box in boxes:
        label = str(box.get("label", ""))
        if label not in class_names:
            raise ValueError(f"Unknown class label: {label!r}")
        xmin = max(0.0, min(float(width), float(box["xmin"])))
        ymin = max(0.0, min(float(height), float(box["ymin"])))
        xmax = max(0.0, min(float(width), float(box["xmax"])))
        ymax = max(0.0, min(float(height), float(box["ymax"])))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = ((xmin + xmax) / 2) / width
        yc = ((ymin + ymax) / 2) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(
            f"{class_names.index(label)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def row_has_class(batch_dir: Path, row: dict, class_name: str, class_names: list[str]) -> bool:
    if class_name == "all":
        return True
    label_path = batch_dir / row["label_reviewed"]
    try:
        class_id = class_names.index(class_name)
    except ValueError:
        return False
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if parts and parts[0].isdigit() and int(parts[0]) == class_id:
            return True
    return False


def metadata_text(row: dict) -> str:
    lines = [
        f"**Source:** `{row.get('source_id', '')}`",
        f"**Reason:** `{row.get('reason', '') or 'n/a'}`",
        f"**Frame:** `{row.get('frame', '') or 'n/a'}`",
        f"**Original confidence:** `{row.get('original_confidence', '') or 'n/a'}`",
    ]
    raw = row.get("original_detections_json", "")
    if raw:
        try:
            detections = json.loads(raw)
        except json.JSONDecodeError:
            detections = []
        if detections:
            lines.append("**Original detections:**")
            for detection in detections:
                confidence = detection.get("confidence")
                suffix = f" ({float(confidence):.3f})" if confidence is not None else ""
                lines.append(f"- {detection.get('class_name', '?')}{suffix}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    batch_dir = resolve_batch(args.batch)
    rows = read_manifest(batch_dir)
    if not rows:
        raise ValueError("Review batch contains no rows.")

    class_names = [str(name) for name in load_model_config()["classes"]]

    try:
        import gradio as gr
        from gradio_image_annotation import image_annotator
    except ImportError as exc:
        raise RuntimeError(
            "Reviewer dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    reasons = sorted({row.get("reason", "") for row in rows if row.get("reason", "")})

    def filtered_indices(reason: str, class_name: str, status: str) -> list[int]:
        indices = []
        for index, row in enumerate(rows):
            if reason != "all" and row.get("reason", "") != reason:
                continue
            row_status = row.get("review_status", "")
            if status == "unreviewed" and row_status:
                continue
            if status not in {"all", "unreviewed"} and row_status != status:
                continue
            if not row_has_class(batch_dir, row, class_name, class_names):
                continue
            indices.append(index)
        return indices

    def load_index(index: int):
        row = rows[index]
        image_path = batch_dir / "images" / row["image"]
        label_path = batch_dir / row["label_reviewed"]
        annotation = {
            "image": str(image_path),
            "boxes": yolo_to_boxes(label_path, image_path, class_names),
        }
        status = row.get("review_status", "") or None
        return annotation, metadata_text(row), status

    def choose_start(indices: list[int]) -> int:
        for position, index in enumerate(indices):
            if not rows[index].get("review_status", ""):
                return position
        return 0

    def apply_filters(reason: str, class_name: str, status: str):
        indices = filtered_indices(reason, class_name, status)
        if not indices:
            raise gr.Error("No frames match these filters.")
        pointer = choose_start(indices)
        annotation, info, review_status = load_index(indices[pointer])
        return indices, pointer, annotation, info, review_status, f"{pointer + 1}/{len(indices)}"

    def save_current(indices: list[int], pointer: int, annotation: dict, review_status: str):
        if not indices:
            raise gr.Error("No active review frame.")
        if review_status not in REVIEW_STATUSES:
            raise gr.Error("Choose a review status before leaving this frame.")
        index = indices[pointer]
        row = rows[index]
        image_path = batch_dir / "images" / row["image"]
        reviewed_path = batch_dir / row["label_reviewed"]
        reviewed_path.write_text(
            boxes_to_yolo(annotation, image_path, class_names), encoding="utf-8"
        )
        row["review_status"] = review_status
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(batch_dir, rows)

    def move(direction: int, indices: list[int], pointer: int, annotation: dict, review_status: str):
        save_current(indices, pointer, annotation, review_status)
        new_pointer = max(0, min(len(indices) - 1, pointer + direction))
        new_annotation, info, new_status = load_index(indices[new_pointer])
        return new_pointer, new_annotation, info, new_status, f"{new_pointer + 1}/{len(indices)}"

    initial_indices = filtered_indices("all", "all", "unreviewed")
    if not initial_indices:
        initial_indices = list(range(len(rows)))
    initial_pointer = choose_start(initial_indices)
    initial_annotation, initial_info, initial_status = load_index(
        initial_indices[initial_pointer]
    )

    with gr.Blocks(title="TrainingModel label reviewer") as demo:
        gr.Markdown(
            "# TrainingModel label reviewer\n"
            "Edits are saved only to `labels_reviewed/`; original labels remain unchanged."
        )
        with gr.Row():
            reason_filter = gr.Dropdown(["all", *reasons], value="all", label="Reason")
            class_filter = gr.Dropdown(["all", *class_names], value="all", label="Class")
            status_filter = gr.Dropdown(
                ["all", "unreviewed", *REVIEW_STATUSES],
                value="unreviewed",
                label="Review status",
            )
            apply_button = gr.Button("Apply filters")

        indices_state = gr.State(initial_indices)
        pointer_state = gr.State(initial_pointer)

        annotator = image_annotator(
            value=initial_annotation,
            label_list=class_names,
            box_min_size=3,
            height=700,
            image_type="filepath",
        )
        with gr.Row():
            review_status = gr.Radio(
                list(REVIEW_STATUSES), value=initial_status, label="Frame status"
            )
            progress = gr.Markdown(f"{initial_pointer + 1}/{len(initial_indices)}")
        info = gr.Markdown(initial_info)

        with gr.Row():
            previous_button = gr.Button("Previous")
            save_button = gr.Button("Save")
            next_button = gr.Button("Save + Next", variant="primary")

        apply_button.click(
            apply_filters,
            inputs=[reason_filter, class_filter, status_filter],
            outputs=[indices_state, pointer_state, annotator, info, review_status, progress],
        )
        save_button.click(
            save_current,
            inputs=[indices_state, pointer_state, annotator, review_status],
            outputs=[],
        )
        previous_button.click(
            lambda indices, pointer, annotation, status: move(
                -1, indices, pointer, annotation, status
            ),
            inputs=[indices_state, pointer_state, annotator, review_status],
            outputs=[pointer_state, annotator, info, review_status, progress],
        )
        next_button.click(
            lambda indices, pointer, annotation, status: move(
                1, indices, pointer, annotation, status
            ),
            inputs=[indices_state, pointer_state, annotator, review_status],
            outputs=[pointer_state, annotator, info, review_status, progress],
        )

    demo.launch(server_name=args.host, server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
