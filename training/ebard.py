from __future__ import annotations

from pathlib import Path
import re

TARGET_CLASS_IDS = {
    "basketball": 0,
    "player": 1,
    "referee": 2,
    "rim": 4,
}

ALIASES = {
    "basketball": "basketball",
    "ball": "basketball",
    "basketballs": "basketball",
    "player": "player",
    "players": "player",
    "referee": "referee",
    "referees": "referee",
    "hoop": "rim",
    "hoops": "rim",
    "rim": "rim",
}


def normalize_class_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return ALIASES.get(cleaned, cleaned)


def build_class_id_map(names: dict | list) -> dict[int, int]:
    """Map E-BARD YOLO class IDs into TrainingModel class IDs."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    result: dict[int, int] = {}
    for source_id, source_name in items:
        normalized = normalize_class_name(str(source_name))
        if normalized in TARGET_CLASS_IDS:
            result[int(source_id)] = TARGET_CLASS_IDS[normalized]
    required = {0, 1, 2, 4}
    if set(result.values()) != required:
        missing = sorted(required - set(result.values()))
        raise ValueError(
            "E-BARD class schema is incomplete for TrainingModel. "
            f"Missing target class IDs: {missing}. Source names: {names!r}"
        )
    return result


def _target_class_id(token: str, class_map: dict[int, int]) -> int | None:
    try:
        source_class = int(token)
    except ValueError:
        normalized = normalize_class_name(token)
        return TARGET_CLASS_IDS.get(normalized)
    return class_map.get(source_class)


def _clip_center_box(values: list[float]) -> tuple[float, float, float, float] | None:
    """Clip normalized center-format xywh to image bounds.

    Returns None if the box has zero area or lies completely outside the image.
    """
    x_center, y_center, box_width, box_height = values
    if box_width <= 0.0 or box_height <= 0.0:
        return None

    x1 = max(0.0, x_center - box_width / 2.0)
    y1 = max(0.0, y_center - box_height / 2.0)
    x2 = min(1.0, x_center + box_width / 2.0)
    y2 = min(1.0, y_center + box_height / 2.0)
    if x2 <= x1 or y2 <= y1:
        return None

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        clipped_width,
        clipped_height,
    )


def remap_yolo_line(
    line: str,
    class_map: dict[int, int],
    image_size: tuple[int, int] | None = None,
) -> str | None:
    """Convert one E-BARD annotation into TrainingModel YOLO format.

    Published E-BARD material can use numeric or named classes, and either
    normalized or absolute-pixel ``x_center y_center width height`` values.
    Absolute coordinates are normalized using the paired image dimensions.
    Slight annotation overflow is clipped to the image boundary. Degenerate or
    completely out-of-frame boxes are skipped.
    """
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) < 5:
        raise ValueError(f"Invalid E-BARD annotation line: {line!r}")

    target_class = _target_class_id(parts[0], class_map)
    if target_class is None:
        return None

    try:
        values = [float(value) for value in parts[1:5]]
    except ValueError as exc:
        raise ValueError(f"Invalid E-BARD box coordinates: {line!r}") from exc

    absolute_pixels = any(abs(value) > 1.0 for value in values)
    if absolute_pixels:
        if image_size is None:
            raise ValueError(
                "Absolute-pixel E-BARD annotation requires image dimensions: "
                f"{line!r}"
            )
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_size!r}")
        x_center, y_center, box_width, box_height = values
        values = [
            x_center / width,
            y_center / height,
            box_width / width,
            box_height / height,
        ]

    clipped = _clip_center_box(values)
    if clipped is None:
        return None
    x_center, y_center, box_width, box_height = clipped

    unchanged = all(abs(a - b) < 1e-12 for a, b in zip(values, clipped))
    if not absolute_pixels and unchanged:
        return " ".join([str(target_class), *parts[1:5]])
    return (
        f"{target_class} {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def infer_game_id(stem: str, source_regex: str | None = None) -> str:
    """Infer a stable game/source ID from an E-BARD image stem."""
    if source_regex:
        match = re.search(source_regex, stem)
        if not match:
            raise ValueError(f"Filename {stem!r} does not match --source-regex.")
        value = match.groupdict().get("game") or (
            match.group(1) if match.groups() else match.group(0)
        )
        if not value:
            raise ValueError(f"Could not extract game ID from {stem!r}.")
        return value

    patterns = (
        r"^(.+?)[_-](?:frame|img|image)[_-]?(\d+)$",
        r"^(.+?)[_-](\d{1,8})$",
    )
    for pattern in patterns:
        match = re.match(pattern, stem, flags=re.IGNORECASE)
        if match and match.group(1):
            return match.group(1)
    raise ValueError(
        f"Could not infer the original E-BARD game from filename {stem!r}. "
        "Re-run with --source-regex containing a capture group for the game ID."
    )


def source_label_path(image_path: Path) -> Path | None:
    """Find the label paired with an image in a conventional images/labels tree."""
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    for index in reversed(image_indices):
        candidate_parts = parts.copy()
        candidate_parts[index] = "labels"
        candidate = Path(*candidate_parts).with_suffix(".txt")
        if candidate.exists():
            return candidate
    sibling = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return sibling if sibling.exists() else None
