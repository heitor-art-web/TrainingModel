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
    """Map E-BARD YOLO class IDs into TrainingModel class IDs.

    E-BARD `hoop` is intentionally mapped to TrainingModel `rim` (class 4).
    Unknown source classes are ignored by the importer.
    """
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
    """Resolve either a numeric source class ID or an E-BARD class name."""
    try:
        source_class = int(token)
    except ValueError:
        normalized = normalize_class_name(token)
        return TARGET_CLASS_IDS.get(normalized)
    return class_map.get(source_class)


def remap_yolo_line(
    line: str,
    class_map: dict[int, int],
    image_size: tuple[int, int] | None = None,
) -> str | None:
    """Convert one E-BARD annotation into TrainingModel YOLO format.

    E-BARD material encountered in the published archive can use either numeric
    or named classes. Coordinates can already be normalized YOLO values, or
    absolute-pixel ``x_center y_center width height`` values. Pixel coordinates
    are normalized using the paired image dimensions.

    Degenerate zero-area boxes are ignored instead of aborting the complete
    import, because they are unusable for YOLO training.
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

    if any(value < 0.0 for value in values):
        raise ValueError(f"Negative E-BARD box coordinate: {line!r}")

    absolute_pixels = any(value > 1.0 for value in values)
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
        if box_width <= 0.0 or box_height <= 0.0:
            return None
        values = [
            x_center / width,
            y_center / height,
            box_width / width,
            box_height / height,
        ]

    x_center, y_center, box_width, box_height = values
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        raise ValueError(f"E-BARD box center outside image: {line!r}")
    if not (box_width <= 1.0 and box_height <= 1.0):
        raise ValueError(f"E-BARD box size outside image: {line!r}")

    if not absolute_pixels:
        return " ".join([str(target_class), *parts[1:5]])
    return (
        f"{target_class} {x_center:.6f} {y_center:.6f} "
        f"{box_width:.6f} {box_height:.6f}"
    )


def infer_game_id(stem: str, source_regex: str | None = None) -> str:
    """Infer a stable game/source ID from an E-BARD image stem.

    A user supplied regex takes precedence and must expose either a named `game`
    group or a first capture group. The automatic mode removes common frame/index
    suffixes while keeping the rest of the filename intact.
    """
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
