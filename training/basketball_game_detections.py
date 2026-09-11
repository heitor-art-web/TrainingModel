from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from training.ebard import source_label_path

SPLIT_KEYS = ("train", "val", "test", "valid", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TARGET_CLASS_IDS = {
    "ball": 0,
    "basketball": 0,
    "player": 1,
    "players": 1,
    "ref": 2,
    "referee": 2,
    "referees": 2,
    "hoop": 4,
    "rim": 4,
}

IGNORED_CLASS_NAMES = {
    "fg attempt",
    "fg made",
    "field goal attempt",
    "field goal made",
}


@dataclass(frozen=True)
class YoloObject:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def corners(self) -> tuple[float, float, float, float]:
        return (
            self.x_center - self.width / 2.0,
            self.y_center - self.height / 2.0,
            self.x_center + self.width / 2.0,
            self.y_center + self.height / 2.0,
        )


def normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def names_to_dict(names: dict | list) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def build_class_map(names: dict | list) -> tuple[dict[int, int], set[int]]:
    """Return source->TrainingModel class mapping plus intentionally ignored IDs."""
    mapping: dict[int, int] = {}
    ignored: set[int] = set()
    for source_id, source_name in names_to_dict(names).items():
        normalized = normalize_class_name(source_name)
        if normalized in TARGET_CLASS_IDS:
            mapping[source_id] = TARGET_CLASS_IDS[normalized]
        elif normalized in IGNORED_CLASS_NAMES:
            ignored.add(source_id)

    required = {0, 1, 2, 4}
    missing = sorted(required - set(mapping.values()))
    if missing:
        raise ValueError(
            "Basketball Game Detections class schema is missing required target "
            f"classes {missing}. Source names: {names!r}"
        )
    return mapping, ignored


def find_dataset_yaml(root: Path) -> tuple[Path, dict]:
    candidates: list[tuple[Path, dict]] = []
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if "names" not in config:
            continue
        try:
            build_class_map(config["names"])
        except (TypeError, ValueError):
            continue
        if not any(config.get(key) for key in SPLIT_KEYS):
            continue
        candidates.append((path, config))
    if not candidates:
        raise FileNotFoundError(
            "Could not find a YOLO dataset YAML containing Ball, Player, Ref and Hoop/Rim."
        )
    candidates.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    return candidates[0]


def _candidate_roots(yaml_path: Path, config: dict) -> list[Path]:
    roots: list[Path] = []
    declared = config.get("path")
    if declared:
        declared_path = Path(str(declared)).expanduser()
        roots.append(
            declared_path.resolve()
            if declared_path.is_absolute()
            else (yaml_path.parent / declared_path).resolve()
        )
    roots.append(yaml_path.parent.resolve())
    parent = yaml_path.parent
    for _ in range(4):
        parent = parent.parent
        roots.append(parent.resolve())
    result: list[Path] = []
    for root in roots:
        if root not in result:
            result.append(root)
    return result


def _resolve_split_path(yaml_path: Path, config: dict, value: str) -> Path:
    raw = Path(str(value)).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()
    for root in _candidate_roots(yaml_path, config):
        candidate = (root / raw).resolve() if not raw.is_absolute() else root / raw.name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve dataset split path {value!r} from {yaml_path}.")


def collect_pairs_from_yaml(
    yaml_path: Path, config: dict
) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    seen: set[Path] = set()
    canonical = {"valid": "val", "validation": "val"}
    for key in SPLIT_KEYS:
        value = config.get(key)
        if not value:
            continue
        values = value if isinstance(value, list) else [value]
        for split_value in values:
            images_root = _resolve_split_path(yaml_path, config, str(split_value))
            candidates = (
                [images_root]
                if images_root.is_file()
                else sorted(path for path in images_root.rglob("*") if path.is_file())
            )
            for image_path in candidates:
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                resolved = image_path.resolve()
                if resolved in seen:
                    continue
                label_path = source_label_path(image_path)
                if label_path is None:
                    raise FileNotFoundError(f"Missing YOLO label for {image_path}")
                seen.add(resolved)
                pairs.append((image_path, label_path, canonical.get(key, key)))
    if not pairs:
        raise ValueError(f"No image/label pairs found via {yaml_path}.")
    return pairs


def _clip_box(values: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    xc, yc, width, height = values
    if width <= 0 or height <= 0:
        return None
    x1 = max(0.0, xc - width / 2.0)
    y1 = max(0.0, yc - height / 2.0)
    x2 = min(1.0, xc + width / 2.0)
    y2 = min(1.0, yc + height / 2.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def parse_and_remap_labels(
    label_path: Path, class_map: dict[int, int], ignored_ids: set[int]
) -> list[YoloObject]:
    result: list[YoloObject] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{label_path}: line {line_number} must contain at least 5 values")
        source_class = int(parts[0])
        if source_class in ignored_ids or source_class not in class_map:
            continue
        values = tuple(float(value) for value in parts[1:5])
        if any(abs(value) > 1.0 for value in values):
            raise ValueError(
                f"{label_path}: line {line_number} uses non-normalized coordinates; "
                "export the source dataset in YOLO normalized format."
            )
        clipped = _clip_box(values)
        if clipped is None:
            continue
        result.append(YoloObject(class_map[source_class], *clipped))
    return result


def yolo_line(obj: YoloObject) -> str:
    return (
        f"{obj.class_id} {obj.x_center:.6f} {obj.y_center:.6f} "
        f"{obj.width:.6f} {obj.height:.6f}"
    )


def ball_size_bucket(objects: list[YoloObject]) -> str:
    balls = [obj for obj in objects if obj.class_id == 0]
    if not balls:
        return "none"
    linear_scale = min((obj.area ** 0.5 for obj in balls), default=1.0)
    if linear_scale <= 0.02:
        return "tiny"
    if linear_scale <= 0.04:
        return "small"
    if linear_scale <= 0.08:
        return "medium"
    return "large"


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def infer_source_group(stem: str, source_regex: str | None = None) -> str | None:
    """Only infer a source when the caller provides a deliberate regex.

    Roboflow filenames are not reliable proof of original game/video identity. We
    therefore avoid heuristic grouping that could create train/test leakage.
    """
    if not source_regex:
        return None
    match = re.search(source_regex, stem)
    if not match:
        return None
    value = match.groupdict().get("source") or (
        match.group(1) if match.groups() else match.group(0)
    )
    return value or None
