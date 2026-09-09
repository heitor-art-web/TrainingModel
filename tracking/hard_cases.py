from __future__ import annotations

from math import hypot


def classify_hard_case(
    row: dict[str, str],
    previous_xy: tuple[float, float] | None,
    frame_diagonal: float,
    low_confidence_threshold: float,
    jump_ratio_threshold: float,
) -> str | None:
    status = row.get("status", "").strip()

    if status == "predicted":
        return "predicted"
    if status == "lost":
        return "lost"

    confidence_text = row.get("confidence", "").strip()
    if status == "detected" and confidence_text:
        if float(confidence_text) < low_confidence_threshold:
            return "low_confidence"

    x_text = row.get("x", "").strip()
    y_text = row.get("y", "").strip()
    if previous_xy is not None and x_text and y_text and frame_diagonal > 0:
        x = float(x_text)
        y = float(y_text)
        distance_ratio = hypot(x - previous_xy[0], y - previous_xy[1]) / frame_diagonal
        if distance_ratio > jump_ratio_threshold:
            return "large_jump"

    return None
