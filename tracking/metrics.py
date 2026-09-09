from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def _longest_streak(statuses: list[str], accepted: set[str]) -> int:
    longest = 0
    current = 0
    for status in statuses:
        if status in accepted:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_trajectory(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    statuses = [row["status"].strip() for row in rows]
    counts = Counter(statuses)
    confidences = [
        float(row["confidence"])
        for row in rows
        if row.get("confidence", "").strip()
    ]
    total = len(rows)
    detected = counts.get("detected", 0)
    predicted = counts.get("predicted", 0)
    lost = counts.get("lost", 0)

    return {
        "total_frames": total,
        "detected_frames": detected,
        "predicted_frames": predicted,
        "lost_frames": lost,
        "detected_ratio": detected / total if total else 0.0,
        "predicted_ratio": predicted / total if total else 0.0,
        "lost_ratio": lost / total if total else 0.0,
        "continuity_ratio": (detected + predicted) / total if total else 0.0,
        "mean_detection_confidence": mean(confidences) if confidences else None,
        "longest_predicted_streak": _longest_streak(statuses, {"predicted"}),
        "longest_lost_streak": _longest_streak(statuses, {"lost"}),
        "longest_continuous_track": _longest_streak(statuses, {"detected", "predicted"}),
    }


def write_tracking_report(csv_path: str | Path, output_path: str | Path) -> dict:
    summary = summarize_trajectory(csv_path)
    target = Path(output_path)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
