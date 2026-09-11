from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUOTAS = {
    "tiny_ball": 180,
    "small_ball": 420,
    "model_fn": 1050,
    "model_fp": 300,
    "no_ball_hard_negative": 350,
    "ball_and_rim": 550,
    "other": 350,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced v1.2.0 selection from an existing Basketball Game "
            "Detections analysis_manifest.csv without re-running model inference."
        )
    )
    parser.add_argument(
        "--analysis",
        default="runs/dataset_analysis/basketball_game_detections/analysis_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="runs/dataset_analysis/basketball_game_detections/refined_selection_manifest.csv",
    )
    parser.add_argument(
        "--summary",
        default="runs/dataset_analysis/basketball_game_detections/refined_selection_summary.json",
    )
    parser.add_argument("--max-select", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def truthy(value: str | bool | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def eligible(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not truthy(row.get("duplicate"))]


def matches_bucket(row: dict, bucket: str) -> bool:
    if bucket == "tiny_ball":
        return row.get("ball_size") == "tiny"
    if bucket == "small_ball":
        return row.get("ball_size") == "small"
    if bucket == "model_fn":
        return truthy(row.get("model_ball_miss"))
    if bucket == "model_fp":
        return truthy(row.get("model_ball_false_positive"))
    if bucket == "no_ball_hard_negative":
        return not truthy(row.get("has_ball")) and int(float(row.get("player_count", 0) or 0)) > 0
    if bucket == "ball_and_rim":
        return truthy(row.get("has_ball")) and truthy(row.get("has_rim"))
    if bucket == "other":
        return True
    raise ValueError(f"Unknown selection bucket: {bucket}")


def choose_balanced(rows: list[dict], max_select: int, seed: int) -> list[dict]:
    usable = eligible(rows)
    if max_select <= 0:
        raise ValueError("--max-select must be greater than zero.")
    max_select = min(max_select, len(usable))
    rng = random.Random(seed)
    selected: list[dict] = []
    selected_keys: set[str] = set()

    def take(bucket: str, quota: int) -> None:
        if len(selected) >= max_select or quota <= 0:
            return
        candidates = [
            row
            for row in usable
            if row.get("image", "") not in selected_keys and matches_bucket(row, bucket)
        ]
        rng.shuffle(candidates)
        # Within the general bucket, prefer richer basketball context but retain
        # deterministic randomness for visual diversity.
        if bucket == "other":
            candidates.sort(
                key=lambda row: (
                    truthy(row.get("has_ball")),
                    truthy(row.get("has_rim")),
                    min(int(float(row.get("player_count", 0) or 0)), 10),
                ),
                reverse=True,
            )
        count = min(quota, max_select - len(selected), len(candidates))
        for row in candidates[:count]:
            copy = dict(row)
            copy["selected"] = "true"
            copy["selection_bucket"] = bucket
            copy["selection_score"] = str(DEFAULT_QUOTAS.get(bucket, 0))
            selected.append(copy)
            selected_keys.add(row.get("image", ""))

    # Small/tiny examples are selected first so they do not disappear inside the
    # much larger model_fn population. The resulting buckets are mutually exclusive.
    for bucket in (
        "tiny_ball",
        "small_ball",
        "model_fn",
        "model_fp",
        "no_ball_hard_negative",
        "ball_and_rim",
        "other",
    ):
        take(bucket, DEFAULT_QUOTAS[bucket])

    if len(selected) < max_select:
        remaining = [row for row in usable if row.get("image", "") not in selected_keys]
        rng.shuffle(remaining)
        for row in remaining[: max_select - len(selected)]:
            copy = dict(row)
            copy["selected"] = "true"
            copy["selection_bucket"] = "other"
            copy["selection_score"] = "0"
            selected.append(copy)
            selected_keys.add(row.get("image", ""))

    return selected[:max_select]


def main() -> None:
    args = parse_args()
    analysis_path = resolve(args.analysis)
    output_path = resolve(args.output)
    summary_path = resolve(args.summary)
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis manifest not found: {analysis_path}. Run the analyzer first."
        )
    if (output_path.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError(
            "Refined selection output already exists. Use --force to replace it."
        )

    with analysis_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or not fieldnames:
        raise ValueError(f"Analysis manifest is empty: {analysis_path}")

    selected = choose_balanced(rows, args.max_select, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    buckets = Counter(row.get("selection_bucket", "other") for row in selected)
    ball_sizes = Counter(row.get("ball_size", "unknown") for row in selected)
    summary = {
        "analysis_manifest": str(analysis_path),
        "selected": len(selected),
        "max_select": args.max_select,
        "seed": args.seed,
        "selection_buckets": dict(buckets),
        "ball_size_distribution": dict(ball_sizes),
        "configured_quotas": DEFAULT_QUOTAS,
        "note": (
            "This selection reuses the existing analysis manifest and does not rerun "
            "YOLO inference. Review the pre-import diagnostic QA before importing."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Refined Basketball Game Detections selection complete.")
    print(f"Selected: {len(selected)}")
    print(f"Buckets: {dict(buckets)}")
    print(f"Ball sizes: {dict(ball_sizes)}")
    print(f"Manifest: {output_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
