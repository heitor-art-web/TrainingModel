from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize review statuses overall and by review reason."
    )
    parser.add_argument("batch", help="Batch name or path under runs/review/.")
    return parser.parse_args()


def resolve_batch(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        direct = PROJECT_ROOT / path
        path = direct if direct.exists() else PROJECT_ROOT / "runs" / "review" / value
    path = path.resolve()
    manifest = path / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Review manifest not found: {manifest}")
    return path


def main() -> None:
    args = parse_args()
    batch_dir = resolve_batch(args.batch)
    manifest_path = batch_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Review batch contains no rows.")

    overall = Counter((row.get("review_status") or "unreviewed") for row in rows)
    by_reason: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        reason = row.get("reason") or "n/a"
        status = row.get("review_status") or "unreviewed"
        by_reason[reason][status] += 1

    print(f"Batch: {batch_dir}")
    print(f"Total: {len(rows)}")
    print("Overall:")
    for status in ("already_correct", "corrected", "skip", "uncertain", "unreviewed"):
        print(f"  {status}: {overall.get(status, 0)}")

    print("By reason:")
    for reason in sorted(by_reason):
        counts = by_reason[reason]
        total = sum(counts.values())
        approved = counts.get("already_correct", 0) + counts.get("corrected", 0)
        correction_rate = (
            counts.get("corrected", 0) / approved if approved else 0.0
        )
        print(f"  {reason}: total={total}")
        print(
            "    "
            + ", ".join(
                f"{status}={counts.get(status, 0)}"
                for status in (
                    "already_correct",
                    "corrected",
                    "skip",
                    "uncertain",
                    "unreviewed",
                )
            )
        )
        print(f"    correction_rate_among_approved={correction_rate:.3f}")


if __name__ == "__main__":
    main()
