from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIORITY_CLASSES = ("basketball", "player", "rim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate evaluation summaries using promotion rules."
    )
    parser.add_argument("baseline", help="Baseline metrics_summary.json")
    parser.add_argument("candidate", help="Candidate metrics_summary.json")
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.0,
        help="Allowed absolute recall drop for player/rim. Default: 0.",
    )
    return parser.parse_args()


def promotion_check(
    baseline: dict, candidate: dict, recall_tolerance: float = 0.0
) -> dict:
    base = baseline["per_class"]
    cand = candidate["per_class"]
    missing = [name for name in PRIORITY_CLASSES if name not in base or name not in cand]
    if missing:
        raise ValueError(f"Missing priority classes in metrics: {', '.join(missing)}")

    checks = {
        "basketball_precision_improves": (
            cand["basketball"]["precision"] > base["basketball"]["precision"]
        ),
        "player_recall_not_worse": (
            cand["player"]["recall"]
            >= base["player"]["recall"] - recall_tolerance
        ),
        "rim_recall_not_worse": (
            cand["rim"]["recall"]
            >= base["rim"]["recall"] - recall_tolerance
        ),
    }
    return {
        "eligible_for_promotion_on_detection_metrics": all(checks.values()),
        "checks": checks,
        "baseline": {name: base[name] for name in PRIORITY_CLASSES},
        "candidate": {name: cand[name] for name in PRIORITY_CLASSES},
        "note": (
            "Promotion still requires tracking/product-video review; this comparison "
            "covers the agreed detection priorities only."
        ),
    }


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = promotion_check(baseline, candidate, args.recall_tolerance)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
