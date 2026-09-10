from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import re
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_SPLITS = ("train", "val", "test")
RESERVED_TEST_ROLES = {"test", "gold_test"}


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    role: str = "pool"
    allow_training: bool = True
    allow_hard_case_training: bool = True
    reference: str | None = None


def load_data_registry(path: str | Path = "configs/data_sources.toml") -> dict:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    with candidate.open("rb") as file:
        registry = tomllib.load(file)
    validate_split_config(registry.get("split", {}))
    return registry


def validate_split_config(config: dict) -> None:
    ratios = [float(config.get(name, 0.0)) for name in VALID_SPLITS]
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("train/val/test ratios must each be between 0 and 1.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0.")


def policies_by_id(registry: dict) -> dict[str, SourcePolicy]:
    result: dict[str, SourcePolicy] = {}
    for raw in registry.get("sources", []):
        source_id = str(raw["id"])
        role = str(raw.get("role", "pool")).lower()
        default_allowed = role not in RESERVED_TEST_ROLES
        result[source_id] = SourcePolicy(
            source_id=source_id,
            role=role,
            allow_training=bool(raw.get("allow_training", default_allowed)),
            allow_hard_case_training=bool(
                raw.get("allow_hard_case_training", default_allowed)
            ),
            reference=str(raw["reference"]) if raw.get("reference") else None,
        )
    return result


def get_source_policy(source_id: str, registry: dict) -> SourcePolicy:
    return policies_by_id(registry).get(source_id, SourcePolicy(source_id=source_id))


def assert_source_allowed(source_id: str, purpose: str, registry: dict) -> None:
    policy = get_source_policy(source_id, registry)
    if purpose == "training" and not policy.allow_training:
        raise ValueError(
            f"Source {source_id!r} is protected ({policy.role}) and cannot be used for training."
        )
    if purpose == "hard_case_training" and not policy.allow_hard_case_training:
        raise ValueError(
            f"Source {source_id!r} is protected ({policy.role}) and cannot be mined for training."
        )


def role_to_split(role: str) -> str | None:
    role = role.lower()
    if role in VALID_SPLITS:
        return role
    if role == "gold_test":
        return "test"
    return None


def sanitize_source_id(source_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("._")
    return cleaned or "source"


def assign_sources(source_counts: dict[str, int], registry: dict) -> dict[str, str]:
    """Assign complete sources to splits while approximating configured frame ratios."""
    if not source_counts:
        raise ValueError("No source counts were provided.")

    split_cfg = registry["split"]
    ratios = {name: float(split_cfg[name]) for name in VALID_SPLITS}
    seed = int(split_cfg.get("seed", 42))
    policies = policies_by_id(registry)
    total = sum(max(0, int(count)) for count in source_counts.values())
    if total <= 0:
        raise ValueError("Sources contain no usable samples.")

    targets = {name: total * ratios[name] for name in VALID_SPLITS}
    assigned_counts = {name: 0 for name in VALID_SPLITS}
    assignments: dict[str, str] = {}
    pool: list[str] = []

    for source_id in sorted(source_counts):
        policy = policies.get(source_id, SourcePolicy(source_id=source_id))
        explicit = role_to_split(policy.role)
        if explicit is None:
            pool.append(source_id)
            continue
        if explicit != "test" and not policy.allow_training:
            raise ValueError(
                f"Protected source {source_id!r} cannot be assigned to {explicit}."
            )
        assignments[source_id] = explicit
        assigned_counts[explicit] += source_counts[source_id]

    rng = random.Random(seed)
    rng.shuffle(pool)
    random_order = {source_id: idx for idx, source_id in enumerate(pool)}
    pool.sort(key=lambda source_id: (-source_counts[source_id], random_order[source_id]))

    for source_id in pool:
        count = source_counts[source_id]
        deficits = {
            split: targets[split] - assigned_counts[split] for split in VALID_SPLITS
        }
        positive = [split for split in VALID_SPLITS if deficits[split] > 0]
        if positive:
            chosen = max(
                positive,
                key=lambda split: (
                    deficits[split] / max(targets[split], 1.0),
                    deficits[split],
                ),
            )
        else:
            chosen = min(
                VALID_SPLITS,
                key=lambda split: assigned_counts[split] / max(targets[split], 1.0),
            )
        assignments[source_id] = chosen
        assigned_counts[chosen] += count

    if len(source_counts) >= 3:
        missing = [split for split in VALID_SPLITS if split not in assignments.values()]
        if missing:
            raise ValueError(
                "Could not create all train/val/test splits at source level. "
                f"Missing: {', '.join(missing)}. Add more independent sources or set explicit roles."
            )
    return assignments


def split_summary(assignments: dict[str, str], source_counts: dict[str, int]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    total = sum(source_counts.values())
    for split in VALID_SPLITS:
        sources = sorted(
            source_id for source_id, value in assignments.items() if value == split
        )
        frames = sum(source_counts[source_id] for source_id in sources)
        result[split] = {
            "sources": sources,
            "frames": frames,
            "ratio": (frames / total) if total else 0.0,
        }
    return result
