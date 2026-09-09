from __future__ import annotations

from pathlib import Path
import tempfile
import tomllib

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def load_model_config() -> dict:
    return load_toml(PROJECT_ROOT / "configs" / "model.toml")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_weights(override: str | None = None) -> Path:
    if override:
        return resolve_project_path(override)
    model_config = load_model_config()
    return resolve_project_path(model_config["weights"])


def select_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def materialize_dataset_yaml(dataset_yaml: str | Path) -> Path:
    """Create a temporary dataset YAML with an absolute dataset root.

    Ultralytics may resolve a relative `path:` against its global datasets
    directory. Converting it to an absolute path here keeps this repository
    portable across machines.
    """
    source = resolve_project_path(dataset_yaml)
    if not source.exists():
        raise FileNotFoundError(f"Dataset config not found: {source}")

    with source.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    dataset_root = Path(config.get("path", "dataset")).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    config["path"] = str(dataset_root.resolve())

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="trainingmodel_dataset_",
        delete=False,
        encoding="utf-8",
    )
    with temp:
        yaml.safe_dump(config, temp, sort_keys=False, allow_unicode=True)

    return Path(temp.name)
