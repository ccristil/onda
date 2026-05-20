from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODELS_PATH = Path(__file__).parent / "models.json"
ONDA_DIR = Path.home() / ".onda"


@dataclass
class ModelEntry:
    name: str
    backend: str
    onnx_url: str
    config_url: str
    size_mb: int
    description: str


def load_registry() -> list[ModelEntry]:
    with open(MODELS_PATH) as f:
        data = json.load(f)
    return [ModelEntry(**m) for m in data]


def get_model(name: str) -> ModelEntry | None:
    for m in load_registry():
        if m.name == name:
            return m
    return None


def model_dir(name: str) -> Path:
    return ONDA_DIR / "models" / name


def is_downloaded(name: str) -> bool:
    d = model_dir(name)
    return (d / f"{name}.onnx").exists() and (d / f"{name}.onnx.json").exists()
