from __future__ import annotations

import json
from datetime import date
from typing import Optional

from onda.registry import ONDA_DIR

CONFIG_PATH = ONDA_DIR / "config.json"


def _read_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_config(data: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_default_model() -> Optional[str]:
    return _read_config().get("default_model") or None


def set_default_model(name: str) -> None:
    data = _read_config()
    data["default_model"] = name
    _write_config(data)


def clear_default_model() -> None:
    data = _read_config()
    data.pop("default_model", None)
    _write_config(data)


def should_show_logo() -> bool:
    return _read_config().get("last_logo_date") != str(date.today())


def mark_logo_shown() -> None:
    data = _read_config()
    data["last_logo_date"] = str(date.today())
    _write_config(data)
