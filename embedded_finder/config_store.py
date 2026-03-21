"""Persistent config file manager for EmbeddedFinder (JSON, XDG paths)."""

import json
import os
from pathlib import Path

_APP_NAME = "embeddedfinder"


def get_config_path() -> Path:
    """Return the path to the config file, respecting $XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(base) / _APP_NAME / "config.json"


def load_config() -> dict:
    """Load the config file. Returns {} if missing or corrupt."""
    path = get_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(data: dict) -> None:
    """Write the full config dict to disk with owner-only permissions."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    path.chmod(0o600)


def get_config_value(key: str):
    """Get a single value from config. Returns None if missing."""
    return load_config().get(key)


def set_config_value(key: str, value) -> None:
    """Set a single value in config (read-modify-write)."""
    data = load_config()
    data[key] = value
    save_config(data)


def delete_config_value(key: str) -> bool:
    """Remove a key from config. Returns True if key existed."""
    data = load_config()
    if key not in data:
        return False
    del data[key]
    save_config(data)
    return True
