"""Tests for persistent config file manager."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from embedded_finder.config_store import (
    load_config,
    save_config,
    get_config_value,
    set_config_value,
    delete_config_value,
    get_config_path,
)


@pytest.fixture
def config_dir(tmp_dir, monkeypatch):
    """Point config to a temp directory."""
    config_path = tmp_dir / "embeddedfinder" / "config.json"
    monkeypatch.setattr(
        "embedded_finder.config_store.get_config_path", lambda: config_path
    )
    return config_path


def test_load_config_returns_empty_when_no_file(config_dir):
    assert load_config() == {}


def test_save_and_load_config(config_dir):
    save_config({"api_key": "test-123", "setting": "value"})
    data = load_config()
    assert data["api_key"] == "test-123"
    assert data["setting"] == "value"


def test_save_config_creates_parent_dirs(config_dir):
    assert not config_dir.parent.exists()
    save_config({"key": "val"})
    assert config_dir.parent.exists()
    assert config_dir.exists()


def test_save_config_sets_owner_only_permissions(config_dir):
    save_config({"api_key": "secret"})
    mode = config_dir.stat().st_mode & 0o777
    assert mode == 0o600


def test_save_config_overwrites_existing(config_dir):
    save_config({"v": 1})
    save_config({"v": 2})
    assert load_config()["v"] == 2


def test_load_config_handles_corrupt_json(config_dir):
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("not valid json {{{")
    assert load_config() == {}


def test_load_config_handles_empty_file(config_dir):
    config_dir.parent.mkdir(parents=True, exist_ok=True)
    config_dir.write_text("")
    assert load_config() == {}


def test_get_config_value_returns_value(config_dir):
    save_config({"api_key": "abc123"})
    assert get_config_value("api_key") == "abc123"


def test_get_config_value_returns_none_when_missing(config_dir):
    save_config({"other": "val"})
    assert get_config_value("api_key") is None


def test_get_config_value_returns_none_when_no_file(config_dir):
    assert get_config_value("api_key") is None


def test_set_config_value_creates_new(config_dir):
    set_config_value("api_key", "new-key")
    assert get_config_value("api_key") == "new-key"


def test_set_config_value_preserves_existing(config_dir):
    save_config({"existing": "keep"})
    set_config_value("api_key", "new")
    data = load_config()
    assert data["existing"] == "keep"
    assert data["api_key"] == "new"


def test_set_config_value_overwrites(config_dir):
    set_config_value("api_key", "old")
    set_config_value("api_key", "new")
    assert get_config_value("api_key") == "new"


def test_delete_config_value_removes_key(config_dir):
    save_config({"api_key": "del-me", "keep": "this"})
    result = delete_config_value("api_key")
    assert result is True
    assert get_config_value("api_key") is None
    assert get_config_value("keep") == "this"


def test_delete_config_value_returns_false_when_missing(config_dir):
    save_config({"other": "val"})
    result = delete_config_value("nonexistent")
    assert result is False


def test_delete_config_value_returns_false_when_no_file(config_dir):
    result = delete_config_value("api_key")
    assert result is False


def test_config_file_is_valid_json(config_dir):
    save_config({"key": "value", "nested": {"a": 1}})
    raw = config_dir.read_text()
    data = json.loads(raw)
    assert data["key"] == "value"
    assert data["nested"]["a"] == 1


def test_config_file_ends_with_newline(config_dir):
    save_config({"key": "value"})
    raw = config_dir.read_text()
    assert raw.endswith("\n")
