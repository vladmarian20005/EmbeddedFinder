"""Tests for configuration module."""

from embedded_finder.config import (
    DEFAULT_DB_DIR,
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    IGNORE_DIRS,
    NATIVE_IMAGE_EXTENSIONS,
    CONVERTIBLE_IMAGE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    NATIVE_EMBED_EXTENSIONS,
    EXTENSION_MIME_MAP,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_VIDEO_DURATION_SECONDS,
    FILES_API_THRESHOLD_MB,
    get_api_key,
    get_db_dir,
)


def test_default_embedding_model():
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2-preview"


def test_supported_extensions_include_common_types():
    assert ".py" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".json" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS


def test_supported_extensions_include_multimodal():
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".mp3" in SUPPORTED_EXTENSIONS
    assert ".mp4" in SUPPORTED_EXTENSIONS


def test_supported_extensions_exclude_unsupported_video():
    assert ".avi" not in SUPPORTED_EXTENSIONS
    assert ".mkv" not in SUPPORTED_EXTENSIONS
    assert ".webm" not in SUPPORTED_EXTENSIONS


def test_native_image_extensions():
    assert NATIVE_IMAGE_EXTENSIONS == {".png", ".jpg", ".jpeg"}


def test_convertible_image_extensions():
    assert ".gif" in CONVERTIBLE_IMAGE_EXTENSIONS
    assert ".webp" in CONVERTIBLE_IMAGE_EXTENSIONS
    assert ".bmp" in CONVERTIBLE_IMAGE_EXTENSIONS


def test_image_extensions_is_union():
    assert IMAGE_EXTENSIONS == NATIVE_IMAGE_EXTENSIONS | CONVERTIBLE_IMAGE_EXTENSIONS


def test_audio_extensions():
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS


def test_video_extensions():
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".mov" in VIDEO_EXTENSIONS
    assert ".avi" not in VIDEO_EXTENSIONS
    assert ".mkv" not in VIDEO_EXTENSIONS
    assert ".webm" not in VIDEO_EXTENSIONS


def test_native_embed_extensions_excludes_convertible_images():
    # NATIVE_EMBED_EXTENSIONS should NOT include convertible image formats
    assert ".gif" not in NATIVE_EMBED_EXTENSIONS
    assert ".webp" not in NATIVE_EMBED_EXTENSIONS
    assert ".bmp" not in NATIVE_EMBED_EXTENSIONS


def test_native_embed_extensions_includes_native_media():
    assert NATIVE_IMAGE_EXTENSIONS.issubset(NATIVE_EMBED_EXTENSIONS)
    assert AUDIO_EXTENSIONS.issubset(NATIVE_EMBED_EXTENSIONS)
    assert VIDEO_EXTENSIONS.issubset(NATIVE_EMBED_EXTENSIONS)


def test_extension_mime_map():
    assert EXTENSION_MIME_MAP[".png"] == "image/png"
    assert EXTENSION_MIME_MAP[".jpg"] == "image/jpeg"
    assert EXTENSION_MIME_MAP[".mp4"] == "video/mp4"
    assert EXTENSION_MIME_MAP[".pdf"] == "application/pdf"


def test_extension_mime_map_no_unsupported_video():
    assert ".avi" not in EXTENSION_MIME_MAP
    assert ".mkv" not in EXTENSION_MIME_MAP
    assert ".webm" not in EXTENSION_MIME_MAP


def test_media_duration_limits():
    assert MAX_AUDIO_DURATION_SECONDS == 80
    assert MAX_VIDEO_DURATION_SECONDS == 120


def test_files_api_threshold():
    assert FILES_API_THRESHOLD_MB == 20


def test_ignore_dirs_include_common_patterns():
    assert ".git" in IGNORE_DIRS
    assert "node_modules" in IGNORE_DIRS
    assert "__pycache__" in IGNORE_DIRS


def test_get_api_key_returns_none_when_unset():
    assert get_api_key() is None


def test_get_api_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"


def test_get_db_dir_returns_default():
    assert str(get_db_dir()) == DEFAULT_DB_DIR


def test_get_db_dir_respects_env(monkeypatch):
    monkeypatch.setenv("EMBEDDEDFINDER_DB_DIR", "/custom/db/path")
    assert str(get_db_dir()) == "/custom/db/path"
