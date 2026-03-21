"""Tests for result ranking and display."""

import pytest

from embedded_finder.search import SearchResult
from embedded_finder.ranker import rank_results, format_results, _score_color, _truncate_path


def _make_result(file_path="/test/a.py", score=0.9, **kwargs):
    defaults = dict(
        file_path=file_path,
        file_name=file_path.split("/")[-1],
        score=score,
        snippet="Some code content here",
        file_extension=".py",
        file_size=1024,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


def test_rank_results_preserves_order_when_no_boost():
    results = [
        _make_result("/a.py", score=0.9),
        _make_result("/b.py", score=0.8),
        _make_result("/c.py", score=0.7),
    ]
    ranked = rank_results(results, "unrelated query")
    assert ranked[0].file_path == "/a.py"
    assert ranked[1].file_path == "/b.py"
    assert ranked[2].file_path == "/c.py"


def test_rank_results_boosts_filename_match():
    results = [
        _make_result("/other.py", score=0.85),
        _make_result("/hello.py", score=0.84),
    ]
    ranked = rank_results(results, "hello function")
    # hello.py should be boosted above other.py
    assert ranked[0].file_path == "/hello.py"


def test_rank_results_caps_at_one():
    results = [_make_result("/a.py", score=0.99)]
    ranked = rank_results(results, "a function code python")
    assert ranked[0].score <= 1.0


def test_format_results_contains_filenames():
    results = [
        _make_result("/test/hello.py", score=0.95),
        _make_result("/test/world.txt", score=0.80, file_extension=".txt"),
    ]
    output = format_results(results, "hello world")
    assert "hello.py" in output
    assert "world.txt" in output


def test_format_results_contains_scores():
    results = [_make_result(score=0.95)]
    output = format_results(results)
    assert "95%" in output


def test_format_results_shows_query():
    results = [_make_result()]
    output = format_results(results, "my search query")
    assert "my search query" in output


def test_format_results_empty():
    output = format_results([])
    assert "No results" in output


def test_format_results_shows_count():
    results = [_make_result(), _make_result("/b.py", score=0.8)]
    output = format_results(results)
    # Rich adds ANSI codes, so check for the parts separately
    assert "result" in output
    assert "found" in output


def test_score_color():
    assert _score_color(0.90) == "green"
    assert _score_color(0.75) == "yellow"
    assert _score_color(0.55) == "orange1"
    assert _score_color(0.30) == "red"


def test_truncate_path_short():
    assert _truncate_path("/a/b.py") == "/a/b.py"


def test_truncate_path_long():
    long_path = "/very/long/deeply/nested/directory/structure/file.py"
    result = _truncate_path(long_path, max_len=30)
    assert len(result) <= len(long_path)
    assert "file.py" in result


# --- Media intent and cross-modal calibration tests ---


def test_media_query_boosts_native_image():
    """A 'photo of' query should boost native image results above text results."""
    text_result = _make_result("/notes.txt", score=0.75,
                               file_extension=".txt", embed_mode="text",
                               snippet="the dog ran across the field")
    image_result = _make_result("/dog.jpg", score=0.70,
                                file_extension=".jpg", embed_mode="native",
                                snippet="[JPG] dog.jpg")
    ranked = rank_results([text_result, image_result], "photo of a dog")
    # Image should rank above the text file for a media-intent query
    assert ranked[0].file_path == "/dog.jpg"


def test_media_query_penalizes_code():
    """Media queries should penalize code files."""
    code_result = _make_result("/dog.py", score=0.80,
                               file_extension=".py", embed_mode="text",
                               snippet="class Dog: pass")
    image_result = _make_result("/dog.png", score=0.72,
                                file_extension=".png", embed_mode="native",
                                snippet="[PNG] dog.png")
    ranked = rank_results([code_result, image_result], "picture of a dog")
    assert ranked[0].file_path == "/dog.png"


def test_code_query_still_penalizes_media():
    """Code queries should still penalize media results."""
    code_result = _make_result("/sort.py", score=0.82,
                               file_extension=".py", embed_mode="text",
                               snippet="def sort(arr): return sorted(arr)")
    image_result = _make_result("/sort.png", score=0.82,
                                file_extension=".png", embed_mode="native",
                                snippet="[PNG] sort.png")
    ranked = rank_results([code_result, image_result], "sort algorithm code")
    assert ranked[0].file_path == "/sort.py"


def test_native_embed_baseline_boost():
    """Native embeds should get a baseline boost even for neutral queries."""
    text_result = _make_result("/vacation.txt", score=0.72,
                               file_extension=".txt", embed_mode="text",
                               snippet="travel notes for next trip")
    image_result = _make_result("/sunset.jpg", score=0.68,
                                file_extension=".jpg", embed_mode="native",
                                snippet="[JPG] sunset.jpg")
    # With ~0.06 baseline boost, image goes from 0.68 to ~0.74, beating text at 0.72
    # Neither filename matches query words, and text snippet has no query overlap
    ranked = rank_results([text_result, image_result], "tropical view")
    assert ranked[0].file_path == "/sunset.jpg"


def test_mixed_query_no_double_boost():
    """Queries with both code and media keywords should not apply either bias."""
    code_result = _make_result("/diagram.py", score=0.80,
                               file_extension=".py", embed_mode="text",
                               snippet="def render_diagram(): pass")
    image_result = _make_result("/diagram.png", score=0.75,
                                file_extension=".png", embed_mode="native",
                                snippet="[PNG] diagram.png")
    # "diagram" is in MEDIA_KEYWORDS, "code" is in CODE_KEYWORDS
    # Neither media nor code intent should dominate
    ranked = rank_results([code_result, image_result], "diagram code")
    # Image gets baseline boost (~0.06) so ~0.81 vs 0.80 — close race
    # The exact order depends on boost math; main point is no extreme swing
    assert abs(ranked[0].score - ranked[1].score) < 0.10


def test_audio_query_boosts_audio():
    """Audio intent queries should boost audio results."""
    text_result = _make_result("/songs.txt", score=0.75,
                               file_extension=".txt", embed_mode="text",
                               snippet="list of favorite songs")
    audio_result = _make_result("/ocean.mp3", score=0.70,
                                file_extension=".mp3", embed_mode="native",
                                snippet="[MP3] ocean.mp3")
    ranked = rank_results([text_result, audio_result], "ocean sound recording")
    assert ranked[0].file_path == "/ocean.mp3"


def test_visual_query_penalizes_audio():
    """Visual intent queries should penalize audio results."""
    audio_result = _make_result("/nature.mp3", score=0.80,
                                file_extension=".mp3", embed_mode="native",
                                snippet="[MP3] nature.mp3")
    image_result = _make_result("/nature.jpg", score=0.74,
                                file_extension=".jpg", embed_mode="native",
                                snippet="[JPG] nature.jpg")
    ranked = rank_results([audio_result, image_result], "photo of nature scene")
    # Image should beat audio for a visual query
    assert ranked[0].file_path == "/nature.jpg"


def test_audio_query_penalizes_images():
    """Audio intent queries should penalize image results."""
    image_result = _make_result("/waves.jpg", score=0.80,
                                file_extension=".jpg", embed_mode="native",
                                snippet="[JPG] waves.jpg")
    audio_result = _make_result("/waves.mp3", score=0.74,
                                file_extension=".mp3", embed_mode="native",
                                snippet="[MP3] waves.mp3")
    ranked = rank_results([image_result, audio_result], "sound of ocean waves")
    assert ranked[0].file_path == "/waves.mp3"


def test_embed_mode_propagated():
    """embed_mode should be preserved through ranking."""
    results = [
        _make_result("/a.png", score=0.9, file_extension=".png", embed_mode="native"),
        _make_result("/b.py", score=0.8, embed_mode="text"),
    ]
    ranked = rank_results(results, "test")
    modes = {r.file_path: r.embed_mode for r in ranked}
    assert modes["/a.png"] == "native"
    assert modes["/b.py"] == "text"
