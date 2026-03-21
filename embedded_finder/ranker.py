"""Result ranking and rich display for EmbeddedFinder."""

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from embedded_finder.config import IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from embedded_finder.search import SearchResult
from embedded_finder.utils import format_size_long


# File type icons (using text symbols for compatibility)
FILE_ICONS = {
    # Code
    ".py": "PY", ".js": "JS", ".ts": "TS", ".java": "JV",
    ".go": "GO", ".rs": "RS", ".rb": "RB", ".c": "C ",
    ".cpp": "C+", ".h": "H ", ".sh": "SH", ".sql": "SQ",
    # Markup / data
    ".html": "HT", ".css": "CS", ".json": "JS",
    ".yaml": "YM", ".yml": "YM", ".xml": "XM",
    ".md": "MD", ".txt": "TX", ".csv": "CV", ".toml": "TM",
    # Documents
    ".pdf": "PD", ".docx": "DC",
    # Images
    ".png": "IM", ".jpg": "IM", ".jpeg": "IM",
    ".gif": "IM", ".webp": "IM", ".bmp": "IM",
    # Audio
    ".mp3": "AU", ".wav": "AU", ".ogg": "AU",
    ".flac": "AU", ".m4a": "AU",
    # Video
    ".mp4": "VD", ".mov": "VD", ".avi": "VD",
    ".mkv": "VD", ".webm": "VD",
}


def _score_color(score: float) -> str:
    """Get color based on similarity score."""
    if score >= 0.85:
        return "green"
    elif score >= 0.70:
        return "yellow"
    elif score >= 0.50:
        return "orange1"
    else:
        return "red"


_format_size = format_size_long


def _truncate_path(path: str, max_len: int = 60) -> str:
    """Truncate a file path for display."""
    if len(path) <= max_len:
        return path
    parts = path.split("/")
    if len(parts) <= 2:
        return "..." + path[-(max_len - 3):]
    # Keep first and last parts, truncate middle
    return parts[0] + "/.../" + "/".join(parts[-2:])


CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".zsh", ".lua", ".pl", ".ex", ".exs", ".r", ".m", ".sql",
}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml",
                   ".toml", ".ini", ".cfg", ".conf", ".xml", ".html", ".css",
                   ".scss", ".less", ".svg"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Keywords that suggest the user wants code/text, not media
CODE_KEYWORDS = {
    "function", "class", "def", "import", "return", "variable", "code",
    "script", "module", "error", "bug", "api", "endpoint", "database",
    "query", "algorithm", "loop", "array", "list", "dict", "string",
    "python", "javascript", "typescript", "java", "rust", "go", "ruby",
    "fibonacci", "sort", "search", "parse", "compile", "deploy", "test",
    "config", "yaml", "json", "html", "css", "sql", "server", "client",
    "http", "request", "response", "auth", "token", "middleware",
}

# Keywords that suggest the user wants visual content (images/photos)
VISUAL_KEYWORDS = {
    "photo", "picture", "image", "screenshot", "diagram", "chart",
    "graph", "illustration", "drawing", "painting", "logo", "icon",
    "thumbnail", "banner", "poster", "selfie", "portrait", "landscape",
    "snapshot", "render", "mockup", "wireframe", "sketch",
    # Descriptive visual cues
    "showing", "depicts", "looks", "wearing", "standing",
    "sitting", "walking", "holding", "background",
}

# Keywords that suggest the user wants audio content
AUDIO_KEYWORDS = {
    "song", "music", "audio", "sound", "recording", "podcast",
    "voice", "speech", "melody", "beat", "track", "listen",
    "hearing", "noise", "tone", "spoken",
}

# Keywords that suggest the user wants video content
VIDEO_KEYWORDS = {
    "video", "movie", "footage", "animation", "screencast",
    "tutorial", "demo", "presentation", "clip",
}

# Union of all media keywords (for general media-intent detection)
MEDIA_KEYWORDS = VISUAL_KEYWORDS | AUDIO_KEYWORDS | VIDEO_KEYWORDS

# Cross-modal calibration: native embeddings (images/audio/video) consistently
# score ~0.05-0.08 lower than text embeddings for equivalent semantic matches.
# This baseline boost compensates for the systematic gap.
NATIVE_EMBED_BASELINE_BOOST = 0.06

# Additional boost for the specific media type the user is looking for
MEDIA_INTENT_BOOST = 0.05
# Penalty for wrong media type when specific intent is detected
WRONG_MEDIA_PENALTY = 0.04


def _query_wants_code(query_words: set[str]) -> bool:
    """Detect if a query is about code/text content rather than media."""
    return bool(query_words & CODE_KEYWORDS)


def _query_wants_media(query_words: set[str]) -> bool:
    """Detect if a query is about visual/audio/video content."""
    return bool(query_words & MEDIA_KEYWORDS)


def _query_wants_visual(query_words: set[str]) -> bool:
    """Detect if a query is specifically about visual/image content."""
    return bool(query_words & VISUAL_KEYWORDS)


def _query_wants_audio(query_words: set[str]) -> bool:
    """Detect if a query is specifically about audio content."""
    return bool(query_words & AUDIO_KEYWORDS)


def _query_wants_video(query_words: set[str]) -> bool:
    """Detect if a query is specifically about video content."""
    return bool(query_words & VIDEO_KEYWORDS)


def rank_results(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Re-rank results with combined scoring.

    Applies boosts based on:
    - Cross-modal calibration (native embeds get baseline boost to close the
      systematic text→image score gap)
    - Media-intent detection (queries about photos/sounds boost media, penalize text)
    - Code-intent detection (queries about code boost code, penalize media)
    - Filename relevance (query words in filename)
    - File path depth (shallower = more likely primary)
    """
    query_words = set(query.lower().split())
    wants_code = _query_wants_code(query_words)
    wants_media = _query_wants_media(query_words)
    wants_visual = _query_wants_visual(query_words)
    wants_audio = _query_wants_audio(query_words)
    wants_video = _query_wants_video(query_words)

    scored = []
    for r in results:
        boost = 0.0
        name_lower = r.file_name.lower()
        ext = r.file_extension.lower()
        is_native = r.embed_mode in ("native", "native_chunked")

        # Cross-modal calibration: native embeddings score systematically lower
        # than text embeddings for equivalent semantic matches. Apply a baseline
        # boost to level the playing field before intent-based adjustments.
        if is_native:
            boost += NATIVE_EMBED_BASELINE_BOOST

        is_image = ext in IMAGE_EXTENSIONS
        is_audio = ext in AUDIO_EXTENSIONS
        is_video = ext in VIDEO_EXTENSIONS

        # Sub-category media intent: boost the specific media type the user wants
        # and penalize other media types that would otherwise crowd results
        if wants_visual and not wants_code:
            if is_image:
                boost += MEDIA_INTENT_BOOST
            elif is_audio or is_video:
                boost -= WRONG_MEDIA_PENALTY
            elif ext in CODE_EXTENSIONS:
                boost -= 0.04
            elif ext in TEXT_EXTENSIONS:
                boost -= 0.03
        elif wants_audio and not wants_code:
            if is_audio:
                boost += MEDIA_INTENT_BOOST
            elif is_image or is_video:
                boost -= WRONG_MEDIA_PENALTY
            elif ext in CODE_EXTENSIONS:
                boost -= 0.04
            elif ext in TEXT_EXTENSIONS:
                boost -= 0.03
        elif wants_video and not wants_code:
            if is_video:
                boost += MEDIA_INTENT_BOOST
            elif is_image or is_audio:
                boost -= WRONG_MEDIA_PENALTY
            elif ext in CODE_EXTENSIONS:
                boost -= 0.04
            elif ext in TEXT_EXTENSIONS:
                boost -= 0.03
        elif wants_media and not wants_code:
            # Generic media intent (no specific sub-type detected)
            if ext in MEDIA_EXTENSIONS:
                boost += 0.04
            elif ext in CODE_EXTENSIONS:
                boost -= 0.04
            elif ext in TEXT_EXTENSIONS:
                boost -= 0.03

        # Code-intent: boost code results, penalize media
        if wants_code and not wants_media:
            if ext in CODE_EXTENSIONS:
                boost += 0.03
            elif ext in TEXT_EXTENSIONS:
                boost += 0.01
            elif ext in MEDIA_EXTENSIONS:
                boost -= 0.05

        # Boost if query words appear in filename
        for word in query_words:
            if word in name_lower:
                boost += 0.03

        # Boost results that have actual text content matching query words
        # (only meaningful for text-embedded results with real snippets)
        if r.snippet and not is_native:
            snippet_lower = r.snippet.lower()
            matching_words = sum(1 for w in query_words if w in snippet_lower)
            boost += matching_words * 0.02

        # Small boost for shorter file paths (less nested = more likely primary)
        depth = r.file_path.count("/")
        if depth < 5:
            boost += 0.005

        new_score = max(0.0, min(r.score + boost, 1.0))
        scored.append(SearchResult(
            file_path=r.file_path,
            file_name=r.file_name,
            score=round(new_score, 4),
            snippet=r.snippet,
            file_extension=r.file_extension,
            file_size=r.file_size,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
            embed_mode=r.embed_mode,
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def format_results(results: list[SearchResult], query: str = "") -> str:
    """Format search results with rich terminal output.

    Returns the formatted string (for testability).
    """
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return buf.getvalue()

    # Header
    if query:
        console.print(f"\n[bold]Results for:[/bold] [cyan]\"{query}\"[/cyan]\n")

    table = Table(show_header=True, header_style="bold", expand=True, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=7)
    table.add_column("Type", width=4)
    table.add_column("File", ratio=3)
    table.add_column("Size", width=8)

    for i, r in enumerate(results, 1):
        color = _score_color(r.score)
        score_text = Text(f"{r.score * 100:.0f}%", style=color)
        icon = FILE_ICONS.get(r.file_extension, "  ")
        path_display = _truncate_path(r.file_path)

        table.add_row(
            str(i),
            score_text,
            f"[dim]{icon}[/dim]",
            path_display,
            _format_size(r.file_size),
        )

    console.print(table)

    # Show snippets for top 3
    for i, r in enumerate(results[:3], 1):
        if r.snippet:
            snippet = r.snippet.replace("\n", " ")[:200]
            console.print(
                Panel(
                    f"[dim]{snippet}[/dim]",
                    title=f"[bold]{r.file_name}[/bold]",
                    border_style="dim",
                    width=100,
                )
            )

    console.print(f"\n[dim]{len(results)} result(s) found.[/dim]")
    return buf.getvalue()


def print_results(results: list[SearchResult], query: str = "") -> None:
    """Print formatted results to the terminal."""
    output = format_results(results, query)
    console = Console()
    console.print(output, highlight=False)
