"""Result ranking and rich display for EmbeddedFinder."""

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from embedded_finder.search import SearchResult


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


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _truncate_path(path: str, max_len: int = 60) -> str:
    """Truncate a file path for display."""
    if len(path) <= max_len:
        return path
    parts = path.split("/")
    if len(parts) <= 2:
        return "..." + path[-(max_len - 3):]
    # Keep first and last parts, truncate middle
    return parts[0] + "/.../" + "/".join(parts[-2:])


def rank_results(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Re-rank results with combined scoring.

    Applies a boost based on:
    - Filename relevance (query words in filename)
    - File extension preference (code files for code queries)
    """
    query_words = set(query.lower().split())

    scored = []
    for r in results:
        boost = 0.0
        name_lower = r.file_name.lower()

        # Boost if query words appear in filename
        for word in query_words:
            if word in name_lower:
                boost += 0.02

        # Small boost for shorter file paths (less nested = more likely primary)
        depth = r.file_path.count("/")
        if depth < 5:
            boost += 0.005

        new_score = min(r.score + boost, 1.0)
        scored.append(SearchResult(
            file_path=r.file_path,
            file_name=r.file_name,
            score=round(new_score, 4),
            snippet=r.snippet,
            file_extension=r.file_extension,
            file_size=r.file_size,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
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
