"""Interactive terminal UI for EmbeddedFinder — Claude Code-style REPL."""

import os
import sys
import time
import shutil
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.markup import escape
from rich import box

from embedded_finder import __version__
from embedded_finder.config import (
    get_api_key, get_db_dir, SUPPORTED_EXTENSIONS,
    IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS,
)

console = Console()

# ── Theme constants ──────────────────────────────────────────────────────────
ACCENT = "cyan"
DIM = "dim"
SCORE_COLORS = {85: "green", 70: "yellow", 50: "bright_yellow", 0: "red"}
TYPE_BADGES = {
    ".py": ("PY", "green"), ".js": ("JS", "yellow"), ".ts": ("TS", "blue"),
    ".java": ("JV", "red"), ".go": ("GO", "cyan"), ".rs": ("RS", "bright_red"),
    ".rb": ("RB", "red"), ".c": ("C", "blue"), ".cpp": ("C+", "blue"),
    ".html": ("HT", "bright_magenta"), ".css": ("CS", "magenta"),
    ".json": ("JS", "yellow"), ".yaml": ("YM", "green"), ".yml": ("YM", "green"),
    ".md": ("MD", "white"), ".txt": ("TX", "white"), ".toml": ("TM", "green"),
    ".sql": ("SQ", "blue"), ".sh": ("SH", "green"), ".xml": ("XM", "magenta"),
    ".pdf": ("PD", "red"), ".docx": ("DC", "blue"), ".csv": ("CV", "green"),
    ".png": ("IMG", "bright_magenta"), ".jpg": ("IMG", "bright_magenta"),
    ".jpeg": ("IMG", "bright_magenta"), ".gif": ("IMG", "bright_magenta"),
    ".webp": ("IMG", "bright_magenta"), ".bmp": ("IMG", "bright_magenta"),
    ".mp3": ("AUD", "bright_cyan"), ".wav": ("AUD", "bright_cyan"),
    ".ogg": ("AUD", "bright_cyan"), ".flac": ("AUD", "bright_cyan"),
    ".mp4": ("VID", "bright_yellow"), ".mov": ("VID", "bright_yellow"),
    ".avi": ("VID", "bright_yellow"), ".mkv": ("VID", "bright_yellow"),
}


def _score_style(score: float) -> str:
    pct = score * 100
    for threshold, color in SCORE_COLORS.items():
        if pct >= threshold:
            return color
    return "red"


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}K"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}M"


def _type_badge(ext: str) -> Text:
    label, color = TYPE_BADGES.get(ext, ("??", "dim"))
    return Text(f" {label} ", style=f"bold {color} on grey23")


def _truncate_path(path: str, max_len: int = 65) -> str:
    if len(path) <= max_len:
        return path
    parts = path.split("/")
    if len(parts) <= 3:
        return "…" + path[-(max_len - 1):]
    return parts[0] + "/…/" + "/".join(parts[-2:])


# ── Banner ───────────────────────────────────────────────────────────────────

def print_banner(doc_count: int = 0, file_count: int = 0):
    """Print the welcome banner."""
    width = min(shutil.get_terminal_size().columns, 80)

    title = Text()
    title.append("◆ ", style=f"bold {ACCENT}")
    title.append("EmbeddedFinder", style="bold white")
    title.append(f"  v{__version__}", style="dim")

    subtitle = Text()
    subtitle.append("  Semantic file search powered by ", style="dim")
    subtitle.append("Gemini Embedding 2", style=f"bold {ACCENT}")

    stats = Text()
    if doc_count > 0:
        stats.append(f"  ● {file_count} files", style="green")
        stats.append(f"  ({doc_count} chunks)", style="dim")
    else:
        stats.append("  ○ No files indexed yet", style="dim yellow")
    db_path = str(get_db_dir())
    stats.append(f"  │  {db_path}", style="dim")

    panel = Panel(
        Text.from_ansi(f"{title}\n{subtitle}\n{stats}"),
        border_style="dim cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def print_help():
    """Print help information."""
    console.print()
    table = Table(
        show_header=False, box=None, padding=(0, 2),
        title="[bold]Commands[/bold]", title_style="white",
    )
    table.add_column(style=f"bold {ACCENT}", min_width=22)
    table.add_column(style="dim")

    table.add_row("[query text]", "Search files by natural language description")
    table.add_row("/index <path>", "Index a directory for searching")
    table.add_row("/reindex <path>", "Re-index only changed files")
    table.add_row("/status", "Show index statistics")
    table.add_row("/clear", "Clear the entire index")
    table.add_row("/watch <path>", "Watch directory for changes")
    table.add_row("/web [port]", "Start web UI (default: 8080)")
    table.add_row("/help", "Show this help")
    table.add_row("/quit  or  Ctrl+C", "Exit")
    console.print(table)
    console.print()


# ── Result display ───────────────────────────────────────────────────────────

def print_results(results, query: str, elapsed: float):
    """Print search results in a compact, information-dense format."""
    if not results:
        console.print(f"\n  [dim]No results found for[/dim] [italic]\"{escape(query)}\"[/italic]\n")
        return

    console.print()
    header = Text()
    header.append(f"  {len(results)} result{'s' if len(results) != 1 else ''}", style="bold")
    header.append(f"  ({elapsed:.1f}s)", style="dim")
    header.append(f"  │  ", style="dim")
    header.append(f"\"{escape(query)}\"", style=f"italic {ACCENT}")
    console.print(header)
    console.print()

    for i, r in enumerate(results):
        _print_result_row(i + 1, r, show_snippet=(i < 5))

    console.print()


def _print_result_row(num: int, r, show_snippet: bool = True):
    """Print a single result row."""
    score_pct = int(r.score * 100)
    color = _score_style(r.score)
    badge = _type_badge(r.file_extension)

    # Line 1: number, score, type badge, filename, size
    line = Text("  ")
    line.append(f"{num:>2}", style="bold dim")
    line.append("  ")
    line.append(f"{score_pct}%", style=f"bold {color}")
    line.append("  ")
    line.append_text(badge)
    line.append("  ")
    line.append(r.file_name, style="bold white")
    line.append(f"  {_format_size(r.file_size)}", style="dim")
    console.print(line)

    # Line 2: path
    path_display = _truncate_path(r.file_path)
    console.print(f"        [dim]{path_display}[/dim]")

    # Line 3: snippet (if available and requested)
    if show_snippet and r.snippet:
        snippet = r.snippet.replace("\n", " ").strip()[:150]
        console.print(f"        [dim italic]▸ {escape(snippet)}[/dim italic]")

    console.print()


# ── Spinner helpers ──────────────────────────────────────────────────────────

def _search_with_spinner(engine, query: str, n_results: int, min_score: float):
    """Run search with animated spinner."""
    from rich.spinner import Spinner

    result_holder = [None, None]  # [results, error]

    import threading

    def do_search():
        try:
            result_holder[0] = engine.search(query, n_results=n_results, min_score=min_score)
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=do_search, daemon=True)
    start = time.time()
    t.start()

    with console.status(f"[{ACCENT}]Searching…[/{ACCENT}]", spinner="dots"):
        t.join()

    elapsed = time.time() - start

    if result_holder[1]:
        raise result_holder[1]

    return result_holder[0], elapsed


def _index_with_progress(indexer, path: str, extensions=None):
    """Run indexing with Rich progress bar."""
    from embedded_finder.crawler import crawl

    # First count files
    console.print(f"\n  [dim]Scanning {escape(path)}…[/dim]")
    files = crawl(path, extensions=extensions)
    total = len(files)
    console.print(f"  [dim]Found {total} files to process[/dim]\n")

    if total == 0:
        return None

    stats_holder = [None]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing", total=total, status="")

        def on_progress(file_info, stats):
            progress.update(task, advance=1, status=file_info.name[:40])

        stats = indexer.index_directory(path, extensions=extensions, on_progress=on_progress)
        stats_holder[0] = stats

    return stats_holder[0]


# ── Component factory ────────────────────────────────────────────────────────

def _create_components(api_key: str | None = None):
    """Create the core components (embedder, store, indexer, search engine)."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.indexer import Indexer
    from embedded_finder.search import SearchEngine
    from embedded_finder.ranker import rank_results

    key = api_key or get_api_key()
    if not key:
        return None, None, None, None
    embedder = Embedder(api_key=key)
    store = VectorStore()
    indexer = Indexer(embedder=embedder, store=store)
    engine = SearchEngine(embedder=embedder, store=store)
    return embedder, store, indexer, engine


# ── Command handlers ─────────────────────────────────────────────────────────

def _handle_search(engine, query: str, n_results: int = 10, min_score: float = 0.0):
    """Handle a search query."""
    from embedded_finder.ranker import rank_results

    results, elapsed = _search_with_spinner(engine, query, n_results, min_score)
    results = rank_results(results, query)
    print_results(results, query, elapsed)


def _handle_index(indexer, path: str):
    """Handle /index command."""
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"  [red]✗[/red] Not a directory: {path}")
        return

    stats = _index_with_progress(indexer, str(resolved))
    if stats is None:
        console.print("  [dim]No supported files found.[/dim]")
        return

    console.print(f"  [green]✓[/green] [bold]{stats.indexed}[/bold] indexed", end="")
    console.print(f"  [dim]│[/dim]  {stats.skipped} skipped", end="")
    console.print(f"  [dim]│[/dim]  {stats.errors} errors", end="")
    console.print(f"  [dim]│[/dim]  {stats.chunks_created} chunks")

    if stats.error_files:
        console.print()
        for err in stats.error_files[:5]:
            console.print(f"  [red]  ✗ {escape(err[:100])}[/red]")
    console.print()


def _handle_status(store):
    """Handle /status command."""
    try:
        count = store.count()
        indexed = store.get_indexed_files()
        unique_files = len(set(indexed.keys()))

        console.print()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim", min_width=16)
        table.add_column(style="bold")

        table.add_row("Documents", str(count))
        table.add_row("Unique files", str(unique_files))
        table.add_row("Database", str(get_db_dir()))
        table.add_row("Model", "gemini-embedding-2-preview")

        # Count by type
        ext_counts: dict[str, int] = {}
        for fp in indexed.keys():
            ext = Path(fp).suffix.lower()
            category = "code"
            if ext in IMAGE_EXTENSIONS:
                category = "images"
            elif ext in AUDIO_EXTENSIONS:
                category = "audio"
            elif ext in VIDEO_EXTENSIONS:
                category = "video"
            elif ext in {".pdf", ".docx"}:
                category = "docs"
            ext_counts[category] = ext_counts.get(category, 0) + 1

        if ext_counts:
            breakdown = "  ".join(
                f"{v} {k}" for k, v in sorted(ext_counts.items(), key=lambda x: -x[1])
            )
            table.add_row("Breakdown", breakdown)

        console.print(Panel(table, border_style="dim", box=box.ROUNDED, padding=(0, 1)))
        console.print()
    except Exception:
        console.print("  [dim]No index found. Use[/dim] [cyan]/index <path>[/cyan] [dim]to create one.[/dim]\n")


def _handle_clear(store):
    """Handle /clear command."""
    console.print()
    try:
        answer = console.input("  [yellow]Clear entire index?[/yellow] [dim](y/N)[/dim] ")
        if answer.strip().lower() in ("y", "yes"):
            store.clear()
            console.print("  [green]✓[/green] Index cleared.\n")
        else:
            console.print("  [dim]Cancelled.[/dim]\n")
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]\n")


def _handle_watch(indexer, path: str):
    """Handle /watch command."""
    from embedded_finder.watcher import FileWatcher

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"  [red]✗[/red] Not a directory: {path}")
        return

    watcher = FileWatcher(indexer, [str(resolved)])
    console.print(f"\n  [dim]Watching[/dim] [{ACCENT}]{resolved}[/{ACCENT}] [dim]for changes… (Ctrl+C to stop)[/dim]\n")
    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        console.print(f"\n  [dim]Stopped watching.[/dim]\n")


def _handle_web(port: int = 8080):
    """Handle /web command."""
    from embedded_finder.web.app import create_app
    console.print(f"\n  [dim]Starting web UI at[/dim] [{ACCENT}]http://127.0.0.1:{port}[/{ACCENT}]")
    console.print(f"  [dim]Press Ctrl+C to stop[/dim]\n")
    app = create_app()
    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except KeyboardInterrupt:
        console.print(f"\n  [dim]Web server stopped.[/dim]\n")


# ── REPL ─────────────────────────────────────────────────────────────────────

def _parse_command(line: str) -> tuple[str, str]:
    """Parse a line into (command, args). Returns ('query', text) for search queries."""
    stripped = line.strip()
    if not stripped:
        return ("empty", "")
    if stripped.startswith("/"):
        parts = stripped[1:].split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        return (cmd, args)
    return ("query", stripped)


def run_interactive(api_key: str | None = None):
    """Run the interactive REPL."""
    console.clear()

    # Try to create components
    key = api_key or get_api_key()
    if not key:
        console.print(Panel(
            "[red bold]GOOGLE_API_KEY not set[/red bold]\n\n"
            f"[dim]Set it in your environment or .env file:[/dim]\n"
            f"  [cyan]export GOOGLE_API_KEY=your-key-here[/cyan]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2),
        ))
        sys.exit(1)

    embedder, store, indexer, engine = _create_components(key)

    # Show banner
    try:
        doc_count = store.count()
        indexed = store.get_indexed_files()
        file_count = len(set(indexed.keys()))
    except Exception:
        doc_count = 0
        file_count = 0

    print_banner(doc_count, file_count)
    console.print()
    console.print(f"  [dim]Type a query to search, or[/dim] [{ACCENT}]/help[/{ACCENT}] [dim]for commands.[/dim]")
    console.print()

    # REPL loop
    while True:
        try:
            line = console.input(f"[bold {ACCENT}]❯[/bold {ACCENT}] ")
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{ACCENT}]Goodbye![/{ACCENT}]\n")
            break

        cmd, args = _parse_command(line)

        try:
            if cmd == "empty":
                continue
            elif cmd == "query":
                _handle_search(engine, args)
            elif cmd == "help":
                print_help()
            elif cmd == "quit" or cmd == "exit" or cmd == "q":
                console.print(f"\n  [{ACCENT}]Goodbye![/{ACCENT}]\n")
                break
            elif cmd == "index":
                if not args:
                    console.print("  [red]Usage:[/red] /index <path>")
                else:
                    _handle_index(indexer, args)
            elif cmd == "reindex":
                if not args:
                    console.print("  [red]Usage:[/red] /reindex <path>")
                else:
                    _handle_index(indexer, args)
            elif cmd == "status":
                _handle_status(store)
            elif cmd == "clear":
                _handle_clear(store)
            elif cmd == "watch":
                if not args:
                    console.print("  [red]Usage:[/red] /watch <path>")
                else:
                    _handle_watch(indexer, args)
            elif cmd == "web":
                port = int(args) if args.strip().isdigit() else 8080
                _handle_web(port)
            else:
                console.print(f"  [dim]Unknown command:[/dim] [red]/{cmd}[/red] [dim]— type[/dim] [{ACCENT}]/help[/{ACCENT}]")
        except KeyboardInterrupt:
            console.print()
            continue
        except Exception as e:
            console.print(Panel(
                f"[red]{escape(str(e))}[/red]",
                title="[red bold]Error[/red bold]",
                border_style="red",
                box=box.ROUNDED,
                padding=(0, 1),
            ))
