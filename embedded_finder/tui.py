"""Interactive terminal UI for EmbeddedFinder — redesigned REPL."""

import getpass
import os
import sys
import time
import shutil
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text
from rich.markup import escape
from rich import box

from embedded_finder import __version__
from embedded_finder.config import (
    get_api_key, get_db_dir, SUPPORTED_EXTENSIONS,
    IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS,
)

console = Console()

# -- Theme constants ----------------------------------------------------------

PRIMARY = "magenta"
SECONDARY = "cyan"
SUCCESS = "green"
DIM = "dim"

SCORE_THRESHOLDS = [(90, "bold green"), (75, "green"), (60, "cyan"), (45, "blue")]

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
    for threshold, style in SCORE_THRESHOLDS:
        if pct >= threshold:
            return style
    return "dim"


def _score_bar(score: float, width: int = 20) -> Text:
    """Generate a mini block-fill score bar."""
    filled = int(score * width)
    style = _score_style(score)
    bar = Text()
    bar.append("\u2588" * filled, style=style)
    bar.append("\u2591" * (width - filled), style="dim")
    return bar


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
        return "\u2026" + path[-(max_len - 1):]
    return parts[0] + "/\u2026/" + "/".join(parts[-2:])


def _mask_key(key: str) -> str:
    """Mask an API key, showing only the first 5 and last 3 chars."""
    if len(key) <= 10:
        return key[:3] + "\u2022" * (len(key) - 3)
    return key[:5] + "\u2022" * (len(key) - 8) + key[-3:]


def _get_key_source() -> str:
    """Return where the active API key comes from."""
    if os.environ.get("GOOGLE_API_KEY"):
        return "env"
    from embedded_finder.config_store import get_config_value
    if get_config_value("api_key"):
        return "config"
    return "none"


# -- Onboarding wizard -------------------------------------------------------

def _validate_api_key(key: str) -> tuple[bool, str]:
    """Test an API key by making a small embed call. Returns (ok, error_msg)."""
    try:
        from embedded_finder.embedder import Embedder
        embedder = Embedder(api_key=key)
        embedder.embed_text("test")
        return True, ""
    except Exception as e:
        return False, str(e)


def _prompt_and_save_key() -> str | None:
    """Prompt for a key, validate, and optionally save. Returns key or None."""
    from embedded_finder.config_store import set_config_value, get_config_path

    console.print()
    try:
        key = getpass.getpass("  Paste your API key (hidden): ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]")
        return None

    if not key:
        console.print("  [red]No key entered.[/red]")
        return None

    # Validate with spinner
    valid = [None, ""]

    def do_validate():
        valid[0], valid[1] = _validate_api_key(key)

    t = threading.Thread(target=do_validate, daemon=True)
    t.start()
    with console.status(f"  [{PRIMARY}]Validating key\u2026[/{PRIMARY}]", spinner="dots"):
        t.join()

    if not valid[0]:
        console.print(f"  [red]Invalid key:[/red] {escape(valid[1])}")
        return None

    console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] Key is valid!")

    # Offer to save
    try:
        save = console.input(f"  Save to config file? [dim](Y/n)[/dim] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return key

    if save not in ("n", "no"):
        set_config_value("api_key", key)
        console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] Saved to [dim {SECONDARY}]{get_config_path()}[/dim {SECONDARY}]")

    return key


def _run_onboarding_wizard() -> str | None:
    """Full first-run onboarding wizard. Returns API key or None."""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold]Welcome to EmbeddedFinder[/bold]  [dim]v{__version__}[/dim]\n\n"
            f"Semantic file search powered by [bold {SECONDARY}]Gemini Embedding 2[/bold {SECONDARY}].\n"
            f"Index your codebase and search by meaning, not keywords.\n\n"
            f"To get started you need a [bold]Google AI API key[/bold] (free tier available).\n"
            f"Get one at: [{SECONDARY}]https://aistudio.google.com/apikey[/{SECONDARY}]"
        ),
        border_style=PRIMARY,
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    max_attempts = 3
    key = None
    for attempt in range(max_attempts):
        key = _prompt_and_save_key()
        if key:
            break
        if attempt < max_attempts - 1:
            try:
                retry = console.input("  Try again? [dim](Y/n)[/dim] ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                console.print()
                break
            if retry in ("n", "no"):
                break

    if not key:
        console.print(f"\n  [dim]You can set your key later via[/dim] [{PRIMARY}]/key set[/{PRIMARY}] "
                      f"[dim]or the[/dim] [{SECONDARY}]GOOGLE_API_KEY[/{SECONDARY}] [dim]env var.[/dim]\n")
        return None

    # Offer to index a directory
    console.print()
    console.print(f"  [{PRIMARY}]What would you like to index?[/{PRIMARY}]")
    console.print(f"  [dim]Enter a directory path, or press Enter to skip.[/dim]")
    console.print()

    try:
        index_path = console.input(f"  [{SECONDARY}]Path:[/{SECONDARY}] ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print()
        return key

    if index_path:
        resolved = Path(index_path).expanduser().resolve()
        if not resolved.is_dir():
            console.print(f"  [red]\u2717[/red] Not a directory: {index_path}")
        else:
            embedder, store, indexer, engine = _create_components(key)
            if indexer:
                stats = _index_with_progress(indexer, str(resolved))
                if stats is None:
                    console.print("  [dim]No supported files found.[/dim]")
                else:
                    console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] [bold]{stats.indexed}[/bold] indexed", end="")
                    console.print(f"  [dim]\u2502[/dim]  {stats.skipped} skipped", end="")
                    console.print(f"  [dim]\u2502[/dim]  {stats.errors} errors", end="")
                    console.print(f"  [dim]\u2502[/dim]  {stats.chunks_created} chunks")
                console.print()

    return key


# -- Banner -------------------------------------------------------------------

def print_banner(doc_count: int = 0, file_count: int = 0):
    """Print the welcome banner."""
    inner = Text()
    inner.append("\n")
    inner.append("   EmbeddedFinder", style="bold white")
    inner.append(f"  v{__version__}\n", style="dim")
    inner.append("   Semantic file search powered by ", style="dim")
    inner.append("Gemini Embedding 2", style=f"bold {SECONDARY}")
    inner.append("\n\n", style="dim")

    if doc_count > 0:
        inner.append(f"   {file_count} files indexed", style=SUCCESS)
        inner.append(f"   {doc_count} chunks", style="dim")
    else:
        inner.append("   No files indexed yet", style="dim yellow")
    inner.append(f"   {get_db_dir()}", style="dim")
    inner.append("\n")

    console.print(Panel(
        inner,
        border_style=f"dim {PRIMARY}",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


# -- Help ---------------------------------------------------------------------

def print_help():
    """Print grouped help information."""
    console.print()

    def section(title: str, rows: list[tuple[str, str]]):
        console.print(f"  [bold {PRIMARY}]{title}[/bold {PRIMARY}]")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style=f"bold {SECONDARY}", min_width=22)
        t.add_column(style="dim")
        for cmd, desc in rows:
            t.add_row(cmd, desc)
        console.print(t)
        console.print()

    section("Search", [
        ("[query text]", "Search files by natural language description"),
    ])
    section("Index Management", [
        ("/index <path>", "Index a directory for searching"),
        ("/reindex <path>", "Re-index only changed files"),
        ("/status", "Show index statistics"),
        ("/clear", "Clear the entire index"),
        ("/watch <path>", "Watch directory for changes"),
    ])
    section("Configuration", [
        ("/key", "Show current API key info"),
        ("/key set", "Set or change API key"),
        ("/key delete", "Remove saved API key"),
        ("/key show", "Show full unmasked key"),
    ])
    section("Other", [
        ("/web [port]", "Start web UI (default: 8080)"),
        ("/help", "Show this help"),
        ("/quit  or  Ctrl+C", "Exit"),
    ])


# -- Result display -----------------------------------------------------------

def print_results(results, query: str, elapsed: float):
    """Print search results with score bars."""
    if not results:
        console.print(f"\n  [dim]No results found for[/dim] [italic]\"{escape(query)}\"[/italic]\n")
        return

    console.print()
    header = Text()
    header.append(f"  {len(results)} result{'s' if len(results) != 1 else ''}", style="bold")
    header.append(f"  ({elapsed:.1f}s)", style="dim")
    header.append(f"  ", style="dim")
    header.append(f"\"{escape(query)}\"", style=f"italic {PRIMARY}")
    console.print(header)
    console.print()

    for i, r in enumerate(results):
        _print_result_row(i + 1, r, show_snippet=(i < 5))

    console.print()


def _print_result_row(num: int, r, show_snippet: bool = True):
    """Print a single result row with score bar."""
    score_pct = int(r.score * 100)
    style = _score_style(r.score)
    badge = _type_badge(r.file_extension)
    bar = _score_bar(r.score)

    # Line 1: number, score, bar, type badge, filename, size
    line = Text("  ")
    line.append(f"{num:>2}", style="bold dim")
    line.append("   ")
    line.append(f"{score_pct}%", style=f"bold {style}")
    line.append("  ")
    line.append_text(bar)
    line.append("   ")
    line.append_text(badge)
    line.append("   ")
    line.append(r.file_name, style="bold white")
    line.append(f"  {_format_size(r.file_size)}", style="dim")
    console.print(line)

    # Line 2: path
    path_display = _truncate_path(r.file_path)
    console.print(f"         [dim {SECONDARY}]{path_display}[/dim {SECONDARY}]")

    # Line 3: snippet
    if show_snippet and r.snippet:
        snippet = r.snippet.replace("\n", " ").strip()[:150]
        console.print(f"         [dim italic]\u25b8 {escape(snippet)}[/dim italic]")

    console.print()


# -- Spinner helpers ----------------------------------------------------------

def _search_with_spinner(engine, query: str, n_results: int, min_score: float):
    """Run search with animated spinner."""
    result_holder = [None, None]

    def do_search():
        try:
            result_holder[0] = engine.search(query, n_results=n_results, min_score=min_score)
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=do_search, daemon=True)
    start = time.time()
    t.start()

    with console.status(f"[{PRIMARY}]Searching\u2026[/{PRIMARY}]", spinner="dots"):
        t.join()

    elapsed = time.time() - start

    if result_holder[1]:
        raise result_holder[1]

    return result_holder[0], elapsed


def _index_with_progress(indexer, path: str, extensions=None):
    """Run indexing with Rich progress bar."""
    from embedded_finder.crawler import crawl

    console.print(f"\n  [dim]Scanning {escape(path)}\u2026[/dim]")
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


# -- Component factory --------------------------------------------------------

def _create_components(api_key: str | None = None):
    """Create the core components."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.indexer import Indexer
    from embedded_finder.search import SearchEngine

    key = api_key or get_api_key()
    if not key:
        return None, None, None, None
    embedder = Embedder(api_key=key)
    store = VectorStore()
    indexer = Indexer(embedder=embedder, store=store)
    engine = SearchEngine(embedder=embedder, store=store)
    return embedder, store, indexer, engine


# -- Command handlers ---------------------------------------------------------

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
        console.print(f"  [red]\u2717[/red] Not a directory: {path}")
        return

    stats = _index_with_progress(indexer, str(resolved))
    if stats is None:
        console.print("  [dim]No supported files found.[/dim]")
        return

    console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] [bold]{stats.indexed}[/bold] indexed", end="")
    console.print(f"  [dim]\u2502[/dim]  {stats.skipped} skipped", end="")
    console.print(f"  [dim]\u2502[/dim]  {stats.errors} errors", end="")
    console.print(f"  [dim]\u2502[/dim]  {stats.chunks_created} chunks")

    if stats.error_files:
        console.print()
        for err in stats.error_files[:5]:
            console.print(f"  [red]  \u2717 {escape(err[:100])}[/red]")
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
        console.print(f"  [dim]No index found. Use[/dim] [{SECONDARY}]/index <path>[/{SECONDARY}] [dim]to create one.[/dim]\n")


def _handle_clear(store):
    """Handle /clear command."""
    console.print()
    try:
        answer = console.input("  [yellow]Clear entire index?[/yellow] [dim](y/N)[/dim] ")
        if answer.strip().lower() in ("y", "yes"):
            store.clear()
            console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] Index cleared.\n")
        else:
            console.print("  [dim]Cancelled.[/dim]\n")
    except (KeyboardInterrupt, EOFError):
        console.print("\n  [dim]Cancelled.[/dim]\n")


def _handle_watch(indexer, path: str):
    """Handle /watch command."""
    from embedded_finder.watcher import FileWatcher

    resolved = Path(path).resolve()
    if not resolved.is_dir():
        console.print(f"  [red]\u2717[/red] Not a directory: {path}")
        return

    watcher = FileWatcher(indexer, [str(resolved)])
    console.print(f"\n  [dim]Watching[/dim] [{SECONDARY}]{resolved}[/{SECONDARY}] [dim]for changes\u2026 (Ctrl+C to stop)[/dim]\n")
    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        console.print("\n  [dim]Stopped watching.[/dim]\n")


def _handle_web(port: int = 8080):
    """Handle /web command."""
    from embedded_finder.web.app import create_app
    console.print(f"\n  [dim]Starting web UI at[/dim] [{SECONDARY}]http://127.0.0.1:{port}[/{SECONDARY}]")
    console.print("  [dim]Press Ctrl+C to stop[/dim]\n")
    app = create_app()
    try:
        app.run(host="127.0.0.1", port=port, debug=False)
    except KeyboardInterrupt:
        console.print("\n  [dim]Web server stopped.[/dim]\n")


def _handle_key(args: str):
    """Handle /key command and subcommands."""
    from embedded_finder.config_store import get_config_value, delete_config_value, get_config_path

    subcmd = args.strip().lower()

    if subcmd == "set":
        key = _prompt_and_save_key()
        if key:
            console.print(f"\n  [{SUCCESS}]\u2713[/{SUCCESS}] Key updated. Restart to use the new key.\n")
        return

    if subcmd == "delete":
        if not get_config_value("api_key"):
            console.print("  [dim]No key saved in config file.[/dim]\n")
            return
        try:
            answer = console.input("  [yellow]Delete saved API key?[/yellow] [dim](y/N)[/dim] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Cancelled.[/dim]\n")
            return
        if answer.strip().lower() in ("y", "yes"):
            delete_config_value("api_key")
            console.print(f"  [{SUCCESS}]\u2713[/{SUCCESS}] Key removed from config file.\n")
        else:
            console.print("  [dim]Cancelled.[/dim]\n")
        return

    if subcmd == "show":
        key = get_api_key()
        if not key:
            console.print("  [dim]No API key configured.[/dim]\n")
            return
        try:
            answer = console.input("  [yellow]Show full API key?[/yellow] [dim](y/N)[/dim] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Cancelled.[/dim]\n")
            return
        if answer.strip().lower() in ("y", "yes"):
            console.print(f"  {key}\n")
        else:
            console.print("  [dim]Cancelled.[/dim]\n")
        return

    # Default: /key — show info
    key = get_api_key()
    source = _get_key_source()
    console.print()
    if key:
        console.print(f"  [bold]Key:[/bold]     {_mask_key(key)}")
        source_label = "environment variable" if source == "env" else "config file"
        console.print(f"  [bold]Source:[/bold]  {source_label}")
        console.print(f"  [bold]Config:[/bold]  [dim {SECONDARY}]{get_config_path()}[/dim {SECONDARY}]")
    else:
        console.print(f"  [dim]No API key configured.[/dim]")
        console.print(f"  [dim]Use[/dim] [{PRIMARY}]/key set[/{PRIMARY}] [dim]or set[/dim] [{SECONDARY}]GOOGLE_API_KEY[/{SECONDARY}] [dim]env var.[/dim]")
    console.print()


# -- Error panels with contextual tips ---------------------------------------

def _print_error(e: Exception):
    """Print an error panel with contextual tips."""
    msg = str(e)
    tip = ""
    lower_msg = msg.lower()
    if "rate" in lower_msg or "429" in lower_msg or "quota" in lower_msg:
        tip = f"\n\n[dim]Tip: Wait a moment and retry, or check your API quota.[/dim]"
    elif "api key" in lower_msg or "401" in lower_msg or "403" in lower_msg:
        tip = f"\n\n[dim]Tip: Run [{PRIMARY}]/key set[/{PRIMARY}] to update your API key.[/dim]"
    elif "connect" in lower_msg or "timeout" in lower_msg:
        tip = f"\n\n[dim]Tip: Check your internet connection and retry.[/dim]"

    console.print(Panel(
        f"[red]{escape(msg)}[/red]{tip}",
        title="[red bold]Error[/red bold]",
        border_style="red",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


# -- REPL ---------------------------------------------------------------------

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

    key = api_key or get_api_key()
    if not key:
        key = _run_onboarding_wizard()
        if not key:
            sys.exit(1)

    embedder, store, indexer, engine = _create_components(key)

    try:
        doc_count = store.count()
        indexed = store.get_indexed_files()
        file_count = len(set(indexed.keys()))
    except Exception:
        doc_count = 0
        file_count = 0

    print_banner(doc_count, file_count)
    console.print()
    console.print(f"  [dim]Type a query to search, or[/dim] [{PRIMARY}]/help[/{PRIMARY}] [dim]for commands.[/dim]")
    console.print()

    while True:
        try:
            line = console.input(f"[bold {PRIMARY}]\u276f[/bold {PRIMARY}] ")
        except (KeyboardInterrupt, EOFError):
            console.print(f"\n  [{PRIMARY}]Goodbye![/{PRIMARY}]\n")
            break

        cmd, args = _parse_command(line)

        try:
            if cmd == "empty":
                continue
            elif cmd == "query":
                _handle_search(engine, args)
            elif cmd == "help":
                print_help()
            elif cmd in ("quit", "exit", "q"):
                console.print(f"\n  [{PRIMARY}]Goodbye![/{PRIMARY}]\n")
                break
            elif cmd == "index":
                if not args:
                    console.print(f"  [red]Usage:[/red] /index <path>")
                else:
                    _handle_index(indexer, args)
            elif cmd == "reindex":
                if not args:
                    console.print(f"  [red]Usage:[/red] /reindex <path>")
                else:
                    _handle_index(indexer, args)
            elif cmd == "status":
                _handle_status(store)
            elif cmd == "clear":
                _handle_clear(store)
            elif cmd == "watch":
                if not args:
                    console.print(f"  [red]Usage:[/red] /watch <path>")
                else:
                    _handle_watch(indexer, args)
            elif cmd == "web":
                port = int(args) if args.strip().isdigit() else 8080
                _handle_web(port)
            elif cmd == "key":
                _handle_key(args)
            else:
                console.print(f"  [dim]Unknown command:[/dim] [red]/{cmd}[/red] [dim]\u2014 type[/dim] [{PRIMARY}]/help[/{PRIMARY}]")
        except KeyboardInterrupt:
            console.print()
            continue
        except Exception as e:
            _print_error(e)
