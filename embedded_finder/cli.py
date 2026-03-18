"""CLI interface for EmbeddedFinder."""

import sys

import click

from embedded_finder import __version__
from embedded_finder.config import get_api_key, get_db_dir


@click.group()
@click.version_option(version=__version__, prog_name="efind")
def cli():
    """EmbeddedFinder - Semantic file search powered by Google Gemini embeddings."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--extensions", "-e", multiple=True, help="File extensions to include (e.g. .py .txt)")
def index(path, extensions):
    """Index files in a directory for semantic search."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.indexer import Indexer

    api_key = get_api_key()
    if not api_key:
        click.echo("Error: GOOGLE_API_KEY environment variable not set.", err=True)
        sys.exit(1)

    ext_set = set(extensions) if extensions else None

    try:
        embedder = Embedder(api_key=api_key)
        store = VectorStore()
        indexer = Indexer(embedder=embedder, store=store)

        click.echo(f"Indexing {path}...")

        def on_progress(file_info, stats):
            total = stats.total_files
            done = stats.indexed + stats.skipped + stats.errors
            click.echo(f"  [{done}/{total}] {file_info.name}", nl=True)

        stats = indexer.index_directory(path, extensions=ext_set, on_progress=on_progress)

        click.echo(f"\nDone! {stats.indexed} files indexed, "
                    f"{stats.skipped} skipped, {stats.errors} errors, "
                    f"{stats.chunks_created} chunks created.")

        if stats.error_files:
            click.echo("\nErrors:")
            for err in stats.error_files[:10]:
                click.echo(f"  - {err}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--top", "-n", default=10, help="Number of results to show.")
@click.option("--min-score", default=0.0, help="Minimum similarity score (0-1).")
@click.option("--plain", is_flag=True, help="Use plain text output (no rich formatting).")
def search(query, top, min_score, plain):
    """Search indexed files using natural language."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.search import SearchEngine
    from embedded_finder.ranker import rank_results, print_results

    api_key = get_api_key()
    if not api_key:
        click.echo("Error: GOOGLE_API_KEY environment variable not set.", err=True)
        sys.exit(1)

    try:
        embedder = Embedder(api_key=api_key)
        store = VectorStore()
        engine = SearchEngine(embedder=embedder, store=store)

        results = engine.search(query, n_results=top, min_score=min_score)
        results = rank_results(results, query)

        if not results:
            click.echo("No results found.")
            return

        if plain:
            click.echo(f"\nFound {len(results)} results for: \"{query}\"\n")
            for i, r in enumerate(results, 1):
                score_pct = f"{r.score * 100:.1f}%"
                click.echo(f"  {i}. [{score_pct}] {r.file_path}")
                click.echo(f"     {r.file_name} ({r.file_extension}, {_format_size(r.file_size)})")
                if r.snippet:
                    snippet = r.snippet.replace("\n", " ")[:120]
                    click.echo(f"     > {snippet}...")
                click.echo()
        else:
            print_results(results, query)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show index status."""
    from embedded_finder.store import VectorStore

    try:
        store = VectorStore()
        count = store.count()
        db_dir = get_db_dir()
        click.echo(f"Index location: {db_dir}")
        click.echo(f"Documents indexed: {count}")
        indexed = store.get_indexed_files()
        unique_files = len(set(indexed.keys()))
        click.echo(f"Unique files: {unique_files}")
    except Exception as e:
        click.echo(f"No index found. Run 'efind index <path>' to create one.")


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to clear the index?")
def clear():
    """Clear the entire index."""
    from embedded_finder.store import VectorStore

    try:
        store = VectorStore()
        store.clear()
        click.echo("Index cleared.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def watch(path):
    """Watch a directory and re-index files on changes."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.indexer import Indexer
    from embedded_finder.watcher import FileWatcher

    api_key = get_api_key()
    if not api_key:
        click.echo("Error: GOOGLE_API_KEY environment variable not set.", err=True)
        sys.exit(1)

    try:
        embedder = Embedder(api_key=api_key)
        store = VectorStore()
        indexer = Indexer(embedder=embedder, store=store)
        watcher = FileWatcher(indexer, [path])

        click.echo(f"Watching {path} for changes... (Ctrl+C to stop)")
        watcher.start()

        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nStopping watcher...")
            watcher.stop()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def reindex(path):
    """Force re-index a directory (only changed files)."""
    from embedded_finder.embedder import Embedder
    from embedded_finder.store import VectorStore
    from embedded_finder.indexer import Indexer

    api_key = get_api_key()
    if not api_key:
        click.echo("Error: GOOGLE_API_KEY environment variable not set.", err=True)
        sys.exit(1)

    try:
        embedder = Embedder(api_key=api_key)
        store = VectorStore()
        indexer = Indexer(embedder=embedder, store=store)

        click.echo(f"Re-indexing {path}...")
        stats = indexer.index_directory(path)

        click.echo(f"\nDone! {stats.indexed} files re-indexed, "
                    f"{stats.skipped} unchanged, {stats.errors} errors.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    cli()
