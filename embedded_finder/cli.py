"""CLI interface for EmbeddedFinder."""

import click

from embedded_finder import __version__


@click.group()
@click.version_option(version=__version__, prog_name="efind")
def cli():
    """EmbeddedFinder - Semantic file search powered by Google Gemini embeddings."""
    pass


@cli.command()
def status():
    """Show index status."""
    click.echo("No index found. Run 'efind index <path>' to create one.")


if __name__ == "__main__":
    cli()
