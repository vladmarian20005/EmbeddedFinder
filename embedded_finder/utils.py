"""Shared utility functions for EmbeddedFinder."""


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form (compact: '2K', '1.4M')."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f}K"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}M"


def format_size_long(size_bytes: int) -> str:
    """Format file size in human-readable form (verbose: '2.0 KB', '1.4 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m = rem // 60
        return f"{h}h {m}m"
