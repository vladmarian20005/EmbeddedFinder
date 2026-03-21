"""File system watching for incremental indexing."""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from embedded_finder.config import SUPPORTED_EXTENSIONS, IGNORE_DIRS
from embedded_finder.crawler import _should_ignore
from embedded_finder.indexer import Indexer

logger = logging.getLogger(__name__)


class _IndexHandler(FileSystemEventHandler):
    """Handles file system events by triggering re-indexing."""

    def __init__(self, indexer: Indexer):
        self._indexer = indexer

    def _should_process(self, path: str) -> bool:
        """Check if a file event should be processed."""
        p = Path(path)
        if not p.is_file():
            return False
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        # Skip files in ignored directories (reuse crawler's fnmatch-based logic)
        for part in p.parts:
            if _should_ignore(part, IGNORE_DIRS):
                return False
        return True

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"New file detected: {event.src_path}")
            try:
                self._indexer.index_file(event.src_path)
            except Exception as e:
                logger.error(f"Error indexing {event.src_path}: {e}")

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            try:
                self._indexer.index_file(event.src_path)
            except Exception as e:
                logger.error(f"Error re-indexing {event.src_path}: {e}")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path_str = str(Path(event.src_path).resolve())
        logger.info(f"File deleted: {path_str}")
        try:
            self._indexer._store.delete_by_file_path(path_str)
        except Exception as e:
            logger.error(f"Error removing {path_str} from index: {e}")


class FileWatcher:
    """Watches directories for file changes and triggers incremental indexing."""

    def __init__(self, indexer: Indexer, paths: list[str | Path]):
        self._indexer = indexer
        self._paths = [str(Path(p).resolve()) for p in paths]
        self._observer = Observer()
        self._handler = _IndexHandler(indexer)
        self._running = False

    def start(self) -> None:
        """Start watching directories."""
        for path in self._paths:
            self._observer.schedule(self._handler, path, recursive=True)
        self._observer.start()
        self._running = True
        logger.info(f"Watching {len(self._paths)} directory(ies) for changes...")

    def stop(self) -> None:
        """Stop watching."""
        if self._running:
            self._observer.stop()
            self._observer.join()
            self._running = False
            logger.info("File watcher stopped.")

    @property
    def is_running(self) -> bool:
        return self._running
