# EmbeddedFinder

Semantic file search for your local filesystem. Ask questions in plain English and find the files you need — across code, documents, images, audio, and video.

Powered by [Google Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) and [ChromaDB](https://www.trychroma.com/).

```
❯ efind

╭─ ◆ EmbeddedFinder  v0.1.0 ─────────────────────────────────────╮
│   Semantic file search powered by Gemini Embedding 2             │
│   ● 142 files  (387 chunks)  │  .embeddedfinder/db              │
╰──────────────────────────────────────────────────────────────────╯

  Type a query to search, or /help for commands.

❯ functions that validate user authentication tokens

  5 results  (0.3s)  │  "functions that validate user authentication tokens"

   1  95%   PY   auth.py  4K
        src/auth/auth.py
        ▸ def validate_token(token: str) -> bool: ...

   2  87%   PY   middleware.py  2K
        src/middleware/middleware.py
        ▸ class AuthMiddleware: def process_request(self, req)...
```

## Features

- **Natural language search** — describe what you're looking for, not just keywords
- **Multimodal** — indexes code, text, PDFs, DOCX, images, audio, and video files
- **Interactive TUI** — Claude Code-style REPL with slash commands, spinners, and color-coded results
- **Incremental indexing** — only re-processes changed files
- **File watching** — automatically re-indexes when files change on disk
- **Web UI** — browser-based search interface via Flask
- **One-shot CLI** — scriptable commands for CI/automation

## Installation

### From PyPI (recommended)

```bash
pip install embedded-finder
```

### From source

```bash
git clone https://github.com/vladmarian20005/EmbeddedFinder.git
cd EmbeddedFinder
pip install .
```

### For development

```bash
git clone https://github.com/vladmarian20005/EmbeddedFinder.git
cd EmbeddedFinder
pip install -e ".[dev]"
```

## Setup

### 1. Get a Google API key

EmbeddedFinder uses the Gemini Embedding API. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Set the API key

Pick one:

```bash
# Option A: environment variable
export GOOGLE_API_KEY=your-key-here

# Option B: .env file in your project root
echo "GOOGLE_API_KEY=your-key-here" > .env
```

### 3. Index your files

```bash
# Interactive mode (launches the TUI)
efind

# Then type:
#   /index ./my-project

# Or one-shot from the command line:
efind index ./my-project
```

That's it. You're ready to search.

## Usage

### Interactive mode (default)

Run `efind` with no arguments to enter the interactive REPL:

```bash
efind
```

Then type natural language queries at the `❯` prompt:

```
❯ database migration scripts
❯ files that handle image resizing
❯ error handling in the payment module
```

#### Slash commands

| Command            | Description                             |
| ------------------ | --------------------------------------- |
| `/index <path>`    | Index a directory                       |
| `/reindex <path>`  | Re-index only changed files             |
| `/status`          | Show index statistics                   |
| `/clear`           | Clear the entire index                  |
| `/watch <path>`    | Watch a directory and auto-reindex      |
| `/web [port]`      | Start the web UI (default: 8080)        |
| `/help`            | Show available commands                 |
| `/quit` or `Ctrl+C`| Exit                                   |

### One-shot CLI commands

For scripting or one-off use:

```bash
# Index a directory
efind index ./src

# Index only specific file types
efind index ./src -e .py -e .ts

# Search from the command line
efind search "authentication middleware"

# Search with options
efind search "config parsing" --top 5 --min-score 0.7

# Plain text output (no colors, good for piping)
efind search "database models" --plain

# Re-index changed files
efind reindex ./src

# Watch for changes
efind watch ./src

# Show index stats
efind status

# Clear the index
efind clear

# Start web UI
efind web --port 3000

# Check version
efind --version
```

### Web UI

```bash
efind web
# → http://127.0.0.1:8080
```

Or from interactive mode:

```
❯ /web 3000
```

Provides a browser-based search interface with the same search backend.

## Supported file types

### Code & text
`.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.c` `.cpp` `.h` `.hpp` `.go` `.rs` `.rb` `.php` `.swift` `.kt` `.scala` `.sh` `.bash` `.zsh` `.lua` `.pl` `.ex` `.exs` `.r` `.m` `.sql` `.html` `.css` `.scss` `.less` `.xml` `.svg` `.json` `.csv` `.yaml` `.yml` `.toml` `.ini` `.cfg` `.conf` `.txt` `.md` `.rst`

### Documents
`.pdf` (text-extracted for >6 pages, natively embedded for ≤6 pages) `.docx`

### Images (native multimodal embedding)
`.png` `.jpg` `.jpeg` `.gif` `.webp` `.bmp`

### Audio (native multimodal embedding)
`.mp3` `.wav` `.ogg` `.flac` `.m4a`

### Video (native multimodal embedding)
`.mp4` `.mov` `.avi` `.mkv` `.webm`

## How it works

1. **Crawl** — recursively walks a directory, skipping `.git`, `node_modules`, `__pycache__`, and other common noise directories
2. **Extract** — reads file content: text extraction for code/docs, raw bytes for multimodal files (images, audio, video)
3. **Chunk** — splits large text files into overlapping chunks (~2000 tokens each) to stay within embedding limits
4. **Embed** — sends content to the Gemini Embedding API (`gemini-embedding-2-preview`, 3072 dimensions) to produce vector representations
5. **Store** — saves embeddings in a local ChromaDB database (`.embeddedfinder/db`)
6. **Search** — embeds your query, performs nearest-neighbor search in ChromaDB, deduplicates by file, and ranks results with a filename-match boost

Files are fingerprinted by content hash, so re-indexing only processes files that have actually changed.

## Configuration

All configuration is via environment variables:

| Variable               | Default                        | Description                    |
| ---------------------- | ------------------------------ | ------------------------------ |
| `GOOGLE_API_KEY`       | —                              | Google AI API key (required)   |
| `EMBEDDEDFINDER_DB_DIR`| `.embeddedfinder/db`           | Path to the ChromaDB database  |

## Project structure

```
embedded_finder/
├── cli.py          # Click CLI — subcommands + TUI launcher
├── tui.py          # Interactive Rich-based REPL
├── config.py       # Settings, supported extensions, env vars
├── crawler.py      # Recursive file discovery
├── extractor.py    # Text extraction, chunking, MIME detection
├── embedder.py     # Gemini Embedding API client
├── store.py        # ChromaDB vector store
├── indexer.py      # Orchestrates crawl → extract → embed → store
├── search.py       # Query embedding + nearest-neighbor search
├── ranker.py       # Result ranking and formatting
├── watcher.py      # Filesystem watcher (watchdog)
└── web/
    └── app.py      # Flask web UI
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=embedded_finder
```

## License

MIT
