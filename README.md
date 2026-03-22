<p align="center">
  <h1 align="center">EmbeddedFinder</h1>
  <p align="center">
    <strong>Semantic file search for your local filesystem.</strong><br>
    Ask questions in plain English - find what you need across code, documents, images, audio, and video.
  </p>
  <p align="center">
    <a href="https://pypi.org/project/embedded-finder/"><img alt="PyPI" src="https://img.shields.io/pypi/v/embedded-finder"></a>
    <a href="https://pypi.org/project/embedded-finder/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/embedded-finder"></a>
    <a href="https://github.com/vladmarian20005/EmbeddedFinder/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  </p>
</p>

Powered by [Google Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) and [ChromaDB](https://www.trychroma.com/).

https://github.com/user-attachments/assets/7641c590-834e-4d8f-94a9-089fa449d863

## Why EmbeddedFinder?

Traditional file search (`grep`, `find`, `ag`) matches exact text. EmbeddedFinder understands *meaning*. Search for "error handling in payments" and find files about exception catching in billing code, even if those exact words never appear.

It works on everything: source code, config files, PDFs, Word documents, images, audio, and video, all in one index.

## Features

- **Natural language search** - describe what you're looking for, not keywords
- **Multimodal indexing** - code, text, PDFs, DOCX, images, audio, and video files
- **Interactive TUI** - rich terminal UI with slash commands, progress bars, and color-coded results
- **First-run setup wizard** - guided onboarding with API key validation
- **Incremental indexing** - content-hashed, only re-processes changed files
- **Batch embedding** - groups chunks into minimal API calls for fast indexing
- **File watching** - auto-reindex when files change on disk
- **One-shot CLI** - scriptable commands for CI/automation
- **Smart ranking** - filename matching, file type relevance, and content-aware scoring

## Quick start

### Install

```bash
pip install embedded-finder
```

Or from source:

```bash
git clone https://github.com/vladmarian20005/EmbeddedFinder.git
cd EmbeddedFinder
pip install .
```

### Run

```bash
efind
```

On first launch, a setup wizard walks you through:

1. Enter your [Google AI API key](https://aistudio.google.com/apikey) (free tier available)
2. The key is validated and saved securely to `~/.config/embeddedfinder/config.json`
3. Optionally index a directory right away

That's it. Start searching.

### Already have a key?

```bash
# Option A: environment variable
export GOOGLE_API_KEY=your-key-here

# Option B: .env file in your project root
echo "GOOGLE_API_KEY=your-key-here" > .env

# Option C: set it interactively
efind
# then type: /key set
```

## Usage

### Interactive mode (default)

```bash
efind
```

Type natural language queries at the `❯` prompt:

```
❯ database migration scripts
❯ files that handle image resizing
❯ error handling in the payment module
❯ screenshots of the dashboard
❯ audio files with speech
```

Results show similarity scores, file types, paths, and content snippets - color-coded by relevance.

### Slash commands

| Command             | Description                                |
| ------------------- | ------------------------------------------ |
| `/index <path>`     | Index a directory                          |
| `/reindex <path>`   | Re-index only changed files                |
| `/status`           | Show index statistics                      |
| `/clear`            | Clear the entire index                     |
| `/watch <path>`     | Watch a directory and auto-reindex         |
| `/key`              | Show current API key info                  |
| `/key set`          | Set or change your API key                 |
| `/key delete`       | Remove saved API key                       |
| `/key show`         | Reveal the full API key                    |
| `/help`             | Show available commands                    |
| `/quit` or `Ctrl+C` | Exit                                      |

### CLI commands

For scripting and one-off use:

```bash
# Index a directory
efind index ./src

# Index specific file types only
efind index ./src -e .py -e .ts

# Search
efind search "authentication middleware"

# Search with options
efind search "config parsing" --top 5 --min-score 0.7

# Plain text output (no colors, good for piping)
efind search "database models" --plain

# Re-index changed files only
efind reindex ./src

# Watch for changes
efind watch ./src

# Show index stats
efind status

# Clear the index
efind clear

# Check version
efind --version
```

## Supported file types

| Category | Extensions |
| -------- | ---------- |
| **Code** | `.py` `.js` `.ts` `.jsx` `.tsx` `.java` `.c` `.cpp` `.h` `.hpp` `.go` `.rs` `.rb` `.php` `.swift` `.kt` `.scala` `.sh` `.bash` `.zsh` `.lua` `.pl` `.ex` `.exs` `.r` `.m` `.sql` |
| **Markup** | `.html` `.css` `.scss` `.less` `.xml` `.svg` |
| **Config** | `.json` `.yaml` `.yml` `.toml` `.ini` `.cfg` `.conf` |
| **Text** | `.txt` `.md` `.rst` `.csv` |
| **Documents** | `.pdf` `.docx` |
| **Images** | `.png` `.jpg` `.jpeg` `.gif` `.webp` `.bmp` |
| **Audio** | `.mp3` `.wav` `.ogg` `.flac` `.m4a` |
| **Video** | `.mp4` `.mov` `.avi` `.mkv` `.webm` |

Images, audio, and video are embedded natively using Gemini's multimodal capabilities - no transcription or OCR needed.

PDFs with 6 or fewer pages are embedded natively; larger PDFs use text extraction for efficiency.

## How it works

```
 Directory          EmbeddedFinder                    ChromaDB
 ─────────     ─────────────────────────     ─────────────────────

  files/ ──→  1. Crawl  (skip .git, etc.)
           ──→  2. Extract  (text / bytes)
           ──→  3. Chunk   (~2000 tokens)
           ──→  4. Hash    (SHA-256 dedup)
           ──→  5. Embed   (Gemini API)   ──→  Store vectors

  query  ──→  6. Embed query              ──→  Nearest-neighbor
           ──→  7. Deduplicate by file             search
           ──→  8. Re-rank & boost        ──→  Results
```

- **Content hashing** - files are fingerprinted with SHA-256; re-indexing skips anything unchanged
- **Batch embedding** - text chunks are grouped into batches (up to 100 per API call) for throughput
- **Rate limiting** - built-in token bucket limiter respects Gemini API quotas
- **Parallel processing** - multi-threaded extraction and embedding with up to 4 workers
- **Smart ranking** - results are boosted by filename match, file type relevance to query, content overlap, and path depth
- **Directory filtering** - hidden directories (starting with `.`) and common non-content directories (`node_modules`, `__pycache__`, `.venv`, `dist`, `build`, etc.) are automatically skipped during crawling

## Configuration

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `GOOGLE_API_KEY` | - | Google AI API key (required) |
| `EMBEDDEDFINDER_DB_DIR` | `.embeddedfinder/db` | Path to the ChromaDB database |

The API key can also be stored via the setup wizard or `/key set`, which saves it to `~/.config/embeddedfinder/config.json` with owner-only permissions.

## Project structure

```
embedded_finder/
├── cli.py            # Click CLI, subcommands + TUI launcher
├── tui.py            # Interactive Rich-based REPL
├── config.py         # Settings, supported extensions, env vars
├── config_store.py   # Persistent config file management
├── crawler.py        # Recursive file discovery
├── extractor.py      # Text extraction, chunking, MIME detection
├── embedder.py       # Gemini Embedding API client + batching
├── store.py          # ChromaDB vector store
├── indexer.py        # Orchestrates crawl → extract → embed → store
├── search.py         # Query embedding + nearest-neighbor search
├── ranker.py         # Result ranking, dedup, and formatting
├── rate_limiter.py   # Token bucket rate limiter
└── watcher.py        # Filesystem watcher (watchdog)
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/vladmarian20005/EmbeddedFinder.git
cd EmbeddedFinder
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=embedded_finder
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request

## License

MIT
