# Ollama Document Repository

A local document processing and Q&A application powered by Ollama. Ingest PDFs, DOCX,
Markdown, HTML, RTF, and text files for summarization, semantic search, and
source-cited question answering — and drive it all from a full-screen **interactive
terminal UI**.

## Features

- **Interactive TUI** — run `./ollama-docs` with no subcommand for a full-screen
  terminal interface: type questions, index documents, pick a document to
  summarize, switch models, and more, with a live database/document panel and
  command log.
- **Multi-database support** — keep several collections side by side (e.g. one for
  manuals, one for music articles) in a single vector store. View, create,
  rename, delete, and **select which database is active** for queries.
- **Per-document management** — add documents to a database without wiping it,
  **remove a single document** from a database, and **move/copy a document
  between databases** (no re-embedding).
- **Thorough Q&A** — answers retrieve more context and are structured across
  multiple source chunks by default (override with `--quick` for concise replies).
- **Multi-format support**: PDF, DOCX, TXT, MD, HTML, RTF
- **Semantic search**: vector embeddings via ChromaDB for relevant document retrieval
- **Local processing**: everything runs against your Ollama instance — no data leaves your network
- **Summarization**: generate summaries of any document in the repository
- **Interactive Q&A**: ask questions about your document collection, with source citations
- **CLI subcommands**: every capability is also available as a one-shot CLI command
- **Export/Import**: export any database and chunks to JSON
- **Raw search**: search documents without LLM answer generation
- **Configurable endpoint & models** via `.env` or environment variables, no code edits

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aspen8494/Ollama_Doc_Repo.git
cd Ollama_Doc_Repo

# Run setup — installs Python dependencies, creates folders, pulls the models.
# Point it at your Ollama instance first if it is not on the default host:
OLLAMA_HOST=http://192.168.68.59:11435 python setup.py

# Add documents to the documents folder
cp /path/to/your/docs/* documents/

# Launch the interactive TUI (no arguments)
./ollama-docs

# ...or use one-shot commands:
./ollama-docs ingest
./ollama-docs ask "What is this about?"
```

## The Interactive TUI

Running `./ollama-docs` with **no subcommand** launches a full-screen TUI
(built on [Textual](https://textual.textual.io/)). Use it like this:

- **Ask a question** — just type a natural-language question and press Enter.
  The TUI retrieves the most relevant chunks and generates a cited answer.
- **Run a command** — prefix a line with `/` (see the list below), or press a
  keybinding (shown in the footer).
- **Pick a document** in the left panel and press **Enter** to summarize it.

All Ollama work runs in background worker threads, so the UI stays responsive.

### TUI commands

| Command | Description |
| --- | --- |
| `/ingest` / `/ingest force` | Index every file in `documents/` (force re-does everything) |
| `/ingest-file <path>` | Index a single file |
| `/search <query>` | Raw semantic search (no LLM answer) |
| `/ask <question>` | Generate a cited LLM answer |
| `/summarize <document>` | Summarize one document |
| `/list` | List documents in the active database with chunk counts |
| `/stats` | Statistics for the active database |
| `/models [name]` | List Ollama models; optionally set the chat model |
| `/embedding-model <name>` | Choose the embedding model |
| `/results <n>` | Set chunks retrieved per question |
| `/export [path.json]` | Export the active database to JSON |
| `/clear-force` | Wipe the active database |
| **Databases** ||
| `/db` | List all databases and show the active one |
| `/db new <name>` | Create a new, empty database |
| `/db use <name>` | Make a database the active one (also `/use <name>`) |
| `/db delete <name>` | Delete a whole database |
| `/db rename <old> <new>` | Rename a database |
| `/remove <document>` | Remove one document from the active database |
| `/reassign <src> <dst>` | **Move** a document from its source database to a destination database |
| `/reassign <src> <dst> copy` | **Copy** a document to a destination database (keep original) |
| `/help` | Show the command list |
| `/quit` | Exit |

### Keybindings

`i` ingest · `s` search · `a` ask · `l` list · `m` models · `d` switch database · `?` help · `q` quit

## CLI Subcommands

Every TUI capability is also a one-shot command.

### `ingest` — Process documents
```bash
./ollama-docs ingest                   # ingest all docs in the folder
./ollama-docs ingest --force           # clear only THIS database first, then re-ingest
./ollama-docs ingest -i /path/document.pdf    # one specific file
./ollama-docs --db music ingest                        # ingest into the 'music' db
./ollama-docs --db manuals ingest -i docs/saw.md       # add one file to 'manuals'
```

> `ingest` **adds** to the target database without touching existing data.
> `--force` clears only that one database, not all of them.

### Databases — create, select, rename, delete
The active database is remembered between runs (`.ollama-docs-state.json`).
Target a specific one on any command with `--db <name>`, or make it default:

```bash
./ollama-docs db new manuals          # create an empty database
./ollama-docs db new music            # e.g. a separate collection
./ollama-docs db list                 # list databases, active one, and counts
./ollama-docs db use manuals          # make 'manuals' the active database
./ollama-docs --db music ask "..."    # or target one per-command
./ollama-docs db rename music musicdb # rename, data preserved
./ollama-docs db delete music         # drop it
```

### `reassign` / `remove` — per-document management
```bash
# Move one document between databases (no re-embedding):
./ollama-docs reassign docs/saw.md music   # move from active db into 'music'

# Copy instead of move:
./ollama-docs reassign docs/saw.md music --copy

# Remove one document without wiping the whole database:
./ollama-docs remove docs/saw.md --from music
```

### `ask` — Query documents
```bash
./ollama-docs ask "What are the main topics?"
./ollama-docs ask -n 10 "Detailed question"      # retrieve 10 chunks
./ollama-docs ask --quick "Quick question"       # concise single-point answer
./ollama-docs ask --no-sources "Quick question" # hide citations
./ollama-docs ask -i                          # interactive prompt-based session
./ollama-docs ask -m qwen3:8b "question"       # override the chat model
./ollama-docs --db music ask "What tracks are mentioned?"  # query one database
```

### `summarize` — Generate summaries
```bash
./ollama-docs summarize                     # list all documents
./ollama-docs summarize documents/report.pdf
```

### `list` — List documents in the active database
```bash
./ollama-docs list                  # active database
./ollama-docs --db music list       # a specific database
```

### `stats` — Database statistics
```bash
./ollama-docs stats            # active database + list of all databases
./ollama-docs db stats music   # a specific database
```

### `clear` — Clear a database
```bash
./ollama-docs --db music clear --confirm   # deletes only 'music'; others kept
```

### `models` — Check Ollama models
```bash
./ollama-docs models
```

### `search` — Raw semantic search (no LLM answer)
```bash
./ollama-docs search "query terms" -n 10
```

### `export` — Export a database to JSON
```bash
./ollama-docs export -o backup.json          # active database
./ollama-docs --db music export -o b.json    # a specific database
```

## Configuration

All settings live in `src/config.py` and can be overridden by
**environment variables** or a project-local **`.env`** file (copy
`.env.example` to `.env` and edit it; `python-dotenv` loads it automatically).

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://192.168.68.59:11435` | Ollama endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `CHAT_MODEL` | `gemma3:4b` | Chat / Q&A / summarization model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `EMBEDDING_BATCH_SIZE` | `10` | Embeddings requested per batch |
| `N_RESULTS` | `8` | Default chunks retrieved per question |
| `MAX_CONTEXT_CHARS` | `24000` | Max context characters per question |
| `DOCUMENTS_PATH` | `documents/` | Folder to ingest from |
| `CHROMA_PATH` | `chroma_db/` | Vector database location |
| `COLLECTION_NAME` | `documents` | Initial/active database name on first run |
| `DB_STATE_FILE` | `.ollama-docs-state.json` | Remembers the active database |

> Defaults point at the local-network Ollama instance at
> `192.168.68.59:11435` and use already-present models
> (`nomic-embed-text` for embeddings, `gemma3:4b` for chat). Point
> `OLLAMA_HOST` elsewhere — e.g. `http://localhost:11435` — to use a local Ollama.
>
> `N_RESULTS` and `MAX_CONTEXT_CHARS` control answer thoroughness: raise them for
> more comprehensive answers, or use `--quick` for a fast one-point reply.

## Directory Structure

```
Ollama_Doc_Repo/
├── documents/                  # Put your documents here
├── chroma_db/                  # Vector store (holds every database/collection)
├── .ollama-docs-state.json     # Remembers the active database (auto-created)
├── src/
│    ├── main.py               # CLI entry point + TUI launcher (no subcommand -> TUI)
│    ├── config.py             # Configuration (env / .env overridable)
│    ├── document_processor.py # Document parsing & chunking
│    ├── vector_store.py       # ChromaDB operations + multi-database manager
│    ├── chat.py               # Q&A / summarization logic
│    └── tui.py                # Interactive Textual TUI
├── setup.py                    # Setup script
├── requirements.txt            # Python dependencies
├── .env.example                # Copy to .env to configure the endpoint/models
└── ollama-docs                 # Launcher script
```

## Requirements

- Python 3.9+ (tested on 3.14)
- Ollama reachable on the configured host (default `192.168.68.59:11435`)
- Models: `nomic-embed-text` and `gemma3:4b` (auto-pulled by `setup.py`)

## Example Workflow

```bash
# 1. Setup
cd Ollama_Doc_Repo
OLLAMA_HOST=http://192.168.68.59:11435 python setup.py

# 2. Create two databases: one for manuals, one for music articles
./ollama-docs db new manuals
./ollama-docs db new music

# 3. Add documents to the right database
./ollama-docs --db manuals ingest -i docs/car_manual.pdf
./ollama-docs --db music  ingest -i docs/great_jazz_2000s.pdf

# 4. Make one the active database (persisted), or just target it per-command
./ollama-docs db use manuals
./ollama-docs ask "What should I do if the engine overheats?"

# 5. Switch to the other database and ask about it
./ollama-docs db use music
./ollama-docs ask "Which jazz record should I buy?"

# 6. Move a document between databases without re-embedding
./ollama-docs reassign docs/shared_notes.pdf manuals --from music

# 7. Remove a single document without wiping the database
./ollama-docs remove docs/old_manual.pdf --from manuals

# 8. Export a database for backup
./ollama-docs --db manuals export -o manuals_backup.json
```

## Troubleshooting

**Ollama connection failed:**
```bash
# Make sure Ollama is running and reachable, then point the app at it:
curl http://192.168.68.59:11435/api/tags      # or your host
OLLAMA_HOST=http://192.168.68.59:11435 ./ollama-docs
```

**Model not found / wrong model:**
```bash
# The defaults match the models on the target instance. Override per-run:
CHAT_MODEL=qwen3:8b EMBEDDING_MODEL=nomic-embed-text ./ollama-docs ask "..."
# Or pull models manually:
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

**Empty answers from a "thinking" model** (qwen3, qwen3.5, lfm2): their output
is written to Ollama's `thinking` field; `chat.py` now pulls the final answer
from that field automatically. Use a non-thinking model like `gemma3:4b`
(default) to keep answers fast and predictable.

**Permission denied on ollama-docs:**
```bash
chmod +x ollama-docs
```

**Documents not found:**
```bash
# Check the documents directory
ls documents/
# Make sure files have supported extensions: .pdf .docx .txt .md .html .htm .rtf
```

## License

MIT
