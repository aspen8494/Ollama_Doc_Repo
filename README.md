# Ollama Document Repository

A local document processing and Q&A application powered by Ollama. Ingest PDFs, DOCX,
Markdown, HTML, RTF, and text files for summarization, semantic search, and
source-cited question answering — and drive it all from a full-screen **interactive
terminal UI**.

## Features

- **Interactive TUI** — run `./ollama-docs` with no subcommand for a full-screen
  terminal interface: type questions, index documents, pick a document to
  summarize, switch models, and more, with a live document panel and command log.
- **Multi-format support**: PDF, DOCX, TXT, MD, HTML, RTF
- **Semantic search**: vector embeddings via ChromaDB for relevant document retrieval
- **Local processing**: everything runs against your Ollama instance — no data leaves your network
- **Summarization**: generate summaries of any document in the repository
- **Interactive Q&A**: ask questions about your document collection, with source citations
- **CLI subcommands**: every capability is also available as a one-shot CLI command
- **Export/Import**: export documents and chunks to JSON
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
| `/list` | List indexed documents with chunk counts |
| `/stats` | Database statistics |
| `/models [name]` | List Ollama models; optionally set the chat model |
| `/embedding-model <name>` | Choose the embedding model |
| `/results <n>` | Set chunks retrieved per question |
| `/export [path.json]` | Export the database to JSON |
| `/clear` then `/clear-force` | Preview, then wipe the database |
| `/help` | Show the command list |
| `/quit` | Exit |

### Keybindings

`i` ingest · `s` search · `a` ask · `l` list · `m` models · `?` help · `q` quit

## CLI Subcommands

Every TUI capability is also a one-shot command.

### `ingest` — Process documents
```bash
./ollama-docs ingest                 # ingest all docs in the folder
./ollama-docs ingest --force         # force reprocess (clears the DB first)
./ollama-docs ingest -i /path/to/document.pdf
./ollama-docs ingest -b 100          # custom embedding batch size
```

### `ask` — Query documents
```bash
./ollama-docs ask "What are the main topics?"
./ollama-docs ask -n 10 "Detailed question"     # retrieve 10 chunks
./ollama-docs ask --no-sources "Quick question" # hide citations
./ollama-docs ask -i                         # interactive prompt-based session
./ollama-docs ask -m qwen3:8b "question"      # override the chat model
```

### `summarize` — Generate summaries
```bash
./ollama-docs summarize                     # list all documents
./ollama-docs summarize documents/report.pdf
```

### `list` — List all documents with chunk counts
```bash
./ollama-docs list
```

### `stats` — Database statistics
```bash
./ollama-docs stats
```

### `clear` — Clear database
```bash
./ollama-docs clear --confirm
```

### `models` — Check Ollama models
```bash
./ollama-docs models
```

### `search` — Raw semantic search (no LLM answer)
```bash
./ollama-docs search "query terms" -n 10
```

### `export` — Export database to JSON
```bash
./ollama-docs export -o backup.json
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
| `MAX_CONTEXT_CHARS` | `12000` | Max context characters per question |
| `DOCUMENTS_PATH` | `documents/` | Folder to ingest from |
| `CHROMA_PATH` | `chroma_db/` | Vector database location |

> Defaults point at the local-network Ollama instance at
> `192.168.68.59:11435` and use already-present models
> (`nomic-embed-text` for embeddings, `gemma3:4b` for chat). Point
> `OLLAMA_HOST` elsewhere — e.g. `http://localhost:11435` — to use a local Ollama.

## Directory Structure

```
Ollama_Doc_Repo/
├── documents/              # Put your documents here
├── chroma_db/              # Vector database (auto-created)
├── src/
│   ├── main.py             # CLI entry point + TUI launcher (no subcommand -> TUI)
│   ├── config.py           # Configuration (env / .env overridable)
│   ├── document_processor.py   # Document parsing & chunking
│   ├── vector_store.py         # ChromaDB operations
│   ├── chat.py               # Q&A / summarization logic
│   └── tui.py              # Interactive Textual TUI
├── setup.py                # Setup script
├── requirements.txt        # Python dependencies
├── .env.example            # Copy to .env to configure the endpoint/models
└── ollama-docs             # Launcher script
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

# 2. Add documents
cp ~/Downloads/*.pdf documents/
cp ~/Projects/notes/*.md documents/

# 3. Launch the TUI and ingest
./ollama-docs            # then type /ingest

# 4. Ask a question right in the TUI, or from the shell
./ollama-docs ask "Summarize the key findings"

# 5. Get a document summary
./ollama-docs summarize documents/important.pdf

# 6. Search without an LLM answer
./ollama-docs search "specific term"

# 7. Export for backup
./ollama-docs export -o backup.json
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
