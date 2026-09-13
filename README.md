# Ollama Document Repository

A local document processing and Q&A application powered by Ollama. Process PDFs, DOCX, Markdown, HTML, RTF, and text files for summarization and semantic search.

## Features

- **Multi-format support**: PDF, DOCX, TXT, MD, HTML, RTF
- **Semantic search**: Vector embeddings via ChromaDB for relevant document retrieval
- **Local processing**: Everything runs locally with Ollama - no data leaves your machine
- **Summarization**: Generate summaries of any document in the repository
- **Interactive Q&A**: Ask questions about your document collection
- **CLI interface**: Simple command-line usage with Rich terminal UI
- **Export/Import**: Export documents and chunks to JSON
- **Raw search**: Search documents without LLM answer generation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aspen8494/Ollama_Doc_Repo.git
cd Ollama_Doc_Repo

# Run setup (installs dependencies, pulls models)
python setup.py

# Add documents to the documents folder
cp /path/to/your/docs/* documents/

# Ingest documents into the database
./ollama-docs ingest

# Ask questions
./ollama-docs ask "What is this document about?"

# Or start interactive session
./ollama-docs ask -i
```

## Commands

### `ingest` - Process documents
```bash
# Ingest all documents in the documents folder
./ollama-docs ingest

# Force reprocess all documents (clears database first)
./ollama-docs ingest --force

# Ingest a specific file
./ollama-docs ingest -i /path/to/document.pdf

# Custom batch size for embeddings
./ollama-docs ingest -b 100
```

### `ask` - Query documents
```bash
# Single question
./ollama-docs ask "What are the main topics?"

# With more search results
./ollama-docs ask -n 10 "Detailed question"

# Hide source citations
./ollama-docs ask --no-sources "Quick question"

# Interactive mode
./ollama-docs ask -i
```

### `summarize` - Generate summaries
```bash
# List all documents
./ollama-docs summarize

# Summarize a specific document
./ollama-docs summarize documents/report.pdf
```

### `list` - List all documents with chunk counts
```bash
./ollama-docs list
```

### `stats` - Database statistics
```bash
./ollama-docs stats
```

### `clear` - Clear database
```bash
./ollama-docs clear --confirm
```

### `models` - Check Ollama models
```bash
./ollama-docs models
```

### `search` - Raw semantic search (no LLM answer)
```bash
./ollama-docs search "query terms" -n 10
```

### `export` - Export database to JSON
```bash
./ollama-docs export -o backup.json
```

## Configuration

Edit `src/config.py` to customize:
- `ollama_host`: Ollama endpoint (default: http://localhost:11435)
- `embedding_model`: Model for embeddings (default: nomic-embed-text)
- `chat_model`: Model for Q&A (default: llama3.2)
- `chunk_size`: Text chunk size (default: 1000)
- `chunk_overlap`: Chunk overlap (default: 200)
- `embedding_batch_size`: Batch size for embedding requests (default: 10)

Paths are relative to the project root by default:
- `chroma_path`: ./chroma_db
- `documents_path`: ./documents

## Directory Structure

```
Ollama_Doc_Repo/
├── documents/          # Put your documents here
├── chroma_db/          # Vector database (auto-created)
├── src/
│   ├── main.py         # CLI entry point
│   ├── config.py       # Configuration
│   ├── document_processor.py  # Document parsing
│   ├── vector_store.py        # ChromaDB operations
│   └── chat.py       # Q&A logic
├── setup.py            # Setup script
├── requirements.txt    # Python dependencies
└── ollama-docs         # Launcher script
```

## Requirements

- Python 3.9+
- Ollama running on port 11435
- Models: `nomic-embed-text` and `llama3.2` (auto-pulled by setup)

## Example Workflow

```bash
# 1. Setup
cd Ollama_Doc_Repo
python setup.py

# 2. Add documents
cp ~/Downloads/*.pdf documents/
cp ~/Projects/notes/*.md documents/

# 3. Ingest
./ollama-docs ingest

# 4. Query
./ollama-docs ask "Summarize the key findings"
./ollama-docs ask -i  # Interactive mode

# 5. Get document summaries
./ollama-docs summarize
./ollama-docs summarize documents/important.pdf

# 6. Search without LLM
./ollama-docs search "specific term"

# 7. Export for backup
./ollama-docs export -o backup.json
```

## Troubleshooting

**Ollama connection failed:**
```bash
# Make sure Ollama is running
ollama serve

# Check it's on port 11435
curl http://localhost:11435/api/tags
```

**Model not found:**
```bash
# Pull required models manually
ollama pull nomic-embed-text
ollama pull llama3.2
```

**Permission denied on ollama-docs:**
```bash
chmod +x ollama-docs
```

**Documents not found:**
```bash
# Check documents directory
ls documents/
# Make sure files have supported extensions: .pdf, .docx, .txt, .md, .html, .htm, .rtf
```

## License

MIT