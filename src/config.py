import os
from dataclasses import dataclass, field
from pathlib import Path

# Load a local .env file (if present) so the endpoint / models can be
# configured without editing this file.  Falls through silently if the file
# or the python-dotenv dependency is not available.
try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).parent.parent
    for _candidate in (_ROOT / ".env", Path.cwd() / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            break
except Exception:   # python-dotenv optional
    pass


def _env(name: str, default: str) -> str:
    """Read an environment variable, falling back to a default."""
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- Ollama endpoint -----------------------------------------------------
    # Default targets the local network instance at 192.168.68.59:11435.
    # Override per-run with OLLAMA_HOST (e.g. http://localhost:11435).
    ollama_host: str = field(default_factory=lambda: _env(
        "OLLAMA_HOST", "http://192.168.68.59:11435"))

    # Path to the project root (two levels up from this file).
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Use project-relative paths by default, fallback is the project root.
    chroma_path: str = field(default_factory=lambda: _env(
        "CHROMA_PATH", str(Path(__file__).parent.parent / "chroma_db")))
    documents_path: str = field(default_factory=lambda: _env(
        "DOCUMENTS_PATH", str(Path(__file__).parent.parent / "documents")))

    # --- Models --------------------------------------------------------------
    # Embedding model: nomic-embed-text is a pure embedding model and is present
    # on the target instance.
    embedding_model: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL", "nomic-embed-text:latest"))
    # Chat model for Q&A / summarization. Default is a non-thinking text model
    # that is actually present on the target instance. Thinking models (qwen3,
    # qwen3.5, lfm2) put their output in a separate field, which the answer
    # extractor now handles, but a non-thinking model is a safer default.
    chat_model: str = field(default_factory=lambda: _env(
        "CHAT_MODEL", "gemma3:4b"))

    # --- Chunking ------------------------------------------------------------
    chunk_size: int = int(_env("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "200"))
     # A "database" is a ChromaDB collection in the one store at chroma_path;
     # multiple databases => multiple collections that coexist, so switching or
     # adding one never wipes another. "documents" is the default active one.
    collection_name: str = _env("COLLECTION_NAME", "documents")

    # The "active" database is remembered across runs so switching it in the
    # TUI/CLI persists. State lives in this file, never in the vector store.
    state_file: str = field(default_factory=lambda: _env(
        "DB_STATE_FILE", str(Path(__file__).parent.parent / ".ollama-docs-state.json")))

    # Batch size for embedding generation.
    embedding_batch_size: int = int(_env("EMBEDDING_BATCH_SIZE", "10"))

    # How many chunks to retrieve per question, and how much of the retrieved
    # context to hand the model. Larger defaults make answers more thorough.
    # Override per-question with /results or --results in the CLI/TUI.
    n_results: int = int(_env("N_RESULTS", "8"))
    # Maximum characters of context to feed the chat model on a single question.
    max_context_chars: int = int(_env("MAX_CONTEXT_CHARS", "24000"))

    def __post_init__(self) -> None:
        # Normalize the host so trailing slashes don't break URL joins.
        self.ollama_host = self.ollama_host.rstrip("/")


config = Config()
