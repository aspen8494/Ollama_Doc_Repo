import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    ollama_host: str = "http://localhost:11435"
    # Use project-relative paths by default, fallback to ~/Documents
    chroma_path: str = str(Path(__file__).parent.parent / "chroma_db")
    documents_path: str = str(Path(__file__).parent.parent / "documents")
    embedding_model: str = "nomic-embed-text:latest"
    chat_model: str = "moondream:latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "documents"
    # Batch size for embedding generation
    embedding_batch_size: int = 10


config = Config()