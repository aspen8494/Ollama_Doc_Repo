import os
from dataclasses import dataclass

@dataclass
class Config:
    ollama_host: str = "http://localhost:11435"
    chroma_path: str = os.path.expanduser("~/Documents/Ollama_Doc_Repo/chroma_db")
    documents_path: str = os.path.expanduser("~/Documents/Ollama_Doc_Repo/documents")
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "documents"

config = Config()