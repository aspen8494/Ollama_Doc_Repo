import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import ollama
from config import config


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=config.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.ollama_client = ollama.Client(host=config.ollama_host)

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (individual requests for compatibility)."""
        embeddings = []
        for text in texts:
            response = self.ollama_client.embeddings(
                model=config.embedding_model,
                prompt=text
            )
            embeddings.append(response['embedding'])
        return embeddings

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama (single text)."""
        response = self.ollama_client.embeddings(
            model=config.embedding_model,
            prompt=text
        )
        return response['embedding']

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        if not documents:
            return

        texts = [doc['text'] for doc in documents]
        metadatas = [
            {
                'source': doc['source'],
                'filename': doc['filename'],
                'chunk_index': doc['chunk_index'],
                'total_chunks': doc['total_chunks']
            }
            for doc in documents
        ]
        ids = [f"{doc['source']}#{doc['chunk_index']}" for doc in documents]

        # Generate embeddings in batch
        embeddings = self._get_embeddings_batch(texts)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        query_embedding = self._get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        if not results['documents'] or not results['documents'][0]:
            return []

        return [
            {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if results['distances'] else None
            }
            for i in range(len(results['documents'][0]))
        ]

    def get_all_sources(self) -> List[str]:
        """Get list of all unique source documents."""
        results = self.collection.get()
        if not results['metadatas']:
            return []
        sources = set(m['source'] for m in results['metadatas'])
        return sorted(sources)

    def delete_source(self, source_path: str) -> None:
        """Delete all chunks from a specific source document."""
        results = self.collection.get(where={"source": source_path})
        if results['ids']:
            self.collection.delete(ids=results['ids'])

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        results = self.collection.get()
        total_chunks = len(results['ids']) if results['ids'] else 0
        sources = set()
        for m in results['metadatas'] or []:
            sources.add(m['source'])
        return {
            'total_chunks': total_chunks,
            'unique_sources': len(sources),
            'sources': sorted(sources)
        }

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


def main():
    store = VectorStore()
    stats = store.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique sources: {stats['unique_sources']}")
    for s in stats['sources']:
        print(f"  - {s}")


if __name__ == '__main__':
    main()