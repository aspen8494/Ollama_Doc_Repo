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
        """Get embeddings for multiple texts."""
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

        meta_keys = ('source', 'source_path', 'filename', 'chunk_index', 'total_chunks')

        # Generate embeddings once for every document.
        raw = self._get_embeddings_batch([doc['text'] for doc in documents])

        seen = set()
        texts, metadatas, ids, embeddings = [], [], [], []
        for doc, emb in zip(documents, raw):
            id_ = f"{doc['source']}#{doc['chunk_index']}"
            if id_ in seen:
                continue
            seen.add(id_)
            texts.append(doc['text'])
            metadatas.append({k: doc.get(k) for k in meta_keys if doc.get(k) is not None})
            ids.append(id_)
            embeddings.append(emb)

        if not ids:
            return

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def search(self, query: str, n_results: int = 5,
            where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search relevant chunks, optionally constrained to a source via ``where``."""
        query_embedding = self._get_embedding(query)

        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

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

    def get_source_counts(self) -> Dict[str, int]:
        """Return a mapping of source key -> chunk count."""
        results = self.collection.get()
        if not results['metadatas']:
            return {}
        counts: Dict[str, int] = {}
        for m in results['metadatas']:
            counts[m['source']] = counts.get(m['source'], 0) + 1
        return counts

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
        print(f"       -> {s}")


if __name__ == '__main__':
    main()
