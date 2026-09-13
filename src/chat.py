import ollama
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from config import config

class DocumentChat:
    def __init__(self):
        self.ollama_client = ollama.Client(host=config.ollama_host)
        self.vector_store = VectorStore()
    
    def ask(self, question: str, n_results: int = 5, show_sources: bool = True) -> Dict[str, Any]:
        """Ask a question about the documents."""
        # Search for relevant chunks
        results = self.vector_store.search(question, n_results=n_results)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'context_used': False
            }
        
        # Build context from results
        context_parts = []
        sources = []
        for r in results:
            context_parts.append(f"[Source: {r['metadata']['filename']}]\n{r['text']}")
            sources.append({
                'filename': r['metadata']['filename'],
                'source': r['metadata']['source'],
                'chunk_index': r['metadata']['chunk_index'],
                'distance': r['distance']
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using Ollama
        system_prompt = """You are a helpful assistant that answers questions based on the provided document context. 
Only use information from the context to answer. If the context doesn't contain enough information, say so.
Be concise but thorough. Cite sources by referencing the filename in your answer."""
        
        user_prompt = f"""Context from documents:
{context}

Question: {question}

Answer based only on the context above:"""
        
        response = self.ollama_client.chat(
            model=config.chat_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.3}
        )
        
        answer = response['message']['content']
        
        return {
            'answer': answer,
            'sources': sources,
            'context_used': True
        }
    
    def summarize_document(self, source_path: str) -> Dict[str, Any]:
        """Generate a summary of a specific document."""
        results = self.vector_store.collection.get(where={"source": source_path})
        
        if not results['documents']:
            return {
                'summary': "Document not found in the database.",
                'source': source_path
            }
        
        # Combine all chunks
        full_text = "\n\n".join(results['documents'])
        
        # Truncate if too long (leave room for prompt)
        max_context = 8000
        if len(full_text) > max_context:
            full_text = full_text[:max_context] + "... [truncated]"
        
        response = self.ollama_client.chat(
            model=config.chat_model,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant that creates concise summaries of documents.'},
                {'role': 'user', 'content': f'Summarize this document:\n\n{full_text}'}
            ],
            options={'temperature': 0.3}
        )
        
        return {
            'summary': response['message']['content'],
            'source': source_path
        }
    
    def list_documents(self) -> List[str]:
        """List all documents in the database."""
        return self.vector_store.get_all_sources()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.vector_store.get_stats()

def main():
    chat = DocumentChat()
    stats = chat.get_stats()
    print(f"Database: {stats['total_chunks']} chunks from {stats['unique_sources']} documents")
    
    while True:
        question = input("\nAsk a question (or 'quit'): ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        result = chat.ask(question)
        print(f"\nAnswer: {result['answer']}")
        if result['sources']:
            print("\nSources:")
            for s in result['sources']:
                print(f"  - {s['filename']} (chunk {s['chunk_index']}, distance: {s['distance']:.4f})")

if __name__ == '__main__':
    main()