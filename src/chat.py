import ollama
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from document_processor import DocumentProcessor
from config import config


def extract_answer(response: Dict[str, Any]) -> str:
    """Pull the answer out of an Ollama chat response.

    Some models (e.g. qwen3, qwen3.5, lfm2) are "thinking" models: they stream
    their reasoning into a separate `thinking` field and may leave
    `message.content` empty. We prefer `content`, then fall back to the final
    answer from the `thinking` field, so a thinking model is handled automatically.
    """
    message = response.get('message') or {}
    content = (message.get('content') or '').strip()
    if content:
        return content

    thinking = message.get('thinking') or response.get('thinking') or ''
    if not thinking:
        return content

    candidates = [ln.strip() for ln in thinking.splitlines() if ln.strip()]
    for ln in candidates:
        low = ln.lower()
        if low.startswith(('answer', 'conclusion', 'final')):
            if ':' in ln:
                return ln.split(':', 1)[-1].strip()
            return ln
    return candidates[-1] if candidates else ''


class DocumentChat:
    def __init__(self, model: Optional[str] = None, n_results: Optional[int] = None):
        self.ollama_client = ollama.Client(host=config.ollama_host)
        self.vector_store = VectorStore()
        # Per-instance overrides so --model / -n actually take effect.
        self.model = model or config.chat_model
        self.n_results = n_results or 5

    def set_model(self, model: str) -> None:
        self.model = model

    def _generate(self, system_prompt: str, user_prompt: str,
                temperature: float = 0.3, think: Optional[bool] = None) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}],
            options={'temperature': temperature})
        if think is not None:
            # Let thinking models use their thinking field (or disable it).
            kwargs['options']['think'] = think
        response = self.ollama_client.chat(**kwargs)
        return extract_answer(response)

    def ask(self, question: str, n_results: Optional[int] = None,
            show_sources: bool = True) -> Dict[str, Any]:
        """Ask a question about the documents."""
        n = n_results or self.n_results
        results = self.vector_store.search(question, n_results=n)

        if not results:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'context_used': False}

        context_parts = []
        sources = []
        for r in results:
            context_parts.append(f"[Source: {r['metadata']['filename']}]\n{r['text']}")
            sources.append({
                'filename': r['metadata']['filename'],
                'source': r['metadata']['source'],
                'chunk_index': r['metadata']['chunk_index'],
                'distance': r['distance']})

        context = "\n\n---\n\n".join(context_parts)

        # Cap context to avoid blowing the model's context window.
        if len(context) > config.max_context_chars:
            context = context[:config.max_context_chars] + "\n... [context truncated]"

        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided "
            "document context. Only use information from the context to answer. If the "
            "context doesn't contain enough information, say so. Be concise but thorough. "
            "Cite sources by referencing the filename in your answer.")

        user_prompt = (f"Context from documents:\n{context}\n\n"
                    f"Question: {question}\n\nAnswer based only on the context above:")

        answer = self._generate(system_prompt, user_prompt)
        if not answer:
            answer = ("I couldn't generate an answer from the retrieved context. "
                    "Try a more specific question.")

        return {'answer': answer, 'sources': sources, 'context_used': True}

    def summarize_document(self, source_path: str) -> Dict[str, Any]:
        """Generate a summary of a specific document.

        source_path may be a relative path, a basename, or the stored 'source'
        key; it is resolved to the stored key first.
        """
        key = DocumentProcessor.match_source(source_path)
        results = self.vector_store.collection.get(where={"source": key})

        if not results.get('documents'):
            # Fall back to the raw query (in case it was already a key).
            results = self.vector_store.collection.get(where={"source": source_path})
        if not results.get('documents'):
            return {'summary': "Document not found in the database.",
                    'source': source_path}

        docs = results['documents'] or []
        metas = results['metadatas'] or []
        paired = list(zip(metas, docs))
        paired.sort(key=lambda pair: (pair[0] or {}).get('chunk_index', 0))

        full_text = "\n\n".join(text for _, text in paired if text)

        # Truncate if too long (leave room for prompt).
        if len(full_text) > config.max_context_chars:
            full_text = full_text[:config.max_context_chars] + " ... [truncated]"

        summary = self._generate(
            "You are a helpful assistant that creates concise summaries of documents. "
            "Provide a clear, structured summary covering the main points.",
            f'Summarize this document:\n\n{full_text}')
        return {'summary': summary, 'source': key}

    def list_documents(self) -> List[str]:
        """List all documents in the database."""
        return self.vector_store.get_all_sources()

    def list_model_names(self) -> List[str]:
        """List available Ollama model names on the configured endpoint."""
        resp = self.ollama_client.list()
        models = getattr(resp, 'models', None)
        if models is None and isinstance(resp, dict):
            models = resp.get('models', [])
        names: List[str] = []
        for m in models or []:
            if isinstance(m, dict):
                names.append(m.get('model') or m.get('name', ''))
            else:
                names.append(getattr(m, 'model', None) or getattr(m, 'name', ''))
        return [n for n in names if n]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.vector_store.get_stats()


def main():
    chat = DocumentChat()
    stats = chat.get_stats()
    print(f"Database: {stats['total_chunks']} chunks from {stats['unique_sources']} documents")

    while True:
        try:
            question = input("\nAsk a question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue

        result = chat.ask(question)
        print(f"\nAnswer: {result['answer']}")
        if result['sources']:
            print("\nSources:")
            for s in result['sources']:
                print(f"       - {s['filename']} (chunk {s['chunk_index']}, distant: {s['distance']:.4f})")


if __name__ == '__main__':
    main()
