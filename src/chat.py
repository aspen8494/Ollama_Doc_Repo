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


# Thoroughness tiers control how aggressively we retrieve and how the prompt asks
# for a complete answer. "quick" matches an old short-answer behaviour; "thorough"
# is the default and retrieves more, asks for structure, and uses all the source
# material the retrieval budget can carry.
THOROUGH_SYSTEM = (
    "You are a thorough research assistant. Answer the question COMPLETELY using "
    "the provided document context. Cover every relevant point, not just the "
    "first one. Organize the answer into clear sections or bullet points. "
    "Draw on all of the supplied context, not only the most similar chunk. If "
    "different documents add different facts, combine them. Only use "
    "information from the context. If the context is missing something, say so "
    "explicitly. When you cite a document, give enough detail to be useful."
)
QUICK_SYSTEM = (
    "You are a helpful assistant that answers a question from the provided "
    "document context. Answer concisely. Only use information from the "
    "context. If the context does not contain enough information, say so."
)


class DocumentChat:
    """Q&A / summarization over one database (ChromaDB collection).

    Pass ``collection_name`` to target a specific database; otherwise the active
    database (``config.collection_name``) is used, so switching databases in the
    TUI/CLI redirects Q&A to the right one.
    """

    def __init__(self, model: Optional[str] = None,
                n_results: Optional[int] = None,
                collection_name: Optional[str] = None,
                thorough: bool = True):
        self.ollama_client = ollama.Client(host=config.ollama_host)
        self.vector_store = VectorStore(collection_name=collection_name)
        self.collection_name = self.vector_store.collection_name
        # Per-instance overrides. Default n_results now comes from config so the
        # retrieval depth is generous (answers are thorough, not one-liners).
        self.model = model or config.chat_model
        self.n_results = n_results or config.n_results or 5
        self.thorough = thorough

    def set_model(self, model: str) -> None:
        self.model = model

    # ------------------------------------------------------------- internal
    def _generate(self, system_prompt: str, user_prompt: str,
                temperature: float = 0.3, think: Optional[bool] = None,
                num_predict: Optional[int] = None) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}],
            options={'temperature': temperature},
        )
        if think is not None:
            kwargs['options']['think'] = think
        if num_predict:
            # Let the model use more output tokens so it can write a full answer.
            kwargs['options']['num_predict'] = 2048
        response = self.ollama_client.chat(**kwargs)
        return extract_answer(response)

    def _build_context(self, results: List[Dict[str, Any]]):
        """Assemble context from retrieved chunks and a per-source record list.

        Keeps every chunk within the char budget but in retrieval order, and
        returns which documents contributed material (for citations and for
        telling the model how many sources it should cover).
        """
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        total = 0

        for r in results:
            meta = r.get('metadata') or {}
            text = (r.get('text') or '').strip()
            if not text:
                continue
            fname = meta.get('filename') or meta.get('source') or '?'
            cidx = meta.get('chunk_index')
            tag = '[Source: ' + str(fname) + ' chunk ' + str(cidx) + ']\n'
            chunk = tag + text
            if total + len(chunk) > config.max_context_chars and parts:
                # Budget exhausted: stop adding, but keep what we have.
                break
            total += len(chunk)
            parts.append(chunk)
            sources.append({
                'filename': fname,
                'source': meta.get('source'),
                'chunk_index': meta.get('chunk_index'),
                'distance': r.get('distance'),
                'db': self.collection_name,
                })
        context = "\n\n---\n\n".join(parts)
        return context, sources

    # ------------------------------------------------------------- public API
    def ask(self, question: str, n_results: Optional[int] = None,
            show_sources: bool = True, thorough: Optional[bool] = None) -> Dict[str, Any]:
        """Ask a question about the documents in this database."""
        n = n_results or self.n_results
        thorough_mode = thorough if thorough is not None else self.thorough

        results = self.vector_store.search(question, n_results=n)

        if not results:
            return {
                'answer': ("I couldn't find any relevant information in the "
                            f"database '{self.collection_name}' to answer your "
                            "question. Try a different question, or check that "
                            "documents are indexed (run /ingest)."),
                'sources': [],
                'context_used': False,
                'truncated': False,
                'db': self.collection_name,
            }

        context, sources = self._build_context(results)

        n_docs = len({(s['source'] or s['filename']) for s in sources})
        system_prompt = THOROUGH_SYSTEM if thorough_mode else QUICK_SYSTEM

        head = (
            f"Context from {n_docs} document(s) in database "
            f"'{self.collection_name}' "
            f"({len(sources)} chunk(s)):")
        user_prompt = (
            f"{head}\n\n{context}\n\n"
            f"Question: {question}\n\n"
            + ("Answer completely, using material from all of the source "
                "documents above where relevant:\n" if thorough_mode else
                "Answer based on the context above:\n"))

        answer = self._generate(system_prompt, user_prompt)
        if not answer:
            answer = ("I couldn't generate an answer from the retrieved context. "
                        "Try a more specific question, or increase the number of "
                        "retrieved chunks.")

        truncated = len(context) >= config.max_context_chars
        return {
            'answer': answer,
            'sources': sources,
            'context_used': True,
            'truncated': truncated,
            'db': self.collection_name,
            'n_chunks': len(sources),
            'n_docs': n_docs,
            'thorough': thorough_mode,
        }

    def summarize_document(self, source_path: str,
                            collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a thorough summary of one document in one database.

        ``source_path`` may be a relative path, a basename, or the stored
        'source' key; it is resolved to the stored key first.
        """
        col = collection_name or self.collection_name
        key = DocumentProcessor.match_source(source_path)
        col_obj = self.vector_store._col(col)
        results = col_obj.get(where={"source": key})
        if not results.get('documents'):
            results = col_obj.get(where={"source": source_path})
        if not results.get('documents'):
            return {'summary': f"Document not found in database '{col}'.",
                    'source': source_path, 'db': col}

        docs = results['documents'] or []
        metas = results['metadatas'] or []
        paired = list(zip(metas, docs))
        paired.sort(key=lambda pair: (pair[0] or {}).get('chunk_index', 0))

        full_text = "\n\n".join(text for _, text in paired if text)

        truncated = False
        if len(full_text) > config.max_context_chars:
            full_text = full_text[:config.max_context_chars] + " ... [truncated]"
            truncated = True

        summary = self._generate(
            "You are a helpful assistant that creates thorough, well-structured "
            "summaries. Cover all the main points and important details; use "
            "headings or bullet points when helpful.",
            f'Summarize this document completely:\n\n{full_text}')
        return {'summary': summary, 'source': key, 'db': col,
                'truncated': truncated}

    def list_documents(self, collection_name: Optional[str] = None) -> List[str]:
        return self.vector_store.get_all_sources()

    def list_model_names(self) -> List[str]:
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
        return self.vector_store.get_stats()


def main():
    chat = DocumentChat()
    stats = chat.get_stats()
    print(f"Database '{stats['collection']}': "
        f"{stats['total_chunks']} chunks from "
        f"{stats['unique_sources']} documents")

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
                print(f"          - {s['filename']} "
                        f"({s.get('db', '')}) chunk {s['chunk_index']} "
                        f"(distance: {s['distance']:.4f})")


if __name__ == '__main__':
    main()
