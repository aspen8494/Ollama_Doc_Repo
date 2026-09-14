"""Textual-based terminal UI for the Ollama Document Repository.

Run with no subcommand:    ./ollama-docs      (or  python -m src.main)

Type a natural-language question and press Enter to ask the document
collection, or prefix a line with "/" to run a command:

    /help                 list all commands
    /ingest [force]       index every file in the documents folder
    /ingest-file <path>   index one specific file
    /search <query>       raw semantic search (no LLM answer)
    /ask <question>       Q&A with an LLM answer + citations
    /summarize <doc>      summarize a document
    /list                 list indexed documents + chunk counts
    /stats                database statistics
    /models [name]        list Ollama models / choose the chat model
    /embedding-model n    choose the embedding model
    /results <n>          default chunks retrieved per question
    /export [path.json]   export the database to JSON
    /clear-force          wipe the database
    /quit                 exit

Global keys:  i ingest,  s search,  a ask,  l list,  m models,  ? help,  q quit
"""

import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
from textual.containers import Horizontal, Vertical
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.worker import WorkerState

from config import config
from document_processor import DocumentProcessor
from chat import DocumentChat


class DocApp(App):
    BINDINGS = [
        Binding("i", "ingest", "Ingest", show=False),
        Binding("s", "search", "Search", show=False),
        Binding("a", "ask", "Ask", show=False),
        Binding("l", "list", "List", show=False),
        Binding("m", "models", "Models", show=False),
        Binding("?", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
        Screen { layout: vertical; }
        #docs-panel { width: 44; dock: left; border: round $primary; }
        #docs { height: 1fr; }
        #log { height: 1fr; }
        #cmd { dock: bottom; border: round $secondary; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.chat: DocumentChat = None
        self.model = config.chat_model
        self.n_results = 5
        self.sources = []
        self._last_question = "?"
        self._kinds = {}

    # -------------------------------------------------------------- compose
    def compose(self):
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="docs-panel"):
                yield Static("INDEXED DOCUMENTS\n(Enter -> summarize)", id="docs-title")
                yield OptionList(id="docs")
            yield RichLog(max_lines=1000, wrap=True, markup=True, auto_scroll=True, id="log")

        yield Input(placeholder="Ask a question, or type /help", id="cmd")
        yield Footer()

    def on_mount(self):
        self.chat = DocumentChat(model=self.model, n_results=self.n_results)
        self._write_welcome()
        self._start("status", self._task_status)
        self._start("refresh_docs", self._task_refresh_docs)
        self.query_one("#cmd").focus()

    # ------------------------------------------------------------- helpers
    def _log(self, renderable, **kwargs):
        # Rich objects passed to RichLog.write are rendered directly.
        self.query_one("#log").write(renderable)

    def _write_welcome(self):
        self._log(Text("Ollama Document Repository - TUI", style="bold cyan"))
        self._log(Text(
            f"Ollama {config.ollama_host}    chat {self.model}    "
            f"embed {config.embedding_model}", style="dim"))
        self._log(Text(f"Documents: {config.documents_path}", style="dim"))
        self._log(Text("Type a question, or /help for all commands.", style="dim"))

    def _start(self, kind, func, *args):
        """Run blocking work in a thread worker so the UI never freezes.
    v8's run_worker takes no args, so we bind the task + its arguments via
    functools.partial. Per-kind grouping keeps each operation exclusive."""
        import functools
        bound = functools.partial(func, *args)
        w = self.run_worker(bound, name=kind, group=kind, thread=True,
                            exclusive=True)
        self._kinds[id(w)] = kind

    def on_worker_state_changed(self, event):
        w = event.worker
        kind = self._kinds.get(id(w))
        if kind is None:
            return
        if w.state == WorkerState.SUCCESS:
            try:
                self._render(kind, w.result)
            except Exception as exc:
                self._log(Text(f"Render error ({kind}): {exc}", style="red"))
        elif w.state == WorkerState.ERROR:
            err = getattr(w, "error", "unknown error")
            self._log(Text(f"  {kind} failed: {err}", style="red"))

    # --------------------------------------------------------- blocking tasks
    def _task_status(self):
        import urllib.request
        connected = False
        try:
            r = urllib.request.urlopen(f"{config.ollama_host}/api/tags", timeout=5)
            connected = r.status == 200
        except Exception:
            connected = False
        stats = self.chat.get_stats()
        return {"connected": connected,
                "docs": stats["unique_sources"],
                "chunks": stats["total_chunks"]}

    def _task_refresh_docs(self):
        counts = self.chat.vector_store.get_source_counts()
        return {"sources": self.chat.list_documents(), "counts": counts}

    def _task_ask(self, question):
        return self.chat.ask(question, n_results=self.n_results)

    def _task_search(self, query, n):
        return self.chat.vector_store.search(query, n_results=n)

    def _task_summarize(self, source):
        return self.chat.summarize_document(source)

    def _task_ingest(self, force, single_file=None):
        from vector_store import VectorStore
        processor = DocumentProcessor()
        vs = VectorStore()
        files = [Path(single_file)] if single_file else processor.scan_documents()
        if force:
            vs.clear()
        out = {"files": 0, "chunks": 0, "skipped": [], "errors": []}
        for fp in files:
            key = DocumentProcessor.source_key(fp)
            if not force and vs.collection.get(where={"source": key}).get("ids"):
                out["skipped"].append(fp.name)
                continue
            try:
                chunks = processor.process_file(fp)
                if chunks:
                    vs.add_documents(chunks)
                    out["files"] += 1
                    out["chunks"] += len(chunks)
            except Exception as exc:
                out["errors"].append(f"{fp.name}: {exc}")
        return out

    def _task_list(self):
        counts = self.chat.vector_store.get_source_counts()
        return {"sources": self.chat.list_documents(), "counts": counts}

    def _task_stats(self):
        return self.chat.get_stats()

    def _task_models(self):
        return self.chat.list_model_names()

    def _task_clear(self):
        self.chat.vector_store.clear()
        return True

    def _task_export(self, path):
        vs = self.chat.vector_store
        results = vs.collection.get()
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        ids = results.get("ids") or []
        grouped = defaultdict(list)
        first_meta = {}
        for doc, meta, id_ in zip(docs, metas, ids):
            src = meta.get("source", "?")
            grouped[src].append({"id": id_, "text": doc, "metadata": meta})
            first_meta.setdefault(src, meta)
        payload = {
            "exported_at": datetime.now().isoformat(),
            "statistics": vs.get_stats(),
            "documents": [
                {"source": s,
                "filename": first_meta.get(s, {}).get("filename", s),
                "chunks": grouped[s]}
                for s in sorted(grouped)
            ],
        }
        out = Path(path) if path else Path("ollama-docs-export.json")
        out.write_text(__import__("json").dumps(payload, indent=2))
        return {"path": str(out), "docs": len(grouped),
                "chunks": vs.get_stats()["total_chunks"]}

    # ----------------------------------------------------------- rendering
    def _render(self, kind, result):
        if kind == "status":
            self._render_status(result)
        elif kind == "refresh_docs":
            self._render_docs(result)
        elif kind == "ask":
            self._render_answer(result)
        elif kind == "search":
            self._render_search(result)
        elif kind == "summarize":
            self._log(Panel(Text(result.get("summary", "").strip() or "(empty)"),
                            title="Summary", border_style="blue"))
        elif kind == "ingest":
            self._render_ingest(result)
        elif kind == "list":
            self._render_docs(result)
        elif kind == "stats":
            self._render_stats(result)
        elif kind == "models":
            self._render_models(result)
        elif kind == "clear":
            self._log(Text("Database cleared.", style="green"))
            self._start("refresh_docs", self._task_refresh_docs)
        elif kind == "export":
            self._log(Text(
                f"Exported {result['docs']} doc(s), "
                f"{result['chunks']} chunks to {result['path']}", style="green"))

    def _render_status(self, r):
        ok = r["connected"]
        self._log(Panel(
            Text.assemble(
                Text("OK" if ok else "UNREACHABLE",
                    style="bold green" if ok else "bold red"),
                Text(f"   {config.ollama_host}\n", style="dim"),
                Text(f"{r['docs']} document(s), {r['chunks']} chunk(s)",
                    style="dim")),
            title="Status",
            border_style="cyan" if ok else "red"))
        self._start("refresh_docs", self._task_refresh_docs)

    def _render_docs(self, result):
        sources = result.get("sources", [])
        counts = result.get("counts", {})
        self.sources = sources
        opts = self.query_one("#docs")
        opts.clear_options()
        if not sources:
            self._log(Text(
                f"No documents yet - run /ingest to index "
                f"the folder {config.documents_path}.", style="dim"))
            return
        for s in sources:
            opts.add_option(f"{s}  ({counts.get(s, '?')} chunks)")

    def _render_answer(self, result):
        self._log(Text(f"Q: {self._last_question}", style="bold cyan"))
        self._log(Panel(
            Text(result.get("answer", "").strip() or "(no answer)"),
            title="Answer", border_style="green"))
        for s in result.get("sources", []):
            d = s.get("distance")
            rel = f"{1 - d:.1%}" if d is not None else "n/a"
            self._log(Text(
                f"    {s['filename']}  chunk {s['chunk_index']}   "
                f"(relevance {rel})", style="dim"))

    def _render_search(self, results):
        if not results:
            self._log(Text("No matching chunks found.", style="yellow"))
            return
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="right", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Rel.", justify="right")
        table.add_column("Chunk")
        for i, r in enumerate(results, 1):
            d = r["distance"]
            rel = f"{1 - d:.1%}" if d is not None else "n/a"
            m = r["metadata"]
            table.add_row(str(i), m["filename"], rel, str(m["chunk_index"]))
        self._log(Panel(table, title="Search", border_style="cyan"))

    def _render_ingest(self, result):
        self._log(Text(
            f"Ingested {result['files']} file(s), {result['chunks']} chunks.",
            style="green"))
        if result["skipped"]:
            self._log(Text("Skipped (already indexed): "
                            + ", ".join(result["skipped"]), style="dim"))
        for err in result["errors"]:
            self._log(Text(f"  error: {err}", style="red"))
        self._start("refresh_docs", self._task_refresh_docs)

    def _render_stats(self, s):
        body = Text.assemble(
            "Total chunks: ", Text(str(s["total_chunks"]), "bold"), "\n",
            "Unique documents: ", Text(str(s["unique_sources"]), "bold"), "\n",
            "Embedding model: ", Text(config.embedding_model, "dim"), "\n",
            "Chat model: ", Text(self.model, "dim"), "\n",
            "Chunk / overlap: ",
            Text(f"{config.chunk_size} / {config.chunk_overlap}", "dim"), "\n",
            "Database: ", Text(str(config.chroma_path), "dim"))
        self._log(Panel(body, title="Database Statistics", border_style="blue"))

    def _render_models(self, names):
        self._log(Text(f"{len(names)} Ollama model(s):", style="bold"))
        for n in sorted(names):
            tag = ""
            if n == self.model:
                tag += " [CHAT]"
            if n == config.embedding_model:
                tag += " [EMBED]"
            style = "cyan bold" if tag else "cyan"
            self._log(Text(f"    {n}{tag}", style=style))

    # ----------------------------------------------------- input / actions
    def on_option_list_option_selected(self, event):
        if event.control.id != "docs":
            return
        idx = event.option_index
        if 0 <= idx < len(self.sources):
            self._last_question = self.sources[idx]
            self._log(Text(f"Summarizing {self.sources[idx]} ...", style="dim"))
            self._start("summarize", self._task_summarize, self.sources[idx])

    def on_input_submitted(self, event):
        raw = (event.value or "").strip()
        if not raw:
            return
        event.input.value = ""
        low = raw.lower()
        if low in ("/quit", "/exit", "/q"):
            self.exit()
            return
        if raw.startswith("/"):
            self._dispatch(raw[1:])
        else:
            self._last_question = raw
            self._log(Text("Searching & generating ...", style="dim"))
            self._start("ask", self._task_ask, raw)

    def _dispatch(self, args):
        parts = args.split()
        cmd = parts[0].lower() if parts else ""
        rest = args[len(parts[0]):].strip() if parts else ""
        if cmd in ("help", "?", "h"):
            self._show_help()
        elif cmd in ("ingest", "i"):
            force = "force" in rest.lower()
            self._log(Text("Ingesting documents ...", style="dim"))
            self._start("ingest", self._task_ingest, force, None)
        elif cmd in ("ingest-file", "ingestfile"):
            if not rest:
                self._log(Text("Usage: /ingest-file <path>", style="yellow"))
                return
            self._log(Text("Ingesting file ...", style="dim"))
            self._start("ingest", self._task_ingest, False, rest)
        elif cmd in ("search", "s"):
            n = self._extract_n(rest)
            q = (rest.split() if n is not None else [rest])[0] if rest else ""
            if not q:
                self._log(Text("Usage: /search <query>", style="yellow"))
                return
            self._log(Text("Searching ...", style="dim"))
            self._start("search", self._task_search, q, n)
        elif cmd in ("ask", "a"):
            if not rest:
                self._log(Text("Usage: /ask <question>", style="yellow"))
                return
            self._last_question = rest
            self._log(Text("Searching & generating ...", style="dim"))
            self._start("ask", self._task_ask, rest)
        elif cmd in ("summarize", "summary"):
            if not rest:
                self._log(Text("Usage: /summarize <document>", style="yellow"))
                return
            self._last_question = rest
            self._log(Text("Summarizing ...", style="dim"))
            self._start("summarize", self._task_summarize, rest)
        elif cmd in ("list", "l"):
            self._start("list", self._task_list)
        elif cmd == "stats":
            self._start("stats", self._task_stats)
        elif cmd in ("models", "m"):
            if rest:
                self.model = rest
                self.chat.set_model(rest)
                self._log(Text(f"Chat model set to {rest}.", style="green"))
            else:
                self._start("models", self._task_models)
        elif cmd in ("embedding-model", "embedding", "embed"):
            if not rest:
                self._log(Text("Usage: /embedding-model <name>", style="yellow"))
                return
            old = config.embedding_model
            config.embedding_model = rest
            import ollama
            self.chat.vector_store.ollama_client = ollama.Client(host=config.ollama_host)
            self._log(Text(f"Embedding model set to {rest} (was {old}).",
                            style="green"))
        elif cmd in ("results", "n", "topn"):
            try:
                self.n_results = int(rest or "5")
            except ValueError:
                self._log(Text("Usage: /results <number>", style="yellow"))
                return
            self.chat.n_results = self.n_results
            self._log(Text(f"Default results per question: {self.n_results}.",
                            style="green"))
        elif cmd == "export":
            self._log(Text("Exporting ...", style="dim"))
            self._start("export", self._task_export, rest or None)
        elif cmd == "clear":
            self._log(Text(
                "This will DELETE all indexed documents. "
                "Run /clear-force to confirm.", style="yellow"))
        elif cmd in ("clear-force", "wipe"):
            self._log(Text("Clearing database ...", style="dim"))
            self._start("clear", self._task_clear)
        else:
            self._log(Text(f"Unknown command /{cmd}. Type /help for the list.",
                            style="yellow"))

    @staticmethod
    def _extract_n(text):
        toks = text.split()
        for i, t in enumerate(toks):
            if t in ("-n", "--n", "-results", "--results") and i + 1 < len(toks):
                try:
                    return int(toks[i + 1])
                except ValueError:
                    return 5
        return 5

    def _show_help(self):
        lines = [
            "Q&A:  type a question and press Enter",
            "",
            "Commands:",
            "  /ingest [force]      index the documents folder",
            "  /ingest-file <path>  index a single file",
            "  /search <query>      raw semantic search (no LLM)",
            "  /ask <question>      LLM answer with citations",
            "  /summarize <doc>     summarize one document",
            "  /list               list indexed documents + counts",
            "  /stats              database statistics",
            "  /models [name]      list models / set chat model",
            "  /embedding-model n  set the embedding model",
            "  /results <n>        default chunks per question",
            "  /export [path]      export database to JSON",
            "  /clear-force        wipe the database",
            "  /help               show this help",
            "  /quit               exit",
            "",
            "Keys: i ingest  s search  a ask  l list  m models  ? help  q quit",
            "Pick a document in the left panel and press Enter to summarize it.",
        ]
        self._log(Panel(Text("\n".join(lines)), title="Help", border_style="yellow"))

    def action_ingest(self):
        self._log(Text("Ingesting documents ...", style="dim"))
        self._start("ingest", self._task_ingest, False, None)

    def action_search(self):
        self.query_one("#cmd").focus()

    def action_ask(self):
        self.query_one("#cmd").focus()

    def action_list(self):
        self._start("list", self._task_list)

    def action_models(self):
        self._start("models", self._task_models)

    def action_help(self):
        self._show_help()


def run_tui():
    DocApp().run()


if __name__ == "__main__":
    run_tui()
