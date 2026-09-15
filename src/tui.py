"""Textual-based terminal UI for the Ollama Document Repository.

Run with no subcommand:  ./ollama-docs  OR  python -m src.main

Type a question and press Enter, or prefix a line with "/" for a command:

    /help                  list all commands
    /db                    list databases, show the active one
    /db new NAME           create a new database
    /db use NAME           make NAME the active database
    /db delete NAME        delete a database
    /db rename OLD NEW     rename a database
    /use NAME              alias of /db use NAME
    /ingest [force]        index every file in the documents folder
    /ingest-file <path>    index a single file
    /search <query>        raw semantic search (no LLM answer)
    /ask <question>        Q&A with an LLM answer + citations
    /summarize <doc>       summarize a document
    /list                  list documents in the active database
    /stats                 statistics for the active database
    /models [name]         list Ollama models / set the chat model
    /embedding-model <n>   set the embedding model
    /results <n>           default chunks retrieved per question
    /remove <doc>          remove one document from the active database
    /reassign <src> <dst>  move a document between databases
    /reassign <src> <dst> copy   copy instead of move
    /export [path.json]    export the active database to JSON
    /clear-force           wipe the active database
    /quit                  exit

Keys: i ingest  s search  a ask  l list  m models  ? help  q quit
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from textual.app import App
from textual.binding import Binding
from textual.widgets import (Footer, Header, Input, OptionList,
                             RichLog, Static)
from textual.containers import Horizontal, Vertical
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.worker import WorkerState
import functools

from config import config
from document_processor import DocumentProcessor
from chat import DocumentChat
from vector_store import DatabaseManager, VectorStore, _sanitize_name


def _resolve_source_key(mgr, db_name, source_arg):
    """Resolve user input (filename / relpath / key) to a stored key."""
    key = DocumentProcessor.match_source(source_arg)
    col = mgr._col(db_name)
    res = col.get() or {}
    stored = {m.get("source") for m in (res.get("metadatas") or
                                        []) if m}
    if key in stored:
        return key
    base = Path(str(source_arg)).name
    for k in stored:
        if k.endswith(base) or base in k:
            return k
    return None


class DocApp(App):
    BINDINGS = [
         Binding("i", "ingest", "Ingest", show=True),
         Binding("s", "search", "Search", show=True),
         Binding("a", "ask", "Ask", show=True),
         Binding("l", "list", "List", show=True),
         Binding("m", "models", "Models", show=True),
         Binding("?", "help", "Help"),
         Binding("d", "switch_db", "Switch DB", show=True),
         Binding("q", "quit", "Quit"),
     ]

    CSS = """
    Screen { layout: vertical; }
    #docs-panel { width: 44; dock: left; border: round $primary; }
    #dbs { height: 4fr; border: round $secondary; }
    #docs { height: 3fr; border: round $secondary; }
    #log { height: 1fr; }
    #cmd { dock: bottom; border: round $secondary; }
    """

    def __init__(self):
        super().__init__()
        self.chat = None
        self.model = config.chat_model
        self.n_results = config.n_results or 8
        self.active_db = "documents"
        self.databases = []
        self.sources = []
        self._last_question = "?"
        self._kinds = {}

    # ------------------------------------------------------------------ compose
    def compose(self):
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="docs-panel"):
                yield Static("DATABASES\n(Enter -> switch)", id="db-title")
                yield OptionList(id="dbs")
                yield Static("DOCUMENTS\n(Enter -> summarize)",
                             id="docs-title")
                yield OptionList(id="docs")
            yield RichLog(max_lines=800, wrap=True, markup=True,
                          auto_scroll=True, id="log")
        yield Input(placeholder="Ask a question, or type /help", id="cmd")
        yield Footer()

    def on_mount(self):
        mgr = DatabaseManager()
        self.active_db = mgr.get_active()
        self.chat = DocumentChat(
            model=self.model, n_results=self.n_results,
            collection_name=self.active_db, thorough=True,
        )
        self._write_welcome()
        self._start("refresh_dbs", self._task_refresh_dbs)
        self._start("refresh_docs", self._task_refresh_docs)
        self._start("status", self._task_status)
        self.query_one("#cmd").focus()

    # ------------------------------------------------------------- helpers
    def _log(self, renderable, **kwargs):
        self.query_one("#log").write(renderable)

    def _write_welcome(self):
        self._log(Text("Ollama Document Repository - TUI",
                       style="bold cyan"))
        self._log(Text(
            f"Ollama {config.ollama_host}  chat {self.model}  "
            f"embed {config.embedding_model}", style="dim"))
        self._log(Text(
            f"Active database: {self.active_db}   "
            f"docs: {config.documents_path}", style="dim"))
        self._log(Text(
            "Type a question, or /help.  "
            "Pick a database on the left and press Enter to switch.",
            style="dim"))

    def _start(self, kind, func, *args):
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
                self._log(Text(f"Render error ({kind}): {exc}",
                               style="red"))
        elif w.state == WorkerState.ERROR:
            err = getattr(w, "error", "unknown error")
            self._log(Text(f"  {kind} failed: {err}", style="red"))

    # ----------------------------------------------------------- blocking tasks
    def _task_status(self):
        import urllib.request
        connected = False
        try:
            r = urllib.request.urlopen(
                f"{config.ollama_host}/api/tags", timeout=5)
            connected = r.status == 200
        except Exception:
            connected = False
        vs = self.chat.vector_store
        s = vs.get_stats()
        return {"connected": connected, "docs": s["unique_sources"],
                "chunks": s["total_chunks"], "db_name": vs.collection_name}

    def _task_refresh_dbs(self):
        return {"dbs": DatabaseManager().list_databases()}

    def _task_refresh_docs(self):
        vs = self.chat.vector_store
        return {"sources": vs.get_all_sources(),
                "counts": vs.get_source_counts()}

    def _task_ask(self, question):
        return self.chat.ask(question, n_results=self.n_results,
                             thorough=True)

    def _task_search(self, query, n):
        return self.chat.vector_store.search(query, n_results=n)

    def _task_summarize(self, source):
        return self.chat.summarize_document(
            source, collection_name=self.active_db)

    def _task_ingest(self, force, single_file=None):
        vs = VectorStore(collection_name=self.active_db)
        proc = DocumentProcessor()
        files = [Path(single_file)] if single_file else proc.scan_documents()
        if force:
            vs.clear()
        out = {"files": 0, "chunks": 0, "skipped": [], "errors": []}
        for fp in files:
            key = DocumentProcessor.source_key(fp)
            if not force and vs.collection.get(
                    where={"source": key}).get("ids"):
                out["skipped"].append(fp.name)
                continue
            try:
                chunks = proc.process_file(fp)
                if chunks:
                    vs.add_documents(chunks)
                    out["files"] += 1
                    out["chunks"] += len(chunks)
            except Exception as exc:
                out["errors"].append(f"{fp.name}: {exc}")
        return out

    def _task_list(self):
        vs = self.chat.vector_store
        return {"sources": vs.get_all_sources(),
                "counts": vs.get_source_counts()}

    def _task_stats(self):
        return self.chat.get_stats()

    def _task_models(self):
        return self.chat.list_model_names()

    def _task_clear(self):
        self.chat.vector_store.clear()
        return {"db": self.active_db}

    def _task_export(self, path):
        vs = self.chat.vector_store
        s = vs.get_stats()
        res = vs.collection.get()
        out_path = Path(path) if path else Path(
            f"ollama-docs-export-{self.active_db}.json")
        import json
        payload = {
            "exported_at": datetime.now().isoformat(),
            "database": self.active_db,
            "statistics": s,
            "documents": [],
        }
        docs_list = []
        for meta, doc, id_ in zip(res.get("metadatas") or [],
                                  res.get("documents") or [],
                                  res.get("ids") or []):
            if not meta:
                continue
            src = meta.get("source", "?")
            docs_list.append({
                "source": src,
                "filename": meta.get("filename", src),
                "id": id_,
                "text": doc,
            })
        payload["documents"] = docs_list
        out_path.write_text(json.dumps(payload, indent=2))
        return {"path": str(out_path), "docs": len(docs_list),
                "chunks": s["total_chunks"]}

    # --------------------------------------------------- database tasks
    def _task_db_use(self, name):
        real = DatabaseManager().set_active(name)
        return {"ok": True, "active": real}

    def _task_db_new(self, name):
        real = DatabaseManager().create(name)
        return {"ok": True, "name": real}

    def _task_db_delete(self, name):
        mgr = DatabaseManager()
        n = _sanitize_name(name)
        if not mgr._exists(n):
            return {"ok": False, "error": f"No database '{name}'"}
        mgr.delete(n)
        mgr.set_active(mgr.get_active())
        return {"ok": True, "deleted": n,
                "active": DatabaseManager().get_active()}

    def _task_db_rename(self, old, new):
        mgr = DatabaseManager()
        try:
            real = mgr.rename(old, new)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "old": old, "new": real}

    def _task_remove_doc(self, source):
        mgr = DatabaseManager()
        key = _resolve_source_key(mgr, self.active_db, source)
        if key is None:
            return {"ok": False, "error": f"No doc '{source}' in "
                                          f"'{self.active_db}'"}
        removed = mgr.collection_delete_source(self.active_db, key)
        return {"ok": True, "removed": removed, "key": key}

    def _task_reassign(self, source, dest, copy):
        mgr = DatabaseManager()
        key = _resolve_source_key(mgr, self.active_db, source)
        if key is None:
            return {"ok": False, "error": f"No doc '{source}' in "
                                          f"'{self.active_db}'"}
        try:
            r = mgr.copy_source(key, dest, src_db=self.active_db,
                                remove_source=not copy)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "copy": copy, **r}

    # ----------------------------------------------------------- rendering
    def _render(self, kind, result):
        if kind == "status":
            self._render_status(result)
        elif kind == "refresh_dbs":
            self._render_dbs(result)
        elif kind == "refresh_docs":
            self._render_docs(result)
        elif kind == "ask":
            self._render_answer(result)
        elif kind == "search":
            self._render_search(result)
        elif kind == "summarize":
            summary = (result.get("summary") or "").strip() or "(empty)"
            self._log(Panel(
                Text(summary), title="Summary",
                border_style="blue"))
        elif kind == "ingest":
            self._render_ingest(result)
        elif kind == "list":
            self._render_docs(result)
        elif kind == "stats":
            self._render_stats(result)
        elif kind == "models":
            self._render_models(result)
        elif kind == "clear":
            self._log(Text(
                f"Cleared database '{result.get('db','?')}'.",
                style="green"))
            self._start("refresh_dbs", self._task_refresh_dbs)
            self._start("refresh_docs", self._task_refresh_docs)
        elif kind == "export":
            self._log(Text(
                f"Exported '{self.active_db}': "
                f"{result['docs']} docs, "
                f"{result['chunks']} chunks -> {result['path']}",
                style="green"))
        elif kind == "db_use":
            if result.get("ok"):
                self._set_active_and_rebind(result["active"])
                self._log(Text(
                    f"Active database: '{result['active']}'.",
                    style="green"))
                self._start("refresh_dbs", self._task_refresh_dbs)
                self._start("refresh_docs", self._task_refresh_docs)
            else:
                self._log(Text(
                    f"Switch failed: {result.get('error')}", style="red"))
        elif kind == "db_new":
            self._log(Text(
                f"Created database '{result.get('name')}'.",
                style="green"))
            self._start("refresh_dbs", self._task_refresh_dbs)
        elif kind == "db_delete":
            if result.get("ok"):
                self._log(Text(
                    f"Deleted '{result.get('deleted')}'.", style="green"))
                self._set_active_and_rebind(result.get("active"))
                self._start("refresh_dbs", self._task_refresh_dbs)
                self._start("refresh_docs", self._task_refresh_docs)
            else:
                self._log(Text(
                    f"Delete failed: {result.get('error')}", style="red"))
        elif kind == "db_rename":
            if result.get("ok"):
                self._log(Text(
                    f"Renamed '{result.get('old')}' -> "
                    f"'{result.get('new')}'.", style="green"))
                self._start("refresh_dbs", self._task_refresh_dbs)
                self._start("refresh_docs", self._task_refresh_docs)
            else:
                self._log(Text(
                    f"Rename failed: {result.get('error')}", style="red"))
        elif kind == "remove_doc":
            if result.get("ok"):
                self._log(Text(
                    f"Removed '{result.get('key')}' "
                    f"({result.get('removed')} chunk(s)) "
                    f"from '{self.active_db}'.", style="green"))
                self._start("refresh_docs", self._task_refresh_docs)
                self._start("refresh_dbs", self._task_refresh_dbs)
            else:
                self._log(Text(
                    f"Remove failed: {result.get('error')}", style="red"))
        elif kind == "reassign":
            if result.get("ok"):
                verb = "Copied" if result["copy"] else "Moved"
                self._log(Text(
                    f"{verb} '{result.get('source_key')}' -> "
                    f"'{result.get('dest_database')}' "
                    f"({result.get('chunks_added')} chunk(s)).",
                    style="green"))
                self._start("refresh_dbs", self._task_refresh_dbs)
                self._start("refresh_docs", self._task_refresh_docs)
            else:
                self._log(Text(
                    f"Reassign failed: {result.get('error')}",
                    style="red"))

    def _render_status(self, r):
        ok = r["connected"]
        db_name = r.get("db_name", self.active_db)
        self._log(Panel(
            Text.assemble(
                Text("OK " if ok else "ERR ",
                     style="bold green" if ok else "bold red"),
                Text(f"{config.ollama_host}\n", style="dim"),
                Text(f"{r['docs']} docs, {r['chunks']} chunks "
                     f"in '{db_name}'", style="dim")),
            title="Status", border_style="cyan" if ok else "red"))

    def _render_dbs(self, result):
        dbs = result.get("dbs", [])
        self.databases = [d["name"] for d in dbs]
        opts = self.query_one("#dbs")
        opts.clear_options()
        for d in dbs:
            active = d.get("active", False)
            label = (f"[*] {d['name']}"
                     if active else f"    {d['name']}")
            label += f"  ({d['chunks']}c/{d.get('docs', '?')}d)"
            opts.add_option(label)
        if not dbs:
            self._log(Text("No databases yet. /db new NAME.",
                           style="dim"))

    def _render_docs(self, result):
        sources = result.get("sources", [])
        counts = result.get("counts", {})
        self.sources = sources
        opts = self.query_one("#docs")
        opts.clear_options()
        if not sources:
            self._log(Text(
                f"No documents in '{self.active_db}'. "
                f"/ingest to index files.",
                style="dim"))
            return
        for s in sorted(sources):
            cnt = counts.get(s, "?")
            opts.add_option(f"{s}  ({cnt}c)")
        self._log(Text(
            f"[{self.active_db}]  {len(sources)} docs, "
            f"{sum(counts.values())} chunks",
            style="bold cyan"))

    def _render_answer(self, result):
        self._log(Text(f"Q: {self._last_question}", style="bold cyan"))
        answer = (result.get("answer") or "").strip() or "(no answer)"
        self._log(Panel(
            Text(answer),
            title=f"Answer  [{result.get('db','?')}]",
            border_style="green"))
        for s in result.get("sources", []):
            dist = s.get("distance")
            rel = (f"{1-dist:.1%}" if dist is not None else "n/a")
            self._log(Text(
                f"  {s.get('filename','?')} chunk "
                f"{s.get('chunk_index','?')}  "
                f"(relevance: {rel})", style="dim"))

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
            dist = r["distance"]
            rel = (f"{1-dist:.1%}" if dist is not None else "n/a")
            m = r.get("metadata") or {}
            table.add_row(str(i),
                          m.get("filename", "?"),
                          rel,
                          str(m.get("chunk_index", "?")))
        self._log(Panel(table, title="Search", border_style="cyan"))

    def _render_ingest(self, result):
        self._log(Text(
            f"Ingested {result.get('files', 0)} "
            f"({result.get('chunks', 0)} chunks) "
            f"into '{self.active_db}'.", style="green"))
        for err in result.get("errors", []):
            self._log(Text(f"  error: {err}", style="red"))
        for skip in result.get("skipped", []):
            self._log(Text(f"  skip: {skip}", style="dim"))
        self._start("refresh_dbs", self._task_refresh_dbs)
        self._start("refresh_docs", self._task_refresh_docs)

    def _render_stats(self, s):
        body = Text.assemble(
            Text("Active database: ", style="dim"),
            Text(s.get("collection", "?"), style="bold"),
            "\n",
            Text("Chunks: ", style="dim"),
            Text(str(s.get("total_chunks", 0)), style="bold"),
            "\n",
            Text("Documents: ", style="dim"),
            Text(str(s.get("unique_sources", 0)), style="bold"),
            "\n",
            Text("Embed model: ", style="dim"),
            Text(config.embedding_model, style="bold"),
            "\n",
            Text("Chat model: ", style="dim"),
            Text(self.model, style="bold"),
        )
        self._log(Panel(
            body,
            title=f"'{s.get('collection','?')}' Stats",
            border_style="blue"))

    def _render_models(self, names):
        self._log(Text(
            f"{len(names)} Ollama model(s):", style="bold"))
        for n in sorted(names):
            tag = ""
            if n == self.model:
                tag = "  [CHAT]"
            if n == config.embedding_model:
                tag = "  [EMBED]"
            style = "bold cyan" if tag else "cyan"
            self._log(Text(f"  {n}{tag}", style=style))

    # ----------------------------------------------------- input / actions
    def _set_active_and_rebind(self, name):
        from vector_store import VectorStore
        self.active_db = name
        self.chat = DocumentChat(
            model=self.model,
            n_results=self.n_results,
            collection_name=name,
            thorough=True,
        )

    def on_option_list_option_selected(self, event):
        if event.control.id == "dbs":
            idx = event.option_index
            if 0 <= idx < len(self.databases):
                name = self.databases[idx]
                self._log(
                    Text(f"Switching to '{name}' ...", style="dim"))
                self._start("db_use", self._task_db_use, name)
            return
        if event.control.id != "docs":
            return
        idx = event.option_index
        if 0 <= idx < len(self.sources):
            src = self.sources[idx]
            self._last_question = src
            self._log(
                Text(f"Summarizing '{src}' ...", style="dim"))
            self._start("summarize", self._task_summarize, src)

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
            q = raw
            self._last_question = q
            self._log(
                Text("[cyan]Searching & generating ...[/cyan]",
                     style="dim"))
            self._start("ask", self._task_ask, q)

    @staticmethod
    def _split_args(args_str):
        parts = args_str.split(maxsplit=4)
        if len(parts) == 0:
            return []
        cmd = parts[0].lower()
        rest = " ".join(parts[1:]).strip()
        return [cmd, rest] if len(parts) > 1 else [cmd, ""]

    def _dispatch(self, args):
        parts = self._split_args(args)
        if not parts:
            return
        cmd, rest = parts
        c = cmd

        if c in ("help", "?", "h"):
            self._show_help()
        elif c in ("ingest", "i"):
            force = "force" in rest.lower()
            self._log(Text("[%dim]Ingesting ...", style="dim"))
            self._start("ingest", self._task_ingest, force, None)
        elif c in ("ingest-file", "ingestfile", "if"):
            if not rest:
                self._log(Text("Usage: /ingest-file <path>",
                               style="yellow"))
                return
            self._log(Text("Ingesting file ...", style="dim"))
            self._start("ingest", self._task_ingest, False, rest)
        elif c in ("search", "s"):
            n = self._extract_n(rest)
            q = rest
            if "\n" in rest:
                q = rest.strip()[:rest.index("\n")]
            if not q:
                self._log(Text("Usage: /search <query>",
                               style="yellow"))
                return
            self._log(Text("Searching ...", style="dim"))
            self._start("search", self._task_search, q, n)
        elif c in ("ask", "a"):
            if not rest:
                self._log(Text("Usage: /ask <question>",
                               style="yellow"))
                return
            self._last_question = rest
            self._log(Text("Searching & generating ...",
                           style="dim"))
            self._start("ask", self._task_ask, rest)
        elif c in ("summarize", "summary", "sum"):
            if not rest:
                self._log(Text("Usage: /summarize <document>",
                               style="yellow"))
                return
            self._last_question = rest
            self._log(Text("Summarizing ...", style="dim"))
            self._start("summarize", self._task_summarize, rest)
        elif c in ("list", "l"):
            self._start("list", self._task_list)
        elif c == "stats":
            self._start("stats", self._task_stats)
        elif c == "models" or c == "m":
            if rest:
                self.model = rest
                self.chat.set_model(rest)
                self._log(
                    Text(f"Chat model: {rest}", style="green"))
            else:
                self._start("models", self._task_models)
        elif c in ("embed", "embedding", "embedding-model"):
            if not rest:
                self._log(Text("Usage: /embed <model>",
                               style="yellow"))
                return
            old = config.embedding_model
            config.embedding_model = rest
            import ollama as _ol
            self.chat.vector_store.ollama_client = _ol.Client(
                host=config.ollama_host)
            self._log(Text(
                f"Embed model: {rest} "
                f"(was {old})", style="green"))
        elif c in ("results", "n", "topn"):
            try:
                self.n_results = int(rest or 8)
            except ValueError:
                self._log(Text("Usage: /results <number>",
                               style="yellow"))
                return
            self.chat.n_results = self.n_results
            self._log(Text(
                f"Default results/question: "
                f"{self.n_results}", style="green"))
        elif c == "db":
            self._dispatch_db(rest)
        elif c == "use":
            if not rest:
                self._log(Text("Usage: /use <name>",
                               style="yellow"))
                return
            self._log(Text(f"Switching to '{rest}' ...", style="dim"))
            self._start("db_use", self._task_db_use, rest)
        elif c in ("remove", "rm", "del"):
            if not rest:
                self._log(Text("Usage: /remove <doc>",
                               style="yellow"))
                return
            doc = rest
            self._log(Text(
                f"Removing '{doc}' from "
                f"'{self.active_db}' ...", style="dim"))
            self._start("remove_doc", self._task_remove_doc, doc)
        elif c in ("reassign", "move"):
            parts2 = rest.split(maxsplit=2)
            if len(parts2) < 2:
                self._log(
                    Text("Usage: /reassign <src> <dst> [copy]",
                         style="yellow"))
                return
            src, dst = parts2[0], parts2[1]
            copy = len(parts2) > 2 and parts2[2].lower() == "copy"
            self._log(Text(
                f"Moving/copying '{src}' -> '{dst}' ...",
                style="dim"))
            self._start("reassign", self._task_reassign,
                        src, dst, copy)
        elif c == "export":
            self._log(Text("Exporting ...", style="dim"))
            self._start("export", self._task_export, rest or None)
        elif c == "clear":
            self._log(Text(
                "Type /clear-force to wipe "
                f"'{self.active_db}'.", style="yellow"))
        elif c in ("clear-force", "wipe"):
            self._log(Text(
                f"Clearing '{self.active_db}' ...",
                style="dim"))
            self._start("clear", self._task_clear)
        else:
            self._log(Text(
                f"Unknown /{c}. Type /help.", style="yellow"))

    def _dispatch_db(self, rest):
        parts = rest.split(maxsplit=4)
        sub = (parts[0].lower() if parts else "list")
        arg = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
        if sub in ("list", "show", "l"):
            self._start("refresh_dbs", self._task_refresh_dbs)
        elif sub in ("new", "create", "add"):
            if not arg:
                self._log(Text("Usage: /db new <name>",
                               style="yellow"))
                return
            self._log(Text(f"Creating database '{arg}' ...",
                           style="dim"))
            self._start("db_new", self._task_db_new, arg)
        elif sub in ("use", "switch", "set"):
            if not arg:
                self._log(Text("Usage: /db use <name>",
                               style="yellow"))
                return
            self._log(Text(f"Switching to '{arg}' ...", style="dim"))
            self._start("db_use", self._task_db_use, arg)
        elif sub in ("delete", "drop", "rm", "del"):
            if not arg:
                self._log(Text("Usage: /db delete <name>",
                               style="yellow"))
                return
            db = _sanitize_name(arg)
            if not DatabaseManager()._exists(db):
                self._log(Text(
                    f"No database '{arg}'.", style="yellow"))
                return
            self._log(Text(
                f"Deleting database '{db}' ...",
                style="dim"))
            self._start("db_delete", self._task_db_delete, db)
        elif sub in ("rename", "rename", "r"):
            p = arg.split(maxsplit=1)
            if len(p) < 2:
                self._log(Text("Usage: /db rename <old> <new>",
                               style="yellow"))
                return
            self._log(Text(f"Renaming '{p[0]}' -> '{p[1]}' ...",
                           style="dim"))
            self._start("db_rename", self._task_db_rename,
                        p[0], p[1])
        elif sub == "stats":
            if not arg:
                self._start("stats", self._task_stats)
                return
            self._log(
                Text(f"'{arg}' stats:", style="dim"))
            try:
                vs = VectorStore(collection_name=arg)
                s = vs.get_stats()
                self._log(Text(
                    f"  {s['total_chunks']} chunks, "
                    f"{s['unique_sources']} docs", style="dim"))
            except Exception as e:
                self._log(Text(f"  error: {e}", style="red"))
        else:
            self._log(Text(
                "db subcommands: list, new, use, delete, "
                "rename, stats", style="yellow"))

    def _extract_n(self, text):
        toks = text.split()
        for i, t in enumerate(toks):
            if t in ("--n", "-n", "--results", "-results"):
                if i + 1 < len(toks):
                    try:
                        return int(toks[i + 1])
                    except ValueError:
                        return 8
        return 8

    def _show_help(self):
        lines = [
            "Q&A:  Type a question and press Enter.",
            "",
            "Databases:",
            "  /db                list databases",
            "  /db new <name>     create a database",
            "  /db use <name>     make a database active",
            "  /use <name>        same as /db use",
            "  /db delete <name>  delete a database",
            "  /db rename <o> <n> rename a database",
            "",
            "Documents:",
            "  /ingest [force]    index files in documents/",
            "  /ingest-file <p>   index a single file",
            "  /list [db]         list documents in active DB",
            "  /summarize <doc>   summarize a document",
            "  /remove <doc>      remove one document from DB",
            "  /reassign <src> <dst> [copy]  move/copy doc between DBs",
            "",
            "Q&A / Search:",
            "  /ask <question>    LLM answer with citations",
            "  /search <query>    raw semantic search, no LLM",
            "",
            "Config:",
            "  /models [name]     list models / set chat model",
            "  /embed <model>     set the embedding model",
            "  /results <n>       default chunks per question",
            "  /export [path]     export active DB to JSON",
            "",
            "Other:",
            "  /stats             stats for active DB",
            "  /clear-force       wipe the active database",
            "  /help              this help",
            "  /quit              exit",
            "",
            "Keys:  i ingest  s search  a ask  l list  "
            "m models  ? help  d switch DB  q quit",
            "Left panel: pick a database to switch to it; pick a document "
            "to summarize it.",
        ]
        self._log(Panel(
            Text("\n".join(lines)),
            title="Help", border_style="yellow"))

    def action_ingest(self):
        self._log(Text("Ingesting ...", style="dim"))
        self._start("ingest", self._task_ingest, False, None)

    def action_search(self):
        self.query_one("#cmd").focus()

    def action_ask(self):
        self.query_one("#cmd").focus()

    def action_list(self):
        self._start("list", self._task_list)

    def action_models(self):
        self._start("models", self._task_models)

    def action_switch_db(self):
        self._start("refresh_dbs", self._task_refresh_dbs)
        # Focus the database option list
        self.query_one("#dbs").focus()

    def action_help(self):
        self._show_help()


def run_tui():
    DocApp().run()


if __name__ == "__main__":
    run_tui()

