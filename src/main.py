#!/usr/bin/env python3
# Ollama Document Repository - CLI entry point + TUI launcher.
# No subcommand -> TUI.  Subcommand -> one-shot action.
# --db NAME on any subcommand targets a specific database for that call only.

import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from vector_store import VectorStore, DatabaseManager, _sanitize_name
from chat import DocumentChat
from config import config

console = Console()

# Per-invocation database target, set by the --db flag. In-memory only, so
# one-shot CLI calls never clobber the remembered active database; use
# "db use" / "db switch" to change the persisted one.
_db_override = None


def launch_tui(db=None):
    pass
    # Lazy-import the TUI so the one-shot CLI stays lightweight.
    try:
        from tui import run_tui
    except Exception as e:
        console.print(f"[red]Failed to start the TUI: {e}[/red]")
        console.print("  Install TUI deps, or use a subcommand. "
                     "See 'ollama-docs --help'.")
        return
    if db:
        DatabaseManager().set_active(db)
    run_tui()


def check_ollama_connection():
    import urllib.request
    try:
        response = urllib.request.urlopen(
            f"{config.ollama_host}/api/tags", timeout=5)
        return response.status == 200
    except Exception:
        return False


def _target_db():
    pass
    # Database this command should operate on: --db override, else active.
    mgr = DatabaseManager()
    return mgr.ensure(_db_override) if _db_override else mgr.get_active()


def _make_store():
    return VectorStore(collection_name=_target_db())


def _make_chat(model=None, n_results=None, thorough=True):
    return DocumentChat(model=model, n_results=n_results,
                       collection_name=_target_db(), thorough=thorough)


@click.group(invoke_without_command=True)
@click.version_option(version='1.2.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--db', 'db_name', default=None,
          help='Target a specific database for this command (creates it on demand).')
@click.pass_context
def cli(ctx, verbose, db_name):
    global _db_override
    _db_override = db_name

    if not check_ollama_connection():
        console.print(f"[red]Warning: Cannot connect to Ollama at "
                     f"{config.ollama_host}[/red]")
        console.print("  Make sure Ollama is running: [cyan]ollama serve[/cyan]")

    if ctx.invoked_subcommand is None:
        launch_tui(db=db_name)
        raise SystemExit(0)


@cli.command()
@click.option('--force', '-f', is_flag=True,
          help='Clear the target database first, then re-ingest')
@click.option('--file', '-i', type=click.Path(exists=True),
          help='Process a specific file instead of scanning the folder')
@click.option('--batch-size', '-b', default=50,
          help='Batch size for embedding generation')
def ingest(force, file, batch_size):
    pass
    # Ingest documents into the target database.
    # Without --force this ADDS to the database without wiping it.
    db = _target_db()
    if _db_override:
        console.print(f"[cyan]Target database: {db}[/cyan]")
    processor = DocumentProcessor()
    vs = VectorStore(collection_name=db)
    files = [Path(file)] if file else processor.scan_documents()
    if not files:
        console.print("[yellow]No documents found in the documents directory.[/yellow]")
        return
    console.print(f"[cyan]Found {len(files)} document(s)[/cyan]")

    if force:
        console.print(f"[yellow]Force: clearing database '{db}'...[/yellow]")
        vs.clear()

    all_chunks = []
    with Progress(SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(files))
        for fp in files:
            progress.update(task, description=f"Processing {fp.name}...")
            key = DocumentProcessor.source_key(fp)
            if not force and vs.collection.get(where={"source": key})["ids"]:
                console.print(f"  [dim]Skip {fp.name} (already in {db})[/dim]")
                progress.advance(task)
                continue
            try:
                chunks = processor.process_file(fp)
                all_chunks.extend(chunks)
                console.print(f"  [green]OK[/green] {fp.name} ({len(chunks)} chunks)")
            except Exception as e:
                console.print(f"  [red]X[/red] {fp.name}: {e}")
            progress.advance(task)

    if all_chunks:
        with Progress(SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(), TaskProgressColumn(), console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(all_chunks))
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                vs.add_documents(batch)
                progress.advance(task, len(batch))
        console.print(
            f"\n[green]Added {len(all_chunks)} chunk(s) from "
            f"{len(files)} file(s) to '{db}'[/green]")
    else:
        console.print("[yellow]No new documents to ingest.[/yellow]")


def _ask_question(chat, question, n_results, show_sources):
    with console.status("[cyan]Searching and answering...[/cyan]"):
        result = chat.ask(question, n_results=n_results)
    console.print(Panel(
        Markdown(result['answer']),
        title=(f"[bold]Answer[/bold]  [dim]db:{result.get('db') or chat.collection_name}"
               f"({result.get('n_chunks',0)} chunks, {result.get('n_docs',0)} docs)"
               + (" [truncated]" if result.get('truncated') else "")),
        border_style="green"))
    if show_sources and result['sources']:
        table = Table(title="[bold]Sources[/bold]", show_header=True)
        table.add_column("Filename", style="cyan")
        table.add_column("Chunk", justify="right")
        table.add_column("Relevance", justify="right")
        for s in result['sources']:
            rel = f"{1 - s['distance']:.2%}" if s['distance'] is not None else "N/A"
            table.add_row(s['filename'], str(s['chunk_index']), rel)
        console.print(table)


@cli.command()
@click.argument('question', required=False)
@click.option('--results', '-n', default=None,
          help='Chunks to retrieve (default: config.N_RESULTS)')
@click.option('--no-sources', is_flag=True, help='Hide source citations')
@click.option('--interactive', '-i', is_flag=True, help='Interactive Q&A mode')
@click.option('--model', '-m', help='Override chat model')
@click.option('--quick', is_flag=True, help='Concise one-point answer')
@click.option('--thorough', is_flag=True,
          help='Thorough multi-source answer (default)')
def ask(question, results, no_sources, interactive, model, quick, thorough):
    pass
    # Thorough is the default. --quick makes the answer single-point.
    thorough_mode = not quick
    chat = _make_chat(model=model, n_results=results, thorough=thorough_mode)
    if interactive or not question:
        console.print(Panel.fit(
            f"[bold]Interactive Q&A - database: "
            f"[cyan]{chat.collection_name}[/cyan]\n"
            f"({chat.vector_store.get_stats()['unique_sources']} docs)\n"
            "Type a question, or 'q' to quit.",
            border_style="cyan"))
        while True:
            try:
                q = console.input("\n[bold cyan]Question:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ('quit', 'exit', 'q'):
                break
            if not q:
                continue
            _ask_question(chat, q, chat.n_results, not no_sources)
    else:
        _ask_question(chat, question, chat.n_results, not no_sources)


@cli.command()
@click.argument('source', required=False)
def summarize(source):
    db = _target_db()
    chat = _make_chat(collection_name=db)
    if source:
        with console.status(f"[cyan]Summarizing {source}...[/cyan]"):
            r = chat.summarize_document(source, collection_name=db)
        console.print(Panel(
            Markdown(r['summary']),
            title=f"[bold]Summary: {Path(source).name}[/bold]  [dim]db:{db}[/dim]",
            border_style="blue"))
    else:
        docs = chat.list_documents()
        if not docs:
            console.print(f"[yellow]No documents in '{db}'.[/yellow]")
            return
        console.print(f"[cyan]Database '{db}' - {len(docs)} docs:[/cyan]\n")
        for s in docs:
            console.print(f"  {s}")


@cli.command(name="list")
def list_documents_cmd():
    db = _target_db()
    vs = VectorStore(collection_name=db)
    counts = vs.get_source_counts()
    if not counts:
        console.print(f"[yellow]No documents in '{db}'.[/yellow]")
        return
    table = Table(title=f"[bold]Database '{db}'[/bold]  "
                  f"({len(counts)} docs, {sum(counts.values())} chunks)",
                  show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right")
    for i, (src, cnt) in enumerate(sorted(counts.items()), 1):
        table.add_row(str(i), src, str(cnt))
    console.print(table)


@cli.command()
def stats():
    db = _target_db()
    vs = VectorStore(collection_name=db)
    s = vs.get_stats()
    console.print(Panel(
        f"[bold]Database:[/bold] {db}\n"
        f"[bold]Chunks:[/bold] {s['total_chunks']}\n"
        f"[bold]Documents:[/bold] {s['unique_sources']}\n"
        f"[bold]Embed model:[/bold] {config.embedding_model}\n"
        f"[bold]Chat model:[/bold] {config.chat_model}\n"
        f"[bold]Storage:[/bold] {config.chroma_path}",
        title="[bold]Stats[/bold]", border_style="blue"))
    _render_db_table("All Databases")


@cli.command()
@click.option('--confirm', is_flag=True, help='Skip the confirmation prompt')
def clear(confirm):
    db = _target_db()
    n = _make_store().get_stats()['total_chunks']
    console.print(
        f"[yellow]Clearing '{db}' ({n} chunks). Other databases NOT affected.[/yellow]")
    if not confirm and not click.confirm("Continue?"):
        return
    _make_store().clear()
    console.print(f"[green]Database '{db}' cleared.[/green]")


@cli.command()
def models():
    try:
        names = DocumentChat().list_model_names()
    except Exception as e:
        console.print(f"[red]Ollama unreachable at {config.ollama_host}: {e}[/red]")
        return
    console.print(Panel(
        f"[bold]Endpoint:[/bold] {config.ollama_host}",
        title="Ollama Status", border_style="cyan"))
    table = Table(show_header=True)
    table.add_column("Model", style="cyan")
    for n in sorted(names):
        tag = ""
        if n == config.chat_model:
            tag = " [CHAT]"
        if n == config.embedding_model:
            tag += " [EMBED]"
        table.add_row(n + tag, "", "")
    console.print(table)
    for req in [config.embedding_model, config.chat_model]:
        ok = any(req in m or m.startswith(req.split(':')[0])
                 for m in names)
        if ok:
            console.print(f"  {req}: [green]Available[/green]")
        else:
            console.print(f"  {req}: [red]Missing[/red]")


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='Number of results')
def search(query, results):
    db = _target_db()
    vs = VectorStore(collection_name=db)
    with console.status(f"[cyan]Searching '{db}': {query}[/cyan]"):
        res = vs.search(query, n_results=results)
    if not res:
        console.print("[yellow]No results found.[/yellow]")
        return
    table = Table(title=f"[bold]Search in '{db}': {query}[/bold]",
                  show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Rel.", justify="right")
    table.add_column("Preview", style="dim")
    for i, r in enumerate(res, 1):
        rel = f"{1-r['distance']:.1%}" if r['distance'] is not None else "N/A"
        prev = r['text'][:120].replace('\n',' ')
        table.add_row(str(i), r['metadata']['filename'], rel, prev)
    console.print(table)


# ========================================================== database commands

def _render_db_table(title):
    mgr = DatabaseManager()
    entries = mgr.list_databases()
    if not entries:
        console.print(
            "[yellow]No databases. Create one with 'db new NAME'.[/yellow]")
        return
    table = Table(title=f"[bold]{title}[/bold]", show_header=True)
    table.add_column("Database", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("Docs", justify="right")
    table.add_column("Sources", style="dim")
    for e in entries:
        name = ("[*]" if e["active"] else "   ") + e["name"]
        srcs = ", ".join(e['sources'])[:50] or "-"
        table.add_row(name, str(e['chunks']), str(len(e['sources'])), srcs)
    console.print(table)
    console.print(f"  [dim]active: {DatabaseManager().get_active()}[/dim]")


@cli.group()
def db():
    pass
    # Manage multiple databases: CRUD, selection, rename.


@db.command(name="new")
@click.argument('name')
def db_new(name):
    real = DatabaseManager().create(name)
    console.print(f"[green]Created empty database '{real}'.[/green]")


@db.command(name="list")
def db_list():
    _render_db_table("Databases")


# The "use" and "switch" commands share one callback but need two names.
@click.argument('name')
def _db_use(name):
    real = DatabaseManager().set_active(name)
    console.print(f"[green]Active database: '{real}' (persisted).[/green]")

db_use = db.command(name="use", help="Make an existing database the active one (persisted).")(_db_use)
db_switch = db.command(name="switch", help="Alias of 'db use': make a database the active one.")(_db_use)


# The "delete" and "drop" commands share one callback but need two names.
@click.argument('name')
@click.option('--confirm', is_flag=True, help='Skip the confirmation prompt')
def _db_delete(name, confirm):
    mgr = DatabaseManager()
    sanitized = _sanitize_name(name)
    if not mgr._exists(sanitized):
        console.print(f"[yellow]No database named '{name}'.[/yellow]")
        return
    n = VectorStore(collection_name=sanitized).get_stats()['total_chunks']
    if n and not confirm and not click.confirm(
            f"Delete '{sanitized}' ({n} chunks)?"):
        return
    mgr.delete(sanitized)
    console.print(f"[green]Deleted '{sanitized}'.[/green]")

db_delete = db.command(
    name="delete",
    help="Delete a whole database.")(_db_delete)
db_drop = db.command(
    name="drop",
    help="Alias of 'db delete' - delete a whole database.")(_db_delete)


@db.command(name="rename")
@click.argument('old')
@click.argument('new')
def db_rename(old, new):
    try:
        real = DatabaseManager().rename(old, new)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return
    console.print(f"[green]Renamed '{old}' -> '{real}'.[/green]")


@db.command(name="stats")
@click.argument('name')
def db_stats(name):
    vs = VectorStore(collection_name=name)
    s = vs.get_stats()
    console.print(Panel(
        f"[bold]Database:[/bold] {s['collection']}\n"
        f"[bold]Chunks:[/bold] {s['total_chunks']}\n"
        f"[bold]Documents:[/bold] {s['unique_sources']}",
        title=f"Database '{s['collection']}'", border_style="blue"))
    for src in sorted(s['sources']):
        console.print(f"  {src}")


# ========================================================= document commands

def _resolve_source_key(db_name, source_arg):
    pass
    # Resolve user input (filename / relpath / key) to a stored source key,
    # or return None if not found.
    mgr = DatabaseManager()
    key = DocumentProcessor.match_source(source_arg)
    col = mgr._col(db_name)
    stored = {m.get('source'): m
              for m in (col.get() or {}).get('metadatas') or []
              if m}
    if key in stored:
        return key
     # Fallback: substring / basename match
    base = Path(str(source_arg)).name
    for k in stored:
        if k.endswith(base) or base in k or k.endswith(str(source_arg)):
            return k
    return None


@cli.command()
@click.argument('source')
@click.option('--from', 'from_db', default=None,
          help='Source database (default: active)')
@click.option('--confirm', is_flag=True, help='Skip confirmation')
def remove(source, from_db, confirm):
     # Remove one document from a database without clearing the whole thing.
    db = from_db or _db_override or DatabaseManager().get_active()
    mgr = DatabaseManager()
    key = _resolve_source_key(db, source)
    if key is None:
        console.print(f"[red]No document matching '{source}' in '{db}'.[/red]")
        return
    n = mgr.collection_delete_source(db, key)
    console.print(f"[green]Removed {n} chunk(s) for '{key}' from '{db}'.[/green]")


@cli.command()
@click.argument('source')
@click.argument('destination')
@click.option('--copy', is_flag=True,
          help='Keep the document in the source database')
@click.option('--from', 'from_db', default=None,
          help='Source database (default: active)')
def reassign(source, destination, copy, from_db):
      # Move or copy one document between databases (carries stored embeddings
      # so no re-embedding is needed).
    target = from_db or _db_override or DatabaseManager().get_active()
    mgr = DatabaseManager()
    key = _resolve_source_key(target, source)
    if key is None:
        console.print(f"[red]No document matching '{source}'.[/red]")
        return
    try:
        res = mgr.copy_source(key, destination, src_db=target,
                              remove_source=not copy)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return
    verb = "Copied" if copy else "Moved"
    console.print(f"[green]{verb} '{res['source_key']}' -> "
                 f"'{res['to']}' ({res['chunks_added']} chunk(s)).[/green]")
    if copy:
        console.print("[dim]  Kept original in source database.[/dim]")


@cli.command()
def databases():
    pass
    # Shortcut alias listing all databases and the active one.
    _render_db_table("Databases")


# ========================================================= export / import

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path (JSON)')
def export(output):
    db = _target_db()
    vs = VectorStore(collection_name=db)
    s = vs.get_stats()
    res = vs.collection.get()
    out_data = {
        "exported_at": datetime.now().isoformat(),
        "database": db,
        "statistics": s,
        "documents": [],
    }
    doc_chunks = defaultdict(list)
    first_meta = {}
    for doc, meta, id_ in zip(res.get('documents') or [],
                               res.get('metadatas') or [],
                               res.get('ids') or []):
        src = (meta or {}).get('source', '?')
        doc_chunks[src].append({'id': id_, 'text': doc, 'metadata': meta})
        first_meta.setdefault(src, meta)
    for src, chunks in doc_chunks.items():
        out_data['documents'].append({
            'source': src,
            'filename': first_meta.get(src, {}).get('filename', src),
            'chunks': chunks,
        })
    out_path = Path(output) if output else Path(f"ollama-docs-export-{db}.json")
    import json
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    console.print(f"[green]Exported '{db}': "
                 f"{s['unique_sources']} docs, "
                 f"{s['total_chunks']} chunks -> {out_path}[/green]")


if __name__ == '__main__':
    cli()
