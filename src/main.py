#!/usr/bin/env python3
"""
Ollama Document Repository - CLI Application
Process local documents for summarization and Q&A using Ollama.
"""

import os
import sys
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat import DocumentChat
from config import config

console = Console()


def check_ollama_connection():
    """Check if Ollama is accessible."""
    import urllib.request
    try:
        response = urllib.request.urlopen(f"{config.ollama_host}/api/tags", timeout=5)
        return response.status == 200
    except Exception:
        return False


@click.group()
@click.version_option(version='1.0.0')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """Ollama Document Repository - Process and query local documents."""
    if not check_ollama_connection():
        console.print(f"[red]Warning: Cannot connect to Ollama at {config.ollama_host}[/red]")
        console.print("  Make sure Ollama is running: [cyan]ollama serve[/cyan]")
    pass


@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force reprocessing of all documents')
@click.option('--file', '-i', type=click.Path(exists=True), help='Process a specific file')
@click.option('--batch-size', '-b', default=50, help='Batch size for embedding generation')
def ingest(force, file, batch_size):
    """Ingest documents into the vector database."""
    processor = DocumentProcessor()
    vector_store = VectorStore()

    if file:
        files = [Path(file)]
    else:
        files = processor.scan_documents()

    if not files:
        console.print("[yellow]No documents found in the documents directory.[/yellow]")
        return

    console.print(f"[cyan]Found {len(files)} document(s) to process[/cyan]")

    if force:
        console.print("[yellow]Force mode: clearing existing data...[/yellow]")
        vector_store.clear()

    all_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing documents...", total=len(files))

        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}...")

            # Check if already processed (unless force)
            if not force:
                existing = vector_store.collection.get(where={"source": str(file_path)})
                if existing['ids']:
                    console.print(f"  [dim]Skipping {file_path.name} (already processed)[/dim]")
                    progress.advance(task)
                    continue

            try:
                chunks = processor.process_file(file_path)
                all_chunks.extend(chunks)
                console.print(f"  [green]✓[/green] {file_path.name} ({len(chunks)} chunks)")
            except Exception as e:
                console.print(f"  [red]✗[/red] {file_path.name}: {e}")

            progress.advance(task)

    if all_chunks:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Generating embeddings and storing...", total=len(all_chunks))

            # Batch insert for efficiency
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                vector_store.add_documents(batch)
                progress.advance(task, len(batch))

        console.print(f"\n[green]Successfully ingested {len(all_chunks)} chunks from {len(files)} document(s)[/green]")
    else:
        console.print("[yellow]No new documents to ingest.[/yellow]")


@cli.command()
@click.argument('question', required=False)
@click.option('--results', '-n', default=5, help='Number of relevant chunks to retrieve')
@click.option('--no-sources', is_flag=True, help='Hide source citations')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive Q&A session')
@click.option('--model', '-m', help='Override chat model')
def ask(question, results, no_sources, interactive, model):
    """Ask a question about the documents."""
    chat = DocumentChat()
    
    if model:
        chat.ollama_client = chat.ollama_client.__class__(host=config.ollama_host)
        # Note: model override would require modifying the chat method

    if interactive or not question:
        console.print(Panel.fit(
            "[bold]Interactive Q&A Session[/bold]\n"
            "Type your questions. Enter 'quit', 'exit', or 'q' to exit.",
            border_style="cyan"
        ))

        while True:
            try:
                question = console.input("\n[bold cyan]Question:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue

            _ask_question(chat, question, results, not no_sources)
    else:
        _ask_question(chat, question, results, not no_sources)


def _ask_question(chat: DocumentChat, question: str, n_results: int, show_sources: bool):
    """Ask a single question and display results."""
    with console.status("[cyan]Searching documents and generating answer...[/cyan]"):
        result = chat.ask(question, n_results=n_results)

    console.print(Panel(
        Markdown(result['answer']),
        title="[bold]Answer[/bold]",
        border_style="green"
    ))

    if show_sources and result['sources']:
        table = Table(title="Sources", show_header=True, header_style="bold")
        table.add_column("Filename", style="cyan")
        table.add_column("Chunk", justify="right")
        table.add_column("Relevance", justify="right")

        for s in result['sources']:
            relevance = f"{1 - s['distance']:.2%}" if s['distance'] is not None else "N/A"
            table.add_row(s['filename'], str(s['chunk_index']), relevance)

        console.print(table)


@cli.command()
@click.argument('source', required=False)
def summarize(source):
    """Summarize a specific document or list all documents."""
    chat = DocumentChat()

    if source:
        # Summarize specific document
        with console.status(f"[cyan]Generating summary for {source}...[/cyan]"):
            result = chat.summarize_document(source)

        console.print(Panel(
            Markdown(result['summary']),
            title=f"[bold]Summary: {Path(source).name}[/bold]",
            border_style="blue"
        ))
    else:
        # List all documents
        sources = chat.list_documents()
        if not sources:
            console.print("[yellow]No documents in database.[/yellow]")
            return

        console.print(f"[cyan]Found {len(sources)} document(s):[/cyan]\n")
        for src in sources:
            console.print(f"  • {src}")


@cli.command()
def list():
    """List all documents in the database with chunk counts."""
    chat = DocumentChat()
    sources = chat.list_documents()

    if not sources:
        console.print("[yellow]No documents in database.[/yellow]")
        return

    stats = chat.get_stats()

    table = Table(title=f"Documents ({stats['unique_sources']} docs, {stats['total_chunks']} chunks)", show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Document", style="cyan")
    table.add_column("Chunks", justify="right")

    # Get chunk count per document
    vector_store = VectorStore()
    for i, src in enumerate(sources, 1):
        results = vector_store.collection.get(where={"source": src})
        chunk_count = len(results['ids']) if results['ids'] else 0
        table.add_row(str(i), src, str(chunk_count))

    console.print(table)


@cli.command()
def stats():
    """Show database statistics."""
    chat = DocumentChat()
    stats = chat.get_stats()

    console.print(Panel(
        f"[bold]Total Chunks:[/bold] {stats['total_chunks']}\n"
        f"[bold]Unique Documents:[/bold] {stats['unique_sources']}\n"
        f"[bold]Embedding Model:[/bold] {config.embedding_model}\n"
        f"[bold]Chat Model:[/bold] {config.chat_model}\n"
        f"[bold]Chunk Size:[/bold] {config.chunk_size}\n"
        f"[bold]Chunk Overlap:[/bold] {config.chunk_overlap}\n"
        f"[bold]Database Path:[/bold] {config.chroma_path}\n"
        f"[bold]Documents Path:[/bold] {config.documents_path}",
        title="[bold]Database Statistics[/bold]",
        border_style="blue"
    ))


@cli.command()
@click.option('--confirm', is_flag=True, help='Confirm without prompt')
def clear(confirm):
    """Clear all documents from the database."""
    if not confirm:
        if not click.confirm("This will delete all documents from the database. Continue?"):
            return

    vector_store = VectorStore()
    vector_store.clear()
    console.print("[green]Database cleared successfully.[/green]")


@cli.command()
def models():
    """Check available Ollama models."""
    import ollama
    client = ollama.Client(host=config.ollama_host)

    try:
        models_response = client.list()
        # Handle the ListResponse object
        models = getattr(models_response, 'models', [])
        if isinstance(models_response, dict):
            models = models_response.get('models', [])

        console.print(Panel(
            f"[bold]Ollama Endpoint:[/bold] {config.ollama_host}\n\n"
            f"[bold]Available Models:[/bold]",
            title="Ollama Status",
            border_style="cyan"
        ))

        table = Table(show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Modified", style="dim")

        model_names = []
        for m in models:
            name = getattr(m, 'model', m.get('model', m.get('name', 'Unknown')) if isinstance(m, dict) else 'Unknown')
            size = getattr(m, 'size', m.get('size', 0) if isinstance(m, dict) else 0)
            modified = getattr(m, 'modified_at', m.get('modified_at', 'Unknown') if isinstance(m, dict) else 'Unknown')
            size_gb = size / (1024**3)
            mod_str = str(modified)[:10] if modified != 'Unknown' else 'Unknown'
            table.add_row(name, f"{size_gb:.1f} GB", mod_str)
            model_names.append(name)

        console.print(table)

        # Check required models
        for required in [config.embedding_model, config.chat_model]:
            status = "[green]✓ Available[/green]" if any(required in m for m in model_names) else "[red]✗ Missing[/red]"
            console.print(f"  {required}: {status}")

    except Exception as e:
        console.print(f"[red]Failed to connect to Ollama at {config.ollama_host}: {e}[/red]")


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path (JSON)')
def export(output):
    """Export all documents and chunks to JSON."""
    import json
    from datetime import datetime
    
    vector_store = VectorStore()
    stats = vector_store.get_stats()
    
    # Get all documents with full content
    results = vector_store.collection.get()
    
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "statistics": stats,
        "documents": []
    }
    
    # Group by source
    from collections import defaultdict
    doc_chunks = defaultdict(list)
    documents = results.get('documents') or []
    metadatas = results.get('metadatas') or []
    ids = results.get('ids') or []
    for i, (doc, meta, id_) in enumerate(zip(documents, metadatas, ids)):
        doc_chunks[meta['source']].append({
            'id': id_,
            'text': doc,
            'metadata': meta
        })
    
    for source, chunks in doc_chunks.items():
        export_data['documents'].append({
            'source': source,
            'filename': chunks[0]['metadata']['filename'],
            'chunks': chunks
        })
    
    output_path = Path(output) if output else Path("ollama-docs-export.json")
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    console.print(f"[green]Exported {stats['unique_sources']} documents ({stats['total_chunks']} chunks) to {output_path}[/green]")


@cli.command()
@click.argument('query')
@click.option('--results', '-n', default=10, help='Number of results to return')
def search(query, results):
    """Search documents without generating an answer (raw semantic search)."""
    vector_store = VectorStore()
    
    with console.status(f"[cyan]Searching for: {query}[/cyan]"):
        search_results = vector_store.search(query, n_results=results)
    
    if not search_results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    console.print(Panel(f"[bold]Search Results for:[/bold] {query}", border_style="cyan"))
    
    table = Table(show_header=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Filename", style="cyan")
    table.add_column("Relevance", justify="right")
    table.add_column("Preview", style="dim")
    
    for i, r in enumerate(search_results, 1):
        relevance = f"{1 - r['distance']:.2%}" if r['distance'] is not None else "N/A"
        preview = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
        preview = preview.replace('\n', ' ')
        table.add_row(str(i), r['metadata']['filename'], relevance, preview)
    
    console.print(table)


if __name__ == '__main__':
    cli()