"""
Science Graph — CLI entry point.
All user-facing output uses src.console for consistent rich formatting.
"""

# Suppress noisy external library output and benign leaked semaphore warnings on shutdown
import os
import warnings

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Suppress benign leaked semaphore warnings on shutdown from multiprocessing/PyTorch/MLX on macOS
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")
if "PYTHONWARNINGS" in os.environ:
    if "ignore:resource_tracker:UserWarning" not in os.environ["PYTHONWARNINGS"]:
        os.environ["PYTHONWARNINGS"] += ",ignore:resource_tracker:UserWarning"
else:
    os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"

import json
import webbrowser
from pathlib import Path
from typing import Optional, List

import typer
from rich.panel import Panel
from rich.table import Table
from rich import box

from src import console as con
from src.config import config
from src.indexer import Indexer
from src.llm_engine import LLMEngine
from src.rag import RAGPipeline
from src.services.container import container

app = typer.Typer(
    help="Science Graph — local AI-powered knowledge base for papers, notes & books",
    no_args_is_help=True,
)


# ── Service factory ───────────────────────────────────────────────────────────

def get_services(load_llm: bool = True, load_embeddings: bool = True, use_cloud: bool = False):
    """Initializes and returns database repositories and engines."""
    graph_repo = container.get_graph_repo()
    vector_repo = container.get_vector_repo()

    embedding_engine = None
    if load_embeddings:
        embedding_engine = container.get_embedding_engine()

    llm_engine = None
    if load_llm:
        try:
            llm_engine = container.get_llm_engine(use_cloud=use_cloud)
        except Exception as e:
            con.error(f"Could not load LLM engine: {e}")

    return graph_repo, vector_repo, embedding_engine, llm_engine


# ── index ─────────────────────────────────────────────────────────────────────

def print_trace_table(source_name: str, trace_info: dict) -> None:
    """Prints a premium, visually satisfying trace table for the ingestion stages."""
    table = Table(
        title=f"Ingestion Trace: {source_name}",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        show_footer=True,
        box=box.ROUNDED,
    )
    table.add_column("Stage", footer="[accent]Total Duration[/accent]")
    table.add_column("Duration", justify="right", style="yellow")
    table.add_column("LLM Tokens", justify="right", style="bold magenta")

    stages = trace_info.get("stages", {})
    tokens = trace_info.get("tokens", {})

    total_duration = sum(stages.values())
    total_tokens = sum(tokens.values())

    chronological_stages = [
        "Document Parsing",
        "NER Author Fallback",
        "Metadata Enrichment",
        "Concept & Tag Extraction",
        "Graph Persistence",
        "Chunking & Embedding",
        "Archiving",
        "Summary Generation",
    ]

    seen_stages = set()

    for stage_name in chronological_stages:
        if stage_name in stages or stage_name in tokens:
            seen_stages.add(stage_name)
            duration_val = stages.get(stage_name, 0.0)
            duration_str = f"{duration_val:.3f}s" if stage_name in stages else "-"
            token_val = tokens.get(stage_name, 0)
            token_str = f"{token_val:,}" if token_val > 0 else "-"
            table.add_row(stage_name, duration_str, token_str)

    # Any other timing log
    for stage_name, duration_val in stages.items():
        if stage_name not in seen_stages:
            seen_stages.add(stage_name)
            token_val = tokens.get(stage_name, 0)
            token_str = f"{token_val:,}" if token_val > 0 else "-"
            table.add_row(stage_name, f"{duration_val:.3f}s", token_str)

    # Any other token log
    for token_stage, token_val in tokens.items():
        if token_stage not in seen_stages:
            table.add_row(token_stage, "-", f"{token_val:,}")

    # Set footers for totals
    table.columns[1].footer = f"[bold yellow]{total_duration:.3f}s[/bold yellow]"
    table.columns[2].footer = f"[bold magenta]{total_tokens:,}[/bold magenta]" if total_tokens > 0 else "-"

    con.blank()
    con.console.print(table)
    con.blank()


# ── index ─────────────────────────────────────────────────────────────────────

def print_session_summary_table(session_traces: List[dict]) -> None:
    """Prints a consolidated summary of the batch index execution using Rich."""
    total_docs = len(session_traces)
    successful = sum(1 for t in session_traces if t.get("success", False))
    skipped_dups = sum(1 for t in session_traces if t.get("skipped_duplicate", False))
    failed = total_docs - successful - skipped_dups
    
    orig_bytes = sum(t.get("original_size", 0) for t in session_traces)
    comp_bytes = sum(t.get("compressed_size", 0) for t in session_traces)
    saved_bytes = max(0, orig_bytes - comp_bytes)
    saved_mb = saved_bytes / (1024 * 1024)
    orig_mb = orig_bytes / (1024 * 1024)
    
    authors = sum(t.get("authors_count", 0) for t in session_traces)
    concepts = sum(t.get("concepts_count", 0) for t in session_traces)
    tags = sum(t.get("tags_count", 0) for t in session_traces)
    refs = sum(t.get("references_count", 0) for t in session_traces)
    
    durations = []
    for t in session_traces:
        if t.get("success", False) and "stages" in t:
            durations.append(sum(t["stages"].values()))
    avg_duration = sum(durations) / len(durations) if durations else 0.0
    total_duration = sum(durations)

    token_totals = {}
    for t in session_traces:
        for stage, count in t.get("tokens", {}).items():
            token_totals[stage] = token_totals.get(stage, 0) + count
            
    con.blank()
    table = Table(
        title="✨ Ingestion Session Summary Table ✨",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        box=box.DOUBLE,
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Summary Value", style="bold green", justify="right")
    
    table.add_row(
        "Processed Documents", 
        f"[bold]{total_docs}[/bold] ([green]{successful} OK[/green], [yellow]{skipped_dups} Dups[/yellow], [red]{failed} Fail[/red])"
    )
    
    if orig_bytes > 0:
        ratio = (saved_bytes / orig_bytes) * 100
        table.add_row(
            "Original PDF Volume", 
            f"{orig_mb:.2f} MB"
        )
        table.add_row(
            "PDF Compression Savings", 
            f"{saved_mb:.2f} MB ([bold yellow]{ratio:.1f}% saved[/bold yellow])"
        )
        
    table.add_row(
        "Knowledge Mapped", 
        f"{authors} Authors, {concepts} Concepts, {tags} Tags"
    )
    table.add_row(
        "Graph Relationships Created", 
        f"{refs} Edges"
    )
    
    if total_duration > 0:
        table.add_row(
            "Average / Total Time",
            f"{avg_duration:.2f}s / {total_duration:.2f}s"
        )
        
    for stage, tokens in token_totals.items():
        table.add_row(
            f"LLM Tokens: {stage}",
            f"{tokens:,}"
        )
        
    con.console.print(table)
    con.blank()


def index_orchestrator(
    target: str,
    use_llm: bool,
    trace: bool,
    cloud: bool,
    chunk_pool_size: Optional[int] = None,
    pdf_parser_type: Optional[str] = None
):
    from src.services.indexing_orchestrator import run_batch_index
    try:
        kwargs = {}
        if pdf_parser_type is not None:
            kwargs["pdf_parser_type"] = pdf_parser_type
        session_traces = run_batch_index(target, use_llm, trace, cloud, chunk_pool_size, **kwargs)
    except Exception as e:
        con.error(f"Failed during batch indexing: {e}")
        raise typer.Exit(1)

    if session_traces:
        if trace:
            for trace_info in session_traces:
                if trace_info.get("success") or trace_info.get("skipped_duplicate"):
                    print_trace_table(trace_info["name"], trace_info)
        print_session_summary_table(session_traces)


@app.command("index")
def index(
    target: str = typer.Argument(..., help="Path to file, directory, or URL to index"),
    use_llm: bool = typer.Option(True, "--use-llm/--no-llm", help="Use LLM to extract concepts (slower)"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Show detailed execution trace with timing and token count"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
    chunk_pool: Optional[int] = typer.Option(1, "--chunk-pool", help="Number of concurrent chunks to process in parallel via LLM"),
    pdf_parser: Optional[str] = typer.Option(None, "--pdf-parser", help="PDF parser to use ('marker' or 'fitz')"),
    legacy_pdf: bool = typer.Option(False, "--legacy-pdf", help="Use legacy Fitz (PyMuPDF) PDF parser (shortcut for --pdf-parser fitz)"),
):
    """Index PDF papers, Markdown notes (.md), EPUB books, or URLs into the knowledge graph."""
    parser_type = "fitz" if legacy_pdf else pdf_parser
    index_orchestrator(target, use_llm, trace, cloud, chunk_pool, parser_type)


# ── reindex ───────────────────────────────────────────────────────────────────

reindex_app = typer.Typer(help="Re-index paper metadata or everything.")
app.add_typer(reindex_app, name="reindex")

@reindex_app.command("meta")
def reindex_meta(
    missing_authors: bool = typer.Option(False, "--missing-authors", help="Reindex only papers without authors"),
    missing_tags: bool = typer.Option(False, "--missing-tags", help="Reindex only papers without topic tags"),
    all_metadata: bool = typer.Option(False, "--all-metadata", help="Reindex metadata for all papers"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of papers to reindex"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for extracting concepts/tags (slower)"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
    chunk_pool: Optional[int] = typer.Option(1, "--chunk-pool", help="Number of concurrent chunks to process in parallel via LLM"),
):
    """Partially re-index paper metadata (authors, year, tags, citations) without regenerating embeddings."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    if not all_metadata and not missing_authors and not missing_tags:
        con.warning("Please specify a filter: --missing-authors, --missing-tags, or --all-metadata")
        raise typer.Exit(0)

    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm, use_cloud=cloud)
    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    indexer.reindex_metadata_batch(
        missing_authors=missing_authors,
        missing_tags=missing_tags,
        limit=limit,
        use_llm=use_llm,
        chunk_pool_size=chunk_pool,
    )


@reindex_app.command("full")
def reindex_full(
    all_papers: bool = typer.Option(False, "--all", help="Reindex all papers"),
    paper_id: Optional[str] = typer.Option(None, "--id", help="Reindex a single paper by ID"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit the number of papers to reindex"),
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for extracting concepts/tags (slower)"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
    chunk_pool: Optional[int] = typer.Option(1, "--chunk-pool", help="Number of concurrent chunks to process in parallel via LLM"),
    pdf_parser: Optional[str] = typer.Option(None, "--pdf-parser", help="PDF parser to use ('marker' or 'fitz')"),
    legacy_pdf: bool = typer.Option(False, "--legacy-pdf", help="Use legacy Fitz (PyMuPDF) PDF parser (shortcut for --pdf-parser fitz)"),
):
    """Fully re-index papers (re-chunk and recreate embeddings) by re-ingesting original files/URLs."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    if not all_papers and not paper_id:
        con.warning("Please specify either --all or --id <paper_id>")
        raise typer.Exit(0)

    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=use_llm, use_cloud=cloud)
    if use_llm and not llm_engine:
        con.warning("Proceeding with regex fallback extraction because LLM engine failed to load.")
    indexer = Indexer(graph_repo, vector_repo, embedding_engine, llm_engine)

    parser_type = "fitz" if legacy_pdf else pdf_parser
    try:
        indexer.reindex_full_batch(
            all_papers=all_papers,
            paper_id=paper_id,
            limit=limit,
            chunk_pool_size=chunk_pool,
            pdf_parser_type=parser_type,
        )
    except ValueError as e:
        con.error(str(e))
        raise typer.Exit(1)



# ── query ─────────────────────────────────────────────────────────────────────

@app.command("query")
def query(
    text: str = typer.Argument(..., help="Your question about the indexed documents"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of context chunks to retrieve"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
    trace: bool = typer.Option(
        False,
        "--t", "-t", "--trace",
        help="Show detailed graph expansion steps, timing metrics, and retrieved neighbors"
    ),
):
    """Answer a question using hybrid RAG over all indexed documents."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True, use_cloud=cloud)
    if not llm_engine:
        con.error("LLM engine is required for query. Check your model path with: graph config")
        raise typer.Exit(1)

    pipeline = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)

    if trace:
        con.SHOW_TIME = True
        
        # Enable and configure Advanced Context Expansion
        from src.services.graph_expander import ExperimentalGraphExpander
        reranker = pipeline._get_reranker()
        pipeline.service.expander = ExperimentalGraphExpander(
            graph_repo=graph_repo,
            vector_repo=vector_repo,
            llm_engine=llm_engine,
            reranker=reranker
        )

    con.blank()
    con.search_msg(f"{text}")
    con.blank()

    response = pipeline.ask(text, limit=limit)

    con.blank()
    con.console.print(Panel(
        response,
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))
    con.blank()


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command("stats")
def stats():
    """Show knowledge base statistics and disk storage details."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    db_stats = graph_repo.get_stats()

    # DB file size
    db_path = Path(config.db_path)
    db_size = ""
    if db_path.exists():
        sz = db_path.stat().st_size
        db_size = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.0f} KB"

    table = Table(
        title="📊 Knowledge Base Statistics",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Category", style="bold white", min_width=24)
    table.add_column("Count", justify="right", style="bold green")

    table.add_row("Papers / Books / Notes", str(db_stats["papers"]))
    table.add_row("Authors", str(db_stats["authors"]))
    table.add_row("Concepts", str(db_stats["concepts"]))
    table.add_row("Graph Edges", str(db_stats["edges"]))
    table.add_row("─" * 24, "─" * 8)
    table.add_row("Database Size", db_size or "—")

    con.blank()
    con.console.print(table)
    con.blank()

    # Storage stats
    storage_stats = config.get_storage_stats()
    
    def format_size(bytes_val: int) -> str:
        if bytes_val >= 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"
        elif bytes_val >= 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.2f} MB"
        elif bytes_val >= 1024:
            return f"{bytes_val / 1024:.2f} KB"
        else:
            return f"{bytes_val} B"

    con.console.print(Panel(
        f"[bold white]📂 Storage Location:[/bold white] [cyan]{storage_stats['storage_dir']}[/cyan]\n"
        f"[bold white]📦 Total Storage Size:[/bold white] [green]{format_size(storage_stats['total_size'])}[/green]",
        title="💾 Disk Storage Information",
        border_style="blue",
        expand=False
    ))
    con.blank()

    # Breakdown by Extensions
    if storage_stats["extensions"]:
        ext_table = Table(
            title="🗂️ Storage Breakdown by File Type (Extensions)",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="blue",
            show_header=True,
            header_style="bold blue",
        )
        ext_table.add_column("Extension", style="bold white")
        ext_table.add_column("Files Count", justify="right", style="magenta")
        ext_table.add_column("Size", justify="right", style="green")
        ext_table.add_column("Percentage", justify="right", style="yellow")

        tot = storage_stats["total_size"]
        for item in storage_stats["extensions"]:
            pct = (item["size"] / tot * 100) if tot > 0 else 0
            ext_table.add_row(
                item["extension"],
                str(item["count"]),
                format_size(item["size"]),
                f"{pct:.1f}%"
            )
        con.console.print(ext_table)
        con.blank()

    # Breakdown by Sources
    if storage_stats["sources"]:
        src_table = Table(
            title="📥 Archive Breakdown by Source Type",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="blue",
            show_header=True,
            header_style="bold blue",
        )
        src_table.add_column("Source Type", style="bold white")
        src_table.add_column("Files Count", justify="right", style="magenta")
        src_table.add_column("Size", justify="right", style="green")

        source_labels = {
            "paper": "📚 Papers / Articles",
            "note": "📝 Notes (Markdown)",
            "book": "📖 Books (EPUB)",
            "webpage": "🌐 Webpages (Scraped)",
            "other": "❓ Other / Uncategorized"
        }

        for item in storage_stats["sources"]:
            lbl = source_labels.get(item["source"], f"❓ {item['source']}")
            src_table.add_row(
                lbl,
                str(item["count"]),
                format_size(item["size"])
            )
        con.console.print(src_table)
        con.blank()


# ── storage ───────────────────────────────────────────────────────────────────

@app.command("storage")
def storage(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of items to display per page"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
):
    """Display indexed data with interactive pagination, deletion and editing."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"

    from src.services.storage_tui import run_storage_tui
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    run_storage_tui(graph_repo, container, limit=limit)


# ── config ────────────────────────────────────────────────────────────────────

@app.command("config")
def show_config():
    """Show current configuration: model paths, database location, and settings."""
    from rich.table import Table
    from rich import box as rbox

    def _check(path: str) -> str:
        """Return ✓ or ✗ depending on whether path exists."""
        return "[green]✓  exists[/green]" if Path(path).exists() else "[red]✗  not found[/red]"

    def _model_info(path: str) -> str:
        """Return a human-readable model name guess from the path."""
        name = Path(path).name.lower()
        if "gemma" in name:
            return "Google Gemma (MLX)"
        if "qwen" in name:
            return "Alibaba Qwen (MLX)"
        if "llama" in name:
            return "Meta LLaMA (MLX)"
        if "mistral" in name:
            return "Mistral (MLX)"
        if "minilm" in name:
            return "MiniLM sentence embedder"
        if "all-mpnet" in name:
            return "MPNet sentence embedder"
        if "mxbai" in name:
            return "MixedBread cross-encoder"
        return "Unknown"

    con.blank()
    con.console.print(Panel(
        "[bold]Science Graph[/bold] — Configuration",
        border_style="cyan",
        padding=(0, 2),
    ))

    # ── Paths ──────────────────────────────────────────────────────────────────
    paths_table = Table(
        title="📂 Paths",
        box=rbox.ROUNDED, border_style="dim",
        show_header=True, header_style="bold cyan",
        expand=True,
    )
    paths_table.add_column("Key", style="bold white", min_width=18)
    paths_table.add_column("Path", style="dim white")
    paths_table.add_column("Status", min_width=14)

    paths_table.add_row("Database",    config.db_path,      _check(config.db_path))
    paths_table.add_row("Archive dir", config.archive_dir,  _check(config.archive_dir))
    paths_table.add_row(
        "Config file",
        str(Path.home() / ".config" / "pdf-graph-analyzer" / "config.yaml"),
        _check(str(Path.home() / ".config" / "pdf-graph-analyzer" / "config.yaml")),
    )
    con.console.print(paths_table)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_table = Table(
        title="⚡ LLM Model (text generation)",
        box=rbox.ROUNDED, border_style="magenta",
        show_header=True, header_style="bold magenta",
        expand=True,
    )
    llm_table.add_column("Setting", style="bold white", min_width=18)
    llm_table.add_column("Value")
    llm_table.add_column("Info", style="dim")

    llm_table.add_row("Active Provider",    config.llm_provider,         "mlx or openai")
    llm_table.add_row("Local Model Path",   config.llm_local_model_path, _check(config.llm_local_model_path) if config.llm_local_model_path else "not set")
    llm_table.add_row("Local Model Type",   _model_info(config.llm_local_model_path) if config.llm_local_model_path else "not set", "local MLX inference")
    
    cloud_provider = "openai"
    if isinstance(config.data.get("llm"), dict) and isinstance(config.data["llm"].get("cloud"), dict):
        cloud_provider = config.data["llm"]["cloud"].get("provider", "openai")
    
    llm_table.add_row("Cloud Provider",     cloud_provider,              "Cloud API provider")
    llm_table.add_row("Cloud Model Name",   config.llm_cloud_model_name, "Model name in cloud API")
    llm_table.add_row("Cloud Base URL",      config.llm_cloud_base_url or "default", "OpenAI compatible API base")
    cloud_api_key_masked = "configured" if config.llm_cloud_api_key else "[yellow]missing[/yellow]"
    llm_table.add_row("Cloud API Key",      cloud_api_key_masked,        "OpenAI / OpenRouter API Key")
        
    llm_table.add_row("Max tokens",  str(config.llm_max_tokens),  "max output length")
    llm_table.add_row("Temperature", str(config.llm_temp),        "0=deterministic, 1=creative")
    con.console.print(llm_table)

    # ── Embedding ─────────────────────────────────────────────────────────────
    emb_table = Table(
        title="🔢 Embedding Model (vector search)",
        box=rbox.ROUNDED, border_style="blue",
        show_header=True, header_style="bold blue",
        expand=True,
    )
    emb_table.add_column("Setting", style="bold white", min_width=18)
    emb_table.add_column("Value")
    emb_table.add_column("Info", style="dim")

    emb_table.add_row("Model name",    config.embedding_model_name, _model_info(config.embedding_model_name))
    emb_table.add_row("Reranker",      "mixedbread-ai/mxbai-rerank-xsmall-v1", "cross-encoder reranking")
    emb_table.add_row("Chunk size",    str(config.chunk_size),   "chars per text chunk")
    emb_table.add_row("Chunk overlap", str(config.chunk_overlap), "overlap between chunks")
    con.console.print(emb_table)

    # ── NLP & Extraction ──────────────────────────────────────────────────────
    nlp_table = Table(
        title="🧠 NLP & Extraction Models (spacy & ner)",
        box=rbox.ROUNDED, border_style="cyan",
        show_header=True, header_style="bold cyan",
        expand=True,
    )
    nlp_table.add_column("Model Task", style="bold white", min_width=18)
    nlp_table.add_column("Model Name / Path")
    nlp_table.add_column("Info", style="dim")

    nlp_table.add_row("spaCy model", config.spacy_model_name, "lemmatization of concepts")
    nlp_table.add_row("NER model", config.ner_model_name, "author name extraction")
    con.console.print(nlp_table)

    # ── Environment ────────────────────────────────────────────────────────────
    env_table = Table(
        title="🌐 Environment",
        box=rbox.ROUNDED, border_style="dim",
        show_header=True, header_style="bold white",
        expand=True,
    )
    env_table.add_column("Variable", style="bold white", min_width=18)
    env_table.add_column("Value")
    env_table.add_column("Effect", style="dim")

    hf_token = os.environ.get("HF_TOKEN", "")
    hf_status = "[green]set[/green]" if hf_token else "[yellow]not set (rate-limited)[/yellow]"
    env_table.add_row("HF_TOKEN",    hf_status,                           "HuggingFace auth")
    env_table.add_row("HF_HUB_VERBOSITY",
                      os.environ.get("HF_HUB_VERBOSITY", "default"),      "HF Hub log level")
    env_table.add_row("TOKENIZERS_PARALLELISM",
                      os.environ.get("TOKENIZERS_PARALLELISM", "default"), "parallelism for tokenizers")
    con.console.print(env_table)

    con.blank()
    con.dim("To change settings, edit: ~/.config/pdf-graph-analyzer/config.yaml")
    con.dim("To set HF_TOKEN: export HF_TOKEN=hf_xxxx")
    con.blank()


# ── visualize ─────────────────────────────────────────────────────────────────

@app.command("visualize")
def visualize(
    output_path: Path = typer.Option(Path.cwd() / "graph.html", "--output", "-o", help="Output HTML file path"),
):
    """Generate an interactive HTML knowledge graph and open it in the browser."""
    from src.services.visualizer import generate_html_graph
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)

    try:
        generate_html_graph(graph_repo, output_path)
    except ValueError as e:
        con.warning(str(e))
        return

    con.success(f"Graph saved to [bold]{output_path}[/bold]")
    try:
        webbrowser.open(Path(output_path).as_uri())
        con.info("Opening in browser …")
    except Exception as e:
        con.warning(f"Could not open browser: {e}")


# ── chat ──────────────────────────────────────────────────────────────────────

@app.command("chat")
def chat(
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
):
    """Start an interactive TUI chat session with RAG memory."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True, use_cloud=cloud)
    if not llm_engine:
        con.error("LLM engine is not available. Run: graph config")
        raise typer.Exit(1)

    rag_pipeline = RAGPipeline(graph_repo, vector_repo, embedding_engine, llm_engine)

    from src.tui import run_tui_chat
    run_tui_chat(rag_pipeline)


# ── review ────────────────────────────────────────────────────────────────────

@app.command("review")
def review(
    topic: str = typer.Argument(..., help="Research topic to review"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max number of chunks to retrieve"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the Markdown report"),
    fast: bool = typer.Option(False, "--fast", help="Skip LLM clustering (single-section mode)"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
):
    """Generate a full Markdown literature review on a topic using the indexed knowledge base."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=True, use_cloud=cloud)
    if not llm_engine:
        con.error("LLM engine is required for review generation. Run: graph config")
        raise typer.Exit(1)

    from src.review_agent import ReviewAgent

    agent = ReviewAgent(graph_repo, vector_repo, embedding_engine, llm_engine)

    if output is None:
        safe_name = topic.lower().replace(" ", "_")[:40]
        output = Path.cwd() / f"review_{safe_name}.md"

    report = agent.run(topic=topic, limit=limit, output_path=output, fast=fast)

    con.blank()
    con.console.print(Panel(
        report[:600] + ("…" if len(report) > 600 else ""),
        title="[bold green]Preview[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
):
    """Start the Science Graph Web UI (FastAPI + interactive vis-network graph)."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    try:
        import uvicorn
    except ImportError:
        con.error("uvicorn is not installed. Run: uv add uvicorn")
        raise typer.Exit(1)

    url = f"http://{host}:{port}"
    con.blank()
    con.console.print(Panel(
        f"[bold]Science Graph Web UI[/bold]\n\n"
        f"  [cyan]URL:[/cyan]    {url}\n"
        f"  [cyan]Reload:[/cyan] {'yes' if reload else 'no'}\n\n"
        f"  Press [bold]Ctrl+C[/bold] to stop.",
        border_style="cyan",
        padding=(1, 2),
    ))
    con.blank()

    if open_browser:
        import threading
        import time
        def _open():
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "src.web_app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning",   # suppress uvicorn INFO noise
    )


# ── serve-mcp ─────────────────────────────────────────────────────────────────

@app.command("serve-mcp")
def serve_mcp(
    sse: bool = typer.Option(False, "--sse", help="Run in SSE/HTTP mode instead of stdio"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to in SSE mode"),
    port: int = typer.Option(8010, "--port", "-p", help="Port to listen on in SSE mode"),
):
    """Start the Science Graph MCP (Model Context Protocol) Server."""
    # Ensure MCP mode is active to redirect logs to stderr
    os.environ["SCIENCE_GRAPH_MCP_MODE"] = "1"
    
    # Import inside to prevent early loading of engines before environment is set up
    from src.mcp_server import mcp
    
    if sse:
        # Run over SSE transport (which is backed by FastAPI under the hood in fastmcp)
        mcp.run(transport="sse", host=host, port=port)
    else:
        # Run over standard stdio transport (perfect for local agents)
        mcp.run(transport="stdio")


@app.command("extract-file")
def extract_file(
    target: str = typer.Argument(..., help="Path to text document"),
    use_llm: bool = typer.Option(True, "--use-llm/--no-llm", help="Use LLM to extract concepts"),
    cloud: bool = typer.Option(False, "--cloud", help="Use cloud provider instead of local model"),
):
    """Extract authors, concepts, and tags from a text document and output as JSON graph."""
    if cloud:
        os.environ["SCIENCE_GRAPH_USE_CLOUD"] = "1"
    import sys
    import re
    # Redirect con output to stderr to keep stdout clean for JSON
    old_file = con.console._file
    con.console._file = sys.stderr
    try:
        path = Path(target)
        if not path.exists():
            con.error(f"File not found: {target}")
            raise typer.Exit(1)
            
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            con.error(f"Failed to read file: {e}")
            raise typer.Exit(1)
            
        # Extract title, abstract, and body text
        first_line = content.split('\n')[0].strip() if content else ""
        if first_line.startswith("# "):
            title = first_line.lstrip("# ").strip()
            full_text = content[len(first_line):].strip()
        else:
            title = path.stem
            full_text = content

        paragraphs = [p.strip() for p in re.split(r'\n\n+', full_text) if p.strip()]
        abstract = paragraphs[0][:800] if paragraphs else ""

        # Load LLM engine if use_llm is True
        llm_engine = None
        if use_llm:
            try:
                llm_engine = LLMEngine(use_cloud=cloud)
            except Exception as e:
                con.warning(f"Could not load LLM engine: {e}. Falling back to regex extraction.")

        from src.services.extraction_service import ExtractionService
        extractor = ExtractionService(llm_engine=llm_engine)
        
        try:
            result = extractor.extract(title, abstract, full_text, use_llm=use_llm)
        except Exception as e:
            con.error(f"Extraction failed: {e}")
            raise typer.Exit(1)

        output_dict = {
            "authors": result.authors,
            "concepts": result.concepts,
            "tags": result.tags,
        }
        
        try:
            sys.stdout.buffer.write(json.dumps(output_dict, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
        except AttributeError:
            sys.stdout.write(json.dumps(output_dict, indent=2, ensure_ascii=False) + "\n")
    finally:
        con.console._file = old_file


@app.command("cleanup")
def cleanup():
    """Remove orphaned Concept nodes with degree 0 (no connected papers/notes)."""
    graph_repo, _, _, _ = get_services(load_llm=False, load_embeddings=False)
    con.dim("Starting database cleanup...")
    deleted_count = graph_repo.cleanup_orphaned_concepts()
    if deleted_count > 0:
        con.success(f"Successfully cleaned up [bold]{deleted_count}[/bold] orphaned concept nodes.")
    else:
        con.info("No orphaned concept nodes found in the database.")


@app.command("reset")
def reset():
    """Completely reset the database, vector index, and local archives."""
    con.warning("This will delete all papers, notes, concepts, embeddings, and cached files!")
    
    # First confirmation
    first_confirm = typer.confirm("Are you sure you want to completely reset the database?", default=False)
    if not first_confirm:
        con.info("Reset cancelled.")
        raise typer.Exit(0)
        
    # Second confirmation
    second_confirm = typer.confirm("WARNING: This action is irreversible. Are you REALLY sure?", default=False)
    if not second_confirm:
        con.info("Reset cancelled.")
        raise typer.Exit(0)

    # Proceed with deletion
    con.dim("Starting database reset...")
    
    # 1. Database path
    db_path = Path(config.db_path)
    if db_path.exists():
        try:
            db_path.unlink()
            con.success(f"Deleted SQLite database: [dim]{db_path}[/dim]")
        except Exception as e:
            con.error(f"Failed to delete SQLite database: {e}")
            
    # Also delete WAL and shared memory files if they exist (graph.db-wal, graph.db-shm)
    for suffix in ["-wal", "-shm"]:
        side_file = Path(str(db_path) + suffix)
        if side_file.exists():
            try:
                side_file.unlink()
            except Exception:
                pass
                
    # 2. Vector index path
    usearch_path = Path(str(db_path).replace(".db", ".usearch"))
    if usearch_path.exists():
        try:
            usearch_path.unlink()
            con.success(f"Deleted vector index: [dim]{usearch_path}[/dim]")
        except Exception as e:
            con.error(f"Failed to delete vector index: {e}")
            
    # 3. Archive directory
    archive_dir = Path(config.archive_dir)
    if archive_dir.exists():
        try:
            import shutil
            for child in archive_dir.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            con.success(f"Cleared archive directory: [dim]{archive_dir}[/dim]")
        except Exception as e:
            con.error(f"Failed to clear archive directory: {e}")
            
    con.success("Database and environment successfully reset to a pristine state.")


@app.command("doctor")
def doctor(
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Actually apply changes to sanitize and clean LLM artifacts and formatters",
    ),
    cloud: bool = typer.Option(
        False,
        "--cloud",
        help="Use cloud provider instead of local model for LLM fixes",
    ),
):
    """
    Scan database through the repository layer, detecting and fixing LLM output artifacts,
    unapplied formatting, and incorrect identifiers due to formatting anomalies.
    """
    graph_repo, vector_repo, embedding_engine, llm_engine = get_services(load_llm=fix, load_embeddings=fix, use_cloud=cloud)
    
    from src.services.doctor_service import DoctorService
    
    con.info("🩺 Starting Science Graph Database Doctor Diagnostics...")
    if fix:
        con.warning("🔧 Running in [bold yellow]FIX[/bold yellow] mode. Anomalies will be corrected in place.")
    else:
        con.info("🔍 Running in [bold cyan]CHECK-ONLY[/bold cyan] mode. No writes will be made. Run with [bold]--fix[/bold] to repair.")
        
    con.blank()
    
    doctor_service = DoctorService(graph_repo, vector_repo, llm_engine=llm_engine, emb_engine=embedding_engine)
    report = doctor_service.run_diagnostics(fix=fix)
    
    # 1. Print Stats Table
    table = Table(
        title="📊 Diagnostics Scan Statistics",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Entity Type", style="bold white", min_width=24)
    table.add_column("Checked", justify="right", style="bold white")
    table.add_column("Fixed/Corrected", justify="right", style="bold green")
    table.add_column("Migrated", justify="right", style="bold yellow")
    table.add_column("Merged", justify="right", style="bold magenta")
    
    stats = report["stats"]
    table.add_row("Papers", str(stats["papers_checked"]), str(stats["papers_fixed"]), "—", "—")
    table.add_row("Authors", str(stats["authors_checked"]), str(stats["authors_fixed"]), str(stats["authors_migrated"]), str(stats["authors_merged"]))
    table.add_row("Concepts / Tags", str(stats["concepts_checked"]), str(stats["concepts_fixed"]), str(stats["concepts_migrated"]), str(stats["concepts_merged"]))
    table.add_row("Chunks", str(stats["chunks_checked"]), str(stats["chunks_fixed"]), "—", "—")
    
    con.console.print(table)
    con.blank()
    
    # 2. Print Detailed Anomalies
    anomalies = report["anomalies"]
    total_issues = (
        len(anomalies["papers"]) +
        len(anomalies["authors"]) +
        len(anomalies["concepts"]) +
        len(anomalies["chunks"])
    )
    
    if total_issues == 0:
        con.success("🎉 No anomalies found! Database texts are completely sanitized and formatted.")
        return
        
    # Detail Papers
    if anomalies["papers"]:
        con.console.print("[bold cyan]📄 Paper Anomalies:[/bold cyan]")
        for paper in anomalies["papers"]:
            con.console.print(f"  • [bold]{paper['id']}[/bold]")
            if paper["old_title"] != paper["new_title"]:
                con.console.print(f"    - Title: [red]\"{paper['old_title']}\"[/red] -> [green]\"{paper['new_title']}\"[/green]")
            if paper["old_abstract"] != paper["new_abstract"]:
                con.console.print("    - Abstract updated")
            if paper["old_authors"] != paper["new_authors"]:
                con.console.print(f"    - Authors: [red]{paper['old_authors']}[/red] -> [green]{paper['new_authors']}[/green]")
            if paper.get("missing_abstract"):
                status = "[green]generated[/green]" if paper.get("generated_abstract") else "[yellow]missing[/yellow]"
                con.console.print(f"    - Abstract: {status}")
            if paper.get("missing_summary"):
                status = "[green]generated[/green]" if paper.get("generated_summary") else "[yellow]missing[/yellow]"
                con.console.print(f"    - Summary: {status}")
        con.blank()
        
    # Detail Authors
    if anomalies["authors"]:
        con.console.print("[bold magenta]👤 Author Anomalies:[/bold magenta]")
        for author in anomalies["authors"]:
            con.console.print(f"  • ID: [bold]{author['id']}[/bold]")
            con.console.print(f"    - Name: [red]\"{author['old_name']}\"[/red] -> [green]\"{author['new_name']}\"[/green]")
            con.console.print(f"    - Action: [bold yellow]{author['action']}[/bold yellow]")
        con.blank()
        
    # Detail Concepts
    if anomalies["concepts"]:
        con.console.print("[bold yellow]💡 Concept/Tag Anomalies:[/bold yellow]")
        for concept in anomalies["concepts"]:
            con.console.print(f"  • ID: [bold]{concept['id']}[/bold]")
            if concept["old_name"] != concept["new_name"]:
                con.console.print(f"    - Name: [red]\"{concept['old_name']}\"[/red] -> [green]\"{concept['new_name']}\"[/green]")
            if concept["old_description"] != concept["new_description"]:
                con.console.print(f"    - Description: [red]\"{concept['old_description']}\"[/red] -> [green]\"{concept['new_description']}\"[/green]")
            con.console.print(f"    - Action: [bold yellow]{concept['action']}[/bold yellow]")
        con.blank()
        
    # Detail Chunks
    if anomalies["chunks"]:
        con.console.print("[bold blue]🧩 Text Chunk Anomalies:[/bold blue]")
        con.console.print(f"  • Found [bold]{len(anomalies['chunks'])}[/bold] chunk text content anomalies containing LLM artifacts or spacing issues.")
        con.blank()
        
    if fix:
        con.success(f"✔️ Successfully corrected [bold]{total_issues}[/bold] anomalies across all tables!")
    else:
        con.warning(f"⚠️ Found [bold]{total_issues}[/bold] anomalies. Run with [bold]--fix[/bold] to repair them.")


@app.command("init")
def init():
    """Bring the configuration file up-to-date with current settings (adding new and removing obsolete fields)."""
    try:
        config.init_config()
        con.success(f"Configuration file [bold]{config.config_file}[/bold] has been successfully updated.")
    except Exception as e:
        con.error(f"Failed to initialize configuration: {e}")
        raise typer.Exit(1)


@app.command("export-db")
def export_db(
    format_type: str = typer.Option("yaml", "--format", "-f", help="Output format: json or yaml"),
    no_chunks: bool = typer.Option(False, "--no-chunks", help="Exclude text chunks from export"),
):
    """Export the database contents (nodes, edges, chunks without embeddings) to stdout as YAML or JSON."""
    graph_repo, vector_repo, _, _ = get_services(load_llm=False, load_embeddings=False)
    
    # 1. Fetch nodes
    raw_nodes = graph_repo.get_all_nodes()
    nodes = []
    for node_id, label, props_json in raw_nodes:
        try:
            props = json.loads(props_json) if props_json else {}
        except Exception:
            props = {}
        nodes.append({
            "id": node_id,
            "label": label,
            "properties": props
        })
        
    # 2. Fetch edges
    raw_edges = graph_repo.get_all_edges()
    edges = []
    for source_id, target_id, edge_type, props_json in raw_edges:
        try:
            props = json.loads(props_json) if props_json else {}
        except Exception:
            props = {}
        edges.append({
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "properties": props
        })
        
    # 3. Fetch chunks (if requested)
    chunks = []
    if not no_chunks:
        with vector_repo._get_connection() as conn:
            rows = conn.execute("SELECT id, paper_id, text_content, page_number FROM chunks").fetchall()
            for r in rows:
                chunk_id = r["id"]
                # Chunk ID is formatted as: paper_id#index. Try to extract index from it
                idx_val = 0
                if chunk_id and "#" in chunk_id:
                    try:
                        idx_val = int(chunk_id.split("#")[-1])
                    except ValueError:
                        pass
                chunks.append({
                    "id": chunk_id,
                    "paper_id": r["paper_id"],
                    "idx": idx_val,
                    "text_content": r["text_content"],
                    "page_number": r["page_number"]
                })
                
    export_data = {
        "nodes": nodes,
        "edges": edges
    }
    if not no_chunks:
        export_data["chunks"] = chunks
        
    import sys
    if format_type.lower() == "json":
        sys.stdout.write(json.dumps(export_data, indent=2, ensure_ascii=False) + "\n")
    else:
        import yaml
        yaml.safe_dump(export_data, sys.stdout, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":  # pragma: no cover
    app()

