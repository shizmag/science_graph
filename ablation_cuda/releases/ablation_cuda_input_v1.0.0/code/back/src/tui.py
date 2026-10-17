import shlex
from pathlib import Path
from typing import List, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.align import Align
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexer import RegexLexer
from pygments.token import Keyword, Text as PygText, String

class CommandLexer(RegexLexer):
    name = 'TUICommand'
    tokens = {
        'root': [
            (r'^/[a-zA-Z0-9_-]+', Keyword.Type),
            (r'(?i)^(exit|quit|clear)\b', Keyword.Reserved),
            (r'".*?"|\'.*?\'', String),
            (r'.', PygText),
        ]
    }

from src.rag import RAGPipeline
from src.indexer import Indexer
from src.config import config

class TUIHistory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: List[Tuple[str, str]] = []

    def add_turn(self, query: str, response: str):
        self.turns.append((query, response))
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def format_for_llm(self) -> str:
        formatted = []
        for q, r in self.turns:
            formatted.append(f"<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n{r}\n<|im_end|>\n")
        return "".join(formatted)

def handle_command(cmd_str: str, rag_pipeline: RAGPipeline, console: Console):
    """Parses and executes a slash command like /index, /config."""
    parts = shlex.split(cmd_str)
    if not parts:
        return
    cmd = parts[0].lower()

    if cmd == "/index":
        if len(parts) < 2:
            console.print("[red]Usage: /index <path_or_url>[/red]")
            return
            
        target = " ".join(parts[1:])
        
        # Determine target type and index
        # Use existing services to build an indexer
        indexer = Indexer(rag_pipeline.graph_repo, rag_pipeline.vector_repo, rag_pipeline.emb_engine, rag_pipeline.llm_engine)
        
        def _index_file(path: Path):
            t = path.suffix.lower().lstrip(".")
            if t == "pdf": indexer.index_pdf(str(path))
            elif t == "md": indexer.index_markdown(str(path))
            elif t == "epub": indexer.index_epub(str(path))
            else: console.print(f"[yellow]Unknown file type: {path.name}[/yellow]")

        try:
            import re
            raw_targets = re.split(r'[,;]', target)
            targets = [t.strip() for t in raw_targets if t.strip()]

            if not targets:
                console.print("[red]No targets provided to index.[/red]")
                return

            for tgt in targets:
                if tgt.startswith("http://") or tgt.startswith("https://"):
                    console.print(f"[bold cyan]Indexing {tgt}...[/bold cyan]")
                    indexer.index_url(tgt)
                else:
                    path = Path(tgt).resolve()
                    if not path.exists():
                        console.print(f"[red]Path not found: {path}[/red]")
                        return
                    if path.is_file():
                        _index_file(path)
                    elif path.is_dir():
                        allowed = {".pdf", ".md", ".epub"}
                        files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in allowed]
                        for f in files:
                            _index_file(f)
            console.print("[bold green]Indexing complete![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to index:[/bold red] {e}")

    elif cmd == "/config":
        console.print("[bold yellow]Interactive Configuration[/bold yellow] (Press Enter to keep current value, Ctrl+C to cancel)")
        from prompt_toolkit.shortcuts import prompt
        try:
            provider = prompt(f"LLM Provider (mlx/openai) [{config.llm_provider}]: ").strip()
            if provider: config.data["llm"]["provider"] = provider
            
            # Ensure nested structures exist
            if "local" not in config.data["llm"] or not isinstance(config.data["llm"]["local"], dict):
                config.data["llm"]["local"] = {}
            if "cloud" not in config.data["llm"] or not isinstance(config.data["llm"]["cloud"], dict):
                config.data["llm"]["cloud"] = {}
            
            local_path = prompt(f"Local MLX Model Path [{config.llm_local_model_path}]: ").strip()
            if local_path: config.data["llm"]["local"]["model_path"] = local_path
            
            cloud_model = prompt(f"Cloud Model Name [{config.llm_cloud_model_name}]: ").strip()
            if cloud_model: config.data["llm"]["cloud"]["model_name"] = cloud_model
            
            masked_key = "*" * len(config.llm_cloud_api_key) if config.llm_cloud_api_key else ""
            api_key = prompt(f"Cloud API Key [{masked_key}]: ", is_password=True).strip()
            if api_key: config.data["llm"]["cloud"]["api_key"] = api_key
            
            base_url = prompt(f"Cloud Base URL [{config.llm_cloud_base_url}]: ").strip()
            if base_url: config.data["llm"]["cloud"]["base_url"] = base_url
                
            # HuggingFace token — always shown
            masked_hf = ("*" * len(config.hf_token)) if config.hf_token else "(not set)"
            hf_token = prompt(f"HuggingFace Token (HF_TOKEN) [{masked_hf}]: ", is_password=True).strip()
            if hf_token:
                config.data["hf_token"] = hf_token
                import os
                os.environ["HF_TOKEN"] = hf_token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                
            config.save()
            console.print("[bold green]Configuration saved![/bold green]")
            console.print("[dim]Note: Some changes (like model) require restarting the application.[/dim]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Configuration change cancelled.[/yellow]")
        
    elif cmd == "/storage":
        from src.cli import storage
        try:
            limit = 50
            if len(parts) > 1 and parts[1].isdigit():
                limit = int(parts[1])
            storage(limit=limit)
        except Exception as e:
            console.print(f"[bold red]Storage error:[/bold red] {e}")
            
    else:
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")


def run_tui_chat(rag_pipeline: RAGPipeline):
    console = Console()
    
    console.clear()
    welcome_text = Text()
    welcome_text.append("🧠 Science Graph Analyzer — Interactive RAG Chat\n", style="bold magenta")
    welcome_text.append("Drag and drop files to index them, or ask questions.\n", style="italic")
    welcome_text.append("Commands: /index <path>, /config, 'exit' to end, 'clear'.\n", style="dim")
    
    try:
        stats = rag_pipeline.graph_repo.get_stats()
        stats_str = f"Database Stats: {stats.get('papers', 0)} papers, {stats.get('authors', 0)} authors, {stats.get('concepts', 0)} concepts, {stats.get('edges', 0)} connections"
        welcome_text.append(f"\n{stats_str}", style="green")
    except Exception:
        pass
        
    console.print(Panel(Align.center(welcome_text), border_style="bold blue", padding=(1, 2)))
    
    history = TUIHistory(max_turns=5)
    
    completer = WordCompleter(['/index', '/config', '/storage', 'exit', 'quit', 'clear'], ignore_case=True)
    
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'bottom-toolbar': 'bg:#2b2b2b #ffffff',
    })
    
    def bottom_toolbar():
        return ' 💡 Commands: /index <path>, /storage, /config | exit, clear '
    
    session = PromptSession(
        style=style,
        completer=completer,
        lexer=PygmentsLexer(CommandLexer),
        bottom_toolbar=bottom_toolbar
    )
    
    while True:
        try:
            user_input = session.prompt('\nQuery > ')
            query = user_input.strip()
            
            # Unquote drag and drop strings like '/path/to/file'
            if query.startswith("'") and query.endswith("'"):
                query = query.strip("'")
            if query.startswith('"') and query.endswith('"'):
                query = query.strip('"')

            if not query:
                continue
                
            if query.lower() in ("exit", "quit"):
                console.print("[bold red]Ending chat session. Goodbye![/bold red]")
                break
                
            if query.lower() == "clear":
                console.clear()
                console.print(Panel(Align.center(welcome_text), border_style="bold blue", padding=(1, 2)))
                continue

            # Command or drag and drop file detection
            if query.startswith("/"):
                handle_command(query, rag_pipeline, console)
                continue
                
            # If user pastes a raw path
            try:
                possible_path = Path(query)
                if possible_path.exists() and str(possible_path).startswith("/"):
                    console.print("[dim]Detected file path, auto-indexing...[/dim]")
                    handle_command(f"/index {shlex.quote(query)}", rag_pipeline, console)
                    continue
            except Exception:
                pass
            
            # Show processing spinner
            with console.status("[bold yellow]Retrieving context & generating answer...[/bold yellow]", spinner="dots"):
                history_str = history.format_for_llm()
                response = rag_pipeline.ask(query, limit=5, history_str=history_str)
                
            history.add_turn(query, response)
            
            console.print("\n")
            console.print(
                Panel(
                    Markdown(response),
                    title="[bold green]Assistant[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2)
                )
            )
            
        except KeyboardInterrupt:
            console.print("\n[bold red]Session interrupted. Goodbye![/bold red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    from src.cli import get_services
    graph, vector, emb, llm = get_services(load_llm=True)
    rag = RAGPipeline(graph, vector, emb, llm)
    run_tui_chat(rag)
